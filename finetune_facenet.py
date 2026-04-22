"""
Finetune FaceNet (InceptionResnetV1) with triplet loss on pre-formed triplets.

Requirements: torch, torchvision, facenet-pytorch, Pillow

Usage: Set the config variables below, then run:
    python finetune_facenet.py
"""

import os
import re
import csv
import time
import random
from collections import defaultdict
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1


# ============================================================
# CONFIGURATION
# ============================================================

# Dataset paths
ANCHORS_FOLDER = "/path/to/anchors"
POSITIVES_FOLDER = "/path/to/positives"
NEGATIVES_FOLDER = "/path/to/negatives"
OUTPUT_DIR = "/path/to/output"

# Split
VAL_PERSONS = 20
TEST_PERSONS = 20

# Training
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-4
MARGIN = 0.2
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

# LR Scheduler (reduces LR when val loss plateaus)
LR_PATIENCE = 3     # epochs to wait before reducing
LR_FACTOR = 0.5     # multiply LR by this factor

# Early Stopping
EARLY_STOP_PATIENCE = 7

# Reproducibility
SEED = 42

# Layers to freeze (everything up to and including repeat_2)
FREEZE_LAYERS = [
    "conv2d_1a", "conv2d_2a", "conv2d_2b", "maxpool_3a",
    "conv2d_3b", "conv2d_4a", "conv2d_4b",
    "repeat_1", "mixed_6a", "repeat_2",
]

# ============================================================
# END OF CONFIGURATION
# ============================================================

ANCHOR_PATTERN = re.compile(r"^[Pp](\d+)_[Aa]", re.IGNORECASE)
POSITIVE_PATTERN = re.compile(r"^[Pp](\d+)_[Pp]\d+", re.IGNORECASE)
NEGATIVE_PATTERN = re.compile(r"^[Pp](\d+)_[Nn]\d+", re.IGNORECASE)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scan_folder(folder_path, pattern):
    result = defaultdict(list)
    for fname in sorted(os.listdir(folder_path)):
        match = pattern.match(fname)
        if match:
            pid = int(match.group(1))
            result[pid].append(fname)
    return result


def build_triplets(person_ids, anchors, positives, negatives):
    triplets = []
    for pid in person_ids:
        a_files = anchors.get(pid, [])
        p_files = positives.get(pid, [])
        n_files = negatives.get(pid, [])

        if not a_files or not p_files or not n_files:
            continue

        anchor_path = os.path.join(ANCHORS_FOLDER, a_files[0])

        max_len = max(len(p_files), len(n_files))
        p_list = [x for _, x in zip(range(max_len), cycle(p_files))]
        n_list = [x for _, x in zip(range(max_len), cycle(n_files))]

        for p_fname, n_fname in zip(p_list, n_list):
            p_path = os.path.join(POSITIVES_FOLDER, p_fname)
            n_path = os.path.join(NEGATIVES_FOLDER, n_fname)
            triplets.append((anchor_path, p_path, n_path))

    return triplets


# ----------------------------------------------------------------
# Transforms
# NOTE: existing inference pipeline uses [0, 1] range (/ 255.0)
#       with NO mean/std normalization. We match that here.
# ----------------------------------------------------------------

base_transform = transforms.Compose([
    transforms.ToTensor(),  # [0, 1]
])

anchor_train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
])

positive_train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.07),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5))], p=0.2),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
])


class TripletDataset(Dataset):
    def __init__(self, triplets, anchor_tf, positive_tf, negative_tf):
        self.triplets = triplets
        self.anchor_tf = anchor_tf
        self.positive_tf = positive_tf
        self.negative_tf = negative_tf

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a_path, p_path, n_path = self.triplets[idx]
        a = self.anchor_tf(Image.open(a_path).convert("RGB"))
        p = self.positive_tf(Image.open(p_path).convert("RGB"))
        n = self.negative_tf(Image.open(n_path).convert("RGB"))
        return a, p, n


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (anchor, positive, negative) in enumerate(loader):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            a_emb = F.normalize(model(anchor), p=2, dim=1)
            p_emb = F.normalize(model(positive), p=2, dim=1)
            n_emb = F.normalize(model(negative), p=2, dim=1)
            loss = criterion(a_emb, p_emb, n_emb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = anchor.size(0)
        total_loss += loss.item() * batch_size

        with torch.no_grad():
            d_ap = (a_emb - p_emb).pow(2).sum(dim=1)
            d_an = (a_emb - n_emb).pow(2).sum(dim=1)
            correct += (d_ap < d_an).sum().item()
            total += batch_size

        if (batch_idx + 1) % 10 == 0:
            print(f"    Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f} | Acc: {correct/total*100:.1f}%")

    return total_loss / total, correct / total * 100


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_d_ap = []
    all_d_an = []

    for anchor, positive, negative in loader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        with torch.cuda.amp.autocast():
            a_emb = F.normalize(model(anchor), p=2, dim=1)
            p_emb = F.normalize(model(positive), p=2, dim=1)
            n_emb = F.normalize(model(negative), p=2, dim=1)
            loss = criterion(a_emb, p_emb, n_emb)

        batch_size = anchor.size(0)
        total_loss += loss.item() * batch_size

        d_ap = (a_emb - p_emb).pow(2).sum(dim=1)
        d_an = (a_emb - n_emb).pow(2).sum(dim=1)
        correct += (d_ap < d_an).sum().item()
        total += batch_size
        all_d_ap.extend(d_ap.cpu().tolist())
        all_d_an.extend(d_an.cpu().tolist())

    return (
        total_loss / total,
        correct / total * 100,
        np.mean(all_d_ap),
        np.mean(all_d_an),
    )


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # --- Scan folders ---
    anchors = scan_folder(ANCHORS_FOLDER, ANCHOR_PATTERN)
    positives = scan_folder(POSITIVES_FOLDER, POSITIVE_PATTERN)
    negatives = scan_folder(NEGATIVES_FOLDER, NEGATIVE_PATTERN)

    complete_pids = sorted(
        set(anchors.keys()) & set(positives.keys()) & set(negatives.keys())
    )
    print(f"  Complete persons: {len(complete_pids)}")

    # --- Split by person ID ---
    random.shuffle(complete_pids)
    test_pids = set(complete_pids[:TEST_PERSONS])
    val_pids = set(complete_pids[TEST_PERSONS : TEST_PERSONS + VAL_PERSONS])
    train_pids = set(complete_pids[TEST_PERSONS + VAL_PERSONS :])

    train_triplets = build_triplets(train_pids, anchors, positives, negatives)
    val_triplets = build_triplets(val_pids, anchors, positives, negatives)
    test_triplets = build_triplets(test_pids, anchors, positives, negatives)

    print(f"  Train: {len(train_pids)} persons, {len(train_triplets)} triplets")
    print(f"  Val:   {len(val_pids)} persons, {len(val_triplets)} triplets")
    print(f"  Test:  {len(test_pids)} persons, {len(test_triplets)} triplets")

    # Save split for reproducibility
    with open(os.path.join(OUTPUT_DIR, "split_info.txt"), "w") as f:
        f.write(f"Seed: {SEED}\n")
        f.write(f"Train ({len(train_pids)}): {sorted(train_pids)}\n")
        f.write(f"Val ({len(val_pids)}): {sorted(val_pids)}\n")
        f.write(f"Test ({len(test_pids)}): {sorted(test_pids)}\n")

    # --- DataLoaders ---
    train_ds = TripletDataset(
        train_triplets, anchor_train_transform, positive_train_transform, base_transform
    )
    val_ds = TripletDataset(val_triplets, base_transform, base_transform, base_transform)
    test_ds = TripletDataset(test_triplets, base_transform, base_transform, base_transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # --- Model ---
    model = InceptionResnetV1(pretrained="vggface2").to(device)

    for name, param in model.named_parameters():
        if any(name.startswith(layer) for layer in FREEZE_LAYERS):
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Parameters: {trainable:,} trainable | {frozen:,} frozen")

    # --- Training setup ---
    criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE,
    )
    scaler = torch.cuda.amp.GradScaler()

    # --- CSV log ---
    log_path = os.path.join(OUTPUT_DIR, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_acc",
            "val_loss", "val_acc",
            "test_loss", "test_acc",
            "avg_d_ap", "avg_d_an", "lr", "time_sec",
        ])

    # --- Baseline (before any training) ---
    print(f"\n  Evaluating baseline (pretrained, no finetuning)...")
    base_val_loss, base_val_acc, _, _ = evaluate(model, val_loader, criterion, device)
    base_test_loss, base_test_acc, base_d_ap, base_d_an = evaluate(model, test_loader, criterion, device)
    print(f"  Baseline Val  Loss: {base_val_loss:.4f} | Acc: {base_val_acc:.2f}%")
    print(f"  Baseline Test Loss: {base_test_loss:.4f} | Acc: {base_test_acc:.2f}%")
    print(f"  Baseline Avg d(a,p): {base_d_ap:.4f} | Avg d(a,n): {base_d_an:.4f}")

    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([
            0, "-", "-",
            f"{base_val_loss:.4f}", f"{base_val_acc:.2f}",
            f"{base_test_loss:.4f}", f"{base_test_acc:.2f}",
            f"{base_d_ap:.4f}", f"{base_d_an:.4f}", f"{LR:.6f}", "-",
        ])

    # --- Training loop ---
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        lr_current = optimizer.param_groups[0]["lr"]

        print(f"\n{'='*60}")
        print(f"  Epoch {epoch}/{EPOCHS} | LR: {lr_current:.6f}")
        print(f"{'='*60}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        print(f"  Train | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")

        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"  Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        test_loss, test_acc, avg_d_ap, avg_d_an = evaluate(
            model, test_loader, criterion, device
        )
        print(f"  Test  | Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%")
        print(f"  Avg d(a,p): {avg_d_ap:.4f} | Avg d(a,n): {avg_d_an:.4f}")

        elapsed = time.time() - start
        scheduler.step(val_loss)

        # Save checkpoint (every epoch, with metrics in filename)
        ckpt_name = (
            f"epoch_{epoch:02d}"
            f"_vloss_{val_loss:.4f}"
            f"_vacc_{val_acc:.2f}"
            f"_tacc_{test_acc:.2f}"
            ".pt"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            },
            os.path.join(OUTPUT_DIR, ckpt_name),
        )
        print(f"  Saved: {ckpt_name}")

        # CSV log
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_loss:.4f}", f"{train_acc:.2f}",
                f"{val_loss:.4f}", f"{val_acc:.2f}",
                f"{test_loss:.4f}", f"{test_acc:.2f}",
                f"{avg_d_ap:.4f}", f"{avg_d_an:.4f}",
                f"{lr_current:.6f}", f"{elapsed:.1f}",
            ])

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"  >> New best val loss!")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping after {epoch} epochs.")
            break

    print(f"\n{'='*60}")
    print(f"  Training complete.")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Baseline test acc: {base_test_acc:.2f}%")
    print(f"  Checkpoints saved to: {OUTPUT_DIR}")
    print(f"  Training log: {log_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
