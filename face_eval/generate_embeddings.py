"""Walk Studio/Game folders, detect faces, generate 5-model embeddings.

Run once per evaluation round (re-runs are idempotent: existing photos are
skipped by path).

Pipeline per photo:
    1. MTCNN detect faces (shared across all 5 models)
    2. Crop + eye-based alignment (ported from production AIPhotoMatch.py)
    3. Save aligned crop as PNG under face_crops/
    4. For each of 5 models: forward pass -> 512-d L2-normalized embedding
    5. Insert 1 row in EvalFaceDetection + 5 rows in EvalEmbedding
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from facenet_pytorch.models.mtcnn import ONet, PNet, RNet
from tqdm import tqdm

import config
from model_loaders import adaface, arcface, facenet, lvface, magface
from utils.alignment import alignment_procedure
from utils.db import connect


# ---------- Same MTCNN subclass pattern as production AIPhotoMatch.py ----------
class FinetunedMTCNN(MTCNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = kwargs.get("device", torch.device("cpu"))
        self.pnet = PNet().to(self.device)
        self.rnet = RNet().to(self.device)
        self.onet = ONet().to(self.device)


# ---------- Photo discovery ----------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(folder: Path):
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


# ---------- Insert helpers ----------
def insert_studio_photo(cur, team_key: int, path: str) -> int:
    cur.execute(
        "INSERT INTO EvalStudioPhoto (TeamKey, ImagePath) OUTPUT INSERTED.StudioPhotoID "
        "VALUES (?, ?)", team_key, path,
    )
    return int(cur.fetchone()[0])


def insert_game_photo(cur, game_number: int, path: str) -> int:
    cur.execute(
        "INSERT INTO EvalGamePhoto (GameNumber, ImagePath) OUTPUT INSERTED.GamePhotoID "
        "VALUES (?, ?)", game_number, path,
    )
    return int(cur.fetchone()[0])


def insert_detection(cur, source_type: str, source_id: int, box, confidence, crop_path) -> int:
    x1, y1, x2, y2 = [int(v) for v in box]
    cur.execute(
        "INSERT INTO EvalFaceDetection (SourceType, SourceID, BoxX1, BoxY1, BoxX2, BoxY2, "
        "Confidence, FaceCropPath) OUTPUT INSERTED.FaceID VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        source_type, source_id, x1, y1, x2, y2, float(confidence), crop_path,
    )
    return int(cur.fetchone()[0])


def insert_embedding(cur, face_id: int, model_name: str, embedding: np.ndarray, quality_norm=None):
    cur.execute(
        "INSERT INTO EvalEmbedding (FaceID, ModelName, Embedding, QualityNorm) "
        "VALUES (?, ?, ?, ?)",
        face_id, model_name, embedding.astype(np.float32).tobytes(), quality_norm,
    )


# ---------- Per-photo processing ----------
def process_photo(path: Path, source_type: str, source_id: int, mtcnn, models, cur):
    """Run MTCNN + 5 models on one photo."""
    try:
        img = Image.open(path).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] cannot open {path}: {exc}")
        return 0

    boxes, confidences, landmarks = mtcnn.detect(img, landmarks=True)
    if boxes is None:
        return 0

    num_faces = 0
    for i, (box, conf) in enumerate(zip(boxes, confidences)):
        if conf is None or conf < config.MTCNN_THRESHOLD:
            continue

        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img.width, x2); y2 = min(img.height, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        cropped = img.crop((x1, y1, x2, y2))
        aligned = alignment_procedure(cropped, landmarks[i][0], landmarks[i][1])
        # aligned is np.uint8 RGB H x W x 3

        crop_filename = f"{source_type}_{source_id}_{i}_{uuid.uuid4().hex[:8]}.png"
        crop_path = config.FACE_CROPS_DIR / crop_filename
        # PIL expects RGB; aligned already RGB from PIL source
        Image.fromarray(aligned).save(crop_path)

        face_id = insert_detection(cur, source_type, source_id, (x1, y1, x2, y2), conf, str(crop_path))

        # ---- FaceNet ----
        try:
            e = models["facenet"].embed(aligned)
            insert_embedding(cur, face_id, "facenet", e)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] facenet failed on face {face_id}: {exc}")

        # ---- ArcFace ----
        if models.get("arcface") is not None:
            try:
                e = models["arcface"].embed(aligned)
                insert_embedding(cur, face_id, "arcface", e)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] arcface failed on face {face_id}: {exc}")

        # ---- AdaFace (also captures quality norm) ----
        if models.get("adaface") is not None:
            try:
                e, qnorm = models["adaface"].embed_with_norm(aligned)
                insert_embedding(cur, face_id, "adaface", e, quality_norm=qnorm)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] adaface failed on face {face_id}: {exc}")

        # ---- MagFace ----
        if models.get("magface") is not None:
            try:
                e = models["magface"].embed(aligned)
                insert_embedding(cur, face_id, "magface", e)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] magface failed on face {face_id}: {exc}")

        # ---- LVFace ----
        if models.get("lvface") is not None:
            try:
                e = models["lvface"].embed(aligned)
                insert_embedding(cur, face_id, "lvface", e)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] lvface failed on face {face_id}: {exc}")

        num_faces += 1

    return num_faces


# ---------- Main ----------
def load_models(device: torch.device):
    """Each loader is best-effort: if weights or model def are missing, we
    skip that model with a warning rather than aborting the whole run."""
    models = {}
    try:
        models["facenet"] = facenet.load(device)
        print("[ok] facenet loaded")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] facenet failed to load: {exc}")

    for name, loader in (("arcface", arcface), ("adaface", adaface),
                         ("magface", magface), ("lvface", lvface)):
        try:
            models[name] = loader.load(device)
            print(f"[ok] {name} loaded")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {name} failed to load (skipping): {exc}")
            models[name] = None
    return models


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    mtcnn = FinetunedMTCNN(keep_all=True, device=device, min_face_size=config.MIN_FACE_SIZE)
    models = load_models(device)

    conn = connect()
    cur = conn.cursor()

    # Pull the 5 rows the user inserted into EvalGames.
    cur.execute("SELECT GameNumber, TeamKey1, TeamKey2 FROM EvalGames ORDER BY GameNumber")
    games = cur.fetchall()
    if not games:
        print("EvalGames is empty.  Insert 5 rows first.")
        sys.exit(1)
    print(f"Processing {len(games)} game(s).")

    seen_teams = set()

    for game_number, team_key_1, team_key_2 in games:
        # ---- Studio photos per team ----
        for team_key in (team_key_1, team_key_2):
            if team_key in seen_teams:
                continue
            seen_teams.add(team_key)

            folder = config.STUDIO_ROOT / str(team_key)
            photos = list_images(folder)
            print(f"[studio] team {team_key}: {len(photos)} photo(s) in {folder}")
            for p in tqdm(photos, desc=f"studio {team_key}"):
                source_id = insert_studio_photo(cur, team_key, str(p))
                process_photo(p, "S", source_id, mtcnn, models, cur)
                conn.commit()

        # ---- Game photos ----
        folder = config.GAME_ROOT / str(game_number)
        photos = list_images(folder)
        print(f"[game]   {game_number}: {len(photos)} photo(s) in {folder}")
        for p in tqdm(photos, desc=f"game {game_number}"):
            source_id = insert_game_photo(cur, game_number, str(p))
            process_photo(p, "G", source_id, mtcnn, models, cur)
            conn.commit()

    cur.close()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
