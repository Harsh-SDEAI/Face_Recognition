"""
Verify triplet pairing across anchor/positive/negative folders.

Each triplet = (anchor, positive_i, negative_i), so positives and negatives
must be equal per person. Reports:
  - Balanced: anchor + equal P and N counts = ready for training
  - Unbalanced: has all three types but P != N count, shows surplus files
  - Incomplete: missing anchor, positives, or negatives entirely
  - Orphans: exists in only one folder
"""

import os
import re
from collections import defaultdict

# ============================================================
# CONFIGURE BEFORE RUNNING
# ============================================================
ANCHORS_FOLDER = "/path/to/anchors"
POSITIVES_FOLDER = "/path/to/positives"
NEGATIVES_FOLDER = "/path/to/negatives"
# ============================================================

ANCHOR_PATTERN = re.compile(r"^[Pp](\d+)_[Aa]", re.IGNORECASE)
POSITIVE_PATTERN = re.compile(r"^[Pp](\d+)_[Pp]\d+", re.IGNORECASE)
NEGATIVE_PATTERN = re.compile(r"^[Pp](\d+)_[Nn]\d+", re.IGNORECASE)


def scan_folder(folder_path: str, pattern: re.Pattern) -> dict[int, list[str]]:
    result = defaultdict(list)
    if not os.path.isdir(folder_path):
        print(f"  ERROR: {folder_path} not found.")
        return result
    for fname in sorted(os.listdir(folder_path)):
        match = pattern.match(fname)
        if match:
            pid = int(match.group(1))
            result[pid].append(fname)
    return result


def verify():
    anchors = scan_folder(ANCHORS_FOLDER, ANCHOR_PATTERN)
    positives = scan_folder(POSITIVES_FOLDER, POSITIVE_PATTERN)
    negatives = scan_folder(NEGATIVES_FOLDER, NEGATIVE_PATTERN)

    all_pids = sorted(set(anchors.keys()) | set(positives.keys()) | set(negatives.keys()))

    balanced = []
    unbalanced = []
    incomplete = []
    orphans = []

    for pid in all_pids:
        a_count = len(anchors.get(pid, []))
        p_count = len(positives.get(pid, []))
        n_count = len(negatives.get(pid, []))

        has_a = a_count > 0
        has_p = p_count > 0
        has_n = n_count > 0
        present_in = sum([has_a, has_p, has_n])

        if has_a and has_p and has_n:
            if p_count == n_count:
                balanced.append((pid, a_count, p_count, n_count))
            else:
                unbalanced.append((pid, a_count, p_count, n_count))
        elif present_in == 1:
            orphans.append((pid, a_count, p_count, n_count))
        else:
            incomplete.append((pid, a_count, p_count, n_count))

    # --- Summary ---
    total_triplets = sum(min(p, n) for _, _, p, n in balanced + unbalanced)
    print(f"\n  Total person IDs: {len(all_pids)}")
    print(f"  Balanced:   {len(balanced)}")
    print(f"  Unbalanced: {len(unbalanced)}")
    print(f"  Incomplete: {len(incomplete)}")
    print(f"  Orphans:    {len(orphans)}")
    print(f"  Total valid triplets: {total_triplets}")

    # --- Balanced ---
    print(f"\n{'='*60}")
    print(f"  BALANCED ({len(balanced)}) - ready for training")
    print(f"{'='*60}")
    if balanced:
        total_pairs = sum(p for _, _, p, _ in balanced)
        print(f"  {total_pairs} triplets from {len(balanced)} persons")
        if len(balanced) <= 20:
            for pid, a, p, n in balanced:
                print(f"    P{pid}: {a}A  {p}P  {n}N  = {min(p,n)} triplets")

    # --- Unbalanced ---
    if unbalanced:
        print(f"\n{'='*60}")
        print(f"  UNBALANCED ({len(unbalanced)}) - P and N counts don't match")
        print(f"{'='*60}")
        total_surplus = 0
        for pid, a, p, n in unbalanced:
            valid = min(p, n)
            if p > n:
                surplus = p - n
                total_surplus += surplus
                surplus_files = positives[pid][-surplus:]
                print(f"    P{pid}: {a}A  {p}P  {n}N  = {valid} triplets, {surplus} extra positive(s) to delete:")
                for f in surplus_files:
                    print(f"      -> {f}")
            else:
                surplus = n - p
                total_surplus += surplus
                surplus_files = negatives[pid][-surplus:]
                print(f"    P{pid}: {a}A  {p}P  {n}N  = {valid} triplets, {surplus} extra negative(s) to delete:")
                for f in surplus_files:
                    print(f"      -> {f}")
        print(f"\n  Total surplus files to delete: {total_surplus}")

    # --- Incomplete ---
    if incomplete:
        print(f"\n{'='*60}")
        print(f"  INCOMPLETE ({len(incomplete)}) - missing one or more types")
        print(f"{'='*60}")
        for pid, a, p, n in incomplete:
            missing = []
            if a == 0: missing.append("anchor")
            if p == 0: missing.append("positives")
            if n == 0: missing.append("negatives")
            print(f"    P{pid}: {a}A  {p}P  {n}N  <- missing {', '.join(missing)}")

    # --- Orphans ---
    if orphans:
        print(f"\n{'='*60}")
        print(f"  ORPHANS ({len(orphans)}) - exists in only one folder, safe to delete")
        print(f"{'='*60}")
        for pid, a, p, n in orphans:
            if a > 0:
                files = anchors[pid]
                folder = "anchors"
            elif p > 0:
                files = positives[pid]
                folder = "positives"
            else:
                files = negatives[pid]
                folder = "negatives"
            print(f"    P{pid}: {a}A  {p}P  {n}N  <- only in {folder}: {', '.join(files)}")

    print()


if __name__ == "__main__":
    verify()
