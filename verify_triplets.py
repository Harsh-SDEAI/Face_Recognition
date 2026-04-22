"""
Verify triplet completeness across anchor/positive/negative folders.

For each person ID, checks if they have:
  - At least 1 anchor
  - At least 1 positive
  - At least 1 negative

Reports: complete pairs, incomplete pairs, and orphan files.
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

    complete = []
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
            complete.append((pid, a_count, p_count, n_count))
        elif present_in == 1:
            orphans.append((pid, a_count, p_count, n_count))
        else:
            incomplete.append((pid, a_count, p_count, n_count))

    # --- Summary ---
    print(f"\n  Total person IDs found: {len(all_pids)}")
    print(f"  Complete:   {len(complete)}")
    print(f"  Incomplete: {len(incomplete)}")
    print(f"  Orphans:    {len(orphans)}")

    # --- Complete ---
    print(f"\n{'='*60}")
    print(f"  COMPLETE ({len(complete)}) - ready for training")
    print(f"{'='*60}")
    total_a, total_p, total_n = 0, 0, 0
    for pid, a, p, n in complete:
        total_a += a
        total_p += p
        total_n += n
    print(f"  Total files: {total_a} anchors, {total_p} positives, {total_n} negatives")
    if len(complete) <= 20:
        for pid, a, p, n in complete:
            print(f"    P{pid}: {a}A  {p}P  {n}N")

    # --- Incomplete ---
    if incomplete:
        print(f"\n{'='*60}")
        print(f"  INCOMPLETE ({len(incomplete)}) - missing one type")
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
                folder = "anchors"
                files = anchors[pid]
            elif p > 0:
                folder = "positives"
                files = positives[pid]
            else:
                folder = "negatives"
                files = negatives[pid]
            print(f"    P{pid}: {a}A  {p}P  {n}N  <- only in {folder}: {', '.join(files)}")

    print()


if __name__ == "__main__":
    verify()
