"""
Redistribute negatives from incomplete persons (no anchor) to unbalanced
persons (more positives than negatives).

Takes negatives from the incomplete pool, renames them, and moves them
to fill the gaps in unbalanced persons.
"""

import os
import re
import shutil
from collections import defaultdict

# ============================================================
# CONFIGURE BEFORE RUNNING
# ============================================================
POSITIVES_FOLDER = "/path/to/positives"
NEGATIVES_FOLDER = "/path/to/negatives"
ANCHORS_FOLDER = "/path/to/anchors"
# ============================================================

ANCHOR_PATTERN = re.compile(r"^[Pp](\d+)_[Aa]", re.IGNORECASE)
POSITIVE_PATTERN = re.compile(r"^[Pp](\d+)_[Pp](\d+)", re.IGNORECASE)
NEGATIVE_PATTERN = re.compile(r"^[Pp](\d+)_[Nn](\d+)", re.IGNORECASE)


def scan_folder(folder_path, pattern):
    result = defaultdict(list)
    for fname in sorted(os.listdir(folder_path)):
        match = pattern.match(fname)
        if match:
            pid = int(match.group(1))
            result[pid].append(fname)
    return result


def redistribute():
    anchors = scan_folder(ANCHORS_FOLDER, ANCHOR_PATTERN)
    positives = scan_folder(POSITIVES_FOLDER, POSITIVE_PATTERN)
    negatives = scan_folder(NEGATIVES_FOLDER, NEGATIVE_PATTERN)

    all_pids = sorted(set(anchors.keys()) | set(positives.keys()) | set(negatives.keys()))

    # Find unbalanced (has anchor, P > N) and incomplete (no anchor, has negatives)
    needs_negatives = []
    donor_pool = []

    for pid in all_pids:
        has_a = len(anchors.get(pid, [])) > 0
        p_count = len(positives.get(pid, []))
        n_count = len(negatives.get(pid, []))

        if has_a and p_count > n_count:
            deficit = p_count - n_count
            needs_negatives.append((pid, deficit, n_count))

        if not has_a and n_count > 0:
            for fname in negatives[pid]:
                donor_pool.append(fname)

    if not needs_negatives:
        print("  No unbalanced persons found. Nothing to do.")
        return

    if not donor_pool:
        print("  No donor negatives available from incomplete persons.")
        return

    total_needed = sum(d for _, d, _ in needs_negatives)
    print(f"\n  Negatives needed:    {total_needed}")
    print(f"  Negatives available: {len(donor_pool)}")

    # Build the redistribution plan
    plan = []
    pool_idx = 0

    for pid, deficit, existing_n_count in needs_negatives:
        # Find the max existing negative number for this person
        max_n_num = 0
        for fname in negatives.get(pid, []):
            match = NEGATIVE_PATTERN.match(fname)
            if match:
                max_n_num = max(max_n_num, int(match.group(2)))

        assigned = 0
        for i in range(deficit):
            if pool_idx >= len(donor_pool):
                break
            donor_file = donor_pool[pool_idx]
            pool_idx += 1
            ext = os.path.splitext(donor_file)[1]
            new_num = max_n_num + 1 + i
            new_name = f"P{pid}_N{new_num}{ext}"
            plan.append((donor_file, new_name))
            assigned += 1

        print(f"  P{pid}: needs {deficit}, assigning {assigned} -> N{max_n_num+1} to N{max_n_num+assigned}")

    leftover = len(donor_pool) - pool_idx
    print(f"\n  Total redistributions: {len(plan)}")
    print(f"  Leftover donor negatives: {leftover}")

    if leftover > 0:
        print(f"  Leftover files:")
        for fname in donor_pool[pool_idx:]:
            print(f"    {fname}")

    # Preview
    print(f"\n  Plan preview:")
    for old, new in plan[:10]:
        print(f"    {old}  ->  {new}")
    if len(plan) > 10:
        print(f"    ... and {len(plan) - 10} more")

    confirm = input("\n  Proceed? (y/n): ").strip().lower()
    if confirm != "y":
        print("  Aborted.")
        return

    # Execute: rename donor files in the negatives folder
    for old, new in plan:
        os.rename(
            os.path.join(NEGATIVES_FOLDER, old),
            os.path.join(NEGATIVES_FOLDER, new),
        )

    print(f"\n  Done. Redistributed {len(plan)} negatives.")

    if leftover > 0:
        delete = input(f"  Delete {leftover} leftover donor files? (y/n): ").strip().lower()
        if delete == "y":
            for fname in donor_pool[pool_idx:]:
                os.remove(os.path.join(NEGATIVES_FOLDER, fname))
            print(f"  Deleted {leftover} leftover files.")


if __name__ == "__main__":
    redistribute()
