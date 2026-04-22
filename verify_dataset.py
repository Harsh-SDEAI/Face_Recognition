"""
Verify that each folder only contains the right type of files.

- Anchors folder: only P{n}_A files
- Positives folder: only P{n}_P{m} files
- Negatives folder: only P{n}_N{m} files

Flags any file that doesn't belong.
"""

import os
import re

# ============================================================
# CONFIGURE BEFORE RUNNING
# ============================================================
ANCHORS_FOLDER = "/path/to/anchors"
POSITIVES_FOLDER = "/path/to/positives"
NEGATIVES_FOLDER = "/path/to/negatives"
# ============================================================

PATTERNS = {
    "anchors":   re.compile(r"^[Pp]\d+_[Aa]\d*\..+$"),
    "positives": re.compile(r"^[Pp]\d+_[Pp]\d+\..+$"),
    "negatives": re.compile(r"^[Pp]\d+_[Nn]\d+\..+$"),
}

IMPOSTERS = {
    "anchors":   [re.compile(r"_[Pp]\d+\.", re.IGNORECASE), re.compile(r"_[Nn]\d+\.", re.IGNORECASE)],
    "positives": [re.compile(r"_[Aa]\d*\.", re.IGNORECASE), re.compile(r"_[Nn]\d+\.", re.IGNORECASE)],
    "negatives": [re.compile(r"_[Aa]\d*\.", re.IGNORECASE), re.compile(r"_[Pp]\d+\.", re.IGNORECASE)],
}

IMPOSTER_LABELS = {
    "anchors":   ["positive", "negative"],
    "positives": ["anchor", "negative"],
    "negatives": ["anchor", "positive"],
}


def check_folder(folder_path: str, folder_type: str):
    if not os.path.isdir(folder_path):
        print(f"  ERROR: {folder_path} not found, skipping.\n")
        return

    files = sorted(os.listdir(folder_path))
    expected = PATTERNS[folder_type]
    imposter_patterns = IMPOSTERS[folder_type]
    imposter_names = IMPOSTER_LABELS[folder_type]

    total = 0
    correct = 0
    imposters = []
    unknown = []

    for fname in files:
        if fname.startswith("."):
            continue
        total += 1

        if expected.match(fname):
            correct += 1
            continue

        found = False
        for pat, label in zip(imposter_patterns, imposter_names):
            if pat.search(fname):
                imposters.append((fname, label))
                found = True
                break

        if not found:
            unknown.append(fname)

    print(f"  [{folder_type.upper()}] {folder_path}")
    print(f"  Total: {total}  |  Correct: {correct}  |  Imposters: {len(imposters)}  |  Unknown: {len(unknown)}")

    if imposters:
        print(f"\n  IMPOSTERS FOUND:")
        for fname, label in imposters:
            print(f"    {fname}  <- looks like a {label}")

    if unknown:
        print(f"\n  UNKNOWN FILES:")
        for fname in unknown:
            print(f"    {fname}")

    if not imposters and not unknown:
        print("  ALL GOOD")

    print()


if __name__ == "__main__":
    print()
    check_folder(ANCHORS_FOLDER, "anchors")
    check_folder(POSITIVES_FOLDER, "positives")
    check_folder(NEGATIVES_FOLDER, "negatives")
