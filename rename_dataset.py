"""
Rename files in a folder to offset person IDs.

Set FOLDER_PATH and START_NUMBER below, then run:
    python rename_dataset.py
"""

import os
import re

# ============================================================
# CONFIGURE THESE BEFORE RUNNING
# ============================================================
FOLDER_PATH = "/path/to/your/folder"   # folder containing the images
START_NUMBER = 632                      # p1 becomes p{START_NUMBER}
# ============================================================


def rename_files(folder_path: str, start_number: int):
    pattern = re.compile(r"^[Pp](\d+)_(.+)$")

    rename_plan = []
    for fname in sorted(os.listdir(folder_path)):
        name, ext = os.path.splitext(fname)
        match = pattern.match(name)
        if not match:
            print(f"  SKIP (no match): {fname}")
            continue

        old_pid = int(match.group(1))
        suffix = match.group(2)
        new_pid = old_pid + (start_number - 1)
        new_name = f"P{new_pid}_{suffix}{ext}"
        rename_plan.append((fname, new_name))

    if not rename_plan:
        print("Nothing to rename.")
        return

    print(f"\n  {len(rename_plan)} files to rename:\n")
    for old, new in rename_plan[:10]:
        print(f"    {old}  ->  {new}")
    if len(rename_plan) > 10:
        print(f"    ... and {len(rename_plan) - 10} more")

    confirm = input("\n  Proceed? (y/n): ").strip().lower()
    if confirm != "y":
        print("  Aborted.")
        return

    temp_plan = []
    for old, new in rename_plan:
        temp_name = f"__temp__{old}"
        os.rename(os.path.join(folder_path, old), os.path.join(folder_path, temp_name))
        temp_plan.append((temp_name, new))

    for temp, new in temp_plan:
        os.rename(os.path.join(folder_path, temp), os.path.join(folder_path, new))

    print(f"\n  Done. Renamed {len(rename_plan)} files.")


if __name__ == "__main__":
    if not os.path.isdir(FOLDER_PATH):
        print(f"Error: {FOLDER_PATH} is not a directory. Update FOLDER_PATH at the top of the script.")
        raise SystemExit(1)

    print(f"\n  Folder: {FOLDER_PATH}")
    print(f"  P1 -> P{START_NUMBER}, P2 -> P{START_NUMBER + 1}, ...")

    rename_files(FOLDER_PATH, START_NUMBER)
