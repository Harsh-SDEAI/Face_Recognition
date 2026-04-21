"""
Rename files in a folder to offset person IDs.

Usage:
    python rename_dataset.py /path/to/folder 632

Renames all files matching p{N}_* pattern in the given folder:
    p1_a1.jpg  -> p632_a1.jpg
    p5_p3.jpg  -> p636_p3.jpg

p1 -> p{start}, p2 -> p{start+1}, etc.
"""

import os
import re
import sys


def rename_files(folder_path: str, start_number: int):
    pattern = re.compile(r"^p(\d+)_(.+)$")

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
        new_name = f"p{new_pid}_{suffix}{ext}"
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
    if len(sys.argv) != 3:
        print("Usage: python rename_dataset.py <folder_path> <start_number>")
        print("Example: python rename_dataset.py ./positives 632")
        sys.exit(1)

    folder = sys.argv[1]
    start = int(sys.argv[2])

    if not os.path.isdir(folder):
        print(f"Error: {folder} is not a directory.")
        sys.exit(1)

    print(f"\n  Folder: {folder}")
    print(f"  p1 -> p{start}, p2 -> p{start + 1}, ...")

    rename_files(folder, start)
