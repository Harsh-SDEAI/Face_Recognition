"""
Rename dataset files to make person IDs continuous across folders.

Usage:
    python rename_dataset.py /path/to/folder 632

This will rename all files in anchors/, positives/, negatives/ subfolders:
    p1_a1.jpg  -> p632_a1.jpg
    p5_p3.jpg  -> p636_p3.jpg
    p1_n2.jpg  -> p632_n2.jpg

The starting number you provide becomes the new ID for p1.
    p1 -> p{start}, p2 -> p{start+1}, etc.
"""

import os
import re
import sys


def rename_files(folder_path: str, start_number: int):
    subfolders = ["anchors", "positives", "negatives"]
    pattern = re.compile(r"^p(\d+)_(.+)$")

    # First pass: collect all files to rename across all subfolders
    rename_plan = []
    for sub in subfolders:
        sub_path = os.path.join(folder_path, sub)
        if not os.path.isdir(sub_path):
            print(f"  WARNING: {sub_path} not found, skipping.")
            continue

        for fname in sorted(os.listdir(sub_path)):
            name, ext = os.path.splitext(fname)
            match = pattern.match(name)
            if not match:
                print(f"  SKIP (no match): {sub}/{fname}")
                continue

            old_pid = int(match.group(1))
            suffix = match.group(2)
            new_pid = old_pid + (start_number - 1)  # p1 -> start_number
            new_name = f"p{new_pid}_{suffix}{ext}"
            rename_plan.append((sub_path, fname, new_name))

    if not rename_plan:
        print("Nothing to rename.")
        return

    # Show preview
    print(f"\n  {len(rename_plan)} files to rename:\n")
    for sub_path, old, new in rename_plan[:10]:
        sub = os.path.basename(sub_path)
        print(f"    {sub}/{old}  ->  {sub}/{new}")
    if len(rename_plan) > 10:
        print(f"    ... and {len(rename_plan) - 10} more")

    confirm = input("\n  Proceed? (y/n): ").strip().lower()
    if confirm != "y":
        print("  Aborted.")
        return

    # Rename in two passes to avoid collisions (old name -> temp -> new name)
    # Pass 1: rename to temp names
    temp_plan = []
    for sub_path, old, new in rename_plan:
        temp_name = f"__temp__{old}"
        os.rename(os.path.join(sub_path, old), os.path.join(sub_path, temp_name))
        temp_plan.append((sub_path, temp_name, new))

    # Pass 2: rename temp to final
    for sub_path, temp, new in temp_plan:
        os.rename(os.path.join(sub_path, temp), os.path.join(sub_path, new))

    print(f"\n  Done. Renamed {len(rename_plan)} files.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python rename_dataset.py <folder_path> <start_number>")
        print("Example: python rename_dataset.py ./tournament2 632")
        sys.exit(1)

    folder = sys.argv[1]
    start = int(sys.argv[2])

    if not os.path.isdir(folder):
        print(f"Error: {folder} is not a directory.")
        sys.exit(1)

    print(f"\n  Folder: {folder}")
    print(f"  p1 -> p{start}, p2 -> p{start + 1}, ...")

    rename_files(folder, start)
