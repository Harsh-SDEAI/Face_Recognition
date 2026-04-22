"""
Rename anchor files to remove the redundant anchor number.

P1_A1.jpg  -> P1_A.jpg
P124_A124.jpg -> P124_A.jpg
"""

import os
import re

# ============================================================
# CONFIGURE BEFORE RUNNING
# ============================================================
FOLDER_PATH = "/path/to/anchors/folder"
# ============================================================


def rename_anchors(folder_path: str):
    pattern = re.compile(r"^([Pp]\d+_[Aa])\d+(\..+)$")

    rename_plan = []
    for fname in sorted(os.listdir(folder_path)):
        match = pattern.match(fname)
        if not match:
            print(f"  SKIP (no match): {fname}")
            continue

        new_name = f"{match.group(1)}{match.group(2)}"
        if fname == new_name:
            continue
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

    for old, new in rename_plan:
        os.rename(os.path.join(folder_path, old), os.path.join(folder_path, new))

    print(f"\n  Done. Renamed {len(rename_plan)} files.")


if __name__ == "__main__":
    if not os.path.isdir(FOLDER_PATH):
        print(f"Error: {FOLDER_PATH} is not a directory. Update FOLDER_PATH at the top of the script.")
        raise SystemExit(1)

    rename_anchors(FOLDER_PATH)
