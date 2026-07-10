"""
Image health checker (read-only diagnostic tool).

Walks the folders listed in CHECK_FOLDERS (from .env) and tries to open and
fully decode every image the exact same way AIPhotoMatch does:

    1. os.path.exists()        (same as detect_align_embed_* line 199/109)
    2. PIL.Image.open()        (same as line 203)
    3. full pixel decode       (same failure point as mtcnn.detect -> np.uint8(img),
                                which forces PIL's lazy load at ImageFile.load)

Nothing is written anywhere - no database, no files. Results go to the console.

.env entry (paths separated by ';' since Windows paths contain ':' and '\\'):

    CHECK_FOLDERS=\\\\172.16.17.136\\PHOTO_ROOT\\2026\\hires\\359\\Game\\375874;\\\\172.16.17.136\\PHOTO_ROOT\\2026\\hires\\359\\Game\\375877

Run:  python check_images.py
"""

import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

load_dotenv()

# Same extensions the service processes (Constellation rows use MediaType = '.jpg')
IMAGE_EXTENSIONS = ('.jpg', '.jpeg')


def file_info(path):
    """Size + last-modified, best effort. Helps spot files still being copied."""
    try:
        st = os.stat(path)
        size_mb = st.st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        return f"size={size_mb:.2f} MB, modified={mtime}"
    except OSError as e:
        return f"stat failed: {e}"


def check_image(path):
    """Open and fully decode one image the same way the service does.
    Returns (ok, problem, detail)."""
    if not os.path.exists(path):
        return False, "MISSING", "os.path.exists() returned False"
    try:
        image = Image.open(path)          # fails here -> corrupt header / locked at open
        image.load()                      # forces full pixel decode, same as np.uint8(img) in mtcnn
        return True, None, None
    except PermissionError as e:
        return False, "LOCKED", f"Permission denied - another process holds a write lock ({e})"
    except UnidentifiedImageError as e:
        return False, "CORRUPT-HEADER", f"PIL cannot identify the file as an image ({e})"
    except FileNotFoundError as e:
        return False, "MISSING", f"File vanished between listing and open ({e})"
    except OSError as e:
        # 'broken data stream' / 'image file is truncated' land here
        return False, "TRUNCATED", f"{e}"
    except Exception as e:
        return False, "OTHER", f"{type(e).__name__}: {e}"


def main():
    raw = os.getenv('CHECK_FOLDERS', '')
    folders = [f.strip() for f in raw.split(';') if f.strip()]
    if not folders:
        print("No folders configured. Add CHECK_FOLDERS to your .env, e.g.")
        print(r"  CHECK_FOLDERS=\\server\share\folder1;\\server\share\folder2")
        sys.exit(1)

    print(f"Checking {len(folders)} folder(s) - started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    total_ok = 0
    bad_files = []

    for folder in folders:
        print(f"\nFOLDER: {folder}")
        if not os.path.isdir(folder):
            print("  !! Folder not found or not accessible - skipping")
            continue

        folder_ok = 0
        folder_bad = 0
        for root, _dirs, files in os.walk(folder):
            for name in sorted(files):
                if not name.lower().endswith(IMAGE_EXTENSIONS):
                    continue
                path = os.path.join(root, name)
                start = time.time()
                ok, problem, detail = check_image(path)
                elapsed = time.time() - start
                if ok:
                    folder_ok += 1
                    total_ok += 1
                else:
                    folder_bad += 1
                    bad_files.append((folder, path, problem))
                    print(f"  [{problem}] {path}")
                    print(f"      -> {detail}")
                    print(f"      -> {file_info(path)} (read attempt took {elapsed:.1f}s)")

        print(f"  Folder summary: {folder_ok} OK, {folder_bad} with issues")

    print("\n" + "=" * 100)
    print(f"DONE. Total OK: {total_ok}, total with issues: {len(bad_files)}")
    if bad_files:
        print("\nAll problem files:")
        for folder, path, problem in bad_files:
            print(f"  [{problem}] {path}")
    else:
        print("All images opened and decoded cleanly - no corrupt or locked files found.")


if __name__ == "__main__":
    main()
