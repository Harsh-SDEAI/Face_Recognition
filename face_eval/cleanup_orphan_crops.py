# cleanup_orphan_crops.py  (put it next to generate_embeddings.py)
from pathlib import Path
from utils.db import connect
import config

conn = connect()
cur = conn.cursor()
cur.execute("SELECT FaceCropPath FROM EvalFaceDetection WHERE FaceCropPath IS NOT NULL")
referenced = {Path(row[0]).name for row in cur.fetchall()}
cur.close(); conn.close()

folder = Path(config.FACE_CROPS_DIR)
all_files = {p.name for p in folder.glob("*.png")}
orphans = all_files - referenced

print(f"Referenced: {len(referenced)}  Total on disk: {len(all_files)}  Orphans: {len(orphans)}")
for name in orphans:
    (folder / name).unlink()
print("Done.")