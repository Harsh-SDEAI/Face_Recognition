"""Central config - loads .env and exposes typed constants."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the face_eval folder regardless of where python was launched
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ---- Database ----
DB_SERVER = os.getenv("DB_SERVER")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")

# ---- Paths ----
ALGOTEST_ROOT = Path(os.getenv("ALGOTEST_ROOT", r"C:\AlgoTest"))
STUDIO_ROOT = ALGOTEST_ROOT / "Studio"
GAME_ROOT = ALGOTEST_ROOT / "Game"

MODEL_DIR = BASE_DIR / os.getenv("MODEL_DIR", "models")
FACE_CROPS_DIR = BASE_DIR / os.getenv("FACE_CROPS_DIR", "face_crops")
FACE_CROPS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Weight filenames (must match what the user downloads / what LVFace pulls) ----
ARCFACE_WEIGHTS = MODEL_DIR / "arcface_ir100_ms1mv3.pth"
ADAFACE_WEIGHTS = MODEL_DIR / "adaface_ir101_webface12m.ckpt"
MAGFACE_WEIGHTS = MODEL_DIR / "magface_epoch_00025.pth"
LVFACE_WEIGHTS = MODEL_DIR / "lvface.pt"

# ---- LVFace HF Hub (PyTorch .pt, loaded via cloned LVFace repo's backbones module) ----
LVFACE_HF_REPO = os.getenv("LVFACE_HF_REPO", "bytedance-research/LVFace")
LVFACE_HF_FILENAME = os.getenv("LVFACE_HF_FILENAME", "LVFace-B_WebFace4M.pt")
LVFACE_MODEL_NAME = os.getenv("LVFACE_MODEL_NAME", "vit_b")
# Path to a local clone of https://github.com/bytedance/LVFace - loader will
# add this to sys.path so it can `from backbones import get_model`.
LVFACE_REPO_DIR = os.getenv("LVFACE_REPO_DIR", "")

# ---- Detection ----
MTCNN_THRESHOLD = float(os.getenv("MTCNN_THRESHOLD", 0.85))
MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", 30))

# ---- Model registry ----
MODEL_NAMES = ["facenet", "arcface", "adaface", "magface", "lvface"]

# ---- Default similarity threshold range (0-1 cosine) ----
DEFAULT_COSINE_THRESHOLD_MIN = 0.20
DEFAULT_COSINE_THRESHOLD_MAX = 0.80
DEFAULT_COSINE_THRESHOLD = 0.40
