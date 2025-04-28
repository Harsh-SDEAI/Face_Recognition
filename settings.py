from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

# Database
SERVER_NAME = os.getenv('DB_SERVER')
DRIVER_NAME = os.getenv('DB_DRIVER')
DATABASE_NAME = os.getenv('DB_NAME')
USER_NAME = os.getenv('DB_USER')
PASSWORD = os.getenv('DB_PASSWORD')

# Paths
LOCAL_IMAGE_PATH = os.getenv('LOCAL_IMAGE_PATH')
DB_PREFIX_PATH = os.getenv('DB_PREFIX_PATH')

# Parameters
MIN_FACE_SIZE = int(os.getenv('MIN_FACE_SIZE', 30))
IDLE_SLEEP_TIME = int(os.getenv('IDLE_SLEEP_TIME', 60))
PROCESSING_SLEEP_TIME = int(os.getenv('PROCESSING_SLEEP_TIME', 5))
RETRY_COUNT = int(os.getenv('RETRY_COUNT', 3))
PROCESSING_TIMESTAMP = int(os.getenv('PROCESSING_TIMESTAMP', 30))
MARGIN = int(os.getenv('MARGIN', 44))

# Face Detection Thresholds
THRESHOLD_STUDIO = float(os.getenv('THRESHOLD_STUDIO', 0.70))
THRESHOLD_GAME = float(os.getenv('THRESHOLD_GAME', 0.96))

# EasyOCR
EASYOCR_LANGUAGES = os.getenv('EASYOCR_LANGUAGES', 'en').split(',')
EASYOCR_GPU = os.getenv('EASYOCR_GPU', 'False').lower() in ('true', '1', 't')

# Jersey detection
JERSEY_CONFIDENCE_THRESHOLD = float(os.getenv('JERSEY_CONFIDENCE_THRESHOLD', 0.90))

# Matching threshold range
DEFAULT_EUCLIDEAN_THRESHOLD_MIN = float(os.getenv('DEFAULT_EUCLIDEAN_THRESHOLD_MIN', 0.5))
DEFAULT_EUCLIDEAN_THRESHOLD_MAX = float(os.getenv('DEFAULT_EUCLIDEAN_THRESHOLD_MAX', 1.5))
