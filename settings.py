from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

# Database
SERVER_NAME_DPP = os.getenv('DB_SERVER_DPP')
DRIVER_NAME_DPP = os.getenv('DB_DRIVER_DPP')
DATABASE_NAME_DPP = os.getenv('DB_NAME_DPP')
USER_NAME_DPP = os.getenv('DB_USER_DPP')
PASSWORD_DPP = os.getenv('DB_PASSWORD_DPP')

# Database
SERVER_NAME_CDPMC = os.getenv('DB_SERVER_CDPMC')
DRIVER_NAME_CDPMC = os.getenv('DB_DRIVER_CDPMC')
DATABASE_NAME_CDPMC = os.getenv('DB_NAME_CDPMC')
USER_NAME_CDPMC = os.getenv('DB_USER_CDPMC')
PASSWORD_CDPMC = os.getenv('DB_PASSWORD_CDPMC')

SERVER_NAME_CDP2000 = os.getenv('DB_SERVER_CDP2000')
DRIVER_NAME_CDP2000 = os.getenv('DB_DRIVER_CDP2000')
DATABASE_NAME_CDP2000 = os.getenv('DB_NAME_CDP2000')
USER_NAME_CDP2000 = os.getenv('DB_USER_CDP2000')
PASSWORD_CDP2000 = os.getenv('DB_PASSWORD_CDP2000')

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
# EASYOCR_LANGUAGES = os.getenv('EASYOCR_LANGUAGES', 'en').split(',')
# EASYOCR_GPU = os.getenv('EASYOCR_GPU', 'False').lower() in ('true', '1', 't')

# Jersey detection
#JERSEY_CONFIDENCE_THRESHOLD = float(os.getenv('JERSEY_CONFIDENCE_THRESHOLD', 0.90))

# Matching threshold range
DEFAULT_EUCLIDEAN_THRESHOLD_MIN = float(os.getenv('DEFAULT_EUCLIDEAN_THRESHOLD_MIN', 0.5))
DEFAULT_EUCLIDEAN_THRESHOLD_MAX = float(os.getenv('DEFAULT_EUCLIDEAN_THRESHOLD_MAX', 1.5))

# Logging-related settings
LOG_FOLDER_PATH = os.getenv("LOG_FOLDER_PATH", "logs")
LOG_ARCHIVE_PATH = os.getenv("LOG_ARCHIVE_PATH", "log_archive")
LOG_MAX_BYTES = log_max_bytes = int(float(os.getenv("LOG_MAX_BYTES_MB", 1)) * 1024 * 1024)  # MB to Bytes

# Scheduler time interval
TASK_RUN_INTERVAL_MINUTES = int(os.getenv("TASK_RUN_INTERVAL_MINUTES", 20))
INSTANCE = int(os.getenv("INSTANCE", 2)) 
GAMES_PER_SCHEDULER = int(os.getenv("GAMES_PER_SCHEDULER", 25)) 