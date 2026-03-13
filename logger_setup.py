import logging
import os
import sys
import re
from pathlib import Path
from datetime import datetime
from concurrent_log_handler import ConcurrentRotatingFileHandler
import settings

def init_logger():
    logger = logging.getLogger()

    # Avoid reinitialization
    if logger.hasHandlers():
        return

    # === Load configuration from settings ===
    log_folder = settings.LOG_FOLDER_PATH
    archive_folder = settings.LOG_ARCHIVE_PATH
    log_max_bytes = settings.LOG_MAX_BYTES  # Already in bytes

    Path(log_folder).mkdir(parents=True, exist_ok=True)
    Path(archive_folder).mkdir(parents=True, exist_ok=True)

    info_log_path = os.path.join(log_folder, "info.log")
    error_log_path = os.path.join(log_folder, "error.log")

    # === Determine next archive index ===
    def get_next_index(log_type):
        pattern = re.compile(rf"{log_type}_(\d+)\.log$")
        existing = [
            int(pattern.match(f).group(1))
            for f in os.listdir(archive_folder)
            if pattern.match(f)
        ]
        return max(existing, default=0) + 1

    # === Custom rotating handler ===
    class CustomArchiveHandler(ConcurrentRotatingFileHandler):
        def __init__(self, base_filename, log_type):
            self.log_type = log_type
            self.base_filename = base_filename
            super().__init__(base_filename, maxBytes=log_max_bytes, backupCount=0)

        def doRollover(self):
            if self.stream:
                self.stream.close()
                self.stream = None
            next_index = get_next_index(self.log_type)
            archive_name = f"{self.log_type}_{next_index}.log"
            archive_path = os.path.join(archive_folder, archive_name)
            os.rename(self.baseFilename, archive_path)
            super().doRollover()

    # === Log Filters ===
    class MaxLevelFilter(logging.Filter):
        def __init__(self, level): self.level = level
        def filter(self, record): return record.levelno <= self.level

    class MinLevelFilter(logging.Filter):
        def __init__(self, level): self.level = level
        def filter(self, record): return record.levelno >= self.level

    # === Formatter ===
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # === File Handlers ===
    info_handler = CustomArchiveHandler(info_log_path, "info")
    info_handler.setFormatter(formatter)
    info_handler.setLevel(logging.DEBUG)
    info_handler.addFilter(MaxLevelFilter(logging.INFO))

    error_handler = CustomArchiveHandler(error_log_path, "error")
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    error_handler.addFilter(MinLevelFilter(logging.ERROR))

    # === Console Handler (new) ===
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)

    # === Register Handlers ===
    logger.setLevel(logging.DEBUG)
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    # === Redirect print() to logger ===
    class StreamToLogger:
        def __init__(self, logger, level):
            self.logger = logger
            self.level = level

        def write(self, message):
            message = message.strip()
            if message:
                self.logger.log(self.level, message)

        def flush(self): pass

    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)
