import os
import logging
from datetime import datetime

LOG_DIRS = "logs"
os.makedirs(LOG_DIRS, exist_ok=True)

LOG_FILE = f"log_{datetime.now().strftime('%Y%m%d-%h%M%S')}.log" # %h = 12 hour gormat and %H = 24 hours format
LOG_PATH = os.path.join(LOG_DIRS, LOG_FILE)

logging.basicConfig(
    filename=LOG_PATH,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    level=logging.INFO,
)

def get_logger():
    return logging.getLogger()