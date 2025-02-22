import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist 
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Define log file name with datestamp
log_file_name = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

# another version of the log file name for different logging of each session
# log_file_name = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

# Configure Logging
logging.basicConfig(
    filename=log_file_name,
    level=logging.DEBUG,
    format="[ %(asctime)s ] - %(lineno)d %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_logger(name):
    """Returns a configured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # console handler for real-time logs in terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # Prevent duplicate logs
    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger
