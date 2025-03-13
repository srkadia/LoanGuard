import logging
import os
from datetime import datetime

class Logger:
    """
    A configurable logging utility class for structured and efficient logging.
    This class ensures logging is set up with both file and console handlers.
    """
    
    LOG_DIR = "logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    
    def __init__(self, name: str, log_level=logging.DEBUG, file_logging=True):
        """
        Initialize the logger with the specified name and logging level.
        
        :param name: Name of the logger (usually module name)
        :param log_level: Logging level (default: DEBUG)
        :param file_logging: Enable or disable file logging (default: True)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Prevent adding multiple handlers to the logger
        if not self.logger.handlers:
            # Console Handler (for real-time logging in terminal)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
            # File Handler (for persistent logging)
            if file_logging:
                log_file_name = os.path.join(self.LOG_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")
                file_handler = logging.FileHandler(log_file_name, mode='a')
                file_handler.setLevel(logging.DEBUG)
                file_formatter = logging.Formatter("[ %(asctime)s ] - %(lineno)d %(name)s - %(levelname)s - %(message)s")
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
    
    def get_logger(self):
        """Returns the configured logger instance."""
        return self.logger

# Usage Example:
# if __name__ == "__main__":
#     logger = Logger(__name__).get_logger()
#     logger.info("This is an info message")
#     logger.debug("This is a debug message")
#     logger.error("This is an error message")
