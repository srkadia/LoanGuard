import sys
import traceback
from utils.logger import Logger

# Initialize Logger
logger = Logger(__name__).get_logger()

class CustomException(Exception):
    """
    Custom Exception class that logs errors with file name & line number automatically.
    """

    def __init__(self, error: Exception):
        """
        Captures detailed error info and logs it automatically.

        :param error: The caught exception object.
        """
        super().__init__(str(error))  # Convert exception object to string
        self.error_message = self.get_detailed_error(error)
        logger.error(self.error_message)  # Auto-log the error

    @staticmethod
    def get_detailed_error(error: Exception) -> str:
        """
        Extracts detailed error message along with stack trace.
        """
        return f"Exception: {error}\nTraceback:\n{traceback.format_exc()}"

    def __str__(self) -> str:
        return self.error_message
