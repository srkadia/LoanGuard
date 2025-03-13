import sys
from utils.logger import Logger

# Initialize Logger
logger = Logger(__name__).get_logger()

class CustomException(Exception):
    """
    Custom Exception class that logs errors with file name & line number automatically.
    """

    def __init__(self, error: Exception, error_detail: sys):
        """
        Captures detailed error info and logs it automatically.
        :param error: The caught exception object.
        :param error_detail: Always pass sys to capture traceback.
        """
        super().__init__(str(error))  # Convert exception object to string
        self.error_message = self.get_detailed_error(error, error_detail)
        logger.error(self.error_message)  # Auto-log the error

    @staticmethod
    def get_detailed_error(error: Exception, error_detail: sys) -> str:
        """Extracts script name & line number for better debugging."""
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        return f"Exception in [ {file_name} ] at line [{line_number}]: {error}"

    def __str__(self) -> str:
        return self.error_message
