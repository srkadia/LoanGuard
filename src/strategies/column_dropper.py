import pandas as pd
from abc import ABC, abstractmethod
from utils.logger import Logger
from utils.exception import CustomException

logger = Logger(__name__).get_logger()


# --- Abstract Base Class ---
class ColumnDropperStrategy(ABC):
    """
    Abstract base class for strategies that involve dropping columns from the DataFrame.
    """

    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to drop columns from the DataFrame.

        :param df: The input DataFrame to clean.
        :return: DataFrame after dropping specified columns.
        """
        pass


# --- Concrete Class ---
class ColumnDropper(ColumnDropperStrategy):
    """
    Concrete class for dropping specified columns from the DataFrame.
    """

    def __init__(self, columns_to_drop: list = None):
        """
        Initializes the DropMissingValuesStrategy with specific parameters.

        :param columns_to_drop: List of column names to drop (default: None, can be set dynamically).
        """
        if not isinstance(columns_to_drop, list):
            raise CustomException("The 'columns_to_drop' must be a list.")
        if not columns_to_drop:
            raise CustomException("The 'columns_to_drop' list cannot be empty.")
        self.columns_to_drop = columns_to_drop

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops the specified columns.

        :param df: The DataFrame to clean.
        :return: DataFrame with specified columns dropped.
        """
        if not isinstance(df, pd.DataFrame):
            raise CustomException("The input data must be a pandas DataFrame.")

        existing_cols_to_drop = [col for col in self.columns_to_drop if col in df.columns]

        if not existing_cols_to_drop:
            logger.warning("No columns to drop found in the DataFrame.")
            return df

        try:
            df.drop(columns=existing_cols_to_drop, axis=1, inplace=True)
            logger.info(f"Dropped columns: {existing_cols_to_drop}")
        except Exception as e:
            raise CustomException(f"Error while dropping columns: {str(e)}")

        return df


# --- Factory Class ---
class ColumnDropperFactory:
    """
    Factory for applying column dropping strategies to the DataFrame.
    """

    def __init__(self, config: dict):
        """
        Initialize the factory with configuration for dropping columns.

        :param config: Configuration dict containing 'columns_to_drop' key with list of columns.
        """
        if not isinstance(config, dict):
            raise CustomException("The config must be a dictionary.")

        self.column_dropper = config.get('columns_to_drop', [])

    def create(self) -> ColumnDropper:
        """
        Creates and returns an instance of ColumnDropper with the provided columns configuration to drop.

        :return: An instance of ColumnDropper.
        """
        if not isinstance(self.column_dropper, list):
            raise CustomException("The 'columns_to_drop' in config must be a list.")
        return ColumnDropper(self.column_dropper)
