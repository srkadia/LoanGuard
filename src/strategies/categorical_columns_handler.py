import pandas as pd
from abc import ABC, abstractmethod
from utils.logger import Logger
from utils.exception import CustomException

logger = Logger(__name__).get_logger()


# --- Abstract Base Class ---
class CategoricalHandlerStrategy(ABC):
    """
    Abstract base class for handling categorical variables in the DataFrame.
    """

    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method for handling categorical variables.

        :param df: The input DataFrame to clean.
        :return: Transformed DataFrame with categorical variables handled.
        """
        pass


# --- Concrete Class ---
class CategoricalHandler(CategoricalHandlerStrategy):
    """
    Concrete class for handling categorical variables via one-hot encoding.
    """

    def __init__(self, categorical_columns: list, drop_first: bool = True):
        # Check if categorical_columns is a list
        if not isinstance(categorical_columns, list):
            raise CustomException("The 'categorical_columns' must be a list.")
        if not categorical_columns:
            raise CustomException("The 'categorical_columns' list cannot be empty.")

        self.categorical_columns = categorical_columns
        self.drop_first = drop_first

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        # Validate that df is a pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            raise CustomException("The input data must be a pandas DataFrame.")

        # Drop 'grade' column if it exists
        if 'grade' in df.columns:
            df.drop(columns=['grade'], inplace=True)
            logger.info("Dropped column: 'grade'")

        # Check if the specified categorical columns exist in the DataFrame
        missing_cols = [col for col in self.categorical_columns if col not in df.columns]
        if missing_cols:
            raise CustomException(f"Missing columns in the DataFrame: {', '.join(missing_cols)}")

        # One-hot encode categorical columns
        df = pd.get_dummies(df, columns=self.categorical_columns, drop_first=self.drop_first)
        logger.info(f"One-hot encoded columns: {self.categorical_columns}")
        return df


# --- Factory Class ---
class CategoricalHandlerFactory:
    """
    Factory for applying categorical handling strategies to the DataFrame.
    """

    def __init__(self, config: dict):
        # Validate config to ensure it's a dictionary
        if not isinstance(config, dict):
            raise CustomException("The config must be a dictionary.")

        self.categorical_columns = config.get('categorical_columns', [])

        # Validate that 'categorical_columns' is a list
        if not isinstance(self.categorical_columns, list):
            raise CustomException("The 'categorical_columns' must be a list.")

    def create(self) -> CategoricalHandler:
        """
        Creates and returns an instance of CategoricalHandler with the provided column's configuration.

        :return: An instance of CategoricalHandler.
        """
        return CategoricalHandler(self.categorical_columns)
