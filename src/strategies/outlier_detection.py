import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from utils.logger import Logger
from utils.exception import CustomException

# Initialize Logger
logger = Logger(__name__).get_logger()


# --- Abstract Base Class ---
class OutlierHandlerStrategy(ABC):
    """
    Abstract base class for outlier detection strategies.
    Defines the structure for outlier handling methods to be implemented by concrete strategies.
    """

    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method for detecting and handling outliers in the DataFrame.

        :param df: The input DataFrame to clean.
        :return: Transformed DataFrame with outliers handled.
        """
        pass


# --- Concrete Class ---
class OutlierHandler(OutlierHandlerStrategy):
    """
    Concrete class for handling outliers in numeric columns by capping values at a specific quantile or custom value.
    """

    def __init__(self, columns: dict, quantile: float = 0.95):
        """
        Initializes the outlier handler with the specified columns to process and the quantile for outlier capping.

        :param columns: A dictionary where keys are column names and values are their respective custom cap values.
        :param quantile: The quantile value used to cap outliers (default is 0.95, which represents the 95th percentile).
        """
        if not isinstance(columns, dict):
            raise CustomException("The 'columns' parameter should be a dictionary with column names as keys and cap values as values.")

        self.columns = columns
        self.quantile = quantile

        if not (0 < quantile < 1):
            raise CustomException("The 'quantile' parameter should be between 0 and 1.")

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles outliers by either capping values in the specified columns to the upper quantile or the custom cap value.

        :param df: The input DataFrame to clean.
        :return: Transformed DataFrame with outliers capped.
        """
        # Check if input is a valid DataFrame
        if not isinstance(df, pd.DataFrame):
            raise CustomException("The input data must be a pandas DataFrame.")

        for col, cap_value in self.columns.items():
            if col in df.columns:
                # If a custom cap value is provided, cap the values directly at that cap value
                if cap_value is not None:
                    df[col] = np.where(df[col] > cap_value, cap_value, df[col])
                    logger.info(f"Outliers in column '{col}' capped at custom value: {cap_value:.2f}")
                else:
                    # Compute the upper quantile (threshold for capping) if quantile is used
                    upper_limit = df[col].quantile(self.quantile)

                    # Cap values above the upper limit to the upper limit
                    df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])

                    # Log the capping process
                    logger.info(
                        f"Outliers in column '{col}' capped at {self.quantile * 100}th percentile value: {upper_limit:.2f}")
            else:
                logger.warning(f"Column '{col}' not found in the DataFrame.")

        return df


# --- Factory Class ---
class OutlierHandlerFactory:
    """
    Factory class for applying outlier detection and handling strategies to the DataFrame.
    This factory creates and applies the OutlierHandler to manage outliers in specified columns.
    """

    def __init__(self, config: dict):
        """
        Initializes the factory with the input configuration settings.

        :param config: A dictionary containing configuration settings for outlier detection.
                       It should have the 'outlier_cols' key with a dictionary of columns and their respective cap values.
                       Optionally, 'outlier_quantile' can be provided for the quantile-based capping.
        """
        if not isinstance(config, dict):
            raise CustomException("The config must be a dictionary.")

        self.columns = config.get('outlier_cols', {})
        self.quantile = config.get('outlier_quantile', 0.95)

        if not self.columns:
            raise CustomException("The 'outlier_cols' dictionary cannot be empty.")

        if not all(isinstance(col, str) for col in self.columns.keys()):
            raise CustomException("All keys in the 'outlier_cols' dictionary must be column names (strings).")

        if not (0 < self.quantile < 1):
            raise CustomException("The 'outlier_quantile' value must be between 0 and 1.")

    def create(self) -> OutlierHandler:
        """
        Creates and returns an instance of OutlierHandler.

        :return: An instance of OutlierHandler.
        """
        return OutlierHandler(self.columns, self.quantile)

    def apply_strategies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply outlier handling strategies to the DataFrame.

        :param df: DataFrame to apply strategies to.
        :return: DataFrame after outlier handling.
        """
        # Validate if the input is a DataFrame
        if not isinstance(df, pd.DataFrame):
            raise CustomException("The input data must be a pandas DataFrame.")

        outlier_handler = OutlierHandler(self.columns, self.quantile)
        df = outlier_handler.handle(df)
        return df
