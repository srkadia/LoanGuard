import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from utils.logger import Logger
from utils.exception import CustomException

logger = Logger(__name__).get_logger()


# --- Abstract Base Class ---
class MissingValueHandlerStrategy(ABC):
    """
    Abstract base class for handling missing values in the DataFrame.
    """

    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method for handling missing values in the DataFrame.

        :param df: The input DataFrame to clean.
        :return: Transformed DataFrame with missing values handled.
        """
        pass


# --- Concrete Class ---
class MissingValueHandler(MissingValueHandlerStrategy):
    """
    Concrete class for handling missing values in the dataset.
    """

    def __init__(self):
        pass

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        # Check if input is a valid DataFrame
        if not isinstance(df, pd.DataFrame):
            raise CustomException("The input data must be a pandas DataFrame.")

        # Log missing values in each column
        for column in df.columns:
            missing = df[column].isna().sum()
            if missing > 0:
                portion = (missing / df.shape[0]) * 100
                logger.warning(f"'{column}': number of missing values '{missing}' ==> '{portion:.3f}%'")

        # Handle 'mort_acc' column based on 'total_acc' column
        if 'total_acc' in df.columns and 'mort_acc' in df.columns:
            total_acc_avg = df.groupby(by='total_acc')['mort_acc'].mean()

            def fill_mort_acc(total_acc, mort_acc):
                if np.isnan(mort_acc):
                    return total_acc_avg.get(total_acc, np.nan)  # Handle missing 'total_acc' values
                else:
                    return mort_acc

            df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
        elif 'total_acc' not in df.columns:
            raise CustomException("'total_acc' column is missing in the DataFrame.")
        elif 'mort_acc' not in df.columns:
            raise CustomException("'mort_acc' column is missing in the DataFrame.")

        # Drop remaining missing values in the DataFrame
        df.dropna(inplace=True)
        logger.info(f"Data shape after handling missing values: {df.shape}")

        return df


# --- Factory Class ---
class MissingValueHandlerFactory:
    """
    Factory for applying missing value handling strategies to the DataFrame.
    """

    def __init__(self, config: dict):
        # Validate if the config is a dictionary
        if not isinstance(config, dict):
            raise CustomException("The config must be a dictionary.")

        # No specific config values used here, but you can extend it later.
        self.config = config

    def create(self) -> MissingValueHandler:
        """
        Creates and returns an instance of MissingValueHandler.

        :return: An instance of MissingValueHandler.
        """
        return MissingValueHandler()

    def apply_strategies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply missing value handling strategies.

        :param df: DataFrame to apply strategies to.
        :return: DataFrame after missing value handling.
        """
        # Validate if the input is a DataFrame
        if not isinstance(df, pd.DataFrame):
            raise CustomException("The input data must be a pandas DataFrame.")

        missing_value_handler = MissingValueHandler()
        df = missing_value_handler.handle(df)
        return df
