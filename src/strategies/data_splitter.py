import pandas as pd
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from utils.logger import Logger

logger = Logger(__name__).get_logger()


# --- Abstract Base Class ---
class DataSplittingStrategy(ABC):
    """
    Abstract base class for defining data splitting strategies.
    """

    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str):
        """
        Abstract method for splitting the dataset.

        :param df: The input DataFrame to be split.
        :param target_column: The name of the target variable column.

        :return: X_train, X_test, y_train, y_test
        """
        pass


# --- Concrete Strategy: Simple Train-Test Split ---
class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    """
    Concrete strategy that implements a simple train-test split.
    """

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_column: str):
        logger.info("Applying Simple Train-Test Split Strategy.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        logger.info("Train-test split completed.")
        return X_train, X_test, y_train, y_test


# --- Concrete Strategy: Stratified Train-Test Split ---
class StratifiedTrainTestSplitStrategy(DataSplittingStrategy):
    """
    Concrete strategy that ensures stratified sampling when splitting the data.
    """

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_column: str):
        logger.info("Applying Stratified Train-Test Split Strategy.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        logger.info("Stratified Train-Test Split completed.")
        return X_train, X_test, y_train, y_test


# --- Factory Class ---
class DataSplitterFactory:
    """
    Factory class to create data splitting strategy instances dynamically.
    """

    @staticmethod
    def get_splitter(strategy_type: str, **kwargs) -> DataSplittingStrategy:
        """
        Returns the appropriate data splitting strategy instance.

        :param strategy_type: The type of strategy ('simple', 'stratified').
        :param kwargs: Additional parameters for the strategy.

        :return: An instance of the selected DataSplittingStrategy.
        """
        strategies = {
            "simple": SimpleTrainTestSplitStrategy,
            "stratified": StratifiedTrainTestSplitStrategy,
        }

        if strategy_type not in strategies:
            raise ValueError(f"Invalid strategy type: {strategy_type}. Choose from {list(strategies.keys())}.")

        return strategies[strategy_type](**kwargs)


# --- Example Usage ---
# if __name__ == "__main__":
#     # Example dataset (replace with actual data)
#     df = pd.read_csv("../../data/raw/data.csv")
#
#     # Select splitting strategy dynamically
#     splitter = DataSplitterFactory.get_splitter(strategy_type="simple", test_size=0.33, random_state=42)
#     X_train, X_test, y_train, y_test = splitter.split_data(df, target_column="loan_status")
#
#     print(f"Train Shape -> {X_train.shape}")
#     print(f"Test Shape -> {X_test.shape}")
