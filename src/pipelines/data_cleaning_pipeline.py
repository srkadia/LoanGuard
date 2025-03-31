import os
import pandas as pd

from strategies.handle_missing_values import MissingValueHandlerFactory
from strategies.feature_engineering import (
    NumericExtractorFactory, ZipCodeHandlerFactory,
    DateFeatureExtractorFactory, TargetEncoderFactory
)
from strategies.categorical_columns_handler import CategoricalHandlerFactory
from strategies.outlier_detection import OutlierHandlerFactory
from strategies.column_dropper import ColumnDropperFactory
from utils.config import ConfigLoader
from utils.exception import CustomException
from utils.logger import Logger

# Initialize Logger
logger = Logger(__name__).get_logger()

class DataCleaningPipeline:
    """
    A pipeline that applies a sequence of data cleaning strategies.
    The pipeline executes strategies in a configurable order to process raw data.
    """

    def __init__(self, df: pd.DataFrame, config: dict):
        """
        Initializes the data cleaning pipeline.

        :param df: The input DataFrame containing raw data.
        :param config: A dictionary specifying configurations for different strategies.
        """
        self.df = df
        self.config = config

    def apply_strategies(self):
        """
        Applies the configured data cleaning strategies in sequence.

        The steps are predefined in the order they should be applied.
        """
        strategy_steps = [
            (ColumnDropperFactory, "clean"),
            (NumericExtractorFactory, "apply_transformation"),
            (MissingValueHandlerFactory, "handle"),
            (OutlierHandlerFactory, "handle"),
            (CategoricalHandlerFactory, "handle"),
            (ZipCodeHandlerFactory, "apply_transformation"),
            (DateFeatureExtractorFactory, "apply_transformation"),
            (TargetEncoderFactory, "apply_transformation")
        ]

        for factory_class, method_name in strategy_steps:
            self.df = self._apply_strategy(factory_class, method_name)

        return self.df

    def _apply_strategy(self, factory_class, method_name):
        """
        Applies a single data cleaning strategy.

        :param factory_class: The factory class responsible for creating the strategy instance.
        :param method_name: The method to be invoked on the strategy instance.
        :return: The transformed DataFrame after applying the strategy.
        """
        try:
            factory = factory_class(self.config)  # Instantiate the factory with config
            strategy_instance = factory.create()  # Create a strategy instance
            self.df = getattr(strategy_instance, method_name)(self.df)  # Apply the transformation
            logger.info(f"Data shape after {factory_class.__name__}: {self.df.shape}")
        except CustomException as e:
            logger.error(f"Error applying {factory_class.__name__}: {e}")
        return self.df

if __name__ == "__main__":
    # # Define input and output file paths
    # RAW_DATA_PATH = os.path.join('data', 'raw', 'data.csv')
    # PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'data_cleaned.csv')
    #
    # # Configuration settings for the pipeline
    # config = {
    #     'columns_to_drop': ['emp_title', 'emp_length', 'title', 'issue_d', 'grade'],
    #     'numeric_transformations': [
    #         {'column_name': 'term', 'pattern': r'(\d+)', 'default_value': 0},
    #         {'column_name': 'int_rate', 'pattern': r'(\d+(\.\d+)?)', 'remove_chars': '%', 'default_value': 0},
    #         {'column_name': 'revol_util', 'pattern': r'(\d+(\.\d+)?)', 'remove_chars': '%', 'default_value': 0}
    #     ],
    #     'outlier_cols': {
    #         'annual_inc': 250000,
    #         'dti': 50,
    #         'open_acc': 40,
    #         'total_acc': 80,
    #         'revol_util': 120,
    #         'revol_bal': 250000
    #     },
    #     'outlier_quantile': 0.95,
    #     'date_columns': {'earliest_cr_line': '%b-%Y'},
    #     'categorical_columns': ['sub_grade', 'verification_status', 'purpose', 'initial_list_status', 'application_type', 'home_ownership'],
    #     'zip_code_column': 'address',
    #     'columns_to_drop_additional': ['zip_code']
    # }

    # Load config from YAML
    config = ConfigLoader.load_config('config.yaml')

    # Set path of raw and processed data
    raw_data_path = config.get('raw_data_path', os.path.join('data', 'raw', 'data.csv'))
    processed_data_path = config.get('processed_data_path', os.path.join('data', 'processed', 'data_cleaned.csv'))

    # Load raw data into a DataFrame
    df = pd.read_csv(raw_data_path)

    # Instantiate and execute the data cleaning pipeline
    pipeline = DataCleaningPipeline(df, config)
    df_cleaned = pipeline.apply_strategies()

    # Extract numerical and categorical features for analysis
    numerical_features = [feature for feature in df_cleaned.columns if df_cleaned[feature].dtype != 'O']
    categorical_features = [feature for feature in df_cleaned.columns if df_cleaned[feature].dtype == 'O']

    # Print feature details
    print(f'We have {len(numerical_features)} numerical features : {numerical_features}')
    print(f'\nWe have {len(categorical_features)} categorical features : {categorical_features}')
    print(f'Cleaned data shape: {df_cleaned.shape}')

    # Save the cleaned data to the processed file path
    df_cleaned.to_csv(processed_data_path, index=False)
    logger.info(f"Cleaned data saved to {processed_data_path}")
