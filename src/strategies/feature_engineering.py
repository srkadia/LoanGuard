import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from utils.logger import Logger
from utils.exception import CustomException

# Initialize Logger
logger = Logger(__name__).get_logger()


# --- Abstract Base Class ---
class FeatureEngineeringStrategy(ABC):
    """
    Abstract base class for all feature engineering strategies.
    Defines the common interface that all concrete feature engineering strategies must implement.
    """

    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method for applying feature engineering transformations.

        :param df: The input DataFrame to clean.
        :return: Transformed DataFrame.
        """
        pass


# --- Concrete Class for Numeric Extraction ---
class NumericExtractor(FeatureEngineeringStrategy):
    """
    Concrete class for extracting numeric values from mixed-type string columns.
    Handles transformations such as extracting numeric values based on regular expressions and cleaning string columns.
    """

    def __init__(self, numeric_transformations: list):
        """
        Initializes the strategy with a list of numeric transformations.

        :param numeric_transformations: A list of dictionaries containing the rules for each numeric transformation.
                                         Each dictionary should contain 'column_name', 'pattern', 'remove_chars', and 'default_value'.
        """
        self.numeric_transformations = numeric_transformations

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations to extract numeric values from specified columns in the DataFrame.

        :param df: The DataFrame to clean.
        :return: DataFrame with numeric values extracted and cleaned.
        """
        transformations = self.numeric_transformations

        for transform in transformations:
            column_name = transform.get('column_name')
            pattern = transform.get('pattern')
            replace_dict = transform.get('replace_dict', None)
            remove_chars = transform.get('remove_chars', None)
            default_value = transform.get('default_value', None)

            try:
                logger.info(f"Starting transformation for column: {column_name}")

                # Check if column exists
                if column_name not in df.columns:
                    raise CustomException(f"Column '{column_name}' not found in the DataFrame.")

                # Apply replacement dictionary if provided
                if replace_dict:
                    logger.debug(f"Replacing values in '{column_name}' based on provided dictionary.")
                    df[column_name] = df[column_name].replace(replace_dict)

                # Remove specific characters if specified
                if remove_chars:
                    logger.debug(f"Removing characters '{remove_chars}' from '{column_name}'.")
                    df[column_name] = df[column_name].astype(str).str.replace(remove_chars, '', regex=True)

                # Extract numeric values using the provided regex pattern
                logger.debug(f"Extracting numeric values from '{column_name}' using pattern: {pattern}")
                extracted_values = df[column_name].astype(str).str.extract(pattern)

                if extracted_values is not None:
                    # Convert extracted values to float and handle NaNs
                    extracted_values = extracted_values[0].astype(float)
                    if default_value is not None:
                        # Replace NaN with the default value
                        extracted_values = extracted_values.fillna(default_value)

                    # Ensure the extracted column has the same length as the original
                    df[column_name] = extracted_values
                    logger.info(f"Transformation for column '{column_name}' completed successfully.")
                else:
                    logger.warning(f"No numeric values extracted for column '{column_name}'.")

            except CustomException as ce:
                logger.error(f"Feature engineering error: {ce.error_message}")
                raise  # Re-raise the custom exception
            except Exception as e:
                logger.error(f"Unexpected error occurred while transforming column '{column_name}': {e}")
                raise CustomException(f"Unexpected error occurred for column '{column_name}': {e}")

        return df


# --- Concrete Class for Zip Code Extraction and One-Hot Encoding ---
class ZipCodeHandler(FeatureEngineeringStrategy):
    """
    Strategy for extracting ZIP codes from the address column and encoding them.
    Extracts the last 5 characters from an address field (assuming ZIP codes) and one-hot encodes them.
    """

    def __init__(self, zip_code_column: str):
        """
        Initializes the strategy with the specified zip code column name.

        :param zip_code_column: The name of the column containing address information from which ZIP codes will be extracted.
        """
        self.zip_code_column = zip_code_column

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts ZIP codes from the address column and performs one-hot encoding.
        The address column is then dropped.

        :param df: The DataFrame to clean.
        :return: DataFrame with processed and transformed data.
        """
        try:
            if self.zip_code_column in df.columns:
                logger.info(f"Extracting ZIP codes from '{self.zip_code_column}' column.")

                # Extract last 5 characters (ZIP code)
                df['zip_code'] = df[self.zip_code_column].apply(lambda x: x[-5:] if isinstance(x, str) else np.nan)

                # One-hot encode ZIP codes
                df = pd.get_dummies(df, columns=['zip_code'], drop_first=True)
                logger.info(f"One-hot encoded '{self.zip_code_column}' column.")

                # Drop the address column
                df.drop(columns=[self.zip_code_column], inplace=True)
                logger.info(f"Dropped column: '{self.zip_code_column}'")
            else:
                raise CustomException(f"Column '{self.zip_code_column}' not found in DataFrame.")

        except CustomException as ce:
            logger.error(f"Feature engineering error: {ce.error_message}")
            raise  # Re-raise the custom exception
        except Exception as e:
            logger.error(f"Unexpected error occurred while extracting and encoding ZIP codes: {e}")
            raise CustomException(f"Unexpected error occurred while processing ZIP codes: {e}")

        return df


# --- Concrete Class for Date Feature Extraction ---
class DateFeatureExtractor(FeatureEngineeringStrategy):
    """
    Strategy for extracting specific features (like year) from date columns.
    Converts date columns to datetime format and extracts the year as a numeric value.
    """

    def __init__(self, date_columns: dict):
        """
        Initializes the strategy with a dictionary of date columns and their respective formats.

        :param date_columns: A dictionary where keys are column names and values are date formats (e.g., '%b-%Y').
        """
        self.date_columns = date_columns

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert date columns to datetime format and extract year as a numeric feature.

        :param df: The DataFrame to clean.
        :return: DataFrame with extracted date features (year).
        """
        try:
            for column, date_format in self.date_columns.items():
                if column in df.columns:
                    logger.info(f"Extracting year from {column} with format {date_format}")

                    # Convert to datetime format
                    df[column] = pd.to_datetime(df[column], format=date_format, errors='coerce')

                    # Extract year and convert to numeric
                    df[column] = df[column].dt.year

                    logger.info(f"Extracted year from {column}.")
                else:
                    raise CustomException(f"Column '{column}' not found in DataFrame.")

        except CustomException as ce:
            logger.error(f"Feature engineering error: {ce.error_message}")
            raise  # Re-raise the custom exception
        except Exception as e:
            logger.error(f"Unexpected error occurred while extracting date features: {e}")
            raise CustomException(f"Unexpected error occurred while processing date features: {e}")

        return df


# --- Concrete Class for Target Encoding ---
class TargetEncoder(FeatureEngineeringStrategy):
    """
    Concrete class for encoding the target variable (e.g., 'loan_status').
    Maps 'Fully Paid' to 0 and 'Charged Off' to 1.
    """

    def __init__(self, target_column: str = 'loan_status'):
        """
        Initializes the encoder with the target column name.

        :param target_column: The column in the DataFrame that needs to be encoded (default: 'loan_status')
        """
        self.target_column = target_column

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes the target variable by mapping 'Fully Paid' to 0 and 'Charged Off' to 1.

        :param df: The DataFrame that contains the target column to encode.
        :return: DataFrame with the encoded target column.
        """
        try:
            logger.info(f"Encoding target column: {self.target_column}")

            if self.target_column not in df.columns:
                raise CustomException(f"The column '{self.target_column}' is not found in the DataFrame.")

            # Perform the target encoding
            df[self.target_column] = df[self.target_column].map({'Fully Paid': 0, 'Charged Off': 1})

            logger.info(f"Target column '{self.target_column}' encoded successfully.")
        except CustomException as ce:
            logger.error(f"Feature engineering error: {ce.error_message}")
            raise  # Re-raise the custom exception
        except Exception as e:
            logger.error(f"Unexpected error occurred while encoding target column '{self.target_column}': {e}")
            raise CustomException(f"Unexpected error occurred while encoding target column: {e}")

        return df


# --- Factory Classes for Feature Engineering Strategies ---
class NumericExtractorFactory:
    """
    Factory class for creating instances of the NumericExtractor class.
    """

    def __init__(self, config: dict):
        """
        Initializes the factory with the configuration dictionary containing numeric transformation rules.

        :param config: The configuration dictionary containing 'numeric_transformations' key.
        """
        self.numeric_transformations = config.get('numeric_transformations', [])

        # Validate configuration
        if not self.numeric_transformations:
            raise CustomException("Configuration missing or invalid: 'numeric_transformations' not found.")

    def create(self) -> NumericExtractor:
        """
        Creates and returns an instance of NumericExtractor with the provided transformation configuration.

        :return: An instance of NumericExtractor.
        """
        return NumericExtractor(self.numeric_transformations)


class ZipCodeHandlerFactory:
    """
    Factory class for creating instances of the ZipCodeHandler class.
    """

    def __init__(self, config: dict):
        """
        Initializes the factory with the configuration dictionary containing the zip code column name.

        :param config: The configuration dictionary containing 'zip_code_column' key.
        """
        self.zip_code_column = config.get('zip_code_column', 'address')

        # Validate configuration
        if not self.zip_code_column:
            raise CustomException("Configuration missing or invalid: 'zip_code_column' not found.")

    def create(self) -> ZipCodeHandler:
        """
        Creates and returns an instance of ZipCodeHandler with the specified zip code column name.

        :return: An instance of ZipCodeHandler.
        """
        return ZipCodeHandler(self.zip_code_column)


class DateFeatureExtractorFactory:
    """
    Factory class for creating instances of the DateFeatureExtractor class.
    """

    def __init__(self, config: dict):
        """
        Initializes the factory with the configuration dictionary containing date columns and formats.

        :param config: The configuration dictionary containing 'date_columns' key.
        """
        self.date_columns = config.get('date_columns', {})

        # Validate configuration
        if not self.date_columns:
            raise CustomException("Configuration missing or invalid: 'date_columns' not found.")

    def create(self) -> DateFeatureExtractor:
        """
        Creates and returns an instance of DateFeatureExtractor with the provided date columns configuration.

        :return: An instance of DateFeatureExtractor.
        """
        return DateFeatureExtractor(self.date_columns)


class TargetEncoderFactory:
    """
    Factory class for creating instances of the TargetEncoder class.
    """

    def __init__(self, config: dict):
        """
        Initializes the factory with the configuration dictionary containing the target column name.

        :param config: The configuration dictionary containing 'target_column' key.
        """
        self.target_column = config.get('target_column', 'loan_status')

        # Validate configuration
        if not self.target_column:
            raise CustomException("Configuration missing or invalid: 'target_column' not found.")

    def create(self) -> TargetEncoder:
        """
        Creates and returns an instance of TargetEncoder with the specified target column name.

        :return: An instance of TargetEncoder.
        """
        return TargetEncoder(self.target_column)