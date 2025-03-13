import os
import sys
import pandas as pd
from abc import ABC, abstractmethod
from utils.logger import Logger
from utils.exception import CustomException

# Initialize Logger
logger = Logger(__name__).get_logger()

# Define an abstract class for Data Ingestor
class DataIngestor(ABC):
    """
    Abstract class for data ingestion.
    """

    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Abstract method to ingest data from a given file."""
        pass


# Implement a concrete class for CSV Ingestion
class CSVDataIngestor(DataIngestor):
    """
    Concrete class for ingesting CSV files.
    """

    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Reads a CSV file and returns its content as a pandas DataFrame.
        """
        try:
            if not file_path.endswith(".csv"):
                raise ValueError("The provided file is not a CSV file.")
            
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded data from {file_path}")
            return df
        except Exception as e:
            raise CustomException(e, sys)


# Implement a Factory to create DataIngestors
class DataIngestorFactory:
    """
    Factory class to get the appropriate DataIngestor based on file type.
    """

    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """
        Returns the appropriate DataIngestor based on file extension.
        """
        try:
            if file_extension == ".csv":
                return CSVDataIngestor()
            else:
                raise ValueError(f"No ingestor available for file extension: {file_extension}")
        except Exception as e:
            raise CustomException(e, sys)


# Example usage
if __name__ == "__main__":
    # try:
    #     # Get data paths
    #     raw_data_path = os.path.join('data', 'raw', 'data.csv')

    #     # Ensure raw data exists
    #     if not os.path.exists(raw_data_path):
    #         raise FileNotFoundError(f"Raw data file not found: {raw_data_path}")

    #     # Determine file extension
    #     file_extension = os.path.splitext(raw_data_path)[1]

    #     # Get the appropriate DataIngestor
    #     data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    #     # Ingest data and load it into a DataFrame
    #     df = data_ingestor.ingest(raw_data_path)

    #     # Now df contains the DataFrame from the extracted CSV
    #     print(df.head())

    #     logger.info("Data ingestion and processing completed successfully.")
    # except Exception as e:
    #     raise CustomException(e, sys)
    pass
