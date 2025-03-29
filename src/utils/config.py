import os
import yaml
from utils.exception import CustomException


class ConfigLoader:
    """
    Utility class to load YAML configuration files dynamically from the config directory.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_DIR = os.path.join(BASE_DIR, "..", "config")

    @staticmethod
    def load_config(filename: str):
        """
        Load a YAML configuration file from the config directory.

        :param filename: Name of the YAML file to load (e.g., 'config.yaml').
        :return: Dictionary containing configuration settings.
        """
        config_path = os.path.join(ConfigLoader.CONFIG_DIR, filename)

        try:
            if not os.path.exists(config_path):
                raise CustomException(f"Configuration file not found: {config_path}")

            with open(config_path, "r") as file:
                return yaml.safe_load(file)

        except Exception as e:
            raise CustomException(f"Error loading {filename}: {str(e)}")

# Example Usage
# if __name__ == "__main__":
#     config = ConfigLoader.load_config("config.yaml")
#     print(config, params, sep='\n')