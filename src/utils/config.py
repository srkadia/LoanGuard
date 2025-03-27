import os
import yaml
from utils.exception import CustomException

# Get absolute path of config.yaml
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of config.py
CONFIG_PATH = os.path.join(BASE_DIR, "..", "config", "config.yaml")

def load_config(config_path=CONFIG_PATH):
    """
    Load configuration settings from a YAML file.
    :param config_path: Path to the YAML configuration file.
    :return: Dictionary containing configuration settings.
    """
    try:
        if not os.path.exists(config_path):
            raise CustomException(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    except Exception as e:
        raise CustomException(e)

# Debugging - Check if path resolves correctly
# if __name__ == "__main__":
#     print(f"Loading config from: {CONFIG_PATH}")
#     config = load_config()
#     print(config)  # Print to verify contents
