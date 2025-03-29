import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from strategies.model_training import ModelTrainerFactory
from strategies.hyperparameter_tuning import LightGBMTuning, ANNTuning
from utils.config import ConfigLoader
from utils.exception import CustomException
from utils.logger import Logger

# Initialize Logger
logger = Logger(__name__).get_logger()


def eval_metrics(model_name, y_true, y_pred, train=True):
    """
    Evaluates model performance using accuracy, classification report, and confusion matrix.

    :param model_name: Name of the model being evaluated.
    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param train: Boolean flag indicating if it's training or test data.
    """
    clf_report_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose()
    eval_type = "Train" if train else "Test"

    logger.info(f"\n{model_name} - {eval_type} Evaluation:\n" + "="*50)
    logger.info(f"Accuracy Score: {accuracy_score(y_true, y_pred) * 100:.2f}%")
    logger.info(f"Classification Report:\n{clf_report_df}")
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}\n")


class ModelTrainingPipeline:
    """
    A pipeline that automates hyperparameter tuning and model training.
    It selects a model dynamically, tunes it if necessary, trains it on the given data,
    and saves the trained model for later use.
    """

    def __init__(self, model_type: str, X_train: pd.DataFrame, y_train: pd.Series, config: dict):
        """
        Initializes the model training pipeline.

        :param model_type: The type of model to train (e.g., 'lightgbm', 'ann').
        :param X_train: Training features.
        :param y_train: Training target labels.
        :param config: Configuration settings for model parameters.
        """
        self.model_type = model_type
        self.X_train = X_train
        self.y_train = y_train
        self.config = config
        self.best_params = None  # Stores best hyperparameters

    def tune_hyperparameters(self):
        """
        Tunes the hyperparameters before training, if applicable.
        """
        logger.info(f"Starting hyperparameter tuning for {self.model_type}...")

        if self.model_type == "lightgbm":
            tuner = LightGBMTuning()
            self.best_params = tuner.tune(self.X_train, self.y_train)

        elif self.model_type == "ann":
            # Convert data to float32 for ANN training
            self.X_train = np.array(self.X_train).astype(np.float32)
            self.y_train = np.array(self.y_train).astype(np.float32)

            tuner = ANNTuning()
            self.best_params = tuner.tune(self.X_train, self.y_train)

        if self.best_params:
            logger.info(f"Best hyperparameters found for {self.model_type}: {self.best_params}")

    def train_model(self):
        """
        Trains the selected model using the best hyperparameters.
        """
        try:
            logger.info(f"Initializing model training for: {self.model_type}")

            # Perform tuning first (if applicable)
            if self.config.get("tune_hyperparameters", False):
                self.tune_hyperparameters()

            # Get model trainer instance with best hyperparameters
            trainer = ModelTrainerFactory.get_trainer(self.model_type, self.best_params)

            # Train the model
            trainer.train(self.X_train, self.y_train)
            logger.info(f"{self.model_type} model training completed successfully.")

            # Make predictions on training data
            y_pred = trainer.predict(self.X_train)

            if self.model_type == "ann":
                y_pred = np.round(y_pred).astype(int)  # Convert ANN probabilities to binary labels

            # Evaluate the model
            eval_metrics(self.model_type.upper(), self.y_train, y_pred, train=True)

            return trainer

        except CustomException as e:
            logger.error(f"Error during {self.model_type} model training: {e}")
            raise e


if __name__ == "__main__":
    # Load config from YAML
    config = ConfigLoader.load_config('config.yaml')

    # Define paths for input datasets
    processed_data_path = config.get('processed_data_path')

    # Load training dataset
    df_train = pd.read_csv(processed_data_path)

    # Define features and target
    target_column = config.get('target_column')  # Adjust based on your dataset
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]

    # Select model type dynamically (can be parameterized)
    model_types = config.get('model_type', ['lightgbm'])
    tune_hyperparameters = config.get("tune_hyperparameters", False)  # Enable/disable tuning via config

    # Instantiate and execute the model training pipeline
    for model_type in model_types:
        try:
            pipeline = ModelTrainingPipeline(model_type, X_train, y_train, config)
            trained_model = pipeline.train_model()
            logger.info(f"Model training pipeline executed successfully for {model_type}.")
        except Exception as e:
            logger.error(f"Skipping {model_type} due to error: {e}")

    logger.info(f"Model training pipeline execution finished successfully!")
