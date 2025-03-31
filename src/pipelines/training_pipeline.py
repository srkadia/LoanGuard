import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from strategies.model_training import ModelTrainerFactory
from strategies.hyperparameter_tuning import LightGBMTuning, ANNTuning
from utils.config import ConfigLoader
from utils.exception import CustomException
from utils.logger import Logger
from joblib import dump

# Initialize Logger
logger = Logger(__name__).get_logger()


class ModelTrainingPipeline:
    """
    A pipeline that automates hyperparameter tuning, model training, and evaluation.
    It selects a model dynamically, tunes it if necessary, trains it on the given data,
    evaluates it, and saves the trained model for later use.
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

        try:
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
        except CustomException as e:
            logger.error(f"Error during hyperparameter tuning for {self.model_type}: {e}")

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

            # Save the trained model
            self.save_model(trainer, self.model_type)

            # Evaluate the model after training
            self.evaluate_model(trainer)

            return trainer

        except CustomException as e:
            logger.error(f"Error during {self.model_type} model training: {e}")

    def evaluate_model(self, trainer):
        """
        Evaluates the model after training using metrics like accuracy and AUC.
        """
        logger.info("Evaluating the trained model...")

        try:
            # Assuming that the trainer has a `predict` method for making predictions
            y_pred = trainer.predict(self.X_train)
            y_pred = np.round(y_pred).astype(int)  # For ANN, convert probabilities to binary labels

            accuracy = accuracy_score(self.y_train, y_pred)
            logger.info(f"Model accuracy: {accuracy}")
            logger.info(f"Classification Report:\n{classification_report(self.y_train, y_pred)}")

            # Optionally, print confusion matrix
            logger.info(f"Confusion Matrix:\n{confusion_matrix(self.y_train, y_pred)}")
        except CustomException as e:
            logger.error(f"Error during model evaluation: {e}")

    def save_model(self, trainer, model_type: str):
        """
        Saves the trained model to a file.
        """
        logger.info(f"Saving the {model_type} model...")

        try:
            # Save the model using joblib or model-specific saving
            if model_type == "ann":
                trainer.model.save(f"{model_type}_model.h5")
            else:
                dump(trainer.model, f"{model_type}_model.joblib")

            logger.info(f"{model_type} model saved successfully.")
        except CustomException as e:
            logger.error(f"Error saving the model: {e}")

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
    # model_types = config.get('model_type', ['lightgbm'])
    model_types = ['ann']
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
