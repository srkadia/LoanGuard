import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import seaborn as sns
import tensorflow.keras.models
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
    def __init__(self, model_type: str, X_train: pd.DataFrame, y_train: pd.Series, config: dict):
        self.model_type = model_type
        self.X_train = X_train
        self.y_train = y_train
        self.config = config
        self.best_params = None
        self.model = None

        # MLflow setup
        mlflow.set_tracking_uri(self.config.get('mlflow_tracking_uri', 'http://mlflow_service:5000'))
        mlflow.set_experiment(self.config.get('mlflow_experiment_name', 'LoanGuard'))

    def tune_hyperparameters(self):
        logger.info(f"Hyperparameter tuning for {self.model_type}...")
        try:
            tuner = LightGBMTuning() if self.model_type == "lightgbm" else ANNTuning()
            self.X_train = np.array(self.X_train).astype(np.float32)
            self.y_train = np.array(self.y_train).astype(np.float32)
            self.best_params = tuner.tune(self.X_train, self.y_train)
            logger.info(f"Best parameters: {self.best_params}")
        except Exception as e:
            raise CustomException(f"Tuning failed: {e}")

    def train_model(self):
        try:
            with mlflow.start_run():
                logger.info(f"Training {self.model_type} model...")
                trainer = ModelTrainerFactory.get_trainer(self.model_type, self.best_params or {})
                trainer.train(self.X_train, self.y_train)
                mlflow.log_params(self.best_params or {})
                self.evaluate_model(trainer)
                self.save_model()
        except Exception as e:
            raise CustomException(f"Training failed: {e}")

    def evaluate_model(self, trainer):
        try:
            y_pred = np.round(trainer.predict(self.X_train)).astype(int)
            accuracy = accuracy_score(self.y_train, y_pred)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_text(classification_report(self.y_train, y_pred), "classification_report.txt")
            self.log_confusion_matrix(y_pred)
            logger.info(f"Accuracy: {accuracy}")
        except Exception as e:
            raise CustomException(f"Evaluation failed: {e}")

    def log_confusion_matrix(self, y_pred):
        cm = confusion_matrix(self.y_train, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        cm_path = "artifacts/confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

    def save_model(self):
        try:
            os.makedirs("artifacts", exist_ok=True)
            model_path = f"artifacts/{self.model_type}_model"
            if self.model_type == "ann":
                self.model.save(f"{model_path}.h5")
                mlflow.tensorflow.log_model(self.model, "model")
            else:
                dump(self.model, f"{model_path}.joblib")
                mlflow.sklearn.log_model(self.model, "model")
            logger.info(f"Model saved: {model_path}")
        except Exception as e:
            raise CustomException(f"Model saving failed: {e}")

if __name__ == "__main__":
    CONFIG_FILE_PATH = os.path.abspath(os.path.join("config.yaml"))
    config = ConfigLoader.load_config(CONFIG_FILE_PATH)
    df_train = pd.read_csv(config.get('processed_data_path'))
    target_column = config.get('target_column')
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]
    model_types = config.get('model_type', ['lightgbm'])

    for model_type in model_types:
        try:
            pipeline = ModelTrainingPipeline(model_type, X_train, y_train, config)
            pipeline.train_model()
            logger.info(f"{model_type} training completed.")
        except Exception as e:
            raise CustomException(f"{model_type} training failed: {e}")