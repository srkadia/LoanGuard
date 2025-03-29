import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from abc import ABC, abstractmethod

from utils.logger import Logger

# Initialize Logger
logger = Logger(__name__).get_logger()


# --- Abstract Base Class ---
class ModelTrainingStrategy(ABC):
    """Abstract base class for model training strategies."""

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        pass

    @abstractmethod
    def predict(self, X_test: pd.DataFrame):
        pass


# --- Concrete Strategies ---
class LightGBMTraining(ModelTrainingStrategy):
    """LightGBM model training strategy (Uses Pre-Tuned Hyperparameters)."""

    def __init__(self, best_params):
        self.model = lgb.LGBMClassifier(**best_params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        logger.info("Training LightGBM Model...")
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame):
        return self.model.predict(X_test)


class ANNTraining(ModelTrainingStrategy):
    """ANN Training with Pre-Tuned Hyperparameters."""

    def __init__(self, best_params):
        logger.info("Initializing ANN Model...")
        self.ann_params = best_params
        self.model = None

    def build_model(self, input_dim, num_labels):
        """Builds ANN model using the best hyperparameters."""
        inp = Input(shape=(input_dim,))
        x = BatchNormalization()(inp)

        # Apply dropout & hidden layers dynamically
        for i in range(self.ann_params['num_layers']):
            units = self.ann_params[f'units_{i}']
            dropout = self.ann_params[f'dropout_{i+1}']

            x = Dense(units, activation='relu', kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout)(x)

        # Output layer
        output_activation = 'sigmoid' if num_labels == 1 else 'softmax'
        x = Dense(num_labels, activation=output_activation, dtype="float32")(x)

        model = Model(inputs=inp, outputs=x)
        optimizer = Adam(learning_rate=self.ann_params['learning_rate'])
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='AUC')])
        return model

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Trains ANN model using the best hyperparameters."""
        X_train = np.array(X_train).astype(np.float32)
        y_train = np.array(y_train).astype(np.float32)

        num_columns = X_train.shape[1]
        num_labels = 1

        logger.info("Training ANN Model...")

        # Build and train model
        self.model = self.build_model(num_columns, num_labels)
        self.model.fit(
            X_train, y_train,
            epochs=self.ann_params['epochs'],
            batch_size=self.ann_params['batch_size'],
            validation_split=self.ann_params['validation_split']
        )
        logger.info("ANN model training completed.")

    def predict(self, X_test: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(X_test).round()


# --- Factory Class ---
class ModelTrainerFactory:
    """Factory to instantiate model training strategies dynamically."""

    @staticmethod
    def get_trainer(model_type: str, best_params) -> ModelTrainingStrategy:
        trainers = {
            "lightgbm": lambda: LightGBMTraining(best_params),
            "ann": lambda: ANNTraining(best_params),
        }

        if model_type not in trainers:
            raise ValueError(f"Invalid model type: {model_type}. Choose from {list(trainers.keys())}.")

        return trainers[model_type]()
