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
    """
    Abstract base class for model training strategies.

    This class defines the interface for different model training strategies.
    All concrete classes must implement the `train` and `predict` methods.
    """

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train the model using the given training data.

        :param X_train: pd.DataFrame
            The training feature dataset.

        :param y_train: pd.Series
            The target labels for the training dataset.

        :raises NotImplementedError: If the method is not overridden in a subclass.
        """
        pass

    @abstractmethod
    def predict(self, X_test: pd.DataFrame):
        """
        Predict using the trained model on the test data.

        :param X_test: pd.DataFrame
            The test feature dataset.

        :return: np.ndarray
            The predictions for the test dataset.

        :raises NotImplementedError: If the method is not overridden in a subclass.
        """
        pass


# --- Concrete Strategies ---
class LightGBMTraining(ModelTrainingStrategy):
    """
    LightGBM model training strategy (Uses Pre-Tuned Hyperparameters).

    This class implements the `ModelTrainingStrategy` for LightGBM. It trains a model using pre-tuned hyperparameters
    and supports making predictions on new data.
    """

    def __init__(self, best_params: dict):
        """
        Initializes the LightGBM model with the provided hyperparameters.

        :param best_params: dict
            A dictionary containing the best hyperparameters for the LightGBM model.
        """
        self.model = lgb.LGBMClassifier(**best_params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train the LightGBM model on the provided training data.

        :param X_train: pd.DataFrame
            The training feature dataset.

        :param y_train: pd.Series
            The target labels for the training dataset.
        """
        logger.info("Training LightGBM Model...")
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame):
        """
        Make predictions using the trained LightGBM model.

        :param X_test: pd.DataFrame
            The test feature dataset.

        :return: np.ndarray
            The predicted values for the test dataset.
        """
        return self.model.predict(X_test)


class ANNTraining(ModelTrainingStrategy):
    """
    ANN model training strategy using pre-tuned hyperparameters.

    This class implements the `ModelTrainingStrategy` for an Artificial Neural Network (ANN) using Keras.
    It trains an ANN model using pre-tuned hyperparameters and supports making predictions on new data.
    """

    def __init__(self, best_params: dict):
        """
        Initializes the ANN model with the provided hyperparameters.

        :param best_params: dict
            A dictionary containing the best hyperparameters for the ANN model.
        """
        logger.info("Initializing ANN Model...")
        self.ann_params = best_params
        self.model = None

    def build_model(self, input_dim: int, num_labels: int) -> Model:
        """
        Builds the ANN model using the pre-tuned hyperparameters.

        This method constructs the neural network model dynamically based on the provided hyperparameters
        such as number of layers, units per layer, dropout rates, etc.

        :param input_dim: int
            The number of features in the input dataset.

        :param num_labels: int
            The number of output labels (1 for binary classification, greater for multi-class classification).

        :return: tf.keras.models.Model
            A compiled Keras model.
        """
        inp = Input(shape=(input_dim,))
        x = BatchNormalization()(inp)

        # Apply dropout and hidden layers dynamically based on the hyperparameters
        for i in range(self.ann_params['num_layers']):
            units = self.ann_params[f'units_{i}']
            dropout = self.ann_params[f'dropout_{i + 1}']

            x = Dense(units, activation='relu', kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout)(x)

        # Output layer with activation function depending on the number of labels
        output_activation = 'sigmoid' if num_labels == 1 else 'softmax'
        x = Dense(num_labels, activation=output_activation, dtype="float32")(x)

        model = Model(inputs=inp, outputs=x)
        optimizer = Adam(learning_rate=self.ann_params['learning_rate'])
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='AUC')])
        return model

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train the ANN model on the provided training data.

        :param X_train: pd.DataFrame
            The training feature dataset.

        :param y_train: pd.Series
            The target labels for the training dataset.
        """
        X_train = np.array(X_train).astype(np.float32)
        y_train = np.array(y_train).astype(np.float32)

        num_columns = X_train.shape[1]
        num_labels = 1

        logger.info("Training ANN Model...")

        # Build and train the model
        self.model = self.build_model(num_columns, num_labels)
        self.model.fit(
            X_train, y_train,
            epochs=self.ann_params['epochs'],
            batch_size=self.ann_params['batch_size'],
            validation_split=self.ann_params['validation_split']
        )
        logger.info("ANN model training completed.")

    def predict(self, X_test: pd.DataFrame):
        """
        Make predictions using the trained ANN model.

        :param X_test: pd.DataFrame
            The test feature dataset.

        :return: np.ndarray
            The predicted values for the test dataset, rounded to nearest integer.

        :raises ValueError: If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(X_test).round()


# --- Factory Class ---
class ModelTrainerFactory:
    """
    Factory class to instantiate model training strategies dynamically.

    This class provides a method to create instances of model training strategies (such as LightGBM and ANN)
    based on the specified model type and pre-tuned hyperparameters.
    """

    @staticmethod
    def get_trainer(model_type: str, best_params: dict) -> ModelTrainingStrategy:
        """
        Return an appropriate model training strategy based on the model type.

        :param model_type: str
            The type of model to train. Can be "lightgbm" or "ann".

        :param best_params: dict
            A dictionary containing the best hyperparameters for the chosen model.

        :return: ModelTrainingStrategy
            An instance of a concrete model training strategy.

        :raises ValueError: If an invalid model type is provided.
        """
        trainers = {
            "lightgbm": lambda: LightGBMTraining(best_params),
            "ann": lambda: ANNTraining(best_params),
        }

        if model_type not in trainers:
            raise ValueError(f"Invalid model type: {model_type}. Choose from {list(trainers.keys())}.")

        return trainers[model_type]()
