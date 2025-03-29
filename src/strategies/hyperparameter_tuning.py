import optuna
import lightgbm as lgb
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from optuna.integration import TFKerasPruningCallback
from abc import ABC, abstractmethod
import numpy as np
import time

class HyperparameterTuningStrategy(ABC):
    """
    Abstract base class for hyperparameter tuning strategies.
    """

    @abstractmethod
    def tune(self, X_train, y_train):
        """
        Run hyperparameter tuning.

        :param X_train: Training feature dataset.
        :param y_train: Training target dataset.
        :return: Best hyperparameters.
        """
        pass

class LightGBMTuning(HyperparameterTuningStrategy):
    """
    Hyperparameter tuning for LightGBM using Optuna.
    """

    def objective(self, trial, X_train, y_train):
        params = {
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 20, 150, step=10),
            "max_depth": trial.suggest_int("max_depth", -1, 15),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10, log=True),
            "random_state": 42,
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1
        }

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_metric="auc",
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        return model.best_score_["valid_0"]["auc"]

    def tune(self, X_train, y_train):
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
        study.optimize(lambda trial: self.objective(trial, X_train, y_train), n_trials=20)
        return study.best_params

class ANNTuning(HyperparameterTuningStrategy):
    """
    Hyperparameter tuning for ANN using Optuna.
    """

    def build_model(self, trial, input_dim, num_labels) -> tf.keras.models.Model:
        inp = tf.keras.layers.Input(shape=(input_dim,))
        x = tf.keras.layers.BatchNormalization()(inp)
        x = tf.keras.layers.Dropout(trial.suggest_float("dropout_1", 0.0, 0.5, step=0.1))(x)

        num_layers = trial.suggest_int("num_layers", 1, 4)
        for i in range(num_layers):
            x = tf.keras.layers.Dense(
                trial.suggest_int(f"units_{i}", 64, 256, step=32),
                activation="relu",
                kernel_initializer="he_normal"
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(trial.suggest_float(f"dropout_{i+1}", 0.0, 0.5, step=0.1))(x)

        output_activation = "sigmoid" if num_labels == 1 else "softmax"
        x = tf.keras.layers.Dense(num_labels, activation=output_activation, dtype="float32")(x)

        model = tf.keras.models.Model(inputs=inp, outputs=x)
        optimizer = Adam(trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True))
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="AUC")])
        return model

    def objective(self, trial, X_train, y_train) -> float:
        X_train = np.array(X_train).astype(np.float32)
        y_train = np.array(y_train).astype(np.float32)

        num_columns = X_train.shape[1]
        num_labels = 1

        model = self.build_model(trial, num_columns, num_labels)
        start_time = time.time()
        history = model.fit(
            X_train,
            y_train,
            validation_split=trial.suggest_float("validation_split", 0.1, 0.3, step=0.05),
            epochs=trial.suggest_int("epochs", 30, 100, step=10),
            batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]),
            verbose=0,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=trial.suggest_float("reduce_lr_factor", 0.1, 0.7, step=0.1),
                    patience=trial.suggest_int("reduce_lr_patience", 2, 6),
                    verbose=0
                ),
                TFKerasPruningCallback(trial, "val_loss")
            ]
        )
        elapsed_time = time.time() - start_time
        print(f"Trial {trial.number} completed in {elapsed_time:.2f} seconds")
        return max(history.history["val_AUC"])

    def tune(self, X_train, y_train):
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
        study.optimize(lambda trial: self.objective(trial, X_train, y_train), n_trials=20)
        return study.best_params
