import numpy as np
import pandas as pd
import mlflow
import mlflow.tensorflow
import tensorflow as tf

from utils.config import ConfigLoader
from utils.exception import CustomException
from utils.logger import Logger

# Initialize Logger
logger = Logger(__name__).get_logger()


class PredictPipeline:
    """
    This class handles the process of prediction by loading a trained model, transforming user input,
    and making predictions.
    """

    def __init__(self, user_input: dict, config: dict):
        """
        Initializes the PredictPipeline with user input and configuration.

        Args:
            user_input (dict): The input data provided by the user.
            config (dict): Configuration parameters (like paths, model settings).
        """
        self.user_input = user_input
        self.config = config
        self.df_input = pd.DataFrame([self.user_input])
        self.model = None

        # Initialize MLflow
        mlflow.set_tracking_uri(self.config.get("mlflow_tracking_uri", "http://127.0.0.1:5000"))
        self.experiment_name = self.config.get("mlflow_experiment_name", "LoanGuard")
        mlflow.set_experiment(self.experiment_name)


    def load_model(self):
        """
        Loads the trained model from disk based on the configuration provided.

        This method will attempt to load a model from the specified path and
        handle errors if the model cannot be found or loaded.
        """
        model_path = self.config.get("trained_model_path", "artifacts/ann_model.h5")
        try:
            mlflow.tensorflow.autolog()
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Model successfully loaded from {model_path}")
        except FileNotFoundError as e:
            error_message = f"Model not found at {model_path}. Please check the file path."
            logger.error(error_message)
            raise CustomException(e)
        except Exception as e:
            error_message = f"Error loading model: {str(e)}"
            logger.error(error_message)
            raise CustomException(e)

    def transform_input(self) -> pd.DataFrame:
        """
        Transforms the user input into a format suitable for model prediction.

        This involves preprocessing steps such as encoding categorical variables,
        handling missing values, and any other transformations defined in the pipeline.

        :returns: The transformed user input in DataFrame format.
        """
        try:
            logger.info("Transforming user input data...")
            transformed_input = self._apply_transforms()
            logger.info("Transformation complete.")
            return transformed_input
        except Exception as e:
            error_message = f"Error transforming input data: {str(e)}"
            logger.error(error_message)
            raise CustomException(e)

    def _apply_transforms(self) -> pd.DataFrame:
        """
        Helper function that applies specific transformations to the user input.

        This method handles the conversion of the user input into the necessary
        format for the model, including one-hot encoding and handling of specific columns.

        :returns: The transformed input dataframe.
        """
        try:
            # Your transformation code (same as previous)
            zip_codes = [5113, 11650, 22690, 29597, 30723, 48052, 70466, 86630, 93700]
            zip_code = int(self.user_input["address"].split()[-1])

            # Start creating the output DataFrame with essential columns
            transformed_data = {
                "loan_amnt": self.user_input["loan_amnt"],
                "term": float(self.user_input["term"].split()[0]),
                "int_rate": self.user_input["int_rate"],
                "installment": self.user_input["installment"],
                "annual_inc": self.user_input["annual_inc"],
                "dti": self.user_input["dti"],
                "earliest_cr_line": int(self.user_input["earliest_cr_line"].split("-")[1]),
                "open_acc": self.user_input["open_acc"],
                "pub_rec": self.user_input["pub_rec"],
                "revol_bal": self.user_input["revol_bal"],
                "revol_util": self.user_input["revol_util"],
                "total_acc": self.user_input["total_acc"],
                "mort_acc": self.user_input["mort_acc"],
                "pub_rec_bankruptcies": self.user_input["pub_rec_bankruptcies"]
            }

            # One-hot encode sub_grades
            sub_grades = [f"{grade}{i}" for grade in ['A', 'B', 'C', 'D', 'E', 'F', 'G'] for i in range(1, 6)]
            for sg in sub_grades:
                transformed_data[f"sub_grade_{sg}"] = True if sg == self.user_input["sub_grade"] else False

            # One-hot encode verification status
            for status in ["Source Verified", "Verified"]:
                transformed_data[f"verification_status_{status.replace(' ', '_')}"] = True if self.user_input["verification_status"] == status else False

            # One-hot encode loan purpose
            purposes = ["credit_card", "debt_consolidation", "educational", "home_improvement", "house",
                        "major_purchase", "medical", "moving", "other", "renewable_energy", "small_business",
                        "vacation", "wedding"]
            for purpose in purposes:
                transformed_data[f"purpose_{purpose}"] = True if self.user_input["purpose"] == purpose else False

            # One-hot encode application type
            for app_type in ["INDIVIDUAL", "JOINT"]:
                transformed_data[f"application_type_{app_type}"] = True if self.user_input["application_type"] == app_type else False

            # One-hot encode home_ownership
            for ownership in ["MORTGAGE", "NONE", "OTHER", "OWN", "RENT"]:
                transformed_data[f"home_ownership_{ownership}"] = True if self.user_input["home_ownership"] == ownership else False

            # One-hot encode zip code
            for zip_code_val in zip_codes:
                transformed_data[f"zip_code_{zip_code_val}"] = True if zip_code == zip_code_val else False

            # Convert the transformed data into a DataFrame
            return pd.DataFrame([transformed_data])

        except Exception as e:
            error_message = f"Error in applying transformations: {str(e)}"
            logger.error(error_message)
            raise CustomException(e)

    def run_pipeline(self) -> np.array:
        """
        Runs the entire prediction pipeline: loads the model, transforms the user input,
        and makes a prediction using the trained model.

        This method handles all steps from loading the model to returning the prediction result.

        :returns:
            np.array: The predicted result from the model.
        """
        try:
            with mlflow.start_run():
                # Step 1: Load the model
                logger.info("Starting pipeline...")
                self.load_model()

                # Step 2: Transform the user input
                transformed_input = self.transform_input()
                transformed_input = np.array(transformed_input).astype(np.float32)

                # Step 3: Log inputs to MLflow
                mlflow.log_params(self.user_input)

                # Step 4: Make the prediction
                prediction = np.round(self.model.predict(transformed_input)).astype(int)

                # Log Prediction
                mlflow.log_metric("prediction", prediction)

                mlflow.end_run()

                return prediction

        except Exception as e:
            logger.error(f"Error in prediction pipeline: {str(e)}")
            raise CustomException(e)


if __name__ == "__main__":
    try:
        # Load the config from the YAML
        config = ConfigLoader.load_config('config.yaml')

        # Get user inputs (for now, we just hardcode it)
        user_input = {
            "loan_amnt": 24375,
            "term": "60 months",
            "int_rate": 17.25,
            "installment": 609.33,
            "sub_grade": "C5",
            "home_ownership": "MORTGAGE",
            "annual_inc": 55000,
            "verification_status": "Verified",
            "purpose": "credit_card",
            "dti": 33.95,
            "earliest_cr_line": "Jun-1999",
            "open_acc": 13.0,
            "pub_rec": 0.0,
            "revol_bal": 24584.0,
            "revol_util": 69.8,
            "total_acc": 43.0,
            "initial_list_status": "w",
            "application_type": "INDIVIDUAL",
            "mort_acc": 1,
            "pub_rec_bankruptcies": 1,
            "address": "1234 Elm Street, Springfield, IL 11650"
        }

        # Create a PredictPipeline instance
        pipeline = PredictPipeline(user_input, config)

        # Run the pipeline and get the prediction
        prediction = pipeline.run_pipeline()

        # Print the result
        logger.info(f"Prediction: {prediction[0][0]}")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise CustomException(e)
