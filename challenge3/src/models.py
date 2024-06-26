from logger import CustomLogger
import pandas as pd
import uuid
from datetime import datetime
from time import time
import json
from dataset import clean_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from typing import Tuple
import numpy as np
import joblib
import pathlib
from abc import ABC, abstractmethod
from xgboost import XGBRegressor
import optuna

AVAILABLE_MODELS = ["linear", "xgboost"]  ## Add new model


class SupervisedModel(ABC):
    """Supervised Learning Model Abstract Class"""

    def __init__(self) -> None:
        self.model_id = self.set_model_id()
        self.model_logger = CustomLogger(self.model_id)
        self.x_train = self.x_test = self.y_train = self.y_test = self.reg = None
        self.trained_succesfully = False

    def set_model_id(
        self,
    ) -> str:
        """Create the unique model ID"""
        
        time_id = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        id = uuid.uuid4()
        model_id = f"{time_id}_{id}"

        return model_id

    def import_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Import the given dataset"""
        try:
            self.model_logger.write_info("Importing dataset ...")
            df = pd.read_csv(dataset_path)
        except Exception as e:
            self.model_logger.conclude_session(
                msg=f"Error downloading the database: {e}"
            )
            raise Exception(f"Error importing the dataset: {e}")

        return df

    def train(
        self,
    ) -> None:
        """Train model"""
        try:
            self.model_logger.write_info("Starting training ...")
            start_time = time()

            self.reg.fit(self.x_train, self.y_train)

            total_time = time() - start_time

            self.trained_succesfully = True
            self.model_logger.write_info(
                f"Training concluded. Total Time: {total_time:.2f} sec"
            )

        except Exception as e:
            self.model_logger.conclude_session(msg=f"Error training model: {e}")
            raise Exception(e)

    def predict(self, x_test: pd.DataFrame) -> pd.Series:
        """Perform prediction base on given input"""
        try:
            self.model_logger.write_info("Making prediction...")

            pred = self.reg.predict(x_test)
            
            return pred

        except Exception as e:
            self.model_logger.conclude_session(msg=f"Error making prediction: {e}")
            raise Exception(e)

    def evaluate_performance(self, pred: pd.Series) -> Tuple[float, float]:
        """Evaluate the model performance. Calculate r2 score and MAE"""
        
        try:
            if self.trained_succesfully:
                self.model_logger.write_info(msg=f"Evaluating model performance ...")
                
                r2score = round(r2_score(self.y_test, pred), 4)
                mae = round(mean_absolute_error(self.y_test, pred), 2)
                
                self.model_logger.write_info(msg=f"R2_Score : {r2score}, MAE : {mae}")
                return r2score, mae
            else:
                raise Exception("Model not trained during this session")
        except Exception as e:
            self.model_logger.conclude_session(
                msg=f"Error evaluating the model performances: {e}"
            )
            raise Exception(e)

    def export_model(
        self,
    ) -> str:
        """Once trained, export the instance class in Models/Artifact/{model_type}/{model_id} """
        
        try:
            class_destination_path = pathlib.Path().parent.joinpath(
                "Models", "Artifact", self.__class__.__name__
            )
            class_destination_path.mkdir(parents=True, exist_ok=True)

            if self.trained_succesfully:
                model_destination_path = class_destination_path.joinpath(
                    f"{self.model_id}.pkl"
                )
                self.model_logger.write_info(
                    msg=f"Exporting model to {model_destination_path}..."
                )

                joblib.dump(self, model_destination_path)

                return str(model_destination_path.absolute())
            else:
                raise Exception("Model not trained during this session")

        except Exception as e:
            self.model_logger.conclude_session(
                msg=f"Error exporting model {self.model_id}: {e}"
            )
            raise Exception(e)

    def get_model_info(
        self,
    ):
        """
        Perform:
        1. prediction
        2. evaluation
        3. export of the model
        Return dictionary with model information
        
        """
        
        pred = self.predict(self.x_test)
        r2, mae = self.evaluate_performance(pred=pred)
        destination = self.export_model()

        data = {
            "modelId": self.model_id,
            "modelType": self.__class__.__name__,
            "trainedModelPath": destination,
            "performance": {"r2Score": r2, "MAE": mae},
            "date": self.model_id.split("_")[0],
        }

        return data

    def model_report(
        self,
    ):
        """Export the model information in the Models/Metrics/report.json file"""
        data = self.get_model_info()

        try:
            self.model_logger.write_info(
                msg=f"Writing json new line report model for model {self.model_id}"
            )
            class_destination_path = (
                pathlib.Path().parent.joinpath("Models", "Metrics").mkdir(exist_ok=True)
            )
            
            with open("Models/Metrics/report.json", "a") as f:
                json.dump(data, f)
                f.write("\n")
                f.close()
                
        except Exception as e:
            self.model_logger.conclude_session(msg=f"Error creating model report: {e}")
            raise Exception(e)

    def predict_from_existing_model(model_artifact_path: str, test_x: pd.DataFrame):
        
        """Static method for prediction uploading model class"""
        model_class: SupervisedModel = joblib.load(model_artifact_path)
        pred = model_class.predict(test_x)
        return pred

    @abstractmethod
    def data_pre_processing(
        self, dataset_path: str, test_size=0.2, random_state=42
    ) -> None:
        """Process and organize the dataset for the supervised learning steps"""
        pass

    @abstractmethod
    def clean_data(
        self, dataset: pd.DataFrame, for_prediction=False, prediction_columns=[]
    ) -> pd.DataFrame:
        """Clean the dataset"""
        pass


class LinearRegressionModel(SupervisedModel):
    """Supervised learning model implementing standard linear regression"""

    def __init__(self) -> None:
        super().__init__()
        self.reg = LinearRegression()

    def clean_data(
        
        self, dataset: pd.DataFrame, for_prediction=False, prediction_columns=[]
    ) -> pd.DataFrame:
        if not for_prediction:
            clean_df = clean_dataset(dataset)
            processed_df = clean_df.drop(columns=["depth", "table", "y", "z"])

            dummy_df = pd.get_dummies(
                processed_df, columns=["cut", "color", "clarity"], drop_first=True
            )
            return dummy_df

        else:
            clean_df = dataset
            processed_df = clean_df.drop(columns=["depth", "table", "y", "z"])

            missing = [c for c in prediction_columns if c not in processed_df.columns]

            for c in missing:
                origin_c = c.split("_")[0]
                processed_df[c] = processed_df[origin_c] == c.split("_")[-1]

            return processed_df.drop(columns=["cut", "color", "clarity"])

    def data_pre_processing(
        
        self, dataset_path: str, test_size=0.2, random_state=42
    ) -> None:
        self.df = self.import_dataset(dataset_path)

        try:
            self.model_logger.write_info("Processing Data")

            # Prepare training dataset
            dummy_df = self.clean_data(self.df)

            # Split the dataset:
            x = dummy_df.drop(columns="price")
            y = dummy_df.price
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                x, y, test_size=test_size, random_state=random_state
            )

            self.model_logger.write_info(
                "Transforming train y values into logarithmic values"
            )
            self.y_train = np.log(self.y_train)

            self.model_logger.write_info("Data Ready for training.")

        except Exception as e:
            self.model_logger.conclude_session(msg=f"Error processing data: {e}")
            raise Exception(e)

    def predict(self, x_test: pd.DataFrame) -> pd.Series:
        
        pred_log = super().predict(x_test)
        self.model_logger.write_info("Transforming logarithmic prediction values")
        pred = np.exp(pred_log)

        return pred


class BoostLinearRegressionModel(SupervisedModel):
    """Supervised learning model implementing XGBoost model"""

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.reg = XGBRegressor()

    def clean_data(
        
        self, dataset: pd.DataFrame, for_prediction=False, prediction_columns=[]
    ) -> pd.DataFrame:
        if not for_prediction:
            clean_df = clean_dataset(dataset)
        else:
            clean_df = dataset

        diamonds_processed_xgb = clean_df.copy()
        diamonds_processed_xgb["cut"] = pd.Categorical(
            diamonds_processed_xgb["cut"],
            categories=["Fair", "Good", "Very Good", "Ideal", "Premium"],
            ordered=True,
        )
        diamonds_processed_xgb["color"] = pd.Categorical(
            diamonds_processed_xgb["color"],
            categories=["D", "E", "F", "G", "H", "I", "J"],
            ordered=True,
        )
        diamonds_processed_xgb["clarity"] = pd.Categorical(
            diamonds_processed_xgb["clarity"],
            categories=["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"],
            ordered=True,
        )

        return diamonds_processed_xgb

    def data_pre_processing(
        self, dataset_path: str, test_size=0.2, random_state=42
    ) -> None:
        
        self.df = self.import_dataset(dataset_path)

        try:
            self.model_logger.write_info("Processing Data")

            diamonds_processed_xgb = self.clean_data(self.df)

            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                diamonds_processed_xgb.drop(columns="price"),
                diamonds_processed_xgb["price"],
                test_size=0.2,
                random_state=42,
            )

        except Exception as e:
            self.model_logger.conclude_session(msg=f"Error processing data: {e}")
            raise Exception(e)

        self.optimize_model_parameters()

    def optimize_model_parameters(
        
        self,
    ):
        """Tuned the parameters to achieve better performance"""
        
        try:
            self.model_logger.write_info("Optimizing parameters ...")

            study = optuna.create_study(
                direction="minimize", study_name="Diamonds XGBoost"
            )
            study.optimize(self.objective, n_trials=100)
            self.training_params = (study.best_params,)
            self.reg = XGBRegressor(
                **study.best_params, enable_categorical=True, random_state=42
            )

            self.model_logger.write_info(f"Best parameters: {self.training_params}")

        except Exception as e:
            self.model_logger.conclude_session(msg=f"Error optimizing parameters: {e}")
            raise Exception(e)

    def objective(self, trial: optuna.trial.Trial) -> float:
        # Define hyperparameters to tune
        param = {
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "colsample_bytree": trial.suggest_categorical(
                "colsample_bytree", [0.3, 0.4, 0.5, 0.7]
            ),
            "subsample": trial.suggest_categorical(
                "subsample", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            ),
            "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "random_state": 42,
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "enable_categorical": True,
        }

        # Split the training data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(
            self.x_train, self.y_train, test_size=0.2, random_state=42
        )

        # Train the model
        model = XGBRegressor(**param)
        model.fit(x_train, y_train)

        # Make predictions
        preds = model.predict(x_val)

        # Calculate MAE
        mae = mean_absolute_error(y_val, preds)

        return mae
