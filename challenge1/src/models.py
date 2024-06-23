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
        time_id = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        id = uuid.uuid4()
        model_id = f"{time_id}_{id}"

        return model_id

    def import_dataset(self, dataset_path: str) -> pd.DataFrame:
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
        try:
            self.model_logger.write_info("Starting training ...")
            start_time = time()

            y_train_log = np.log(self.y_train)
            self.reg.fit(self.x_train, y_train_log)

            total_time = time() - start_time

            self.trained_succesfully = True
            self.model_logger.write_info(
                f"Training concluded. Total Time: {total_time:.2f} sec"
            )

        except Exception as e:
            self.model_logger.conclude_session(msg=f"Error training model: {e}")
            raise Exception(e)

    def predict(self, x_test: pd.DataFrame) -> pd.Series:
        try:
            self.model_logger.write_info("Making prediction...")

            pred_log = self.reg.predict(x_test)
            pred = np.exp(pred_log)

            return pred

        except Exception as e:
            self.model_logger.conclude_session(msg=f"Error making prediction: {e}")
            raise Exception(e)

    def evaluate_performance(self, pred: pd.Series) -> Tuple[float, float]:
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
        try:
            class_destination_path = pathlib.Path().parent.joinpath(
                "Models", "Artifact", __class__.__name__
            )
            class_destination_path.mkdir(parents=True, exist_ok=True)

            if self.trained_succesfully:
                model_destination_path = class_destination_path.joinpath(
                    f"{self.model_id}.pkl"
                )
                self.model_logger.write_info(
                    msg=f"Exporting model to {model_destination_path}..."
                )

                joblib.dump(self.reg, model_destination_path)

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
        pred = self.predict(self.x_test)
        r2, mae = self.evaluate_performance(pred=pred)
        destination = self.export_model()

        data = {
            "modelId": self.model_id,
            "modelType": __class__.__name__,
            "trainedModelPath": destination,
            "performance": {"r2Score": r2, "MAE": mae},
            "date": self.model_id.split("_")[0],
        }

        return data

    def model_report(
        self,
    ):
        data = self.get_model_info()

        try:
            self.model_logger.write_info(
                msg=f"Writing json new line report model for model {self.model_id}"
            )
            class_destination_path = (
                pathlib.Path().parent.joinpath("Models", "Metrics").mkdir(exist_ok=True)
            )
            with open("Models/Metrics/report.json", "a") as f:
                json.dump(data, f, indent=4)
                f.write("\n")
                f.close()
        except Exception as e:
            self.model_logger.conclude_session(msg=f"Error creating model report: {e}")
            raise Exception(e)

    def predict_from_existing_model(model_artifact_path: str, test_x: pd.DataFrame):
        
        print("IN")
        reg : LinearRegression= joblib.load(model_artifact_path)
        pred = reg.predict(test_x)
        return pred
        
        
    @abstractmethod
    def data_pre_processing(
        self, dataset_path: str, test_size=0.2, random_state=42
    ) -> None:
        pass


class LinearRegressionModel(SupervisedModel):
    """Supervised learning model implementing standard linear regression"""
    
    def __init__(self) -> None:
        super().__init__()
        self.reg = LinearRegression()

    def data_pre_processing(
        self, dataset_path: str, test_size=0.2, random_state=42
    ) -> None:
        self.df = self.import_dataset(dataset_path)

        try:
            
            self.model_logger.write_info("Processing Data")
            clean_df = clean_dataset(self.df)
            processed_df = clean_df.drop(columns=["depth", "table", "y", "z"])
            dummy_df = pd.get_dummies(
                processed_df, columns=["cut", "color", "clarity"], drop_first=True
            )
            # Split the dataset:
            x = dummy_df.drop(columns="price")
            y = dummy_df.price
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                x, y, test_size=test_size, random_state=random_state
            )
            self.model_logger.write_info("Data Ready for training.")

        except Exception as e:
            self.model_logger.conclude_session(msg=f"Error processing data: {e}")
            raise Exception(e)

