import json
from typing import List, Dict
from models import SupervisedModel
import joblib
import pandas as pd

COLUMNS = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]


def retrieve_all_trained_models() -> List[Dict[str, str | Dict[str, float]]]:
    """ Read the record.json file content"""
    trained_models = []
    with open("Models/Metrics/report.json") as f:  # "Models/Metrics/report.json"
        for line in f:
            j_content = json.loads(line)
            trained_models.append(j_content)

    return trained_models


def load_trained_model_instance(model_artifact_path: str) -> SupervisedModel:
    """ Load the class instance from the artifact"""
    try:
        model_class: SupervisedModel = joblib.load(model_artifact_path)

    except Exception as e:
        print(e)

    return model_class


def pick_best_model() -> Dict | None:
    """Select the best performing model. It will be used by the api for prediction and dataset checks"""
    trained_models = retrieve_all_trained_models()

    if not trained_models:
        return None

    try:
        best_model = trained_models[0]

        for actual_model in trained_models[1:]:
            if best_model["performance"]["MAE"] > actual_model["performance"]["MAE"]:
                best_model = actual_model

    except Exception as e:
        print(e)

    return best_model
