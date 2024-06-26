from models import (
    LinearRegressionModel,
    BoostLinearRegressionModel,
    SupervisedModel,
    AVAILABLE_MODELS,
)
import sys


WEB_DATASET = "https://raw.githubusercontent.com/xtreamsrl/xtream-ai-assignment-engineer/main/datasets/diamonds/diamonds.csv"


def get_model_instance(model_name: str) -> SupervisedModel:
    """Model validation based on model names : xgboost or linear"""
    
    
    if not model_name.lower() in AVAILABLE_MODELS:
        raise Exception(
            f"No model named {model_name}. Available models: {AVAILABLE_MODELS}"
        )

    if model_name.lower() == "linear":
        return LinearRegressionModel()
    elif model_name.lower() == "xgboost":
        return BoostLinearRegressionModel()

    ## Add for more models


def automated_pipeline(dataset_path: str) -> None:
    """Main class that starts the training pipeline"""
    
    
    try:
        # Get input model from pipeline

        choosen_model = sys.argv[1]

        # Initiate the model class
        model = get_model_instance(model_name=choosen_model)

        # Learning steps
        # 1. Process dataset
        model.data_pre_processing(dataset_path=dataset_path)
        # # 2. Train model
        model.train()
        # # # 3. Create Report: Prediction -> Performance Analisys -> Model Export -> Model Session Information Export
        model.model_report()
        # # # # 4. End Session
        model.model_logger.conclude_session(success=True)

    except IndexError as ie:
        print("No model indicated. Available models:")
        for model in AVAILABLE_MODELS:
            print(model)
    except Exception as e:
        print(e)


automated_pipeline(WEB_DATASET)
