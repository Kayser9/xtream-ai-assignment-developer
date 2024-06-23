from models import LinearRegressionModel

WEB_DATASET = "https://raw.githubusercontent.com/xtreamsrl/xtream-ai-assignment-engineer/main/datasets/diamonds/diamonds.csv"


def automated_pipeline(dataset_path: str) -> None:
    try:
        
        # Initiate the model class
        l = LinearRegressionModel()
        
        # Learning steps

        # 1. Process dataset
        l.data_pre_processing(dataset_path=dataset_path)

        # # 2. Train model
        l.train()
        # # 3. Create Report: Prediction -> Performance Analisys -> Model Export -> Model Session Information Export
        l.model_report()

        l.model_logger.conclude_session(success=True)

    except Exception as e:
        print(e)


automated_pipeline(WEB_DATASET)
