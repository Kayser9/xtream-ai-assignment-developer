from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Tuple
from utils import COLUMNS, pick_best_model, load_trained_model_instance
import json
import pandas as pd


class Input(BaseModel):
    data: List[List]  # [Tuple[float,str,str,str,float,float,float,float,float]]
    


app = FastAPI()


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware function:
    1. Pick the best available model
    2. Load the best model
    """
    #TODO
    # Connection to MongoDB to store api logs 
    
    try:
        # Get the best model
        best_model_info = pick_best_model()

        # Load the selected model class instance
        model_instance = load_trained_model_instance(
            best_model_info["trainedModelPath"]
        )

        # Set them as state parametes
        request.state.model_info = best_model_info
        request.state.model = model_instance

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "message": "No Model Available now. Wait a minute and/or Train a new one( i.e. docker-compose run --build pipeline)"
            },
        )
    #TODO
    # Connection to MongoDB to store api logs 
    
    response = await call_next(request)
    return response


@app.post("/prediction", status_code=200)
async def prediction(request: Request, input: Input):
    """Endpoint generating the prediction:
    1. Prepare input data for prediction
    2. Generate Prediction
    """

    try:
        # create datafram from input data
        df = pd.DataFrame(input.data, columns=COLUMNS)

        # process data depending on the model type
        res_df = request.state.model.clean_data(
            df,
            for_prediction=True,
            prediction_columns=request.state.model.x_train.columns,
        )

        # make prediction
        prediction = request.state.model.predict(res_df)
        prediction.astype(float)

    except Exception as e:
        print(e)
        return JSONResponse(
            status_code=400,
            content={
                "message": f"Bad Request: Data must be in the format <[{COLUMNS}, {COLUMNS}, ...]>  "
            },
        )

    return JSONResponse(
        {"model": request.state.model_info, "prediction": prediction.tolist()}
    )


@app.get("/diamonds", status_code=200)
async def get_training_data(
    request: Request,
    carat: float,
    cut: str,
    color: str,
    clarity: str,
    justraining=False,
):
    """Query the dataset depending on the 4 main characteristics: carat, cut, colo and clarity.
    It's possible to choose between the whole dataset or just the fold used for the training by leveraging the 'justtraining' parameter
    """
    try:
        # Choose dataset
        if not justraining:
            training_df = request.state.model.df
        else:
            training_df = request.state.model.x_train

        # Query dataset
        res: pd.DataFrame = training_df[
            (training_df["carat"] == carat)
            & (training_df["cut"] == cut)
            & (training_df["color"] == color)
            & (training_df["clarity"] == clarity)
        ]

        # Process data for response
        json_df = json.loads(
            res.to_json(
                orient="index",
            )
        )

    except Exception:
        return JSONResponse(
            status_code=500, content={"message": "Internal Server Error"}
        )

    return JSONResponse(json_df)
