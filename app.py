import os
import sys

import certifi
ca = certifi.where()

from dotenv import load_dotenv

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)

import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipelines.training_pipeline import TrainingPipeline
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from networksecurity.utils.main_utils.utils import load_object

client =  pymongo.MongoClient(MONGO_DB_URL, tlsCAFile = ca)

from networksecurity.constants.training_pipeline import DATA_INGESTION_COLLECTION_NAME, DATA_INGESTION_DATABASE_NAME

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/", tags = ["authentication"])
async def index():
    return RedirectResponse(url= "/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is Successful.")
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        preprocessor = load_object("final_models/preprocessing.pkl")
        final_model = load_object("final_models/model.pkl")
        network_model = NetworkModel (preprocessor= preprocessor, model = final_model)
        print(df.iloc[0])
        y_pred = network_model.predict(df)
        print(y_pred)
        df['predicted_column'] = y_pred
        print(df['predicted_column'])
        df.to_csv("prediction_output/output.csv")
        table_html = df.to_html(classes='table table-striped')

        return templates.TemplateResponse("table.html", {"request":request, "table": table_html})
    except Exception as e:
        raise NetworkSecurityException(e, sys)

    

if __name__ == "__main__":
    app_run(app, host="localhost", port=8000)
