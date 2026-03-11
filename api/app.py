from fastapi import FastAPI, UploadFile, File
import shutil
import os

from src.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI()

pipeline = PredictionPipeline()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def home():
    return {"message": "Document Classification API Running"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction, confidence = pipeline.predict(file_path)

    return {
        "document_type": str(prediction),
        "confidence": float(confidence)
    }