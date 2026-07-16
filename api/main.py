from pathlib import Path
from fastapi import FastAPI
from api.schemas import PredictRequest
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
bundle = joblib.load(BASE_DIR / "artifacts" / "best_model.joblib") # it runs once at startup, not per request.

app = FastAPI()

@app.get("/health")

def health_check():
    return {
        "status": "ok",
        "model_name": bundle["model_name"],
        "mape_percent": bundle["test_mape_percent"],
    }

@app.post("/predict")

def predict(request : PredictRequest):
    return {"received": request.model_dump()}

