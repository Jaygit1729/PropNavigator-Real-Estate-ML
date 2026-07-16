"""PropNavigator price-prediction API.

    GET  /health   - is the service up, and which model is loaded?
    GET  /sectors  - the sector names /predict accepts
    POST /predict  - price estimate + 90% range for one property

Run locally from the project root:
    uvicorn api.main:app --reload
"""

from fastapi import FastAPI, HTTPException

from api.schemas import PredictRequest, PredictResponse
from api import inference   # loads the model + reference tables once, at import

app = FastAPI(
    title="PropNavigator Price API",
    description="Estimate Gurgaon property prices with the deployed model.",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_name": inference.model_name,
        "mape_percent": inference.mape_percent,
        "n_features": len(inference.FEATURES),
    }


@app.get("/sectors")
def sectors():
    """Valid values for the `sector` field in /predict."""
    return {"count": len(inference.VALID_SECTORS), "sectors": inference.VALID_SECTORS}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        return inference.predict_price(request.model_dump())
    except ValueError as e:
        # e.g. a sector we don't know -> the caller's fault, not ours
        raise HTTPException(status_code=422, detail=str(e))
