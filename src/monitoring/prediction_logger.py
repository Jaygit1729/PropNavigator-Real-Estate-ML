# src/monitoring/prediction_logger.py

import os
import csv
from datetime import datetime


LOG_PATH = "logs/prediction_log.csv"

FIELDS = [
    "timestamp", "model_name",
    "property_type", "sector", "area", "bedRoom",
    "predicted_price_cr", "lower_bound_cr", "upper_bound_cr",
]


def log_prediction(
    model_name: str,
    input_features: dict,
    prediction: dict
):
    """
    Appends one prediction row to the CSV log.
    Schema is intentionally flat — easy to load into pandas for
    drift analysis or post-hoc accuracy checks.
    """
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    file_exists = os.path.exists(LOG_PATH)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "property_type": input_features.get("property_type", ""),
        "sector": input_features.get("sector", ""),
        "area": input_features.get("area", ""),
        "bedRoom": input_features.get("bedRoom", ""),
        "predicted_price_cr": prediction.get("predicted_price", ""),
        "lower_bound_cr": prediction.get("lower_bound", ""),
        "upper_bound_cr": prediction.get("upper_bound", ""),
    }

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
