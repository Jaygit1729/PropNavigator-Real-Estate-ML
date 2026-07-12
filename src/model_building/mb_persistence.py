# src/model_building/mb_persistence.py

import joblib
import os
import csv
from datetime import datetime
from src.logger_utils import setup_logger

logger = setup_logger(__name__, "logs/mb_persistence.log")

EXPERIMENT_LOG = "artifacts/experiment_log.csv"


def _log_experiment(model_name, metric, filepath, status, **extra):
    """Appends one row to the experiment log CSV."""
    os.makedirs(os.path.dirname(EXPERIMENT_LOG), exist_ok=True)
    file_exists = os.path.exists(EXPERIMENT_LOG)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "test_mape_percent": metric,
        "artifact_path": filepath,
        "status": status,
        **extra
    }

    with open(EXPERIMENT_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_model(model_pipeline, model_name, metric, filepath, **kwargs):
    """
    Saves the trained model pipeline only if it has the best MAPE.
    Also logs every experiment run (saved or not) for auditability.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if os.path.exists(filepath):
            existing_artifact = joblib.load(filepath)
            best_mape = existing_artifact.get("test_mape_percent", float("inf"))

            if metric >= best_mape:
                logger.info(
                    f"Model '{model_name}' not saved. "
                    f"Existing model has better MAPE ({best_mape:.2f}%)."
                )
                _log_experiment(model_name, metric, filepath, "skipped_worse")
                return

            logger.info(
                f"New model '{model_name}' improved MAPE "
                f"from {best_mape:.2f}% to {metric:.2f}%."
            )

        artifact = {
            "model_name": model_name,
            "test_mape_percent": round(metric, 2),
            "pipeline": model_pipeline,
            "residual_quantiles": kwargs.get("residual_quantiles", None),
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        }

        # Versioned copy
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_path = filepath.replace(
            ".joblib", f"_{ts}.joblib"
        )
        joblib.dump(artifact, versioned_path)

        # Overwrite latest
        joblib.dump(artifact, filepath)

        _log_experiment(model_name, metric, filepath, "saved_best")

        logger.info(
            f"Best model '{model_name}' saved at: {filepath} "
            f"(version: {versioned_path}) with MAPE {metric:.2f}%"
        )

    except Exception as e:
        logger.error(f"Error saving model: {e}", exc_info=True)
        raise


def load_model(filepath):
    """
    Loads saved model artifact.
    """

    try:
        artifact = joblib.load(filepath)
        logger.info(f"Model loaded from: {filepath}")
        return artifact

    except FileNotFoundError:
        logger.error(f"Model file not found at: {filepath}")
        return None

    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        raise