import joblib
import os
from src.logger_utils import setup_logger

logger = setup_logger("logs/mb_persistence.log")


def save_model(model_pipeline, model_name, metric, filepath):
    """
    Saves the trained model pipeline only if it has the best MAPE.
    """

    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # If model already exists, compare MAPE
        if os.path.exists(filepath):

            existing_artifact = joblib.load(filepath)
            best_mape = existing_artifact.get("test_mape_percent", float("inf"))

            # If new model is worse, do not overwrite
            if metric >= best_mape:
                logger.info(
                    f"Model '{model_name}' not saved. Existing model has better MAPE ({best_mape:.2f}%)."
                )
                return

            logger.info(
                f"New model '{model_name}' improved MAPE from {best_mape:.2f}% to {metric:.2f}%."
            )

        # Save new best model
        artifact = {
            "model_name": model_name,
            "test_mape_percent": round(metric, 2),
            "pipeline": model_pipeline
        }

        joblib.dump(artifact, filepath)

        logger.info(
            f"Best model '{model_name}' saved successfully at: {filepath} with MAPE {metric:.2f}%"
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