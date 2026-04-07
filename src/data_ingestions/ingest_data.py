# src/data_ingestions/ingest_data.py

import os
import pandas as pd
from src.logger_utils import setup_logger

logger = setup_logger(__name__, 'logs/data_ingestions.log')
logger.info("Data ingestion module loaded.")


def load_data(file_path: str) :
    """
    Loads a CSV file into a DataFrame.
    Returns None if the file doesn't exist or loading fails.
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None

        df = pd.read_csv(file_path)
        
        logger.info(f"Loaded '{file_path}' — shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Failed to load '{file_path}': {e}")
        return None


def save_data(df: pd.DataFrame, output_path: str):
    """
    Saves a DataFrame to CSV at the given path.
    Creates intermediate directories if they don't exist.
    Returns True on success, False on failure so the caller can react.
    """
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        df.to_csv(output_path, index=False)
        logger.info(f"Saved data to '{output_path}' — shape: {df.shape}")
        return True

    except Exception as e:
        logger.error(f"Failed to save data to '{output_path}': {e}")
        return False