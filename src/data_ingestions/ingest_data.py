from src.logger_utils import setup_logger
import pandas as pd
import os

# Initialize the logger
logger = setup_logger('logs/data_ingestions.log')
logger.info("Logging setup for data ingestion successfully!")

def load_data(file_path):
    """
    Loads the dataset from a CSV file.
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f'File not Found: {file_path}')
            return None
        df = pd.read_csv(file_path)
        logger.info(f'Data Loaded Successfully from {file_path}')
        return df
    except Exception as e:
        logger.error(f'Error Loading Data from {file_path}: {e}')
        return None

def save_data(df, output_path):
    """
    Saves the dataset to the specified path.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f'Data Saved Successfully to {output_path}')
    except Exception as e:
        logger.error(f'Error Saving data to {output_path}: {e}')
