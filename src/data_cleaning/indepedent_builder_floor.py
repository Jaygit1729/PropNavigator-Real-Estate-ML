# src/data_cleaning/indepedent_builder_floor.py

import pandas as pd
from src.data_ingestions.ingest_data import load_data
from src.logger_utils import setup_logger
from src.data_cleaning.base_cleaner import apply_column_cleaning

logger = setup_logger(__name__, "logs/independent_builder_floor_cleaning.log")


def column_cleaning(df: pd.DataFrame):
    """
    Cleaning pipeline for Independent Builder Floor data.
    Applies shared base cleaning — no additional steps needed
    for this property type.
    """
    try:
        df = apply_column_cleaning(df)
        if df is None:
            return None

        logger.info("Independent Builder Floor column cleaning done.")
        return df

    except Exception as e:
        logger.error(f"Error during Independent Builder Floor column cleaning: {e}")
        return None


def clean_builder_data(file_path: str):
    """
    Full cleaning pipeline for Independent Builder Floor data.
    Loads CSV and applies column cleaning.
    """
    try:
        df = load_data(file_path)
        if df is None:
            logger.warning(f"Data not found: {file_path}")
            return None

        df = column_cleaning(df)
        if df is not None:
            logger.info(f"Independent Builder Floor data cleaned — shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error cleaning Independent Builder Floor data '{file_path}': {e}")
        return None