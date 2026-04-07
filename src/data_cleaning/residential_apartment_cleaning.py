# src/data_cleaning/residential_apartment_cleaning.py

import pandas as pd
from src.data_ingestions.ingest_data import load_data
from src.logger_utils import setup_logger
from src.data_cleaning.base_cleaner import apply_column_cleaning

logger = setup_logger(__name__, "logs/flat_cleaning.log")


def column_cleaning(df: pd.DataFrame):
    """
    Cleaning pipeline for Residential Apartment data.
    Applies base cleaning first, then standardizes the society column
    which is specific to this property type.
    """
    try:
        # Step 1 — apply shared cleaning logic
        
        df = apply_column_cleaning(df)
        if df is None:
            return None

        # Step 2 — residential apartments have a society column
        # that needs its own standardization

        df = df.assign(
            society=lambda df_: df_['society']
                .str.strip()
                .str.lower()
        )

        logger.info("Residential Apartment column cleaning done.")
        return df

    except Exception as e:
        logger.error(f"Error during Residential Apartment column cleaning: {e}")
        return None


def clean_flat_data(file_path: str):
    """
    Full cleaning pipeline for Residential Apartment data.
    Loads CSV and applies column cleaning.
    """
    try:
        df = load_data(file_path)
        if df is None:
            logger.warning(f"Data not found: {file_path}")
            return None

        df = column_cleaning(df)
        if df is not None:
            logger.info(f"Residential Apartment data cleaned — shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error cleaning Residential Apartment data '{file_path}': {e}")
        return None