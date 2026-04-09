# src/data_cleaning/base_cleaner.py

import numpy as np
import pandas as pd
from src.logger_utils import setup_logger

logger = setup_logger(__name__, "logs/base_cleaner.log")


def apply_column_cleaning(df: pd.DataFrame):
    """
    Shared column cleaning logic applied to all property types:
    - Renames price and area columns
    - Converts price to crores (handles both Cr and Lac formats)
    - Extracts numeric price per sqft
    - Converts bedrooms, bathrooms to numeric
    - Cleans balcony, additional_room, facing columns

    This function is the single source of truth for column cleaning.
    Each property type cleaner calls this first, then applies any
    type-specific transformations on top.
    """
    try:
        df = (
            df
            .rename(columns={
                "price": "price_in_cr",
                "area": "price_per_sqft"
            })

            .assign(
                price_in_cr=lambda df_: df_['price_in_cr']
                    .str.replace('₹', '', regex=False)
            )

            .assign(
                price_in_cr=lambda df_: round(
                    pd.to_numeric(
                        df_['price_in_cr'].apply(
                            lambda value: (
                                float(str(value).replace('Cr', '').strip())
                                if 'Cr' in str(value)
                                else float(str(value).replace('Lac', '').strip()) / 100
                            ) if value not in ['Price on Request', None, np.nan]
                            else np.nan
                        )
                    ), 2
                ),

                price_per_sqft=lambda df_: pd.to_numeric(
                    df_['price_per_sqft']
                    .astype(str)
                    .str.split("₹").str.get(1)
                    .str.replace("/sqft", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.strip()
                ),

                bedrooms=lambda df_: pd.to_numeric(
                    df_['bedrooms'].str.split(" ").str.get(0)
                ),

                bathrooms=lambda df_: pd.to_numeric(
                    df_['bathrooms'].str.split(" ").str.get(0)
                ),

                balcony=lambda df_: df_['balcony']
                    .str.split(" ").str.get(0)
                    .replace("No", 0),

                additional_room=lambda df_: df_['additional_room']
                    .fillna('not available')
                    .str.lower(),

                facing=lambda df_: df_['facing']
                    .fillna("not available")
                    .str.lower(),

                society=lambda df_: df_['society']
                    .str.strip()
                    .str.lower()
            )
        )

        logger.info("Base column cleaning applied successfully.")
        return df

    except Exception as e:
        logger.error(f"Error during base column cleaning: {e}")
        return None