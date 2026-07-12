# src/data_cleaning/residential_apartment_cleaning.py

import pandas as pd
from src.data_ingestions.ingest_data import load_data
from src.logger_utils import setup_logger


logger = setup_logger(__name__, "logs/flat_cleaning.log")


# Columns that together identify the same flat (Option A). Rows identical across all of
# these are the same flat reposted by different agents, and are dropped after cleaning.

DEDUP_KEY = ['property_name', 'society', 'price_in_cr',
             'areaWithType', 'floorNum', 'facing', 'overlooking']


def apply_column_cleaning(df: pd.DataFrame):
    """
    - Renames price and area columns
    - Converts price to crores (handles both Cr and Lac formats)
    - Extracts numeric price per sqft
    - mapping facing, furnishing and agePossession columns to more readable formats

    """
    try:

        AGE_POSSESSION_MAP = {0: 'Undefined', 1: '1 to 5 Year Old', 2: '5 to 10 Year Old',
                              3: '10+ Year Old', 5: 'Under Construction', 6: '0 to 1 Year Old'}
        df = (
            df
            .rename(columns = {
                                "price": "price_in_cr",
                                "area": "price_per_sqft"
            })

            .assign(society = lambda df_: df_['society'].str
                                                        .strip()
                                                        .str
                                                        .lower())

            .assign(price_in_cr = lambda df_: df_['price_in_cr'].str
                                                                .strip())

            .loc[lambda df_: (df_['price_in_cr'].notna() &
                              (df_['price_in_cr'] != 'Price on Request') &
                                (df_['price_in_cr'] != '25,000'))]

            .assign(price_in_cr = lambda df_: round(pd.to_numeric(df_['price_in_cr']
                                    .apply(lambda value:float(str(value).replace('Cr', '').strip())
                                    if 'Cr' in str(value)
                                    else float(str(value).replace('Lac', '').strip()) / 100
                                    )),4))

            .assign(price_per_sqft = lambda df_:pd.to_numeric(df_['price_per_sqft']
                                                                            .str
                                                                            .split("₹")
                                                                            .str
                                                                            .get(1)
                                                                            .str
                                                                            .replace("/sqft","")
                                                                            .str
                                                                            .replace("L","")
                                                                            .str
                                                                            .replace(",","")
                                                                            .str
                                                                            .strip())
            ,facing  = lambda df_: df_['facing'].replace('0',"Not Available")
                                                .str
                                                .lower()

            ,furnishing = lambda df_: df_['furnishing'].replace('0',"Not Available")
                                                        .str
                                                        .lower()
            ,agePossession = lambda df_: df_['agePossession'].map(AGE_POSSESSION_MAP))

            )

        logger.info("Column cleaning applied successfully.")
        return df

    except Exception as e:
        logger.error(f"Error during base column cleaning: {e}")
        return None


def clean_flat_data(file_path: str):
    """
    Full cleaning pipeline for Residential Apartment data.
    Loads CSV, applies column cleaning, then drops duplicate repostings.
    """
    try:
        df = load_data(file_path)
        if df is None:
            logger.warning(f"Data not found: {file_path}")
            return None

        # Column cleaning (also drops rows with missing / placeholder price)
        before = len(df)
        df = apply_column_cleaning(df)
        if df is None:
            return None
        logger.info(f"Column cleaning dropped {before - len(df)} rows — {before} -> {len(df)}.")

        # Deduplication — drop the same flat reposted by different agents
        before = len(df)
        df = df.drop_duplicates(subset=DEDUP_KEY, keep='first').reset_index(drop=True)
        logger.info(f"Deduplication dropped {before - len(df)} rows — {before} -> {len(df)}.")

        logger.info(f"Residential Apartment data cleaned — shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error cleaning Residential Apartment data '{file_path}': {e}")
        return None
