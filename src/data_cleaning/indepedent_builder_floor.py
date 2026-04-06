import numpy as np
import pandas as pd
from src.data_ingestions.ingest_data import load_data
from src.logger_utils import setup_logger

logger = setup_logger("logs/indepedent_builder_floor_cleaning.py")



def column_cleaning(df):
    """
    Cleans and transforms specific columns in the Indepedent Builder Floor dataset:
    - Standardizes columns : society, price in cr, price per sqft, bedRoom, bathroom, balcony, additionalRoom, facing
    - Converts price in Crores
    - Renames columns for clarity
    
    """

    try:
        df = (
            df
            .rename(columns={

                "price": "price_in_cr"
                ,
                "area": "price_per_sqft"
        
            })

            .assign(
                price_in_cr = lambda df_: df_['price_in_cr']
                                            .str
                                            .replace('₹', '', regex=False)
                    )

        
            .assign(
                price_in_cr = lambda df_: round(pd.to_numeric(df_['price_in_cr']
                                            .apply(lambda value: (float(str(value)
                                            .replace('Cr', '')
                                            .strip())
                                            if 'Cr' in str(value)
                                            else float(str(value)
                                            .replace('Lac', '')
                                            .strip()) / 100)
                                            if value not in ['Price on Request', None, np.nan] 
                                            else np.nan)),2)
                    

                ,
                price_per_sqft = lambda df_: pd.to_numeric(df_['price_per_sqft']
                                            .astype(str)
                                            .str
                                            .split("₹")
                                            .str
                                            .get(1)
                                            .str
                                            .replace("/sqft", "", regex=False)
                                            .str
                                            .replace(",", "", regex=False)
                                            .str
                                            .strip())
                ,
                bedrooms = lambda df_: pd.to_numeric(df_['bedrooms']
                                            .str
                                            .split(" ")
                                            .str
                                            .get(0))
                    

                ,
                bathrooms = lambda df_: pd.to_numeric(df_['bathrooms']
                                            .str
                                            .split(" ")
                                            .str
                                            .get(0))
                    
                ,
                balcony = lambda df_: df_['balcony']
                                            .str
                                            .split(" ")
                                            .str
                                            .get(0)
                                            .replace("No",0)
                    
                ,
                additional_room = lambda df_: df_['additional_room']
                                            .fillna('not available')
                                            .str
                                            .lower()
                ,
                facing  =  lambda df_: df_['facing']
                                            .fillna("not available")
                                            .str
                                            .lower())
                )
        
        logger.info("Independent Builder Floor data column cleaning done.")
        return df
    except Exception as e:
        logger.error(f"Error during Independent Builder Floor column cleaning: {e}")
        return None


def clean_house_data(file_path):
    """
    Runs the full data cleaning pipeline for Independent Builder Floor data:
    - Load CSV
    - Apply column-specific cleaning

    """
    try:
        df = load_data(file_path)
        if df is None:
            logger.warning(f"Data not found for file path: {file_path}")
            return None
        df = column_cleaning(df)
        if df is not None:
            logger.info(f"Independent Builder Floor data cleaned and shape is {df.shape}.")
        return df
    except Exception as e:
        logger.error(f"Error during Independent Builder Floor data cleaning for file {file_path}: {e}")
        return None
