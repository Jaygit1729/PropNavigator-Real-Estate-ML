import pandas as pd
import os
from src.logger_utils import setup_logger

logger = setup_logger("logs/merge_data.log")

def merge_cleaned_datasets(flat_path, house_path, indepedent_builder_floor_path, output_path, shuffle=True):
    try:
    
        flats_df = pd.read_csv(flat_path)
        houses_df = pd.read_csv(house_path)
        indepedent_builder_floor_df = pd.read_csv(indepedent_builder_floor_path)

        logger.info("Successfully loaded flat, house and builder floor datasets.")

        flats_df["property_type"] = "Flat"
        houses_df["property_type"] = "Independent House"
        indepedent_builder_floor_df["property_type"] = "Independent Builder Floor"
        
        merged_df = pd.concat([flats_df, houses_df, indepedent_builder_floor_df], ignore_index = True, sort = False)

        if shuffle:
            merged_df = merged_df.sample(frac = 1, random_state=42).reset_index(drop = True)
            logger.info("Merged dataset shuffled.")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        merged_df.to_csv(output_path, index=False)
        logger.info(f"Merged dataset saved successfully at: {output_path} having shape of {merged_df.shape}")

        return merged_df

    except Exception as e:
        logger.error(f"Failed to merge datasets: {e}")
        return None
