# src/data_cleaning/merge_data.py

import pandas as pd
from src.logger_utils import setup_logger
from src.data_ingestions.ingest_data import load_data, save_data

logger = setup_logger(__name__, "logs/merge_data.log")


def merge_cleaned_datasets(
    flat_path: str,
    house_path: str,
    builder_floor_path: str,
    output_path: str,
    shuffle: bool = True
                        ):
    """
    Loads cleaned datasets for all three property types,
    tags each with its property_type, merges them, and saves
    the result to output_path.
    """
    try:
        flats_df = load_data(flat_path)
        houses_df = load_data(house_path)
        builder_floor_df = load_data(builder_floor_path)

        # If any dataset failed to load, abort early
        if any(df is None for df in [flats_df, houses_df, builder_floor_df]):
            logger.error("One or more datasets failed to load. Aborting merge.")
            return None

        logger.info(
            f"Loaded datasets — flats: {flats_df.shape}, "
            f"houses: {houses_df.shape}, "
            f"builder floors: {builder_floor_df.shape}"
        )

        # Tag each dataset with its property type before merging
        flats_df["property_type"] = "Flat"
        houses_df["property_type"] = "Independent House"
        builder_floor_df["property_type"] = "Independent Builder Floor"

        merged_df = pd.concat(
            [flats_df, houses_df, builder_floor_df],
            ignore_index=True,
            sort=False
        )

        if shuffle:
            merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
            logger.info("Merged dataset shuffled.")

        # Use save_data for consistent saving with error handling
        saved = save_data(merged_df, output_path)
        if not saved:
            logger.error("Failed to save merged dataset.")
            return None

        logger.info(f"Merged dataset saved — shape: {merged_df.shape}")
        return merged_df

    except Exception as e:
        logger.error(f"Failed to merge datasets: {e}")
        return None