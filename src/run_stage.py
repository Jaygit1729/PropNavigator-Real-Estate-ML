# src/run_stages.py

import sys

from src.logger_utils import setup_logger
from src.data_ingestions.ingest_data import load_data, save_data
from src.data_cleaning.residential_apartment_cleaning import clean_flat_data
from src.data_cleaning.indepedent_builder_floor import clean_builder_data
from src.data_cleaning.house_cleaning import clean_house_data
from src.data_cleaning.merge_data import merge_cleaned_datasets
from src.feature_engineering.feature_eng import feature_engineering
from src.data_preprocessing.pre_process_data import preprocessing
from src.feature_selection.feature_selection import select_features

logger = setup_logger(__name__,"logs/pipeline.log")

def stage_clean_flats():
    """
    Stage 1: Clean residential apartment data.
    """
    cleaned_flats = clean_flat_data("data/web_scraping/flats_gurgaon.csv")
    save_data(cleaned_flats, "data/data_cleaning/cleaned_residential_apartment.csv")
    logger.info("Stage completed: clean_flats")

def stage_clean_builder():

    """
    Stage 2: Clean independent builder floor data.
    """
    cleaned_builder = clean_builder_data("data/web_scraping/builder_floor_gurgaon.csv")
    save_data(cleaned_builder, "data/data_cleaning/cleaned_independent_builder_floor.csv")
    logger.info("Stage completed: clean_builder")

def stage_clean_house():
    """
    Stage 3: Clean house data.
    """
    cleaned_house = clean_house_data("data/web_scraping/independent_house_gurgaon.csv")
    save_data(cleaned_house, "data/data_cleaning/cleaned_independent_houses.csv")
    logger.info("Stage completed: clean_house")

def stage_merge():

    """
    Stage 4: Merge cleaned datasets."""

    merge_cleaned_datasets(
        "data/data_cleaning/cleaned_residential_apartment.csv",
        "data/data_cleaning/cleaned_independent_houses.csv",
        "data/data_cleaning/cleaned_independent_builder_floor.csv",
        "data/data_cleaning/cleaned_properties.csv",
    )

    logger.info("Stage completed: merge")


def stage_feature_engineering():

    """
    Stage 5: Feature engineering on merged data."""
    merged_data = load_data("data/data_cleaning/cleaned_properties.csv")
    featured_data = feature_engineering(merged_data)
    save_data(featured_data, "data/fe/featured_properties.csv")

    logger.info("Stage completed: feature_engineering")

def stage_preprocess():

    """
    Stage 6: Preprocess featured data."""
    featured_data = load_data("data/fe/featured_properties.csv")
    preprocessed_data = preprocessing(featured_data)
    save_data(preprocessed_data, "data/pp/preprocessed_properties.csv")

    logger.info("Stage completed: preprocess")


def stage_feature_selection():

    """
    Stage 7: Select features from preprocessed data.""" 
    preprocessed_data = load_data("data/pp/preprocessed_properties.csv")
    selected_data = select_features(preprocessed_data)
    save_data(selected_data, "data/fs/feature_selected_properties.csv")

    logger.info("Stage completed: feature_selection")

STAGES = {
    "clean_flats": stage_clean_flats,
    "clean_builder": stage_clean_builder,
    "clean_house": stage_clean_house,
    "merge": stage_merge,
    "feature_engineering": stage_feature_engineering,
    "preprocess": stage_preprocess,
    "feature_selection": stage_feature_selection,
}


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in STAGES:
        raise SystemExit(f"Unknown stage '{name}'. Options: {', '.join(STAGES)}")
    STAGES[name]()
