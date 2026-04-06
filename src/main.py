from src.data_cleaning.residential_apartment_cleaning import clean_flat_data
from src.data_cleaning.house_cleaning import clean_house_data
from src.data_cleaning.merge_data import merge_cleaned_datasets
from src.feature_engineering.feature_eng import feature_engineering
from src.data_preprocessing.pre_process_data import preprocessing as data_preprocessing
from src.data_ingestions.ingest_data import save_data
from src.logger_utils import setup_logger
from src.feature_selection.feature_selection import feature_selection_pipeline
from src.model_building.mb_main import run_model_building


logger = setup_logger("logs/main_pipeline.log")


def main():

    """Main execution pipeline"""

    try:
        logger.info("Pipeline started.")

        # 1. Clean Residential Apartment Data

        logger.info("Step 1: Cleaning Residential Apartment data...")
        cleaned_flats = clean_flat_data("data/web_scraping/residential_apartment.csv")
        if cleaned_flats is None:
            logger.error("Residential Apartment data cleaning failed (returned None).")
            return
        save_data(cleaned_flats, "data/data_cleaning/cleaned_residential_apartment.csv")
        logger.info("Residential Apartment data cleaned and saved.")

        # 2. Clean House Data

        logger.info("Step 2: Cleaning Independent House data...")
        cleaned_houses = clean_house_data("data/web_scraping/independent_house.csv")
        if cleaned_houses is None:
            logger.error("House data cleaning failed (returned None).")
            return
        save_data(cleaned_houses, "data/data_cleaning/cleaned_independent_houses.csv")
        logger.info("Independent House data cleaned and saved.")

        # 3. Clean Indepedent Builder Floor Data

        logger.info("Step 3: Cleaning Independent Builder Floor data...")
        cleaned_houses = clean_house_data("data/web_scraping/independent_builder_floor.csv")
        if cleaned_houses is None:
            logger.error("Independent Builder Floor data cleaning failed (returned None).")
            return
        save_data(cleaned_houses, "data/data_cleaning/cleaned_independent_builder_floor.csv")
        logger.info("Independent Builder Floor data cleaned and saved.")

        # 4. Merge Datasets

        logger.info("Step 4: Merging datasets...")
        merged_df = merge_cleaned_datasets(
            "data/data_cleaning/cleaned_residential_apartment.csv",
            "data/data_cleaning/cleaned_independent_houses.csv",
            "data/data_cleaning/cleaned_independent_builder_floor.csv",
            "data/data_cleaning/cleaned_properties.csv"
        )
        if merged_df is None:
            logger.error("Dataset merging failed (returned None).")
            return
        logger.info("Datasets merged successfully.")


        # 5. Feature Engineering

        logger.info("Step 5: Performing feature engineering...")
        fe_df = feature_engineering(merged_df)
        if fe_df is None:
            logger.error("Feature engineering failed (returned None).")
            return
        save_data(fe_df, "data/fe/featured_properties.csv")
        logger.info("Feature engineering completed and data saved.")
   
         # 6. Preprocessing
        
        logger.info("Step 6: Preprocessing data...")
        processed_df = data_preprocessing(fe_df)
        if processed_df is None:
            logger.error("Preprocessing failed (returned None).")
            return
        save_data(processed_df, "data/pp/preprocessed_properties.csv")
        logger.info("Preprocessing completed and data saved.")
    
        
        # 7. Feature Selection
        
        logger.info("Step 7: Feature Selection...")
        fs_df, importance_df = feature_selection_pipeline(processed_df)
        if fs_df is None:
            logger.error("Feature Selection failed (returned None).")
            return 
        save_data(fs_df, "data/fs/feature_selected_properties.csv")
        save_data(importance_df, "data/fs/feature_importance_table.csv")
        logger.info("Feature selection completed and data saved.") 

        # 8. Model Building
        
        logger.info("Step 8-11: Executing Model Building Pipeline...")
        run_model_building(fs_df)
        
        logger.info("Pipeline finished successfully.")

    except Exception as e:
        logger.error(f"Critical pipeline failure: {str(e)}", exc_info=True) 
        
if __name__ == "__main__":
    main()
