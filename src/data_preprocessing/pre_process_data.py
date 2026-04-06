import re
import numpy as np
import pandas as pd
from src.logger_utils import setup_logger


logger = setup_logger('logs/pre_processing.log')
logger.info("Logging set up successfully for Pre-Processing Module!")

def create_area_missing_flags(df):

    df = df.copy()

    area_cols = [
        'super_built_up_area',
        'built_up_area',
        'carpet_area',
        'plot_area'
    ]

    for col in area_cols:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    logger.info("Area missing flags created.")

    return df

def create_primary_area(df):

    df = df.copy()

    df["total_area_sqft"] = np.nan

    flat_mask = df["property_type"].isin(
        ["Flat", "Independent Builder Floor"]
    )

    df.loc[flat_mask, "total_area_sqft"] = \
        df.loc[flat_mask, "super_built_up_area"]

    house_mask = df["property_type"] == "Independent House"

    df.loc[house_mask, "total_area_sqft"] = \
        df.loc[house_mask, "plot_area"]

    logger.info("Primary area assigned using property-type logic.")

    return df

def fallback_area_creation(df):

    df = df.copy()

    fallback_cols = [
        "super_built_up_area",
        "built_up_area",
        "carpet_area"
    ]

    df["total_area_sqft"] = df["total_area_sqft"].fillna(
        df[fallback_cols].median(axis=1)
    )

    logger.info("Fallback area consolidation applied.")

    return df


def ensure_numeric_columns(df):
    """Ensure critical columns are numeric to avoid type errors."""
    df = df.copy()

    cols_to_fix = ["bedrooms", "floornum", "total_area_sqft"]
    for col in cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Converted bedrooms, floornum, and total_area_sqft to numeric (coerce errors).")
    return df

def dropna(df):

    "Removes rows from the DataFrame where specific critical columns have null values."

    initial_rows = df.shape[0]
    df = df.dropna(subset=['price_in_cr','price_per_sqft'])
    dropped_count = initial_rows - len(df)
    
    if dropped_count > 0:
        logger.info(f"Successfully dropped {dropped_count} rows with null values in 'price_in_cr'.")
    else:
        logger.info("No null values found in 'price_in_cr'. No rows were dropped.")
        
    return df


def remove_total_area_outliers(df, max_area = 15000):
    """
    Removes rows where total_area_sqft exceeds a domain-defined upper bound.
    """
    df = df.copy()

    before = df.shape[0]

    df = df[df["total_area_sqft"] <= max_area]

    after = df.shape[0]

    logger.info(
        f"Removed {before - after} rows where total_area_sqft <= {max_area}."
    )

    return df


def remove_price_per_sqft_outliers(df, max_ppsf = 100000):
    """
    Removes rows where price_per_sqft exceeds a domain-defined upper bound.
    """
    df = df.copy()

    before = df.shape[0]

    df = df[df["price_per_sqft"] <= max_ppsf]

    after = df.shape[0]

    logger.info(
        f"Removed {before - after} rows where price_per_sqft > {max_ppsf}."
    )

    return df


def remove_area_bedroom_ratio_outliers( df, min_ratio = 200, dense_threshold = 275 ):
    """
    Cleans structurally implausible layouts based on area_per_bedroom.

    Steps:
    1. Remove rows where area_per_bedroom < min_ratio 
       (physically unrealistic density).
    2. Create a dense_house_flag for independent houses
       where area_per_bedroom < dense_threshold.
    """
    df = df.copy()

    before = df.shape[0]

    df["area_per_bedroom"] = df["total_area_sqft"] / df["bedrooms"]

    # Remove structurally impossible density

    df = df[df["area_per_bedroom"] >= min_ratio]

    after = df.shape[0]

    logger.info(
        f"Removed {before - after} rows where area_per_bedroom < {min_ratio}."
    )

    # Create density flag

    df["dense_house_flag"] = (
        (df["area_per_bedroom"] < dense_threshold) &
        (df["property_type"] == "Independent House")
    ).astype(int)

    logger.info(
        f"Dense house flag created using threshold {dense_threshold}."
    )

    return df

def fill_missing_floornum(df):

    """Fill missing floornum using median value."""

    df = df.copy()

    median_floor = df['floornum'].median()
    df['floornum'] = df['floornum'].fillna(median_floor)

    logger.info(f"Filled missing floornum values with median: {median_floor}.")
    return df

def mode_based_imputation(row, df):

    if row['age_possession_category'] == 'Undefined':
        mode_value = df[
            (df['sector'] == row['sector']) & (df['property_type'] == row['property_type'])
        ]['age_possession_category'].mode()
        return mode_value.iloc[0] if not mode_value.empty else row['age_possession_category']
    return row['age_possession_category']


def mode_based_imputation2(row, df):
    if row['age_possession_category'] == 'Undefined':
        mode_value = df[df['sector'] == row['sector']]['age_possession_category'].mode()
        return mode_value.iloc[0] if not mode_value.empty else row['age_possession_category']
    return row['age_possession_category']


def mode_based_imputation3(row, df):
    if row['age_possession_category'] == 'Undefined':
        mode_value = df[df['property_type'] == row['property_type']]['age_possession_category'].mode()
        return mode_value.iloc[0] if not mode_value.empty else row['age_possession_category']
    return row['age_possession_category']


def impute_age_possession_category(df):
    """Apply 3-pass mode-based imputation for age_possession_category."""
    df = df.copy()

    df['age_possession_category'] = df.apply(lambda row: mode_based_imputation(row, df), axis=1)
    df['age_possession_category'] = df.apply(lambda row: mode_based_imputation2(row, df), axis=1)
    df['age_possession_category'] = df.apply(lambda row: mode_based_imputation3(row, df), axis=1)

    logger.info("Applied 3-pass mode-based imputation for age_possession_category.")
    return df

def create_luxury_category(df):
    logger.info("Creating luxury_category.")
    df = df.copy()
    df['luxury_category'] = pd.qcut(
        df['luxury_score'],
        q=3,
        labels=['Budget', 'Semi-Luxury', 'Luxury'],
        duplicates='drop'
    )
    return df


def categorize_floornum(df):
    logger.info("Categorizing floornum.")
    df = df.copy()

    def _cat(floor):
        if pd.isna(floor):
            return "Undefined"
        f = float(floor)
        if f <= 5:
            return "Low-rise"
        elif f <= 15:
            return "Mid-rise"
        else:
            return "High-rise"

    df['floornum_category'] = df['floornum'].apply(_cat)
    return df



def reorder_columns(df):
    """Reorder columns into a clean and consistent structure."""

    desired_order = [
        'property_type','society', 'sector', 'price_in_cr', 'price_per_sqft',
        'total_area_sqft', 'bedrooms', 'bathrooms', 'balcony', 'floornum_category',
        'study_room', 'servant_room', 'store_room', 'pooja_room', 'others',
        'facing', 'furnishing_type','age_possession_category',
        'super_built_up_area_missing','built_up_area_missing','carpet_area_missing',
        'plot_area_missing','area_per_bedroom','dense_house_flag' ,'luxury_category'
    ]

    df = df[[col for col in desired_order if col in df.columns]]
    logger.info("Reordered columns for final dataset.")
    return df



def preprocessing(df):
    """Main preprocessing pipeline combining all cleaning and correction steps."""
    try:
        df = (
            df
            .pipe(create_area_missing_flags)
            .pipe(create_primary_area)
            .pipe(fallback_area_creation)
            .pipe(ensure_numeric_columns)
            .pipe(dropna)
            .pipe(remove_total_area_outliers)
            .pipe(remove_area_bedroom_ratio_outliers)
            .pipe(remove_price_per_sqft_outliers)      
            .pipe(fill_missing_floornum)
            .pipe(impute_age_possession_category)
            .pipe(create_luxury_category)
            .pipe(categorize_floornum)
            .drop(
                columns=['property_id', 'link', 'areawithtype','price_per_sqft','plot_area', 'super_built_up_area',
        'built_up_area', 'carpet_area'],
                errors='ignore'
            )
            .pipe(reorder_columns)
            .rename(columns={
                'age_possession_category': 'age_possession'
            })
        )

        logger.info(f"Data Pre-Processing completed. Final shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Data Pre-Processing failed: {e}", exc_info=True)
        return df
