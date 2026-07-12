# src/data_preprocessing/pre_process_data.py

import numpy as np
import pandas as pd
from src.logger_utils import setup_logger


logger = setup_logger(__name__, 'logs/pre_processing.log')
logger.info("Logging set up successfully for Pre-Processing Module.")


DIST_COLS = ['dist_to_cyber_city', 'dist_to_golf_road', 'dist_to_airport', 'dist_to_manesar']

# Per-type sane area range (sqft). Flats/floors are single units; houses sit on plots.

AREA_MIN = 180
AREA_CAP = {'Flat': 16000, 'Independent Builder Floor': 10000, 'Independent House': 30000}

# implied price-per-sqft (price / area) must stay in a sane band
PPSF_MIN, PPSF_MAX = 1500, 250000

# a bedroom needs ~150 sqft even in cramped layouts
MIN_AREA_PER_BEDROOM = 150


DROP_COLS = [
    'property_name', 'link', 'areaWithType', 'carpetArea', 'address', 'floorNum',
    'overlooking', 'agePossession', 'cornerProperty', 'parking', 'nearbyLocations',
    'description', 'features', 'property_id',
    'price_per_sqft', 'latitude', 'longitude', 'floornum',
]

# final, clean column order
DESIRED_ORDER = [
    'property_type', 'society', 'sector', 'price_in_cr', 'area',
    'bedRoom', 'bathroom', 'balcony', 'floornum_category', 'total_floor',
    'facing', 'furnishing', 'age_possession_category', 'is_corner',
    'covered_parking', 'open_parking', 'total_parking',
    'has_ac', 'has_power_backup', 'has_gym', 'has_pool', 'has_club',
    'has_lift', 'has_servant_room', 'is_gated',
    'ov_park', 'ov_pool', 'ov_club', 'ov_main_road', 'ov_sea', 'ov_others',
    'dist_to_cyber_city', 'dist_to_golf_road', 'dist_to_airport', 'dist_to_manesar',
]



def fix_bathroom_zero(df):

    """A home can't have 0 bathrooms — fall back to the bedroom count."""
    df = df.copy()
    mask = df['bathroom'] == 0
    df.loc[mask, 'bathroom'] = df.loc[mask, 'bedRoom']
    logger.info(f"bathroom == 0 fixed on {int(mask.sum())} rows.")
    return df



def fill_floor_nulls(df):

    """Missing floors mean 'ground / none' (independent houses) -> 0."""
    df = df.copy()
    df[['floornum', 'total_floor']] = df[['floornum', 'total_floor']].fillna(0)
    logger.info("floornum / total_floor nulls filled with 0.")
    return df


def fill_parking_nulls(df):
    
    """No parking info recorded -> 0 parking."""
    df = df.copy()
    df[['covered_parking', 'open_parking', 'total_parking']] = \
        df[['covered_parking', 'open_parking', 'total_parking']].fillna(0)
    logger.info("parking nulls filled with 0.")
    return df


def fill_balcony_nulls(df):
    
    """Missing balcony -> 0."""
    df = df.copy()
    df['balcony'] = df['balcony'].fillna(0)
    logger.info("balcony nulls filled with 0.")
    return df


def impute_distance_nulls(df):
    
    """Rows with un-fixable coordinates -> median distance per landmark."""
    df = df.copy()
    for col in DIST_COLS:
        df[col] = df[col].fillna(df[col].median())
    logger.info("distance nulls filled with median.")
    return df


def clean_facing(df):
    
    """'not available' / missing facing -> explicit 'unknown' category."""
    df = df.copy()
    df['facing'] = df['facing'].fillna('unknown').replace(
        {'not available': 'unknown', 'na': 'unknown', '': 'unknown'})
    logger.info("facing cleaned ('not available' -> 'unknown').")
    return df


def clean_furnishing(df):
    
    """'not available' / missing furnishing -> explicit 'unknown' category."""
    df = df.copy()
    df['furnishing'] = df['furnishing'].fillna('unknown').replace(
        {'not available': 'unknown', 'na': 'unknown', '': 'unknown'})
    logger.info("furnishing cleaned ('not available' -> 'unknown').")
    return df


# age possession: 3-pass mode imputation
# Pass 1: mode within same sector AND property_type (most specific)
# Pass 2: mode within same sector
# Pass 3: mode within same property_type (broadest)

def mode_based_imputation(row, df):
    if row['age_possession_category'] == 'Undefined':
        mode_value = df[
            (df['sector'] == row['sector']) &
            (df['property_type'] == row['property_type'])
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


#  society cardinality 

def cap_rare_societies(df: pd.DataFrame, min_count: int = 5):
    
    """Group missing / rare societies (< min_count) into 'other'."""
    
    df = df.copy()
    df['society'] = df['society'].fillna('other')
    counts = df['society'].value_counts()
    rare = counts[counts < min_count].index
    df['society'] = df['society'].where(~df['society'].isin(rare), 'other')
    logger.info(f"Rare societies (<{min_count}) grouped into 'other' "
                f"-> {df['society'].nunique()} unique.")
    return df


# outlier removal 

def remove_area_outliers(df):
    
    """Univariate: area must be physically possible for its property type."""
    before = len(df)
    cap = df['property_type'].map(AREA_CAP)
    df = df[(df['area'] >= AREA_MIN) & (df['area'] <= cap)]
    logger.info(f"Area outliers removed: {before - len(df)} rows.")
    return df.reset_index(drop=True)


def remove_price_area_outliers(df):
    
    """Bivariate (price vs area): implied price/sqft must be in a sane band."""
    before = len(df)
    implied_ppsf = df['price_in_cr'] * 1e7 / df['area']
    df = df[implied_ppsf.between(PPSF_MIN, PPSF_MAX)]
    logger.info(f"Price-vs-area outliers removed: {before - len(df)} rows.")
    return df.reset_index(drop=True)


def remove_area_bedroom_outliers(df, min_ratio: int = MIN_AREA_PER_BEDROOM):
    """Bivariate (area vs bedrooms): < min_ratio sqft/bedroom is an impossible layout."""
    before = len(df)
    df = df[(df['area'] / df['bedRoom']) >= min_ratio]
    logger.info(f"Area-per-bedroom outliers removed: {before - len(df)} rows.")
    return df.reset_index(drop=True)


# feature shaping 

def categorize_floornum(df: pd.DataFrame):
    """Categorize floornum into Low-rise, Mid-rise, and High-rise."""
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
    logger.info("Categorized floornum into Low-rise, Mid-rise, High-rise.")
    return df


def reorder_columns(df: pd.DataFrame):
    """Reorder columns into a clean and consistent structure."""
    df = df[[col for col in DESIRED_ORDER if col in df.columns]]
    logger.info("Reordered columns for final dataset.")
    return df


#  pipeline 

def preprocessing(df: pd.DataFrame):
    """
    Preprocessing pipeline for the feature-engineered data.
    Cleans, imputes, removes outliers (univariate + bivariate), shapes features,
    and drops unused columns. Does NOT encode or scale — those are fit on the
    training split inside the model pipeline (leakage-safe).
    """
    try:
        logger.info(f"Preprocessing started — input shape: {df.shape}")

        df = (
            df
            .pipe(fix_bathroom_zero)
            .pipe(fill_floor_nulls)
            .pipe(fill_parking_nulls)
            .pipe(fill_balcony_nulls)
            .pipe(impute_distance_nulls)
            .pipe(clean_facing)
            .pipe(clean_furnishing)
            .pipe(impute_age_possession_category)
            .pipe(cap_rare_societies)
            .pipe(remove_area_outliers)            # univariate
            .pipe(remove_price_area_outliers)      # bivariate: price vs area
            .pipe(remove_area_bedroom_outliers)    # bivariate: area vs bedrooms
            .pipe(categorize_floornum)
            .drop(columns=DROP_COLS, errors='ignore')
            .drop_duplicates()                      
            .reset_index(drop=True)
            .pipe(reorder_columns)
        )

        logger.info(f"Preprocessing completed — output shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        return df
