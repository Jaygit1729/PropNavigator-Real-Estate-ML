# src/data_preprocessing/pre_process_data.py

import numpy as np
import pandas as pd
from src.logger_utils import setup_logger


logger = setup_logger(__name__, 'logs/pre_processing.log')
logger.info("Logging set up successfully for Pre-Processing Module.")


def create_area_missing_flags(df: pd.DataFrame):
    """Creates binary missing-value flags for all four area columns."""
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


def create_primary_area(df: pd.DataFrame):
    """
    Assigns total_area_sqft based on property type:
    - Flats and Builder Floors use super_built_up_area
    - Independent Houses use plot_area
    """
    df = df.copy()
    df["total_area_sqft"] = np.nan

    flat_mask = df["property_type"].isin(["Flat", "Independent Builder Floor"])
    df.loc[flat_mask, "total_area_sqft"] = df.loc[flat_mask, "super_built_up_area"]

    house_mask = df["property_type"] == "Independent House"
    df.loc[house_mask, "total_area_sqft"] = df.loc[house_mask, "plot_area"]

    logger.info("Primary area assigned using property-type logic.")
    return df


def fallback_area_creation(df: pd.DataFrame):
    """
    For rows where total_area_sqft is still missing after primary assignment,
    fills using row-wise median of super_built_up_area, built_up_area, carpet_area.
    """
    df = df.copy()
    fallback_cols = ["super_built_up_area", "built_up_area", "carpet_area"]
    df["total_area_sqft"] = df["total_area_sqft"].fillna(
        df[fallback_cols].median(axis=1)
    )
    logger.info("Fallback area consolidation applied.")
    return df


def ensure_numeric_columns(df: pd.DataFrame):
    """Ensure critical columns are numeric to avoid type errors downstream."""
    df = df.copy()
    cols_to_fix = ["bedrooms", "floornum", "total_area_sqft"]
    for col in cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info("Converted bedrooms, floornum, total_area_sqft to numeric.")
    return df


def dropna(df: pd.DataFrame):
    """Removes rows where critical price columns have null values."""
    initial_rows = df.shape[0]
    df = df.dropna(subset=['price_in_cr', 'price_per_sqft'])
    dropped_count = initial_rows - len(df)

    if dropped_count > 0:
        logger.info(
            f"Dropped {dropped_count} rows with nulls in price_in_cr or price_per_sqft."
        )
    else:
        logger.info("No nulls found in price_in_cr or price_per_sqft. No rows dropped.")
    return df


def remove_total_area_outliers(df: pd.DataFrame, max_area: int = 15000):
    """Removes rows where total_area_sqft exceeds a domain-defined upper bound."""
    df = df.copy()
    before = df.shape[0]
    df = df[df["total_area_sqft"] <= max_area]
    after = df.shape[0]
    logger.info(
        f"Removed {before - after} rows where total_area_sqft > {max_area}."
    )
    return df


def remove_price_per_sqft_outliers(df: pd.DataFrame, max_ppsf: int = 100000):
    """Removes rows where price_per_sqft exceeds a domain-defined upper bound."""
    df = df.copy()
    before = df.shape[0]
    df = df[df["price_per_sqft"] <= max_ppsf]
    after = df.shape[0]
    logger.info(
        f"Removed {before - after} rows where price_per_sqft > {max_ppsf}."
    )
    return df


def remove_area_bedroom_ratio_outliers(
    df: pd.DataFrame,
    min_ratio: int = 200,          

):
    df = df.copy()
    before = df.shape[0]

    df["area_per_bedroom"] = df["total_area_sqft"] / df["bedrooms"]

    # Condition 1 — global minimum ratio (raised to 250)
    # Less than 250 sqft/bedroom is practically impossible
    # for any residential layout including kitchen + bathrooms
    ratio_filter = df["area_per_bedroom"] >= min_ratio

    # Condition 2 — high bedroom count filter
    # 7+ bedrooms need proportionally more space
    high_bedroom_filter = ~(
        (df["bedrooms"] >= 7) &
        (df["area_per_bedroom"] < 300)
    )

    df = df[ratio_filter & high_bedroom_filter]

    after = df.shape[0]
    logger.info(f"Removed {before - after} implausible layout rows.")

    return df

def fill_missing_floornum(df: pd.DataFrame):
    """Fill missing floornum using median value."""
    df = df.copy()
    median_floor = df['floornum'].median()
    df['floornum'] = df['floornum'].fillna(median_floor)
    logger.info(f"Filled missing floornum values with median: {median_floor}.")
    return df


# Three-pass imputation strategy for age_possession_category:
# Pass 1 — impute using mode within same sector AND property_type (most specific)
# Pass 2 — impute using mode within same sector only (broader fallback)
# Pass 3 — impute using mode within same property_type only (broadest fallback)
# Any remaining 'Undefined' values had too little data to impute reliably

def mode_based_imputation(row: pd.Series, df: pd.DataFrame):
    if row['age_possession_category'] == 'Undefined':
        mode_value = df[
            (df['sector'] == row['sector']) &
            (df['property_type'] == row['property_type'])
        ]['age_possession_category'].mode()
        return mode_value.iloc[0] if not mode_value.empty else row['age_possession_category']
    return row['age_possession_category']


def mode_based_imputation2(row: pd.Series, df: pd.DataFrame):
    if row['age_possession_category'] == 'Undefined':
        mode_value = df[
            df['sector'] == row['sector']
        ]['age_possession_category'].mode()
        return mode_value.iloc[0] if not mode_value.empty else row['age_possession_category']
    return row['age_possession_category']


def mode_based_imputation3(row: pd.Series, df: pd.DataFrame):
    if row['age_possession_category'] == 'Undefined':
        mode_value = df[
            df['property_type'] == row['property_type']
        ]['age_possession_category'].mode()
        return mode_value.iloc[0] if not mode_value.empty else row['age_possession_category']
    return row['age_possession_category']


def impute_age_possession_category(df: pd.DataFrame):
    """Apply 3-pass mode-based imputation for age_possession_category."""
    df = df.copy()
    df['age_possession_category'] = df.apply(
        lambda row: mode_based_imputation(row, df), axis=1
    )
    df['age_possession_category'] = df.apply(
        lambda row: mode_based_imputation2(row, df), axis=1
    )
    df['age_possession_category'] = df.apply(
        lambda row: mode_based_imputation3(row, df), axis=1
    )
    logger.info("Applied 3-pass mode-based imputation for age_possession_category.")
    return df

def cap_rare_societies(df: pd.DataFrame, min_count: int = 5) -> pd.DataFrame:
    """
    Replaces rare society values with 'Other' to reduce cardinality.
    Societies appearing fewer than min_count times are grouped together.

    Why this matters:
        With 970 unique societies and ~4400 training rows, many societies
        appear only 1-2 times. Tree models memorize these rare patterns
        during training but cannot generalize them to test data — causing
        overfitting. Grouping rare societies into 'Other' forces the model
        to rely on generalizable patterns instead.

    Args:
        df:        Preprocessed dataframe with society column
        min_count: Minimum occurrences to keep a society label.
                   Societies below this threshold become 'Other'.
    """
    df = df.copy()

    society_counts = df['society'].value_counts()
    rare_societies = society_counts[society_counts < min_count].index
    original_count = df['society'].nunique()

    df['society'] = df['society'].apply(
        lambda x: x if x not in rare_societies else 'Other'
    )

    new_count = df['society'].nunique()
    logger.info(
        f"Capped rare societies — cardinality reduced from "
        f"{original_count} to {new_count} "
        f"({len(rare_societies)} societies grouped as 'Other', "
        f"threshold={min_count})."
    )
    return df


def create_luxury_category(df: pd.DataFrame):
    """Bins luxury_score into 3 quantile-based categories."""
    df = df.copy()
    df['luxury_category'] = pd.qcut(
        df['luxury_score'],
        q=3,
        labels=['Budget', 'Semi-Luxury', 'Luxury'],
        duplicates='drop'
    )
    logger.info("Created luxury_category using quantile binning.")
    return df


def categorize_floornum(df: pd.DataFrame):
    """Categorizes floornum into Low-rise, Mid-rise, and High-rise."""
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
    desired_order = [
        'property_type', 'society', 'sector', 'price_in_cr', 'price_per_sqft',
        'total_area_sqft', 'bedrooms', 'bathrooms', 'balcony', 'floornum_category',
        'study_room', 'servant_room', 'store_room', 'pooja_room', 'others',
        'facing', 'furnishing_type', 'age_possession_category',
        'super_built_up_area_missing', 'built_up_area_missing',
        'carpet_area_missing', 'plot_area_missing',
        'area_per_bedroom', 'dense_house_flag', 'luxury_category'
    ]
    df = df[[col for col in desired_order if col in df.columns]]
    logger.info("Reordered columns for final dataset.")
    return df


def preprocessing(df: pd.DataFrame):
    """Main preprocessing pipeline combining all cleaning and correction steps."""
    try:
        logger.info(f"Preprocessing started — input shape: {df.shape}")

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
            .pipe(cap_rare_societies)              # ← add here

            .pipe(create_luxury_category)
            .pipe(categorize_floornum)
            .drop(
                columns=[
                    'property_id', 'link', 'areawithtype', 'price_per_sqft',
                    'plot_area', 'super_built_up_area', 'built_up_area', 'carpet_area'
                ],
                errors='ignore'
            )
            .pipe(reorder_columns)
            .rename(columns={'age_possession_category': 'age_possession'})
        )

        logger.info(f"Preprocessing completed — output shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        return df