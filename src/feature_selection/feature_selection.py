# src/feature_selection/feature_selection.py

import pandas as pd
from src.logger_utils import setup_logger


logger = setup_logger(__name__, 'logs/feature_selection.log')
logger.info("Feature Selection logging initialized.")


SELECTED_FEATURES = [
    'area', 'dist_to_golf_road', 'total_floor', 'bathroom', 'property_type',
    'dist_to_cyber_city', 'covered_parking', 'dist_to_manesar', 'sector',
    'dist_to_airport', 'bedRoom', 'furnishing', 'balcony', 'age_possession_category',
    'facing', 'has_ac', 'total_parking', 'open_parking', 'has_power_backup',
    'is_corner', 'floornum_category', 'ov_main_road', 'has_pool', 'ov_others',
]


def select_features(df: pd.DataFrame, target: str = 'price_in_cr'):
    """
    Keep the top features (+ target), then drop duplicate properties.

    The de-duplication matters: earlier cleaning de-duplicates on the FULL
    column set, but once we narrow to the modelling features two distinct
    listings can collapse into identical rows. If such twins land on opposite
    sides of the train/test split the model can memorise the answer, which
    inflates the test score. De-duplicating here — on the final feature set,
    before any split — closes that leak.
    """
    try:
        keep = [c for c in SELECTED_FEATURES if c in df.columns]
        missing = [c for c in SELECTED_FEATURES if c not in df.columns]
        if missing:
            logger.warning(f"Selected features not found in data (skipped): {missing}")

        fs_df = df[keep + [target]].copy()
        logger.info(f"Selected {len(keep)} features + target — shape: {fs_df.shape}")

        before = len(fs_df)
        fs_df = fs_df.drop_duplicates(subset=keep).reset_index(drop=True)
        logger.info(
            f"Dropped {before - len(fs_df)} duplicate feature rows "
            f"(prevents train/test memorisation) — shape: {fs_df.shape}"
        )
        return fs_df

    except Exception as e:
        logger.error(f"Feature selection failed: {e}")
        return None
