# src/feature_selection/feature_selection.py

import numpy as np
import pandas as pd
from src.logger_utils import setup_logger
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE


logger = setup_logger(__name__, 'logs/feature_selection.log')
logger.info("Feature Selection logging initialized.")


def encode_for_feature_selection(
    X: pd.DataFrame,
    y: pd.Series
):
    """
    Ordinally encodes categorical columns and applies log1p to target.
    log1p on target reduces the influence of price skew on tree-based
    importance scores — features should rank by pattern, not scale.
    Note: encoder is fitted and discarded since this is an offline
    feature ranking step, not production inference.
    """
    X = X.copy()
    y = np.log1p(y)

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        encoder = OrdinalEncoder()
        X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

    return X, y


def compute_importances(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_n: int = 15
):
    """
    Computes ensemble feature importance using four methods:
    - Random Forest importance
    - Gradient Boosting importance
    - Permutation importance (on training set)
    - RFE ranking

    All methods are trained on X_train only to prevent leakage.
    Scores are normalized to 0-1 before averaging so no single
    method dominates due to scale differences.
    """
    logger.info("Computing feature importances on training data only.")

    # Random Forest importance
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_imp = pd.Series(rf.feature_importances_, index=X_train.columns, name="rf")

    # Gradient Boosting importance
    gb = GradientBoostingRegressor(random_state=42)
    gb.fit(X_train, y_train)
    gb_imp = pd.Series(gb.feature_importances_, index=X_train.columns, name="gb")

    # Permutation importance on training set
    perm = permutation_importance(
        rf, X_train, y_train,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    perm_imp = pd.Series(perm.importances_mean, index=X_train.columns, name="perm")

    # RFE ranking
    rfe = RFE(
        RandomForestRegressor(n_estimators=200, random_state=42),
        n_features_to_select=top_n
    )
    rfe.fit(X_train, y_train)
    rfe_rank = pd.Series(rfe.ranking_, index=X_train.columns, name="rfe_rank")

    # Combine and normalize magnitude-based importances
    importance_df = pd.concat([rf_imp, gb_imp, perm_imp], axis=1)
    importance_df = (importance_df - importance_df.min()) / (
        importance_df.max() - importance_df.min()
    )

    importance_df["avg_importance"] = importance_df.mean(axis=1)
    importance_df["rfe_score"] = 1 / rfe_rank
    importance_df["final_score"] = (
        importance_df["avg_importance"] + importance_df["rfe_score"]
    ) / 2
    importance_df = importance_df.sort_values("final_score", ascending=False)

    logger.info("Feature importance computation completed.")
    return importance_df


def feature_selection_pipeline(
    df: pd.DataFrame,
    target: str = 'price_in_cr',
    top_n: int = 15
):
    """
    Leakage-free feature selection pipeline.

    Splits data internally, computes importance on training split only,
    then applies selected features to the full dataset.

    """
    try:
        logger.info("Starting leakage-free feature selection pipeline.")

        df = df.copy()
        X = df.drop(columns=[target])
        y = df[target]

        # Split happens here once — feature importance is computed
        # on training data only so test rows are never seen during selection
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.info(f"Training shape for feature ranking: {X_train.shape}")

        # Encode training data only
        X_train_enc, y_train_enc = encode_for_feature_selection(X_train, y_train)

        # Compute importance using training data only
        importance_df = compute_importances(X_train_enc, y_train_enc, top_n)

        # Select top features
        selected_features = importance_df.head(top_n).index.tolist()
        logger.info(f"Selected top {top_n} features:")
        for f in selected_features:
            logger.info(f"  - {f}")

        # Reduce full dataset to selected features
        fs_df = df[selected_features + [target]].copy()

        logger.info(f"Final shape after feature selection: {fs_df.shape}")
        logger.info("Feature selection pipeline completed successfully.")
        return fs_df, importance_df

    except Exception as e:
        logger.error(f"Feature selection pipeline failed: {e}")
        return None, None