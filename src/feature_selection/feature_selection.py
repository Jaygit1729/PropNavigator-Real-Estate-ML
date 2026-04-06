import numpy as np
import pandas as pd
from src.logger_utils import setup_logger
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE


logger = setup_logger('logs/feature_selection.log')
logger.info("Feature Selection logging initialized.")


#  Encoding 

def encode_for_feature_selection(X, y):

    X = X.copy()
    y = np.log1p(y)

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    if len(categorical_cols) > 0:
        encoder = OrdinalEncoder()
        X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

    return X, y


#  Compute Importances 

def compute_importances(X_train, y_train, top_n=15):

    logger.info("Computing feature importances on TRAINING DATA ONLY.")

    # Random Forest

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_imp = pd.Series(rf.feature_importances_, index=X_train.columns, name="rf")

    # Gradient Boosting

    gb = GradientBoostingRegressor(random_state=42)
    gb.fit(X_train, y_train)
    gb_imp = pd.Series(gb.feature_importances_, index=X_train.columns, name="gb")

    # Permutation Importance (on training only)

    perm = permutation_importance(
        rf,
        X_train,
        y_train,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    perm_imp = pd.Series(perm.importances_mean, index=X_train.columns, name="perm")

    # RFE

    rfe = RFE(
        RandomForestRegressor(n_estimators=200, random_state=42),
        n_features_to_select=top_n
    )
    rfe.fit(X_train, y_train)
    rfe_rank = pd.Series(rfe.ranking_, index=X_train.columns, name="rfe_rank")

    # Combine magnitude-based importances

    importance_df = pd.concat([rf_imp, gb_imp, perm_imp], axis=1)

    # Normalize

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



#  Main Pipeline 
 
def feature_selection_pipeline(df, target='price_in_cr', top_n=15):

    logger.info("Starting Leakage-Free Feature Selection Pipeline.")

    df = df.copy()

    X = df.drop(columns=[target])
    y = df[target]


    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info(f"Training shape for feature ranking: {X_train.shape}")

    # Encode training data only

    X_train_enc, y_train_enc = encode_for_feature_selection(X_train, y_train)

    # Compute importance using training only

    importance_df = compute_importances(X_train_enc, y_train_enc, top_n)

    # Select top features

    selected_features = importance_df.head(top_n).index.tolist()

    logger.info(f"Selected top {top_n} features:")
    for f in selected_features:
        logger.info(f"  - {f}")

    # Reduce full dataset using selected feature list

    fs_df = df[selected_features + [target]].copy()

    logger.info(f"Final shape after feature selection: {fs_df.shape}")
    logger.info("Feature Selection Pipeline completed successfully.")

    return fs_df, importance_df