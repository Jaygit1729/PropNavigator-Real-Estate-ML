# src/model_building/mb_evaluation.py

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)
from sklearn.pipeline import Pipeline
from src.logger_utils import setup_logger
from .mb_preprocessing import inverse_transform_target


logger = setup_logger(__name__, "logs/mb_evaluation.log")


def scorer(
    model_name: str,
    model,
    preprocessor,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_log: pd.Series,
    y_test_log: pd.Series
) -> dict:
    """
    Builds a pipeline, runs cross-validation on training data,
    fits on full training set, then evaluates on both train and test.

    All metrics are computed on original price scale (after inverse
    log transform) so they are interpretable in rupees/crores.

    Returns a dict of metrics for comparison across models.
    """
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    # Cross-validation on training set only
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2_log = cross_val_score(
        pipeline, X_train, y_train_log,
        cv=kf, scoring="r2", n_jobs=-1
    ).mean()

    # Fit on full training data
    pipeline.fit(X_train, y_train_log)

    # Training metrics on original price scale
    y_train_pred = inverse_transform_target(pipeline.predict(X_train))
    y_train_true = inverse_transform_target(y_train_log)
    train_r2 = r2_score(y_train_true, y_train_pred)
    train_mae = mean_absolute_error(y_train_true, y_train_pred)
    train_mape = mean_absolute_percentage_error(y_train_true, y_train_pred) * 100

    # Test metrics on original price scale
    y_pred = inverse_transform_target(pipeline.predict(X_test))
    y_true = inverse_transform_target(y_test_log)
    test_r2 = r2_score(y_true, y_pred)
    test_mae = mean_absolute_error(y_true, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    test_mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    logger.info(
        f"{model_name} — CV R2: {cv_r2_log:.4f} | "
        f"Test R2: {test_r2:.4f} | Test MAPE: {test_mape:.2f}%"
    )

    return {
        "Model": model_name,
        "CV R2 (log)": round(cv_r2_log, 4),
        "Train R2": round(train_r2, 4),
        "Train MAPE": round(train_mape, 2),
        "Test R2": round(test_r2, 4),
        "Test MAE": round(test_mae, 4),
        "Test RMSE": round(test_rmse, 4),
        "Test MAPE": round(test_mape, 2)
    }