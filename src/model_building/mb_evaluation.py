import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from .mb_preprocessing import inverse_transform_target


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculates Mean Absolute Percentage Error (MAPE).
    Handles division by zero by masking zero true values.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def scorer(model_name, model, preprocessor,X_train, X_test,y_train_log, y_test_log):

    """
    Fits a pipeline, performs cross-validation, and scores the model on 
    training and test sets (on original price scale).
    """
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    # Cross-validation on training set

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2_log = cross_val_score(
        pipeline,
        X_train,
        y_train_log,
        cv=kf,
        scoring="r2",
        n_jobs=-1
    ).mean()

    # Fit on full training data
    pipeline.fit(X_train, y_train_log)

    # Training predictions
    train_pred_log = pipeline.predict(X_train)
    y_train_pred = inverse_transform_target(train_pred_log)
    y_train_true = inverse_transform_target(y_train_log)

    train_r2 = r2_score(y_train_true, y_train_pred)
    train_mae = mean_absolute_error(y_train_true, y_train_pred)
    train_mape = mean_absolute_percentage_error(y_train_true, y_train_pred)

    # Test predictions
    test_pred_log = pipeline.predict(X_test)
    y_pred = inverse_transform_target(test_pred_log)
    y_true = inverse_transform_target(y_test_log)

    test_r2 = r2_score(y_true, y_pred)
    test_mae = mean_absolute_error(y_true, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    test_mape = mean_absolute_percentage_error(y_true, y_pred)

    print(f"\n--- {model_name} ---")
    print(f"CV R2 (log): {cv_r2_log:.4f}")
    print(f"Test R2: {test_r2:.4f}")
    print(f"Test MAPE: {test_mape:.2f}%")

    return {
        "Model": model_name,
        "CV R2 (log)": round(cv_r2_log, 4),
        "Train R2": round(train_r2, 4),
        "Train MAPE": round(train_mape, 2),
        "Test R2": round(test_r2, 4),
        "Test MAE": round(test_mae, 4),
        "Test RMSE": round(test_rmse, 4),
        "Test MAPE": round(test_mape, 2)}