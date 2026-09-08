# src/model_building/model_building.py

import csv
import os
import warnings
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    make_scorer,
    r2_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.logger_utils import setup_logger


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

logger = setup_logger(__name__, "logs/model_building.log")

TARGET_COL = "price_in_cr"
EXPERIMENT_NAME = "propnavigator-model-building"
REGISTERED_MODEL_NAME = "propnavigator-price-model"
MODEL_PATH = "artifacts/best_model.joblib"
EXPERIMENT_LOG = "artifacts/experiment_log.csv"

RANDOM_STATE = 42
N_ITER = 25
N_SPLITS = 3


# ---------------------------------------------------------------------
# Target transformation
# ---------------------------------------------------------------------

def transform_target(y):
    """Convert price to log scale for model training."""
    return np.log1p(y)


def inverse_transform_target(y_log):
    """Convert predictions back to the original price scale."""
    return np.expm1(y_log)


# ---------------------------------------------------------------------
# Feature preprocessing
# ---------------------------------------------------------------------

def get_feature_lists(X):
    """Identify numerical and categorical columns."""
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    return numerical_features, categorical_features


def create_preprocessor(numerical_features, categorical_features):
    """
    Encode categorical features and pass numerical features through.
    Unknown categories are handled safely at prediction time.
    """
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                categorical_features,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------

PARAMETER_SPACES = {
    "XGBoost": {
        "regressor__learning_rate": sp_uniform(0.01, 0.05),
        "regressor__n_estimators": sp_randint(500, 1000),
        "regressor__max_depth": sp_randint(3, 6),
        "regressor__subsample": sp_uniform(0.6, 0.4),
        "regressor__colsample_bytree": sp_uniform(0.6, 0.4),
        "regressor__reg_alpha": [0.1, 0.5, 1, 5],
        "regressor__reg_lambda": [1, 5, 10],
        "regressor__min_child_weight": sp_randint(3, 8),
    },

    "LightGBM": {
        "regressor__learning_rate": sp_uniform(0.01, 0.09),
        "regressor__n_estimators": sp_randint(400, 3000),
        "regressor__max_depth": sp_randint(4, 10),
        "regressor__num_leaves": sp_randint(20, 80),
        "regressor__subsample": sp_uniform(0.6, 0.4),
        "regressor__subsample_freq": [0, 1, 5],
        "regressor__colsample_bytree": sp_uniform(0.6, 0.4),
        "regressor__reg_alpha": [0, 0.1, 0.5, 1, 5],
        "regressor__reg_lambda": [0, 1, 5, 10],
        "regressor__min_child_samples": sp_randint(5, 30),
    },

    "CatBoost": {
        "regressor__learning_rate": sp_uniform(0.01, 0.09),
        "regressor__iterations": sp_randint(500, 1200),
        "regressor__depth": sp_randint(4, 8),
        "regressor__l2_leaf_reg": [1, 3, 5, 7, 10],
        "regressor__bagging_temperature": sp_uniform(0, 1),
        "regressor__random_strength": sp_uniform(0, 2),
    },
}


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def mape_on_price_scale(y_log_true, y_log_pred):
    """Calculate MAPE after converting values back to price scale."""
    y_true = inverse_transform_target(y_log_true)
    y_pred = inverse_transform_target(y_log_pred)

    return mean_absolute_percentage_error(y_true, y_pred)


neg_mape_scorer = make_scorer(
    mape_on_price_scale,
    greater_is_better=False,
)


def calculate_metrics(model, X, y_log):
    """Calculate regression metrics on the original price scale."""
    y_true = inverse_transform_target(y_log)
    y_pred = inverse_transform_target(model.predict(X))

    return {
        "r2": round(r2_score(y_true, y_pred), 4),
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(
            np.sqrt(mean_squared_error(y_true, y_pred)),
            4,
        ),
        "mape": round(
            mean_absolute_percentage_error(y_true, y_pred) * 100,
            2,
        ),
    }


# ---------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------

def create_train_val_test_split(df):
    """
    Create a 60/20/20 train/validation/test split.

    Price quintiles are used only for stratification so that
    all three datasets have a similar price distribution.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    y_log = transform_target(y)

    # Five price groups used only for stratification.
    price_bins = pd.qcut(
        y,
        q=5,
        labels=False,
    )

    # Step 1: reserve 20% as untouched test data.
    X_temp, X_test, y_temp_log, y_test_log, bins_temp, _ = (
        train_test_split(
            X,
            y_log,
            price_bins,
            test_size=0.20,
            stratify=price_bins,
            random_state=RANDOM_STATE,
        )
    )

    # Step 2: split remaining 80% into 60% train + 20% validation.
    X_train, X_val, y_train_log, y_val_log = train_test_split(
        X_temp,
        y_temp_log,
        test_size=0.25,
        stratify=bins_temp,
        random_state=RANDOM_STATE,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train_log,
        y_val_log,
        y_test_log,
    )


# ---------------------------------------------------------------------
# Model tuning
# ---------------------------------------------------------------------

def tune_model(
    model_name,
    model,
    X_train,
    y_train_log,
    X_val,
    y_val_log,
    numerical_features,
    categorical_features,
):
    """Tune one model and evaluate its best version on validation data."""

    logger.info(f"Tuning started for {model_name}.")

    preprocessor = create_preprocessor(
        numerical_features,
        categorical_features,
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", model),
        ]
    )

    # Use price quintiles so CV folds have similar price distributions.
    price_bins = pd.qcut(
        inverse_transform_target(y_train_log),
        q=5,
        labels=False,
    )

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Materialize the same folds so every model is compared fairly.
    cv_splits = list(cv.split(X_train, price_bins))

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=PARAMETER_SPACES[model_name],
        n_iter=N_ITER,
        scoring=neg_mape_scorer,
        cv=cv_splits,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train_log)

    best_model = search.best_estimator_

    train_metrics = calculate_metrics(
        best_model,
        X_train,
        y_train_log,
    )

    val_metrics = calculate_metrics(
        best_model,
        X_val,
        y_val_log,
    )

    cv_mape = round(
        -search.best_score_ * 100,
        2,
    )

    logger.info(
        f"{model_name} | "
        f"CV MAPE: {cv_mape}% | "
        f"Train MAPE: {train_metrics['mape']}% | "
        f"Val MAPE: {val_metrics['mape']}%"
    )

    logger.info(
        f"{model_name} best parameters: "
        f"{search.best_params_}"
    )

    return {
        "pipeline": best_model,
        "best_params": search.best_params_,
        "cv_mape": cv_mape,
        "train_mape": train_metrics["mape"],
        "val_mape": val_metrics["mape"],
        "val_r2": val_metrics["r2"],
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
    }


# ---------------------------------------------------------------------
# Prediction interval calibration
# ---------------------------------------------------------------------

def calculate_residual_quantiles(model, X_val, y_val_log):
    """
    Estimate prediction-error quantiles from validation data.

    These are later used to create approximate prediction intervals.
    """
    y_true = inverse_transform_target(y_val_log)
    y_pred = inverse_transform_target(model.predict(X_val))

    # Percentage error relative to predicted price.
    pct_errors = (y_true - y_pred) / y_pred

    return {
        "q05": float(np.percentile(pct_errors, 5)),
        "q95": float(np.percentile(pct_errors, 95)),
        "q10": float(np.percentile(pct_errors, 10)),
        "q90": float(np.percentile(pct_errors, 90)),
    }


# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------

def save_model(
    pipeline,
    model_name,
    val_mape,
    test_mape,
    residual_quantiles,
    filepath=MODEL_PATH,
):
    """Write the winning model plus the metadata the app reads from it."""

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    artifact = {
        "model_name": model_name,
        "val_mape_percent": val_mape,
        "test_mape_percent": test_mape,
        "pipeline": pipeline,
        "residual_quantiles": residual_quantiles,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }

    # Dated copy first, so a previous model is never the only casualty
    # of a failed write.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = filepath.replace(".joblib", f"_{timestamp}.joblib")

    # Second-resolution stamps collide if two saves land in the same second.
    collision = 2
    while os.path.exists(versioned_path):
        versioned_path = filepath.replace(
            ".joblib", f"_{timestamp}_{collision}.joblib"
        )
        collision += 1

    joblib.dump(artifact, versioned_path)
    joblib.dump(artifact, filepath)

    _append_experiment_log(model_name, val_mape, test_mape, filepath)

    logger.info(
        f"Saved {model_name} to {filepath} "
        f"(version: {versioned_path}) | "
        f"Val MAPE: {val_mape}% | Test MAPE: {test_mape}%"
    )


def _append_experiment_log(model_name, val_mape, test_mape, filepath):
    """Append one row per run. Local fallback for when MLflow is unreachable."""

    os.makedirs(os.path.dirname(EXPERIMENT_LOG), exist_ok=True)
    is_new = not os.path.exists(EXPERIMENT_LOG)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "val_mape_percent": val_mape,
        "test_mape_percent": test_mape,
        "artifact_path": filepath,
    }

    with open(EXPERIMENT_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------

def log_to_mlflow(
    results,
    best_model_name,
    best_test_mape,
    best_test_r2,
    n_features,
):
    """Log model comparison results and register the winner."""

    try:
        mlflow.set_experiment(EXPERIMENT_NAME)

        for model_name, result in results.items():

            with mlflow.start_run(run_name=model_name):

                mlflow.log_params(
                    {
                        "model_type": model_name,
                        "split": "60/20/20 train/val/test",
                        "selected_on": "validation",
                        "random_state": RANDOM_STATE,
                        "n_features": n_features,
                    }
                )

                mlflow.log_params(result["best_params"])

                mlflow.log_metrics(
                    {
                        "cv_mape": result["cv_mape"],
                        "train_mape": result["train_mape"],
                        "val_mape": result["val_mape"],
                        "val_r2": result["val_r2"],
                        "val_mae": result["val_mae"],
                        "val_rmse": result["val_rmse"],
                    }
                )

                if model_name == best_model_name:

                    mlflow.log_param(
                        "is_best",
                        True,
                    )

                    # Only the winner gets a test score.
                    mlflow.log_metrics(
                        {
                            "test_mape": best_test_mape,
                            "test_r2": best_test_r2,
                        }
                    )

                    mlflow.sklearn.log_model(
                        result["pipeline"],
                        name="model",
                        serialization_format="cloudpickle",
                        registered_model_name=REGISTERED_MODEL_NAME,
                    )

        logger.info("MLflow logging complete.")

    except Exception as e:
        logger.warning(
            f"MLflow logging failed: {e}. "
            "Model is already saved locally."
        )


# ---------------------------------------------------------------------
# Main model-building pipeline
# ---------------------------------------------------------------------

def run_model_building(fs_df: pd.DataFrame):
    """
    Complete model-building workflow:

    1. Split data into train/validation/test.
    2. Identify feature types.
    3. Tune XGBoost, LightGBM and CatBoost.
    4. Select the winner using validation MAPE.
    5. Evaluate the winner once on untouched test data.
    6. Calibrate prediction intervals using validation residuals.
    7. Save the winning model.
    8. Log experiments to MLflow.
    """

    try:
        logger.info("Model building pipeline started.")
        logger.info(f"Input shape: {fs_df.shape}")

        # -------------------------------------------------------------
        # 1. Split data
        # -------------------------------------------------------------

        (
            X_train,
            X_val,
            X_test,
            y_train_log,
            y_val_log,
            y_test_log,
        ) = create_train_val_test_split(fs_df)

        logger.info(
            f"Train: {X_train.shape} | "
            f"Validation: {X_val.shape} | "
            f"Test: {X_test.shape}"
        )

        # -------------------------------------------------------------
        # 2. Identify feature types
        # -------------------------------------------------------------

        numerical_features, categorical_features = get_feature_lists(
            X_train
        )

        logger.info(
            f"Numerical features: {numerical_features}"
        )
        logger.info(
            f"Categorical features: {categorical_features}"
        )

        # -------------------------------------------------------------
        # 3. Define candidate models
        # -------------------------------------------------------------

        models = {
            "XGBoost": XGBRegressor(
                random_state=RANDOM_STATE,
                objective="reg:squarederror",
                tree_method="hist",
            ),

            "LightGBM": LGBMRegressor(
                random_state=RANDOM_STATE,
                verbose=-1,
            ),

            "CatBoost": CatBoostRegressor(
                random_seed=RANDOM_STATE,
                verbose=0,
                allow_writing_files=False,
            ),
        }

        # -------------------------------------------------------------
        # 4. Tune every candidate model
        # -------------------------------------------------------------

        results = {}

        for model_name, model in models.items():

            try:
                results[model_name] = tune_model(
                    model_name=model_name,
                    model=model,
                    X_train=X_train,
                    y_train_log=y_train_log,
                    X_val=X_val,
                    y_val_log=y_val_log,
                    numerical_features=numerical_features,
                    categorical_features=categorical_features,
                )

            except Exception as e:
                logger.error(
                    f"{model_name} tuning failed: {e}",
                    exc_info=True,
                )

        if not results:
            logger.error(
                "All models failed tuning. No model saved."
            )
            return {}

        # -------------------------------------------------------------
        # 5. Select winner using validation MAPE
        # -------------------------------------------------------------

        best_model_name = min(
            results,
            key=lambda name: results[name]["val_mape"],
        )

        best_result = results[best_model_name]
        best_pipeline = best_result["pipeline"]
        best_val_mape = best_result["val_mape"]

        logger.info(
            f"Best model: {best_model_name} | "
            f"Validation MAPE: {best_val_mape}%"
        )

        # -------------------------------------------------------------
        # 6. Evaluate winner on untouched test set
        # -------------------------------------------------------------

        test_metrics = calculate_metrics(
            best_pipeline,
            X_test,
            y_test_log,
        )

        best_test_mape = test_metrics["mape"]
        best_test_r2 = test_metrics["r2"]

        logger.info(
            f"HELD-OUT TEST | "
            f"{best_model_name} | "
            f"MAPE: {best_test_mape}% | "
            f"R2: {best_test_r2}"
        )

        # -------------------------------------------------------------
        # 7. Calibrate prediction intervals
        # -------------------------------------------------------------

        residual_quantiles = calculate_residual_quantiles(
            best_pipeline,
            X_val,
            y_val_log,
        )

        logger.info(
            f"Residual quantiles: "
            f"{residual_quantiles}"
        )

        # -------------------------------------------------------------
        # 8. Save winning model
        # -------------------------------------------------------------

        save_model(
            pipeline=best_pipeline,
            model_name=best_model_name,
            val_mape=best_val_mape,
            test_mape=best_test_mape,
            residual_quantiles=residual_quantiles,
        )

        # -------------------------------------------------------------
        # 9. Log experiments and register winner
        # -------------------------------------------------------------

        log_to_mlflow(
            results=results,
            best_model_name=best_model_name,
            best_test_mape=best_test_mape,
            best_test_r2=best_test_r2,
            n_features=X_train.shape[1],
        )

        logger.info(
            "Model building pipeline completed successfully."
        )

        # -------------------------------------------------------------
        # 10. Return summary
        # -------------------------------------------------------------

        return {
            "best_model_name": best_model_name,
            "best_val_mape": best_val_mape,
            "best_test_mape": best_test_mape,
            "best_test_r2": best_test_r2,
            "all_val_results": {
                name: result["val_mape"]
                for name, result in results.items()
            },
        }

    except Exception as e:
        logger.error(
            f"Model building pipeline failed: {e}",
            exc_info=True,
        )
        raise
