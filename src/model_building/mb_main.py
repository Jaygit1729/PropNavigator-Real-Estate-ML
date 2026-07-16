# src/model_building/mb_main.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from src.logger_utils import setup_logger
from .mb_preprocessing import (
    get_feature_lists,
    transform_target,
    inverse_transform_target,
)
from .mb_tuning import tune_model
from .mb_persistence import save_model

import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

load_dotenv()


logger = setup_logger(__name__, "logs/mb_main.log")

TARGET_COL = "price_in_cr"
EXPERIMENT_NAME = "propnavigator-model-building"
REGISTERED_MODEL_NAME = "propnavigator-price-model"


def create_train_test_split(df: pd.DataFrame):
    """
    Single source of truth for the train/test split.
    Stratifies on price bins so the same rows land in train vs test
    consistently across the pipeline.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    y_log = transform_target(y)
    price_bins = pd.qcut(y, q=5, labels=False)

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log,
        stratify=price_bins,
        test_size=0.2,
        random_state=42
    )
    return X_train, X_test, y_train_log, y_test_log


def run_model_building(fs_df: pd.DataFrame):
    """
    Model building pipeline:
        1. Split data (stratified on price bins)
        2. Derive feature lists from training data
        3. Tune XGBoost, LightGBM, CatBoost (RandomizedSearchCV)
        4. Pick the single best model by test MAPE
        5. Log every tuned model to MLflow; register the winner
        6. Save best model via MAPE-gated persistence

    Note: RandomForest and stacking were intentionally dropped. On this
    tabular data the gradient-boosting trio wins, RandomForest was the
    slowest to tune and the weakest, and a single model is simpler to
    serve and to explain with SHAP than a stacked ensemble.
    """
    try:
        logger.info("Model building pipeline started.")
        logger.info(f"Input shape: {fs_df.shape}")

        X_train, X_test, y_train_log, y_test_log = create_train_test_split(
            fs_df
        )
        logger.info(
            f"Train shape: {X_train.shape} | Test shape: {X_test.shape}"
        )

        # Feature lists derived dynamically from training data, so changes
        # in feature selection never break model building.
        numerical_features, categorical_features = get_feature_lists(X_train)
        logger.info(
            f"Numerical features ({len(numerical_features)}): "
            f"{numerical_features}"
        )
        logger.info(
            f"Categorical features ({len(categorical_features)}): "
            f"{categorical_features}"
        )

        # Candidate models — all tree-based, all use the tree preprocessor.
        models_to_tune = {
            "XGBoost": XGBRegressor(
                random_state=42,
                objective="reg:squarederror",
                tree_method="hist"
            ),
            "LightGBM": LGBMRegressor(
                random_state=42,
                verbose=-1
            ),
            "CatBoost": CatBoostRegressor(
                random_seed=42,
                verbose=0,
                allow_writing_files=False
            ),
        }

        # Tune each model and collect its results.
        results = {}
        for name, model in models_to_tune.items():
            info = tune_model(
                model_name=name,
                model=model,
                X_train=X_train,
                y_train_log=y_train_log,
                X_test=X_test,
                y_test_log=y_test_log,
                numerical_features=numerical_features,
                categorical_features=categorical_features
            )
            if info is None:
                logger.warning(f"Skipping {name} — tuning returned None.")
                continue
            results[name] = info
            logger.info(f"{name} tuned Test MAPE: {info['test_mape']}%")

        if not results:
            logger.error("All models failed tuning. No model saved.")
            return {}

        # Pick the single best model by test MAPE.
        best_model_name = min(
            results, key=lambda n: results[n]["test_mape"]
        )
        best_info = results[best_model_name]
        best_pipeline = best_info["pipeline"]
        best_test_mape = best_info["test_mape"]
        logger.info(
            f"Best model: {best_model_name} ({best_test_mape}% MAPE)"
        )

        # Residual quantiles for calibrated prediction intervals.
        y_pred = inverse_transform_target(best_pipeline.predict(X_test))
        y_true = inverse_transform_target(y_test_log)
        pct_errors = (y_true - y_pred) / y_pred
        residual_quantiles = {
            "q05": float(np.percentile(pct_errors, 5)),
            "q95": float(np.percentile(pct_errors, 95)),
            "q10": float(np.percentile(pct_errors, 10)),
            "q90": float(np.percentile(pct_errors, 90)),
        }
        logger.info(
            f"Residual quantiles (90% CI): "
            f"[{residual_quantiles['q05']:.3f}, "
            f"{residual_quantiles['q95']:.3f}]"
        )

        # MLflow: one run per tuned model; the winner's model is logged
        # and registered in the Model Registry.
        mlflow.set_experiment(EXPERIMENT_NAME)
        for name, info in results.items():
            with mlflow.start_run(run_name=name):
                mlflow.log_param("model_type", name)
                mlflow.log_param("test_size", 0.2)
                mlflow.log_param("random_state", 42)
                # Log the feature count so every run self-describes in the UI
                # (24 = society dropped, 25 = society included).
                mlflow.log_param("n_features", X_train.shape[1])
                mlflow.log_params(info["best_params"])
                mlflow.log_metric("test_mape", info["test_mape"])
                mlflow.log_metric("test_r2", info["test_r2"])
                mlflow.log_metric("train_mape", info["train_mape"])
                if name == best_model_name:
                    mlflow.log_param("is_best", True)
                    mlflow.sklearn.log_model(
                        best_pipeline,
                        name="model",
                        serialization_format="cloudpickle",
                        registered_model_name=REGISTERED_MODEL_NAME
                    )
        logger.info("MLflow logging complete.")

        # Save best model via MAPE-gated persistence (unchanged artifact
        # contract: dict with pipeline / model_name / test_mape_percent /
        # residual_quantiles — the Streamlit pages depend on this).
        save_model(
            model_pipeline=best_pipeline,
            model_name=best_model_name,
            metric=round(best_test_mape, 2),
            filepath="artifacts/best_model.joblib",
            residual_quantiles=residual_quantiles
        )

        logger.info("Model building pipeline completed successfully.")

        return {
            "best_model_name": best_model_name,
            "best_test_mape": best_test_mape,
            "all_results": {
                n: i["test_mape"] for n, i in results.items()
            },
        }

    except Exception as e:
        logger.error(
            f"Model building pipeline failed: {e}",
            exc_info=True
        )
        raise
