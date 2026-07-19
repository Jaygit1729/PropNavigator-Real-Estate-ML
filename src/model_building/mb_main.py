# src/model_building/mb_main.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score
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


def create_train_val_test_split(df: pd.DataFrame):
    """
    Single source of truth for the 60/20/20 train / validation / test split.

    Why three splits and not two: every time data is used to MAKE A CHOICE it
    can no longer give an honest score for what was chosen. Hyperparameters are
    chosen by CV on train; the winning model FAMILY is chosen on validation;
    test is touched exactly once, at the end, to report. Selecting the family on
    test — as this pipeline previously did — makes the headline metric
    optimistic, because whichever model got luckiest on those rows wins.

    Stratified on price quintiles so all three splits span the price range.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    y_log = transform_target(y)
    price_bins = pd.qcut(y, q=5, labels=False)

    # First carve off the test set (20%) and leave it alone until the very end.
    X_temp, X_test, y_temp_log, y_test_log, bins_temp, _ = train_test_split(
        X, y_log, price_bins,
        stratify=price_bins,
        test_size=0.2,
        random_state=42
    )

    # Then split the remainder into train (60% of all) and validation (20% of all).
    X_train, X_val, y_train_log, y_val_log = train_test_split(
        X_temp, y_temp_log,
        stratify=bins_temp,
        test_size=0.25,          # 0.25 of the remaining 80% = 20% overall
        random_state=42
    )
    return X_train, X_val, X_test, y_train_log, y_val_log, y_test_log


def run_model_building(fs_df: pd.DataFrame):
    """
    Model building pipeline:
        1. Split data 60/20/20 into train / validation / test
        2. Derive feature lists from training data
        3. Tune XGBoost, LightGBM, CatBoost (RandomizedSearchCV on train)
        4. Pick the single best model by VALIDATION MAPE
        5. Score the winner on the untouched test set — the reported number
        6. Log every tuned model to MLflow; register the winner
        7. Save best model via MAPE-gated persistence

    Note: RandomForest and stacking were intentionally dropped. On this
    tabular data the gradient-boosting trio wins, RandomForest was the
    slowest to tune and the weakest, and a single model is simpler to
    serve and to explain with SHAP than a stacked ensemble.
    """
    try:
        logger.info("Model building pipeline started.")
        logger.info(f"Input shape: {fs_df.shape}")

        (X_train, X_val, X_test,
         y_train_log, y_val_log, y_test_log) = create_train_val_test_split(fs_df)
        logger.info(
            f"Train shape: {X_train.shape} | Val shape: {X_val.shape} | "
            f"Test shape: {X_test.shape}"
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
                X_val=X_val,
                y_val_log=y_val_log,
                numerical_features=numerical_features,
                categorical_features=categorical_features
            )
            if info is None:
                logger.warning(f"Skipping {name} — tuning returned None.")
                continue
            results[name] = info
            logger.info(f"{name} tuned Validation MAPE: {info['val_mape']}%")

        if not results:
            logger.error("All models failed tuning. No model saved.")
            return {}

        # Pick the winner on VALIDATION — test stays untouched so the number
        # we report is not inflated by having been used to choose.
        best_model_name = min(
            results, key=lambda n: results[n]["val_mape"]
        )
        best_info = results[best_model_name]
        best_pipeline = best_info["pipeline"]
        best_val_mape = best_info["val_mape"]
        logger.info(
            f"Best model by validation: {best_model_name} "
            f"({best_val_mape}% val MAPE)"
        )

        # Now — and only now — score the winner on the held-out test set.
        # This single number is the honest, reportable performance.
        y_test_pred = inverse_transform_target(best_pipeline.predict(X_test))
        y_test_true = inverse_transform_target(y_test_log)
        best_test_mape = round(
            mean_absolute_percentage_error(y_test_true, y_test_pred) * 100, 2
        )
        best_test_r2 = round(r2_score(y_test_true, y_test_pred), 4)
        logger.info(
            f"HELD-OUT TEST — {best_model_name}: "
            f"MAPE {best_test_mape}% | R2 {best_test_r2}"
        )

        # Residual quantiles for prediction intervals, calibrated on VALIDATION
        # so the test set is used for reporting only.
        y_pred = inverse_transform_target(best_pipeline.predict(X_val))
        y_true = inverse_transform_target(y_val_log)
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
                mlflow.log_param("split", "60/20/20 train/val/test")
                mlflow.log_param("selected_on", "validation")
                mlflow.log_param("random_state", 42)
                # Log the feature count so every run self-describes in the UI
                # (24 = society dropped, 25 = society included).
                mlflow.log_param("n_features", X_train.shape[1])
                mlflow.log_params(info["best_params"])
                mlflow.log_metric("val_mape", info["val_mape"])
                mlflow.log_metric("val_r2", info["val_r2"])
                mlflow.log_metric("train_mape", info["train_mape"])
                if name == best_model_name:
                    mlflow.log_param("is_best", True)
                    # Only the winner gets a test score — logged here so the
                    # UI shows exactly one honest, held-out number.
                    mlflow.log_metric("test_mape", best_test_mape)
                    mlflow.log_metric("test_r2", best_test_r2)
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
            "best_val_mape": best_val_mape,
            "best_test_mape": best_test_mape,
            "best_test_r2": best_test_r2,
            "all_val_results": {
                n: i["val_mape"] for n, i in results.items()
            },
        }

    except Exception as e:
        logger.error(
            f"Model building pipeline failed: {e}",
            exc_info=True
        )
        raise
