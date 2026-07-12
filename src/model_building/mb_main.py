# src/model_building/mb_main.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error
from src.logger_utils import setup_logger
from .mb_preprocessing import (
    get_feature_lists,
    transform_target,
    inverse_transform_target,
    get_tree_preprocessor,
    get_linear_preprocessor,
    get_catboost_preprocessor
)
from .mb_evaluation import scorer
from .mb_tuning import tune_model
from .mb_persistence import save_model
from .mb_error_analysis import run_error_analysis


logger = setup_logger(__name__, "logs/mb_main.log")

TARGET_COL = "price_in_cr"


def create_train_test_split(df: pd.DataFrame):
    """
    Single source of truth for the train/test split.
    Used by both feature selection and model building to guarantee
    the same rows land in train vs test across the entire pipeline.
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
    Full model building pipeline:
        1. Splits data with stratification on price bins
        2. Derives feature lists dynamically from training data
        3. Evaluates base models
        4. Tunes all models with RandomizedSearchCV
        5. Builds stacking ensemble from tuned models
        6. Selects best model by test MAPE
        7. Saves best model via MAPE-gated persistence
    """
    try:
        logger.info("Model building pipeline started.")
        logger.info(f"Input shape: {fs_df.shape}")

        X_train, X_test, y_train_log, y_test_log = create_train_test_split(
            fs_df
        )
        logger.info(
            f"Train shape: {X_train.shape} | "
            f"Test shape: {X_test.shape}"
        )

        # Derive feature lists dynamically from training data
        # This means changing top_n in feature selection never
        # breaks the model building pipeline
        numerical_features, categorical_features = get_feature_lists(X_train)
        logger.info(
            f"Numerical features ({len(numerical_features)}): "
            f"{numerical_features}"
        )
        logger.info(
            f"Categorical features ({len(categorical_features)}): "
            f"{categorical_features}"
        )

        # Build preprocessors from actual column lists
        tree_preprocessor = get_tree_preprocessor(
            numerical_features, categorical_features
        )
        linear_preprocessor = get_linear_preprocessor(
            numerical_features, categorical_features
        )
        catboost_preprocessor = get_catboost_preprocessor(
            numerical_features, categorical_features
        )

        # Base model evaluation
        model_dict = {
            "RandomForest": (
                RandomForestRegressor(random_state=42),
                tree_preprocessor
            ),
            "XGBoost": (
                XGBRegressor(
                    random_state=42,
                    objective="reg:squarederror",
                    tree_method="hist"
                ),
                tree_preprocessor
            ),
            "LightGBM": (
                LGBMRegressor(
                    random_state=42,
                    verbose=-1
                ),
                tree_preprocessor
            ),
            "CatBoost": (
                CatBoostRegressor(
                    random_seed=42,
                    verbose=0,
                    allow_writing_files=False
                ),
                catboost_preprocessor
            ),
            # SVR dropped: O(n^2), impractically slow to tune on ~31k rows and the
            # weakest fit — tree models dominate this problem.
        }

        results = []
        for name, (model, preprocessor) in model_dict.items():
            logger.info(f"Evaluating base model: {name}")
            res = scorer(
                model_name=name,
                model=model,
                preprocessor=preprocessor,
                X_train=X_train,
                X_test=X_test,
                y_train_log=y_train_log,
                y_test_log=y_test_log
            )
            results.append(res)

        results_df = pd.DataFrame(results).sort_values("Test MAPE")
        logger.info(f"Base model results:\n{results_df.to_string()}")
        logger.info("Base model evaluation completed.")

        # Hyperparameter tuning
        # Pass numerical and categorical feature lists so each
        # tuning run builds its preprocessor from actual columns
        tuned_models = {}
        for name, (model, _) in model_dict.items():
            tuned_models[name] = tune_model(
                model_name=name,
                model=model,
                X_train=X_train,
                y_train_log=y_train_log,
                X_test=X_test,
                y_test_log=y_test_log,
                numerical_features=numerical_features,
                categorical_features=categorical_features
            )

        logger.info("Tuning completed.")

        # Evaluate each tuned model and track scores
        best_model_name = None
        best_test_mape = float("inf")
        best_pipeline = None
        candidate_pipelines = {}

        for name, pipeline in tuned_models.items():
            if pipeline is None:
                logger.warning(
                    f"Skipping {name} — tuning returned None."
                )
                continue

            y_pred = inverse_transform_target(pipeline.predict(X_test))
            y_true = inverse_transform_target(y_test_log)
            test_mape = mean_absolute_percentage_error(
                y_true, y_pred
            ) * 100

            logger.info(
                f"{name} tuned Test MAPE: {round(test_mape, 2)}%"
            )
            candidate_pipelines[name] = (pipeline, test_mape)

            if test_mape < best_test_mape:
                best_test_mape = test_mape
                best_model_name = name
                best_pipeline = pipeline

        # Build stacking ensemble from the top 3 tuned models.
        # CatBoost is excluded because its parameter handling is
        # incompatible with sklearn.base.clone() which StackingRegressor
        # requires internally for CV-based stacking.
        CLONE_INCOMPATIBLE = {"CatBoost"}
        stackable = {
            name: (pipe, mape)
            for name, (pipe, mape) in candidate_pipelines.items()
            if name not in CLONE_INCOMPATIBLE
        }

        if len(stackable) >= 2:
            logger.info("Building stacking ensemble...")
            ranked = sorted(
                stackable.items(),
                key=lambda x: x[1][1]
            )
            top_estimators = [
                (name, pipe) for name, (pipe, _) in ranked[:3]
            ]

            stacking = StackingRegressor(
                estimators=top_estimators,
                final_estimator=Ridge(alpha=1.0),
                cv=5,
                n_jobs=-1
            )
            stacking.fit(X_train, y_train_log)

            y_pred_stack = inverse_transform_target(
                stacking.predict(X_test)
            )
            y_true = inverse_transform_target(y_test_log)
            stack_mape = mean_absolute_percentage_error(
                y_true, y_pred_stack
            ) * 100

            logger.info(f"Stacking Test MAPE: {round(stack_mape, 2)}%")

            if stack_mape < best_test_mape:
                best_test_mape = stack_mape
                best_model_name = "Stacking"
                best_pipeline = stacking

        if best_pipeline is None:
            logger.error("All models failed tuning. No model saved.")
            return {}

        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Best test MAPE: {round(best_test_mape, 2)}%")

        # Compute residual quantiles for calibrated prediction intervals
        import numpy as _np
        _y_pred = inverse_transform_target(best_pipeline.predict(X_test))
        _y_true = inverse_transform_target(y_test_log)
        _pct_errors = (_y_true - _y_pred) / _y_pred
        residual_quantiles = {
            "q05": float(_np.percentile(_pct_errors, 5)),
            "q95": float(_np.percentile(_pct_errors, 95)),
            "q10": float(_np.percentile(_pct_errors, 10)),
            "q90": float(_np.percentile(_pct_errors, 90)),
        }
        logger.info(f"Residual quantiles (90% CI): [{residual_quantiles['q05']:.3f}, {residual_quantiles['q95']:.3f}]")

        # Save best model via MAPE-gated persistence
        save_model(
            model_pipeline=best_pipeline,
            model_name=best_model_name,
            metric=round(best_test_mape, 2),
            filepath="artifacts/best_model.joblib",
            residual_quantiles=residual_quantiles
        )

        # Error analysis on best model
        logger.info("Running error analysis on best model...")
        error_results = run_error_analysis(
            pipeline=best_pipeline,
            X_test=X_test,
            y_test_log=y_test_log
        )

        logger.info("Model building pipeline completed successfully.")

        return {
            "base_results": results_df,
            "best_model_name": best_model_name,
            "best_test_mape": best_test_mape,
            "error_analysis": error_results
        }

    except Exception as e:
        logger.error(
            f"Model building pipeline failed: {e}",
            exc_info=True
        )
        raise