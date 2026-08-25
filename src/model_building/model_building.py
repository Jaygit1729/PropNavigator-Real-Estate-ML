# src/model_building/model_building.py
"""
Model building, end to end, in the order it actually happens:

    1. split          60/20/20, stratified on price quintiles
    2. encode         ordinal for categories, numbers passed through
    3. tune           randomised search, cross-validated inside train only
    4. pick           best validation score chooses the winning family
    5. evaluate       the winner is scored on test once, and that is the number
    6. persist        saved only if it beats the incumbent on validation

"""

import warnings

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from scipy.stats import randint as sp_randint, uniform as sp_uniform
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    make_scorer,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from src.logger_utils import setup_logger
from .persistence import save_model

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

logger = setup_logger(__name__, "logs/model_building.log")


# ---------------------------------------------------------------- encoding ---

def get_feature_lists(X):
    """
    Derives numerical and categorical feature lists dynamically
    from the dataframe passed in.

    """
    numerical_features = X.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()

    return numerical_features, categorical_features


def get_tree_preprocessor(numerical_features: list, categorical_features: list):
    """
    Preprocessor for the tree-based models (XGBoost, LightGBM, CatBoost):
    - Ordinal encodes all categorical features
    - Passes numerical features through unchanged (no scaling needed
      since tree models are scale-invariant)
    - unknown_value=-1 handles unseen categories at inference time
    """
    tree_preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                ),
                categorical_features
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    )
    return tree_preprocessor


def transform_target(y):
    """
    Applies log1p transformation to the target variable.
    log1p compresses the price distribution, reducing the influence
    of expensive outliers on model training.
    """
    return np.log1p(y)


def inverse_transform_target(y_log):
    """
    Reverses log1p transformation using expm1.
    Applied after prediction to get back to original price scale
    before computing evaluation metrics.
    """
    return np.expm1(y_log)


# ------------------------------------------------------------------ tuning ---

def _mape_in_rupees(y_log_true, y_log_pred):
    """MAPE on the original price scale, from log-space inputs.

    The search fits on log1p(price), so a plain MAPE scorer computes relative
    error between LOG values — a different objective from the one reported.
    Because log1p compresses the range, the cheapest properties have tiny
    denominators and dominate: measured on the current test set they take 34%
    of the log-space objective versus 28% of the rupee objective. Tuning was
    therefore optimising a metric nobody reports, tilted toward cheap listings.

    Inverting the transform inside the scorer makes the tuning objective and
    the reported metric the same quantity.
    """
    return mean_absolute_percentage_error(
        inverse_transform_target(y_log_true),
        inverse_transform_target(y_log_pred),
    )


neg_mape_scorer = make_scorer(
    _mape_in_rupees,
    greater_is_better=False
)


def get_param_grid(model_name: str):
    """
    Returns the hyperparameter search space for the given model.
    Ranges are based on empirical tuning for real estate price prediction.
    """
    if model_name == "XGBoost":
        return {
            "regressor__learning_rate": sp_uniform(0.01, 0.05),
            "regressor__n_estimators": sp_randint(500, 1000),
            "regressor__max_depth": sp_randint(3, 6),
            "regressor__subsample": sp_uniform(0.6, 0.4),
            "regressor__colsample_bytree": sp_uniform(0.6, 0.4),
            "regressor__reg_alpha": [0.1, 0.5, 1, 5],
            "regressor__reg_lambda": [1, 5, 10],
            "regressor__min_child_weight": sp_randint(3, 8)
        }
    elif model_name == "LightGBM":
        return {
            "regressor__learning_rate": sp_uniform(0.01, 0.09),
            "regressor__n_estimators": sp_randint(400, 1000),
            "regressor__max_depth": sp_randint(4, 10),
            "regressor__num_leaves": sp_randint(20, 80),
            "regressor__subsample": sp_uniform(0.6, 0.4),
            "regressor__colsample_bytree": sp_uniform(0.6, 0.4),
            "regressor__reg_alpha": [0, 0.1, 0.5, 1, 5],
            "regressor__reg_lambda": [0, 1, 5, 10],
            "regressor__min_child_samples": sp_randint(5, 30)
        }
    elif model_name == "CatBoost":
        return {
            "regressor__learning_rate": sp_uniform(0.01, 0.09),
            "regressor__iterations": sp_randint(500, 1200),
            "regressor__depth": sp_randint(4, 8),
            "regressor__l2_leaf_reg": [1, 3, 5, 7, 10],
            "regressor__bagging_temperature": sp_uniform(0, 1),
            "regressor__random_strength": sp_uniform(0, 2),
        }
    return {}


def tune_model(
    model_name: str,
    model,
    X_train,
    y_train_log,
    X_val,
    y_val_log,
    numerical_features: list,
    categorical_features: list
):
    """
    Runs RandomizedSearchCV for the given model and evaluates the best
    estimator on train and VALIDATION sets (metrics on original price scale).

    Deliberately never sees the test set: the caller picks the winning model
    family from these validation scores, and only the winner is scored on test.

    Returns a dict:
        {
            "pipeline":    fitted best pipeline,
            "best_params": winning hyperparameters,
            "val_mape":    validation MAPE (%),
            "val_r2":      validation R2,
            "train_mape":  train MAPE (%),
        }
    or None if tuning fails.
    """
    try:
        logger.info(f"Tuning started for {model_name}.")

        # All candidate models are tree-based → same tree preprocessor.
        preprocessor = get_tree_preprocessor(
            numerical_features, categorical_features
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])

        param_grid = get_param_grid(model_name)
        # 3-fold keeps tuning tractable on ~23k train rows. Folds are stratified
        # on price quintiles: the target is skewed (~9.9), so plain KFold can
        # give folds with noticeably different luxury-segment shares, which makes
        # CV scores noisier than the differences being compared.
        price_bins = pd.qcut(inverse_transform_target(y_train_log), q=5, labels=False)
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=25,
            scoring=neg_mape_scorer,
            cv=list(kf.split(X_train, price_bins)),
            verbose=1,
            random_state=42,
            n_jobs=-1
        )

        random_search.fit(X_train, y_train_log)
        best_model = random_search.best_estimator_

        # Train metrics on original price scale
        y_train_pred = inverse_transform_target(best_model.predict(X_train))
        y_train_true = inverse_transform_target(y_train_log)
        train_r2 = r2_score(y_train_true, y_train_pred)
        train_mape = mean_absolute_percentage_error(
            y_train_true, y_train_pred
        ) * 100

        # Validation metrics on original price scale (used to pick the winner)
        y_pred = inverse_transform_target(best_model.predict(X_val))
        y_true = inverse_transform_target(y_val_log)
        val_r2 = r2_score(y_true, y_pred)
        val_mae = mean_absolute_error(y_true, y_pred)
        val_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        val_mape = mean_absolute_percentage_error(y_true, y_pred) * 100

        logger.info(
            f"{model_name} best CV MAPE: "
            f"{round(-random_search.best_score_ * 100, 2)}%"
        )
        logger.info(f"{model_name} best params: {random_search.best_params_}")
        logger.info(
            f"{model_name} Train — R2: {round(train_r2, 4)} | "
            f"MAPE: {round(train_mape, 2)}%"
        )
        logger.info(
            f"{model_name} Validation — R2: {round(val_r2, 4)} | "
            f"MAE: {round(val_mae, 4)} | "
            f"RMSE: {round(val_rmse, 4)} | "
            f"MAPE: {round(val_mape, 2)}%"
        )

        return {
            "pipeline": best_model,
            "best_params": random_search.best_params_,
            "val_mape": round(val_mape, 2),
            "val_r2": round(val_r2, 4),
            "train_mape": round(train_mape, 2),
        }

    except Exception as e:
        logger.error(f"Tuning failed for {model_name}: {e}", exc_info=True)
        return None


# ------------------------------------------------------------- orchestration ---

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

        # Persist the model FIRST. Training costs minutes; MLflow logging talks to
        # a remote server and can fail for reasons that have nothing to do with the
        # model (network, auth, console encoding). A logging failure must never
        # destroy a trained model, so the local artifact is written before any of it.
        #
        # Artifact contract is unchanged (dict with pipeline / model_name /
        # test_mape_percent / residual_quantiles) — the Streamlit pages depend on it.
        save_model(
            model_pipeline=best_pipeline,
            model_name=best_model_name,
            metric=round(best_test_mape, 2),
            val_mape_percent=best_val_mape,
            filepath="artifacts/best_model.joblib",
            residual_quantiles=residual_quantiles
        )

        # MLflow: one run per tuned model; the winner's model is logged and
        # registered. Wrapped so a tracking-server problem degrades to a warning
        # rather than failing the whole pipeline.
        try:
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
        except Exception as e:
            logger.warning(
                f"MLflow logging failed ({e}). The model is already saved locally; "
                f"continuing."
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
