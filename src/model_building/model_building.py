# src/model_building/model_building.py


# 1. IMPORTS

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


# 2. CONFIGURATION

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

logger = setup_logger(
    __name__,
    "logs/model_building.log",
)

TARGET_COL = "price_in_cr"
EXPERIMENT_NAME = "propnavigator-model-building"
REGISTERED_MODEL_NAME = "propnavigator-price-model"
MODEL_PATH = "artifacts/best_model.joblib"
RANDOM_STATE = 42
N_ITER = 25
N_SPLITS = 3


# 3. DATA PREPARATION

def transform_target(y):
    """
    Convert price to log scale for model training.
    """
    return np.log1p(y)


def inverse_transform_target(y_log):
    """
    Convert log-scale values back to original price scale.
    """
    return np.expm1(y_log)


def create_train_val_test_split(df):
    """
    Create a 60/20/20 train/validation/test split.
    """

    # Separate features and target

    X = df.drop(columns=[TARGET_COL])

    y = df[TARGET_COL]

    # Transform target for model training

    y_log = transform_target(y)

    # Create price groups for stratification

    price_bins = pd.qcut(
        y,
        q=5,
        labels=False,
    )

    # Step 1:
    # Reserve 20% as untouched test data. When creating the 80/20 split, preserve the distribution of these five price groups as much as possible.

    (
        X_temp,
        X_test,
        y_temp_log,
        y_test_log,
        bins_temp,
        _,
    ) = train_test_split(
        X,
        y_log,
        price_bins,
        test_size=0.20,
        stratify=price_bins,
        random_state=RANDOM_STATE,
    )

    # Step 2:
    # Split remaining 80% into:

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


# 4. FEATURE PREPROCESSING


def get_feature_lists(X):
    """
    Identify numerical and categorical columns.
    """

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    return numerical_features, categorical_features


def create_preprocessor(
    numerical_features,
    categorical_features,
):
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


# 5. MODEL DEFINITIONS


def create_models():
    """
    Create the candidate regression models.
    """

    return {
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


# 5A. HYPERPARAMETER SEARCH SPACES


PARAMETER_SPACES = {

    "XGBoost": {

        "regressor__learning_rate":
            sp_uniform(0.01, 0.05),

        "regressor__n_estimators":
            sp_randint(500, 1000),

        "regressor__max_depth":
            sp_randint(3, 6),

        "regressor__subsample":
            sp_uniform(0.6, 0.4),

        "regressor__colsample_bytree":
            sp_uniform(0.6, 0.4),

        "regressor__reg_alpha":
            [0.1, 0.5, 1, 5],

        "regressor__reg_lambda":
            [1, 5, 10],

        "regressor__min_child_weight":
            sp_randint(3, 8),
    },


    "LightGBM": {

        "regressor__learning_rate":
            sp_uniform(0.01, 0.09),

        "regressor__n_estimators":
            sp_randint(400, 3000),

        "regressor__max_depth":
            sp_randint(4, 10),

        "regressor__num_leaves":
            sp_randint(20, 80),

        "regressor__subsample":
            sp_uniform(0.6, 0.4),

        "regressor__subsample_freq":
            [0, 1, 5],

        "regressor__colsample_bytree":
            sp_uniform(0.6, 0.4),

        "regressor__reg_alpha":
            [0, 0.1, 0.5, 1, 5],

        "regressor__reg_lambda":
            [0, 1, 5, 10],

        "regressor__min_child_samples":
            sp_randint(5, 30),
    },


    "CatBoost": {

        "regressor__learning_rate":
            sp_uniform(0.01, 0.09),

        "regressor__iterations":
            sp_randint(500, 1200),

        "regressor__depth":
            sp_randint(4, 8),

        "regressor__l2_leaf_reg":
            [1, 3, 5, 7, 10],

        "regressor__bagging_temperature":
            sp_uniform(0, 1),

        "regressor__random_strength":
            sp_uniform(0, 2),
    },
}


# 6. EVALUATION

# The model is trained on log(price),but business metrics are calculated on actual price.


def mape_on_price_scale(
    y_log_true,
    y_log_pred,
):
    """
    Calculate MAPE after converting values back
    to the original price scale.
    """

    y_true = inverse_transform_target(
        y_log_true
    )

    y_pred = inverse_transform_target(
        y_log_pred
    )

    return mean_absolute_percentage_error(
        y_true,
        y_pred,
    )


neg_mape_scorer = make_scorer(
    mape_on_price_scale,
    greater_is_better=False,
)


def calculate_metrics(
    model,
    X,
    y_log,
):
    """
    Calculate regression metrics on the
    original price scale.
    """

    y_true = inverse_transform_target(
        y_log
    )

    y_pred = inverse_transform_target(
        model.predict(X)
    )

    return {
        "r2": round(
            r2_score(y_true, y_pred),
            4,
        ),

        "mae": round(
            mean_absolute_error(y_true, y_pred),
            4,
        ),

        "rmse": round(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            ),
            4,
        ),

        "mape": round(
            mean_absolute_percentage_error(
                y_true,
                y_pred,
            ) * 100,
            2,
        ),
    }


# 7. CROSS-VALIDATION + HYPERPARAMETER TUNING


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
    """
    Tune one model using randomized hyperparameter
    search with stratified cross-validation.

    The best tuned model is then evaluated
    on the separate validation set.
    """

    logger.info(
        f"Tuning started for {model_name}."
    )

    # 1. Create preprocessing

    preprocessor = create_preprocessor(
        numerical_features,
        categorical_features,
    )

    # 2. Combine preprocessing + model


    pipeline = Pipeline(
        [
            (
                "preprocessor",
                preprocessor,
            ),

            (
                "regressor",
                model,
            ),
        ]
    )

    # 3. Create price bins for CV stratification

    price_bins = pd.qcut(
        inverse_transform_target(y_train_log),
        q=5,
        labels=False,
    )

    # 4. Define cross-validation strategy

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # 5. Materialize the same folds

    cv_splits = list(
        cv.split(
            X_train,
            price_bins,
        )
    )

    # 6. Create randomized hyperparameter search

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

    # 7. Run the actual search

    search.fit(
        X_train,
        y_train_log,
    )

    # 8. Extract the best pipeline

    best_model = search.best_estimator_

    # 9. Evaluate best configuration on training data

    train_metrics = calculate_metrics(
        best_model,
        X_train,
        y_train_log,
    )

    # 10. Evaluate best configuration on validation data

    val_metrics = calculate_metrics(
        best_model,
        X_val,
        y_val_log,
    )

    # 11. Convert negative CV MAPE back to positive %

    cv_mape = round(
        -search.best_score_ * 100,
        2,
    )

    # 12. Logging

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

    # 13. Return everything needed later

    return {
        "pipeline": best_model,

        "best_params":
            search.best_params_,

        "cv_mape":
            cv_mape,

        "train_mape":
            train_metrics["mape"],

        "val_mape":
            val_metrics["mape"],

        "val_r2":
            val_metrics["r2"],

        "val_mae":
            val_metrics["mae"],

        "val_rmse":
            val_metrics["rmse"],
    }



# 8. MODEL SELECTION


def select_best_model(results):
    """
    Select the model with the lowest validation MAPE.
    """

    best_model_name = min(
        results,
        key=lambda name:
            results[name]["val_mape"],
    )

    return (
        best_model_name,
        results[best_model_name],
    )


# 9. PREDICTION INTERVAL CALIBRATION


def calculate_residual_quantiles(
    model,
    X_val,
    y_val_log,
):
    """
    Estimate prediction-error quantiles
    from validation data.
    """

    y_true = inverse_transform_target(
        y_val_log
    )

    y_pred = inverse_transform_target(
        model.predict(X_val)
    )

    # Percentage error relative to predicted price.
    pct_errors = (
        (y_true - y_pred)
        / y_pred
    )

    return {
        "q05": float(
            np.percentile(
                pct_errors,
                5,
            )
        ),

        "q95": float(
            np.percentile(
                pct_errors,
                95,
            )
        ),

        "q10": float(
            np.percentile(
                pct_errors,
                10,
            )
        ),

        "q90": float(
            np.percentile(
                pct_errors,
                90,
            )
        ),
    }


# 10. MODEL PERSISTENCE
#
# Save:
#
#   trained pipeline
#   model name
#   validation MAPE
#   test MAPE
#   residual quantiles
#   training timestamp
#
# The saved pipeline contains:
#
#   preprocessing + model
#


def save_model(
    pipeline,
    model_name,
    val_mape,
    test_mape,
    residual_quantiles,
    filepath=MODEL_PATH,
):
    """
    Save the winning model and metadata.
    """

    os.makedirs(
        os.path.dirname(filepath),
        exist_ok=True,
    )

    artifact = {
        "model_name":
            model_name,

        "val_mape_percent":
            val_mape,

        "test_mape_percent":
            test_mape,

        "pipeline":
            pipeline,

        "residual_quantiles":
            residual_quantiles,

        "trained_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),
    }

    # --------------------------------------------------------
    # Save timestamped copy
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    joblib.dump(
        artifact,
        filepath.replace(
            ".joblib",
            f"_{timestamp}.joblib",
        ),
    )

    # Save current/best model

    joblib.dump(
        artifact,
        filepath,
    )

    logger.info(
        f"Saved {model_name} to {filepath} | "
        f"Val MAPE: {val_mape}% | "
        f"Test MAPE: {test_mape}%"
    )


# 11. MLFLOW EXPERIMENT TRACKING
#
# Log:
#
#   model type
#   hyperparameters
#   CV metrics
#   training metrics
#   validation metrics
#   test metrics for winner
#


def log_to_mlflow(
    results,
    best_model_name,
    best_test_mape,
    best_test_r2,
    n_features,
):
    """
    Log model comparison results and
    register the winning model.
    """

    try:

        mlflow.set_experiment(
            EXPERIMENT_NAME
        )

        # Create one MLflow run per candidate model

        for model_name, result in results.items():

            with mlflow.start_run(
                run_name=model_name
            ):

                # General experiment parameters

                mlflow.log_params(
                    {
                        "model_type":
                            model_name,

                        "split":
                            "60/20/20 train/val/test",

                        "selected_on":
                            "validation",

                        "random_state":
                            RANDOM_STATE,

                        "n_features":
                            n_features,
                    }
                )

                # Best hyperparameters

                mlflow.log_params(
                    result["best_params"]
                )

                # Model metrics

                mlflow.log_metrics(
                    {
                        "cv_mape":
                            result["cv_mape"],

                        "train_mape":
                            result["train_mape"],

                        "val_mape":
                            result["val_mape"],

                        "val_r2":
                            result["val_r2"],

                        "val_mae":
                            result["val_mae"],

                        "val_rmse":
                            result["val_rmse"],
                    }
                )

                # Only the selected winner receives the untouched test score.

                if model_name == best_model_name:

                    mlflow.log_param(
                        "is_best",
                        True,
                    )

                    mlflow.log_metrics(
                        {
                            "test_mape":
                                best_test_mape,

                            "test_r2":
                                best_test_r2,
                        }
                    )

                    # Register winning model

                    mlflow.sklearn.log_model(
                        result["pipeline"],
                        name="model",
                        serialization_format="cloudpickle",
                        registered_model_name=
                            REGISTERED_MODEL_NAME,
                    )

        logger.info(
            "MLflow logging complete."
        )

    except Exception as e:

        logger.warning(
            f"MLflow logging failed: {e}. "
            "Model is already saved locally."
        )


# 12. MAIN MODEL-BUILDING ORCHESTRATION



def run_model_building(
    fs_df: pd.DataFrame,
):
    """
    Complete model-building workflow.
    """

    try:

        logger.info(
            "Model building pipeline started."
        )

        logger.info(
            f"Input shape: {fs_df.shape}"
        )


        # STEP 1 — SPLIT DATA

        (
            X_train,
            X_val,
            X_test,
            y_train_log,
            y_val_log,
            y_test_log,
        ) = create_train_val_test_split(
            fs_df
        )

        logger.info(
            f"Train: {X_train.shape} | "
            f"Validation: {X_val.shape} | "
            f"Test: {X_test.shape}"
        )


        # STEP 2 — IDENTIFY FEATURE TYPES

        (
            numerical_features,
            categorical_features,
        ) = get_feature_lists(
            X_train
        )

        logger.info(
            f"Numerical features: "
            f"{numerical_features}"
        )

        logger.info(
            f"Categorical features: "
            f"{categorical_features}"
        )


        # STEP 3 — CREATE CANDIDATE MODELS

        models = create_models()


        # STEP 4 — TUNE EVERY MODEL

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

                    numerical_features=
                        numerical_features,

                    categorical_features=
                        categorical_features,
                )

            except Exception as e:

                logger.error(
                    f"{model_name} tuning failed: {e}",
                    exc_info=True,
                )


        # STEP 5 — CHECK WHETHER ANY MODEL SUCCEEDED

        if not results:

            logger.error(
                "All models failed tuning. "
                "No model saved."
            )

            return {}


        # STEP 6 — SELECT BEST MODEL

        (
            best_model_name,
            best_result,
        ) = select_best_model(
            results
        )

        best_pipeline = (
            best_result["pipeline"]
        )

        best_val_mape = (
            best_result["val_mape"]
        )

        logger.info(
            f"Best model: "
            f"{best_model_name} | "
            f"Validation MAPE: "
            f"{best_val_mape}%"
        )


        
        # STEP 7 — FINAL TEST EVALUATION
    

        test_metrics = calculate_metrics(
            best_pipeline,
            X_test,
            y_test_log,
        )

        best_test_mape = (
            test_metrics["mape"]
        )

        best_test_r2 = (
            test_metrics["r2"]
        )

        logger.info(
            f"HELD-OUT TEST | "
            f"{best_model_name} | "
            f"MAPE: {best_test_mape}% | "
            f"R2: {best_test_r2}"
        )


        # STEP 8 — CALIBRATE PREDICTION INTERVALS

        residual_quantiles = (
            calculate_residual_quantiles(
                best_pipeline,
                X_val,
                y_val_log,
            )
        )

        logger.info(
            f"Residual quantiles: "
            f"{residual_quantiles}"
        )


        # STEP 9 — SAVE WINNING MODEL

        save_model(
            pipeline=best_pipeline,

            model_name=best_model_name,

            val_mape=best_val_mape,

            test_mape=best_test_mape,

            residual_quantiles=
                residual_quantiles,
        )


        # STEP 10 — LOG TO MLFLOW
    

        log_to_mlflow(
            results=results,

            best_model_name=
                best_model_name,

            best_test_mape=
                best_test_mape,

            best_test_r2=
                best_test_r2,

            n_features=
                X_train.shape[1],
        )


        # STEP 11 — RETURN SUMMARY

        logger.info(
            "Model building pipeline "
            "completed successfully."
        )

        return {

            "best_model_name":
                best_model_name,

            "best_val_mape":
                best_val_mape,

            "best_test_mape":
                best_test_mape,

            "best_test_r2":
                best_test_r2,

            "all_val_results": {
                name:
                    result["val_mape"]
                for name, result
                in results.items()
            },
        }


    except Exception as e:

        logger.error(
            f"Model building pipeline failed: {e}",
            exc_info=True,
        )

        raise