import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_percentage_error

from src.logger_utils import setup_logger
from .mb_preprocessing import (
    transform_target,
    inverse_transform_target,
    get_tree_preprocessor,
    get_linear_preprocessor
)
from .mb_evaluation import scorer
from .mb_tuning import tune_model
from .mb_persistence import save_model


logger = setup_logger("logs/mb_main.log")

TARGET_COL = "price_in_cr"


def run_model_building(fs_df):

    try:
        logger.info("Model Building Pipeline Started.")

        #  Data Preparation

        X = fs_df.drop(columns=[TARGET_COL])
        y = fs_df[TARGET_COL]
        y_log = transform_target(y)

        price_bins = pd.qcut(y, q=5, labels=False)

        X_train, X_test, y_train_log, y_test_log = train_test_split(
            X,
            y_log,
            stratify=price_bins,
            test_size=0.2,
            random_state=42
        )

        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        #  Base Model Experiments

        tree_preprocessor = get_tree_preprocessor()
        linear_preprocessor = get_linear_preprocessor()

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
            "SVR": (
                SVR(),
                linear_preprocessor
            )
        }

        results = []

        for name, (model, preprocessor) in model_dict.items():

            logger.info(f"Evaluating {name}")

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

        print("\n===== BASE MODEL RESULTS =====")
        print(results_df)
        print("================================\n")

        logger.info("Base model evaluation completed.")

        #  Hyperparameter Tuning

        tuned_models = {}

        logger.info("Starting tuning for RandomForest")
        tuned_models["RandomForest"] = tune_model(
            "RandomForest",
            RandomForestRegressor(random_state=42),
            X_train,
            y_train_log,
            X_test,
            y_test_log
        )

        logger.info("Starting tuning for XGBoost")
        tuned_models["XGBoost"] = tune_model(
            "XGBoost",
            XGBRegressor(
                random_state=42,
                objective="reg:squarederror",
                tree_method="hist"
            ),
            X_train,
            y_train_log,
            X_test,
            y_test_log
        )

        logger.info("Starting tuning for SVR")
        tuned_models["SVR"] = tune_model(
            "SVR",
            SVR(),
            X_train,
            y_train_log,
            X_test,
            y_test_log
        )

        logger.info("Tuning completed successfully.")

        # Select Best Tuned Model

        best_model_name = None
        best_test_mape = float("inf")
        best_pipeline = None

        for name, pipeline in tuned_models.items():

            test_pred_log = pipeline.predict(X_test)
            y_pred = inverse_transform_target(test_pred_log)
            y_true = inverse_transform_target(y_test_log)

            test_mape = mean_absolute_percentage_error(y_true, y_pred)

            if test_mape < best_test_mape:
                best_test_mape = test_mape
                best_model_name = name
                best_pipeline = pipeline

        logger.info(f"Best Tuned Model: {best_model_name}")
        logger.info(f"Best Test MAPE: {round(best_test_mape * 100, 2)}%")

        # 5. Save Best Model

        save_model(
            model_pipeline=best_pipeline,
            model_name=best_model_name,
            metric=round(best_test_mape * 100, 2),
            filepath="artifacts/best_model.joblib"
        )

        logger.info("Best model saved successfully.")
        logger.info("Model Building Pipeline Completed Successfully.")

        return {
            "base_results": results_df,
            "best_model_name": best_model_name,
            "best_test_mape": best_test_mape
        }

    except Exception as e:
        logger.error(f"Model Building failed: {str(e)}", exc_info=True)
        raise