# src/model_building/mb_tuning.py

import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    make_scorer
)
from scipy.stats import randint as sp_randint, uniform as sp_uniform
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from src.logger_utils import setup_logger
from .mb_preprocessing import (
    get_tree_preprocessor,
    get_linear_preprocessor,
    inverse_transform_target
)


logger = setup_logger(__name__, "logs/mb_tuning.log")

neg_mape_scorer = make_scorer(
    mean_absolute_percentage_error,
    greater_is_better=False
)


def get_param_grid(model_name: str):
    """
    Returns hyperparameter search space for the given model.
    Ranges are based on empirical tuning experience for
    real estate price prediction tasks.
    """
    if model_name == "RandomForest":
        return {
            "regressor__n_estimators": sp_randint(400, 900),
            "regressor__max_depth": sp_randint(10, 25),
            "regressor__min_samples_split": sp_randint(5, 25),
            "regressor__min_samples_leaf": sp_randint(2, 15),
            "regressor__max_features": ["sqrt", 1.0]
        }
    elif model_name == "XGBoost":
        return {
                "regressor__learning_rate": sp_uniform(0.01, 0.05),
                "regressor__n_estimators": sp_randint(500, 1000),     # ← middle ground
                "regressor__max_depth": sp_randint(3, 6),
                "regressor__subsample": sp_uniform(0.6, 0.4),
                "regressor__colsample_bytree": sp_uniform(0.6, 0.4),
                "regressor__reg_alpha": [0.1, 0.5, 1, 5],            # ← moderate
                "regressor__reg_lambda": [1, 5, 10],                  # ← moderate
                "regressor__min_child_weight": sp_randint(3, 8)       # ← keep this
            }
        
            
    elif model_name == "SVR":
        return {
            "regressor__C": sp_uniform(1, 100),
            "regressor__epsilon": sp_uniform(0.01, 0.5),
            "regressor__gamma": ["scale", "auto"]
        }
    return {}


def tune_model(
    model_name: str,
    model,
    X_train,
    y_train_log,
    X_test,
    y_test_log,
    numerical_features: list,
    categorical_features: list
):
    """
    Runs RandomizedSearchCV for the given model, evaluates the best
    estimator on train and test sets, and returns the fitted pipeline.

    """
    try:
        logger.info(f"Tuning started for {model_name}.")

        # Preprocessor is built from actual column lists derived at runtime
        # not from hardcoded lists — so top_n changes in feature selection
        # automatically flow through without breaking this step
        preprocessor = (
            get_tree_preprocessor(numerical_features, categorical_features)
            if model_name in ["RandomForest", "XGBoost"]
            else get_linear_preprocessor(numerical_features, categorical_features)
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])

        param_grid = get_param_grid(model_name)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        random_search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=param_grid,
                    n_iter=50,              # ← increase from 25 to 50
                    scoring=neg_mape_scorer,
                    cv=kf,
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

        # Test metrics on original price scale
        y_pred = inverse_transform_target(best_model.predict(X_test))
        y_true = inverse_transform_target(y_test_log)
        test_r2 = r2_score(y_true, y_pred)
        test_mae = mean_absolute_error(y_true, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        test_mape = mean_absolute_percentage_error(y_true, y_pred) * 100

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
            f"{model_name} Test — R2: {round(test_r2, 4)} | "
            f"MAPE: {round(test_mape, 2)}%"
        )

        return best_model

    except Exception as e:
        logger.error(f"Tuning failed for {model_name}: {e}", exc_info=True)
        return None