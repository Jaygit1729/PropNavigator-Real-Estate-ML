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
from src.logger_utils import setup_logger
from .mb_preprocessing import (
    get_tree_preprocessor,
    inverse_transform_target
)


logger = setup_logger(__name__, "logs/mb_tuning.log")

neg_mape_scorer = make_scorer(
    mean_absolute_percentage_error,
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
    X_test,
    y_test_log,
    numerical_features: list,
    categorical_features: list
):
    """
    Runs RandomizedSearchCV for the given model and evaluates the best
    estimator on train and test sets (metrics on original price scale).

    Returns a dict:
        {
            "pipeline":    fitted best pipeline,
            "best_params": winning hyperparameters,
            "test_mape":   test MAPE (%),
            "test_r2":     test R2,
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
        # 3-fold keeps tuning tractable on ~31k rows.
        kf = KFold(n_splits=3, shuffle=True, random_state=42)

        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=25,
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
            f"MAE: {round(test_mae, 4)} | "
            f"RMSE: {round(test_rmse, 4)} | "
            f"MAPE: {round(test_mape, 2)}%"
        )

        return {
            "pipeline": best_model,
            "best_params": random_search.best_params_,
            "test_mape": round(test_mape, 2),
            "test_r2": round(test_r2, 4),
            "train_mape": round(train_mape, 2),
        }

    except Exception as e:
        logger.error(f"Tuning failed for {model_name}: {e}", exc_info=True)
        return None
