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

from .mb_preprocessing import (
    get_tree_preprocessor,
    get_linear_preprocessor,
    inverse_transform_target
)


# Custom MAPE scorer

neg_mape_scorer = make_scorer(
    mean_absolute_percentage_error,
    greater_is_better=False
)



# Parameter Grids


def get_param_grid(model_name):

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
            "regressor__n_estimators": sp_randint(800, 1600),
            "regressor__max_depth": sp_randint(3, 6),
            "regressor__subsample": sp_uniform(0.7, 0.3),
            "regressor__colsample_bytree": sp_uniform(0.7, 0.3),
            "regressor__reg_alpha": [0, 0.1, 1, 5],
            "regressor__reg_lambda": [0.5, 1, 5]
        }

    elif model_name == "SVR":
        return {
            "regressor__C": sp_uniform(1, 100),
            "regressor__epsilon": sp_uniform(0.01, 0.5),
            "regressor__gamma": ["scale", "auto"]
        }

    return {}
    

# Tuning Function


def tune_model(model_name, model, X_train, y_train_log, X_test, y_test_log):

    print(f"\n===== TUNING {model_name} =====")

    # Choose correct preprocessor
    if model_name in ["RandomForest", "XGBoost"]:
        preprocessor = get_tree_preprocessor()
    else:
        preprocessor = get_linear_preprocessor()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    param_grid = get_param_grid(model_name)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

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

    # TRAIN METRICS

    train_pred_log = best_model.predict(X_train)
    y_train_pred = inverse_transform_target(train_pred_log)
    y_train_true = inverse_transform_target(y_train_log)

    train_r2 = r2_score(y_train_true, y_train_pred)
    train_mae = mean_absolute_error(y_train_true, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
    train_mape = mean_absolute_percentage_error(y_train_true, y_train_pred)

    # TEST METRICS

    test_pred_log = best_model.predict(X_test)
    y_pred = inverse_transform_target(test_pred_log)
    y_true = inverse_transform_target(y_test_log)

    test_r2 = r2_score(y_true, y_pred)
    test_mae = mean_absolute_error(y_true, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    test_mape = mean_absolute_percentage_error(y_true, y_pred)

    print("\nBest CV MAPE:", round(-random_search.best_score_ * 100, 2), "%")
    print("Best Parameters:", random_search.best_params_)

    print("\n--- TRAIN ---")
    print("R2:", round(train_r2, 4))
    print("MAPE:", round(train_mape * 100, 2), "%")

    print("\n--- TEST ---")
    print("R2:", round(test_r2, 4))
    print("MAPE:", round(test_mape * 100, 2), "%")
    print("====================================\n")

    return best_model