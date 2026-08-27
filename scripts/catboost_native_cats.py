"""Does CatBoost do better with its own categorical handling?

The shared preprocessor ordinal-encodes categories before any model sees them.
That is even-handed, but it disables the feature CatBoost exists for: ordered
target statistics, which replace a category with a running target average
computed only from rows that came earlier in a random permutation -- target
encoding that cannot leak.

This runs the same search budget, folds and scorer as the pipeline, changing
only how categories reach CatBoost. Anything else would confound the comparison.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import randint as sp_randint, uniform as sp_uniform
from sklearn.metrics import make_scorer, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.model_building.model_building import create_train_val_test_split

SEED = 42


def mape_rupees(y_log_true, y_log_pred):
    return mean_absolute_percentage_error(np.expm1(y_log_true), np.expm1(y_log_pred))


scorer = make_scorer(mape_rupees, greater_is_better=False)


def main():
    fs = pd.read_csv("data/fs/feature_selected_properties.csv")
    X_train, X_val, _, y_train, y_val, _ = create_train_val_test_split(fs)

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    print("categorical features handled natively:", cat_cols)
    print("cardinalities:", {c: X_train[c].nunique() for c in cat_cols})

    # CatBoost wants categoricals as strings and no NaN in those columns.
    for df in (X_train, X_val):
        for c in cat_cols:
            df[c] = df[c].astype(str)

    grid = {
        "learning_rate": sp_uniform(0.01, 0.09),
        "iterations": sp_randint(500, 1200),
        "depth": sp_randint(4, 8),
        "l2_leaf_reg": [1, 3, 5, 7, 10],
        "bagging_temperature": sp_uniform(0, 1),
        "random_strength": sp_uniform(0, 2),
    }

    # cat_features must be passed at fit time, not in the constructor: sklearn's
    # clone() rejects estimators whose constructor modifies a parameter, and
    # CatBoost normalises cat_features internally.
    model = CatBoostRegressor(
        random_seed=SEED, verbose=0, allow_writing_files=False,
    )

    bins = pd.qcut(np.expm1(y_train), q=5, labels=False)
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    search = RandomizedSearchCV(
        model, grid, n_iter=25, scoring=scorer,
        cv=list(kf.split(X_train, bins)), random_state=SEED, n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train, cat_features=cat_cols)

    best = search.best_estimator_
    pred = np.expm1(best.predict(X_val))
    true = np.expm1(y_val)
    val_mape = mean_absolute_percentage_error(true, pred) * 100
    val_r2 = r2_score(true, pred)

    print(f"\nbest CV MAPE : {-search.best_score_*100:.2f}%")
    print(f"best params  : {search.best_params_}")
    print(f"\nNATIVE CATEGORICALS  -> val MAPE {val_mape:.2f}%  R2 {val_r2:.4f}")
    print(f"ORDINAL (pipeline)   -> val MAPE 12.18%  R2 0.9315")
    print(f"LightGBM (incumbent) -> val MAPE 10.86%  R2 0.9275")
    print(f"\ndelta vs ordinal CatBoost: {val_mape - 12.18:+.2f} points")


if __name__ == "__main__":
    main()
