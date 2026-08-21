"""
Re-derive the feature ranking that justifies SELECTED_FEATURES.

Three corrections over the original notebook:

1. The target is log-transformed ONCE. The notebook applied log1p when
   separating X and y, then applied it again after the split, so the models
   were ranked against log(log(price)).

2. Ranking uses exactly the rows the model is allowed to see. The notebook
   took its own 80/20 unstratified split of the preprocessed table, which is
   a different partition from the model's stratified 60/20/20 on the
   de-duplicated table -- so final test rows helped choose the features.
   Here the production selection + de-duplication + split are reproduced
   positionally, and ranking runs on the train+val rows only.

3. Permutation importance on held-out rows replaces RFE. Impurity importance
   is biased toward high-cardinality features; permutation measures what
   actually happens to error when a feature is scrambled.

4. LightGBM replaces sklearn's GradientBoostingRegressor as the second signal.
   It is the deployed model family, so its gain importances describe the model
   actually being shipped -- and it handles the 184 rows with unresolved
   coordinates natively, which sklearn's booster cannot.

Outputs land in data/fs/ and replace artifacts that described an older
pipeline (they still ranked luxury_score, pooja_room, study_room ...).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.feature_selection.feature_selection import SELECTED_FEATURES

TARGET = "price_in_cr"
PP_PATH = "data/pp/preprocessed_properties.csv"
SEED = 42


def reproduce_model_rows(pp):
    """Return the pp-index of rows that land in the model's train and val splits.

    Mirrors select_features() then split_data() exactly. The index is kept
    (production resets it) so selected rows can be mapped back to pp and read
    with all 34 candidate features rather than only the 24 chosen ones.
    """
    keep = [c for c in SELECTED_FEATURES if c in pp.columns]
    fs = pp[keep + [TARGET]].drop_duplicates(subset=keep)

    y = fs[TARGET]
    bins = pd.qcut(y, q=5, labels=False)

    idx_temp, _, bins_temp, _ = train_test_split(
        fs.index, bins, stratify=bins, test_size=0.2, random_state=SEED
    )
    # train (60%) + val (20%) = everything except test; both are fair game
    # for feature ranking, test is not.
    return idx_temp, len(fs)


def compute_ranking(pp=None):
    """Return the feature ranking table. Importable so the notebook shows
    exactly what the script saves, instead of keeping a second copy of this
    logic that can drift (and did)."""
    if pp is None:
        pp = pd.read_csv(PP_PATH)
    candidates = [c for c in pp.columns if c != TARGET]
    print(f"preprocessed: {pp.shape}  candidates: {len(candidates)}")

    fit_idx, n_fs = reproduce_model_rows(pp)
    print(f"feature-selected rows: {n_fs}  ranking on train+val: {len(fit_idx)}")

    X = pp.loc[fit_idx, candidates].copy()
    y = np.log1p(pp.loc[fit_idx, TARGET])          # ONCE

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X[cat_cols] = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ).fit_transform(X[cat_cols])

    # inner split so permutation importance is measured on rows the ranking
    # models did not fit on
    X_fit, X_hold, y_fit, y_hold = train_test_split(
        X, y, test_size=0.25, random_state=SEED
    )

    print("fitting random forest ...")
    rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf.fit(X_fit, y_fit)

    print("fitting lightgbm ...")
    gb = LGBMRegressor(
        n_estimators=200, random_state=SEED, n_jobs=-1,
        importance_type="gain", verbose=-1,
    )
    gb.fit(X_fit, y_fit)

    print("permutation importance ...")
    perm = permutation_importance(
        rf, X_hold, y_hold, n_repeats=5, random_state=SEED, n_jobs=-1
    )

    scores = pd.DataFrame(
        {
            "rf": rf.feature_importances_,
            "lgbm_gain": gb.feature_importances_,
            "perm": perm.importances_mean,
        },
        index=X.columns,
    )

    norm = (scores - scores.min()) / (scores.max() - scores.min())
    scores["final_score"] = norm.mean(axis=1)
    scores = scores.sort_values("final_score", ascending=False)
    scores.index.name = "feature"

    scores["rank"] = range(1, len(scores) + 1)
    scores["in_selected"] = scores.index.isin(SELECTED_FEATURES)

    return scores


def main():
    scores = compute_ranking()
    scores.to_csv("data/fs/feature_ranking.csv")
    print(scores.round(4).to_string())

    chosen = set(SELECTED_FEATURES)
    top_n = set(scores.head(len(SELECTED_FEATURES)).index)
    print(f"\ntop-{len(SELECTED_FEATURES)} vs hardcoded list")
    print("  in top-N but NOT selected:", sorted(top_n - chosen))
    print("  selected but NOT in top-N:", sorted(chosen - top_n))


if __name__ == "__main__":
    main()
