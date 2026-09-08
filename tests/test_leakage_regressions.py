"""Regression tests for the data-leakage bugs found in the 2026-07-19 audit.

Each test here corresponds to a real bug that was in the pipeline and got fixed.
They exist so nobody silently reintroduces one — a leak is invisible at runtime
(everything still "works", the metric just quietly lies).
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_selection.feature_selection import SELECTED_FEATURES, select_features
from src.model_building.model_building import create_train_val_test_split


# --- Bug 1: model family was selected on the test set -------------------------

def test_split_is_three_way_60_20_20(fs_df):
    """A validation set must exist, so family selection never touches test."""
    X_tr, X_val, X_te, y_tr, y_val, y_te = create_train_val_test_split(fs_df)
    total = len(X_tr) + len(X_val) + len(X_te)

    assert total == len(fs_df), "split lost or duplicated rows"
    assert 0.58 < len(X_tr) / total < 0.62, f"train share {len(X_tr)/total:.3f}, expected ~0.60"
    assert 0.18 < len(X_val) / total < 0.22, f"val share {len(X_val)/total:.3f}, expected ~0.20"
    assert 0.18 < len(X_te) / total < 0.22, f"test share {len(X_te)/total:.3f}, expected ~0.20"


def test_splits_do_not_overlap_by_index(fs_df):
    """The three splits must be disjoint — no row may appear in two of them."""
    X_tr, X_val, X_te, *_ = create_train_val_test_split(fs_df)
    tr, val, te = set(X_tr.index), set(X_val.index), set(X_te.index)

    assert not (tr & val), "train and validation share rows"
    assert not (tr & te), "train and test share rows"
    assert not (val & te), "validation and test share rows"


# --- Bug 2: duplicate properties straddled train and test ---------------------

def test_feature_selection_drops_duplicate_rows(raw_pp_df):
    """Narrowing 35 columns -> 24 creates new duplicates; they must be dropped.

    Before the fix, 252 test rows (3.23%) had an identical feature-twin in train,
    which the model could memorise.
    """
    fs = select_features(raw_pp_df)
    feats = [c for c in fs.columns if c != "price_in_cr"]
    assert fs.duplicated(subset=feats).sum() == 0, "duplicate feature rows survived selection"


def test_no_identical_feature_rows_across_train_and_test(fs_df):
    """The memorisation leak itself: zero test rows may have a train twin."""
    X_tr, X_val, X_te, *_ = create_train_val_test_split(fs_df)
    feats = list(X_tr.columns)

    train_keys = set(map(tuple, X_tr[feats].astype(str).to_numpy()))
    overlap = sum(1 for k in map(tuple, X_te[feats].astype(str).to_numpy()) if k in train_keys)

    assert overlap == 0, f"{overlap} test rows have an identical feature-row in train"


# --- Bug 3: society was unobtainable at serving time --------------------------

def test_society_is_not_a_model_feature():
    """society can't be obtained at inference (guessed correctly only ~31% of the
    time), so it must stay out of the feature set."""
    assert "society" not in SELECTED_FEATURES


# --- Bug 4: imputation was fitted on the full dataset -------------------------

def test_distance_nulls_are_not_imputed(raw_pp_df):
    """Distances must keep their NaNs.

    They used to be filled with the median of the WHOLE dataset — test statistics
    leaking into training inputs. Tree models handle NaN natively, so we leave it.
    """
    dist_cols = ["dist_to_cyber_city", "dist_to_golf_road",
                 "dist_to_airport", "dist_to_manesar"]
    total_nulls = sum(raw_pp_df[c].isna().sum() for c in dist_cols)
    assert total_nulls > 0, (
        "distance NaNs have disappeared — has median imputation been reintroduced? "
        "That leaks full-dataset statistics into training."
    )


def test_undefined_age_category_is_preserved(raw_pp_df):
    """'Undefined' must survive as its own category.

    It used to be replaced by a sector/type mode computed over the whole dataset.
    Keeping it matches how `facing` and `furnishing` already handle unknowns.
    """
    assert (raw_pp_df["age_possession_category"] == "Undefined").sum() > 0, (
        "'Undefined' age_possession_category is gone — has mode imputation returned?"
    )


# --- Bug 4: the save gate compared TEST scores across runs --------------------

def test_save_gate_requires_a_validation_score():
    """Persistence must refuse to gate on anything but validation.

    Gating on test MAPE means that across repeated runs the artifact keeps
    whichever model happened to score best on test — the reported number becomes
    a maximum over runs rather than a held-out estimate. Each individual run
    still looks correct, which is what makes it easy to miss.
    """
    from src.model_building.persistence import save_model

    with pytest.raises(ValueError, match="val_mape_percent"):
        save_model(
            model_pipeline=object(),
            model_name="dummy",
            metric=10.0,
            filepath="artifacts/_gate_probe.joblib",
        )


# --- Bug 5: the tuning objective was not the reported metric ------------------

def test_cv_scorer_measures_error_in_rupees_not_log_space():
    """The search fits on log1p(price); a naive MAPE scorer would score logs.

    Relative error between log values is a different objective — log1p
    compresses the range, so cheap listings get tiny denominators and dominate.
    The scorer must invert the transform so tuning optimises what gets reported.
    """
    from sklearn.metrics import mean_absolute_percentage_error

    from src.model_building.model_building import mape_on_price_scale

    y_true_rupees = np.array([1.0, 10.0, 100.0])
    y_log_true = np.log1p(y_true_rupees)
    y_log_pred = np.log1p(y_true_rupees * 1.10)      # uniformly 10% over

    # A uniform 10% overprediction is 10% error, whatever the price level.
    assert mape_on_price_scale(y_log_true, y_log_pred) == pytest.approx(0.10, abs=1e-9)

    # Scoring the logs directly does not give 10% — that is the bug this guards.
    log_space = mean_absolute_percentage_error(y_log_true, y_log_pred)
    assert log_space < 0.06, (
        "log-space MAPE happens to match rupee MAPE here; the test cannot "
        "distinguish the two objectives and needs rewriting"
    )
