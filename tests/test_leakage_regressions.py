"""Regression tests for the data-leakage bugs found in the 2026-07-19 audit.

Each test here corresponds to a real bug that was in the pipeline and got fixed.
They exist so nobody silently reintroduces one — a leak is invisible at runtime
(everything still "works", the metric just quietly lies).
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_selection.feature_selection import SELECTED_FEATURES, select_features
from src.model_building.mb_main import create_train_val_test_split


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
