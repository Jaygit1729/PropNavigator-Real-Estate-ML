"""Guards on the feature contract.

SELECTED_FEATURES is hardcoded — the ranking notebook is the justification, the
constant is the decision, and nothing previously connected the two. If an
upstream stage renames or drops a column, the selection step just skips it with
a warning and the model trains on fewer features than intended. These tests turn
that silent degradation into a failure.
"""

import pandas as pd

from src.feature_selection.feature_selection import SELECTED_FEATURES, select_features

TARGET = "price_in_cr"


def test_every_selected_feature_exists_upstream(raw_pp_df):
    """Preprocessing must supply every feature the model expects."""
    missing = [c for c in SELECTED_FEATURES if c not in raw_pp_df.columns]
    assert not missing, (
        f"preprocessed data is missing {missing} — selection would silently "
        "skip these and the model would train on a smaller feature set"
    )


def test_selected_features_are_unique():
    """A duplicated name would silently produce a duplicated column."""
    dupes = [c for c in set(SELECTED_FEATURES) if SELECTED_FEATURES.count(c) > 1]
    assert not dupes, f"duplicated feature names: {dupes}"


def test_target_is_not_a_feature():
    assert TARGET not in SELECTED_FEATURES, "target leaked into the feature list"


def test_price_per_sqft_never_reaches_the_model(fs_df):
    """price_per_sqft * area reconstructs the target exactly — it must not survive."""
    assert "price_per_sqft" not in fs_df.columns, (
        "price_per_sqft is in the modelling data; it encodes the target"
    )


def test_society_is_excluded(fs_df):
    """Society is ~a lookup of price and is not reliably known at serve time.

    Measured: 10.65% MAPE with true values, 14.29% with the ~31%-accurate guess
    production could actually obtain, 11.18% retrained without it.
    """
    assert "society" not in fs_df.columns, (
        "society is in the modelling data — it inflates offline scores and "
        "degrades production, see the ablation in the feature-selection notebook"
    )


def test_shipped_dataset_matches_the_hardcoded_contract(fs_df):
    """The dataset on disk must have exactly the agreed columns, no drift."""
    expected = set(SELECTED_FEATURES) | {TARGET}
    assert set(fs_df.columns) == expected, (
        f"unexpected: {sorted(set(fs_df.columns) - expected)}, "
        f"absent: {sorted(expected - set(fs_df.columns))}"
    )


def test_selection_removes_duplicates_created_by_narrowing(raw_pp_df):
    """Narrowing columns collapses rows that were distinct in the wider table.

    Earlier de-duplication passes cannot catch these: the rows genuinely differ
    on columns feature selection is about to drop. If a pair straddles the
    train/test split the model recalls rather than predicts.
    """
    out = select_features(raw_pp_df, target=TARGET)
    keep = [c for c in SELECTED_FEATURES if c in raw_pp_df.columns]

    assert len(out) < len(raw_pp_df), "narrowing produced no duplicates — suspicious"
    assert not out.duplicated(subset=keep).any(), "duplicates survived selection"
