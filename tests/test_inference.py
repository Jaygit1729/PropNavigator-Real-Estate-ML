"""Tests for the serving path (api/inference.py).

This is the code that turns a handful of user inputs into the 24 features the model
demands. Most serving bugs are silent — a wrong column order still returns a number,
it's just the wrong number — so these assertions matter more than they look.
"""

import numpy as np
import pytest


def test_feature_list_comes_from_the_model(service):
    """FEATURES must be read off the fitted pipeline, not hardcoded.

    This is what let the API survive the 25 -> 24 feature change without edits.
    """
    assert service.FEATURES == list(service.pipeline.feature_names_in_)
    assert len(service.FEATURES) == 24
    assert "society" not in service.FEATURES


def test_built_row_matches_model_columns_exactly(service, valid_request):
    """Column set AND order must match the model.

    Order matters: the ColumnTransformer passes untouched columns through by
    position, so a shuffled frame silently feeds values to the wrong features.
    """
    row = service.build_input_row(valid_request)
    assert list(row.columns) == service.FEATURES
    assert len(row) == 1


def test_predict_returns_sane_bounded_price(service, valid_request):
    out = service.predict_price(valid_request)

    assert out["lower_bound_cr"] < out["predicted_price_cr"] < out["upper_bound_cr"]
    assert 0.1 < out["predicted_price_cr"] < 100, "price outside any plausible range"
    assert out["model_name"] == service.model_name
    # JSON-serialisable floats, not numpy scalars
    assert all(isinstance(out[k], float)
               for k in ("predicted_price_cr", "lower_bound_cr", "upper_bound_cr"))


def test_house_without_floors_uses_per_type_default(service, valid_request):
    """Independent houses have no building floors; omitting total_floor must fall
    back to the per-property-type default rather than crashing or using the flat one."""
    house = {**valid_request, "property_type": "Independent House", "total_floor": None}
    row = service.build_input_row(house)

    assert row["total_floor"].iloc[0] == 0.0
    assert row["floornum_category"].iloc[0] == "Low-rise"


def test_unknown_sector_is_rejected(service, valid_request):
    """An unknown sector must raise, not silently predict from NaN distances."""
    with pytest.raises(ValueError, match="Unknown sector"):
        service.predict_price({**valid_request, "sector": "atlantis"})


def test_distances_are_looked_up_from_the_sector(service, valid_request):
    """The four distances are geographic facts of the chosen sector — they must be
    populated, and must differ between genuinely different sectors."""
    row_a = service.build_input_row(valid_request)
    row_b = service.build_input_row({**valid_request, "sector": "dlf cyber city"})

    for d in service.DIST:
        assert not np.isnan(row_a[d].iloc[0]), f"{d} not populated"
    assert (row_a[service.DIST].iloc[0].values
            != row_b[service.DIST].iloc[0].values).any(), "distances identical across sectors"


def test_amenities_default_to_absent(service):
    """Omitted amenity flags must default to 0, not error or default to 1."""
    minimal = {"property_type": "Flat", "sector": "sector 49",
               "area": 1500, "bedRoom": 3, "bathroom": 2}
    row = service.build_input_row(minimal)

    for flag in ("has_ac", "has_power_backup", "has_pool", "is_corner"):
        assert row[flag].iloc[0] == 0


def test_bigger_area_predicts_higher_price(service, valid_request):
    """A directional sanity check: all else equal, more area should cost more.

    Catches catastrophic wiring errors (swapped columns, inverted target) that a
    single-prediction test would happily pass.
    """
    small = service.predict_price({**valid_request, "area": 1000})["predicted_price_cr"]
    large = service.predict_price({**valid_request, "area": 3000})["predicted_price_cr"]
    assert large > small
