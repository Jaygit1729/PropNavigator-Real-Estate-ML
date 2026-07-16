"""Turn caller inputs into a price.

This is the single place that owns the "user inputs -> 24 model features -> price"
logic, so the API (and later the Streamlit page) can never disagree about what a
property is worth.

Loaded once at import: the model and the two small reference tables.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Loaded once, at import (not per request) ---
bundle = joblib.load(BASE_DIR / "artifacts" / "best_model.joblib")
pipeline = bundle["pipeline"]
model_name = bundle["model_name"]
mape_percent = float(bundle["test_mape_percent"])
residual_q = bundle.get("residual_quantiles", {})

sector_ref = pd.read_csv(BASE_DIR / "data" / "price_prediction" / "sector_reference.csv")
type_ref = pd.read_csv(BASE_DIR / "data" / "price_prediction" / "type_reference.csv")
VALID_SECTORS = sorted(sector_ref["sector"].unique())

# Ask the MODEL which columns it wants, rather than hardcoding a list.
# If the feature set changes again, this adapts automatically.
FEATURES = list(pipeline.feature_names_in_)

DIST = ['dist_to_cyber_city', 'dist_to_golf_road', 'dist_to_airport', 'dist_to_manesar']

# Weak features we don't bother asking about — a fixed value costs us nothing.
CONST = dict(facing='unknown', open_parking=0.0, ov_main_road=0, ov_others=0)


def build_input_row(inp: dict) -> pd.DataFrame:
    """Assemble the full model row from the caller's handful of inputs."""
    sr = sector_ref.loc[sector_ref['sector'] == inp['sector']]
    tr = type_ref.loc[type_ref['property_type'] == inp['property_type']]
    tr = tr.iloc[0] if not tr.empty else type_ref.iloc[0]

    covered = float(inp.get('covered_parking', 1))
    # Caller's value for flats/builder floors; per-type default for houses (-> 0).
    total_floor = (
        float(inp['total_floor'])
        if inp.get('total_floor') is not None
        else float(tr['total_floor'])
    )

    row = {
        # straight from the caller
        'area': float(inp['area']),
        'bedRoom': int(inp['bedRoom']),
        'bathroom': int(inp['bathroom']),
        'property_type': inp['property_type'],
        'sector': inp['sector'],
        'furnishing': inp.get('furnishing', 'semi-furnished'),
        'age_possession_category': inp.get('age_possession_category', 'New Property'),
        'covered_parking': covered,
        'total_floor': total_floor,
        # per-property-type defaults
        'balcony': float(tr['balcony']),
        'floornum_category': tr['floornum_category'],
        # constants for weak features
        'facing': CONST['facing'],
        'open_parking': CONST['open_parking'],
        'ov_main_road': CONST['ov_main_road'],
        'ov_others': CONST['ov_others'],
    }

    # derived
    row['total_parking'] = covered + CONST['open_parking']

    # amenity toggles
    for flag in ['has_ac', 'has_power_backup', 'has_pool', 'is_corner']:
        row[flag] = int(inp.get(flag, 0))

    # geographic facts of the sector the caller picked
    for d in DIST:
        row[d] = float(sr[d].iloc[0])

    # Reindex to the model's exact column order.
    return pd.DataFrame([row])[FEATURES]


def predict_price(inp: dict) -> dict:
    """Point estimate + calibrated 90% range for one property."""
    if inp['sector'] not in set(sector_ref['sector']):
        raise ValueError(f"Unknown sector '{inp['sector']}'. See GET /sectors.")

    input_df = build_input_row(inp)

    # The model predicts log(price); expm1 undoes the log1p used in training.
    price = float(np.expm1(pipeline.predict(input_df)[0]))

    # Residual quantiles were measured on the test set, so the range is
    # calibrated from real errors rather than a guessed +/- percentage.
    lo = round(float(price * np.exp(residual_q.get('q05', 0))), 2)
    hi = round(float(price * np.exp(residual_q.get('q95', 0))), 2)

    return {
        "predicted_price_cr": round(price, 2),
        "lower_bound_cr": lo,
        "upper_bound_cr": hi,
        "model_name": model_name,
        "mape_percent": round(mape_percent, 2),
    }
