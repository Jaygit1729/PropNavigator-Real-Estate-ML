"""Regenerate the reference tables the API and the Streamlit pages read.

Two are load-bearing -- the model cannot be called without them, because they
supply inputs the user never types:

    sector_reference.csv   sector -> the four landmark distances
    type_reference.csv     property type -> default floors / balcony / band

Four drive charts only:

    sector_premiums.csv           priciest sectors        (insight page)
    society_premiums.csv          priciest societies      (insight page)
    shap_global_importance.csv    what moves predictions  (insight page)
    analytics_base.csv            everything              (analytics page)

The load-bearing pair is written to BOTH data/price_prediction/ (read by the
API and the price-prediction page) and data/insight_module/ (read by the
insight page), because each consumer looks in its own folder. Generating both
copies here is what stops them drifting apart -- the previous copies were
maintained separately and ended up two months stale.

Premium definition
------------------
The previous sector_premiums.csv could not be reproduced from the data: its
values imply a hidden baseline and look like model predictions on a reference
property that was never recorded. society_premiums.csv was plainly data
derived. Both are now defined the same way, so the two charts are comparable:

    price_cr     median price of that sector / society
    premium_pct  median price per sqft against the city median, as a percentage

Price per sqft drives the premium rather than price, so a sector of large
houses is not flattered simply because its units are bigger.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src.model_building.model_building import MODEL_PATH, create_train_val_test_split

ROOT = Path(__file__).resolve().parents[1]

PRICE_PREDICTION_DIR = ROOT / "data/price_prediction"
INSIGHT_DIR = ROOT / "data/insight_module"
ANALYTICS_DIR = ROOT / "data/analytics_module"

DIST_COLS = [
    "dist_to_cyber_city",
    "dist_to_golf_road",
    "dist_to_airport",
    "dist_to_manesar",
]

ANALYTICS_COLS = [
    "property_type", "society", "sector", "price_in_cr", "price_per_sqft",
    "area", "bedRoom", "bathroom", "balcony", "latitude", "longitude",
    "features", *DIST_COLS,
]

MIN_LISTINGS = 5   # a median over fewer rows than this is noise, not a premium


def write(df, *dirs, name):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        df.to_csv(d / name, index=False)
    where = ", ".join(str(d.relative_to(ROOT)).replace("\\", "/") for d in dirs)
    print(f"  {name:<30} {len(df):>6} rows -> {where}")


def build_sector_reference(pp):
    """Sector -> median distance to each landmark.

    Distance is a geographic property of the sector, so a median over its
    listings is a stable lookup. This is what lets a user pick a sector
    without being asked for four distances the model requires.
    """
    return (
        pp.groupby("sector")[DIST_COLS].median()
        .reset_index()
        .sort_values("sector")
        .reset_index(drop=True)
    )


def build_type_reference(pp):
    """Property type -> defaults for fields the forms do not collect."""
    return (
        pp.groupby("property_type")
        .agg(
            total_floor=("total_floor", "median"),
            balcony=("balcony", "median"),
            floornum_category=("floornum_category", lambda s: s.mode().iat[0]),
        )
        .reset_index()
    )


def build_premiums(pp, key):
    """Median price and price-per-sqft premium against the city, by key."""
    city_ppsf = pp["ppsf"].median()

    out = (
        pp.groupby(key)
        .agg(
            price_cr=("price_in_cr", "median"),
            ppsf=("ppsf", "median"),
            listings=("price_in_cr", "size"),
        )
        .reset_index()
    )
    out = out[out["listings"] >= MIN_LISTINGS].copy()
    out["premium_pct"] = (out["ppsf"] / city_ppsf * 100 - 100).round(1)
    out["price_cr"] = out["price_cr"].round(3)

    return (
        out.sort_values("premium_pct", ascending=False)
        [[key, "price_cr", "premium_pct", "listings"]]
        .reset_index(drop=True)
    )


def build_shap_importance(fs):
    """Mean absolute SHAP value per feature, measured on the test split.

    Computed on held-out rows, so the chart describes how the model behaves on
    data it has not seen. Values are contributions to log-price, the scale the
    model predicts on -- they rank features, they are not rupees.
    """
    import shap

    bundle = joblib.load(ROOT / MODEL_PATH)
    pipeline = bundle["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]

    *_, X_test, _, _, _ = create_train_val_test_split(fs)

    encoded = preprocessor.transform(X_test)
    names = list(preprocessor.get_feature_names_out())

    shap_values = shap.TreeExplainer(regressor).shap_values(encoded)

    return (
        pd.DataFrame({"feature": names,
                      "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def build_analytics_base(fe):
    """The analytics page reads feature-engineered rows directly.

    Kept at the feature-engineering stage rather than preprocessed, because the
    page describes the market as listed -- it needs society, coordinates and the
    raw amenity text, all of which preprocessing drops.
    """
    return fe[[c for c in ANALYTICS_COLS if c in fe.columns]].copy()


def main():
    pp = pd.read_csv(ROOT / "data/pp/preprocessed_properties.csv")
    fs = pd.read_csv(ROOT / "data/fs/feature_selected_properties.csv")
    fe = pd.read_csv(ROOT / "data/fe/featured_properties.csv")
    pp["ppsf"] = pp["price_in_cr"] * 1e7 / pp["area"]

    print(f"sources: pp {len(pp):,} rows | fs {len(fs):,} | fe {len(fe):,}")
    print(f"city median price {pp['price_in_cr'].median():.2f} Cr | "
          f"city median Rs/sqft {pp['ppsf'].median():,.0f}\n")

    print("load-bearing (model cannot run without these):")
    write(build_sector_reference(pp), PRICE_PREDICTION_DIR, INSIGHT_DIR,
          name="sector_reference.csv")
    write(build_type_reference(pp), PRICE_PREDICTION_DIR, INSIGHT_DIR,
          name="type_reference.csv")

    print("\ncharts:")
    write(build_premiums(pp, "sector"), INSIGHT_DIR, name="sector_premiums.csv")
    write(build_premiums(pp, "society"), INSIGHT_DIR, name="society_premiums.csv")
    write(build_analytics_base(fe), ANALYTICS_DIR, name="analytics_base.csv")

    print("\ncomputing SHAP values on the test split ...")
    write(build_shap_importance(fs), INSIGHT_DIR, name="shap_global_importance.csv")


if __name__ == "__main__":
    main()
