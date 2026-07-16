import os
import sys

import numpy as np
import pandas as pd
import joblib
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.monitoring.prediction_logger import log_prediction


# Page Config

st.set_page_config(page_title="PropNavigator | Price Estimator", layout="wide")


# Constants

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "best_model.joblib")
SECTOR_REF = os.path.join(BASE_DIR, "data", "price_prediction", "sector_reference.csv")
TYPE_REF   = os.path.join(BASE_DIR, "data", "price_prediction", "type_reference.csv")

FEATURES = [
    'area', 'dist_to_golf_road', 'total_floor', 'bathroom', 'property_type',
    'dist_to_cyber_city', 'covered_parking', 'dist_to_manesar', 'sector',
    'dist_to_airport', 'bedRoom', 'furnishing', 'balcony', 'age_possession_category',
    'facing', 'has_ac', 'total_parking', 'open_parking', 'has_power_backup',
    'is_corner', 'floornum_category', 'ov_main_road', 'has_pool', 'ov_others',
]
DIST = ['dist_to_cyber_city', 'dist_to_golf_road', 'dist_to_airport', 'dist_to_manesar']

PROPERTY_TYPES = ['Flat', 'Independent Builder Floor', 'Independent House']
FURNISHING     = ['unfurnished', 'semi-furnished', 'furnished']   # no 'unknown' shown to user
AGE_OPTIONS    = ['New Property', 'Relatively New', 'Moderately Old', 'Old Property', 'Under Construction']

# Fields we no longer ask the user — filled internally
CONST = dict(facing='unknown', open_parking=0.0, ov_main_road=0, ov_others=0)


# Loaders

@st.cache_resource(show_spinner=False)
def load_model(path):
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path)


# Feature assembly + prediction  (mirrors notebooks/price_prediction/price_prediction.ipynb)

def build_input_row(inp, sector_ref, type_ref):
    pt = inp['property_type']
    sr = sector_ref.loc[sector_ref['sector'] == inp['sector']]
    tr = type_ref.loc[type_ref['property_type'] == pt]
    tr = tr.iloc[0] if not tr.empty else type_ref.iloc[0]

    covered = float(inp.get('covered_parking', 1))
    # total_floor: user value for flats/builder floors; per-type default for houses (-> 0)
    total_floor = float(inp['total_floor']) if inp.get('total_floor') is not None else float(tr['total_floor'])
    row = {
        'area': float(inp['area']),
        'total_floor': total_floor,
        'bathroom': int(inp['bathroom']),
        'property_type': pt,
        'covered_parking': covered,
        'open_parking': CONST['open_parking'],
        'sector': inp['sector'],
        'bedRoom': int(inp['bedRoom']),
        'furnishing': inp.get('furnishing', 'semi-furnished'),
        'balcony': float(tr['balcony']),                         # per-type default
        'age_possession_category': inp.get('age_possession_category', 'New Property'),
        'facing': CONST['facing'],                               # weak feature -> not asked
        'floornum_category': tr['floornum_category'],            # per-type default
        'total_parking': covered + CONST['open_parking'],
    }
    for flag in ['has_ac', 'has_power_backup', 'has_pool', 'is_corner']:
        row[flag] = int(inp.get(flag, 0))
    row['ov_main_road'] = CONST['ov_main_road']
    row['ov_others'] = CONST['ov_others']
    for d in DIST:
        row[d] = float(sr[d].iloc[0]) if not sr.empty else np.nan
    return pd.DataFrame([row])[FEATURES]


def predict(pipeline, input_df, residual_q):
    price = float(np.expm1(pipeline.predict(input_df)[0]))
    if residual_q:
        return round(price, 2), round(price * np.exp(residual_q.get('q05', 0)), 2), \
               round(price * np.exp(residual_q.get('q95', 0)), 2)
    return round(price, 2), round(price * 0.8, 2), round(price * 1.2, 2)


# Load

bundle     = load_model(MODEL_PATH)
sector_ref = load_csv(SECTOR_REF)
type_ref   = load_csv(TYPE_REF)
pipeline   = bundle['pipeline']
model_name = bundle['model_name']
mape_pct   = bundle['test_mape_percent']
residual_q = bundle.get('residual_quantiles', {})
sectors    = sorted(sector_ref['sector'].tolist())


# UI — Header

st.title("🏡 PropNavigator: Property Price Estimator")
st.caption(
    f"Estimate a Gurgaon property's price with the deployed model "
    f"(**{model_name}**, ~{mape_pct:.1f}% MAPE). Just the essentials — location and "
    "building details are inferred from the sector for you."
)
st.divider()


# UI — Inputs (reduced set)

st.subheader("🔹 Property Details")
c1, c2, c3 = st.columns(3)

with c1:
    property_type = st.selectbox("Property Type", PROPERTY_TYPES)
    sector = st.selectbox("Sector", sectors,
                          index=sectors.index("sector 49") if "sector 49" in sectors else 0)
    area = st.number_input("Area (sqft)", min_value=200.0, max_value=27000.0, value=1500.0, step=50.0)

with c2:
    bedRoom  = st.selectbox("Bedrooms", list(range(1, 11)), index=2)
    bathroom = st.selectbox("Bathrooms", list(range(1, 11)), index=1)
    furnishing = st.selectbox("Furnishing", FURNISHING, index=1)

with c3:
    age_possession_category = st.selectbox("Age / Possession", AGE_OPTIONS)
    covered_parking = st.number_input("Parking Spaces", min_value=0, max_value=10, value=1, step=1)
    # Total floors applies to flats / builder floors, not independent houses.
    if property_type != "Independent House":
        _default_tf = int(type_ref.loc[type_ref['property_type'] == property_type, 'total_floor'].iloc[0])
        total_floor = st.number_input("Total Floors in Building", min_value=1, max_value=90,
                                      value=max(_default_tf, 1), step=1)
    else:
        total_floor = None

st.markdown("#### ✨ Amenities")
a1, a2, a3, a4 = st.columns(4)
with a1:
    has_ac = st.checkbox("Air Conditioning", value=True)
with a2:
    has_power_backup = st.checkbox("Power Backup", value=True)
with a3:
    has_pool = st.checkbox("Swimming Pool", value=False)
with a4:
    is_corner = st.checkbox("Corner Property", value=False)

if property_type == "Independent House":
    st.caption("ℹ️ Floor details don't apply to independent houses — handled automatically.")


# UI — Prediction

st.divider()

if st.button("💰 Estimate Price", type="primary", use_container_width=True):
    with st.spinner("Estimating price..."):
        inp = dict(
            property_type=property_type, sector=sector, area=area, bedRoom=bedRoom,
            bathroom=bathroom, furnishing=furnishing, age_possession_category=age_possession_category,
            covered_parking=covered_parking, total_floor=total_floor,
            has_ac=int(has_ac), has_power_backup=int(has_power_backup),
            has_pool=int(has_pool), is_corner=int(is_corner),
        )
        try:
            input_df = build_input_row(inp, sector_ref, type_ref)
            price, lo, hi = predict(pipeline, input_df, residual_q)

            st.success(f"### 💵 Estimated Price: ₹ {price} Crore")
            st.info(f"📊 90% Price Range: **₹ {lo} Cr — ₹ {hi} Cr**")
            st.caption(f"Model: **{model_name}**  ·  sector: *{input_df['sector'].iloc[0]}*")

            log_prediction(
                model_name,
                {"property_type": property_type, "sector": sector, "area": area, "bedRoom": bedRoom},
                {"predicted_price": price, "lower_bound": lo, "upper_bound": hi},
            )
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")


# Footer

st.divider()
st.caption("🔍 Predictive estimate only. Actual prices vary with market conditions, finish, and negotiation.")
