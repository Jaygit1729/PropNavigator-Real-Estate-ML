"""Price estimator UI.

This page owns NO model logic. It collects inputs, POSTs them to the FastAPI
service, and renders the answer. The "inputs -> 24 features -> price" logic
lives in exactly one place: api/inference.py. That's what stops the website and
the API from ever quoting two different prices for the same property.
"""

import os
import sys

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.monitoring.prediction_logger import log_prediction


# Page Config

st.set_page_config(page_title="PropNavigator | Price Estimator", layout="wide")


# Constants

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TYPE_REF = os.path.join(BASE_DIR, "data", "price_prediction", "type_reference.csv")

# Where the prediction API lives. Overridable so the same code works locally,
# in docker-compose (http://api:8000), and on the server.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

PROPERTY_TYPES = ['Flat', 'Independent Builder Floor', 'Independent House']
FURNISHING     = ['unfurnished', 'semi-furnished', 'furnished']   # no 'unknown' shown to user
AGE_OPTIONS    = ['New Property', 'Relatively New', 'Moderately Old', 'Old Property', 'Under Construction']


# API helpers

@st.cache_data(ttl=60, show_spinner=False)
def api_get(path: str):
    r = requests.get(f"{API_URL}{path}", timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path)


# Ask the API what model is serving and which sectors it accepts.
try:
    health = api_get("/health")
    sectors = api_get("/sectors")["sectors"]
except Exception as e:
    st.error(
        f"**Can't reach the prediction API at `{API_URL}`.**\n\n"
        f"Start it from the project root:\n\n"
        f"```\nuvicorn api.main:app --reload\n```\n\n"
        f"Details: `{e}`"
    )
    st.stop()

model_name = health["model_name"]
mape_pct = health["mape_percent"]
type_ref = load_csv(TYPE_REF)   # UI prefill only (default floors per property type)


# UI — Header

st.title("🏡 PropNavigator: Property Price Estimator")
st.caption(
    f"Estimate a Gurgaon property's price with the deployed model "
    f"(**{model_name}**, ~{mape_pct:.1f}% MAPE), served live from the prediction API. "
    "Just the essentials — location and building details are inferred from the sector for you."
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
        # Omitted from the request -> the API fills the per-type default (ground).
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


# UI — Prediction (via the API)

st.divider()

if st.button("💰 Estimate Price", type="primary", use_container_width=True):
    with st.spinner("Estimating price..."):
        payload = dict(
            property_type=property_type, sector=sector, area=area, bedRoom=bedRoom,
            bathroom=bathroom, furnishing=furnishing,
            age_possession_category=age_possession_category,
            covered_parking=covered_parking, total_floor=total_floor,
            has_ac=int(has_ac), has_power_backup=int(has_power_backup),
            has_pool=int(has_pool), is_corner=int(is_corner),
        )
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)

            if r.status_code == 422:
                # The API's contract rejected the input — show why.
                st.error(f"❌ Invalid input: {r.json().get('detail')}")
            else:
                r.raise_for_status()
                out = r.json()
                price = out["predicted_price_cr"]
                lo, hi = out["lower_bound_cr"], out["upper_bound_cr"]

                st.success(f"### 💵 Estimated Price: ₹ {price} Crore")
                st.info(f"📊 90% Price Range: **₹ {lo} Cr — ₹ {hi} Cr**")
                st.caption(f"Model: **{out['model_name']}**  ·  sector: *{sector}*  ·  served by the API")

                log_prediction(
                    out["model_name"],
                    {"property_type": property_type, "sector": sector, "area": area, "bedRoom": bedRoom},
                    {"predicted_price": price, "lower_bound": lo, "upper_bound": hi},
                )
        except requests.RequestException as e:
            st.error(f"❌ Could not reach the prediction API: {e}")


# Footer

st.divider()
st.caption("🔍 Predictive estimate only. Actual prices vary with market conditions, finish, and negotiation.")
