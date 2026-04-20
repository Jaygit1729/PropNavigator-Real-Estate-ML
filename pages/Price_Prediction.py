import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st



# Page Config

st.set_page_config(
    page_title="PropNavigator | Price Estimator",
    layout="wide"
)



# Constants

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "best_model.joblib")
DATA_PATH  = os.path.join(BASE_DIR, "data", "fs", "feature_selected_properties.csv")

# Simplified age/possession labels shown in UI → mapped to model's trained values
# Model was trained on original categories — we map simplified labels back before prediction
AGE_POSSESSION_MAP = {
    "New"               : "Relatively New",    # covers: New Property + Relatively New
    "Old"               : "Moderately Old",     # covers: Moderately Old + Old Property
    "Under Construction": "Under Construction"
}



# Loaders

@st.cache_data(show_spinner=False)
def load_dataframe(path: str):
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_artifact(path: str):
    return joblib.load(path)


# Helpers

def get_options(df: pd.DataFrame, col: str):
    """Returns sorted unique non-null values for a given column."""
    return sorted(df[col].dropna().unique().tolist())


def get_society_for_sector(df: pd.DataFrame, sector: str):
    """
    Returns the most common society for a given sector as a backend default.
    Society is not shown in the UI — it is derived internally.
    Falls back to 'other' if no meaningful society found for the sector.
    """
    sector_societies = (
        df[df["sector"] == sector]["society"]
        .dropna()
        .value_counts()
    )

    # Filter out "other" — it's a data artifact, not a real society
    meaningful = sector_societies[
        sector_societies.index.str.lower() != "other"
    ]

    if not meaningful.empty:
        return meaningful.index[0]  # most common real society for this sector

    return "other"  # safe fallback the model knows


def derive_features(total_area: float, bedrooms: int):
    """
    Derives features the model needs but the user never provides directly.
    - area_per_bedroom : total_area / bedrooms
    - plot_area_missing: always 0 — user inputs never include plot area
    """
    return {
        "area_per_bedroom" : total_area / bedrooms if bedrooms > 0 else total_area,
        "plot_area_missing": 0
    }


def build_input_df(raw_inputs: dict, derived: dict):
    """
    Combines raw user inputs and derived features into
    a single-row DataFrame matching the model's expected columns.

    Model expects exactly these 15 features:
    Numerical (7) : total_area_sqft, plot_area_missing, bathrooms,
                    area_per_bedroom, bedrooms, servant_room, pooja_room
    Categorical (8): property_type, sector, society, luxury_category,
                     balcony, facing, furnishing_type, age_possession
    """
    return pd.DataFrame([{**raw_inputs, **derived}])


def predict_price(pipeline, input_df: pd.DataFrame, mape: float):
    """
    Runs inference and returns predicted price with confidence bounds.
    Model was trained on log1p(price) — inverse transform applied here.
    """
    log_price       = pipeline.predict(input_df)[0]
    predicted_price = round(float(np.expm1(log_price)), 2)
    lower_bound     = round(predicted_price / (1 + mape), 2)
    upper_bound     = round(predicted_price * (1 + mape), 2)

    return {
        "predicted_price": predicted_price,
        "lower_bound"    : lower_bound,
        "upper_bound"    : upper_bound
    }


# Load Data & Model

df       = load_dataframe(DATA_PATH)
artifact = load_artifact(MODEL_PATH)

pipeline     = artifact["pipeline"]
model_name   = artifact["model_name"]
mape_percent = artifact["test_mape_percent"]
mape         = mape_percent / 100


# UI — Header

st.title("🏡 PropNavigator: Property Price Estimator")
st.caption("Estimate Gurgaon property prices using ML-powered insights")
st.divider()


st.subheader("🔹 Property Details")

col1, col2, col3 = st.columns(3)

with col1:

    property_type = st.selectbox(
        "Property Type",
        get_options(df, "property_type")
    )

    sector = st.selectbox(
        "Sector",
        get_options(df, "sector"),
        index=get_options(df, "sector").index("sector 1")
        if "sector 1" in get_options(df, "sector") else 0
    )

    age_possession_label = st.selectbox(
        "Age / Possession",
        list(AGE_POSSESSION_MAP.keys())
    )

    luxury_category = st.selectbox(
        "Luxury Category",
        get_options(df, "luxury_category")
    )

with col2:

    total_area = st.number_input(
        "Total Area (Sqft)",
        min_value=100.0,
        max_value=15000.0,
        value=800.0,
        step=50.0
    )

    bedroom_options = get_options(df, "bedrooms")
    bedrooms = st.selectbox(
        "Bedrooms",
        bedroom_options,
        index=bedroom_options.index(2) if 2 in bedroom_options else 0
    )

    bathrooms = st.selectbox(
        "Bathrooms",
        get_options(df, "bathrooms")
    )

with col3:

    furnishing_type_options = get_options(df, "furnishing_type")
    furnishing_type = st.selectbox(
        "Furnishing Type",
        furnishing_type_options,
        index=furnishing_type_options.index("Semi-Furnished")
        if "Semi-Furnished" in furnishing_type_options else 0
    )

    balcony = st.selectbox(
        "Balcony",
        get_options(df, "balcony"),
        index=0
    )

    facing_options = [
        f for f in get_options(df, "facing")
        if f.lower() != "not available"
    ]
    facing = st.selectbox(
        "Facing",
        facing_options,
        index=0
    )

# Additional rooms — full width row below the three columns
st.markdown("#### 🚪 Additional Rooms")

room_col1, room_col2, room_col3 = st.columns(3)

with room_col1:
    servant_room = st.selectbox(
        "Servant Room",
        get_options(df, "servant_room"),
        index=0
    )

with room_col2:
    pooja_room = st.selectbox(
        "Pooja Room",
        get_options(df, "pooja_room"),
        index=0
    )



# UI — Prediction Button

st.divider()

if st.button("💰 Estimate Price", use_container_width=True):

    with st.spinner("Estimating price..."):

        # Map simplified age label → model's trained category value
        age_possession = AGE_POSSESSION_MAP[age_possession_label]

        # Derive society internally — not shown in UI
        society = get_society_for_sector(df, sector)

        raw_inputs = {
            "property_type" : property_type,
            "society"       : society,         # derived, not from UI
            "sector"        : sector,
            "total_area_sqft": total_area,
            "bedrooms"      : bedrooms,
            "bathrooms"     : bathrooms,
            "servant_room"  : servant_room,
            "pooja_room"    : pooja_room,      
            "balcony"       : balcony,
            "facing"        : facing,
            "furnishing_type": furnishing_type,
            "age_possession": age_possession,  # mapped from simplified label
            "luxury_category": luxury_category
        }

        derived  = derive_features(total_area, bedrooms)
        input_df = build_input_df(raw_inputs, derived)

        try:
            result = predict_price(pipeline, input_df, mape)

            st.success(f"### 💵 Estimated Price: ₹ {result['predicted_price']} Crore")

            st.info(
                f"📊 Price Range: **₹ {result['lower_bound']} Cr — ₹ {result['upper_bound']} Cr** "
                f"(± {mape_percent:.2f}% model error)"
            )

            st.caption(f"Model used: **{model_name}**")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")


# UI — Footer

st.divider()
st.caption("🔍 Predictive estimate only. Actual prices may vary based on market conditions.")