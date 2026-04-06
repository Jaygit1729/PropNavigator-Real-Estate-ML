import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np


# 1. Page Config

st.set_page_config(
    page_title="PropNavigator | Price Estimator",
    layout="wide"
)

st.title("🏡 PropNavigator: Property Price Estimator")
st.caption("Estimate Gurgaon property prices using ML-powered insights")


# 2. Path Setup

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "best_model.joblib"
)

DF_PATH = os.path.join(
    BASE_DIR,
    "data",
    "fs",
    "feature_selected_properties.csv"
)


# 3. Cached Loaders

@st.cache_resource(show_spinner=False)
def load_model(path):
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_dataframe(path):
    return pd.read_csv(path)


# 4. Load Model & Data

df = load_dataframe(DF_PATH)

artifact = load_model(MODEL_PATH)

model_pipeline = artifact["pipeline"]
model_name = artifact["model_name"]
mape_percent = artifact["test_mape_percent"]

# Convert percentage to decimal

mape = mape_percent / 100


def get_options(col):
    return sorted(df[col].dropna().unique())


# 5. Property Details

st.header("🔹 Property Details")

col1, col2, col3 = st.columns(3)

with col1:

    property_type = st.selectbox(
        "Property Type",
        get_options("property_type")
    )

    sector = st.selectbox(
        "Sector",
        get_options("sector")
    )

    # Society filtered by sector

    sector_societies = sorted(
        df[df["sector"] == sector]["society"].dropna().unique()
    )

    society = st.selectbox(
        "Society",
        sector_societies
    )

    age_possession = st.selectbox(
        "Age Possession",
        get_options("age_possession")
    )


with col2:

    total_area = st.number_input(
        "Total Area (Sqft)",
        min_value=300.0,
        value=800.0,
        step=50.0
    )

    bedrooms = st.selectbox(
        "Bedrooms",
        get_options("bedrooms")
    )

    luxury_category = st.selectbox(
        "Luxury Category",
        get_options("luxury_category")
    )


with col3:

    bathrooms = st.selectbox(
        "Bathrooms",
        get_options("bathrooms")
    )

    furnishing_type = st.selectbox(
        "Furnishing Type",
        get_options("furnishing_type")
    )


# 6. Advanced Options

with st.expander("⚙️ Advanced Options (Optional)"):

    col4, col5, col6 = st.columns(3)

    with col4:

        servant_room = st.selectbox(
            "Servant Room",
            get_options("servant_room"),
            index=0
        )

        pooja_room = st.selectbox(
            "Pooja Room",
            get_options("pooja_room"),
            index=0
        )

        balcony = st.selectbox(
            "Balcony",
            get_options("balcony"),
            index=0
        )

    with col6:

        facing = st.selectbox(
            "Facing",
            [
                f for f in get_options("facing")
                if f.lower() not in ["not available", "unknown"]
            ],
            index=0
        )


# 7. Auto Derived Features

area_per_bedroom = total_area / bedrooms if bedrooms > 0 else 0
plot_area_missing = 0



# 8. Price Prediction

st.markdown("---")

if st.button("💰 Estimate Price"):

    
    if total_area > 15000:
        st.warning("⚠️ Extremely large area detected. Prediction may be unreliable.")

    input_df = pd.DataFrame([{
        "property_type": property_type,
        "society": society,
        "sector": sector,
        "total_area_sqft": total_area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "balcony": balcony,
        "servant_room": servant_room,
        "pooja_room": pooja_room,
        "facing": facing,
        "furnishing_type": furnishing_type,
        "age_possession": age_possession,
        "luxury_category": luxury_category,
        "area_per_bedroom": area_per_bedroom,
        "plot_area_missing": plot_area_missing
    }])

    try:

        log_price = model_pipeline.predict(input_df)[0]

        final_price = np.expm1(log_price)

        st.success(f"### 💵 Estimated Price: ₹{final_price:.2f} Crore")

        # Price Range using MAPE

        lower = final_price / (1 + mape)
        upper = final_price * (1 + mape)

        st.info(

            f"📊 Estimated price range: **₹{lower:.2f} Cr – ₹{upper:.2f} Cr** "
            f"(±{mape_percent:.2f}% model error)"
        )

        st.caption(f"Model used: **{model_name}**")

    except Exception as e:

        st.error(f"❌ Prediction failed: {e}")


# 9. Footer

st.markdown("---")

st.caption(
    "🔍 This is a predictive estimate. Actual prices may vary based on market conditions."
)