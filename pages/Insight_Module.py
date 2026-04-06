import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "fs",
    "feature_selected_properties.csv"
)

COEF_PATH = os.path.join(
    BASE_DIR,
    "notebooks",
    "insight_module",
    "insight_coefficients.csv"
)

# Load Data

df = pd.read_csv(DATA_PATH)

coef_df = pd.read_csv(COEF_PATH)
COEFS = dict(zip(coef_df["feature"], coef_df["coef"]))

st.header("📊 Pricing Insights")

# Area Impact

BASE_PRICE = df["price_in_cr"].median() * 1e7


st.subheader("📐 Impact of Area Increase")

delta_area = st.slider(
    "Increase in area (sqft)",
    min_value=100,
    max_value=1000,
    step=100,
    value=100
)

area_std = df["total_area_sqft"].std()

scaled_delta = delta_area / area_std

log_price_change = COEFS["num__total_area_sqft"] * scaled_delta

price_multiplier = np.exp(log_price_change)

pct_change = (price_multiplier - 1) * 100
abs_change = BASE_PRICE * (price_multiplier - 1)

st.metric(
    label="Estimated Price Impact",
    value=f"₹{abs_change/1e5:.1f} L",
    delta=f"{pct_change:.2f}%"
)

st.caption(
    "An increase of 100 sqft is estimated to increase property value by approximately 1.7%, based on the model."
)

# Bedroom Impact

st.subheader("🏠 Impact of Bedroom Change")

BEDROOM_OPTIONS = [1, 2, 3, 4, 5]

current_bed = st.selectbox(
    "Current Bedrooms",
    BEDROOM_OPTIONS,
    index=1
)

target_bed = st.selectbox(
    "Target Bedrooms",
    BEDROOM_OPTIONS,
    index=2
)

delta_bed = target_bed - current_bed

if delta_bed != 0:

    bedroom_std = df["bedrooms"].std()
    scaled_delta = delta_bed / bedroom_std

    log_price_change = COEFS["num__bedrooms"] * scaled_delta

    price_multiplier = np.exp(log_price_change)

    pct_change = (price_multiplier - 1) * 100
    abs_change = BASE_PRICE * (price_multiplier - 1)

    st.metric(
        "Estimated Price Impact",
        f"₹{abs_change/1e5:.1f} L",
        f"{pct_change:.2f}%"
    )

    st.caption(
        f"Change of {delta_bed:+d} bedroom(s) evaluated at median market price."
    )

else:
    st.caption("No change in bedroom count selected.")

# Utility Feature Premiums

st.subheader("✨ Utility Feature Premiums")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🧹 Servant Room")

    if st.checkbox("Add Servant Room", key="servant_premium"):

        servant_std = df["servant_room"].std()
        scaled_delta = 1 / servant_std

        log_price_change = COEFS["num__servant_room"] * scaled_delta

        price_multiplier = np.exp(log_price_change)

        pct_change = (price_multiplier - 1) * 100
        abs_change = BASE_PRICE * (price_multiplier - 1)

        st.metric(
            "Estimated Price Impact",
            f"₹{abs_change/1e5:.1f} L",
            f"{pct_change:.2f}%"
        )

with col2:
    st.markdown("### 📦 Pooja Room")

    if st.checkbox("Add Pooja Room", key="pooja_premium"):

        pooja_std = df["pooja_room"].std()
        scaled_delta = 1 / pooja_std

        log_price_change = COEFS["num__pooja_room"] * scaled_delta

        price_multiplier = np.exp(log_price_change)

        pct_change = (price_multiplier - 1) * 100
        abs_change = BASE_PRICE * (price_multiplier - 1)

        st.metric(
            "Estimated Price Impact",
            f"₹{abs_change/1e5:.1f} L",
            f"{pct_change:.2f}%"
        )

# Upgrade Scenario

st.subheader("📈 Property Upgrade Scenario")

col1, col2, col3 = st.columns(3)

with col1:
    add_area = st.slider("Extra Area (sqft)", 0, 500, 0, step=50)

with col2:
    add_bedroom = st.checkbox("Add Bedroom", key="upgrade_bedroom")

with col3:
    add_servant = st.checkbox("Add Servant Room", key="upgrade_servant")

total_log_change = 0

# Area Impact
if add_area > 0:
    area_std = df["total_area_sqft"].std()
    scaled_delta = add_area / area_std
    total_log_change += COEFS["num__total_area_sqft"] * scaled_delta

# Bedroom Impact
if add_bedroom:
    bedroom_std = df["bedrooms"].std()
    scaled_delta = 1 / bedroom_std
    total_log_change += COEFS["num__bedrooms"] * scaled_delta

# Servant Room Impact
if add_servant:
    servant_std = df["servant_room"].std()
    scaled_delta = 1 / servant_std
    total_log_change += COEFS["num__servant_room"] * scaled_delta

price_multiplier = np.exp(total_log_change)

pct_change = (price_multiplier - 1) * 100
abs_change = BASE_PRICE * (price_multiplier - 1)

st.metric(
    "Total Upgrade Value",
    f"₹{abs_change/1e5:.1f} L",
    f"{pct_change:.2f}%"
)

st.caption(
    "Estimated combined impact of selected property upgrades."
)

# Sector Price Premium Chart

st.subheader("🏆 Top 10 Most Expensive Sectors vs Market Average")

sector_coefs = {
    k.replace("cat__sector_", ""): v
    for k, v in COEFS.items()
    if k.startswith("cat__sector_")
}

sector_df = pd.DataFrame({
    "sector": sector_coefs.keys(),
    "coef": sector_coefs.values()
})

avg_coef = sector_df["coef"].mean()

sector_df["coef_vs_market"] = sector_df["coef"] - avg_coef

sector_df["price_premium_%"] = (
    np.exp(sector_df["coef_vs_market"]) - 1
) * 100

top_sectors = sector_df.sort_values(
    "price_premium_%", ascending=False
).head(10)

fig = px.bar(
    top_sectors.sort_values("price_premium_%"),
    x="price_premium_%",
    y="sector",
    orientation="h",
    title="Top 10 Most Expensive Sectors vs Market Average"
)

fig.update_layout(
    xaxis_title="Price Premium (%)",
    yaxis_title="Sector"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("🏢 Top 10 Premium Societies")

society_coefs = {
    k.replace("cat__society_", ""): v
    for k, v in COEFS.items()
    if k.startswith("cat__society_")
}

society_df = pd.DataFrame({
    "society": society_coefs.keys(),
    "coef": society_coefs.values()
})

avg_coef = society_df["coef"].mean()

society_df["coef_vs_market"] = society_df["coef"] - avg_coef

society_df["premium_%"] = (
    np.exp(society_df["coef_vs_market"]) - 1
) * 100

top_societies = society_df.sort_values(
    "premium_%", ascending=False
).head(10)

fig = px.bar(
    top_societies.sort_values("premium_%"),
    x="premium_%",
    y="society",
    orientation="h",
    title="Top 10 Premium Societies vs Market Average"
)

fig.update_layout(
    xaxis_title="Price Premium (%)",
    yaxis_title="Society"
)

st.plotly_chart(fig, use_container_width=True)
