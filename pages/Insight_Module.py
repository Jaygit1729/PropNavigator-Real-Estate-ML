import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Page Config

st.set_page_config(
    page_title="PropNavigator | Pricing Insights",
    layout="wide"
)


# Constants

BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(BASE_DIR, "data", "fs", "feature_selected_properties.csv")
COEF_PATH = os.path.join(BASE_DIR, "data", "insight_module", "insight_coefficients.csv")

BEDROOM_OPTIONS = [1, 2, 3, 4, 5]



# Loaders

@st.cache_data(show_spinner=False)
def load_dataframe(path: str):
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_coefficients(path: str):
    coef_df = pd.read_csv(path)
    return dict(zip(coef_df["feature"], coef_df["coef"]))



# Helpers


def compute_price_impact(coef: float, scaled_delta: float, base_price_rs: float) :
    """
    Converts a Ridge coefficient and scaled feature delta
    into a percentage change and absolute rupee change.

    Returns (pct_change, abs_change_in_rupees)
    """
    log_price_change = coef * scaled_delta
    price_multiplier = np.exp(log_price_change)
    pct_change       = (price_multiplier - 1) * 100
    abs_change       = base_price_rs * (price_multiplier - 1)
    return pct_change, abs_change


def precompute_stds(df: pd.DataFrame):
    """
    Precomputes standard deviations for all numerical features used
    in insight calculations. Avoids repeated inline df.std() calls.
    """
    return {
        "total_area_sqft": df["total_area_sqft"].std(),
        "bedrooms"        : df["bedrooms"].std(),
        "servant_room"    : df["servant_room"].std(),
        "pooja_room"      : df["pooja_room"].std()
    }


def build_premium_df(coefs: dict, prefix: str, label_col: str):
    """
    Builds a DataFrame of price premiums vs market average
    for a given feature prefix (e.g. 'cat__sector_' or 'cat__society_').
    """
    filtered = {
        k.replace(prefix, ""): v
        for k, v in coefs.items()
        if k.startswith(prefix)
    }

    premium_df = pd.DataFrame({
        label_col: filtered.keys(),
        "coef"    : filtered.values()
    })

    avg_coef                  = premium_df["coef"].mean()
    premium_df["coef_vs_avg"] = premium_df["coef"] - avg_coef
    premium_df["premium_%"]   = (np.exp(premium_df["coef_vs_avg"]) - 1) * 100

    return premium_df


# Load Data

df   = load_dataframe(DATA_PATH)
COEFS = load_coefficients(COEF_PATH)

# Base price in rupees — median market price used for absolute impact calculations
# Stored in rupees so abs_change / 1e5 gives Lakhs directly
BASE_PRICE_RS = df["price_in_cr"].median() * 1e7

# Precompute std values once — reused across all insight sections
STDS = precompute_stds(df)


# UI — Header

st.title("📈 PropNavigator: Pricing Insights")
st.caption(
    "Understand what drives property prices in Gurgaon — "
    "how area, bedrooms, amenities and location affect what you pay."
)
st.caption(
    "⚠️ Insights are derived from a Ridge regression trained on the same 15 features "
    "as the production model. Ridge is used because it produces interpretable coefficients. "
    "All impacts are estimated at the **median market price**."
)
st.divider()



# Section 1 — Area Impact

st.subheader("📐 Impact of Area Increase")
st.caption("How much does price change when you increase the property area?")

delta_area = st.slider(
    "Increase in area (sqft)",
    min_value=100,
    max_value=1000,
    step=100,
    value=100
)

scaled_delta        = delta_area / STDS["total_area_sqft"]
pct_change, abs_chg = compute_price_impact(
    COEFS["total_area_sqft"], scaled_delta, BASE_PRICE_RS
)

st.metric(
    label="Estimated Price Impact",
    value=f"₹{abs_chg / 1e5:.1f} L",
    delta=f"{pct_change:.2f}%"
)

st.caption(
    f"An increase of **{delta_area} sqft** is estimated to change "
    f"property value by approximately **{pct_change:.1f}%** at median market price."
)

st.divider()


# Section 2 — Bedroom Impact

st.subheader("🏠 Impact of Bedroom Change")
st.caption("How much does adding or removing a bedroom affect price?")

col1, col2 = st.columns(2)

with col1:
    current_bed = st.selectbox("Current Bedrooms", BEDROOM_OPTIONS, index=1)

with col2:
    target_bed = st.selectbox("Target Bedrooms", BEDROOM_OPTIONS, index=2)

delta_bed = target_bed - current_bed

if delta_bed != 0:
    scaled_delta        = delta_bed / STDS["bedrooms"]
    pct_change, abs_chg = compute_price_impact(
        COEFS["bedrooms"], scaled_delta, BASE_PRICE_RS
    )

    st.metric(
        label="Estimated Price Impact",
        value=f"₹{abs_chg / 1e5:.1f} L",
        delta=f"{pct_change:.2f}%"
    )

    st.caption(
        f"A change of **{delta_bed:+d} bedroom(s)** is estimated to "
        f"change property value by approximately **{pct_change:.1f}%** at median market price."
    )

else:
    st.info("Select different bedroom counts to see price impact.")

st.divider()


# Section 3 — Utility Feature Premiums

st.subheader("✨ Utility Feature Premiums")
st.caption("How much do additional rooms add to property value?")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🧹 Servant Room")

    if st.checkbox("Add Servant Room", key="servant_premium"):
        scaled_delta        = 1 / STDS["servant_room"]
        pct_change, abs_chg = compute_price_impact(
            COEFS["servant_room"], scaled_delta, BASE_PRICE_RS
        )
        st.metric(
            label="Estimated Price Impact",
            value=f"₹{abs_chg / 1e5:.1f} L",
            delta=f"{pct_change:.2f}%"
        )
        st.caption(f"Adding a servant room is estimated to change price by ~{pct_change:.1f}%.")

with col2:
    st.markdown("#### 🛕 Pooja Room")

    if st.checkbox("Add Pooja Room", key="pooja_premium"):
        scaled_delta        = 1 / STDS["pooja_room"]
        pct_change, abs_chg = compute_price_impact(
            COEFS["pooja_room"], scaled_delta, BASE_PRICE_RS
        )
        st.metric(
            label="Estimated Price Impact",
            value=f"₹{abs_chg / 1e5:.1f} L",
            delta=f"{pct_change:.2f}%"
        )
        st.caption(f"Adding a pooja room is estimated to change price by ~{pct_change:.1f}%.")

st.divider()



# Section 4 — Combined Upgrade Scenario

st.subheader("📈 Property Upgrade Scenario")
st.caption("Combine multiple upgrades and see their cumulative price impact.")

col1, col2, col3 = st.columns(3)

with col1:
    add_area = st.slider("Extra Area (sqft)", 0, 500, 0, step=50)

with col2:
    add_bedroom = st.checkbox("Add Bedroom", key="upgrade_bedroom")

with col3:
    add_servant = st.checkbox("Add Servant Room", key="upgrade_servant")

total_log_change = 0

if add_area > 0:
    total_log_change += COEFS["total_area_sqft"] * (add_area / STDS["total_area_sqft"])

if add_bedroom:
    total_log_change += COEFS["bedrooms"] * (1 / STDS["bedrooms"])

if add_servant:
    total_log_change += COEFS["servant_room"] * (1 / STDS["servant_room"])

if total_log_change != 0:
    price_multiplier = np.exp(total_log_change)
    pct_change       = (price_multiplier - 1) * 100
    abs_chg          = BASE_PRICE_RS * (price_multiplier - 1)

    st.metric(
        label="Total Upgrade Value",
        value=f"₹{abs_chg / 1e5:.1f} L",
        delta=f"{pct_change:.2f}%"
    )
    st.caption("Estimated combined impact of all selected upgrades at median market price.")

else:
    st.info("Select at least one upgrade above to see combined price impact.")

st.divider()



# Section 5 — Sector Price Premium Chart

st.subheader("🏆 Top 10 Most Expensive Sectors vs Market Average")
st.caption(
    "Sectors are ranked by price premium relative to the market average. "
    "A premium of 50% means properties in that sector cost 50% more than average."
)

sector_df   = build_premium_df(COEFS, "sector_", "sector")
top_sectors = sector_df.nlargest(10, "premium_%").sort_values("premium_%")

fig_sector = px.bar(
    top_sectors,
    x="premium_%",
    y="sector",
    orientation="h",
    labels={
        "premium_%": "Price Premium (%)",
        "sector"   : "Sector"
    }
)

fig_sector.update_layout(
    height=400,
    xaxis_title="Price Premium (%)",
    yaxis_title="Sector"
)

st.plotly_chart(fig_sector, use_container_width=True)

st.divider()



# Section 6 — Society Price Premium Chart

st.subheader("🏢 Top 10 Premium Societies vs Market Average")
st.caption(
    "Societies are ranked by price premium relative to the market average. "
    "Reflects the brand, amenity, and location value a society commands."
)

society_df   = build_premium_df(COEFS, "society_", "society")
top_societies = society_df.nlargest(10, "premium_%").sort_values("premium_%")

fig_society = px.bar(
    top_societies,
    x="premium_%",
    y="society",
    orientation="h",
    labels={
        "premium_%": "Price Premium (%)",
        "society"  : "Society"
    }
)

fig_society.update_layout(
    height=400,
    xaxis_title="Price Premium (%)",
    yaxis_title="Society"
)

st.plotly_chart(fig_society, use_container_width=True)



# Footer


st.divider()
st.caption(
    "🔍 All insights are estimates based on Ridge regression coefficients. "
    "Actual price impacts may vary based on specific property and market conditions."
)
