import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np


# Page Config

st.set_page_config(
    page_title="PropNavigator | Apartment Recommender",
    layout="wide"
)


# Constants

BASE_DIR         = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RECOMMENDER_PATH = os.path.join(BASE_DIR, "data", "recommender")

# Similarity engine weights — must match what was used in the notebook
WEIGHTS = {"facilities": 30, "price": 20, "location": 8}


# Loaders

@st.cache_data(show_spinner=False)
def load_pickle(filepath: str):
    """Loads a pickle file safely using context manager."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_all_artifacts(path: str):
    """
    Loads all recommender artifacts from disk.
    Cached so disk reads only happen once per session.
    Returns: (location_df, price_df, sim_features, sim_price, sim_location)
    """
    location_df  = load_pickle(os.path.join(path, "location_distance.pkl"))
    price_df     = load_pickle(os.path.join(path, "price_df.pkl"))
    sim_features = load_pickle(os.path.join(path, "cosine_sim_top_features.pkl"))
    sim_price    = load_pickle(os.path.join(path, "cosine_sim_price.pkl"))
    sim_location = load_pickle(os.path.join(path, "cosine_sim_location.pkl"))

    return location_df, price_df, sim_features, sim_price, sim_location


@st.cache_data(show_spinner=False)
def build_combined_matrix(
    sim_features: np.ndarray,
    sim_price   : np.ndarray,
    sim_location: np.ndarray,
    w_features  : int,
    w_price     : int,
    w_location  : int
):
    """
    Precomputes the weighted combined similarity matrix once at startup.
    Never recomputed per request — cached for the session.

    Weights: Facilities (30) > Price (20) > Location (8)
    Facilities weighted highest — amenity profile is the strongest
    signal for lifestyle similarity in real estate.
    """
    return (
        w_features * sim_features +
        w_price    * sim_price    +
        w_location * sim_location
    )


# Recommendation Logic

def get_recommendations(
    property_name  : str,
    location_df    : pd.DataFrame,
    price_df       : pd.DataFrame,
    combined_matrix: np.ndarray,
    top_n          : int = 10
):
    """
    Returns top_n most similar properties to the selected property.
    Uses precomputed combined similarity matrix — no recomputation per call.
    """
    idx        = location_df.index.get_loc(property_name)
    sim_scores = list(enumerate(combined_matrix[idx]))
    top_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

    results = []
    for i, score in top_scores:
        prop_name = location_df.index[i]

        # Safely retrieve price data — property may have NaN values
        avg_price = price_df.loc[prop_name, "avg_price_cr"] if prop_name in price_df.index else np.nan
        avg_area  = price_df.loc[prop_name, "avg_area_sqft"] if prop_name in price_df.index else np.nan
        min_bhk   = price_df.loc[prop_name, "min_bhk"] if prop_name in price_df.index else np.nan
        max_bhk   = price_df.loc[prop_name, "max_bhk"] if prop_name in price_df.index else np.nan

        results.append({
            "PropertyName"   : prop_name,
            "SimilarityScore": round(score, 2),
            "AvgPrice"       : round(avg_price, 2) if not np.isnan(avg_price) else "N/A",
            "AvgArea"        : int(avg_area) if not np.isnan(avg_area) else "N/A",
            "MinBHK"         : int(min_bhk) if not np.isnan(min_bhk) else "N/A",
            "MaxBHK"         : int(max_bhk) if not np.isnan(max_bhk) else "N/A"
        })

    return pd.DataFrame(results)


# Load Artifacts & Precompute

location_df, price_df, sim_features, sim_price, sim_location = load_all_artifacts(
    RECOMMENDER_PATH
)

combined_matrix = build_combined_matrix(
    sim_features,
    sim_price,
    sim_location,
    WEIGHTS["facilities"],
    WEIGHTS["price"],
    WEIGHTS["location"]
)


# Session State — persists recommendations across rerenders

if "recommendations" not in st.session_state:
    st.session_state["recommendations"] = None

if "selected_property" not in st.session_state:
    st.session_state["selected_property"] = None


# UI — Header

st.title("🏢 PropNavigator: Apartment Recommender")
st.caption(
    "Discover similar properties based on amenities, price profile, and location. "
    "Select an apartment below and find your best matches."
)
st.divider()



# UI — Search

selected_apartment = st.selectbox(
    "🔍 Search for an Apartment",
    sorted(location_df.index.tolist()),
    help="Select a property to find similar apartments"
)

st.caption(
    f"Similarity is computed across three dimensions — "
    f"**Amenities** (weight: {WEIGHTS['facilities']}), "
    f"**Price Profile** (weight: {WEIGHTS['price']}), "
    f"**Location** (weight: {WEIGHTS['location']}). "
    f"Higher similarity score = more similar overall profile."
)

if st.button("🔎 Find Similar Apartments", type="primary", use_container_width=True):
    with st.spinner("Finding similar properties..."):
        st.session_state["recommendations"] = get_recommendations(
            property_name  =selected_apartment,
            location_df    =location_df,
            price_df       =price_df,
            combined_matrix=combined_matrix
        )
        st.session_state["selected_property"] = selected_apartment

st.divider()


# UI — Results

if st.session_state["recommendations"] is not None:
    recom_df      = st.session_state["recommendations"]
    selected_prop = st.session_state["selected_property"]

    st.subheader(f"Top {len(recom_df)} Matches for **{selected_prop}**")

    st.caption(
        "💡 **Similarity Score** reflects combined similarity across amenities, "
        "price, and location. Scores are relative — compare across results, "
        "not as absolute percentages."
    )

    # Render in a 2-column card grid
    for i in range(0, len(recom_df), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(recom_df):
                row = recom_df.iloc[i + j]
                with cols[j]:
                    with st.container(border=True):

                        st.markdown(f"### {row['PropertyName']}")

                        # Similarity score with context
                        st.metric(
                            label="Similarity Score",
                            value=row["SimilarityScore"],
                            help="Combined score across amenities, price and location"
                        )

                        st.markdown("---")

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write(f"💰 **Avg Price**")
                            price_display = (
                                f"₹{row['AvgPrice']} Cr"
                                if row["AvgPrice"] != "N/A"
                                else "Price on Request"
                            )
                            st.write(price_display)

                            st.write(f"📐 **Avg Area**")
                            area_display = (
                                f"{row['AvgArea']} sqft"
                                if row["AvgArea"] != "N/A"
                                else "N/A"
                            )
                            st.write(area_display)

                        with col_b:
                            st.write(f"🛏️ **BHK Range**")
                            bhk_display = (
                                f"{row['MinBHK']} – {row['MaxBHK']} BHK"
                                if row["MinBHK"] != "N/A"
                                else "N/A"
                            )
                            st.write(bhk_display)



# Footer

st.divider()
st.caption(
    "🔍 Recommendations are based on property features, price profile, and "
    "proximity to landmarks — not on user behaviour or transaction history."
)