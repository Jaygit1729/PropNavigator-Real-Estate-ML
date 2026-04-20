import streamlit as st


# Page Config

st.set_page_config(
    page_title="PropNavigator",
    page_icon="🏡",
    layout="wide"
)


# Header

st.title("🏡 PropNavigator")
st.subheader("Real Estate Decision Intelligence Platform for Gurgaon")

st.markdown(
    """
    Most real estate platforms are listing tools — they let you filter and browse.
    **PropNavigator goes further.** It transforms raw property data into structured
    market intelligence, helping homebuyers and investors make decisions that are
    grounded in data rather than intuition.
    """
)

st.divider()


# Module Overview

st.markdown("### 🧩 What's Inside")

col1, col2, col3, col4 = st.columns(4)

with col1:
    with st.container(border=True):
        st.markdown("#### 📊 Analytics")
        st.caption(
            "Understand the market before you search. "
            "Explore sector-level pricing, supply distribution, "
            "price volatility, and amenity trends across Gurgaon."
        )

with col2:
    with st.container(border=True):
        st.markdown("#### 💰 Price Prediction")
        st.caption(
            "Get an ML-powered price estimate for any property "
            "configuration — sector, area, BHK, furnishing, and more. "
            "Includes a confidence range based on model accuracy."
        )

with col3:
    with st.container(border=True):
        st.markdown("#### 🔎 Recommendations")
        st.caption(
            "Discover similar properties based on amenity profile, "
            "price band, and location proximity — not just filters. "
            "Built on a three-engine cosine similarity system."
        )

with col4:
    with st.container(border=True):
        st.markdown("#### 📈 Insights")
        st.caption(
            "Understand what drives prices. Quantify the impact of "
            "adding a bedroom, increasing area, or choosing a premium "
            "sector — derived from Ridge regression coefficients."
        )

st.divider()


# How It Was Built

st.markdown("### ⚙️ How It Was Built")

st.markdown(
    """
    PropNavigator is built on a complete end-to-end ML pipeline — from raw web scraping
    to a deployed interactive application. Every stage was designed with production
    engineering principles in mind.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Data Collection**")
    st.caption(
        "Custom two-layer Selenium scraper built with undetected-chromedriver "
        "to bypass bot detection on 99acres. Batch-based persistence strategy "
        "ensured no data loss during multi-hour scraping sessions."
    )

    st.markdown("**Feature Engineering**")
    st.caption(
        "Domain-driven features including sector standardization across 350+ raw labels, "
        "area consolidation across multiple measurement types, and a price-driven "
        "luxury score computed from amenity pricing premiums."
    )

    st.markdown("**Furnishing Classification**")
    st.caption(
        "KMeans clustering on 18 furnishing item counts to classify properties "
        "as Furnished, Semi-Furnished, or Unfurnished — validated using "
        "silhouette analysis and domain knowledge."
    )

with col2:
    st.markdown("**Model Building**")
    st.caption(
        "Three models evaluated — RandomForest, XGBoost, SVR — with log1p "
        "target transformation and stratified train-test split. Hyperparameter "
        "tuning via RandomizedSearchCV. Best model: SVR at 13.84% MAPE."
    )

    st.markdown("**Recommendation Engine**")
    st.caption(
        "Content-based system combining three cosine similarity engines — "
        "TF-IDF on amenities, summarized price profiles, and normalized "
        "landmark distances — with weighted combination (30 / 20 / 8)."
    )

    st.markdown("**Pricing Insights**")
    st.caption(
        "Ridge regression trained on the same 15 features as the production SVR "
        "to generate interpretable coefficients. Used to quantify price impact "
        "of area changes, bedroom upgrades, and utility room additions."
    )

st.divider()


# Tech Stack

st.markdown("### 🛠️ Tech Stack")

t1, t2, t3, t4 = st.columns(4)

with t1:
    st.markdown("**Data**")
    st.caption("Python · Pandas · NumPy")

with t2:
    st.markdown("**ML**")
    st.caption("Scikit-learn · XGBoost · KMeans")

with t3:
    st.markdown("**Scraping**")
    st.caption("Selenium · undetected-chromedriver")

with t4:
    st.markdown("**App**")
    st.caption("Streamlit · Plotly · WordCloud")

st.divider()


# Call to Action


st.info(
    "👈 Use the sidebar to navigate between modules. "
    "Start with **Analytics** to understand the market, "
    "then use **Price Prediction** to estimate a specific property."
)