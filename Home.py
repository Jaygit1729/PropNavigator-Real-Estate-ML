import streamlit as st

st.set_page_config(page_title="PropNavigator", layout="wide")

st.title("🏡 PropNavigator")
st.subheader("AI-Powered Real Estate Decision Intelligence Platform")

st.markdown("""
PropNavigator is an end-to-end analytics and machine learning platform 
designed to help homebuyers and investors make data-driven property decisions in Gurgaon.

It combines **market analytics, price prediction, similarity-based recommendations, 
and explainable insights** into a single interactive web application.
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Analytics Module")
    st.write("Explore sector-wise pricing trends, area vs price relationships, and distribution insights.")

    st.markdown("### 🔮 Price Prediction")
    st.write("Predict property price ranges based on sector, BHK, area, and amenities.")

with col2:
    st.markdown("### 🤝 Recommendation Engine")
    st.write("Discover similar societies using feature similarity and engineered attributes.")

    st.markdown("### 📈 Insight Engine")
    st.write("Understand price sensitivity — BHK upgrades, area increases, servant room impact, and more.")

st.divider()

st.markdown("### 🛠 Tech Stack")
st.write("""
- Python  
- Pandas & Scikit-learn  
- Feature Engineering  
- Regression Modeling  
- Similarity-Based Recommendation  
- Streamlit Deployment  
""")

st.success("👉 Use the sidebar to explore modules and start analyzing properties!")