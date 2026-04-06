import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# 1. Page Configuration

st.set_page_config(page_title="PropNavigator | Recommend Apartments", layout="wide")

#

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RECOMMENDER_PATH = os.path.join(
    BASE_DIR,
    "data",
    "recommender"
)


# 2. Cached Data Loading

@st.cache_resource
def load_data():

    base_path = RECOMMENDER_PATH

    # Loading DataFrames

    location_df = pickle.load(open(os.path.join(base_path, "location_distance.pkl"), "rb"))
    price_df = pickle.load(open(os.path.join(base_path, "price_df.pkl"), "rb"))

    # Loading Similarity Matrices

    sim_features = pickle.load(open(os.path.join(base_path, "cosine_sim_top_features.pkl"), "rb"))
    sim_price = pickle.load(open(os.path.join(base_path, "cosine_sim_price.pkl"), "rb"))
    sim_location = pickle.load(open(os.path.join(base_path, "cosine_sim_location.pkl"), "rb"))

    return location_df, price_df, sim_features, sim_price, sim_location

location_df, price_df, cosine_sim_top_features, cosine_sim_price, cosine_sim_location = load_data()

# 3. Initialize Session State for Persistence

if 'recommendations' not in st.session_state:
    st.session_state['recommendations'] = None
if 'selected_property' not in st.session_state:
    st.session_state['selected_property'] = None

# 4. Recommendation Logic

def recommend_properties_with_scores(property_name, top_n=10):
    
    # Weighted Similarity Matrix: Facilities (30), Price (20), Location (8)
    
    cosine_sim_matrix = (30 * cosine_sim_top_features + 
                         20 * cosine_sim_price + 
                         8 * cosine_sim_location)
    
    idx = location_df.index.get_loc(property_name)
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    recom_data = []
    for i, score in sorted_scores:
        prop_name = location_df.index[i]
        avg_price = price_df.loc[prop_name, 'avg_price_cr']
        avg_area = price_df.loc[prop_name, 'avg_area_sqft'] 
        min_bhk = price_df.loc[prop_name, 'min_bhk']
        max_bhk = price_df.loc[prop_name, 'max_bhk']
        
        
        recom_data.append({
            'PropertyName': prop_name,
            'SimilarityScore': round(score, 2),
            'AvgPrice': round(avg_price, 2),
            'AvgArea': round(avg_area),
            'MinBHK': min_bhk,
            'MaxBHK': max_bhk
        })
    
    return pd.DataFrame(recom_data)


st.title('🏢 PropNavigator: Apartment Recommender')

selected_appartment = st.selectbox(
    'Search for an Apartment:',
    sorted(location_df.index.to_list())
)

if st.button('Find Similar Apartments', type="primary"):

    st.session_state['recommendations'] = recommend_properties_with_scores(selected_appartment)
    st.session_state['selected_property'] = selected_appartment

st.divider()

# --- DYNAMIC RESULTS DISPLAY ---

if st.session_state['recommendations'] is not None:
    recom_df = st.session_state['recommendations']
    st.subheader(f"Top Matches for {st.session_state['selected_property']}")

    # Image of a property recommendation grid layout

    for i in range(0, len(recom_df), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(recom_df):
                row = recom_df.iloc[i + j]
                with cols[j]:
                    with st.container(border=True):
                        st.write(f"### {row['PropertyName']}")
                        st.metric("Similarity Score", row['SimilarityScore'])
                        
                        # Populate Area Range and Average BHK instead of Min Area

                        st.write(f"💰 **Avg Price:** ₹{row['AvgPrice']} Cr")
                        st.write(f"📐 **Avg Area:** {row['AvgArea']} sq.ft.")
                        st.write(f"🛏️ **Min BHK:** {row['MinBHK']}")
                        st.write(f"🛏️ **Max BHK:** {row['MaxBHK']}")
                        
                      