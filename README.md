🏡 Prop Navigator

Real Estate Decision Intelligence Platform

📌 Overview

Prop Navigator is an end-to-end real estate analytics platform designed to transform raw property listings into structured insights that support informed property decisions.

Most real estate marketplaces such as 99acres and Magicbricks primarily function as listing platforms. They allow users to search properties using filters like price, number of bedrooms (BHK), or property type. However, these platforms largely stop at displaying information and offer limited analytical support for understanding the market.

Prop Navigator addresses this gap by integrating data engineering, analytics, machine learning, and recommendation systems into a unified platform.

Instead of simply displaying property listings, the platform:

Analyzes market trends

Predicts reasonable price ranges

Recommends similar properties

Explains key factors influencing property pricing

🎯 Objective:

Make real estate decisions more structured, transparent, and data-driven rather than purely intuition-based.

🚀 Key Features

📊 1. Market Analytics Module

The Analytics Module helps users understand the broader real estate market rather than viewing listings in isolation.

It provides insights such as:

📍 Sector-wise price distribution

📐 Relationship between area and price

🏢 Bedroom configuration trends

⭐ Frequency of amenities across properties

💰 Identification of premium vs affordable sectors

This module transforms raw listing data into meaningful market insights.

💰 2. Price Prediction Module

The Price Prediction Module estimates a reasonable price range for a property based on its key attributes:

Location (Sector)

Total Area

Number of Bedrooms

Furnishing Level

Property Type

Amenities

A regression-based machine learning model is trained on historical listing data to generate price estimates.

Users can adjust property attributes and observe how predicted prices change accordingly.

🔍 3. Insight Module (Explainable Pricing)

The Insight Module explains why property prices vary.

Instead of only showing a predicted price, it provides interpretable insights such as:

📈 How price changes when area increases

🏠 Impact of moving from 2BHK to 3BHK

📍Top 10 Premium Sector/Socierties


🧩 Impact of additional rooms and utilities

These insights are derived using regression coefficients and feature-level analysis, making the pricing process transparent.

🔎 4. Recommendation Module

The Recommendation Module helps users discover similar properties based on a selected listing.

Comparable societies are identified using similarity across:

Price range

Property size

Configuration

Amenities

Location

This allows users to explore relevant alternatives without manually browsing through multiple listings.

🗂 Data Collection

The dataset for this project was collected using a custom-built web scraping pipeline.

Two-Layer Scraping Architecture

Layer 1 – Listing Scraping

Extracts summary information from listing pages:

Property links

Basic property attributes

Listing-level details

Layer 2 – Property Detail Scraping

Visits individual property pages and extracts detailed attributes:

Bedrooms

Bathrooms

Facing direction

Furnishing details

Amenities

Floor information

To ensure reliability:

Batch processing was implemented

Retry logic handled page failures

Cooldown intervals prevented blocking

📊 Final Dataset

Approximately 6,000 residential properties across multiple property types:

Flats

Independent Houses

Independent Builder Floors

⚙️ Feature Engineering

Several domain-driven features were engineered to improve model performance.

📍 Sector Standardization

Sector names across listings were inconsistent.

350+ raw sector labels

Consolidated into 102 standardized sectors

This reduced noise and improved model generalization.

📐 Area Consolidation

Listings contained multiple area measurements:

Super Built-Up Area

Built-Up Area

Carpet Area

Plot Area

These were consolidated into a single feature:

total_area_sqft

Property-type logic and fallback strategies were applied to select the most reliable area measurement.

⭐ Luxury Score

A custom Luxury Score was developed to quantify the impact of amenities on property valuation.

Initial Approach

Amenities were weighted based on rarity.

Issue

EDA revealed that rarity did not always correspond to pricing impact.

Final Approach

A price-driven weighting strategy was implemented:

Each amenity weight is based on its observed pricing premium

Amenities contributing higher price-per-sqft receive higher weight

This aligns the feature with real market behavior.

📊 Exploratory Data Analysis

Extensive EDA was performed to understand pricing patterns.

Key analyses included:

Price distribution analysis (skewness and kurtosis)

Outlier detection

Quantile analysis

Price-per-square-foot comparison

Feature impact validation

Multivariate interaction analysis

Log transformation was applied to stabilize the price distribution and improve model performance.

🛠 Technology Stack

Programming

Python


Data Processing

Pandas

NumPy

Machine Learning

Scikit-learn

Web Scraping

Selenium

Visualization

Plotly

Matplotlib

Seaborn

Deployment

Streamlit

💻 Application Interface

The platform is deployed as a multi-page Streamlit application that includes:

📊 Analytics Dashboard

💰 Price Prediction Interface

🔎 Recommendation Engine

📈 Pricing Insight Module

Users can explore the market, estimate property prices, and discover similar properties within a single interactive interface.

👨‍💻 Author

Jay Patel
Data Science & Analytics Professional

GitHub:
https://github.com/Jaygit1729

LinkedIn:
https://www.linkedin.com/in/jay-patel1729
