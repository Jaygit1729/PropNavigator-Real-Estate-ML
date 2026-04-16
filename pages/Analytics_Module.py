import os
import ast
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import streamlit as st


# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Gurgaon Real Estate Analytics",
    layout="wide"
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(BASE_DIR, "data", "analytics_module", "gurgaon_property_geomap.csv")
WC_PATH   = os.path.join(BASE_DIR, "data", "analytics_module", "wordcloud.csv")

# Minimum listings a sector must have to appear in price dispersion heatmap
MIN_LISTINGS_FOR_DISPERSION = 15

# Area segment bin edges and labels for Section 5
AREA_BINS   = [0, 1000, 2200, 3500, float("inf")]
AREA_LABELS = ["Small (<1000)", "Mid (1000–2200)", "Large (2200–3500)", "Luxury (3500+)"]

PROPERTY_TYPES = ["All", "Independent House", "Independent Builder Floor", "Flat"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_dataframe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_wordcloud_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)[["sector", "features_list"]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def filter_by_property_type(df: pd.DataFrame, property_type: str) -> pd.DataFrame:
    """Returns full df if 'All' selected, else filters by property type."""
    if property_type == "All":
        return df
    return df[df["property_type"] == property_type]


def add_area_segment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates area_segment column on a copy of df.
    Always operates on a copy — never mutates the cached original.
    """
    df = df.copy()
    df["area_segment"] = pd.cut(
        df["total_area_sqft"],
        bins=AREA_BINS,
        labels=AREA_LABELS
    )
    return df


def extract_amenity_words(wc_df: pd.DataFrame, sector: str) -> list:
    """
    Extracts all amenity words for a given sector (or all sectors).
    Returns a flat list of amenity strings.
    """
    filtered = wc_df if sector == "All" else wc_df[wc_df["sector"] == sector]
    words = []
    for lst in filtered["features_list"].dropna().apply(ast.literal_eval):
        words.extend(lst)
    return words


def get_sector_options(wc_df: pd.DataFrame) -> list:
    """Returns sorted list of sectors with 'All' as first option."""
    return ["All"] + sorted(wc_df["sector"].dropna().unique().tolist())


# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------

df    = load_dataframe(DATA_PATH)
wc_df = load_wordcloud_data(WC_PATH)


# ---------------------------------------------------------------------------
# UI — Header
# ---------------------------------------------------------------------------

st.header("📊 Gurgaon Real Estate Analytics")
st.caption(
    "A structured analytics module to understand **market structure, pricing behavior, "
    "configuration trends, and value dynamics** across Gurgaon's residential market."
)
st.divider()


# ---------------------------------------------------------------------------
# Section 1 — Market Structure: Property Mix by Sector
# ---------------------------------------------------------------------------

st.subheader("🏘️ Market Structure Overview")
st.caption("Shows **total supply and property-type composition** within each sector.")
st.info(
    "💡 **How to read this:** Longer bars indicate more active sectors. "
    "Color segments reveal whether a sector is flat-heavy, builder-floor dominant, or mixed."
)

sector_total = (
    df.groupby("sector")
    .size()
    .reset_index(name="total_listings")
)

sector_mix = (
    df.groupby(["sector", "property_type"])
    .size()
    .reset_index(name="listing_count")
    .merge(sector_total, on="sector")
)

sector_mix["pct_of_sector"] = (
    sector_mix["listing_count"] / sector_mix["total_listings"] * 100
).round(1)

fig_mix = px.bar(
    sector_mix,
    x="listing_count",
    y="sector",
    color="property_type",
    orientation="h",
    color_discrete_sequence=px.colors.qualitative.Bold,
    custom_data=["property_type", "total_listings", "pct_of_sector"],
    labels={
        "listing_count" : "Listings",
        "sector"        : "Sector",
        "property_type" : "Property Type"
    }
)

fig_mix.update_traces(
    hovertemplate=
    "<b>%{y}</b><br>"
    "Property Type: %{customdata[0]}<br>"
    "Listings: %{x}<br>"
    "Share of Sector: %{customdata[2]}%<br>"
    "Total Listings: %{customdata[1]}"
    "<extra></extra>"
)

fig_mix.update_layout(
    height=1500,
    yaxis={"categoryorder": "total ascending"},
    legend_title_text="Property Type"
)

st.plotly_chart(fig_mix, use_container_width=True)


# ---------------------------------------------------------------------------
# Section 2 — Price Dispersion Heatmap
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🔥 Sector Price Dispersion")
st.caption("Uses **IQR** to highlight **price volatility within sectors**.")
st.info(
    "💡 **How to read this:** Darker sectors indicate **high price uncertainty or mixed inventory**. "
    "Lighter sectors suggest **stable and predictable pricing**."
)

price_dispersion = (
    df.groupby("sector")
    .agg(
        p25=("price_per_sqft", lambda x: x.quantile(0.25)),
        p75=("price_per_sqft", lambda x: x.quantile(0.75)),
        listings=("price_per_sqft", "count")
    )
    .reset_index()
)

price_dispersion["iqr"] = price_dispersion["p75"] - price_dispersion["p25"]
price_dispersion = price_dispersion[
    price_dispersion["listings"] >= MIN_LISTINGS_FOR_DISPERSION
]

fig_disp = px.imshow(
    price_dispersion
        .sort_values("iqr", ascending=False)
        .set_index("sector")[["iqr"]],
    color_continuous_scale=px.colors.sequential.Magma,
    aspect="auto",
    labels={"color": "IQR (₹/sqft)", "y": "Sector", "x": ""}
)

fig_disp.update_layout(height=800)
st.plotly_chart(fig_disp, use_container_width=True)


# ---------------------------------------------------------------------------
# Section 3 — Bedroom × Property Type Configuration Heatmap
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🧩 Bedroom × Property Type Configuration")
st.caption("Normalized heatmap showing **dominant bedroom configurations** within each property type.")

bhk_df = df[df["bedrooms"].between(1, 6)]

bhk_matrix = (
    bhk_df
    .groupby(["property_type", "bedrooms"])
    .size()
    .reset_index(name="count")
)

bhk_matrix["pct"] = (
    bhk_matrix.groupby("property_type")["count"]
    .transform(lambda x: (x / x.sum()) * 100)
).round(1)

heatmap_df = bhk_matrix.pivot(
    index="property_type",
    columns="bedrooms",
    values="pct"
).fillna(0)

fig_bhk = px.imshow(
    heatmap_df,
    text_auto=True,
    aspect="auto",
    color_continuous_scale=px.colors.sequential.Viridis,
    labels={"color": "% of Listings"}
)

fig_bhk.update_layout(height=400)
st.plotly_chart(fig_bhk, use_container_width=True)


# ---------------------------------------------------------------------------
# Section 4 — Price Distribution by Property Type (Violin)
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📊 Price Distribution by Property Type")
st.caption("Violin plots compare **price density, spread, and outliers** across property types.")

fig_violin = px.violin(
    df,
    x="property_type",
    y="price_in_cr",
    color="property_type",
    box=True,
    points="outliers",
    color_discrete_sequence=px.colors.qualitative.Set2
)

fig_violin.update_layout(height=550, showlegend=False)
st.plotly_chart(fig_violin, use_container_width=True)


# ---------------------------------------------------------------------------
# Section 5 — Area Segment vs Price Efficiency
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📐 Area Segment vs Price Efficiency")
st.caption(
    "Shows **price-per-sqft behavior across home size segments**. "
    "Helps identify where buyers pay premiums vs get efficiency."
)

df_with_segments = add_area_segment(df)

fig_area = px.box(
    df_with_segments.dropna(subset=["area_segment"]),
    x="area_segment",
    y="price_per_sqft",
    color="area_segment",
    color_discrete_sequence=px.colors.qualitative.Set2
)

fig_area.update_layout(height=550, showlegend=False)
st.plotly_chart(fig_area, use_container_width=True)


# ---------------------------------------------------------------------------
# Section 6 — Geomap
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🔍 Listing Exploration using Geomap")

ptype_map = st.selectbox(
    "Select Property Type (Map)",
    PROPERTY_TYPES,
    key="ptype_map"
)

map_df = filter_by_property_type(df, ptype_map)

geo_df = (
    map_df.groupby("sector", as_index=False)
    .mean(numeric_only=True)[
        ["sector", "price_per_sqft", "total_area_sqft", "latitude", "longitude"]
    ]
)

st.caption(
    f"This map shows **average property prices and sizes** for **{ptype_map}** "
    "across Gurgaon sectors.\n\n"
    "- **Color** : Average Price per Sqft\n"
    "- **Bubble size** : Average Property Size (In SQFT)\n"
    "- Each point corresponds to a **sector-level aggregation**"
)

st.info(
    "💡 **How to read this map:** Darker sectors indicate higher average price per sqft. "
    "Larger circles represent sectors with bigger average property sizes."
)

fig_map = px.scatter_map(
    geo_df,
    lat="latitude",
    lon="longitude",
    color="price_per_sqft",
    size="total_area_sqft",
    size_max=20,
    zoom=11,
    map_style="open-street-map",
    color_continuous_scale=px.colors.cyclical.IceFire,
    hover_name="sector",
    hover_data={
        "latitude"      : False,
        "longitude"     : False,
        "price_per_sqft": ":.2f",
        "total_area_sqft": ":.0f"
    },
    labels={
        "price_per_sqft" : "Avg Price/Sqft",
        "total_area_sqft": "Avg Area (Sqft)"
    }
)

fig_map.update_layout(
    width=1000,
    height=600,
    margin=dict(l=0, r=0, t=30, b=0)
)

st.plotly_chart(fig_map, use_container_width=True)


# ---------------------------------------------------------------------------
# Section 7 — Amenities Word Cloud
# ---------------------------------------------------------------------------

st.divider()
st.subheader("☁️ Common Amenities by Sector")

sector_wc = st.selectbox(
    "Select Sector (Amenities)",
    get_sector_options(wc_df),
    key="sector_wc"
)

words = extract_amenity_words(wc_df, sector_wc)

st.caption(
    f"This word cloud highlights the **most commonly mentioned amenities** "
    f"for properties in **{sector_wc}**.\n\n"
    "- **Larger words** indicate amenities that appear more frequently\n"
    "- Aggregated across all properties in the selected sector"
)

st.info(
    "💡 **How to read this:** Amenities that appear larger are more prevalent in listings, "
    "indicating common facilities or selling points emphasized by builders."
)

if words:
    wc = WordCloud(
        width=800,
        height=500,
        background_color="white"
    ).generate(" ".join(words))

    st.image(wc.to_array(), use_container_width=True)

else:
    st.warning("No amenity data available for the selected sector.")


# ---------------------------------------------------------------------------
# Section 8 — Area vs Price Scatter
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🔎 Area vs Price — Micro Price Behavior")

ptype_scatter = st.selectbox(
    "Select Property Type (Scatter)",
    PROPERTY_TYPES,
    key="ptype_scatter"
)

scatter_df = filter_by_property_type(df, ptype_scatter)

st.caption(
    "This scatter plot shows **how price scales with size**, while highlighting "
    "**bedroom-driven clustering and anomalies**."
)

st.info(
    "💡 **How to read this:** Each point is a listing. Vertical stacks indicate standardized layouts, "
    "while isolated points signal **overpriced or luxury outliers**."
)

fig_scatter = px.scatter(
    scatter_df,
    x="total_area_sqft",
    y="price_in_cr",
    color="bedrooms",
    color_continuous_scale=px.colors.sequential.Viridis,
    opacity=0.75,
    labels={
        "total_area_sqft": "Total Area (Sqft)",
        "price_in_cr"    : "Price (Cr)",
        "bedrooms"       : "Bedrooms"
    }
)

fig_scatter.update_layout(
    height=600,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)"
)

fig_scatter.update_xaxes(showgrid=False)
fig_scatter.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")

st.plotly_chart(fig_scatter, use_container_width=True)


# ---------------------------------------------------------------------------
# Section 9 — Bedroom-wise Price Distribution
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📦 Price Distribution by Bedroom Configuration")

ptype_box = st.selectbox(
    "Select Property Type (Bedroom Pricing)",
    PROPERTY_TYPES,
    key="ptype_box"
)

box_df = filter_by_property_type(df, ptype_box)
box_df = box_df[box_df["bedrooms"].between(1, 4)]

st.caption(
    "Shows **price spread and median behavior** across bedroom configurations. "
    "Useful to identify **sweet spots vs diminishing returns**."
)

st.info(
    "💡 **How to read this:** If price jumps sharply from 2→3 BHK but flattens after, "
    "it signals **demand saturation** beyond a configuration."
)

fig_box = px.box(
    box_df,
    x="bedrooms",
    y="price_in_cr",
    color="bedrooms",
    color_discrete_sequence=px.colors.qualitative.Set2,
    labels={
        "bedrooms"   : "Bedrooms",
        "price_in_cr": "Price (Cr)"
    }
)

fig_box.update_layout(height=500, showlegend=False)
st.plotly_chart(fig_box, use_container_width=True)


# ---------------------------------------------------------------------------
# Section 10 — Bedroom Distribution Pie Chart
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🛏️ Bedroom Distribution Across Listings")

st.caption(
    "This chart shows the **distribution of property listings by bedroom count**. "
    "Helps identify which unit types dominate the market."
)

st.info(
    "💡 **How to read this:** A market dominated by 2–3 BHK units typically reflects "
    "end-user demand, while higher counts indicate premium or luxury segments."
)

sector_pie = st.selectbox(
    "Select Sector (Bedroom Distribution)",
    get_sector_options(wc_df),
    key="sector_pie"
)

pie_df = filter_by_property_type(
    df if sector_pie == "All" else df[df["sector"] == sector_pie],
    "All"
)

fig_pie = px.pie(
    data_frame=pie_df,
    names="bedrooms",
    hole=0.4,
    labels={"bedrooms": "Bedrooms"}
)

fig_pie.update_layout(height=500, legend_title_text="Bedrooms")
st.plotly_chart(fig_pie, use_container_width=True)