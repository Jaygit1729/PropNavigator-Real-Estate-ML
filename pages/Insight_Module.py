import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import joblib
import shap


# Page Config

st.set_page_config(page_title="PropNavigator | Pricing Insights", layout="wide")


# Constants

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "best_model.joblib")
DATA_DIR   = os.path.join(BASE_DIR, "data", "insight_module")
FS_PATH    = os.path.join(BASE_DIR, "data", "fs", "feature_selected_properties.csv")
TARGET     = "price_in_cr"
BASE_MODEL = "LightGBM"

FEATURES = [
    'area', 'dist_to_golf_road', 'total_floor', 'bathroom', 'property_type',
    'dist_to_cyber_city', 'society', 'covered_parking', 'dist_to_manesar', 'sector',
    'dist_to_airport', 'bedRoom', 'furnishing', 'balcony', 'age_possession_category',
    'facing', 'has_ac', 'total_parking', 'open_parking', 'has_power_backup',
    'is_corner', 'floornum_category', 'ov_main_road', 'has_pool', 'ov_others',
]
DIST = ['dist_to_cyber_city', 'dist_to_golf_road', 'dist_to_airport', 'dist_to_manesar']
PROPERTY_TYPES = ['Flat', 'Independent Builder Floor', 'Independent House']
FURNISHING     = ['unfurnished', 'semi-furnished', 'furnished']   # no 'unknown' shown to user
CONST          = dict(facing='unknown', open_parking=0.0, ov_main_road=0, ov_others=0)


# SHAP runtime (self-contained)

@dataclass
class ExplainerBundle:
    explainer: object
    preprocessor: object
    feat_names: list
    base_value: float


def build_explainer(pipeline) -> ExplainerBundle:
    # Deployed model is a single LightGBM Pipeline (preprocessor + regressor),
    # so SHAP's TreeExplainer runs directly on its regressor.
    pre = pipeline.named_steps["preprocessor"]; reg = pipeline.named_steps["regressor"]
    feat_names = [f.split("__")[-1] for f in pre.get_feature_names_out()]
    expl = shap.TreeExplainer(reg)
    return ExplainerBundle(expl, pre, feat_names, float(np.ravel(expl.expected_value)[0]))


def explain_row(bundle: ExplainerBundle, row_df: pd.DataFrame):
    X_ord = row_df[bundle.feat_names]
    sv = bundle.explainer.shap_values(bundle.preprocessor.transform(row_df))[0]
    contrib = (pd.DataFrame({"feature": bundle.feat_names, "value": X_ord.iloc[0].values,
                             "pct_effect": (np.exp(sv) - 1) * 100})
                 .reindex(np.argsort(np.abs(sv))[::-1]).reset_index(drop=True))
    return contrib, float(np.expm1(bundle.base_value + sv.sum()))


# Feature assembly + prediction (mirrors notebooks/insight_module/insight_module.ipynb)

def build_input_row(inp, ref, type_ref):
    r = ref.loc[ref['sector'] == inp['sector']]
    tr = type_ref.loc[type_ref['property_type'] == inp['property_type']]
    tr = tr.iloc[0] if not tr.empty else type_ref.iloc[0]
    covered = float(inp.get('covered_parking', 1))
    total_floor = float(inp['total_floor']) if inp.get('total_floor') is not None else float(tr['total_floor'])
    row = {
        'area': float(inp['area']), 'total_floor': total_floor,
        'bathroom': int(inp['bathroom']), 'property_type': inp['property_type'],
        'society': inp.get('society') or (r['society'].iloc[0] if not r.empty else 'other'),
        'covered_parking': covered, 'open_parking': CONST['open_parking'],
        'sector': inp['sector'], 'bedRoom': int(inp['bedRoom']),
        'furnishing': inp.get('furnishing', 'semi-furnished'), 'balcony': float(tr['balcony']),
        'age_possession_category': inp.get('age_possession_category', 'New Property'),
        'facing': CONST['facing'], 'floornum_category': tr['floornum_category'],
    }
    row['total_parking'] = covered + CONST['open_parking']
    for flag in ['has_ac', 'has_power_backup', 'is_corner', 'has_pool']:
        row[flag] = int(inp.get(flag, 0))
    row['ov_main_road'] = CONST['ov_main_road']; row['ov_others'] = CONST['ov_others']
    for d in DIST:
        row[d] = float(r[d].iloc[0]) if not r.empty else np.nan
    return pd.DataFrame([row])[FEATURES]


def predict_many(pipeline, inps, ref, type_ref):
    batch = pd.concat([build_input_row(i, ref, type_ref) for i in inps], ignore_index=True)
    return np.expm1(pipeline.predict(batch))


# Loaders

@st.cache_resource(show_spinner=False)
def load_model_and_explainer():
    bundle = joblib.load(MODEL_PATH)
    return bundle, build_explainer(bundle["pipeline"])


@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path)


bundle, explainer = load_model_and_explainer()
pipeline   = bundle["pipeline"]
model_name = bundle["model_name"]
mape_pct   = bundle["test_mape_percent"]

sector_ref  = load_csv(os.path.join(DATA_DIR, "sector_reference.csv"))
type_ref    = load_csv(os.path.join(DATA_DIR, "type_reference.csv"))
sec_prem    = load_csv(os.path.join(DATA_DIR, "sector_premiums.csv"))
soc_prem    = load_csv(os.path.join(DATA_DIR, "society_premiums.csv"))
global_imp  = load_csv(os.path.join(DATA_DIR, "shap_global_importance.csv"))
sectors     = sorted(sector_ref["sector"].tolist())


# UI — Header

st.title("📈 PropNavigator: Pricing Insights")
st.caption(
    "Understand and simulate what drives Gurgaon prices. Every number is computed by "
    f"**re-running the deployed model** ({model_name}, ~{mape_pct:.1f}% MAPE) — not a proxy."
)
st.divider()


# Section 1 — Price Impact Simulator

st.subheader("🧪 Price Impact Simulator")
st.caption("Set a baseline property, then see how specific upgrades change its price — model-true "
           "**what-if** analysis (baseline vs modified prediction).")

c1, c2 = st.columns(2)
with c1:
    b_property_type = st.selectbox("Property Type", PROPERTY_TYPES, key="sim_pt")
    b_sector = st.selectbox("Sector", sectors,
                            index=sectors.index("sector 49") if "sector 49" in sectors else 0, key="sim_sec")
    b_area = st.number_input("Area (sqft)", 200.0, 27000.0, 1500.0, 50.0, key="sim_area")
with c2:
    b_bed  = st.selectbox("Bedrooms", list(range(1, 11)), index=2, key="sim_bed")
    b_bath = st.selectbox("Bathrooms", list(range(1, 11)), index=1, key="sim_bath")
    b_furn = st.selectbox("Furnishing", FURNISHING, index=1, key="sim_furn")

# baseline = just the essentials; floors / balcony / facing are derived automatically
baseline = dict(property_type=b_property_type, sector=b_sector, area=b_area, bedRoom=b_bed,
                bathroom=b_bath, furnishing=b_furn, covered_parking=1, has_ac=1, has_power_backup=1)

scenarios = {
    "+500 sqft area":        {"area": b_area + 500},
    "+1 BHK (+bath)":        {"bedRoom": b_bed + 1, "bathroom": b_bath + 1},
    "Add swimming pool":     {"has_pool": 1},
    "Corner property":       {"is_corner": 1},
    "+1 covered parking":    {"covered_parking": 2},
    "Upgrade to furnished":  {"furnishing": "furnished"},
    "Under construction":    {"age_possession_category": "Under Construction"},
}
inps = [baseline] + [{**baseline, **chg} for chg in scenarios.values()]
prices = predict_many(pipeline, inps, sector_ref, type_ref)
base_price, scen_prices = prices[0], prices[1:]

st.metric("Baseline estimated price", f"₹{base_price:.2f} Cr")
sim = pd.DataFrame({
    "Change": list(scenarios.keys()),
    "New Price (Cr)": np.round(scen_prices, 2),
    "Δ (%)": np.round((scen_prices / base_price - 1) * 100, 1),
}).sort_values("Δ (%)")

fig_sim = px.bar(sim, x="Δ (%)", y="Change", orientation="h", color="Δ (%)",
                 color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                 labels={"Δ (%)": "Price impact vs baseline (%)", "Change": ""})
fig_sim.update_layout(height=400, coloraxis_showscale=False)
st.plotly_chart(fig_sim, use_container_width=True)
st.caption("📖 **How to read:** each bar is the % change in the estimated price if you make that one "
           "change to your baseline property — bars to the **right (green) add value**, bars to the "
           "**left (red) reduce it**. Impacts are non-linear (from the real model); amenity toggles "
           "barely move, since amenities cluster by property type (Simpson's paradox).")

st.divider()


# Section 2 & 3 — Premium rankings

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🏆 Top Sectors by Premium")
    st.caption("The *same* property priced in every sector, vs the median sector.")
    top_sec = sec_prem.head(12).sort_values("premium_pct")
    fig = px.bar(top_sec, x="premium_pct", y="sector", orientation="h", color="premium_pct",
                 color_continuous_scale="Tealgrn", labels={"premium_pct": "Premium vs median (%)", "sector": ""})
    fig.update_layout(height=460, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("🏢 Top Societies by Premium")
    st.caption("Brand/quality premium of each society for the same unit.")
    top_soc = soc_prem.head(12).sort_values("premium_pct")
    fig = px.bar(top_soc, x="premium_pct", y="society", orientation="h", color="premium_pct",
                 color_continuous_scale="Purp", labels={"premium_pct": "Premium vs median (%)", "society": ""})
    fig.update_layout(height=460, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

st.divider()


# Section 4 — SHAP: global drivers + why this price

st.subheader("🌍 What the Model Relies On (SHAP)")
st.caption("Average influence of each feature on the model's price (mean |SHAP|).")
top_global = global_imp.head(15).sort_values("mean_abs_shap")
fig_g = px.bar(top_global, x="mean_abs_shap", y="feature", orientation="h", color="mean_abs_shap",
               color_continuous_scale="Blues", labels={"mean_abs_shap": "Mean |SHAP| (log-price)", "feature": ""})
fig_g.update_layout(height=520, coloraxis_showscale=False)
st.plotly_chart(fig_g, use_container_width=True)

st.markdown("#### 🔍 Why this price? — explain a real listing")
st.caption("Pick a listing to see the additive, model-true breakdown of its predicted price.")

sample = load_csv(FS_PATH).sample(n=300, random_state=7).reset_index(drop=True)
sample["label"] = (sample["sector"].astype(str) + " · " + sample["area"].round(0).astype(int).astype(str)
                   + " sqft · " + sample["bedRoom"].astype(int).astype(str) + " BHK · ₹"
                   + sample[TARGET].round(2).astype(str) + " Cr")
choice = st.selectbox("Select a listing", sorted(sample["label"].tolist()))
row = sample.loc[sample["label"] == choice].iloc[[0]]
contrib, predicted = explain_row(explainer, row[FEATURES])

m1, m2 = st.columns(2)
m1.metric("Model estimate", f"₹{predicted:.2f} Cr")
m2.metric("Actual listed price", f"₹{float(row[TARGET].iloc[0]):.2f} Cr")

plot_df = contrib.head(10).copy()
plot_df["label"] = plot_df["feature"] + "  (" + plot_df["value"].astype(str) + ")"
fig_l = px.bar(plot_df.sort_values("pct_effect"), x="pct_effect", y="label", orientation="h",
               color="pct_effect", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
               labels={"pct_effect": "Price effect (%)", "label": ""})
fig_l.update_layout(height=440, coloraxis_showscale=False)
st.plotly_chart(fig_l, use_container_width=True)


# Footer

st.divider()
st.caption(f"🔍 All figures are computed live from the deployed stacking model "
           f"(test MAPE ≈ {mape_pct:.1f}%). Estimates reflect learned patterns, not causal guarantees.")
