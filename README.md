# PropNavigator: Real Estate Decision-Support Platform

Listing sites show properties but don't help buyers understand the market.
PropNavigator is a decision-support platform for the Gurgaon property market,
built on ~39,800 scraped listings.

**Live app:** https://propnavigator-web.onrender.com
(free hosting: the first visit can take about a minute to wake up)

## What it does
| Module | Buyer question | How |
|---|---|---|
| Analytics | How is the market structured? | 10 views, incl. price spread by sector (IQR) |
| Price Prediction | What should this property cost? | LightGBM, 24 features, price + 90% range |
| Insight | Why is it priced that way? | SHAP on the deployed model + what-if simulator |
| Recommendation | What else should I look at? | Similarity across 654 societies |

## Results
- ~10.8% MAPE, R² 0.93 on a held-out test set
- 41% lower error than the best rule-based lookup (18.4%)

## Architecture
```
GitHub push → GitHub Actions → downloads model + app data (Hugging Face Hub)
            → builds 2 Docker images → publishes to GHCR
Render runs: Streamlit (UI) → calls → FastAPI (model service)
```
- **Training:** DVC + DagsHub for data versions, MLflow for experiments
- **Deployment artifacts:** private Hugging Face Hub repo
- **Configuration:** the UI finds the API through the `API_URL` environment variable

## Tech stack
Python · pandas · scikit-learn · LightGBM · SHAP · FastAPI · Streamlit ·
Docker · GitHub Actions · DVC · MLflow · Hugging Face Hub · Render

## Run locally
```
docker compose up --build
```
Needs the model and data files locally (`dvc pull`).
UI: http://localhost:8501 · API docs: http://localhost:8000/docs
