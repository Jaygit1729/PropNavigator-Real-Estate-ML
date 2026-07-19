# Phase 2 — MLOps & Deployment Roadmap

**Scope:** everything after "here is a trained model" — versioning, tracking, serving, packaging,
automation, cloud, monitoring. Phase 1 (data → model) has its own plan:
[`phase1_prep_2week_plan.md`](phase1_prep_2week_plan.md).

**Order chosen: the industry lifecycle** — version your code and data, track experiments, serve the
model, package it, automate the checks, deploy, then watch it in production. Each step only makes
sense once the previous one exists.

---

## Where you are

| # | Step | Status | What exists |
|---|---|---|---|
| 0 | **Git + GitHub** | ✅ | Own repo, clean `.gitignore` (venv/data/artifacts excluded) |
| 1 | **DVC** | ✅ | Data + model versioned, DagsHub remote, 7-stage `dvc.yaml` |
| 2 | **MLflow** | ✅ | Hosted on DagsHub, one run per model, registry (`propnavigator-price-model`) |
| 3 | **FastAPI** | ✅ | `/health`, `/sectors`, `/predict`; Streamlit calls the API |
| 3b | **Tests** | ✅ | 31 pytest tests incl. leakage regression tests |
| 4 | **Docker** | ⬜ | **next** |
| 5 | **CI/CD** | ⬜ | GitHub Actions |
| 6 | **Cloud deploy** | ⬜ | AWS EC2 free tier |
| 7 | **Monitoring** | ⬜ | Evidently drift reports |

**Architecture:** Streamlit (UI) → HTTP → FastAPI (`api/inference.py` owns all inference logic) →
`best_model.joblib`. The UI holds **no** model logic, so the website and the API can never quote
different prices.

---

# Remaining steps

## Step 4 — Docker

**The problem it solves:** "works on my machine." Your API needs Python 3.11, specific versions of
scikit-learn and LightGBM, and two reference CSVs. A container packages all of it so the thing that
runs on EC2 is byte-identical to what you tested.

**What you'll build:**
- `Dockerfile` for the API — **multi-stage** (build deps in one layer, copy only what's needed into
  a slim runtime image)
- `.dockerignore` — keep `prop/`, `notebooks/`, `data/` out of the build context
- `docker-compose.yml` — two services: `api` and `web` (Streamlit), on one network so Streamlit
  reaches the API at `http://api:8000`

**Already prepared for this:** `requirements-api.txt` is lean (no xgboost/catboost/streamlit/shap)
and version-pinned to what you actually run. `Price_Prediction.py` reads `API_URL` from an
environment variable, so it works locally *and* in compose with no code change.

**Watch out for:** the model artifact. `best_model.joblib` is DVC-tracked, not in Git — so either
`dvc pull` during the build or mount `artifacts/` as a volume. Decide deliberately.

**Interview questions:** image vs container; why multi-stage; why not `FROM python:3.11` plain;
how do you get a 4 MB model into an image; what belongs in the image vs a volume vs an env var.

---

## Step 5 — CI/CD (GitHub Actions)

**The problem it solves:** proving that every push still works, without remembering to check.

**What you'll build:** `.github/workflows/ci.yml` that on each push installs deps, runs `pytest`,
and builds the Docker image.

**You already have the hard part** — 31 tests, including regression tests that fail if a leak is
reintroduced. CI is what makes them matter.

**Watch out for:** most of your tests need `data/` and `artifacts/`, which aren't in Git. Options:
(a) `dvc pull` in CI using a DagsHub token in GitHub Secrets, (b) commit a tiny fixture dataset,
(c) mark data-dependent tests to skip in CI. Option (a) is the realistic one; (b) makes CI fast and
hermetic. Worth doing both — a small fixture for unit tests, `dvc pull` for integration.

**Interview questions:** CI vs CD; what should block a merge; how do you keep CI fast; how do you
handle secrets; would you retrain in CI? (No — training is slow and non-deterministic; retraining
belongs in a scheduled or manual pipeline.)

---

## Step 6 — Deploy to AWS EC2

**Chosen over Render deliberately:** you want real AWS experience, and running compose on a box you
provisioned teaches more than a managed platform.

**What you'll do:** launch a `t2.micro`/`t3.micro` free-tier instance, configure a security group
(22 for SSH, 80/8000 for the API — **do not open everything**), install Docker + compose, pull the
repo, `dvc pull` the model, `docker compose up -d`. Optionally nginx as a reverse proxy.

**Watch out for:** free tier is **1 GB RAM**. LightGBM + pandas + FastAPI is tight, and Streamlit
alongside may not fit. Have a plan: run the API only on EC2 and host Streamlit on Community Cloud,
or add swap. Also: an EC2 public IP changes on stop/start unless you attach an Elastic IP.

**Interview questions:** why EC2 over Lambda (model cold starts) or App Runner (less control);
how do you get the model onto the box; what happens when the instance reboots (restart policies);
how would you deploy a new model version without downtime?

---

## Step 7 — Monitoring (Evidently)

**The problem it solves:** models decay silently. Nothing errors — predictions just drift from
reality as the market moves.

**What you'll build:** wire `src/monitoring/prediction_logger.py` into the API so every prediction
is logged, then generate Evidently reports comparing live input distributions against the training
distribution.

**Concepts to know:**
- **Data drift** — inputs change (buyers start asking about different sectors)
- **Concept drift** — the input→price relationship changes (a market crash)
- **Training-serving skew** — you have a *lived example* in the `society` story; use it
- Why you often can't measure accuracy live: ground truth (the actual sale price) arrives months
  later, if ever. So you monitor *inputs* as a proxy.

**Interview questions:** how do you know your model is still good without labels; what would trigger
a retrain; how do you avoid retraining on your own predictions.

---

# Phase 2 interview preparation

Once Steps 4–7 are built, prepare these — roughly one day each.

## Day A — Git & DVC
Why data can't live in Git (size, diffing). What a `.dvc` pointer file is. `dvc add` vs pipeline
`outs`. **`dvc commit` vs `dvc repro`** — commit records outputs you produced by other means; repro
re-runs stages whose dependencies changed. **Key gotcha you hit:** DVC deletes a stage's output
before re-running it, so a broken stage command leaves you with nothing.
*Open item:* `dvc.yaml` stops at feature selection — training isn't reproducible via `dvc repro`.
Say so rather than overclaiming "fully reproducible."

## Day B — MLflow & the registry
Experiment / run / params / metrics / artifacts / tags. Why the registry is separate from
experiments. **Your best MLflow lesson:** the first six runs were all named `LightGBM` with nothing
recording whether `society` was included — unreadable. Fixed by logging `n_features`, `split` and
`selected_on` so every run self-describes. **And:** MLflow ranks by metric and has no idea which
metrics are trustworthy — sorting by `test_mape` puts the *fantasy* 10.65% run on top.

## Day C — FastAPI & serving
REST basics; GET vs POST. Pydantic as a contract that validates for free and generates `/docs`.
**Load the model once at startup, never per request.** Why the serving logic lives in exactly one
module (`api/inference.py`) — you were bitten by duplicated `build_input_row` when `society` was
dropped and had to fix it in two files.
**Your best design detail:** `FEATURES = list(pipeline.feature_names_in_)` — the API asks the model
what it wants instead of hardcoding, so a feature-set change needs no API edit.

## Day D — Docker & CI/CD
Image vs container vs volume. Layer caching and build order. Why a lean requirements file matters.
What CI should block on. Secrets handling.

## Day E — Cloud & monitoring
Why containers make deployment portable. Security groups as a firewall. Drift types. What triggers
a retrain. The full loop: data → train → track → register → serve → monitor → retrain.

---

## The Phase 2 questions you must never fumble

1. Why version data separately from code?
2. What's in an MLflow run, and what's the registry for?
3. Why does the API load the model at startup rather than per request?
4. What does Docker actually solve here?
5. What should CI block a merge on?
6. How do you detect a decaying model without ground-truth labels?
7. Walk me through what happens from `git push` to a new model serving traffic.

---

## Known gaps — say these before they're found

| Gap | Honest framing |
|---|---|
| `dvc.yaml` has no train stage | "Stages 1–7 are reproducible; training is still manual because it's slow and overwrites the artifact." |
| MAPE gate compares across methodologies | "It blocked a *better* model twice because it can't tell that the new number was measured under stricter rules. I'd store the methodology alongside the metric and gate only within matching methodology." |
| No load testing | "I don't know my API's throughput or p99 latency." |
| No auth / rate limiting on the API | "Fine for a portfolio; a public deployment needs an API key and a rate limit." |
| Registry vs reality can diverge | "MLflow auto-registers a new version even when the local gate refuses to save it — so the registry can claim a version that isn't serving. I'd reconcile the two." |
