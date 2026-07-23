# Phase 2 — MLOps Interview Prep (Plain-English Guide with Examples)

**This is the "explain it like it finally made sense" version.** Every tool here is really just a
manual thing you already do, handed to a machine so it happens automatically and identically every
time. Each section has: **the problem → what you'd do by hand → the tool's version → a real example
→ interview Q&A.**

Companion docs:
- Phase 1 (data → model): [`phase1_prep_2week_plan.md`](phase1_prep_2week_plan.md)
- Build roadmap (what to build next): [`phase2_mlops_roadmap.md`](phase2_mlops_roadmap.md)

**The one sentence that ties it all together:** MLOps is just doing to your *whole ML system* what
Git already does to your *code* — version it, test it, package it, ship it, and watch it — so that
"it works on my machine" becomes "it works everywhere, provably."

---

## The big picture — how the pieces connect

```
   CODE          DATA + MODEL        EXPERIMENTS
   Git/GitHub    DVC (+DagsHub)      MLflow (+DagsHub)
      |               |                   |
      +-------+-------+-------------------+
              |
       [ Docker image ]  <- packages the API + its exact world
              |
       [ GitHub Actions ] <- on every push: run tests, build the image
              |
       [ AWS EC2 ]        <- runs the container in the cloud
              |
       [ Evidently ]      <- watches live data for drift
```

Read it top to bottom: version everything, package it, automate the checks, ship it, monitor it.
Each layer stands on the one above.

---

## 0. Git + GitHub ✅ DONE

**The problem.** Code changes constantly; you need history, and you need it somewhere safe that
isn't only your laptop.

**By hand you'd:** keep zipped folders named `project_final`, `project_final_v2`, `project_REALLY_final`.

**The tool.** Git records every change as a **commit**; GitHub stores those commits in the cloud
and lets others see them.

**Your example.**
```
git add -A
git commit -m "Fix selection bias and duplicate leakage; honest test MAPE 11.39%"
git push
```

**Interview Q&A.**
- *What's in a commit?* A snapshot of your tracked files plus a message and a parent pointer.
- *What should NOT be in Git?* Big data, the trained model, secrets, the virtualenv. (Yours are all
  in `.gitignore` — `prop/`, `data/`, `artifacts/`, `.env`.)
- *Why not commit the 4 MB model?* Git is built for text diffs; binary blobs bloat history forever.
  That's exactly what DVC is for.

---

## 1. DVC ✅ DONE

**The problem.** Git can't sensibly store your 500 MB of data or your 4 MB model — but you still
need to version them, so "the model" always matches "the data it was trained on."

**By hand you'd:** keep `data_v1/`, `data_v2/` folders and *hope* you remember which model came from
which.

**The tool.** DVC stores the **big files elsewhere** (a "remote" — yours is DagsHub) and keeps only
a tiny **pointer file** in Git. The pointer is a fingerprint: "the real data lives on DagsHub, and
here's its exact hash." Git versions the pointer; DVC versions the data.

**Your example.**
```
dvc add data/                # start tracking; creates data.dvc pointer
dvc commit -f                # record the current data/model as the new truth
dvc push                     # upload the actual bytes to DagsHub
```
Git commits the small `.dvc` pointers; `dvc push` ships the real files.

**The mental model to say out loud:**
- `docker images` = boxes on the shelf · `dvc push` = data on the shelf
- Git holds the *label*; DVC holds the *box*.

**Interview Q&A.**
- *Why version data separately from code?* Different sizes, different tools. Git diffs text; DVC
  content-addresses blobs.
- *`dvc commit` vs `dvc repro`?* `commit` records outputs you produced *some other way* (you ran
  `python -m src.main` directly); `repro` re-runs pipeline stages whose inputs changed. **You used
  `commit` all session** because you ran training by hand, not via `dvc repro`.
- *A gotcha you actually hit:* DVC **deletes a stage's output before re-running it**, so a broken
  stage command leaves you with nothing and an "output does not exist" error.
- *Honest limitation:* your `dvc.yaml` stops at feature selection — training isn't a stage, so the
  full "raw data → model" chain isn't `dvc repro`-able yet.

---

## 2. MLflow ✅ DONE

**The problem.** You train dozens of models. Which one had which settings? Which scored best? Where
is that model file? Without tracking, it's sticky notes and guesswork.

**By hand you'd:** keep a spreadsheet of "model_v3, learning_rate 0.05, MAPE 11.4%" and lose it.

**The tool.** MLflow is that spreadsheet, automated and hosted (yours on DagsHub). Every training
run records four things:

| MLflow term | Plain meaning | Your example |
|---|---|---|
| **Parameter** | a knob you set going in | `learning_rate=0.06`, `n_features=24` |
| **Metric** | a number that came out | `test_mape=11.57`, `test_r2=0.9187` |
| **Artifact** | a file produced | the trained model itself |
| **Tag** | a sticky note for organising | `society=without_society` |

The **Model Registry** is separate — the "official releases shelf": a named model
(`propnavigator-price-model`) with numbered versions (v1, v2, …), each linked back to the run that
made it.

**Your example (inside `mb_main.py`):**
```python
with mlflow.start_run(run_name="LightGBM"):
    mlflow.log_param("n_features", 24)
    mlflow.log_metric("test_mape", 11.57)
    mlflow.sklearn.log_model(best_pipeline, registered_model_name="propnavigator-price-model")
```

**Your best MLflow stories (say these):**
- *Runs must self-describe.* Your first six runs were all named `LightGBM` with nothing recording
  whether `society` was included — the UI was unreadable. Fixed by logging `n_features`, `split`,
  `selected_on`.
- *MLflow ranks by metric, not by truth.* Sorting by `test_mape` floated the **fantasy** 10.65%
  society run to the top — a number that was never real. The tool can't know which metrics are
  trustworthy; you have to.
- *A robustness bug you fixed:* logging ran *before* saving the model, so a logging crash destroyed
  a 7-minute training run. Now the model saves locally first; MLflow is wrapped in try/except.

**Interview Q&A.**
- *Experiment vs run?* Experiment = a folder for one question; run = one attempt inside it.
- *Why a registry separate from experiments?* Experiments are messy exploration; the registry is the
  curated "this is production" shelf with an audit trail back to the run.

---

## 3. FastAPI ✅ DONE

**The problem.** Your model is trapped inside your Streamlit app. Nothing else can use it, and you
can't deploy the model independently.

**By hand you'd:** copy the model-loading code into every app that needs a prediction (and they'd
drift apart — which bit you this session when `society` was dropped).

**The tool.** FastAPI turns the model into a small web service with a "front desk." Anyone sends
property details to a URL; it sends back a price.

**The two ideas:**
- **GET vs POST.** GET = *visit and read* (a browser opening a page). POST = *send data and get a
  reply*. `/health` is GET; `/predict` is POST because it carries the property details.
- **The contract (Pydantic).** You describe the input shape once; FastAPI validates every request
  for free and auto-builds interactive docs at `/docs`.

**Your example.**
```python
@app.post("/predict")
def predict(request: PredictRequest):          # Pydantic validates automatically
    return inference.predict_price(request.model_dump())
```
```
POST /predict {"property_type":"Flat","sector":"sector 49","area":1500,"bedRoom":3,"bathroom":2}
-> {"predicted_price_cr": 2.43, "lower_bound_cr": 1.97, "upper_bound_cr": 3.18, ...}
```

**Two design points worth stating:**
- **Load the model once at startup, not per request.** Reading a 4 MB file on every call would make
  the API crawl. It's loaded once when `api/inference.py` is imported.
- **`FEATURES = list(pipeline.feature_names_in_)`** — the API *asks the model* which columns it
  wants instead of hardcoding a list. That's why dropping `society` (25→24 features) needed **zero**
  API edits.

**Interview Q&A.**
- *Why an API instead of loading the model in Streamlit?* Reuse (many clients), independent scaling,
  and independent deployment. It's the standard microservice split.
- *What does Pydantic buy you?* Free validation + free docs + a single source of truth for the
  input contract.

---

## 4. Docker ⬜ NEXT

**The problem.** "Works on my machine." Your API needs Python 3.11, exact library versions, a system
library for LightGBM, and two CSVs. A fresh AWS server has none of that.

**By hand on a new laptop you'd:** install Python → `pip install` requirements → copy your code and
model → run uvicorn.

**The tool.** A Dockerfile is *exactly that list, written down* so any machine can follow it and get
an identical result. Five keywords cover everything:

| What you'd do by hand | Dockerfile |
|---|---|
| Start with a machine that has Python 3.11 | `FROM python:3.11-slim` |
| Go into my project folder | `WORKDIR /app` |
| Copy requirements in, pip install | `COPY requirements-api.txt .` + `RUN pip install ...` |
| Copy my code + model in | `COPY api/ ./api/` etc. |
| Run uvicorn | `CMD ["uvicorn", "api.main:app", ...]` |

**Vocabulary:** an **image** is the sealed box (built, sitting on disk); a **container** is a
*running* instance of that box.

**Your example — `Dockerfile`:**
```dockerfile
FROM python:3.11-slim

# LightGBM needs the OpenMP runtime; the slim base image doesn't ship it.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Deps first (change rarely) so a code edit doesn't trigger a full reinstall.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Only what the API needs at runtime.
COPY api/ ./api/
COPY artifacts/best_model.joblib ./artifacts/best_model.joblib
COPY data/price_prediction/ ./data/price_prediction/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run it:**
```
docker build -t propnav-api .        # build the image, tag it "propnav-api"
docker run -p 8000:8000 propnav-api  # run it; map laptop port 8000 -> container port 8000
```

**The three details that trip everyone up (all good interview answers):**
1. **Layer-caching order** — copy `requirements-api.txt` and install *before* copying code, so code
   edits reuse the cached install layer. Turns a 3-minute rebuild into 3 seconds.
2. **`libgomp1`** — LightGBM crashes on startup without this system library on slim Debian.
3. **`--host 0.0.0.0`** — inside a container, uvicorn's default `127.0.0.1` is only reachable *inside
   the box*. `0.0.0.0` lets your laptop reach it. Same idea for `-p 8000:8000`: it connects your
   laptop's port to the container's.

**`.dockerignore`** (companion file) — a list of what to leave OUT of the box: `prop/` (your 32k-file
venv), `notebooks/`, `data/` (except the reference CSVs), `.git/`. Keeps the image small and the
build fast.

**One decision to make deliberately:** `best_model.joblib` is DVC-tracked, not in Git. So the build
either needs the file physically present (fine locally) or a `dvc pull` step (needed in CI/cloud).

**`docker-compose.yml`** (final Docker piece) — runs *two* boxes together, `api` and `web`
(Streamlit), on one network so Streamlit reaches the API at `http://api:8000`. Your Streamlit page
already reads `API_URL` from an env var, so this needs no code change.

**Interview Q&A.**
- *Image vs container?* Image = the recipe's output on disk; container = a running instance of it.
- *Why multi-stage builds?* Build tools (compilers) bloat the final image; multi-stage builds in one
  layer and copies only the finished artifacts into a slim runtime layer.
- *Why does `EXPOSE`/`-p` matter?* Containers are isolated by default; you must explicitly connect a
  host port to a container port.
- *How does the model get into the image?* Either `COPY` a present file (local) or `dvc pull` during
  build (CI). State the trade-off.

---

## 5. CI/CD (GitHub Actions) ⬜

**The problem.** You keep *remembering* to run the tests and rebuild before shipping. Humans forget.

**By hand you'd:** run `pytest` and `docker build` yourself on every change, and skip it when you're
tired.

**The tool.** GitHub Actions runs a checklist **automatically on every push** — a robot that does
the boring, critical steps for you, every single time.
- **CI** (Continuous Integration) = automatically test + build on every change.
- **CD** (Continuous Deployment) = automatically ship it if the tests pass.

**Your example — `.github/workflows/ci.yml`:**
```yaml
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements-api.txt -r requirements-dev.txt
      - run: pytest                       # your 31 tests
      - run: docker build -t propnav-api . # prove the image still builds
```

**You already have the hard part:** 31 tests, including regression tests that *fail* if a leak is
reintroduced. CI is what makes them matter — nobody can merge a change that resurrects the `society`
bug without the pipeline going red.

**The one real snag:** most tests need `data/` and `artifacts/`, which aren't in Git. Options: (a)
`dvc pull` in CI using a DagsHub token stored in **GitHub Secrets**; (b) commit a tiny fixture
dataset so unit tests run without any pull. Best answer: both — a fixture for fast unit tests, a
`dvc pull` for integration.

**Interview Q&A.**
- *CI vs CD?* CI = test/build on every change; CD = auto-deploy when green.
- *What should block a merge?* Failing tests, a broken build, lint errors.
- *Would you retrain in CI?* No — training is slow and non-deterministic. Retraining belongs in a
  scheduled/manual pipeline, not the merge gate.
- *How do you handle secrets?* GitHub Secrets (encrypted), never in the repo.

---

## 6. AWS EC2 ⬜

**The problem.** Your container runs on your laptop. Your laptop isn't on 24/7 and nobody else can
reach it. You need it on a machine that's always up, with a public address.

**By hand you'd:** buy a server, plug it in somewhere, keep it running.

**The tool.** EC2 rents you a Linux computer in Amazon's data centre, by the hour (free tier
available). You install Docker on it and run your container there — same box, now in the cloud.

**Why EC2 over the easy options (a real interview point):** you chose EC2 over Render/App Runner
*deliberately*, to run the container on a machine you provisioned yourself — closer to how real
production works, and better learning. The trade-off: more setup and upkeep than a managed platform.

**The rough flow:**
```
# 1. Launch a t2.micro (free tier), open ports in the security group:
#    22 (SSH), 8000 (API). Do NOT open everything.
# 2. SSH in:
ssh -i key.pem ec2-user@<public-ip>
# 3. Install Docker, pull your repo, dvc pull the model:
sudo yum install -y docker && sudo service docker start
git clone <repo> && cd PropNavigator... && dvc pull
# 4. Run it:
docker compose up -d
```

**The gotchas that will actually bite (name them before they ask):**
- **1 GB RAM on free tier.** LightGBM + pandas + FastAPI is tight; Streamlit alongside may not fit.
  Plan: run only the API on EC2 and host Streamlit on Community Cloud, or add swap.
- **The public IP changes** on stop/start unless you attach an **Elastic IP**.
- **Security groups are a firewall** — only open the ports you actually need.

**Interview Q&A.**
- *Why EC2 over Lambda?* Lambda cold-starts would reload your model on every idle-then-call; a
  long-running container keeps it warm.
- *What happens on reboot?* Use a restart policy (`restart: always` in compose) so the container
  comes back automatically.
- *How would you deploy a new model version with no downtime?* Build a new image, start it
  alongside, switch traffic once healthy (blue-green), then retire the old one.

---

## 7. Monitoring (Evidently) ⬜

**The problem.** Models decay *silently*. Nothing errors — predictions just drift from reality as
the market moves. And you usually can't measure accuracy live, because the true sale price arrives
months later, if ever.

**By hand you'd:** occasionally eyeball predictions and hope.

**The tool.** Evidently compares the data your API sees *now* against the data it was *trained on*,
and flags when they diverge. Since you can't check accuracy without labels, you watch the **inputs**
as an early-warning proxy.

**The vocabulary (know all three):**
- **Data drift** — the *inputs* shift (buyers start asking about different sectors than you trained on).
- **Concept drift** — the *input→price relationship* shifts (a market crash: same inputs, different price).
- **Training-serving skew** — training and serving see systematically different data. **You have a
  lived example: the `society` story.** Trained on true society, served a guess — that gap *is*
  train/serve skew, and it degraded production from an advertised 10.65% to a real 14.29%.

**Your example (the hook already exists — `src/monitoring/prediction_logger.py`):**
```python
log_prediction(model_name, inputs, outputs)   # every /predict call is recorded
# later: Evidently reads the log, compares live input dists vs training dists -> drift report
```

**Interview Q&A.**
- *How do you know the model is still good without labels?* Monitor input drift as a proxy; alert on
  significant shifts; spot-check with any ground truth you can get.
- *What would trigger a retrain?* A drift threshold breach, a scheduled cadence, or a measured
  accuracy drop once labels arrive.
- *What's the danger of retraining on your own predictions?* A feedback loop — the model reinforces
  its own biases. Retrain on real outcomes, not model outputs.

---

## Known gaps — say these before an interviewer finds them

| Gap | Honest one-liner |
|---|---|
| `dvc.yaml` has no train stage | "Stages 1–7 are reproducible via `dvc repro`; training is still manual because it's slow and overwrites the artifact." |
| MAPE gate compares across methodologies | "It blocked a *better* model twice — it can't tell the new number was measured under stricter rules. I'd store methodology with the metric and gate only within a match." |
| Registry vs reality can diverge | "MLflow auto-registers a new version even when the local save-gate refuses it, so the registry can name a version that isn't actually serving. I'd reconcile the two." |
| No load testing | "I don't know my API's throughput or p99 latency." |
| No auth / rate limiting | "Fine for a portfolio; a public deployment needs an API key and a rate limit." |

---

## The 7 Phase-2 questions you must never fumble

1. Why version data separately from code? (DVC)
2. What's in an MLflow run, and what's the registry for?
3. Why does the API load the model at startup rather than per request?
4. What does Docker actually solve here — and what's the difference between an image and a container?
5. What should CI block a merge on?
6. How do you detect a decaying model when you have no ground-truth labels?
7. Walk me through what happens from `git push` to a new model serving traffic.
   *(push → GitHub Actions runs tests + builds the image → image deployed to EC2 → container restarts
   serving the new version → Evidently watches the incoming data.)*

---

## The whole lifecycle in one breath (practice saying this)

> *"I version code in Git and data/models in DVC, so any model always traces back to the exact data
> that made it. I track every experiment in MLflow and register the winner. I serve the model with
> FastAPI, package it and its exact environment in Docker so it runs identically anywhere, and
> automate testing and building with GitHub Actions. It deploys as a container on AWS EC2, and
> Evidently watches the live inputs for drift so I know when to retrain. The thread running through
> all of it: make the system reproducible, testable, and observable — not just accurate on my
> laptop."*
