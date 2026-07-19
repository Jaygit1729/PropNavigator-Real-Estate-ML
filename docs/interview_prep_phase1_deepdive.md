# Interview Prep — Phase 1: Problem Statement → Model Building

Scope: everything from "why does this project exist" to "here is the trained model artifact."
Out of scope (Phase 2): MLflow/DVC deep-dive, FastAPI serving, Streamlit, monitoring, drift, retraining.

How to use this: work top-to-bottom. Each stage has (a) **what your code actually does**,
(b) **the decisions an interviewer will probe**, (c) **questions with the answer you should give**.
Answer out loud, not in your head. If you can't answer without reading, you don't own it yet.

---

## The 6-stage map of your Phase 1

| # | Stage | Your code | The one thing they'll dig into |
|---|-------|-----------|-------------------------------|
| 0 | Problem framing | `README.md` | Why regression? Why MAPE? Who uses it? |
| 1 | Data collection | `notebooks/web_scraping_v3/`, `src/data_ingestions/` | Scraping ethics, reliability, sampling bias |
| 2 | Data cleaning | `src/data_cleaning/*.py` (3 type-specific + merge) | Why 3 separate cleaners, not one |
| 3 | Feature engineering | `src/feature_engineering/feature_eng.py` (560 lines) | Area standardization, geo distances, amenity flags |
| 4 | Preprocessing | `src/data_preprocessing/pre_process_data.py` | Imputation choices, outlier rules, **leakage** |
| 5 | Feature selection | `src/feature_selection/feature_selection.py` | How were the 24 chosen? Society ablation |
| 6 | Model building | `src/model_building/mb_*.py` | Log target, stratified split, MAPE, tuning |

---

## Stage 0 — Problem Statement & Framing

**Your story (tighten this to 60 seconds):**
Listing portals (99acres, Magicbricks) are search interfaces, not decision tools. A buyer sees a
₹2.4 Cr asking price and has no way to judge whether it's fair. PropNavigator turns ~39k scraped
Gurgaon listings into four decision aids: market analytics, a price estimate with an interval,
an explanation of what drives that price, and comparable properties.

**Framing decisions you must be able to defend:**

- **Supervised regression**, target = `price_in_cr`, continuous. Not classification — buyers need a
  number, not a "cheap/expensive" bucket.
- **Target is asking price, not transaction price.** This is your biggest honest limitation. Say it
  before they find it: the model learns what sellers *ask*, which carries listing-side optimism.
  Ground truth would be registry/sale-deed data.
- **MAPE as the headline metric.** Because error is naturally proportional in real estate — being
  ₹20L off on a ₹1 Cr flat is bad, on a ₹10 Cr villa is fine. RMSE would let the luxury tail
  dominate the loss.
- **Business framing of 11.57% MAPE:** on a ₹1 Cr property that's ±₹11 lakh. Good enough to
  sanity-check an asking price, not good enough to underwrite a loan. Know where your model
  should *not* be used.

**Questions:**
1. Why is this an ML problem and not a lookup table of sector averages? → Because price is a
   non-linear interaction of area × sector × type × age × amenities; a sector average ignores all
   of it. **And you can now quantify it:** the strongest lookup table (median ₹/sqft by sector ×
   property type × bedrooms) gets **19.4%** MAPE; your model gets **11.6%** — a **40% reduction in
   error**. A naive sector-median-price table gets 47.8%. See
   `notebooks/error_analysis/baseline_comparison.ipynb`.
2. Who is the user and what decision changes because of your output?
3. What's the cost of a wrong prediction in each direction? (Over-predict → buyer overpays;
   under-predict → seller under-lists. Asymmetric? Argue it.)
4. Why price in crores and not price per sqft as the target? (PPSF target would make area a
   near-deterministic denominator and hide the model's real work; also you'd re-multiply by area
   at inference and compound error.)
5. What would make you say this project failed?

---

## Stage 1 — Data Collection

**What your code does:** two-layer Selenium scraper. Layer 1 collects listing URLs + summary
fields; Layer 2 visits each property page for bedrooms, bathrooms, facing, furnishing, amenities,
floor. Batch processing, retry logic, cooldown intervals. Three property types scraped separately
(flats, independent houses, builder floors). **~39,800 listings scraped**, Gurgaon.

**✅ Resolved (2026-07-19):** an earlier draft flagged a contradiction — `mb_tuning.py` said
"~31k rows" while the README said ~6,000. The **code was right**: 39,621 rows after cleaning,
39,066 after outliers, 38,239 after de-duplication, of which 22,943 are training rows (the "~31k"
comment predated the validation split, when train was 80%). The README's "6,000" was stale from an
early scrape and has been corrected.

**Questions:**
1. Walk me through the scraper architecture. Why two layers instead of one? → Listing pages are
   cheap and paginated; detail pages are expensive. Two layers lets you resume, dedupe by URL, and
   avoid re-hitting detail pages on retry.
2. How did you handle being blocked / rate limits? → Batches, cooldowns, retries. Be honest about
   whether you used proxies or rotating user agents.
3. **Is scraping this legal/ethical?** Have a real answer: public listing data, no login bypass,
   respectful rate, personal/educational project, no PII stored, no redistribution of raw listings.
   Mention robots.txt and ToS as a known risk you'd resolve with an official data agreement in a
   commercial setting.
4. **What sampling bias does scraping introduce?** This is the strong question. Listings ≠ market.
   You over-sample what's currently unsold (possibly overpriced), under-sample fast-moving segments,
   and get whatever the portal's ranking surfaced. Also single-city, single-time-snapshot.
5. How would you know if the data is stale? What's the refresh strategy?
6. Duplicates: the same property relisted by 3 brokers. How did you dedupe? (You call
   `drop_duplicates()` in preprocessing — is that enough? Exact-row match won't catch near-dupes
   with different descriptions. Know this gap.)

---

## Stage 2 — Data Cleaning

**What your code does:** three separate cleaners —
`residential_apartment_cleaning.py`, `Independent_house_cleaning.py` (as
`house_cleaning.py`), `indepedent_builder_floor.py` — then `merge_data.py` unions them into
`cleaned_properties.csv`. Documented in `docs/cleaning_process.md`.

**Questions:**
1. **Why three cleaners instead of one?** → Different property types have genuinely different raw
   schemas and semantics: a flat has a society and a floor number, an independent house sits on a
   plot and has neither. Forcing one cleaner would mean a mess of `if property_type ==` branches.
   Cleaning per-type and unioning on a common schema is more readable and independently testable.
2. What's the common schema after merge? Which columns are type-specific and how are they filled?
3. What did you do about the same physical attribute reported in different units/strings?
4. How do you guarantee the three cleaners stay in sync when you add a column? (Honest answer: you
   don't, today — it's convention. A shared schema contract / test would fix it. Saying this shows
   engineering maturity.)
5. Where in the pipeline does the row count drop and by how much? Be able to trace
   39,795 scraped → 39,621 cleaned → 39,066 after outliers → 38,239 after de-dup → 22,943 train.

---

## Stage 3 — Feature Engineering (your strongest storytelling material)

Four sub-stories. Each one is a "tell me about a hard technical decision" answer.

### 3a. Area standardization (`_process_area`, feature_eng.py:154)
Raw listings report area as super built-up, built-up, or carpet — not comparable. You extract the
number and its type from `areaWithType` via regex, then **learn the conversion ratios from the data
itself**: on rows where both `carpetArea` and a typed area exist, you take the median
`carpet/super` and `carpet/builtup` ratios, derive `builtup→super` as their quotient, and rescale
every row to super built-up sqft.

Why this is good: you didn't hardcode the industry rule-of-thumb (~0.7 carpet/super) — you
estimated it from your own market's data.

**Probes:**
- Why median and not mean? (Ratios have a heavy tail from bad listings; median is robust.)
- **Is this leakage?** The ratio is estimated on the full dataset before the train/test split.
  Strictly, yes — a whisper of test information leaks into the training features. Defend it as
  low-risk (a global unit-conversion constant, not a target-derived statistic) but *acknowledge*
  the correct fix: fit the ratio on train only. Do not pretend it's clean.
- What if `areaWithType` doesn't match the regex? What happens to that row?
- Why standardize to super built-up rather than carpet? (Most common in the data → fewest rows
  transformed → least conversion noise.)

### 3b. Sector standardization
350+ raw sector labels → 102 canonical sectors, via `SECTOR_ALIASES`, block/pocket stripping
(`"block c"`, `"pocket d"` hide the real locality), and grouping localities below a minimum
listing count into `"other"`.

**Probes:** Why not just one-hot the 350? (Sparse, high-variance, many categories with 1–2 rows →
the model memorizes noise.) How did you validate two aliases really are the same locality? What
does `"other"` cost you at inference for a genuinely new sector? (Handled downstream by
`unknown_value=-1` in the ordinal encoder.)

### 3c. Geo distance features (`_process_geo`, feature_eng.py:488)
Four landmarks — Cyber City (office hub), Golf Course Road (luxury belt), the airport, Manesar
(cheap industrial edge) — with straight-line (haversine-style) km to each. Chosen to point in
*different directions* so the four distances together triangulate a home's position.

**Probes:**
- Why distances instead of raw lat/long? → Trees split on axis-aligned thresholds; raw lat/long
  makes a diagonal price gradient expensive to approximate. Distance-to-anchor encodes the
  economic gradient directly in one feature.
- Why straight-line and not driving distance? (No routing API; straight-line is a monotone proxy.
  Named limitation: it ignores the highway/metro corridors that actually drive Gurgaon prices.)
- Aren't the four distances collinear? (Yes, partly — that's fine for trees, would be a problem
  for the linear model in your Insight Module. Know which models care.)
- Rows with bad coordinates → median-imputed in preprocessing. What's wrong with that? (Imputes a
  *location* with a global median — geographically meaningless. Small row count, but say so.)

### 3d. Amenity handling — and a correction to an earlier draft of this doc

**⚠️ An earlier version of this document claimed you had a target-derived "luxury score" that was
target encoding and your biggest leakage risk. That was checked against the code and is FALSE.
Do not raise it in an interview — you would be confessing a flaw you don't have, and there is no
code to show if they ask.**

What actually exists:
- **`AMENITY_FLAGS`** — plain binary columns (`has_ac`, `has_power_backup`, `has_pool`,
  `ov_main_road`, `ov_others`), built from a vocabulary of amenities appearing in ≥20 listings.
  Leakage-free, and these are what feed the shipped 24-feature model.
- **`luxury_count`** — exists only in `notebooks/feature_engineering/fe.ipynb` (exploratory). It's
  a count of hand-curated keywords (`LUXURY_KEYWORDS = ['swimming pool', 'club', 'gym', 'spa', ...]`).
  **No price is involved.** It is not target-derived, is not in `src/`, and is not in the model.

**If asked "did you consider a composite luxury feature?"** — Yes, explored in the notebook as a
keyword count. It correlated weakly with price because amenity lists are largely society-level
boilerplate, so individual binary flags were kept instead. SHAP confirms the amenity flags are weak
drivers regardless — `area` and the engineered distances dominate.

**The general principle to state:** any amenity weighting derived from observed price *would* be
target encoding, and would have to be fit on the training fold only (ideally out-of-fold). That's
exactly why the simpler leakage-free flags were preferred.

---

## Stage 4 — Preprocessing (`pre_process_data.py`)

**Imputation, and the reason for each:**

| Column | Strategy | Justification |
|---|---|---|
| `bathroom == 0` | ← bedroom count | A home can't have 0 bathrooms; it's a missing-value sentinel |
| `floornum`, `total_floor` | 0 | Missing means "ground / not applicable" (independent houses) |
| parking, balcony | 0 | Absent info = none recorded |
| distances | median | Un-fixable coordinates |
| `facing`, `furnishing` | `"unknown"` category | Missingness is itself informative — don't hide it in the mode |
| `age_possession_category` | **3-pass mode** | See below |
| `society` | rare (<5) → `"other"` | Cardinality control |

**The 3-pass mode imputation** is your most interesting choice: pass 1 = mode within
(sector × property_type), pass 2 = mode within sector, pass 3 = mode within property_type. Most
specific context first, falling back progressively.

**Probes:**
- Why hierarchical instead of a single global mode? → Property age clusters by locality; a global
  mode would flatten real signal.
- **Is this leakage?** It's computed on the full frame before splitting and uses no target — so no
  *target* leakage, but it is train/test contamination of a feature statistic. Same honest answer
  as 3a.
- It's `df.apply(..., axis=1)` three times over the whole frame — O(n) row-wise passes each doing a
  filtered mode. That's slow. How would you vectorize? (`groupby().transform()` with the mode.)
  Expect a "make this faster" follow-up.
- Why `"unknown"` for facing but median for distance? (Categorical missingness can be a level;
  numeric can't be.)

**Outlier removal — one univariate, two bivariate:**
1. `remove_area_outliers` — area in [180, per-type cap]: Flat 16k, Builder Floor 10k, House 30k.
2. `remove_price_area_outliers` — implied ₹/sqft must be in [1500, 250000].
3. `remove_area_bedroom_outliers` — ≥150 sqft per bedroom.

**Probes:**
- Why domain rules instead of IQR / z-score / isolation forest? → IQR on a right-skewed price
  distribution deletes legitimate luxury properties. Domain bounds delete *impossible* rows only.
  This is the right answer and it's a strong one — lead with it.
- Rule 2 uses price (the target) to filter rows. Defend it: you're removing data-entry errors, and
  it's applied identically to every row before splitting, not fitted. But note it does mean your
  test set is not a fair sample of "all listings including garbage ones."
- Why per-type area caps rather than one? (A 20,000 sqft "flat" is a typo; a 20,000 sqft house
  isn't.)
- How many rows does each rule remove? **Know these numbers** — they're in `logs/pre_processing.log`.
- Would you clip instead of drop? Trade-off?

**The leakage answer you should volunteer:**
The docstring at `pre_process_data.py:238` is your best line — *"Does NOT encode or scale — those
are fit on the training split inside the model pipeline (leakage-safe)."* Encoding and scaling live
in the sklearn `Pipeline`, so they're fit inside each CV fold. That's the part you got right;
say it confidently, then be honest about the global-statistic cases above.

---

## Stage 5 — Feature Selection

24 features + target, hardcoded in `SELECTED_FEATURES`, with `society` deliberately dropped
(commit `e764c05`, plus `notebooks/error_analysis/society_ablation.ipynb`).

**Probes:**
1. **How were these 24 chosen?** Go re-read `notebooks/feature_selection/feature_selection.ipynb`
   and be able to name the methods (correlation, mutual information, tree importance, RFE,
   permutation importance — whichever you actually used) and where they disagreed. "The notebook
   picked them" is a failing answer.
2. Why hardcode the list instead of selecting at runtime? → Reproducibility and a stable serving
   contract; the API schema depends on exactly these 24. Selection is a decision you made once and
   version-controlled, not something to re-roll every training run.
3. **The society ablation is a great story.** Society is a high-cardinality categorical that
   almost certainly *improved* offline metrics. You dropped it anyway. Be ready to explain why:
   cardinality/overfitting, unusable for new societies at inference, and a worse UX (users don't
   always know the society). Have the before/after MAPE numbers ready.
4. Multicollinearity: `total_parking` vs `covered_parking` + `open_parking`; `bedRoom` vs
   `bathroom`; the four distances. Why keep them? (Trees tolerate it; it splits importance across
   correlated features, which distorts your SHAP/importance story but not accuracy.)
5. What would you drop if you had to get to 10 features for a simpler UI?

---

## Stage 6 — Model Building

**The pipeline** (`mb_main.py` → `mb_tuning.py` → `mb_persistence.py`):

1. **Split** — **60/20/20 train / validation / test**, `random_state=42`, **stratified on
   `pd.qcut(y, 5)` price quintiles** (`create_train_val_test_split`). Sizes: 22,943 / 7,648 / 7,648.
2. **Target transform** — `log1p(price)`, inverted with `expm1` before every metric.
3. **Feature lists derived dynamically** from `X_train` dtypes — feature selection can change
   without breaking model building.
4. **Preprocessor** — `ColumnTransformer`: `OrdinalEncoder(handle_unknown="use_encoded_value",
   unknown_value=-1)` on categoricals, numericals passed through unscaled.
5. **Candidates** — XGBoost, LightGBM, CatBoost. RandomForest and stacking intentionally dropped.
6. **Tuning** — `RandomizedSearchCV`, `n_iter=25`, `KFold(3, shuffle=True)`, scoring = negative
   MAPE, fit on **train only**.
7. **Selection** — lowest **validation** MAPE. LightGBM won (11.40% val, vs XGBoost 12.72%,
   CatBoost 12.82%).
8. **Reporting** — the winner alone is scored on the untouched test set: **11.57% MAPE, R² 0.9187**.
9. **Prediction intervals** — percentage residual quantiles (q05/q95, q10/q90) computed on
   **validation**, so the test set is used for reporting only.
9. **Persistence** — `save_model` only overwrites if the new MAPE beats the stored artifact;
   every run (saved or skipped) is appended to `artifacts/experiment_log.csv`; a timestamped
   version is kept alongside `best_model.joblib`.

### The questions that decide this round

**Q. Why log-transform the target?**
Price is right-skewed. In log space the error the model minimizes becomes roughly *relative*
error, which matches MAPE — the metric you actually care about. It also stops ₹15 Cr villas from
dominating the squared loss.

**Q. You trained on log price but report MAPE on the original scale. Any problem?**
Yes — `expm1(E[log y])` is the conditional **median**, not the mean, so back-transformed
predictions are systematically biased low (Jensen's inequality). For MAPE, a median-unbiased
predictor is arguably *what you want*. But know the term "retransformation bias" and know the fix
(Duan's smearing estimator). **This question separates candidates. Prepare it cold.**

**Q. Why stratify the split on price quintiles?**
Small dataset + heavy tail. A plain random split can put an uneven share of luxury properties in
test, making MAPE swing run to run. Stratifying on quintiles keeps the price distribution matched.

**Q. Isn't stratifying on the target a form of leakage?**
No — it affects only which rows go where, not what the model sees. But it does make your test
MAPE slightly optimistic relative to a truly random future sample. Fair to concede.

**Q. Why ordinal-encode categoricals for tree models instead of one-hot?**
Trees don't need one-hot; ordinal keeps dimensionality low and lets a single node split isolate
category groups. One-hotting 102 sectors would create 102 sparse columns, each a weak split.
The imposed ordering is meaningless but trees can carve arbitrary subsets with enough depth.
Caveat to volunteer: CatBoost has native categorical handling you're not using — you're feeding
it ordinal codes instead. That's a real thing to improve.

**Q. `unknown_value=-1` — what happens at inference for an unseen sector?**
It maps to −1, which sits outside the training range, so the tree routes it to the leftmost branch.
It won't crash, but the prediction is essentially "some arbitrary corner of the feature space."
Better: fall back to a sector-level median, or refuse to predict and say so in the UI.

**Q. Why no scaling on the numeric features?**
Trees are scale-invariant — splits depend on order, not magnitude. (Your Insight Module's linear
model is a different story; scaling matters there.)

**Q. Why RandomizedSearchCV and not GridSearchCV or Bayesian optimization?**
8-dimensional continuous search space; a grid is combinatorially hopeless and spends budget
uniformly on dimensions that don't matter. Random search hits good regions far faster
(Bergstra & Bengio). Optuna/TPE would be the next step and you should say you'd use it.

**Q. Why only 3 CV folds and 25 iterations?**
Compute budget across three model families. Trade-off you accepted: fewer folds = noisier CV
estimate. With a small dataset, 5-fold would be more stable and you'd have preferred it.

**Q. How do you avoid selecting on the test set?**
**This used to be a real flaw in this pipeline, and fixing it is now one of your best stories.**

Originally: hyperparameters were tuned with CV on train (correct), but the winning *family* was
then chosen by **test** MAPE — so the reported 11.18% was inflated by a decision the test set had
participated in. Fixed with a three-way split: family chosen on **validation**, test scored exactly
once. Honest number moved 11.18% → **11.57%**.

**Say it like this:** *"I audited my own pipeline and found I was selecting the model family on the
test set. I moved to a 60/20/20 split where selection happens on validation and test is touched
once. My number went from 11.18% to 11.57% — that 0.2 points was the inflation I'd been unknowingly
reporting. Small, but I'd rather quote a number I can defend."*

Two follow-ups worth having ready:
- **"Why not nested CV or select on the CV score instead?"** Both valid. Selecting on the train CV
  score costs no data and is standard; I chose an explicit validation split because with ~38k rows I
  can afford it and it's unambiguous to explain. Nested CV is the rigorous option at higher compute.
- **"Your test came out better than validation (11.39 vs 11.79) — isn't that suspicious?"** No, and
  it's reassuring. They're two different 7,647-row samples, so a few tenths of difference is normal
  sampling noise. If test had been *much worse* than validation, that would indicate I'd overfit the
  selection to validation. It didn't.

**Q. Train MAPE vs test MAPE — are you overfitting?**
Both are logged (`mb_tuning.py`). **Know the actual gap.** If train is far below test, name the
regularization levers you already have: `reg_alpha`, `reg_lambda`, `min_child_samples`,
`subsample`, `colsample_bytree`, capped `max_depth`.

**Q. Why did LightGBM win over XGBoost and CatBoost?**
Honest answer: marginally, on this data and these search budgets — it's not a law of nature. If
the three are within ~0.5% MAPE, the "winner" is partly search-budget noise. Saying that is
stronger than inventing a theoretical reason. Mention LightGBM's leaf-wise growth suiting the
non-linear price surface, but don't oversell it.

**Q. Why drop RandomForest and stacking?**
Documented in the `run_model_building` docstring: RF was slowest to tune and weakest; a single
model is simpler to serve and to explain with SHAP than a stacked ensemble. Interpretability is a
product requirement here (the Insight Module), so it's a real trade-off, not laziness.

**Q. Explain your prediction intervals.**
Not a model-based interval — empirical. You compute percentage errors `(y_true − y_pred)/y_pred`
on the test set and take the 5th/95th (and 10th/90th) percentiles, then apply those as
multiplicative bands at inference.
Weaknesses to volunteer: they're **global**, so a ₹50L flat and a ₹12 Cr villa get the same
relative band even though the tail is far less certain; and they're fit on the same test set you
report metrics on. Conformal prediction (or quantile regression with `objective="quantile"`) would
give properly calibrated, feature-conditional intervals. This is your best "what would you do
next" answer.

**Q. Walk me through `save_model`'s MAPE gate.**
New artifact is written only if it beats the stored MAPE; every attempt is logged to
`experiment_log.csv` with `saved_best` / `skipped_worse`; a timestamped copy is kept for rollback.
**Weakness to own:** the gate compares MAPE across *different test splits over time*. If the data
changes, the comparison isn't apples-to-apples, and a model can be blocked by a stale record. A
fixed holdout or a challenger/champion evaluation on the same data would be correct.

**Q. Metrics — you compute R², MAE, RMSE, MAPE. When would you report each?**
MAPE for the business conversation (relative error). MAE in ₹ for "how far off in rupees." RMSE
when large errors are disproportionately costly. R² for variance explained — but note R² is
inflated by the wide price range here and is the least useful of the four for your use case.

**Q. What's MAPE's known failure mode?**
It's asymmetric — it punishes over-prediction more than under-prediction, so optimizing it biases
the model to under-predict. It also explodes as y→0 (not an issue here; no ₹0 properties). If you
wanted symmetry: SMAPE, or MAE on log price.

---

## Your 8 must-know numbers (verified 2026-07-19 — know these cold)

1. **Row funnel:** 39,795 scraped (28,531 flats + 9,210 builder floors + 2,054 houses) →
   **39,621** after cleaning & merge → **39,066** after outlier removal → **38,239** after
   de-duplicating on the final feature set → **22,940 train / 7,647 val / 7,647 test**.
   *(The old "~6,000 properties" in the README was stale and has been corrected.)*
2. **Features:** 24 + target. **18 numerical, 6 categorical** (`property_type`, `sector`,
   `furnishing`, `age_possession_category`, `facing`, `floornum_category`).
3. **Validation MAPE by model** — LightGBM **11.40%**, XGBoost **12.72%**, CatBoost **12.82%**.
   Selection happens here.
4. **Winner train vs test** — train MAPE **7.35%**, test MAPE **11.57%**. A real gap: the model
   fits train harder than it generalises. Levers already in use: `reg_alpha`, `reg_lambda`,
   `min_child_samples`, `subsample`, `colsample_bytree`, capped `max_depth`.
5. **Held-out test (the number you quote): MAPE 11.57%, R² 0.9187.** Validation MAE ₹0.45 Cr,
   RMSE ₹1.27 Cr.
6. **Residual quantiles** q05 **−0.210** / q95 **+0.269** → "the 90% interval is roughly
   **−21% / +27%** around the point estimate." Calibrated on validation.
7. **Rows removed by each rule:** area 193, price-vs-area 48, area-per-bedroom 279 (520 total),
   plus **827** duplicate feature-rows dropped before splitting.
8. **Baseline to beat — COMPUTED ✅** (`notebooks/error_analysis/baseline_comparison.ipynb`).
   Lookup tables built from **training-set** medians, scored on the same test set:

   | Approach | Test MAPE | Knows |
   |---|---|---|
   | Global median price | 72.3% | nothing |
   | Sector median price (naive) | 47.8% | location only |
   | Global median ₹/sqft × area | 33.3% | size only |
   | Sector median ₹/sqft × area | 23.6% | location + size |
   | Sector × type median ₹/sqft × area | 21.4% | + property type |
   | **Sector × type × BHK median ₹/sqft × area** | **19.4%** | strongest lookup |
   | **ML model (LightGBM)** | **11.6%** | all 24 features + interactions |

   **The model beats the best lookup table by 7.8 points — a 40% reduction in error.**
   Note also: **area alone (33.3%) beats sector alone (47.8%)**, which independently corroborates
   SHAP putting `area` as the dominant driver.

Sources: `logs/mb_tuning.log`, `logs/pre_processing.log`, `artifacts/experiment_log.csv`,
MLflow experiment `propnavigator-model-building` on DagsHub.

---

## Weaknesses — what you FIXED, and what's still open

Naming your own flaws is the highest-leverage interview move. But now you can do something better
than confess: **show the audit and the fix.** Lead with these two.

### ✅ Found and fixed (tell these as stories)

**1. Model family was selected on test MAPE.** Found during a self-audit. Fixed with a 60/20/20
split — selection on validation, test scored once. Honest number 11.18% → **11.57%**. *"The 0.2
points was the inflation I'd been unknowingly reporting."*

**2. Duplicate rows straddled train and test.** De-duplication ran on the full 35-column table, but
narrowing to 24 features created new duplicates: **252 test rows (3.23%) had an identical twin in
train** — the model could memorise them. Fixed by de-duplicating on the final feature set before
splitting (827 rows). Overlap now **zero**.

**3. `society` was a train/serve-skew liability.** Trained on true society, but production could
only guess it from sector — correct **31%** of the time. Measured: 10.65% (true society, a fantasy)
vs **14.29%** (guessed — what production really delivered) vs **11.18%** (retrained without it).
Dropping it improved production accuracy by ~3 points. *"Never train on a feature you can't obtain
honestly at serving time."* Full experiment: `notebooks/error_analysis/society_ablation.ipynb`,
logged in MLflow.

### ⚠️ Still open (volunteer honestly)

**4. Imputation is fitted before the split.** `fillna(df[col].median())` for distances and the
3-pass mode imputation for `age_possession_category` are computed over train **and** test. Real
leakage, but tiny — ~187 rows, about 0.5%. Correct fix: move imputation inside the sklearn pipeline
so it's fitted on train only. Not yet done.

**5. Target is asking price, not sale price.** Structural — asking prices run above transaction
prices, so the model predicts *listing* behaviour, not market-clearing value. (Stage 0)

**6. Outlier removal uses the target.** `remove_price_area_outliers` filters on price/sqft.
Mitigating detail worth stating: the thresholds are **fixed domain constants** (₹1,500–250,000/sqft),
not data-derived quantiles, so no statistics leak. But you can't filter production requests by price,
so the test set is a slightly cleaner population than reality.

**7. No test suite.** Zero pytest coverage. Needed before CI.

**8. Prediction intervals are global, not conditional** — one ±22%/+24% band for a ₹0.5 Cr flat and
a ₹15 Cr villa. Conditional (quantile-regression) intervals would be the upgrade.

For each open item: state the flaw, state the impact honestly, state the fix. Three sentences, no
defensiveness.

---

## Prep schedule (5 days)

- **Day 1** — All eight must-know numbers are filled in and verified (2026-07-19), baseline
  included. So: rehearse the three **"found and fixed"** stories out loud — selection bias,
  duplicate leakage, and the `society` train/serve skew — plus the **baseline comparison**
  (19.4% → 11.6%, 40% less error). Those four are your strongest material: they show you audit
  your own work and can prove your model's value rather than assert it.
- **Day 2** — Stages 0–2. Write your 60-second problem pitch and say it out loud 10 times. Trace
  the row count through cleaning.
- **Day 3** — Stage 3–4. The four FE stories + every imputation and outlier rule with its "why."
  Do the leakage audit: list every statistic computed before the split.
- **Day 4** — Stage 5–6. Re-read the feature-selection notebook until you can name the methods.
  Then drill the hard six: log-transform bias, stratified split, ordinal vs one-hot, random vs grid
  search, test-set selection, interval construction.
- **Day 5** — Full mock. Whiteboard the pipeline from memory, no notes. Then say the 5 weaknesses
  out loud with fix for each. If you can do both, you're ready.

**Whiteboard drill** — you should be able to draw this in 90 seconds:
`scrape → 3 type-specific cleaners → merge → feature engineering (area/sector/geo/amenities) →
preprocess (impute + outliers, no encoding) → select 24 → stratified split → log1p(y) →
ColumnTransformer(ordinal) → 3 GBMs × RandomizedSearchCV(3-fold, neg-MAPE) → best by MAPE →
residual quantiles → MAPE-gated joblib artifact`
