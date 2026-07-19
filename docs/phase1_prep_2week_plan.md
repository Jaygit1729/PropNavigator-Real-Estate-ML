# Phase 1 — 2-Week Interview Prep Plan

**Scope:** problem framing → web scraping → EDA → cleaning → feature engineering → preprocessing →
feature selection → model building. Everything up to "here is the trained model artifact."
Phase 2 (MLflow, DVC, FastAPI, Docker, CI/CD, cloud, monitoring) has its own roadmap:
[`phase2_mlops_roadmap.md`](phase2_mlops_roadmap.md).

**How each day is structured:**

| Block | Purpose | Time |
|---|---|---|
| **Fundamentals** | The ML concept itself — what you'd need to know even without this project | 45 min |
| **Your code** | What *your* implementation actually does, read with the concept in mind | 45 min |
| **Q&A drill** | Answer the listed questions **out loud**, no notes | 30 min |
| **Prove it** | A small concrete task. If you can't do it, you don't own the day yet | 30 min |

~2.5 hrs/day. If you only have 1 hour, do Fundamentals + Q&A and skip Prove-it — but never skip
saying answers out loud. Reading feels like understanding and isn't.

---

## Your verified numbers (know these cold — they anchor every answer)

| | Value |
|---|---|
| Rows: scraped → cleaned → outliers → de-dup | 39,795 → 39,621 → 39,066 → **38,239** |
| Train / validation / test | 22,943 / 7,648 / 7,648 (60/20/20, stratified on price quintiles) |
| Features | **24** = 18 numerical + 6 categorical |
| Validation MAPE | LightGBM **11.41%** · XGBoost 12.72% · CatBoost 12.82% |
| **Held-out test (the number you quote)** | **11.57% MAPE, R² 0.9187** |
| Train MAPE (winner) | 7.35% → a real train/test gap, know your regularization levers |
| Best lookup-table baseline | **19.37%** → the model cuts error by **~40%** |
| 90% prediction interval | −21% / +27% (residual quantiles, calibrated on validation) |
| Test suite | 31 tests, ~16s |

---

# WEEK 1 — Data: from raw web pages to a modelling table

## Day 1 — Problem framing & why MAPE

**Fundamentals.** Regression vs classification. The three error metrics and what each punishes:
MAE (absolute, robust), RMSE (squares errors → outliers dominate), MAPE (relative → percentage
error). Why the *metric choice is a business decision*, not a technical one.

**Your code.** `README.md`, the target `price_in_cr`.

**Q&A.**
1. Why regression and not "cheap / fair / expensive" classification?
2. Why MAPE over RMSE? → error in property pricing is naturally proportional; ₹20L off on a ₹1 Cr
   flat is a disaster, on a ₹10 Cr villa it's noise. RMSE would let the luxury tail dominate.
3. **What's wrong with MAPE?** (Know this — it's the follow-up.) It's asymmetric: it punishes
   over-prediction more than under-prediction, and it explodes as the true value approaches zero.
   Alternatives: sMAPE, or MAE on log-price.
4. Your target is **asking price, not sale price**. What does that mean for the model? → it learns
   listing-side optimism, not market-clearing value. Ground truth would be registry data.
5. Business framing: 11.57% on a ₹1 Cr property is ±₹11.6 lakh. Where should this model *not* be
   used? (Loan underwriting, legal valuation.)

**Prove it.** Write your 60-second project pitch. Say it out loud 10 times, timed.

---

## Day 2 — Web scraping & sampling bias

**Fundamentals.** How scraping works (HTML → parsing → structured rows). Static vs JS-rendered
pages, and why the latter needs Selenium. Rate limiting and bot detection. **Sampling bias** — the
single most important concept of this day.

**Your code.** `notebooks/web_scraping_v3/` — the two-layer scraper (Layer 1: listing URLs +
summary; Layer 2: detail pages).

**Q&A.**
1. Why two layers instead of one? → listing pages are cheap and paginated, detail pages expensive.
   Two layers lets you resume, dedupe by URL, and avoid re-hitting detail pages on retry.
2. Why Selenium + undetected-chromedriver rather than `requests` + BeautifulSoup?
3. **Is this legal/ethical?** Have a real answer: public data, no login bypass, respectful rate,
   educational use, no PII, no redistribution. Name robots.txt and ToS as the known risk you'd
   resolve with a data agreement commercially.
4. **What sampling bias does scraping introduce?** ← the strong question. Listings ≠ market. You
   over-sample unsold (possibly overpriced) inventory, under-sample fast-moving stock, and inherit
   the portal's ranking. Single city, single time snapshot.
5. How would you detect stale data? What's the refresh strategy?

**Prove it.** Name three ways your dataset is *not* a random sample of Gurgaon property.

---

## Day 3 — EDA fundamentals

**Fundamentals.** Distributions and skew; why right-skewed targets hurt squared-error models.
Log transforms. Correlation ≠ causation; Pearson vs Spearman. Univariate vs bivariate outliers.
**Simpson's paradox** — you have a real example (amenities look valuable overall but are weak
within a sector, because they're society-level boilerplate).

**Your code.** `notebooks/EDA/eda_propnavigator.ipynb`.

**Q&A.**
1. Describe your target's distribution. Why does that shape matter?
2. What surprised you in EDA and how did it change what you built?
3. Which features looked predictive in EDA but weren't in the model — and why?
4. How did you decide something was an outlier vs a genuine luxury property?

**Prove it.** Sketch the price distribution from memory. State its skew direction and your fix.

---

## Day 4 — Data cleaning

**Fundamentals.** Schema normalisation. Types of duplicates (exact, near, semantic). Idempotency —
running a cleaner twice must not change the result.

**Your code.** Three cleaners (`residential_apartment_cleaning.py`, `house_cleaning.py`,
`indepedent_builder_floor.py`) + `merge_data.py`. Also `docs/cleaning_process.md`.

**Q&A.**
1. **Why three cleaners instead of one?** → the raw schemas genuinely differ: a flat has a society
   and a floor number; an independent house sits on a plot and has neither. One cleaner would be a
   thicket of `if property_type ==` branches.
2. How do you keep the three in sync when a column is added? (Honest: you don't — it's convention.
   A shared schema contract or test would fix it. Saying this shows maturity.)
3. Same property listed by three brokers — how do you dedupe? What does exact-match miss?
4. Trace the row count: 39,795 → 39,621 → 39,066 → 38,239. What was removed at each step?

**Prove it.** Recite the row funnel and what each drop represents.

---

## Day 5 — Feature engineering I: area & sector

**Fundamentals.** Domain-driven features vs automated ones. Unit normalisation. Cardinality —
why 350+ raw sector labels is a problem.

**Your code.** `feature_eng.py` → `_process_area` (carpet/built-up/super built-up → one scale using
ratios **learned from the data**: carpet/super = 0.725, builtup/super = 0.925), sector
standardisation → 131 clean sectors.

**Q&A.**
1. Why standardise area at all? What breaks if you don't?
2. You learned the conversion ratios from the data — **is that leakage?** (Think it through: the
   ratios come from rows that include your test set. It's a mild version of the same issue you
   fixed elsewhere. Be ready to concede it and describe the fix: compute ratios on train only.)
3. How did 350+ raw sector strings become 131? What did you do with unmatched ones?
4. Why is `sector` kept as a categorical rather than encoded by average price? → that would be
   target encoding, and it needs out-of-fold computation to avoid leakage.

**Prove it.** Explain the area conversion, including where you'd defend or concede on leakage.

---

## Day 6 — Feature engineering II: geo & amenities

**Fundamentals.** Haversine distance. Why distance-to-landmark encodes location better than raw
lat/long for tree models. Collinearity and why it destabilises *linear* models but not trees.
Binary flags vs composite scores.

**Your code.** `_process_geo` → four distances (Cyber City, Golf Course Road, airport, Manesar);
fixed 255 swapped coordinate pairs, 187 unresolvable → NaN. Amenity flags from a vocabulary of
amenities appearing ≥20 times.

**Q&A.**
1. Why distances to four landmarks instead of raw coordinates? → trees split on axis-aligned
   thresholds; raw lat/long forces awkward staircase boundaries, while "8 km from Cyber City" is a
   single meaningful split.
2. Those four distances are highly correlated. Why is that acceptable here? → tree ensembles are
   robust to collinearity; it would break coefficient interpretation in a linear model. (This is
   exactly why you used SHAP rather than Ridge coefficients for the Insight module.)
3. Why keep binary amenity flags instead of one composite luxury score? → **any score weighted by
   observed price would be target encoding.** The flags are leakage-free, and SHAP shows amenities
   are weak drivers anyway.
4. ⚠️ **Do not claim you built a price-weighted luxury score.** You didn't — the only luxury code is
   a hand-curated keyword count in an exploratory notebook, never in `src/` or the model.

**Prove it.** Explain why 187 rows keep NaN distances instead of being imputed.

---

## Day 7 — Review + Mock #1

No new material. Do a full mock covering Days 1–6: *"Walk me through your data pipeline from
scraping to a clean table."* Record yourself. Note every stumble; those are Week 2's homework.

---

# WEEK 2 — Modelling: from a clean table to a defensible number

## Day 8 — Preprocessing & data leakage ⭐

**This is your strongest technical day. Leakage is the concept that separates candidates.**

**Fundamentals.** What leakage is: any information reaching training that wouldn't be available at
prediction time. Its forms — target leakage, train/test contamination, temporal leakage,
train/serve skew. **Golden rule: anything *fitted* (a median, a scaler, an encoder) must be fit on
training data only.**

**Your code.** `pre_process_data.py` — outlier rules (area, price-vs-area, area-per-bedroom, all
using **fixed domain constants**), and the two imputations you *removed*.

**Q&A.**
1. What is data leakage? Give three distinct forms.
2. **You removed imputation entirely — why?** → distance medians were computed over the whole
   dataset, so test statistics shaped training inputs. Tree models handle NaN natively and more
   informatively than a median. Same for `'Undefined'` age: kept as its own category, matching how
   `facing`/`furnishing` treat unknowns.
3. What's the trade-off? → it couples preprocessing to NaN-tolerant models. A linear model would
   need imputation back, fitted on train only.
4. **Your outlier rules use price. Isn't that target leakage?** → the thresholds are fixed domain
   constants (₹1,500–250,000/sqft), not data-derived quantiles, so no statistics leak. **But
   concede the real caveat:** you can't filter production requests by price, so your test set is a
   slightly cleaner population than reality.
5. Why remove outliers before the split rather than after?

**Prove it.** Explain all three leaks you found and fixed, with numbers.

---

## Day 9 — Feature selection & the `society` story ⭐

**This is your best story. Rehearse it until it's effortless.**

**Fundamentals.** Filter / wrapper / embedded selection. Tree feature importance and why it's
biased toward high-cardinality features. **High cardinality** — 660 societies over 38k rows.
**Train/serve skew.**

**Your code.** `feature_selection.py` (24 features + de-duplication),
`notebooks/error_analysis/society_ablation.ipynb`.

**The story, in order:**
1. Model trained with `society`, scored **10.65%** — looked great.
2. But production can't know the society (660 options), so the app **guessed** it from the sector.
3. That guess was right **30.9%** of the time (605 real societies collapse to 107 possible guesses).
4. Re-scored under production conditions: **14.29%** — the app was delivering far worse than advertised.
5. Retrained without it: **11.18%**.
6. So the real choice was 14.29 vs 11.18 — dropping it **improved production by ~3 points.**

**Q&A.**
1. Walk me through the society decision.
2. "But your offline metric got worse (10.65 → 11.18) — didn't you make the model worse?" → that
   comparison is invalid; 10.65 used information production never has.
3. Why is a *wrong* value worse than *no* value? → a model can't distinguish a fact from a guess.
   With no feature it leans on area/sector. With a wrong one it acts on a confident lie.
4. How would you keep society if you really wanted it? → capture it at input (bad UX), or use a
   coarse, obtainable proxy (builder tier, project age band).

**Prove it.** Tell the whole story in 90 seconds with all four numbers, no notes.

---

## Day 10 — How gradient boosting actually works

**Fundamentals — the most likely place to be caught out.**
- Decision tree: recursive splits minimising impurity/variance.
- Bagging (Random Forest): many deep trees on bootstrap samples, averaged → reduces variance.
- **Boosting:** shallow trees fitted **sequentially, each on the previous ensemble's residuals** →
  reduces bias. Know this difference cold.
- XGBoost: second-order gradients, regularised objective, pre-sorted/histogram splits.
- LightGBM: histogram binning + **leaf-wise** growth (vs level-wise) → faster, more prone to
  overfit on small data, controlled by `num_leaves` / `min_child_samples`.
- CatBoost: ordered boosting + native categorical handling.
- Key hyperparameters and what each does.

**Your code.** `mb_main.py` (candidates), `mb_tuning.py` (search spaces).

**Q&A.**
1. Explain boosting to a non-technical person, then to an ML engineer.
2. Bagging vs boosting — which reduces bias, which reduces variance?
3. Why did you drop RandomForest? → slowest to tune (~28 of 40 min) *and* weakest. In industry RF
   is a quick baseline, not a tuned contender on tabular data.
4. Why drop stacking? → it beat the best single model by 0.1% — noise. Not worth the complexity;
   and a single model is far easier to explain with SHAP.
5. Why did LightGBM win? → honestly: on this data with this budget. It won in both the 25- and
   24-feature runs, and by a clear margin (11.41 vs 12.72/12.82), so it's not just noise.
6. Why ordinal-encode categoricals for trees rather than one-hot?

**Prove it.** Draw boosting on paper: 3 sequential trees, showing what each fits.

---

## Day 11 — Training methodology & your audit story ⭐

**Fundamentals.** Train/validation/test and *why three*. Cross-validation. Grid vs random search
(Bergstra & Bengio: random wins when few hyperparameters matter). **Selection bias** — using data
to choose makes that data optimistic.

**Your code.** `create_train_val_test_split`, `RandomizedSearchCV(n_iter=25, cv=KFold(3))`.

**Q&A.**
1. Why three splits, not two? → every decision made on data makes that data optimistic. Tuning uses
   CV on train; family selection uses validation; test is touched **once**.
2. **The audit story:** *"I was selecting the model family on test MAPE. I fixed it with a 60/20/20
   split — my number moved 11.18% → 11.39%. That 0.2 points was inflation I'd been unknowingly
   reporting."*
3. **The duplicate story:** de-duplication ran on 35 columns, but narrowing to 24 created new
   duplicates — **252 test rows (3.23%) had an identical twin in train**, memorisable. Fixed by
   de-duplicating on the final feature set. Overlap now zero, enforced by a test.
4. "Test came out better than validation (11.57 vs 11.41) — suspicious?" → no: two different
   7,648-row samples differ by tenths naturally. Test being *much worse* would indicate overfitting
   to validation.
5. Why 3 folds and 25 iterations? → compute budget across three families. 5-fold would be more
   stable; a fair trade to concede.
6. Why stratify on price quintiles? Isn't stratifying on the target leakage? → it affects only
   which rows go where, not what the model sees. But it does make test slightly optimistic vs a
   truly random future sample.

**Prove it.** Explain what changes if you delete the validation set.

---

## Day 12 — Evaluation, baselines & interpretation

**Fundamentals.** Metric suite (MAPE/MAE/RMSE/R²) and what each hides. **Why a baseline is
mandatory.** Residual analysis. SHAP: additive per-feature attribution, TreeExplainer.
Retransformation bias: `expm1(E[log y])` is the conditional **median**, not the mean (Jensen's
inequality) — know the term and the fix (Duan's smearing estimator).

**Your code.** `notebooks/error_analysis/baseline_comparison.ipynb`, `error_analysis.ipynb`,
the SHAP section of `insight_module.ipynb`.

**The baseline ladder:**

| Approach | MAPE | Knows |
|---|---|---|
| Global median price | 72.3% | nothing |
| Sector median price | 47.8% | location only |
| Global median ₹/sqft × area | 33.3% | size only |
| Sector median ₹/sqft × area | 23.6% | location + size |
| + property type | 21.4% | |
| **+ bedrooms (best lookup)** | **19.4%** | |
| **Your model** | **11.6%** | 24 features + interactions |

**Q&A.**
1. "Why not just a spreadsheet of sector averages?" → the best lookup gets 19.4%; the model gets
   11.6% — **40% less error.**
2. Why does area alone (33.3%) beat sector alone (47.8%)? → size dominates price, independently
   confirming SHAP's ranking.
3. Where is the model weakest? → sparse sectors (rajendra park ~85% MAPE on 7 listings) and the
   luxury tail. It's a data-coverage problem as much as a model problem.
4. What does SHAP actually compute? Global vs local?
5. You trained on log price and report MAPE on the original scale — any problem? ← **prepare cold.**

**Prove it.** Recite the baseline ladder and the 40% figure.

---

## Day 13 — Weaknesses, trade-offs & your four stories

**Your four stories (rehearse each to 90 seconds):**
1. **`society` / train-serve skew** — 31% guess accuracy, 14.29% vs 11.18%
2. **Selection bias** — found by self-audit, three-way split, 11.18 → 11.39
3. **Duplicate leakage** — 252 test rows with train twins, now zero, test-enforced
4. **Baseline comparison** — 19.4% → 11.6%, 40% error reduction

**Still-open weaknesses (volunteer these):**
- Target is asking price, not sale price
- Outlier rules use the target (via fixed constants — state the caveat)
- Area conversion ratios learned on the full dataset
- Prediction intervals are global, not conditional (one ±band for a ₹0.5 Cr flat and a ₹15 Cr villa)
- Single city, single time snapshot — no temporal validation
- `dvc.yaml` has no train stage, so the full chain isn't `dvc repro`-able

**Format for each:** flaw → honest impact → fix. Three sentences, no defensiveness.

---

## Day 14 — Full mock interview

45 minutes, end to end, recorded. Structure: 60-second pitch → data pipeline walkthrough →
"tell me about a hard decision" → "what's wrong with your model?" → "what would you do with another
month?" Play it back and grade yourself on clarity, honesty and *whether you used your numbers*.

---

## The seven questions you must never fumble

1. Why MAPE, and what's wrong with MAPE?
2. What is data leakage — and where did you find it in your own project?
3. Why did you drop `society`?
4. How do you know your model beats a lookup table?
5. Bagging vs boosting?
6. Why three splits instead of two?
7. What's the biggest weakness of this project?
