# Data Cleaning — Process Documentation

The raw data is scraped from 99acres and is messy because it's typed by thousands of different
agents. Cleaning happens in **two stages**:

1. **Stage 1 — Individual cleaning:** each property type (flats, builder floors, houses) is
   cleaned separately, because they're scraped into separate files.
2. **Stage 2 — Post-merge cleaning:** after the three are combined, we clean the columns that
   only make sense on the full dataset (e.g. the `sector` location column).

---

## Stage 1 — Individual Property Cleaning

**Files:**
- `src/data_cleaning/residential_apartment_cleaning.py` (flats)
- `src/data_cleaning/indepedent_builder_floor.py` (builder floors)
- `src/data_cleaning/house_cleaning.py` (independent houses)

All three follow the **same recipe** (with tiny type-specific tweaks). Each step below shows a
*before → after* example.

### Step 1 — Rename columns to what they really are
The raw column names are misleading: `price` is fine, but `area` actually holds the price-per-sqft.
We rename them so the code reads clearly.
- `price` → `price_in_cr`
- `area` → `price_per_sqft`

### Step 2 — Normalise the society (project) name
Same project is written in different cases/spacing. We trim spaces and lowercase it so they match.
- `"  Eldeco Accolade "` → `"eldeco accolade"`

### Step 3 — Drop rows with no usable price *(logged)*
Some listings have no real price — they say "Price on Request", a placeholder, or (for houses) a
range. We can't train on those, so we drop them and log how many.
- `"Price on Request"` → dropped
- `"25,000"` / `"70,000"` (placeholder) → dropped
- `"17.57 - 25.57 Cr"` (a range, houses only) → dropped

> Example log line: `Price filter / column cleaning dropped 24 rows — 28531 -> 28507.`

### Step 4 — Convert price to a single number (crores)
Prices come as text in two units — crore and lakh. We turn them into one consistent number in
crores (1 lakh = 0.01 crore). We keep enough decimals so 32.7 and 32.8 lakh stay different.
- `"2.5 Cr"` → `2.5`
- `"32.8 Lac"` → `0.328`

### Step 5 — Convert price-per-sqft to a number
The text like `₹9,850 /sqft` becomes a plain number.
- `"₹9,850 /sqft"` → `9850`

For **builder floors and houses** there are two extra cases handled:
- Values in **lakhs** (an `L`) are multiplied by 100,000 → `"₹1.2 L /sqft"` → `120000`
- Values per **square yard** (`/sqyd`) are converted to per-sqft by dividing by 9 (1 sqyd ≈ 9 sqft)

### Step 6 — Clean the categorical fields
- **Facing / Furnishing:** the site uses `"0"` to mean "not provided" — we relabel it.
  - `"0"` → `"not available"`
- **Age / Possession:** stored as codes 0–6 — we map them to readable labels.
  - `1` → `"1 to 5 Year Old"`, `6` → `"0 to 1 Year Old"`, `5` → `"Under Construction"`

### Step 7 — Remove duplicate listings *(logged)*
The same flat is often posted by several agents, creating duplicate rows. If we leave them, the
same property can land in both training and test data and the model "cheats" (this is data
leakage). There's no flat number in our data, so we treat two rows as the same property when a
set of identifying details **plus the price** all match:
`property_name + society + price + areaWithType + floorNum + facing + overlooking`.
We keep the first and drop the rest.

> Example log line: `Deduplication dropped 74 rows — 28507 -> 28433.`

### Step 8 — Save + full transparency
Every step that drops rows logs the count and reason, so the row count is always auditable —
nothing disappears silently. The cleaned file is then saved to `data/data_cleaning/`.

**Net effect (flats example):** 28,531 → 28,507 (price filter) → 28,433 (dedup).

---

## Stage 2 — Post-Merge Cleaning (in Feature Engineering)

After the three cleaned files are merged into one dataset (~39,600 rows), we clean columns that
need the whole dataset to make sense. The first big one is **location**.

### Sector / Location cleaning
Location is the **biggest driver of price**, so this column had to be clean — but it was the
messiest, with 320+ distinct values for a city that really has ~115 sectors. The mess came from
inconsistent agent typing. We cleaned it in passes:

1. **Lowercase and remove "gurgaon"** (it's on every row → no information).
2. **Strip building-block bits** — `"Block C"`, `"A Block"`, `"Pocket D"` describe a spot
   *inside* a society, not a locality, so they're noise.
   - `"Block S Uppals Southend"` → `"uppals southend"`
   - `"Block C, Sushant Lok Phase 1"` → `"sushant lok phase 1"`
3. **Standardise sector numbers and drop letter suffixes.**
   - `"sector33"`, `"sector-33"` → `"sector 33"`
   - `"sector 89a"`, `"sector 89 a"` → `"sector 89"`
4. **Keep genuine named localities** (no sector number) as-is.
   - `"dlf phase 5"`, `"nirvana country"` → unchanged
5. **Keep Sohna's sectors separate** from Gurgaon's — Sohna is a different, cheaper township
   with its own numbering, so merging would corrupt the price signal.
   - `"sector33 sohna"` → `"sohna sector 33"` (not `"sector 33"`)
6. **Bucket the rare long tail** (localities with <10 listings) into a single `"other"` group,
   so we don't end up with hundreds of tiny categories.

**Net effect:** 320+ messy values → ~130 clean categories (~115 Gurgaon sectors + Sohna sectors
+ ~40 named localities + `other`).

### Area cleaning (one comparable size column)
A property's size is advertised in one of three ways, and they are **not comparable** — for the
*same* home, **carpet < built-up < super built-up** (carpet = usable floor; super built-up also
counts walls, lobby, stairs). In our data each listing gives only one of these, so the raw size
mixes apples and oranges (1,200 sqft carpet is a bigger home than 1,200 sqft super built-up).

So we convert every row to **one common unit — super built-up square feet:**
1. Rows already in super built-up (≈78%) are **kept as-is**.
2. Carpet and built-up rows are **scaled up** using conversion ratios.
3. The ratios are **learned from the data** (rows that list both a primary area and a carpet
   area), not guessed — they came out at carpet ≈ 0.725× and built-up ≈ 0.925× of super built-up,
   which matches real-estate norms (super built-up carries ~25–30% extra "loading").

**Net effect:** one clean numeric `area` column for every property, all on the same scale.

#### Worked example

**Step 1 — how the ratio is learned (from rows that list both numbers):**
Some listings give the primary area *and* a separate carpet area, which shows the relationship
directly:
> A listing says **super built-up = 1,380 sqft** and **carpet = 1,000 sqft**
> → ratio = carpet ÷ super = 1000 ÷ 1380 ≈ **0.72**

Doing this across ~15,000 such rows and taking the median gives **0.725** (and similarly
built-up ÷ super = **0.925**). The ratios are measured from data, not guessed.

> **Why only ~15,000 rows (not all 39,600)?** To *learn* the ratio, a row must list **both** a
> super built-up area and a carpet area — only then can you compute carpet ÷ super. Of the 30,728
> super built-up rows, only ~15,161 also carry a separate carpet figure (the rest list just one
> number). So ~15k is simply the count of rows where both measurements coexist. The remaining
> rows can't help *learn* the ratio, but they are still *converted* in the apply step.

**Step 2 — how the conversion is applied (target unit = super built-up sqft):**

| Property | Listed as | Conversion | Final `area` |
|----------|-----------|------------|--------------|
| A | `1000 sqft Carpet` | 1000 ÷ 0.725 | **1,379** |
| B | `1200 sqft Builtup` | 1200 ÷ 0.925 | **1,297** |
| C | `1500 sqft Superbuiltup` | kept as-is | **1,500** |

So a 1,000 sqft *carpet* listing is actually the same size as a ~1,379 sqft *super built-up*
listing — after conversion they all sit on one comparable scale.

> **In one line:** `area` is each property's size expressed in a single common unit (super
> built-up sqft) — rows already in that unit are kept, and carpet/built-up rows are converted up
> using ratios learned from the data.

> More Stage-2 steps (floor split, parking, amenities) follow the same idea — done after merge
> because they feed analytics and the price model together.

### Other Stage-2 feature steps (summary)
Each adds a new column without removing anything:

| Raw column | New column(s) | What it does |
|------------|---------------|--------------|
| `floorNum` | `floornum`, `total_floor` | splits "3 of 14 Floors"; G/L/B → 0/-1/-2; `total_floor` 0 → NaN |
| `agePossession` | `age_possession_category` | readable labels (New Property, Relatively New, …) |
| `cornerProperty` | `is_corner` | `Y` → 1, blank → 0 |
| `parking` | `covered_parking`, `open_parking`, `total_parking` | parses `{"C":1,"O":1}`; `["N"]` → 0; NaN kept |
| `features` | `luxury_count` | counts premium amenities (after dropping free-text location noise) |
| `overlooking` | `ov_park`, `ov_pool`, `ov_club`, `ov_main_road`, `ov_sea`, `ov_others` | multi-hot view flags |
| `address` | `sector` | clean sector / locality (see Sector section) |

### Keep-all in FE, drop in preprocessing
Feature engineering **only adds columns — it never drops any**. The reason: the enriched output is
shared by several modules:
- **price model** → engineered numeric columns
- **analytics** → `society`, `sector`, `price_per_sqft`, `description`
- **recommender** → `nearbyLocations`, `description`, `society`, `link`

Dropping is model-specific, so it happens in **preprocessing** (the price-model path only). Columns
to drop there once features are built:
`property_name`, `address`, `link`, `areaWithType`, `carpetArea`, `floorNum`, `cornerProperty`,
`parking`, `overlooking`, `features`, `agePossession`, `description`, `nearbyLocations`.

> `description` and `nearbyLocations` are kept through FE for the analytics / recommender modules,
> and only dropped on the price-model branch.

---

## Quick interview summary
> "Cleaning is in two stages. First I clean each property type separately — standardise prices
> into one numeric unit, parse price-per-sqft, map coded categorical fields to labels, and remove
> duplicate repostings of the same flat to avoid leakage — logging every dropped row so it's
> auditable. Then, after merging the three types, I clean dataset-wide columns; the biggest is
> location, where I collapsed 320+ messy values into ~130 clean sectors and localities, since
> location is the top price driver."
