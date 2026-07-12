# Data Overview & Understanding

## 1. Source

All data is scraped from **99acres.com** for **Gurgaon**, across three residential property
types. Each row is one property listing.

---

## 2. Dataset Shapes

### Raw (as scraped) — `data/web_scraping/`

| File | Property type | Rows | Columns |
|------|---------------|------|---------|

| `flats_gurgaon.csv` | Flats / apartments | 28,531 | 24 |

| `builder_floor_gurgaon.csv` | Independent builder floors | 9,210 | 24 |

| `independent_house_gurgaon.csv` | Independent houses / villas | 2,054 | 25 |

### Cleaned — `data/data_cleaning/`

| File | Rows (after cleaning) | Dropped | Why dropped |

|------|-----------------------|---------|-------------|

| `cleaned_residential_apartment.csv` | 28,433 | 98 | 24 invalid price + 74 duplicate repostings |

| `cleaned_independent_builder_floor.csv` | 9,168 | 42 | 9 invalid price + 33 duplicate repostings |

| `cleaned_independent_houses.csv` | 2,020 | 34 | 16 invalid/range price + 18 duplicate repostings |

After cleaning, the houses file is also reduced from 25 → **24 columns** (the leftover raw
`pricePerSqft` column is dropped), so all three cleaned files share an identical 24-column schema.

**Combined (all three merged):** ~39,600 rows used for feature engineering and modelling.

> Note: the raw `independent_house` file has one extra column (`pricePerSqft`); it is dropped
> during cleaning so the three cleaned datasets line up exactly. This is why combining them is
> straightforward.

---

## 3. Data Dictionary (what each feature means)

Examples and fill-rates below are from the raw flats file (the schema is the same across types).

| Column | Meaning | Example | Fill % | Notes / use |

|--------|---------|---------|--------|-------------|

| `property_id` | Unique listing ID from 99acres | `M89360483` | 100% | Row identifier; not a model feature |

| `link` | URL of the listing | `https://www.99acres.com/...` | 100% | Reference only |

| `property_name` | Listing heading (BHK + locality) | `2 BHK Flat in Sohna, Gurgaon` | 100% | Source for **sector** extraction |

| `society` | Project / building name | `Eldeco Accolade` | ~100% | High-cardinality location feature (1,298 values) |

| `price` | Listed price as text (target) | `1.25 Cr`, `32.8 Lac` | 100% | Cleaned to **`price_in_cr`** (numeric, in crores) — **the target** |

| `area` | Price per sqft as text | `₹9,850 /sqft` | 99% | Cleaned to **`price_per_sqft`** (numeric). Note: derived from price/area, so collinear with target |

| `areaWithType` | Size + area type | `1269 sqft Superbuiltup Area` | 100% | Main size feature; types: Superbuiltup / Carpet / Builtup |

| `carpetArea` | Usable internal area (sqft) | `726.0` | 64% | Numeric; partially filled |

| `bedRoom` | Number of bedrooms (BHK) | `2` | 100% | Core feature |

| `bathroom` | Number of bathrooms | `2` | 100% | Core feature |

| `balcony` | Number of balconies | `4` | 100% | `0` means none |

| `address` | Locality / sector text | `Sohna, Gurgaon` | 100% | Source for **sector** extraction (location = top price driver) |

| `floorNum` | Floor of total floors | `2 of 18 Floors` | 100% | Split into floor + total in feature engineering |

| `facing` | Direction the unit faces | `North-East` | 100% | `0` recoded to "Not Available" |

| `overlooking` | What the unit overlooks | `Park/Garden, Main Road` | 86% | Multi-value text |

| `agePossession` | Age / possession status (coded) | `1` | 100% | Codes mapped to labels (e.g. 1 = "1 to 5 Year Old") |

| `cornerProperty` | Corner unit flag | `Y` | 65% | Only `Y` present; missing treated as `N` |

| `furnishing` | Furnishing status | `Semi-Furnished` | 100% | Furnished / Semi / Unfurnished; `0` → "Not Available" |

| `parking` | Parking (JSON) | `{"C":1}` | 99% | `C` = covered, `O` = open; parsed in FE |

| `nearbyLocations` | Nearby landmarks (list) | `['GD Goenka University', ...]` | 100% | Text list; useful for recommender/analytics |

| `description` | Free-text listing description | `The flat is north-East-Facing...` | 100% | Text; for recommender / NLP features |

| `features` | Amenities (list) | `['Full Power Backup', 'Gym', ...]` | 98% | Parsed into amenity flags/counts in FE |

| `latitude` | Geo latitude | `28.259939` | 100% | Spatial feature (distance, clustering) |

| `longitude` | Geo longitude | `77.06448` | 100% | Spatial feature |

> The raw `independent_house` file also has a `pricePerSqft` column, but it is **dropped during
> cleaning** (redundant with `price_per_sqft`), so it does not appear in the cleaned datasets.

---

## 4. Key points to remember

- **Target:** `price` → cleaned to `price_in_cr` (numeric, crores).

- **Top price driver:** location — captured via `address` / `property_name` → cleaned into a
  `sector` feature (~130 clean categories from 320+ messy values).

- **Size feature:** `areaWithType` (mixed area types) + `carpetArea`.

- **Care items:** `price_per_sqft` is derived from price (collinear — keep for checks, not as a
  naive predictor); `carpetArea` is only ~64% filled; `cornerProperty` is effectively Y/N.

- **Every dropped row is logged** in the cleaning pipeline, so the row count is fully auditable.
