# src/feature_engineering/feature_eng.py

import re
import ast
import pandas as pd
import numpy as np
from collections import Counter
from src.logger_utils import setup_logger


logger = setup_logger(__name__, 'logs/feature_eng.log')
logger.info("Logging set up successfully for Feature Engineering Module.")


# Matches building-block that hide the real locality,

# e.g. "block c", "a block", "pocket d", "block m 2"

BLOCK_RE = re.compile(r'\b([a-z]\s*block|block\s*[a-z]?\s*\d?|pocket\s*[a-z]?\s*\d?)\b')

# Localities with fewer than this many listings are grouped into "other".

RARE_SECTOR_THRESHOLD = 10

# Same locality written in different ways -> one canonical label.

SECTOR_ALIASES = {
    'dlf city phase 1':      'dlf phase 1',
    'dlf city phase 2':      'dlf phase 2',
    'sushant lok 3':         'sushant lok phase 3',
    'sushant lok phase iii': 'sushant lok phase 3',
}


def _clean_sector_value(text):
    """
    Convert a raw Gurgaon address into a standardized sector/locality label

    1. Clean text.
    2. Find sector number.
    3. Return "sector X" or "sohna sector X".
    4. Otherwise return locality name.

    Examples:
    --------
    'Block C, Sector 56, Gurgaon' -> 'sector 56'
    'Sohna Sector 36'             -> 'sohna sector 36'
    'DLF Phase 5, Gurgaon'        -> 'dlf phase 5'
    """

    if pd.isna(text):
        return np.nan

    # basic cleaning

    text = str(text).lower()
    text = text.replace('gurgaon', '')
    text = text.replace('-', ' ')

    # remove block/pocket information
    # Example:
    # 'Block C, Sector 56' -> 'Sector 56'
    text = BLOCK_RE.sub(' ', text)

    # normalize spaces

    text = text.replace(',', ' ')
    text = re.sub(r'\s+', ' ', text).strip()

    if not text:
        return np.nan

    # check if it belongs to Sohna
    # Sohna Sector 36 and Gurgaon Sector 36 should be treated as different locations.

    is_sohna = 'sohna' in text

    # extract sector number

    sector_match = re.search(r'sector\s*(\d+)', text)

    if sector_match:
        sector_no = sector_match.group(1)

        if is_sohna:
            return f"sohna sector {sector_no}"

        return f"sector {sector_no}"

    # not a sector — return locality name, merging known duplicate spellings

    return SECTOR_ALIASES.get(text, text)

def _extract_sector(df: pd.DataFrame):

    """
    Create a standardized sector/locality feature from the raw address.

    Steps:
    ------
    1. Extract a clean sector/locality label from each address.
       Examples:
       - 'Block C, Sector 56, Gurgaon' -> 'sector 56'
       - 'Sohna Sector 36'             -> 'sohna sector 36'
       - 'DLF Phase 5, Gurgaon'        -> 'dlf phase 5'

    2. Group infrequent sectors/localities into a single 'other' bucket
       to reduce cardinality and improve model generalization.

    3. Return the dataframe with the transformed sector column.
    """

    df['sector'] = df['address'].map(_clean_sector_value)

    sector_counts = df['sector'].value_counts()

    df['sector'] = df['sector'].where(
        df['sector'].map(sector_counts) >= RARE_SECTOR_THRESHOLD,
        'other'
    )

    logger.info(
        f"Sector extraction: {df['sector'].nunique()} unique sectors."
    )

    return df

def _extract_floor_info(df: pd.DataFrame):
    """
    Extract floor number and total floors from the `floorNum` column.

    'floorNum' looks like "3 of 5 Floors". We extract:
    - `floornum`   : numeric floor (Ground=0, Lower Ground=-1, Basement=-2)
    - `total_floor`: total floors in the building (a 0 total is invalid -> NaN)
    """
    df = df.assign(
        floornum=lambda df_: pd.to_numeric(
            df_['floorNum'].str.split('of').str.get(0).str.strip()
                .str.replace('L', '-1', regex=False)   # Lower Ground
                .str.replace('G', '0',  regex=False)   # Ground
                .str.replace('B', '-2', regex=False),  # Basement
            errors='coerce'
        ),
        total_floor=lambda df_: pd.to_numeric(
            df_['floorNum'].str.split('of').str.get(1).str.extract(r'(\d+)')[0],
            errors='coerce'
        ).replace(0, np.nan),   # "G of 0 Floors" -> invalid total -> missing
    )

    logger.info("Floor info extracted: floornum + total_floor.")
    return df


def _process_area(df: pd.DataFrame):
    
    """
        Standardize all property areas to super built-up square feet.
        The raw data has multiple area columns with different types (super built-up, built-up, carpet)

        Example:
    --------
    1000 sqft Carpet Area      -> 1380 sqft Super Built-up
    1300 sqft Built-up Area    -> 1405 sqft Super Built-up
    1400 sqft Super Built-up   -> 1400 sqft Super Built-up
    """

    # Extract numeric area and area type from the listing text.
    #
    # Example:
    # "1450 sqft Builtup Area"
    #   -> area = 1450
    #   -> type = builtup
    

    area = pd.to_numeric(
        df['areaWithType']
        .str.extract(r'([\d,\.]+)\s*sqft')[0]
        .str.replace(',', '', regex=False),
        errors='coerce'
    )

    area_type = (
        df['areaWithType']
        .str.extract(r'sqft\s+(.*)$')[0]
        .replace({
            'Super Area': 'super',
            'Superbuiltup Area': 'super',
            'Builtup Area': 'builtup',
            'Carpet Area': 'carpet'
        })
    )

    carpet_area = pd.to_numeric(df['carpetArea'], errors='coerce')

    valid = area.between(100, 50000) & carpet_area.between(100, 50000)

    super_rows = valid & (area_type == 'super')
    builtup_rows = valid & (area_type == 'builtup')

    carpet_to_super = (
        carpet_area[super_rows] / area[super_rows]
    ).median()

    carpet_to_builtup = (
        carpet_area[builtup_rows] / area[builtup_rows]
    ).median()

    builtup_to_super = carpet_to_super / carpet_to_builtup

    # Convert every area measurement to super built-up sqft.
    # Super area rows remain unchanged.
    # Carpet and built-up rows are scaled using learned ratios.

    multiplier = pd.Series(1.0, index=df.index)

    multiplier[area_type == 'carpet'] = 1 / carpet_to_super
    multiplier[area_type == 'builtup'] = 1 / builtup_to_super

    df = df.assign(
        area=(area * multiplier).round()
    )

    logger.info(
        f"Area standardized to super built-up sqft "
        f"(carpet/super={carpet_to_super:.3f}, "
        f"builtup/super={builtup_to_super:.3f})"
    )

    return df



AGE_LABEL_MAP = {
    'Under Construction': 'Under Construction',
    '10+ Year Old': 'Old Property',
    '5 to 10 Year Old': 'Moderately Old',
    '1 to 5 Year Old': 'Relatively New',
    '0 to 1 Year Old': 'New Property',
    'Undefined': 'Undefined',
}


def _process_age_possession(df: pd.DataFrame):
    """Add a readable `age_possession_category` column (keeps the original agePossession)."""
    df = df.assign(
        age_possession_category=lambda df_: df_['agePossession'].map(AGE_LABEL_MAP).fillna('Undefined')
    )
    logger.info("Age possession categorized: age_possession_category.")
    return df


def _process_corner(df: pd.DataFrame):
    """
    Encode cornerProperty as a binary flag.
    The site stores 'Y' for corner units and leaves it blank otherwise, so blank = not corner.
    """
    df = df.assign(
        is_corner = lambda df_: (df_['cornerProperty'] == 'Y').astype(int)
    )
    logger.info("Corner flag created: is_corner.")
    return df

def _parse_parking(val):
    """
    Convert one raw parking value into separate covered and open
    parking counts.

    Examples:
    --------
    '{"C":2}'           -> (2, 0)
    '{"O":1,"C":1}'     -> (1, 1)
    '["N"]'             -> (0, 0)
    NaN                 -> (NaN, NaN)
    """

    # Missing parking information
    if pd.isna(val):
        return (np.nan, np.nan)

    # Convert the string representation into a Python object.
    # Example:
    # '{"C":2}' -> {'C': 2}
    try:
        obj = ast.literal_eval(val)
    except Exception:
        return (np.nan, np.nan)

    # Dictionary format stores parking counts.
    # C = Covered parking
    # O = Open parking
    if isinstance(obj, dict):
        return (
            int(obj.get('C', 0) or 0),
            int(obj.get('O', 0) or 0)
        )

    # Any non-dictionary value (e.g. ["N"]) is treated as no parking.
    return (0, 0)

def _process_parking(df: pd.DataFrame):
    """
    Create parking-related features from the raw parking column.

    Why?
    ----
    Parking information is stored as a JSON-like string and cannot
    be used directly by a machine learning model.

    We split it into:

    - covered_parking : number of covered parking spaces
    - open_parking    : number of open parking spaces
    - total_parking   : total parking spaces

    Example:
    --------
    '{"O":1,"C":2}'

    becomes

    covered_parking = 2
    open_parking    = 1
    total_parking   = 3
    """

    # Parse the raw parking string into (covered, open) counts.
    parsed = df['parking'].apply(_parse_parking)

    # Expand the tuple into separate model-friendly features and
    # compute the total number of parking spaces.
    df = df.assign(
        covered_parking=parsed.str[0],
        open_parking=parsed.str[1],
        total_parking=lambda df_: (
            df_['covered_parking'] + df_['open_parking']
        )
    )

    logger.info(
        "Parking parsed: covered_parking, open_parking, total_parking."
    )

    return df


AMENITY_MIN_FREQ = 20

# Individual binary flags — one per amenity that has a clear, independent price impact.
# Each key becomes a column (1 = property has this amenity, 0 = does not).
# Keyword matching is case-insensitive substring search against valid amenities only.

AMENITY_FLAGS = {
    'has_ac':           'centrally air conditioned',
    'has_power_backup': 'power backup',
    'has_gym':          'gym',
    'has_pool':         'swimming pool',
    'has_club':         'club',
    'has_lift':         'lift',
    'has_servant_room': 'servant',
    'is_gated':         'gated',
}


def _parse_features(val):
    """
    Convert the raw features string into a Python list.

    Example:
    --------
    "['Swimming Pool', 'Gym']"
        ->
    ['Swimming Pool', 'Gym']
    """
    if pd.isna(val):
        return []

    try:
        parsed = ast.literal_eval(val)
        return parsed if isinstance(parsed, list) else []

    except Exception:
        return []


def _process_features(df):
    """
    Create binary amenity flags from the features column.
    """

    logger.info("Starting amenity flag creation.")

    # Parse features
    parsed_features = df['features'].apply(_parse_features)


    # Create one flag at a time
    for col_name, amenity_name in AMENITY_FLAGS.items():


        flags = []

        for amenities in parsed_features:

            if any(amenity_name in amenity.lower() for amenity in amenities):
                flags.append(1)
            else:
                flags.append(0)

        df[col_name] = flags

        

    logger.info(
        "Amenity processing completed. Created %s flags.",
        len(AMENITY_FLAGS)
    )

    return df

# Views a property can overlook -> one binary flag each (the raw column is multi-valued).
OVERLOOKING_FLAGS = {
    'ov_park': 'park/garden',
    'ov_pool': 'pool',
    'ov_club': 'club',
    'ov_main_road': 'main road',
    'ov_sea': 'sea facing',
    'ov_others': 'others',
}


def _process_overlooking(df: pd.DataFrame):
    """
    Create binary flags for the different types of views a property overlooks.

    Example:
    --------
    "Park/Garden, Pool"

    becomes

    ov_park = 1
    ov_pool = 1
    ov_club = 0
    ov_main_road = 0
    """

    ov = df['overlooking'].fillna('').str.lower()

    flags = {}

    for column, keyword in OVERLOOKING_FLAGS.items():
        flags[column] = (
            ov.str.contains(keyword, regex=False)
            .astype(int)
        )

    logger.info(
        "Overlooking views encoded: park, pool, club, main road, sea."
    )

    return df.assign(**flags)


# Key Gurgaon landmarks (latitude, longitude). Distance to each becomes a feature.
# Chosen to point in different directions so together they pin down a home's location:
# the office hub, the luxury belt, the airport, and the cheap industrial edge.
GURGAON_LANDMARKS = {
    'dist_to_cyber_city': (28.4956, 77.0880),   # main office hub (jobs)
    'dist_to_golf_road':  (28.4430, 77.1030),   # posh / luxury belt
    'dist_to_airport':    (28.5457, 77.1092),   # Delhi airport (IGI)
    'dist_to_manesar':    (28.3540, 76.9370),   # far industrial edge (cheap end)
}

# Gurgaon's valid coordinate box: latitude ~28, longitude ~77.
GGN_LAT_RANGE = (28.0, 29.0)
GGN_LON_RANGE = (76.0, 78.0)


def _straight_line_km(lat, lon, place_lat, place_lon):
    """
    Approximate distance in km between two GPS points.
    Uses a flat-earth approximation (1 degree ~ 111 km) — accurate enough within a
    single small city like Gurgaon, and far simpler than the haversine formula.
    """
    return np.sqrt((lat - place_lat) ** 2 + (lon - place_lon) ** 2) * 111


def _process_geo(df: pd.DataFrame):
    """
    Turn raw GPS coordinates into distance-to-landmark features.

    Steps:
    ------
    1. Repair rows where latitude/longitude were written in the wrong order
       (a Gurgaon point looks like (28.xx, 77.xx); some rows have it flipped).
    2. Drop coordinates that still fall outside Gurgaon (set to NaN).
    3. Add one `dist_to_*` column per landmark.

    Example:
    --------
    (28.47, 77.08) -> dist_to_cyber_city = 2.3 km, dist_to_golf_road = 3.1 km, ...
    """
    lat = pd.to_numeric(df['latitude'], errors='coerce')
    lon = pd.to_numeric(df['longitude'], errors='coerce')

    # 1. Fix swapped coordinates (latitude sitting in the longitude slot and vice-versa).
    swapped = lat.between(*GGN_LON_RANGE) & lon.between(*GGN_LAT_RANGE)
    lat_fixed = lat.where(~swapped, lon)
    lon_fixed = lon.where(~swapped, lat)

    # 2. Keep only coordinates inside Gurgaon; anything else becomes NaN.
    valid = lat_fixed.between(*GGN_LAT_RANGE) & lon_fixed.between(*GGN_LON_RANGE)
    lat_fixed = lat_fixed.where(valid)
    lon_fixed = lon_fixed.where(valid)

    df = df.assign(latitude=lat_fixed, longitude=lon_fixed)

    # 3. Distance to each landmark.
    for name, (place_lat, place_lon) in GURGAON_LANDMARKS.items():
        df[name] = _straight_line_km(df['latitude'], df['longitude'], place_lat, place_lon)

    logger.info(
        "Geo features created: fixed %d swapped coords, %d invalid -> NaN, "
        "added %d distance columns (%s).",
        int(swapped.sum()),
        int((~valid).sum()),
        len(GURGAON_LANDMARKS),
        ", ".join(GURGAON_LANDMARKS),
    )
    return df


def feature_engineering(df: pd.DataFrame):
    """
    Feature engineering pipeline.
    sector + floor info + area normalization + age category + corner flag
    + parking + 8 amenity binary flags + overlooking flags + geo distance features.
    """
    try:
        logger.info(f"Feature engineering started — input shape: {df.shape}")

        df = (
            df
            .pipe(_extract_sector)
            .pipe(_extract_floor_info)
            .pipe(_process_area)
            .pipe(_process_age_possession)
            .pipe(_process_corner)
            .pipe(_process_parking)
            .pipe(_process_features)
            .pipe(_process_overlooking)
            .pipe(_process_geo)
        )

        logger.info(f"Feature engineering completed — output shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        return df
