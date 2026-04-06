import re
import ast
import pandas as pd
import numpy as np
from src.logger_utils import setup_logger
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer


logger = setup_logger('logs/feature_eng.log')
logger.info("Logging set up Successfully for Feature Engineering Module!")


sector_mapping = {
    'nirvana country': 'sector 50',
    'palam vihar': 'sector 1',
    'dlf phase 1': 'sector 26',
    'dlf phase 2': 'sector 25',
    'sector 67a': 'sector 67',
    'sushant lok phase 1': 'sector 43',
    'sector 70a': 'sector 70',
    'sohna': 'sector 33',
    'dlf phase 4': 'sector 28',
    'block c sushant lok phase 1': 'sector 43',
    'sector33 sohna': 'sector 33',
    'malibu town': 'sector 47',
    'south city 2': 'sector 49',
    'south city 1': 'sector 41',
    'sector 37d': 'sector 37',
    'sushant lok phase 3': 'sector 57',
    'dlf phase 3': 'sector 24',
    'sector 110 a': 'sector 110',
    'a block sushant lok phase 1': 'sector 43',
    'rajendra park': 'sector 105',
    'b block sushant lok phase 1': 'sector 43',
    'dlf phase 5': 'sector 43',
    'greenwood city': 'sector 45',
    'sector 37c': 'sector 37',
    'sushant lok 3 extension': 'sector 57',
    'blocka dlf city phase 1': 'sector 26',
    'krishna colony': 'sector 7',
    'uppals southend': 'sector 49',
    'suncity': 'sector 54',
    'maruti kunj': 'sector 105',
    'rosewood city': 'sector 49',
    'ardee city': 'sector 52',
    'vishnu garden': 'sector 105',
    'sector 9a': 'sector 9',
    'sushant lok phase 2': 'sector 55',
    'g block dlf city phase 1': 'sector 26',
    'laxman vihar phase 2': 'sector 3',
    'urban estate': 'sector 4',
    'sushant lok 2sector 55': 'sector 55',
    'gwal pahari': 'gwal pahari',
    'sector 10a': 'sector 10',
    'block m dlf phase 2': 'sector 25',
    'patel nagar': 'sector 15',
    'new colony': 'sector 7',
    'v block dlf phase 3': 'sector 24',
    'pocket h nirvana country': 'sector 50',
    'sector 89 a': 'sector 89',
    'surat nagar phase 1': 'sector 104',
    'block b sushant lok 3': 'sector 57',
    'mayfield garden': 'sector 51',
    'saraswati enclave': 'sector 37',
    'block e dlf city phase 1': 'sector 26',
    'sector 15 part 2': 'sector 15',
    'sector 7 extension': 'sector 7',
    'jyoti park': 'sector 7',
    'dlf cyber city': 'sector 24',
    'block a south city 2': 'sector 49',
    'block n mayfield garden': 'sector 51',
    'block m south city 1': 'sector 41',
    'mg road': 'sector 28',
    'block c ardee city': 'sector 52',
    'block c2 sector 3': 'sector 3',
    'shivpuri': 'sector 7',
    'new palam vihar': 'sector 110',
    'c block mayfield garden': 'sector 50',
    'u block dlf phase 3': 'sector 24',
    'galleria market': 'sector 28',
    'block c sushant lok phase 3': 'sector 57',
    'block g sector57': 'sector 57',
    'c block sector 43': 'sector 43',
    'laxman vihar': 'sector 4',
    'block q south city 1': 'sector 41',
    'sohna road': 'sector 48',
    'garhi harsaru': 'sector 92',
    'bhondsi': 'bhondsi',
    'shivaji nagar': 'sector 11',
    'sector7 ext': 'sector 7',
    'block g rajendra park': 'sector 105',
    'arjun nagar': 'sector 8',
    'sector 88a': 'sector 88',
    'sector 36a': 'sector 36',
    'block j pocket c palam vihar': 'sector 1',
    'jacobpura': 'sector 12',
    'farukhnagar': 'farukhnagar',
    'b block pocket b palam vihar': 'sector 1',
    'sector 95a': 'sector 95',
    'sector 17b': 'sector 17',
    'sector 15 part 1': 'sector 15',
    'baldev nagar': 'sector 7',
    'block p south city 1': 'sector 41',
    'manesar': 'sector 1',
    'imt manesar': 'sector 1',
    'imt manesar':'sector 1',
    'sector 1 imt manesar': 'sector 1',
    'sector 1a imt manesar': 'sector 1',
    'pataudi': 'sector 1',
    'pataudi road': 'sector 1',
    'pataudi road':'sector 1',
    'block s uppals southend': 'sector 49',
    'block b rajendra park': 'sector 105',
    'sector 99a': 'sector 99',
    'sector 23a': 'sector 23',
    't block dlf phase 3': 'sector 24',
    's block dlf phase 3': 'sector 24',
    'ashok vihar phase 1': 'sector 5',
    'block k1 new palam vihar phase 1': 'sector 110',
    'b1 block sector 57': 'sector 57',
    'pocket e sector 2 palam vihar': 'sector 2',
    'saraswati vihar': 'sector 28',
    'block b palam vihar': 'sector 1',
    'dayanand colony': 'sector 6',
    'block n south city 1': 'sector 41',
    'block c palam vihar': 'sector 1',
    'b block mayfield garden': 'sector 50',
    'block d sector 56': 'sector 56',
    'old dlf colony': 'sector 14',
    'sushant lok': 'sector 43',
    'block e greenwood city': 'sector 46',
    'surat nagar phase 2': 'sector 104',
    'block a greenwood city': 'sector 45',
    'madanpuri': 'sector 7',
    'block d rajendra park': 'sector 105',
    'block c 1 palam vihar': 'sector 1',
    'om nagar': 'sector 11',
    'ashok vihar phase 3 extension': 'sector 3',
    'pocket i nirvana country': 'sector 50',
    'subhash nagar': 'sector 12',
    'pratap nagar': 'sector 8',
    'south city': 'sector 49',
    'rajiv nagar': 'sector 13',
    'chakkarpur': 'sector 28',
    'huda': 'sector 14',
    'hous': 'sector 7',
    'block b sector 56': 'sector 56',
    'block i south city 1': 'sector 41',
    'sector35 sohna': 'sector 35',
    'golf course ext road': 'sector 61',
    'block h dlf city phase 1': 'sector 26',
    'devilal colony': 'sector 9',
    'sector2 sohna': 'sector 2',
    'block e sector 56': 'sector 56',
    'sector22a': 'sector 22',
    'jhajjar road': 'sector 15',
    'block n new palam vihar phase 1': 'sector 110',
    'dhankot': 'sector 102',
    'mohyal colony': 'sector 40',
    'block c sheetla colony': 'sector 5',
    'shivji park colony': 'sector 10',
    'block j ashok vihar phase 3 extension': 'sector 3',
    'block k south city 1': 'sector 41',
    'manohar nagar': 'sector 8',
    'garden estate': 'sector 24',
    'block c greenwood city': 'sector 45',
    'g block sushant lok 3': 'sector 57',
    'block f sushant lok phase iii': 'sector 57',
    'sihi village': 'sector 84',
    'phase 3 sector 22a': 'sector 22',
    'block f new palam vihar phase 1': 'sector 110',
    'rattan garden': 'sector 7',
    'sector 21 pocket a': 'sector 21',
    'civil l': 'sector 15',
    'ambedkar nagar': 'sector 9',
    'om vihar': 'sector 23',
    'model town': 'sector 11',
    'feroz gandhi colony': 'sector 9',
    'rajiv chowk': 'sector 15',
    'sheetla colony': 'sector 5',
    'block a new palam vihar phase 2': 'sector 110',
    'part 3 sector 5': 'sector 5',
    'block f sector 57': 'sector 57',
    'sector 36 sohna': 'sector 36',
    'supermart1': 'sector 28',
    'new palam vihar phase 2': 'sector 110',
    'dlf golf course': 'sector 42',
    'block t sector109': 'sector 109',
    'heritage city': 'sector 25',
    'new amanpura': 'sector 6',
    'ashok vihar': 'sector 3',
    'village': 'sector 6',
    'alipur': 'sector 105',
    'acharya puri extension': 'sector 7',
    'sector 48 49 gurugram': 'sector 48',
    'block q dlf city phase2': 'sector 25',
    'dlf city': 'sector 26',
    'kirti nagar': 'sector 15',
    'pocket j palam vihar': 'sector 1',
    'shyam kunj': 'sector 105',
    'khandsha': 'sector 37',
    'surya vihar': 'sector 21',
    'begampur khatola': 'sector 74',
    'surat nagar': 'sector 104',
    'h block sector 82': 'sector 82',
    'dwarka expressway': 'sector 109',
    'trimurti villas': 'sector 11',
    'behrampur': 'sector 71',
    'block e sheetla colony': 'sector 5',
    'carterpuri village': 'sector 23',
    'block a surya vihar': 'sector 9',
    'bhawani enclave': 'sector 9',
    'sector 5 imt manesar': 'sector 5',
    'sahib kunj': 'sector 110',
    'sadar bazar': 'sector 12',
    'shankar vihar': 'sector 104',
    'block t new palam vihar phase 2': 'sector 110',
    'ashok vihar phase 2': 'sector 5',
    'jakhopur': 'sector 6',
    'kanahi': 'sector 44',
    'islampur colony': 'sector 38',
    'new basti': 'sector 8',
    'huda sector': 'sector 55',
    'sector 2 palam vihar': 'sector 2',
    'basai village': 'sector 101',
    'baharampur naya': 'sector 61',
    'block e dharam colony': 'sector 10',
    'friends colony': 'sector 15',
    'palam vihar extension': 'sector 110',
    'islampur village': 'sector 38',
    'near medanta hospital': 'sector 38',
    'block m new palam vihar phase 1': 'sector 110',
    'sector 3a': 'sector 3',
    'sanjay colony': 'sector 6',
    'adarsh nagar': 'sector 12',
    'rajiv colony': 'sector 33',
    'shanti kunj': 'sector 110',
    'mianwali colony': 'sector 12',
    'block d sector 50': 'sector 50',
    'vatika kunj extension': 'sector 105',
    'railyway road gurgaon': 'sector 11',
    'sai kunj block c': 'sector 110',
    'block c harijan basti': 'sector 17',
    'block d greenwood city': 'sector 46',
    's block garden estate': 'sector 24',
    'block c sector 17': 'sector 17',
    'jharsa': 'sector 39',
    'laxmi garden': 'sector 11',
    'west rajiv nagar': 'sector 12',
    'ansal': 'sector 49',
    'block b sheetla colony': 'sector 5',
    'c block sector50': 'sector 50',
    'sector 72a': 'sector 72',
    'jawahar nagar': 'sector 12',
    'medicity': 'sector 38',
    'kadipur industrial area': 'sector 31',
    'sector 22b': 'sector 22',
    'a block sushant lok phase 3': 'sector 43',
    'jail road': 'sector 55',
    'block c dlf phase 1': 'sector 26',
    'hasanpur': 'sector 105',
    'sarhaul abadi village': 'sector 18',
    'dhumaspur': 'sector 68',
    'shanti nagar': 'sector 11',
    'laxman vihar industrial area': 'sector 3',
    'vijay park': 'sector 8',
    'uday nagar': 'sector 45',
    'heera nagar': 'sector 11',
    'block h south city 2': 'sector 70',
    'new palam vihar phase 3': 'sector 110',
    'block h sector57': 'sector 57',
    'gandhi nagar': 'sector 28',
    'kadipur': 'sector 10',
    'nihali colony': 'sector 110',
    'gurgaon sector 12': 'sector 12',
    'sector 63a': 'sector 63',
    'sector 82a': 'sector 82',
    'block a ashok vihar phase 3 extension': 'sector 3',
    'block c sushant lok phase  3':'sector 57',
    'sector 48  49 gurugram':'sector 49',
    'a block sushant lok phase  3':'sector 43',
    'block g new palam vihar phase 1':'sector 110',
    'new':'sector 70'
}


def _extract_sector(df):
    
    """ Extract and clean sector information """
    
    df = df.assign(

                sector = lambda df_: df_['property_name']
                        .str
                        .split('in')
                        .str
                        .get(1)
                        .str
                        .replace('Gurgaon', '')
                        .str
                        .replace("-","")
                        .str
                        .replace(",","")
                        .str
                        .strip()
                        .str
                        .lower())
    
    return df


def _apply_sector_mapping(df):
    
    """ mapping sector values """
    
    df = df.assign(
        
        sector = lambda df_: df_['sector'].replace(sector_mapping))
    
    return df


def _process_floornum(df):

    """ Handle floor number transformations"""

    df =  (df.assign(

        floornum = lambda df_: df_['floor_info']
                            .str
                            .replace("Lower Ground", "-1", regex=True)
                            .str
                            .replace("Ground", "0", regex=True)
                            .str
                            .extract(r'(-?\d+)')))
    
    return df

def _process_area_types(df):
    """Handle area type transformations"""

    def extract_area(pattern, text):
        if not isinstance(text, str):
            return None
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            sqft = match.group(1).replace(',', '')
            return float(sqft)
        return None

    return (
        df
        .assign(
            super_built_up_area = lambda df_: df_['areawithtype']
                .apply(lambda x: extract_area(r'([\d,\.]+)\s*sqft\s*\([\d,\.]+\s*sqm\)\s*Super Built-up Area', x)),

            built_up_area = lambda df_: df_['areawithtype']
                .apply(lambda x: extract_area(r'([\d,\.]+)\s*sqft\s*\([\d,\.]+\s*sqm\)\s*Built-up Area', x)),

            carpet_area = lambda df_: df_['areawithtype']
                .apply(lambda x: extract_area(r'([\d,\.]+)\s*sqft\s*\([\d,\.]+\s*sqm\)\s*Carpet Area', x)),

            plot_area = lambda df_: df_['areawithtype']
                .apply(lambda x: extract_area(r'([\d,\.]+)\s*sqft\s*\([\d,\.]+\s*sqm\)\s*Plot Area', x)),
        )
    )


def _process_additionalRoom(df):

    """
    Extract boolean flags for additional rooms
    """

    return (
        df
        .assign(
            additionalRoom = df['additional_room'].fillna('Not Available')
        )
        .assign(
            study_room = lambda df_: df_['additional_room'].str.contains('study room', case=False).astype(int),
            servant_room = lambda df_: df_['additional_room'].str.contains('servant room', case=False).astype(int),
            store_room = lambda df_: df_['additional_room'].str.contains('store room', case=False).astype(int),
            pooja_room = lambda df_: df_['additional_room'].str.contains('pooja room', case=False).astype(int),
            others = lambda df_: df_['additional_room'].str.contains('others', case=False).astype(int),
        )
    )


def _categorize_age_possession(value):
    
    if pd.isna(value):
        return "Undefined"
    
    if "0 to 1 Year Old" in value or "Within 6 months" in value or "Within 3 months" in value:
        return "New Property"
    
    if "1 to 5 Year Old" in value:
        return "Relatively New"
    
    if "5 to 10 Year Old" in value:
        return "Moderately Old"
    
    if "10+ Year Old" in value:
        return "Old Property"
    
    if "Under Construction" in value or "By" in value:
        return "Under Construction"
    
    try:
        int(value.split(" ")[-1])
        return "Under Construction"
    
    except:
        return "Undefined"
    
def _process_age_possession(df):
    
    return df.assign(age_possession_category = lambda df_: df_['property_age'].apply(_categorize_age_possession))


def _process_furnish_details(df):
    """
    Processes the furnishing details and applies KMeans clustering to classify properties
    into furnishing types: Furnished, Semi-Furnished, and Unfurnished.
    """

    furnishing_items = [
        'Fan', 'Exhaust Fan', 'Geyser', 'Light', 'Chimney', 'Wardrobe', 'AC', 'Bed',
        'Curtains', 'Dining Table', 'Modular Kitchen', 'Microwave', 'Fridge',
        'Sofa', 'Stove', 'TV', 'Washing Machine', 'Water Purifier'
    ]

    # Step 1: Clean the furnishDetails column
    df = df.assign(
        cleaned_furnishDetails=lambda df_: df_['furnishing_details'].apply(
            lambda x: (
                x.replace('[', '').replace(']', '').replace("'", "").strip()
                if isinstance(x, str) and x.strip() not in ["", "[]"]
                else "No Info"
            )
        )
    )

    # Step 2: Separate known and unknown

    known_df = df[df['cleaned_furnishDetails'] != "No Info"].copy()
    unknown_df = df[df['cleaned_furnishDetails'] == "No Info"].copy()

    # Step 3: Feature extraction helper

    def get_furnishing_count(details, furnishing):
        if not isinstance(details, str) or details.strip().lower() == "no info":
            return 0
        if f"No {furnishing}" in details:
            return 0
        parts = details.split(',')
        for item in parts:
            item = item.strip()
            if furnishing in item:
                tokens = item.split()
                if tokens[0].isdigit():
                    return int(tokens[0])
                else:
                    return 1
        return 0

    # Step 4: Create furnishing item features

    for item in furnishing_items:
        known_df[item] = known_df['cleaned_furnishDetails'].apply(lambda x: get_furnishing_count(x, item))

    # Step 5: KMeans clustering

    scaler = StandardScaler()
    scaled_known = scaler.fit_transform(known_df[furnishing_items])

    kmeans = KMeans(n_clusters=3, random_state=42)
    known_df['furnish_cluster'] = kmeans.fit_predict(scaled_known)

    # Step 6: Analyze cluster centroids

    cluster_centroids = pd.DataFrame(kmeans.cluster_centers_, columns=furnishing_items)
    cluster_centroids['total_items'] = cluster_centroids.sum(axis=1)
    cluster_centroids.index.name = "cluster_id"

    # Step 7: Map clusters to labels

    cluster_map = (
        cluster_centroids['total_items']
        .sort_values(ascending=False)
        .reset_index()
        .assign(furnishing_type=['Furnished', 'Semi-Furnished', 'Unfurnished'])
    )

    known_df = known_df.merge(
        cluster_map[['cluster_id', 'furnishing_type']],
        left_on='furnish_cluster',
        right_on='cluster_id',
        how='left'
    )

    # Step 8: Assign unknowns

    unknown_df['furnishing_type'] = 'Unfurnished'

    # Step 9: Combine all back

    final_df = pd.concat([known_df, unknown_df], axis=0, sort=False)

    # Step 10: Merge back on property_id to ensure alignment

    final_df = final_df[['property_id', 'furnishing_type']]
    df = df.merge(final_df, on='property_id', how='left')

    return df


def _compute_luxury_score(df):
    """
    Computes a price-driven luxury score based on observed
    pricing premium of each amenity.
    """

    try:
        

        df = df.copy()

        # Step 1: Parse amenities list safely

        df["features_list"] = df["features"].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) else []
        )

        # Step 2: Multi-hot encode amenities
        

        mlb = MultiLabelBinarizer()
        binary_features = mlb.fit_transform(df["features_list"])

        features_encoded = pd.DataFrame(
            binary_features,
            columns=mlb.classes_,
            index=df.index
        )

        
        # Step 3: Compute price-driven premium weights

        amenity_weights = {}

        for col in features_encoded.columns:

            present_mask = features_encoded[col] == 1
            absent_mask = features_encoded[col] == 0

            # Ignore extremely rare amenities
            if present_mask.sum() > 30:

                median_present = df.loc[present_mask, "price_per_sqft"].median()
                median_absent = df.loc[absent_mask, "price_per_sqft"].median()

                premium = median_present - median_absent

                # Only keep positive pricing impact
                amenity_weights[col] = max(premium, 0)

            else:
                amenity_weights[col] = 0

        weights_series = pd.Series(amenity_weights)

        # Step 4: Normalize weights

        if weights_series.max() > 0:
            weights_series = weights_series / weights_series.max()

       
        # Step 5: Compute final luxury score
        

        df["luxury_score"] = np.log1p(
            features_encoded.mul(weights_series, axis=1).sum(axis=1)
        )

        return df

    except Exception as e:
        print(f"Error computing luxury score: {e}")
        return df

    except Exception as e:
        logger.error(f"Luxury score computation failed: {e}")
        return df



def _reorder_columns(df):
    desired_order = [
         'property_id','property_type','link','society', 'sector', 'price_in_cr', 'price_per_sqft',
         'areawithtype', 'plot_area','super_built_up_area', 'built_up_area','carpet_area',
         'bedrooms','bathrooms','balcony','floornum','study_room', 'servant_room', 'store_room', 
         'pooja_room', 'others','facing','furnishing_type', 'age_possession_category', 'features',
         'luxury_score'
    ]      
    
    return df[desired_order]


def feature_engineering(df):
    """ Main feature engineering pipeline """

    try:

        df = (
            df
            .pipe(_extract_sector)
            .pipe(_apply_sector_mapping)
            .pipe(_process_floornum)
            .pipe(_process_area_types)
            .pipe(_process_additionalRoom)
            .pipe(_process_age_possession)
            .pipe(_process_furnish_details)
            .pipe(_compute_luxury_score)
            .drop(
                columns=[
                    'property_name',
                    'additional_room', 'property_age', 'nearby_location',
                     'furnishing_details',
                    'cleaned_furnishDetails', 'features_list'
                ],
                errors='ignore'
            )
            .pipe(_reorder_columns)
        )

        logger.info(f"Feature engineering completed. Final shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        return df
