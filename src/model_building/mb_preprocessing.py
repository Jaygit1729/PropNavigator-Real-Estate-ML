
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


#  Feature Definitions

numerical_features = [

    "total_area_sqft",
    "plot_area_missing",
    "bathrooms",
    "area_per_bedroom",
    "servant_room",
    "bedrooms",
    "pooja_room"
]

categorical_features = [

    "property_type",
    "sector",
    "society",
    "balcony",
    "luxury_category",
    "furnishing_type",
    "facing",
    "age_possession",
    
]

#  Tree Preprocessor 

def get_tree_preprocessor():
    """
    Preprocessor for Tree Models:
    - Ordinal encode ALL categorical features
    - Numerical features passed through unchanged
    - No scaling
    - No target encoding
    """

    tree_preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                ),
                categorical_features
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    )

    return tree_preprocessor



# 3 Linear Preprocessor 

def get_linear_preprocessor():
    """
    Preprocessor for Linear Models:
    - StandardScaler for numerical features
    - OneHotEncoding for categorical features
    """

    linear_preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return linear_preprocessor


#  Target Transformation

def transform_target(y):
    """Log(1+y) transformation"""
    return np.log1p(y)


def inverse_transform_target(y_log):
    """Inverse of log(1+y)"""
    return np.expm1(y_log)