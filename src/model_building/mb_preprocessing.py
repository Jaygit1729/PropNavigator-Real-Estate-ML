# src/model_building/mb_preprocessing.py

import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def get_feature_lists(X):
    """
    Derives numerical and categorical feature lists dynamically
    from the dataframe passed in.

    This avoids hardcoding column names — if feature selection
    changes which columns are selected, the preprocessor adapts
    automatically without any manual updates.

    Args:
        X: Feature dataframe after feature selection

    Returns:
        numerical_features: list of numeric column names
        categorical_features: list of categorical/object column names
    """
    numerical_features = X.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()

    return numerical_features, categorical_features


def get_tree_preprocessor(numerical_features: list, categorical_features: list):
    """
    Preprocessor for tree-based models (RandomForest, XGBoost):
    - Ordinal encodes all categorical features
    - Passes numerical features through unchanged (no scaling needed
      since tree models are scale-invariant)
    - unknown_value=-1 handles unseen categories at inference time
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


def get_linear_preprocessor(numerical_features: list, categorical_features: list):
    """
    Preprocessor for linear models (SVR):
    - StandardScaler for numerical features (linear models are
      sensitive to feature scale)
    - OneHotEncoder for categorical features
    - drop='first' avoids multicollinearity (dummy variable trap)
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


def transform_target(y):
    """
    Applies log1p transformation to the target variable.
    log1p compresses the price distribution, reducing the influence
    of expensive outliers on model training.
    """
    return np.log1p(y)


def inverse_transform_target(y_log):
    """
    Reverses log1p transformation using expm1.
    Applied after prediction to get back to original price scale
    before computing evaluation metrics.
    """
    return np.expm1(y_log)