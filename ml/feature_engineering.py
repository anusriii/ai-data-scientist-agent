import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.base import BaseEstimator, TransformerMixin


# ============================================================
# CONSTANT / LOW VARIANCE FEATURE REMOVER
# ============================================================

class LowVarianceRemover(
    BaseEstimator,
    TransformerMixin,
):
    """
    Removes numerical features that contain
    very little useful variation.

    This transformer learns which columns to remove
    ONLY from the training data, preventing leakage.
    """

    def __init__(
        self,
        threshold=0.0,
    ):
        self.threshold = threshold

    def fit(
        self,
        X,
        y=None,
    ):

        X_array = np.asarray(X)

        if X_array.ndim == 1:

            X_array = X_array.reshape(
                -1,
                1,
            )

        variances = np.nanvar(
            X_array.astype(float),
            axis=0,
        )

        self.keep_mask_ = (
            variances > self.threshold
        )

        # Always retain at least one feature.
        if not self.keep_mask_.any():

            self.keep_mask_[
                np.argmax(variances)
            ] = True

        return self

    def transform(self, X):

        X_array = np.asarray(X)

        if X_array.ndim == 1:

            X_array = X_array.reshape(
                -1,
                1,
            )

        return X_array[
            :,
            self.keep_mask_,
        ]


# ============================================================
# COLUMN DETECTION
# ============================================================

def detect_feature_columns(
    df,
    target_column,
):
    """
    Automatically identify numerical and categorical
    feature columns.
    """

    if target_column not in df.columns:

        raise ValueError(
            f"Target column '{target_column}' "
            "not found."
        )

    X = df.drop(
        columns=[target_column]
    )

    numerical_columns = (
        X.select_dtypes(
            include=[
                "number",
            ]
        ).columns.tolist()
    )

    categorical_columns = (
        X.select_dtypes(
            include=[
                "object",
                "category",
                "bool",
            ]
        ).columns.tolist()
    )

    return (
        numerical_columns,
        categorical_columns,
    )


# ============================================================
# NUMERICAL PIPELINE
# ============================================================

def build_numeric_pipeline(
    scale=True,
):
    """
    Build preprocessing pipeline for numerical features.
    """

    steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
    ]

    if scale:

        steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    return Pipeline(
        steps=steps
    )


# ============================================================
# CATEGORICAL PIPELINE
# ============================================================

def build_categorical_pipeline():
    """
    Build preprocessing pipeline for categorical features.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                    add_indicator=False,
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )


# ============================================================
# COMPLETE PREPROCESSOR
# ============================================================

def build_preprocessor(
    df,
    target_column,
    scale_numeric=True,
):
    """
    Build a leakage-safe ColumnTransformer.

    Numerical:
        median imputation
        missing indicators
        scaling

    Categorical:
        most-frequent imputation
        one-hot encoding
    """

    (
        numerical_columns,
        categorical_columns,
    ) = detect_feature_columns(
        df,
        target_column,
    )

    transformers = []

    # --------------------------------------------------------
    # Numerical features
    # --------------------------------------------------------

    if numerical_columns:

        numerical_pipeline = (
            build_numeric_pipeline(
                scale=scale_numeric,
            )
        )

        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_columns,
            )
        )

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    if categorical_columns:

        categorical_pipeline = (
            build_categorical_pipeline()
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    if not transformers:

        raise ValueError(
            "No usable feature columns were found."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ============================================================
# FEATURE NAMES
# ============================================================

def get_feature_names(
    preprocessor,
):
    """
    Return feature names generated by the fitted
    ColumnTransformer.
    """

    try:

        return (
            preprocessor
            .get_feature_names_out()
            .tolist()
        )

    except Exception:

        return []


# ============================================================
# FEATURE SUMMARY
# ============================================================

def feature_summary(
    df,
    target_column,
):
    """
    Generate a summary of the feature space.
    """

    (
        numerical_columns,
        categorical_columns,
    ) = detect_feature_columns(
        df,
        target_column,
    )

    summary = {
        "target": target_column,
        "total_features": (
            len(numerical_columns)
            + len(categorical_columns)
        ),
        "numerical_features": len(
            numerical_columns
        ),
        "categorical_features": len(
            categorical_columns
        ),
        "numerical_columns": (
            numerical_columns
        ),
        "categorical_columns": (
            categorical_columns
        ),
    }

    return summary


# ============================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================

def build_advanced_preprocessor(
    df,
    target_column,
    scale_numeric=True,
):
    """
    Advanced preprocessing entry point.

    This function is intentionally kept separate from the
    basic preprocessor so future functionality can be added:

        - feature selection
        - polynomial features
        - date feature extraction
        - interaction features
        - dimensionality reduction
        - text embeddings

    without changing the rest of the ML architecture.
    """

    return build_preprocessor(
        df=df,
        target_column=target_column,
        scale_numeric=scale_numeric,
    )


# ============================================================
# TRANSFORM DATA
# ============================================================

def transform_features(
    df,
    target_column,
    scale_numeric=True,
):
    """
    Fit the feature preprocessing pipeline and transform
    the dataset.

    Returns
    -------
    X_transformed
    y
    preprocessor
    feature_names
    """

    data = df.copy()

    if target_column not in data.columns:

        raise ValueError(
            f"Target column '{target_column}' "
            "not found."
        )

    X = data.drop(
        columns=[target_column]
    )

    y = data[target_column]

    preprocessor = (
        build_advanced_preprocessor(
            data,
            target_column,
            scale_numeric=scale_numeric,
        )
    )

    X_transformed = (
        preprocessor.fit_transform(X)
    )

    feature_names = (
        get_feature_names(
            preprocessor
        )
    )

    return (
        X_transformed,
        y,
        preprocessor,
        feature_names,
    )