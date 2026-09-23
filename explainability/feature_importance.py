import numpy as np
import pandas as pd


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(
    model,
    feature_names,
):
    """
    Extract feature importance from models that expose
    feature_importances_ or coef_.
    """

    if not feature_names:
        return pd.DataFrame(
            columns=[
                "Feature",
                "Importance",
            ]
        )

    importance = None

    # --------------------------------------------------------
    # Tree-based models
    # --------------------------------------------------------

    if hasattr(
        model,
        "feature_importances_",
    ):

        importance = (
            model.feature_importances_
        )

    # --------------------------------------------------------
    # Linear models
    # --------------------------------------------------------

    elif hasattr(
        model,
        "coef_",
    ):

        coefficients = np.asarray(
            model.coef_
        )

        if coefficients.ndim == 1:

            importance = np.abs(
                coefficients
            )

        else:

            importance = np.mean(
                np.abs(coefficients),
                axis=0,
            )

    if importance is None:

        return pd.DataFrame(
            columns=[
                "Feature",
                "Importance",
            ]
        )

    importance = np.asarray(
        importance
    )

    if len(importance) != len(
        feature_names
    ):

        size = min(
            len(importance),
            len(feature_names),
        )

        importance = importance[
            :size
        ]

        feature_names = feature_names[
            :size
        ]

    result = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance,
        }
    )

    result = result.sort_values(
        "Importance",
        ascending=False,
    )

    return result.reset_index(
        drop=True
    )


# ============================================================
# TOP FEATURES
# ============================================================

def get_top_features(
    model,
    feature_names,
    top_n=20,
):
    """
    Return the most important features.
    """

    importance = get_feature_importance(
        model,
        feature_names,
    )

    return importance.head(
        top_n
    )


# ============================================================
# NORMALIZED IMPORTANCE
# ============================================================

def normalize_importance(
    importance_df,
):
    """
    Normalize feature importance so that the values
    sum to 1.
    """

    result = importance_df.copy()

    if result.empty:
        return result

    total = result[
        "Importance"
    ].sum()

    if total > 0:

        result[
            "Normalized Importance"
        ] = (
            result["Importance"]
            / total
        )

    else:

        result[
            "Normalized Importance"
        ] = 0.0

    return result