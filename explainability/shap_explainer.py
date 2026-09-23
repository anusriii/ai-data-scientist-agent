import numpy as np
import pandas as pd


# ============================================================
# SHAP IMPORT
# ============================================================

def _load_shap():

    try:

        import shap

        return shap

    except ImportError:

        raise ImportError(
            "SHAP is not installed. "
            "Run: pip install shap"
        )


# ============================================================
# CREATE EXPLAINER
# ============================================================

def create_explainer(
    model,
    X_background=None,
):
    """
    Automatically choose an appropriate SHAP explainer.
    """

    shap = _load_shap()

    # --------------------------------------------------------
    # Tree models
    # --------------------------------------------------------

    model_for_explanation = model

    if hasattr(
        model,
        "named_steps",
    ):

        model_for_explanation = (
            model.named_steps.get(
                "model",
                model,
            )
        )

    if hasattr(
        model_for_explanation,
        "feature_importances_",
    ):

        return shap.TreeExplainer(
            model_for_explanation
        )

    # --------------------------------------------------------
    # Generic fallback
    # --------------------------------------------------------

    if X_background is None:

        raise ValueError(
            "X_background is required for "
            "generic SHAP explainers."
        )

    return shap.Explainer(
        model,
        X_background,
    )


# ============================================================
# SHAP VALUES
# ============================================================

def calculate_shap_values(
    model,
    X,
    X_background=None,
):
    """
    Calculate SHAP values.
    """

    shap = _load_shap()

    explainer = create_explainer(
        model=model,
        X_background=X_background,
    )

    values = explainer(
        X
    )

    return {
        "explainer": explainer,
        "values": values,
        "shap_values": values.values,
        "base_values": values.base_values,
    }


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

def global_importance(
    shap_values,
    feature_names,
):
    """
    Calculate mean absolute SHAP importance.
    """

    values = np.asarray(
        shap_values
    )

    # --------------------------------------------------------
    # Binary / regression
    # --------------------------------------------------------

    if values.ndim == 2:

        importance = np.mean(
            np.abs(values),
            axis=0,
        )

    # --------------------------------------------------------
    # Multi-class
    # --------------------------------------------------------

    elif values.ndim == 3:

        importance = np.mean(
            np.abs(values),
            axis=(0, 2),
        )

    else:

        raise ValueError(
            "Unsupported SHAP value shape."
        )

    size = min(
        len(importance),
        len(feature_names),
    )

    result = pd.DataFrame(
        {
            "Feature": feature_names[
                :size
            ],
            "SHAP Importance": (
                importance[:size]
            ),
        }
    )

    return result.sort_values(
        "SHAP Importance",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ============================================================
# LOCAL EXPLANATION
# ============================================================

def local_explanation(
    shap_values,
    X,
    feature_names,
    row_index=0,
):
    """
    Explain one individual prediction.
    """

    values = np.asarray(
        shap_values
    )

    if row_index < 0 or row_index >= len(X):

        raise IndexError(
            "row_index is outside the dataset."
        )

    row_values = X.iloc[
        row_index
    ] if isinstance(
        X,
        pd.DataFrame,
    ) else X[row_index]

    # --------------------------------------------------------
    # Single-output
    # --------------------------------------------------------

    if values.ndim == 2:

        contributions = values[
            row_index
        ]

    # --------------------------------------------------------
    # Multi-class
    # --------------------------------------------------------

    elif values.ndim == 3:

        contributions = np.mean(
            values[row_index],
            axis=1,
        )

    else:

        raise ValueError(
            "Unsupported SHAP value shape."
        )

    size = min(
        len(contributions),
        len(feature_names),
    )

    result = pd.DataFrame(
        {
            "Feature": feature_names[
                :size
            ],
            "Value": np.asarray(
                row_values
            )[:size],
            "SHAP Contribution": (
                contributions[:size]
            ),
        }
    )

    result[
        "Absolute Contribution"
    ] = np.abs(
        result[
            "SHAP Contribution"
        ]
    )

    return result.sort_values(
        "Absolute Contribution",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ============================================================
# POSITIVE / NEGATIVE CONTRIBUTIONS
# ============================================================

def contribution_summary(
    explanation_df,
    top_n=10,
):
    """
    Separate positive and negative feature contributions.
    """

    if explanation_df.empty:

        return {
            "positive": explanation_df,
            "negative": explanation_df,
        }

    positive = (
        explanation_df[
            explanation_df[
                "SHAP Contribution"
            ] > 0
        ]
        .head(top_n)
    )

    negative = (
        explanation_df[
            explanation_df[
                "SHAP Contribution"
            ] < 0
        ]
        .sort_values(
            "SHAP Contribution"
        )
        .head(top_n)
    )

    return {
        "positive": positive,
        "negative": negative,
    }