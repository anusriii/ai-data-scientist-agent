# ml/models.py

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# ============================================================
# CLASSIFICATION MODELS
# ============================================================

def get_classification_models(random_state=42):

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=random_state,
        ),
    }


# ============================================================
# REGRESSION MODELS
# ============================================================

def get_regression_models(random_state=42):

    return {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            random_state=random_state,
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=random_state,
        ),
    }


# ============================================================
# MODEL REGISTRY
# ============================================================

def get_model_registry(
    problem_type,
    random_state=42,
):
    """
    Return the appropriate model registry
    for classification or regression.
    """

    problem_type = str(
        problem_type
    ).lower().strip()

    if problem_type == "classification":

        return get_classification_models(
            random_state=random_state
        )

    if problem_type == "regression":

        return get_regression_models(
            random_state=random_state
        )

    raise ValueError(
        f"Unsupported problem type: {problem_type}"
    )