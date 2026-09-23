import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def evaluate_classification(
    y_true,
    y_pred,
    y_probability=None,
):
    """
    Calculate comprehensive classification metrics.
    """

    metrics = {}

    metrics["accuracy"] = accuracy_score(
        y_true,
        y_pred,
    )

    metrics["balanced_accuracy"] = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    metrics["precision"] = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    metrics["recall"] = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    metrics["f1"] = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    metrics["roc_auc"] = np.nan

    if y_probability is not None:

        try:

            probabilities = np.asarray(
                y_probability
            )

            if probabilities.ndim == 1:

                metrics["roc_auc"] = (
                    roc_auc_score(
                        y_true,
                        probabilities,
                    )
                )

            elif probabilities.shape[1] == 2:

                metrics["roc_auc"] = (
                    roc_auc_score(
                        y_true,
                        probabilities[:, 1],
                    )
                )

            else:

                metrics["roc_auc"] = (
                    roc_auc_score(
                        y_true,
                        probabilities,
                        multi_class="ovr",
                        average="weighted",
                    )
                )

        except Exception:
            pass

    return metrics


# ============================================================
# REGRESSION METRICS
# ============================================================

def evaluate_regression(
    y_true,
    y_pred,
):
    """
    Calculate comprehensive regression metrics.
    """

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_true,
        y_pred,
    )

    # --------------------------------------------------------
    # MAPE
    # --------------------------------------------------------

    y_true_array = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred_array = np.asarray(
        y_pred,
        dtype=float,
    )

    non_zero = (
        y_true_array != 0
    )

    if non_zero.any():

        mape = np.mean(
            np.abs(
                (
                    y_true_array[non_zero]
                    - y_pred_array[non_zero]
                )
                / y_true_array[non_zero]
            )
        ) * 100

    else:

        mape = np.nan

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }


# ============================================================
# CONFUSION MATRIX
# ============================================================

def get_confusion_matrix(
    y_true,
    y_pred,
):
    """
    Return confusion matrix and class labels.
    """

    labels = np.unique(
        np.concatenate(
            [
                np.asarray(y_true),
                np.asarray(y_pred),
            ]
        )
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    return matrix, labels


# ============================================================
# CLASSIFICATION REPORT TABLE
# ============================================================

def classification_report_table(
    y_true,
    y_pred,
):
    """
    Create a class-level performance table.
    """

    from sklearn.metrics import classification_report

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    return pd.DataFrame(
        report
    ).transpose()


# ============================================================
# REGRESSION RESIDUALS
# ============================================================

def calculate_residuals(
    y_true,
    y_pred,
):
    """
    Calculate prediction residuals.
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    residuals = (
        y_true - y_pred
    )

    return residuals


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    problem_type,
):
    """
    Evaluate a fitted model.

    Returns predictions and metrics.
    """

    y_pred = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if problem_type == "classification":

        y_probability = None

        if hasattr(
            model,
            "predict_proba",
        ):

            try:

                y_probability = (
                    model.predict_proba(
                        X_test
                    )
                )

            except Exception:
                y_probability = None

        metrics = evaluate_classification(
            y_test,
            y_pred,
            y_probability,
        )

        confusion, labels = (
            get_confusion_matrix(
                y_test,
                y_pred,
            )
        )

        report = (
            classification_report_table(
                y_test,
                y_pred,
            )
        )

        return {
            "predictions": y_pred,
            "probabilities": y_probability,
            "metrics": metrics,
            "confusion_matrix": confusion,
            "labels": labels,
            "classification_report": report,
        }

    # --------------------------------------------------------
    # Regression
    # --------------------------------------------------------

    if problem_type == "regression":

        metrics = evaluate_regression(
            y_test,
            y_pred,
        )

        residuals = calculate_residuals(
            y_test,
            y_pred,
        )

        return {
            "predictions": y_pred,
            "probabilities": None,
            "metrics": metrics,
            "residuals": residuals,
        }

    raise ValueError(
        "problem_type must be "
        "'classification' or 'regression'."
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

def create_comparison_table(
    evaluations,
    problem_type,
):
    """
    Convert multiple model evaluations into
    a comparison DataFrame.

    Parameters
    ----------
    evaluations:
        Dictionary:

        {
            "Random Forest": evaluation,
            "Logistic Regression": evaluation,
        }

    problem_type:
        classification or regression
    """

    rows = []

    for model_name, evaluation in (
        evaluations.items()
    ):

        row = {
            "Model": model_name
        }

        row.update(
            evaluation.get(
                "metrics",
                {},
            )
        )

        rows.append(row)

    if not rows:

        return pd.DataFrame()

    table = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    if problem_type == "classification":

        if "f1" in table.columns:

            table = table.sort_values(
                "f1",
                ascending=False,
            )

    elif problem_type == "regression":

        if "r2" in table.columns:

            table = table.sort_values(
                "r2",
                ascending=False,
            )

    return table.reset_index(
        drop=True
    )


# ============================================================
# BEST MODEL
# ============================================================

def identify_best_model(
    comparison_table,
    problem_type,
):
    """
    Identify the model with the best primary metric.

    Classification:
        F1

    Regression:
        R2
    """

    if comparison_table.empty:

        return None

    if problem_type == "classification":

        metric = "f1"

    elif problem_type == "regression":

        metric = "r2"

    else:

        raise ValueError(
            "Invalid problem type."
        )

    if metric not in comparison_table.columns:

        return None

    valid = comparison_table.dropna(
        subset=[metric]
    )

    if valid.empty:

        return None

    return valid.iloc[0]["Model"]


# ============================================================
# METRIC FORMATTER
# ============================================================

def format_metrics(
    metrics,
):
    """
    Convert metrics into display-friendly values.
    """

    formatted = {}

    for name, value in metrics.items():

        if value is None:

            formatted[name] = "N/A"
            continue

        try:

            if np.isnan(value):

                formatted[name] = "N/A"

            else:

                formatted[name] = round(
                    float(value),
                    4,
                )

        except (
            TypeError,
            ValueError,
        ):

            formatted[name] = value

    return formatted