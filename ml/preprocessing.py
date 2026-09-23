import pandas as pd

from .feature_engineering import (
    build_advanced_preprocessor,
    detect_feature_columns,
    feature_summary,
)


def prepare_features(
    df: pd.DataFrame,
    target_column: str,
    scale_numeric: bool = True,
):
    """
    Prepare the dataset for machine learning.

    Returns
    -------
    X : DataFrame
        Feature matrix.

    y : Series
        Target.

    preprocessor : ColumnTransformer
        Unfitted preprocessing pipeline.

    """

    data = df.copy()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if data.empty:
        raise ValueError(
            "Cannot prepare an empty dataset."
        )

    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "does not exist."
        )

    # --------------------------------------------------------
    # Remove rows with missing target
    # --------------------------------------------------------

    data = data.dropna(
        subset=[target_column]
    ).reset_index(drop=True)

    if data.empty:
        raise ValueError(
            "No rows remain after removing "
            "missing target values."
        )

    # --------------------------------------------------------
    # Separate X and y
    # --------------------------------------------------------

    X = data.drop(
        columns=[target_column]
    )

    y = data[target_column]

    # --------------------------------------------------------
    # Detect feature types
    # --------------------------------------------------------

    numerical_columns, categorical_columns = (
        detect_feature_columns(
            data,
            target_column,
        )
    )

    if not numerical_columns and not categorical_columns:
        raise ValueError(
            "No usable features were detected."
        )

    # --------------------------------------------------------
    # Build preprocessing pipeline
    # --------------------------------------------------------

    preprocessor = build_advanced_preprocessor(
        df=data,
        target_column=target_column,
        scale_numeric=scale_numeric,
    )

    return (
        X,
        y,
        preprocessor,
    )


def get_dataset_feature_summary(
    df: pd.DataFrame,
    target_column: str,
):
    """
    Return a high-level feature summary.
    """

    return feature_summary(
        df,
        target_column,
    )


def validate_dataset(
    df: pd.DataFrame,
    target_column: str,
):
    """
    Validate a dataset before model training.
    """

    errors = []
    warnings = []

    # --------------------------------------------------------
    # Empty dataset
    # --------------------------------------------------------

    if df.empty:

        errors.append(
            "Dataset is empty."
        )

        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
        }

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    if target_column not in df.columns:

        errors.append(
            f"Target column '{target_column}' "
            "does not exist."
        )

        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
        }

    # --------------------------------------------------------
    # Target missing values
    # --------------------------------------------------------

    target_missing = (
        df[target_column]
        .isna()
        .sum()
    )

    if target_missing > 0:

        warnings.append(
            f"Target contains "
            f"{target_missing:,} missing values. "
            "These rows will be removed before "
            "training."
        )

    # --------------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------------

    duplicates = (
        df.duplicated()
        .sum()
    )

    if duplicates > 0:

        warnings.append(
            f"Dataset contains "
            f"{duplicates:,} duplicate rows."
        )

    # --------------------------------------------------------
    # Constant columns
    # --------------------------------------------------------

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(
            dropna=False
        ) <= 1
    ]

    if constant_columns:

        warnings.append(
            "Constant columns detected: "
            + ", ".join(
                map(
                    str,
                    constant_columns,
                )
            )
        )

    # --------------------------------------------------------
    # High-cardinality categorical columns
    # --------------------------------------------------------

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "category",
            ]
        ).columns
    )

    for column in categorical_columns:

        unique_count = (
            df[column]
            .nunique(
                dropna=True
            )
        )

        if unique_count > 100:

            warnings.append(
                f"'{column}' has "
                f"{unique_count:,} unique "
                "categorical values."
            )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def prepare_training_data(
    df: pd.DataFrame,
    target_column: str,
    scale_numeric: bool = True,
):
    """
    High-level entry point used by the ML trainer.

    Performs validation and prepares X, y,
    and the preprocessing pipeline.
    """

    validation = validate_dataset(
        df,
        target_column,
    )

    if not validation["valid"]:

        raise ValueError(
            "Dataset validation failed: "
            + "; ".join(
                validation["errors"]
            )
        )

    X, y, preprocessor = prepare_features(
        df,
        target_column,
        scale_numeric=scale_numeric,
    )

    return {
        "X": X,
        "y": y,
        "preprocessor": preprocessor,
        "validation": validation,
        "feature_summary": (
            get_dataset_feature_summary(
                df,
                target_column,
            )
        ),
    }