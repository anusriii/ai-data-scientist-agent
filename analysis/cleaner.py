import pandas as pd


def fill_missing(
    df: pd.DataFrame,
    strategy: str,
    custom_value=None,
) -> pd.DataFrame:
    """Return a cleaned copy with missing values handled."""

    cleaned = df.copy()

    if strategy == "drop":
        return cleaned.dropna()

    for column in cleaned.columns:

        if cleaned[column].isna().sum() == 0:
            continue

        if strategy == "mean":
            if pd.api.types.is_numeric_dtype(cleaned[column]):
                cleaned[column] = cleaned[column].fillna(
                    cleaned[column].mean()
                )

        elif strategy == "median":
            if pd.api.types.is_numeric_dtype(cleaned[column]):
                cleaned[column] = cleaned[column].fillna(
                    cleaned[column].median()
                )

        elif strategy == "most_frequent":
            mode = cleaned[column].mode()

            if not mode.empty:
                cleaned[column] = cleaned[column].fillna(mode.iloc[0])

        elif strategy == "custom":
            cleaned[column] = cleaned[column].fillna(custom_value)

    return cleaned


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe without duplicate rows."""

    return df.copy().drop_duplicates()


def detect_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
) -> dict:
    """Detect outliers in numerical columns."""

    if method != "iqr":
        raise ValueError("Only IQR outlier detection is currently supported.")

    results = {}

    numeric_columns = df.select_dtypes(include="number").columns

    for column in numeric_columns:

        series = df[column].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (df[column] < lower) | (df[column] > upper)

        results[column] = {
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "outlier_count": int(mask.sum()),
            "outlier_percentage": round(
                float(mask.mean() * 100),
                2,
            ),
        }

    return results


def remove_outliers(
    df: pd.DataFrame,
    columns=None,
) -> pd.DataFrame:
    """Remove IQR outliers from selected numerical columns."""

    cleaned = df.copy()

    if columns is None:
        columns = cleaned.select_dtypes(
            include="number"
        ).columns.tolist()

    mask = pd.Series(True, index=cleaned.index)

    for column in columns:

        if column not in cleaned.columns:
            continue

        if not pd.api.types.is_numeric_dtype(cleaned[column]):
            continue

        series = cleaned[column].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask &= (
            cleaned[column].isna()
            | cleaned[column].between(lower, upper)
        )

    return cleaned.loc[mask].copy()