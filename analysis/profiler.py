import pandas as pd
def generate_profile(df: pd.DataFrame) -> dict:
    """Generate detailed per-column profiling information."""

    profile = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "total_missing": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": {},
    }

    for column in df.columns:
        series = df[column]

        column_info = {
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "missing": int(series.isna().sum()),
            "missing_pct": round(float(series.isna().mean() * 100), 2),
            "unique": int(series.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(series):
            column_info.update({
                "mean": safe_float(series.mean()),
                "median": safe_float(series.median()),
                "std": safe_float(series.std()),
                "min": safe_float(series.min()),
                "max": safe_float(series.max()),
                "q1": safe_float(series.quantile(0.25)),
                "q3": safe_float(series.quantile(0.75)),
                "iqr": safe_float(
                    series.quantile(0.75) - series.quantile(0.25)
                ),
            })

        elif pd.api.types.is_datetime64_any_dtype(series):
            valid = series.dropna()

            if not valid.empty:
                column_info.update({
                    "min_date": str(valid.min()),
                    "max_date": str(valid.max()),
                    "date_range_days": int(
                        (valid.max() - valid.min()).days
                    ),
                })

        else:
            frequencies = series.value_counts(dropna=True).head(10)

            column_info.update({
                "top_value": (
                    str(frequencies.index[0])
                    if not frequencies.empty
                    else None
                ),
                "top_value_count": (
                    int(frequencies.iloc[0])
                    if not frequencies.empty
                    else 0
                ),
                "top_values": {
                    str(index): int(value)
                    for index, value in frequencies.items()
                },
            })

        profile["columns"][column] = column_info

    return profile


def safe_float(value):
    """Convert numeric values safely for display/JSON."""
    if pd.isna(value):
        return None

    return float(value)


def profile_to_dataframe(profile: dict) -> pd.DataFrame:
    """Convert detailed profile dictionary into a display dataframe."""

    rows = []

    for column, info in profile["columns"].items():
        rows.append({
            "Column": column,
            "Data Type": info.get("dtype"),
            "Non-Null": info.get("non_null"),
            "Missing": info.get("missing"),
            "Missing %": info.get("missing_pct"),
            "Unique": info.get("unique"),
            "Mean": info.get("mean"),
            "Median": info.get("median"),
            "Std": info.get("std"),
            "Min": info.get("min"),
            "Max": info.get("max"),
            "Q1": info.get("q1"),
            "Q3": info.get("q3"),
            "IQR": info.get("iqr"),
        })

    return pd.DataFrame(rows)