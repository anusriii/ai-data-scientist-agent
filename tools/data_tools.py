import pandas as pd
import streamlit as st
import plotly.express as px
from typing import Dict, List, Any


def load_dataset(uploaded_file) -> pd.DataFrame:
    """
    Read CSV or Excel (via openpyxl) into a pandas DataFrame.
    """
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        raise ValueError("Unsupported file type")
    return df


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return a dictionary with a few basic profiling statistics.
    """
    n_rows, n_cols = df.shape

    total_missing = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Very simple target‑column heuristic
    target = None
    low_cardinality = [
        c for c in categorical_cols if df[c].nunique() < 20
    ]
    if "target" in df.columns:
        target = "target"
    elif low_cardinality:
        target = low_cardinality[0]

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "total_missing": total_missing,
        "duplicate_rows": duplicate_rows,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "target_column": target,
    }


def plot_numeric_histograms(df: pd.DataFrame, numeric_cols: List[str]) -> None:
    """
    Render a Plotly histogram for each numeric column inside Streamlit.
    """
    if not numeric_cols:
        st.info("No numeric columns detected – nothing to plot.")
        return

    for col in numeric_cols:
        fig = px.histogram(df, x=col, nbins=30, title=f"Histogram of {col}")
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)