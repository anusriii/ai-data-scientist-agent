import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_histogram(df: pd.DataFrame, column: str):
    return px.histogram(
        df,
        x=column,
        title=f"Distribution of {column}",
        marginal="box",
    )


def create_boxplot(df: pd.DataFrame, column: str):
    return px.box(
        df,
        y=column,
        title=f"Box Plot — {column}",
    )


def create_bar_chart(df: pd.DataFrame, column: str, top_n: int = 10):
    counts = df[column].value_counts().head(top_n).reset_index()
    counts.columns = [column, "Count"]

    return px.bar(
        counts,
        x=column,
        y="Count",
        title=f"Top {top_n} Values — {column}",
    )


def create_scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
):
    return px.scatter(
        df,
        x=x,
        y=y,
        title=f"{x} vs {y}",
    )


def create_correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return None

    correlation = numeric_df.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.columns,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            text=correlation.round(2).values,
            texttemplate="%{text}",
        )
    )

    fig.update_layout(
        title="Correlation Matrix",
        height=600,
    )

    return fig


def run_eda(df: pd.DataFrame):
    """Generate a limited set of automatic EDA figures."""

    figures = []

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    for column in numeric_columns[:10]:
        figures.append({
            "type": "histogram",
            "column": column,
            "figure": create_histogram(df, column),
        })

        figures.append({
            "type": "boxplot",
            "column": column,
            "figure": create_boxplot(df, column),
        })

    for column in categorical_columns[:10]:
        figures.append({
            "type": "bar",
            "column": column,
            "figure": create_bar_chart(df, column),
        })

    heatmap = create_correlation_heatmap(df)

    if heatmap is not None:
        figures.append({
            "type": "correlation",
            "column": None,
            "figure": heatmap,
        })

    return figures