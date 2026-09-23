import streamlit as st
import pandas as pd
import numpy as np

from tools.data_tools import (
    load_dataset,
    profile_dataset,
    plot_numeric_histograms,
)

from agent.orchestrator import DataScientistAgent
from reports.report_generator import ReportGenerator


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Data Scientist Agent",
    page_icon="🧪",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "df": None,
    "agent": None,
    "analysis_result": None,
    "report_path": None,
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HEADER
# ============================================================

st.title("🧪 AI Data Scientist Agent")

st.caption(
    "Upload your dataset and let the AI automatically "
    "analyze, model, explain and report."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📂 Data Upload")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"],
    )

    st.divider()

    st.markdown(
        """
        ### 🤖 Automatic Mode

        The agent automatically handles:

        - Data profiling
        - Data quality checks
        - Preprocessing
        - Problem detection
        - Model selection
        - Model training
        - Evaluation
        - Deep Learning
        - Explainability
        - Report generation
        """
    )


# ============================================================
# NO DATA
# ============================================================

if uploaded_file is None:

    st.info(
        "👈 Upload a CSV or Excel dataset to get started."
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:

    with st.spinner("Reading dataset..."):

        df = load_dataset(
            uploaded_file
        )

    st.session_state.df = df

except Exception as error:

    st.error(
        f"Could not load dataset: {error}"
    )

    st.stop()


# ============================================================
# CURRENT DATA
# ============================================================

df = st.session_state.df


# ============================================================
# DATASET OVERVIEW
# ============================================================

profile = profile_dataset(
    df
)

st.header("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Rows",
    profile.get(
        "n_rows",
        len(df),
    ),
)

c2.metric(
    "Columns",
    profile.get(
        "n_cols",
        len(df.columns),
    ),
)

c3.metric(
    "Missing Values",
    profile.get(
        "total_missing",
        int(
            df.isna()
            .sum()
            .sum()
        ),
    ),
)

c4.metric(
    "Duplicate Rows",
    profile.get(
        "duplicate_rows",
        int(
            df.duplicated()
            .sum()
        ),
    ),
)


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "🔎 Overview",
        "📊 Profiling",
        "📈 EDA",
        "🧹 Cleaning",
        "🤖 AI Analysis",
        "🧠 Explainability",
        "📄 Report",
    ]
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tabs[0]:

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(20),
        width="stretch",
    )

    st.subheader(
        "Column Information"
    )

    info_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [
                str(dtype)
                for dtype in df.dtypes
            ],
            "Missing": [
                int(
                    df[column]
                    .isna()
                    .sum()
                )
                for column in df.columns
            ],
            "Unique Values": [
                int(
                    df[column]
                    .nunique(
                        dropna=True
                    )
                )
                for column in df.columns
            ],
        }
    )

    st.dataframe(
        info_df,
        width="stretch",
    )


# ============================================================
# TAB 2 — PROFILING
# ============================================================

with tabs[1]:

    st.subheader(
        "📊 Automatic Data Profiling"
    )

    numeric_cols = profile.get(
        "numeric_cols",
        [],
    )

    categorical_cols = profile.get(
        "categorical_cols",
        [],
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### 🔢 Numerical Columns"
        )

        if numeric_cols:

            st.write(
                numeric_cols
            )

        else:

            st.info(
                "No numerical columns detected."
            )

    with col2:

        st.markdown(
            "### 🔤 Categorical Columns"
        )

        if categorical_cols:

            st.write(
                categorical_cols
            )

        else:

            st.info(
                "No categorical columns detected."
            )

    st.divider()

    st.subheader(
        "Statistical Summary"
    )

    try:

        statistics = (
            df.describe(
                include="all"
            ).T
        )

        st.dataframe(
            statistics,
            width="stretch",
        )

    except Exception:

        st.info(
            "Statistics unavailable."
        )


# ============================================================
# TAB 3 — EDA
# ============================================================

with tabs[2]:

    st.subheader(
        "📈 Exploratory Data Analysis"
    )

    numeric_cols = profile.get(
        "numeric_cols",
        [],
    )

    if not numeric_cols:

        st.info(
            "No numerical columns available."
        )

    else:

        selected_column = st.selectbox(
            "Select column to inspect",
            numeric_cols,
            key="eda_column",
        )

        if selected_column:

            st.subheader(
                f"Distribution — {selected_column}"
            )

            try:

                values = (
                    df[selected_column]
                    .dropna()
                )

                histogram_df = pd.DataFrame(
                    {
                        selected_column: values
                    }
                )

                st.bar_chart(
                    histogram_df[
                        selected_column
                    ]
                    .value_counts()
                    .sort_index()
                    .head(100)
                )

            except Exception as error:

                st.warning(
                    f"Could not create chart: {error}"
                )

        if len(numeric_cols) >= 2:

            st.subheader(
                "Correlation Matrix"
            )

            correlation = (
                df[numeric_cols]
                .corr()
            )

            st.dataframe(
                correlation,
                width="stretch",
            )

        st.subheader(
            "Numeric Histograms"
        )

        try:

            plot_numeric_histograms(
                df,
                numeric_cols,
            )

        except Exception as error:

            st.warning(
                f"Histogram generation failed: {error}"
            )


# ============================================================
# TAB 4 — CLEANING
# ============================================================

with tabs[3]:

    st.subheader(
        "🧹 Data Cleaning"
    )

    st.info(
        "The agent can automatically handle common "
        "data-quality problems during modeling."
    )

    missing_count = int(
        df.isna()
        .sum()
        .sum()
    )

    duplicate_count = int(
        df.duplicated()
        .sum()
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Missing Values",
        missing_count,
    )

    c2.metric(
        "Duplicate Rows",
        duplicate_count,
    )

    st.divider()

    st.markdown(
        """
        ### Automatic cleaning

        During AI analysis, the system can automatically:

        - Handle missing numerical values
        - Handle categorical values
        - Encode categorical features
        - Scale numerical features
        - Remove problematic rows when required
        - Detect unsuitable columns
        - Prepare training data
        """
    )


# ============================================================
# TAB 5 — AI ANALYSIS
# ============================================================

with tabs[4]:

    st.subheader(
        "🤖 AI Data Scientist"
    )

    st.markdown(
        """
        ### Let the agent do the work

        You only need to select the **target column**.

        The AI will automatically determine the appropriate
        preprocessing and machine-learning strategy.
        """
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    target_column = st.selectbox(
        "🎯 Target column",
        df.columns.tolist(),
        key="target_column",
    )

    # --------------------------------------------------------
    # Advanced settings
    # --------------------------------------------------------

    with st.expander(
        "⚙️ Advanced Settings"
    ):

        st.caption(
            "Normal users can leave these settings unchanged."
        )

        training_mode = st.selectbox(
            "Training mode",
            [
                "auto",
                "fast",
                "advanced",
            ],
            index=0,
            key="training_mode",
        )

        enable_dl = st.checkbox(
            "Enable Deep Learning",
            value=True,
            key="enable_dl",
        )

        st.info(
            "Auto mode lets the agent choose the "
            "appropriate training strategy."
        )

    # --------------------------------------------------------
    # Main button
    # --------------------------------------------------------

    st.divider()

    run_analysis = st.button(
        "🚀 Run AI Data Scientist",
        type="primary",
        width="stretch",
        key="run_agent",
    )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    if run_analysis:

        progress = st.progress(
            0
        )

        status = st.empty()

        try:

            status.write(
                "🔍 Loading dataset..."
            )

            progress.progress(
                10
            )

            agent = DataScientistAgent(
                mode=training_mode,
                use_deep_learning=enable_dl,
                random_state=42,
            )

            agent.load_data(
                df
            )

            status.write(
                "📊 Profiling dataset..."
            )

            progress.progress(
                25
            )

            result = agent.run(
                target_column=target_column,
                train_models=True,
            )

            progress.progress(
                100
            )

            status.write(
                "✅ Analysis completed."
            )

            st.session_state.agent = (
                agent
            )

            st.session_state.analysis_result = (
                result
            )

            st.success(
                "🎉 AI Data Scientist analysis completed!"
            )

        except Exception as error:

            progress.empty()

            status.empty()

            st.error(
                f"Analysis failed: {error}"
            )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    agent = st.session_state.agent

    if agent is not None:

        st.divider()

        st.subheader(
            "🎯 AI Analysis Results"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Target",
            str(
                agent.target_column
            ),
        )

        c2.metric(
            "Problem Type",
            str(
                agent.problem_type
            ),
        )

        if agent.ml_trainer is not None:

            c3.metric(
                "Best Model",
                str(
                    agent.ml_trainer
                    .best_model_name
                ),
            )

        # ----------------------------------------------------
        # Model comparison
        # ----------------------------------------------------

        if agent.ml_trainer is not None:

            comparison = (
                agent.ml_trainer.comparison
            )

            if (
                comparison is not None
                and not comparison.empty
            ):

                st.subheader(
                    "🏆 Automatically Selected Models"
                )

                st.dataframe(
                    comparison,
                    width="stretch",
                )

                if (
                    agent.problem_type
                    == "classification"
                    and "f1"
                    in comparison.columns
                ):

                    chart_data = (
                        comparison[
                            [
                                "Model",
                                "f1",
                            ]
                        ]
                        .set_index(
                            "Model"
                        )
                    )

                    st.bar_chart(
                        chart_data
                    )

                elif (
                    agent.problem_type
                    == "regression"
                    and "r2"
                    in comparison.columns
                ):

                    chart_data = (
                        comparison[
                            [
                                "Model",
                                "r2",
                            ]
                        ]
                        .set_index(
                            "Model"
                        )
                    )

                    st.bar_chart(
                        chart_data
                    )

        # ----------------------------------------------------
        # Deep learning
        # ----------------------------------------------------

        if agent.dl_trainer is not None:

            st.subheader(
                "🧠 Deep Learning"
            )

            dl = (
                agent.dl_trainer
            )

            st.write(
                f"Device: `{dl.device}`"
            )

            st.write(
                f"Best Epoch: `{dl.best_epoch}`"
            )

            if dl.history:

                history_df = pd.DataFrame(
                    dl.history
                )

                columns = [
                    column
                    for column in [
                        "train_loss",
                        "validation_loss",
                    ]
                    if column
                    in history_df.columns
                ]

                if columns:

                    st.line_chart(
                        history_df[
                            columns
                        ]
                    )


# ============================================================
# TAB 6 — EXPLAINABILITY
# ============================================================

with tabs[5]:

    st.subheader(
        "🧠 Explainability"
    )

    agent = st.session_state.agent

    if agent is None:

        st.info(
            "Run the AI Data Scientist first."
        )

    elif agent.ml_trainer is None:

        st.info(
            "No trained model available."
        )

    else:

        st.write(
            "Best model:",
            agent.ml_trainer.best_model_name,
        )

        try:

            best_model = (
                agent.ml_trainer.best_model
            )

            estimator = best_model

            if hasattr(
                best_model,
                "named_steps",
            ):

                estimator = (
                    best_model
                    .named_steps
                    .get(
                        "model",
                        best_model,
                    )
                )

            feature_names = (
                df.drop(
                    columns=[
                        agent.target_column
                    ]
                )
                .columns
                .tolist()
            )

            if hasattr(
                best_model,
                "named_steps",
            ):

                preprocessor = (
                    best_model
                    .named_steps
                    .get(
                        "preprocessor"
                    )
                )

                if (
                    preprocessor
                    is not None
                ):

                    try:

                        feature_names = (
                            preprocessor
                            .get_feature_names_out()
                            .tolist()
                        )

                    except Exception:

                        pass

            from explainability.feature_importance import (
                get_feature_importance,
            )

            importance = (
                get_feature_importance(
                    estimator,
                    feature_names,
                )
            )

            if importance.empty:

                st.info(
                    "Feature importance is not available for this model."
                )

            else:

                st.dataframe(
                    importance.head(30),
                    width="stretch",
                )

                st.bar_chart(
                    importance
                    .head(15)
                    .set_index(
                        "Feature"
                    )[
                        "Importance"
                    ]
                )

        except Exception as error:

            st.warning(
                f"Explainability unavailable: {error}"
            )


# ============================================================
# TAB 7 — REPORT
# ============================================================

with tabs[6]:

    st.subheader(
        "📄 AI Report"
    )

    agent = st.session_state.agent

    if agent is None:

        st.info(
            "Run the AI Data Scientist first."
        )

    else:

        if st.button(
            "📄 Generate Report",
            type="primary",
            width="stretch",
            key="generate_report",
        ):

            try:

                with st.spinner(
                    "Generating report..."
                ):

                    generator = (
                        ReportGenerator()
                    )

                    report_path = (
                        generator.generate_html(
                            agent
                        )
                    )

                    st.session_state.report_path = (
                        report_path
                    )

                st.success(
                    "Report generated."
                )

            except Exception as error:

                st.error(
                    f"Report generation failed: {error}"
                )

        report_path = (
            st.session_state.report_path
        )

        if report_path is not None:

            try:

                report_data = (
                    report_path.read_bytes()
                )

                st.download_button(
                    "⬇️ Download Report",
                    data=report_data,
                    file_name=report_path.name,
                    mime="text/html",
                    width="stretch",
                    key="download_report",
                )

            except Exception as error:

                st.warning(
                    f"Download unavailable: {error}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🧪 AI Data Scientist Agent | "
    "Automated ML • Deep Learning • Explainability"
)