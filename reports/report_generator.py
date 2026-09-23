from pathlib import Path
from datetime import datetime

import pandas as pd


# ============================================================
# REPORT GENERATOR
# ============================================================

class ReportGenerator:
    """
    Generate a professional HTML data-science report.
    """

    def __init__(
        self,
        output_dir="outputs/reports",
    ):

        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # HTML HELPERS
    # ========================================================

    @staticmethod
    def _escape(value):

        text = str(value)

        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    @staticmethod
    def _table_from_dataframe(
        df,
        max_rows=50,
    ):

        if df is None or df.empty:

            return (
                "<p>No data available.</p>"
            )

        display_df = df.head(
            max_rows
        ).copy()

        return display_df.to_html(
            index=False,
            classes="data-table",
            border=0,
        )

    # ========================================================
    # DATASET SECTION
    # ========================================================

    def _dataset_section(
        self,
        agent,
    ):

        df = agent.df

        if df is None:

            return ""

        rows = len(df)
        columns = len(df.columns)

        missing = int(
            df.isna().sum().sum()
        )

        duplicates = int(
            df.duplicated().sum()
        )

        return f"""
        <section>
            <h2>Dataset Overview</h2>

            <div class="cards">

                <div class="card">
                    <span>Rows</span>
                    <strong>{rows:,}</strong>
                </div>

                <div class="card">
                    <span>Columns</span>
                    <strong>{columns:,}</strong>
                </div>

                <div class="card">
                    <span>Missing Values</span>
                    <strong>{missing:,}</strong>
                </div>

                <div class="card">
                    <span>Duplicate Rows</span>
                    <strong>{duplicates:,}</strong>
                </div>

            </div>

            <h3>Columns</h3>

            {self._table_from_dataframe(
                pd.DataFrame(
                    {
                        "Column": df.columns,
                        "Data Type": [
                            str(dtype)
                            for dtype in df.dtypes
                        ],
                        "Missing": [
                            int(
                                df[col].isna().sum()
                            )
                            for col in df.columns
                        ],
                        "Unique Values": [
                            int(
                                df[col].nunique(
                                    dropna=True
                                )
                            )
                            for col in df.columns
                        ],
                    }
                )
            )}

        </section>
        """

    # ========================================================
    # TARGET SECTION
    # ========================================================

    def _target_section(
        self,
        agent,
    ):

        target = (
            agent.target_column
        )

        problem = (
            agent.problem_type
        )

        if target is None:

            return ""

        return f"""
        <section>

            <h2>Problem Definition</h2>

            <div class="info-box">

                <p>
                    <strong>Target column:</strong>
                    {self._escape(target)}
                </p>

                <p>
                    <strong>Problem type:</strong>
                    {self._escape(problem)}
                </p>

            </div>

        </section>
        """

    # ========================================================
    # QUALITY SECTION
    # ========================================================

    def _quality_section(
        self,
        agent,
    ):

        quality = (
            agent.data_quality_report()
        )

        missing = quality.get(
            "missing_values",
            {},
        )

        outliers = quality.get(
            "outliers",
            {},
        )

        missing_rows = []

        for column, count in missing.items():

            if count > 0:

                missing_rows.append(
                    {
                        "Column": column,
                        "Missing Values": count,
                    }
                )

        outlier_rows = []

        for column, count in outliers.items():

            if count > 0:

                outlier_rows.append(
                    {
                        "Column": column,
                        "Outliers": count,
                    }
                )

        missing_df = pd.DataFrame(
            missing_rows
        )

        outlier_df = pd.DataFrame(
            outlier_rows
        )

        return f"""
        <section>

            <h2>Data Quality</h2>

            <h3>Missing Values</h3>

            {self._table_from_dataframe(
                missing_df
            )}

            <h3>Outliers</h3>

            {self._table_from_dataframe(
                outlier_df
            )}

            <p>
                Duplicate rows detected:
                <strong>
                    {quality.get(
                        "duplicate_rows",
                        0
                    )}
                </strong>
            </p>

        </section>
        """

    # ========================================================
    # MODEL SECTION
    # ========================================================

    def _model_section(
        self,
        agent,
    ):

        trainer = (
            agent.ml_trainer
        )

        if trainer is None:

            return """
            <section>
                <h2>Machine Learning</h2>
                <p>No models were trained.</p>
            </section>
            """

        comparison = (
            trainer.comparison
        )

        best_model = (
            trainer.best_model_name
        )

        return f"""
        <section>

            <h2>Machine Learning</h2>

            <div class="info-box">

                <p>
                    <strong>Best model:</strong>
                    {self._escape(
                        best_model
                    )}
                </p>

                <p>
                    <strong>Training mode:</strong>
                    {self._escape(
                        trainer.mode
                    )}
                </p>

            </div>

            <h3>Model Comparison</h3>

            {self._table_from_dataframe(
                comparison
            )}

        </section>
        """

    # ========================================================
    # MODEL DETAILS
    # ========================================================

    def _model_details(
        self,
        agent,
    ):

        trainer = (
            agent.ml_trainer
        )

        if trainer is None:

            return ""

        rows = []

        for (
            model_name,
            result,
        ) in trainer.results.items():

            if "error" in result:

                rows.append(
                    {
                        "Model": model_name,
                        "Status": "Failed",
                        "Details": result[
                            "error"
                        ],
                    }
                )

            else:

                rows.append(
                    {
                        "Model": model_name,
                        "Status": "Success",
                        "Details": "Trained successfully",
                    }
                )

        return f"""
        <section>

            <h2>Training Details</h2>

            {self._table_from_dataframe(
                pd.DataFrame(rows)
            )}

        </section>
        """

    # ========================================================
    # SUMMARY
    # ========================================================

    def _summary_section(
        self,
        agent,
    ):

        summary = agent.summary()

        return f"""
        <section>

            <h2>Analysis Summary</h2>

            <ul class="summary-list">

                <li>
                    Rows:
                    <strong>
                        {summary["rows"]:,}
                    </strong>
                </li>

                <li>
                    Columns:
                    <strong>
                        {summary["columns"]:,}
                    </strong>
                </li>

                <li>
                    Target:
                    <strong>
                        {self._escape(
                            summary["target"]
                        )}
                    </strong>
                </li>

                <li>
                    Problem Type:
                    <strong>
                        {self._escape(
                            summary["problem_type"]
                        )}
                    </strong>
                </li>

                <li>
                    Best ML Model:
                    <strong>
                        {self._escape(
                            summary["ml_model"]
                        )}
                    </strong>
                </li>

            </ul>

        </section>
        """

    # ========================================================
    # FULL HTML
    # ========================================================

    def generate_html(
        self,
        agent,
        filename=None,
    ):

        timestamp = datetime.now()

        if filename is None:

            filename = (
                "data_science_report_"
                + timestamp.strftime(
                    "%Y%m%d_%H%M%S"
                )
                + ".html"
            )

        output_path = (
            self.output_dir
            / filename
        )

        html = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width,
               initial-scale=1.0">

<title>
AI Data Scientist Report
</title>

<style>

body {{
    font-family:
        Arial,
        Helvetica,
        sans-serif;

    margin: 0;

    padding: 0;

    background: #f4f6f8;

    color: #202124;
}}

.container {{
    max-width: 1200px;

    margin: auto;

    padding: 40px;
}}

header {{
    background: #ffffff;

    padding: 30px;

    border-radius: 12px;

    margin-bottom: 25px;

    box-shadow:
        0 2px 10px
        rgba(0,0,0,0.08);
}}

header h1 {{
    margin-top: 0;
}}

section {{
    background: #ffffff;

    padding: 30px;

    margin-bottom: 25px;

    border-radius: 12px;

    box-shadow:
        0 2px 10px
        rgba(0,0,0,0.06);
}}

.cards {{
    display: grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(180px, 1fr)
        );

    gap: 15px;

    margin: 20px 0;
}}

.card {{
    background: #f7f9fb;

    border-radius: 10px;

    padding: 20px;
}}

.card span {{
    display: block;

    color: #666;

    margin-bottom: 8px;
}}

.card strong {{
    font-size: 28px;
}}

.info-box {{
    background: #f1f7ff;

    border-left: 5px solid #4285f4;

    padding: 15px 20px;

    border-radius: 6px;
}}

.data-table {{
    width: 100%;

    border-collapse: collapse;

    margin-top: 15px;

    font-size: 14px;
}}

.data-table th,
.data-table td {{
    border-bottom:
        1px solid #ddd;

    padding: 10px;

    text-align: left;
}}

.data-table th {{
    background: #f5f5f5;
}}

.summary-list {{
    line-height: 2;
}}

.footer {{
    text-align: center;

    color: #777;

    padding: 20px;
}}

</style>

</head>

<body>

<div class="container">

<header>

<h1>
AI Data Scientist Report
</h1>

<p>
Automatically generated analysis report
</p>

<p>
Generated:
{timestamp.strftime(
    "%Y-%m-%d %H:%M:%S"
)}
</p>

</header>

{self._summary_section(agent)}

{self._dataset_section(agent)}

{self._target_section(agent)}

{self._quality_section(agent)}

{self._model_section(agent)}

{self._model_details(agent)}

<div class="footer">

AI Data Scientist Agent

</div>

</div>

</body>

</html>
"""

        output_path.write_text(
            html,
            encoding="utf-8",
        )

        return output_path

    # ========================================================
    # SAVE REPORT
    # ========================================================

    def save(
        self,
        agent,
        filename=None,
    ):

        return self.generate_html(
            agent=agent,
            filename=filename,
        )