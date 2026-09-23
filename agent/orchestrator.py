import pandas as pd
import numpy as np


class DataScientistAgent:
    """
    Main orchestration layer for the AI Data Scientist Agent.

    The agent coordinates:

        1. Dataset profiling
        2. Data quality analysis
        3. Target detection
        4. Problem-type detection
        5. Classical ML
        6. Deep Learning
        7. Model comparison
        8. Explainability
    """

    def __init__(
        self,
        mode="auto",
        use_deep_learning=True,
        random_state=42,
    ):
        self.mode = mode
        self.use_deep_learning = (
            use_deep_learning
        )
        self.random_state = random_state

        self.df = None
        self.profile = None
        self.target_column = None
        self.problem_type = None

        self.ml_trainer = None
        self.dl_trainer = None

        self.results = {}
        self.status = "initialized"

    # ========================================================
    # LOAD DATA
    # ========================================================

    def load_data(self, df):
        """
        Load a pandas DataFrame into the agent.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "df must be a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "The dataset is empty."
            )

        self.df = df.copy()

        self.status = "data_loaded"

        return self.df

    # ========================================================
    # PROFILE
    # ========================================================

    def profile_data(self):
        """
        Generate dataset profiling information.
        """

        if self.df is None:
            raise RuntimeError(
                "Load data before profiling."
            )

        try:
            from tools.data_tools import (
                profile_dataset,
            )

            self.profile = profile_dataset(
                self.df
            )

        except Exception:

            self.profile = {
                "n_rows": len(self.df),
                "n_cols": len(self.df.columns),
                "total_missing": int(
                    self.df.isna().sum().sum()
                ),
                "duplicate_rows": int(
                    self.df.duplicated().sum()
                ),
                "numeric_cols": (
                    self.df
                    .select_dtypes(
                        include=np.number
                    )
                    .columns
                    .tolist()
                ),
                "categorical_cols": (
                    self.df
                    .select_dtypes(
                        exclude=np.number
                    )
                    .columns
                    .tolist()
                ),
            }

        self.status = "profiled"

        return self.profile

    # ========================================================
    # TARGET DETECTION
    # ========================================================

    def detect_target(
        self,
        target_column=None,
    ):
        """
        Detect or validate the target column.

        If target_column is supplied, it is always used.
        Otherwise, attempt sensible automatic detection.
        """

        if self.df is None:
            raise RuntimeError(
                "Load data before target detection."
            )

        if target_column is not None:

            if target_column not in self.df.columns:

                raise ValueError(
                    f"Target column "
                    f"'{target_column}' "
                    f"does not exist."
                )

            self.target_column = (
                target_column
            )

            return self.target_column

        # ----------------------------------------------------
        # Common target names
        # ----------------------------------------------------

        target_candidates = [
            "target",
            "label",
            "class",
            "y",
            "outcome",
            "response",
            "churn",
            "default",
            "price",
            "sales",
            "revenue",
        ]

        lower_columns = {
            str(column).lower(): column
            for column in self.df.columns
        }

        for candidate in target_candidates:

            if candidate in lower_columns:

                self.target_column = (
                    lower_columns[candidate]
                )

                return self.target_column

        # ----------------------------------------------------
        # Fallback: last column
        # ----------------------------------------------------

        self.target_column = (
            self.df.columns[-1]
        )

        return self.target_column

    # ========================================================
    # PROBLEM TYPE
    # ========================================================

    def detect_problem_type(
        self,
        target_column=None,
    ):
        """
        Determine classification vs regression.
        """

        if self.df is None:
            raise RuntimeError(
                "Load data first."
            )

        if target_column is not None:
            self.target_column = (
                target_column
            )

        if self.target_column is None:
            self.detect_target()

        y = self.df[
            self.target_column
        ]

        # ----------------------------------------------------
        # Non-numeric target
        # ----------------------------------------------------

        if not pd.api.types.is_numeric_dtype(y):

            self.problem_type = (
                "classification"
            )

            return self.problem_type

        # ----------------------------------------------------
        # Numeric target
        # ----------------------------------------------------

        unique_count = y.nunique(
            dropna=True
        )

        row_count = len(y)

        # Binary / small categorical numeric target
        if unique_count <= 10:

            self.problem_type = (
                "classification"
            )

        # High-cardinality numeric target
        else:

            self.problem_type = (
                "regression"
            )

        return self.problem_type

    # ========================================================
    # DATA QUALITY
    # ========================================================

    def data_quality_report(self):

        if self.df is None:
            raise RuntimeError(
                "Load data first."
            )

        missing = (
            self.df.isna()
            .sum()
            .sort_values(
                ascending=False
            )
        )

        duplicates = int(
            self.df.duplicated().sum()
        )

        numeric_columns = (
            self.df
            .select_dtypes(
                include=np.number
            )
            .columns
        )

        outliers = {}

        for column in numeric_columns:

            series = self.df[
                column
            ].dropna()

            if len(series) < 4:
                continue

            q1 = series.quantile(
                0.25
            )

            q3 = series.quantile(
                0.75
            )

            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = (
                q1 - 1.5 * iqr
            )

            upper = (
                q3 + 1.5 * iqr
            )

            count = int(
                (
                    (series < lower)
                    | (series > upper)
                ).sum()
            )

            outliers[column] = count

        return {
            "missing_values": missing.to_dict(),
            "duplicate_rows": duplicates,
            "outliers": outliers,
        }

    # ========================================================
    # RUN ML
    # ========================================================

    def train_ml(
        self,
        selected_models=None,
    ):
        """
        Train classical ML models.
        """

        if self.df is None:
            raise RuntimeError(
                "Load data first."
            )

        if self.target_column is None:
            self.detect_target()

        if self.problem_type is None:
            self.detect_problem_type()

        from ml.trainer import MLTrainer

        self.ml_trainer = MLTrainer(
            problem_type=self.problem_type,
            mode=self.mode,
            random_state=self.random_state,
        )

        self.ml_trainer.fit(
            df=self.df,
            target_column=self.target_column,
            selected_models=selected_models,
        )

        self.results[
            "classical_ml"
        ] = self.ml_trainer

        return self.ml_trainer

    # ========================================================
    # DEEP LEARNING
    # ========================================================

    def train_deep_learning(
        self,
        X_train,
        y_train,
        X_validation,
        y_validation,
    ):
        """
        Train the PyTorch tabular neural network.

        Data passed here should already be numerically
        transformed and scaled.
        """

        if not self.use_deep_learning:

            return None

        if self.problem_type is None:

            raise RuntimeError(
                "Detect problem type first."
            )

        from deep_learning.trainer import (
            DeepLearningTrainer,
        )

        self.dl_trainer = (
            DeepLearningTrainer(
                task=self.problem_type,
                random_state=self.random_state,
            )
        )

        self.dl_trainer.fit(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
        )

        self.results[
            "deep_learning"
        ] = self.dl_trainer

        return self.dl_trainer

    # ========================================================
    # EXPLAIN MODEL
    # ========================================================

    def explain_model(
        self,
        model=None,
        X=None,
        feature_names=None,
    ):
        """
        Generate model explanations.

        Uses feature importance when available and
        SHAP when possible.
        """

        if model is None:

            if self.ml_trainer is None:

                raise RuntimeError(
                    "No trained ML model exists."
                )

            model = (
                self.ml_trainer.best_model
            )

        if feature_names is None:

            if X is not None and hasattr(
                X,
                "columns",
            ):

                feature_names = (
                    X.columns.tolist()
                )

            else:

                feature_names = []

        explanation = {}

        # ----------------------------------------------------
        # Classical feature importance
        # ----------------------------------------------------

        try:

            from explainability.feature_importance import (
                get_feature_importance,
            )

            # Pipeline support
            estimator = model

            if hasattr(
                model,
                "named_steps",
            ):

                estimator = (
                    model.named_steps.get(
                        "model",
                        model,
                    )
                )

            importance = (
                get_feature_importance(
                    estimator,
                    feature_names,
                )
            )

            explanation[
                "feature_importance"
            ] = importance

        except Exception as error:

            explanation[
                "feature_importance_error"
            ] = str(error)

        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        if X is not None:

            try:

                from explainability.shap_explainer import (
                    calculate_shap_values,
                    global_importance,
                )

                shap_result = (
                    calculate_shap_values(
                        model=model,
                        X=X,
                    )
                )

                explanation[
                    "shap_values"
                ] = shap_result[
                    "shap_values"
                ]

                explanation[
                    "shap_importance"
                ] = global_importance(
                    shap_result[
                        "shap_values"
                    ],
                    feature_names,
                )

            except Exception as error:

                explanation[
                    "shap_error"
                ] = str(error)

        return explanation

    # ========================================================
    # FULL ANALYSIS
    # ========================================================

    def run(
        self,
        target_column=None,
        selected_models=None,
        train_models=True,
    ):
        """
        Run the complete classical analysis pipeline.
        """

        if self.df is None:
            raise RuntimeError(
                "Load data before running agent."
            )

        # ----------------------------------------------------
        # Profile
        # ----------------------------------------------------

        self.profile_data()

        # ----------------------------------------------------
        # Quality
        # ----------------------------------------------------

        quality = (
            self.data_quality_report()
        )

        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        self.detect_target(
            target_column
        )

        # ----------------------------------------------------
        # Problem type
        # ----------------------------------------------------

        self.detect_problem_type()

        # ----------------------------------------------------
        # ML
        # ----------------------------------------------------

        if train_models:

            self.train_ml(
                selected_models=selected_models
            )

        self.status = "completed"

        return {
            "profile": self.profile,
            "quality": quality,
            "target": self.target_column,
            "problem_type": self.problem_type,
            "ml": self.ml_trainer,
            "deep_learning": self.dl_trainer,
            "status": self.status,
        }

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(self):

        return {
            "status": self.status,
            "rows": (
                len(self.df)
                if self.df is not None
                else 0
            ),
            "columns": (
                len(self.df.columns)
                if self.df is not None
                else 0
            ),
            "target": self.target_column,
            "problem_type": self.problem_type,
            "ml_model": (
                self.ml_trainer.best_model_name
                if self.ml_trainer is not None
                else None
            ),
            "deep_learning": (
                self.dl_trainer is not None
            ),
        }