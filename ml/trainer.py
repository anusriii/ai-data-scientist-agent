import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from .preprocessing import prepare_training_data
from .models import get_model_registry
from .tuning import tune_model
from .evaluator import evaluate_model


# ============================================================
# TRAINER
# ============================================================

class MLTrainer:
    """
    Advanced machine-learning training engine.

    Supports:

        Fast       -> train models with default parameters
        Advanced   -> cross-validation + hyperparameter tuning
        Auto       -> automatically choose strategy
    """

    def __init__(
        self,
        problem_type,
        mode="fast",
        test_size=0.2,
        random_state=42,
        cv_folds=5,
        n_iter=10,
    ):

        if problem_type not in [
            "classification",
            "regression",
        ]:
            raise ValueError(
                "problem_type must be "
                "'classification' or 'regression'."
            )

        if mode not in [
            "fast",
            "advanced",
            "auto",
        ]:
            raise ValueError(
                "mode must be "
                "'fast', 'advanced', or 'auto'."
            )

        self.problem_type = problem_type
        self.mode = mode
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.n_iter = n_iter

        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.comparison = None


    # ========================================================
    # MODE SELECTION
    # ========================================================

    def _resolve_mode(
        self,
        n_rows,
        n_features,
    ):
        """
        Decide the training strategy.

        Auto mode chooses based on dataset size.
        """

        if self.mode != "auto":
            return self.mode

        # Small datasets
        if n_rows < 1000:
            return "advanced"

        # Medium datasets
        if n_rows < 50000:
            return "advanced"

        # Large datasets
        return "fast"


    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    def _split_data(
        self,
        X,
        y,
    ):

        stratify = None

        if self.problem_type == "classification":

            # Stratification requires at least
            # two samples per class.
            class_counts = (
                y.value_counts()
            )

            if (
                len(class_counts) > 1
                and class_counts.min() >= 2
            ):
                stratify = y

        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )


    # ========================================================
    # CREATE PIPELINE
    # ========================================================

    def _create_pipeline(
        self,
        preprocessor,
        model,
    ):

        return Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    model,
                ),
            ]
        )


    # ========================================================
    # TRAIN SINGLE MODEL
    # ========================================================

    def _train_single_model(
        self,
        model_name,
        model_info,
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        training_mode,
    ):

        model = model_info["model"]

        pipeline = self._create_pipeline(
            preprocessor=preprocessor,
            model=model,
        )

        search_object = None

        # ----------------------------------------------------
        # Advanced tuning
        # ----------------------------------------------------

        if (
            training_mode == "advanced"
            and model_info.get("params")
        ):

            search_object = tune_model(
                pipeline=pipeline,
                X_train=X_train,
                y_train=y_train,
                problem_type=self.problem_type,
                param_grid=model_info[
                    "params"
                ],
                search_type="random",
                folds=self.cv_folds,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )

            fitted_model = (
                search_object.best_estimator_
            )

        # ----------------------------------------------------
        # Fast training
        # ----------------------------------------------------

        else:

            pipeline.fit(
                X_train,
                y_train,
            )

            fitted_model = pipeline

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        evaluation = evaluate_model(
            model=fitted_model,
            X_test=X_test,
            y_test=y_test,
            problem_type=self.problem_type,
        )

        return {
            "model": fitted_model,
            "evaluation": evaluation,
            "search": search_object,
            "training_mode": training_mode,
        }


    # ========================================================
    # TRAIN ALL MODELS
    # ========================================================

    def fit(
        self,
        df,
        target_column,
        selected_models=None,
    ):
        """
        Train and evaluate multiple models.

        Parameters
        ----------
        df:
            Input DataFrame.

        target_column:
            Prediction target.

        selected_models:
            Optional list of model names.

        Returns
        -------
        results dictionary
        """

        # ----------------------------------------------------
        # Prepare data
        # ----------------------------------------------------

        prepared = prepare_training_data(
            df=df,
            target_column=target_column,
            scale_numeric=True,
        )

        X = prepared["X"]
        y = prepared["y"]
        preprocessor = prepared[
            "preprocessor"
        ]

        # ----------------------------------------------------
        # Split
        # ----------------------------------------------------

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = self._split_data(
            X,
            y,
        )

        # ----------------------------------------------------
        # Resolve mode
        # ----------------------------------------------------

        training_mode = self._resolve_mode(
            n_rows=len(X_train),
            n_features=X_train.shape[1],
        )

        # ----------------------------------------------------
        # Model registry
        # ----------------------------------------------------

        registry = get_model_registry(
            self.problem_type
        )

        if selected_models:

            registry = {
                name: info
                for name, info in registry.items()
                if name in selected_models
            }

        if not registry:

            raise ValueError(
                "No valid models were selected."
            )

        # ----------------------------------------------------
        # Train models
        # ----------------------------------------------------

        for (
            model_name,
            model_info,
        ) in registry.items():

            try:

                result = (
                    self._train_single_model(
                        model_name=model_name,
                        model_info=model_info,
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                        preprocessor=preprocessor,
                        training_mode=training_mode,
                    )
                )

                self.results[
                    model_name
                ] = result

            except Exception as error:

                self.results[
                    model_name
                ] = {
                    "error": str(error)
                }

        # ----------------------------------------------------
        # Comparison
        # ----------------------------------------------------

        self.comparison = (
            self._create_comparison()
        )

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        self._select_best_model()

        return self


    # ========================================================
    # COMPARISON
    # ========================================================

    def _create_comparison(self):

        rows = []

        for (
            model_name,
            result,
        ) in self.results.items():

            if "error" in result:
                continue

            metrics = result[
                "evaluation"
            ].get(
                "metrics",
                {},
            )

            row = {
                "Model": model_name,
            }

            row.update(
                metrics
            )

            rows.append(
                row
            )

        if not rows:

            return pd.DataFrame()

        comparison = pd.DataFrame(
            rows
        )

        if (
            self.problem_type
            == "classification"
        ):

            if "f1" in comparison.columns:

                comparison = (
                    comparison.sort_values(
                        "f1",
                        ascending=False,
                    )
                )

        else:

            if "r2" in comparison.columns:

                comparison = (
                    comparison.sort_values(
                        "r2",
                        ascending=False,
                    )
                )

        return comparison.reset_index(
            drop=True
        )


    # ========================================================
    # BEST MODEL
    # ========================================================

    def _select_best_model(self):

        if (
            self.comparison is None
            or self.comparison.empty
        ):
            return

        best_name = (
            self.comparison.iloc[0][
                "Model"
            ]
        )

        if best_name not in self.results:
            return

        self.best_model_name = (
            best_name
        )

        self.best_model = (
            self.results[
                best_name
            ]["model"]
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(
        self,
        X,
    ):
        """
        Generate predictions using
        the selected best model.
        """

        if self.best_model is None:

            raise RuntimeError(
                "No trained best model exists."
            )

        return self.best_model.predict(
            X
        )


    # ========================================================
    # PROBABILITY PREDICTION
    # ========================================================

    def predict_proba(
        self,
        X,
    ):
        """
        Generate class probabilities.
        """

        if self.best_model is None:

            raise RuntimeError(
                "No trained best model exists."
            )

        if not hasattr(
            self.best_model,
            "predict_proba",
        ):

            raise AttributeError(
                "The selected model does not "
                "support probability prediction."
            )

        return self.best_model.predict_proba(
            X
        )


    # ========================================================
    # TRAINING SUMMARY
    # ========================================================

    def summary(self):

        successful = [
            name
            for name, result
            in self.results.items()
            if "error" not in result
        ]

        failed = [
            name
            for name, result
            in self.results.items()
            if "error" in result
        ]

        return {
            "problem_type": self.problem_type,
            "requested_mode": self.mode,
            "models_attempted": len(
                self.results
            ),
            "models_successful": len(
                successful
            ),
            "models_failed": len(
                failed
            ),
            "successful_models": successful,
            "failed_models": failed,
            "best_model": self.best_model_name,
            "comparison": self.comparison,
        }