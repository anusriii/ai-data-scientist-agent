import pandas as pd

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    KFold,
)


# ============================================================
# CROSS-VALIDATION
# ============================================================

def create_cv(
    problem_type,
    folds=5,
    random_state=42,
):
    """
    Create an appropriate cross-validation strategy.

    Classification:
        StratifiedKFold

    Regression:
        KFold
    """

    if folds < 2:
        raise ValueError(
            "CV folds must be at least 2."
        )

    if problem_type == "classification":

        return StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=random_state,
        )

    if problem_type == "regression":

        return KFold(
            n_splits=folds,
            shuffle=True,
            random_state=random_state,
        )

    raise ValueError(
        "problem_type must be "
        "'classification' or 'regression'."
    )


# ============================================================
# DEFAULT SCORING
# ============================================================

def get_scoring(
    problem_type,
):
    """
    Return the primary optimization metric.
    """

    if problem_type == "classification":

        return "f1_weighted"

    if problem_type == "regression":

        return "r2"

    raise ValueError(
        "Invalid problem type."
    )


# ============================================================
# GRID SEARCH
# ============================================================

def grid_search(
    pipeline,
    param_grid,
    X_train,
    y_train,
    problem_type,
    folds=5,
    n_jobs=-1,
    verbose=0,
):
    """
    Perform exhaustive GridSearchCV.
    """

    cv = create_cv(
        problem_type=problem_type,
        folds=folds,
    )

    scoring = get_scoring(
        problem_type
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
        return_train_score=True,
    )

    search.fit(
        X_train,
        y_train,
    )

    return search


# ============================================================
# RANDOM SEARCH
# ============================================================

def random_search(
    pipeline,
    param_distributions,
    X_train,
    y_train,
    problem_type,
    folds=5,
    n_iter=20,
    n_jobs=-1,
    random_state=42,
    verbose=0,
):
    """
    Perform randomized hyperparameter search.

    Useful when the parameter space is large.
    """

    cv = create_cv(
        problem_type=problem_type,
        folds=folds,
        random_state=random_state,
    )

    scoring = get_scoring(
        problem_type
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
        refit=True,
        return_train_score=True,
    )

    search.fit(
        X_train,
        y_train,
    )

    return search


# ============================================================
# SEARCH RESULTS
# ============================================================

def get_search_results(
    search,
    top_n=10,
):
    """
    Convert hyperparameter search results
    into a clean DataFrame.
    """

    results = pd.DataFrame(
        search.cv_results_
    )

    columns = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score",
        "params",
    ]

    available = [
        column
        for column in columns
        if column in results.columns
    ]

    results = results[
        available
    ].sort_values(
        "rank_test_score"
    )

    return results.head(
        top_n
    ).reset_index(
        drop=True
    )


# ============================================================
# BEST PARAMETERS
# ============================================================

def get_best_parameters(
    search,
):
    """
    Return the best hyperparameters.
    """

    return search.best_params_


# ============================================================
# BEST SCORE
# ============================================================

def get_best_cv_score(
    search,
):
    """
    Return the best cross-validation score.
    """

    return search.best_score_


# ============================================================
# TUNING SUMMARY
# ============================================================

def tuning_summary(
    search,
):
    """
    Return a compact tuning summary.
    """

    return {
        "best_score": search.best_score_,
        "best_parameters": search.best_params_,
        "total_candidates": (
            len(search.cv_results_[
                "params"
            ])
        ),
        "best_estimator": search.best_estimator_,
    }


# ============================================================
# SAFE TUNING
# ============================================================

def tune_model(
    pipeline,
    X_train,
    y_train,
    problem_type,
    param_grid=None,
    search_type="grid",
    folds=5,
    n_iter=20,
    n_jobs=-1,
    random_state=42,
):
    """
    Unified hyperparameter tuning interface.

    Parameters
    ----------
    pipeline:
        sklearn Pipeline.

    X_train:
        Training features.

    y_train:
        Training target.

    problem_type:
        classification / regression.

    param_grid:
        Parameters for GridSearchCV or
        distributions for RandomizedSearchCV.

    search_type:
        "grid" or "random".

    Returns
    -------
    search:
        Fitted search object.
    """

    if param_grid is None:

        raise ValueError(
            "param_grid must be provided."
        )

    if search_type == "grid":

        return grid_search(
            pipeline=pipeline,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
            problem_type=problem_type,
            folds=folds,
            n_jobs=n_jobs,
        )

    if search_type == "random":

        return random_search(
            pipeline=pipeline,
            param_distributions=param_grid,
            X_train=X_train,
            y_train=y_train,
            problem_type=problem_type,
            folds=folds,
            n_iter=n_iter,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    raise ValueError(
        "search_type must be "
        "'grid' or 'random'."
    )