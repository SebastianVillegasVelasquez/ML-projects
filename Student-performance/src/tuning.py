"""tuning.py

Utilities to perform hyperparameter search for estimators inside a
preprocessing pipeline. Provides a convenience wrapper for randomized
search and a function that returns parameter distributions for an SVR model.
"""

def fine_tune_model(pipeline, X_train, y_train, param_grid, cv=5, scoring='neg_mean_squared_error'):
    """Perform RandomizedSearchCV on the provided pipeline and training data.

    This function wraps sklearn's RandomizedSearchCV and returns the best
    estimator found after fitting.

    Args:
        pipeline: sklearn Pipeline that contains a final estimator under the name 'model'.
        X_train: training features (DataFrame/ndarray)
        y_train: training targets
        param_grid: parameter distributions passed to RandomizedSearchCV
        cv: cross-validation folds
        scoring: scoring metric for model selection

    Returns:
        Best estimator (fitted) from the RandomizedSearchCV.
    """
    from sklearn.model_selection import RandomizedSearchCV
    grid_search = RandomizedSearchCV(estimator=pipeline,
                               param_distributions=param_grid,
                               cv=cv,
                                n_iter=30,
                               scoring=scoring,
                               n_jobs=-1,
                               verbose=0,
                                random_state=42,
                                error_score='raise')

    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def get_param_distributions_svr():
    """Return a parameter distribution mapping suitable for SVR tuning.

    Uses scipy.stats distributions for efficient sampling during randomized search.
    """
    from scipy.stats import loguniform, uniform
    return {
        'model__C': loguniform(1e0, 1e3),  # C from 1 to 1000 (log-uniform)
        'model__gamma': loguniform(1e-4, 1e-1), # gamma from 0.0001 to 0.1 (log-uniform)
        'model__epsilon': uniform(0.1, 0.5) # epsilon from 0.1 to 0.6 (uniform distribution for a range of 0.5 starting at 0.1)
    }