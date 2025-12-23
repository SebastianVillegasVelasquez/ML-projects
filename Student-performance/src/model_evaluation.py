"""model_evaluation.py

Utilities to evaluate regression models used in the project. Exposes
`evaluate_model` which computes R2, MSE and RMSE for a fitted estimator on
test data.
"""
from sklearn.metrics import r2_score, mean_squared_error

import pandas as pd
import numpy as np


def evaluate_model(
    model,
    test_data: tuple[pd.DataFrame, pd.Series]
) -> dict:
    """
    Evaluate the given model on the test data and return R2, MSE and RMSE metrics.

    Args:
        model: Trained machine learning model with a predict method.
        test_data: Tuple containing test features (X_test) and test target (y_test).

    Returns:
        dict: Dictionary containing keys 'r2', 'mse' and 'rmse'.
    """
    X_test, y_test = test_data
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    return {
        "r2": r2_score(y_test, y_pred),
        "mse": mse,
        "rmse": np.sqrt(mse)
    }
