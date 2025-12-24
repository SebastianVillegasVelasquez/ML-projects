"""featureEngineering.py

Feature engineering utilities for the student performance dataset.
Contains a scikit-learn compatible transformer `FeatureExtractor` which
creates new features and drops unwanted columns.
"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

from features.feature_definitions import (
    consistency_score,
    motivation_adjusted_study,
)


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Create and remove features in a scikit-learn transformer wrapper.

    This transformer is compatible with sklearn pipelines. It drops a list
    of unwanted columns and computes new features defined in
    `self.new_features_funcs`.

    Attributes:
        feature_names_in_ (np.ndarray): stored input feature names after fit.
        new_features_funcs (dict): mapping new feature name -> function(X) that computes it.
        columns_to_drop (list): columns removed during transform.
    """
    def __init__(self):
        self.feature_names_in_ = None
        self.new_features_funcs = {
            'Consistency_Score': consistency_score,
            'Motivation_Adjusted_Study': motivation_adjusted_study,
        }

        self.columns_to_drop = [
            'Gender',
            'School_Type',
            'Sleep_Hours',
            'Physical_Activity',
        ]

    def fit(self, X: pd.DataFrame, y=None):
        """Fit method (sklearn API) — records input feature names.

        Args:
            X: pandas.DataFrame of input features.
            y: Ignored, present for sklearn compatibility.

        Returns:
            self
        """
        # sklearn convention
        self.feature_names_in_ = X.columns.to_numpy()
        return self

    def transform(self, X: pd.DataFrame):
        """Apply the feature engineering steps and return transformed DataFrame.

        Steps performed:
        - copy input DataFrame
        - drop columns listed in `self.columns_to_drop` (ignore missing)
        - compute and append new features from `self.new_features_funcs`

        Args:
            X: pandas.DataFrame to transform.

        Returns:
            pandas.DataFrame with dropped columns removed and new features added.
        """
        X_transformed = X.copy()

        # Drop columns
        X_transformed.drop(columns=self.columns_to_drop, errors='ignore', inplace=True)

        # Create new features USING X_transformed
        for feature_name, func in self.new_features_funcs.items():
            X_transformed[feature_name] = func(X_transformed)

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """Return output feature names after transform.

        If `input_features` is None, the saved `feature_names_in_` is used.
        Returned names are the remaining original columns (after drop)
        followed by the newly created features.
        """
        if input_features is None:
            input_features = self.feature_names_in_

        remaining_columns = [
            col for col in input_features if col not in self.columns_to_drop
        ]

        return np.array(remaining_columns + list(self.new_features_funcs.keys()))