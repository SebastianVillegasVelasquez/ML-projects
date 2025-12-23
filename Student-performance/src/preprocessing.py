"""preprocessing.py

Module that builds and runs the preprocessing pipeline for the student
performance dataset. Contains helpers to construct column transformers,
compose the full preprocessing pipeline (optionally with a model), and
apply the pipeline to a pandas DataFrame returning a processed DataFrame.
"""

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import  Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


from src.featureEngineering import FeatureExtractor

from src.io import (
save_preprocessing_pipeline,
save_preprocessed_data,
)

import pandas as pd
import numpy as np


def build_column_transformer():
    """Builds and returns a ColumnTransformer that applies different
    preprocessing pipelines depending on column types.

    The returned transformer handles:
    - ordinal columns: impute most frequent, then ordinal encode
    - nominal columns: impute most frequent, then one-hot encode (ignore unknowns)
    - binary columns: impute most frequent, then map to 0/1 using BinaryMapper
    - numerical columns: impute mean, then standard scale
    - distribution columns: impute most frequent, apply log1p-like transform, then standard scale

    The function uses selector callables imported from `src.utils` so the
    transformer can be fit on a DataFrame directly.
    """

    ordinal_pipeline = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OrdinalEncoder()
    )

    # Pipeline for Nominal columns
    nominal_pipeline = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore')
    )

    numerical_pipeline = make_pipeline(
        SimpleImputer(strategy='mean'),
        StandardScaler()
    )

    distribution_pipeline = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        FunctionTransformer(log1p_transform, feature_names_out='one-to-one'),
        StandardScaler()
    )

    mapping_dict = {
        'Yes': 1,
        'No': 0,
        'Public': 1,
        'Private': 0,
        'Male': 1,
        'Female': 0
    }

    binary_pipelines = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('mapper', BinaryMapper(mapping_dict))
    ])

    preprocessing = ColumnTransformer([
        ('ordinal', ordinal_pipeline, get_ordinal_columns),
        ('nominal', nominal_pipeline, get_nominal_columns),
        ('binary', binary_pipelines, get_binary_columns),
        ('num', numerical_pipeline, get_numerical_columns),
        ('log1p', distribution_pipeline, get_distribution_columns)

    ], remainder='passthrough')

    return preprocessing


def build_preprocessing_pipeline(model=None):
    """Compose the full preprocessing pipeline.

    This creates a Pipeline that first applies feature engineering
    (via `FeatureExtractor`), then the column-wise preprocessing created by
    `build_column_transformer`. If a scikit-learn estimator `model` is
    provided, it is appended as the final step.

    Args:
        model: Optional scikit-learn estimator to append to the pipeline.

    Returns:
        sklearn.pipeline.Pipeline: the composed pipeline.
    """
    print("Building preprocessing pipeline...")
    preprocessing = build_column_transformer()
    feature_extractor = FeatureExtractor()

    if model is not None:
        pipeline = Pipeline(steps=[
            ('feature_engineering', feature_extractor),
            ('preprocessing', preprocessing),
            ('model', model)
        ])
    else:
        pipeline = Pipeline(steps=[
            ('feature_engineering', feature_extractor),
            ('preprocessing', preprocessing)
        ])


    return pipeline


def preprocess_data(X: pd.DataFrame):
    """Fit the preprocessing pipeline on the provided DataFrame and return
    a processed pandas DataFrame.

    This function will:
    - build the preprocessing pipeline
    - fit it on `X` and transform `X`
    - convert the resulting numpy array back to a pandas DataFrame using
      the pipeline's feature names
    - persist the transformed data and the fitted pipeline using utility
      functions from `src.utils`

    Args:
        X: pandas.DataFrame with raw input features.

    Returns:
        pandas.DataFrame: the preprocessed data with column names derived
        from the preprocessing pipeline.
    """
    preprocessing_pipeline = build_preprocessing_pipeline()
    X_processed = preprocessing_pipeline.fit_transform(X)

    X_processed_df = pd.DataFrame(X_processed, columns=preprocessing_pipeline.get_feature_names_out())
    save_preprocessed_data(X=X_processed_df)
    save_preprocessing_pipeline(pipeline=preprocessing_pipeline)
    return X_processed_df


def get_ordinal_columns(X: pd.DataFrame) -> list:
  """Identify ordinal categorical columns in the DataFrame.

  A column is considered ordinal if its non-null unique values are a subset
  of the expected ordered levels ['Low', 'Medium', 'High'].

  Args:
      X: pandas.DataFrame to inspect.

  Returns:
      list: column names detected as ordinal categorical features.
  """
  ordinal_cols = []
  expected_levels = ['Low', 'Medium', 'High']
  for col in X.select_dtypes(include='object'):
    unique_values = X[col].unique()
    unique_values_filtered = [val for val in unique_values if pd.notna(val)]

    if set(unique_values_filtered).issubset(set(expected_levels)):
      ordinal_cols.append(col)

  return ordinal_cols

def get_binary_columns(X: pd.DataFrame) -> list:
  """Detect binary categorical columns in the DataFrame.

  A binary column is any object-typed column with exactly two distinct
  non-null values.

  Args:
      X: pandas.DataFrame to inspect.

  Returns:
      list: column names detected as binary categorical features.
  """
  binary_cols = []
  for col in X.select_dtypes(include='object'):
    unique_values = X[col].unique()
    unique_values_filtered = [val for val in unique_values if pd.notna(val)] # Filter NaNs here too
    if len(unique_values_filtered) == 2:
      binary_cols.append(col)

  return binary_cols

def get_nominal_columns(X: pd.DataFrame) -> list:
  """Identify nominal categorical columns in the DataFrame.

  A nominal column is an object-typed column with more than two distinct
  non-null values that are not a subset of the ordinal expected levels.

  Args:
      X: pandas.DataFrame to inspect.

  Returns:
      list: column names detected as nominal categorical features.
  """
  nominal_cols = []
  expected_levels = ['Low', 'Medium', 'High']
  for col in X.select_dtypes(include='object'):
    unique_values = X[col].unique()
    unique_values_filtered = [val for val in unique_values if pd.notna(val)]
    if len(unique_values_filtered) > 2 and not set(unique_values_filtered).issubset(set(expected_levels)):
      nominal_cols.append(col)

  return nominal_cols

def get_numerical_columns(X: pd.DataFrame) -> list:
  """Return the list of numerical columns for preprocessing.

  This function intentionally excludes the 'Tutoring_Sessions' column
  from the numerical set (it is handled separately by the distribution
  pipeline).

  Args:
      X: pandas.DataFrame to inspect.

  Returns:
      list: numerical column names (dtype numeric) excluding 'Tutoring_Sessions'.
  """
  X = X.drop(columns=['Tutoring_Sessions'], errors='ignore').copy()
  return X.select_dtypes(include=np.number).columns.tolist()

def get_distribution_columns(X: pd.DataFrame) -> list:
    """Return columns that require a log1p-like transform.

    Currently this returns ['Tutoring_Sessions'] when that column exists.

    Args:
        X: pandas.DataFrame to inspect.

    Returns:
        list: column names to be processed by the distribution (log1p) pipeline.
    """
    return ['Tutoring_Sessions'] if 'Tutoring_Sessions' in X.columns else []

class BinaryMapper(BaseEstimator, TransformerMixin):
    """A small transformer that maps binary categorical values to numeric codes.

    The transformer accepts either a DataFrame or NumPy array on transform:
    - If a DataFrame is passed, values are replaced in-place and the
      DataFrame is returned.
    - If a NumPy array is passed, it is converted to a temporary DataFrame
      using stored input feature names (or generated names) so the mapping
      can be applied, then converted back to a NumPy array.

    Attributes:
        mapping (dict): mapping from original values to numeric codes.
        feature_names_in_ (np.ndarray or None): set during fit when a
            DataFrame with named columns is provided.
    """
    def __init__(self, mapping):
        """Initialize the mapper with a value -> code mapping.

        Args:
            mapping: dict mapping original categorical values to numeric codes.
        """
        self.mapping = mapping

    def fit(self, X, y=None):
        """Fit the transformer (no-op) and store feature names if available.

        Args:
            X: pandas.DataFrame or array-like used to infer column names.
            y: Ignored, present for compatibility.

        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_numpy()
        else:
            self.feature_names_in_ = None
        return self

    def transform(self, X):
        """Apply the mapping to X and return transformed data.

        Accepts either a pandas DataFrame or numpy array. The return type
        follows the input type (DataFrame returned for DataFrame input,
        NumPy array for NumPy input).

        Args:
            X: pandas.DataFrame or numpy.ndarray to transform.

        Returns:
            Transformed object of the same high-level type as the input.

        Raises:
            TypeError: if the input type is unsupported.
        """
        if isinstance(X, pd.DataFrame):
            return X.replace(self.mapping)

        if isinstance(X, np.ndarray):
            if self.feature_names_in_ is None:
                cols = [f"binary_{i}" for i in range(X.shape[1])]
            else:
                cols = self.feature_names_in_

            df = pd.DataFrame(X, columns=cols)
            df = df.replace(self.mapping)
            return df.to_numpy()

        raise TypeError(f"Unsupported input type: {type(X)}")

    def get_feature_names_out(self, input_features=None):
        """Return output feature names for compatibility with sklearn.

        If input_features is provided, it is returned as an ndarray. Otherwise
        stored feature names (if any) are returned. If none are available,
        an empty list is returned.
        """
        if input_features is not None:
            return np.asarray(input_features)

        if self.feature_names_in_ is not None:
            return self.feature_names_in_

        return []


def log1p_transform(X):
    """Apply a log1p transform to numeric input.

    This wrapper exists so it can be used as a FunctionTransformer target.

    Args:
        X: numeric array-like to transform.

    Returns:
        Transformed array with np.log1p applied elementwise.
    """
    return np.log1p(X)