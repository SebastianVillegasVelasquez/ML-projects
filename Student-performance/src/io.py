"""io.py

Small I/O helpers for persisting the preprocessing pipeline and
preprocessed data artifacts.
"""

from src.paths import PROCESSED_DATA_PATH, PIPELINE_PATH
import pandas as pd

def save_preprocessing_pipeline(pipeline) -> None:
    """Save the fitted preprocessing pipeline to disk using joblib.

    Ensures the pipeline directory exists before writing.

    Args:
        pipeline: fitted sklearn pipeline to persist.
    """
    import joblib
    if not PIPELINE_PATH.exists():
        PIPELINE_PATH.mkdir(parents=True, exist_ok=True)
    filepath = PIPELINE_PATH / 'preprocessing_pipeline.joblib'
    joblib.dump(pipeline, filepath)

def save_preprocessed_data(X: pd.DataFrame) -> None:
    """Persist preprocessed DataFrame to CSV under the processed data folder.

    Args:
        X: pandas.DataFrame to write as CSV.
    """
    if not PROCESSED_DATA_PATH.exists():
        PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    filepath = PROCESSED_DATA_PATH / 'preprocessed_data.csv'
    X.to_csv(filepath, index=False)
    print(f"Preprocessed data saved to {filepath}")