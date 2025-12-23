"""data_loader.py

Utilities to load the student performance dataset.

Behavior:
- If the raw dataset exists locally under `RAW_DATA_PATH`, it is loaded from disk.
- Otherwise the dataset is downloaded using `kagglehub` and saved locally.

This module exposes a single convenience function `load_data()` which
returns a pandas DataFrame.
"""

from kagglehub import kagglehub
from src.paths import RAW_DATA_PATH
from config import DATASET_FILE, DATASET_SLUG
import pandas as pd


def load_data() -> pd.DataFrame:
    """Load the dataset into a pandas DataFrame.

    The function first tries to read the CSV from `RAW_DATA_PATH / DATASET_FILE`.
    If the file is absent, it downloads the file using `kagglehub.dataset_download`
    and stores a local copy under `RAW_DATA_PATH` for subsequent runs.

    Returns:
        pandas.DataFrame: loaded dataset.
    """
    if RAW_DATA_PATH.exists():
        print(f"Loading dataset from {RAW_DATA_PATH / DATASET_FILE}...")
        df = pd.read_csv(RAW_DATA_PATH / DATASET_FILE)
        print("Dataset loaded from local storage.")
        return df
    else:
        dataset_slug = DATASET_SLUG
        dataset_file = DATASET_FILE

        print(f"Downloading dataset file '{dataset_file}' from '{dataset_slug}'...")
        file_path = kagglehub.dataset_download(dataset_slug, dataset_file)
        print("Download complete.")

        df = pd.read_csv(file_path)

        if RAW_DATA_PATH.exists() is False:
            RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

            df.to_csv(RAW_DATA_PATH / dataset_file, index=False)
            print(f"Dataset saved to {RAW_DATA_PATH / dataset_file}")

        return df