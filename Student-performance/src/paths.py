"""paths.py

Defines project-local filesystem paths used to read/write data and artifacts.
Paths are defined relative to the repository root.
"""

from pathlib import Path

ROOT_PATH = Path.cwd().parent
DATA_PATH = ROOT_PATH / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
MODELS_PATH = ROOT_PATH / "models"
PIPELINE_PATH = ROOT_PATH / "artifacts"