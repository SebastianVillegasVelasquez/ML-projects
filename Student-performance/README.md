# Student Performance Prediction

## Overview

This repository implements a reproducible pipeline to predict student exam performance (target: `Exam_Score`) using academic, personal, family, and environmental features. Code and utilities are under `src/` and experiments live in `notebooks/`.

## Goals

- Predict `Exam_Score` (regression) and evaluate using R² and RMSE.
- Provide a modular preprocessing pipeline and reproducible training/evaluation scripts.

## Dataset

- Source: Kaggle — Student Performance Factors
- Samples: ~6,600
- Features: ~20 (numerical, ordinal, nominal, binary)
- Target: `Exam_Score`

## Methodology (brief)

1. Exploratory data analysis (EDA).
2. Preprocessing: imputation, encoding (ordinal/one-hot), scaling, and a ColumnTransformer pipeline.
3. Feature engineering and selection.
4. Model selection and hyperparameter tuning.
5. Final evaluation on a hold-out test set.

## Final Model Performance

The final model was evaluated on a hold-out test set that was not used during
training or hyperparameter tuning.

- RMSE: **1.8080**
- R²: **0.7687**

These results indicate that the model is able to explain approximately 77% of
the variance in students' exam scores, with an average prediction error of
approximately 1.8 points.

## Project structure

Current repository structure (top-level):

- `data/`
  - `raw/`
    - `StudentPerformanceFactors.csv`
  - `processed/`
    - `preprocessed_data.csv`
- `features/`
  - `feature_registry.yml`
  - `feature_store.md`
- `notebooks/`
  - `student_perfomance.ipynb`
- `src/`
  - `data_loader.py`
  - `featureEngineering.py`
  - `io.py`
  - `main.py`
  - `model_evaluation.py`
  - `paths.py`
  - `preprocessing.py`
  - `tuning.py`
- `config.py`
- `README.md`
- `requirements.txt`
- `.gitignore`

Notes:
- `src/` contains the reusable code for loading, preprocessing, training, and evaluating models.
- `notebooks/` contains analysis and experimentation.

## How to run

1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the main pipeline:

```powershell
python src/main.py
```

Depending on how you prefer to call scripts you may also run `python main.py` from the project root if configured.

## Author

Johan Sebastian Villegas Velasquez — Software Engineering Student
