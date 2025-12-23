# Student Performance Prediction

## Description

Project to predict students' exam performance (`Exam_Score`) using academic, personal, family, and environmental features. The repository separates experimentation (notebooks) from production-ready code (`src/`).

## Goal

Solve a regression problem to predict the exam score (`Exam_Score`). Main evaluation metrics are R² and RMSE.

## Dataset

- Source: Kaggle — Student Performance Factors
- Samples: ~6,607
- Features: ~20
- Target: `Exam_Score` (approximate range 55–101)

The dataset contains numerical features, ordinal categorical variables, nominal categorical variables, and binary variables.

## Methodology (summary)

1. Exploratory Data Analysis (EDA): distributions, correlations, and data quality checks.
2. Modular preprocessing using pipelines (`ColumnTransformer`): missing value imputation, ordinal encoding, one-hot encoding for nominal variables, binary mapping, scaling and transformations (e.g., log for skewed features).
3. Feature engineering: ablation studies and interaction features; in this project, engineered features provided little improvement.
4. Model selection: comparison of several regressors; SVR (RBF kernel) showed the best stable performance.
5. Hyperparameter tuning: `RandomizedSearchCV` over C, gamma, and epsilon.
6. Final evaluation on a held-out test set with R² and RMSE.

## Results (summary)

- Selected model: SVR (RBF)
- Test R²: ~0.769
- Test RMSE: ~1.81

> Interpretation: the model explains around 77% of the variance and predicts exam scores with an average error below 2 points.

## Project structure

Current repository structure (summary):

- config.py
- README.md
- requirements.txt
- artifacts/
  - preprocessing_pipeline.joblib
- data/
  - raw/
    - StudentPerformanceFactors.csv
  - processed/
    - preprocessed_data.csv
- notebooks/
  - student_perfomance.ipynb
- src/
  - data_loader.py
  - featureEngineering.py
  - io.py
  - main.py
  - model_evaluation.py
  - paths.py
  - preprocessing.py
  - tuning.py
- __pycache__/

Notes: `src/` contains reusable code and `notebooks/` contains experimentation and analysis.

## How to run

1. (Optional) Create and activate a virtual environment:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the pipeline/entrypoint:

```powershell
python src/main.py
```

(You can also run `python main.py` depending on how you prefer to call the top-level script.)

## Best practices and recommendations

- Keep experiments and notebooks separate from reusable code in `src/`.
- Version important artifacts and document preprocessing changes.
- Add unit tests and CI to ensure refactors do not break the pipeline.

## Author

Sebastián — Software Engineering Student

---

This project focuses on reproducibility and engineering best practices for machine learning.
