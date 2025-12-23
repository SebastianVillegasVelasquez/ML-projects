"""main.py

Entrypoint for running the end-to-end training and evaluation pipeline.
"""

import logging

from src.data_loader import  load_data
from src.preprocessing import build_preprocessing_pipeline
from src.model_evaluation import evaluate_model
from src.tuning import fine_tune_model
from src.tuning import get_param_distributions_svr

from sklearn.model_selection import train_test_split

from sklearn.svm import SVR

logging.basicConfig(level=logging.INFO)

def main() -> dict:
    """Run the full training and evaluation flow.

    Steps:
    - load raw data
    - split into train/test
    - construct pipeline with SVR model
    - fine-tune hyperparameters
    - evaluate final estimator on test set

    Returns:
        dict with evaluation metrics.
    """
    df_raw = load_data()

    X = df_raw.drop(columns=["Exam_Score"])
    y = df_raw["Exam_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        shuffle=True,
        test_size=0.2,
        random_state=42
    )

    model = SVR(kernel="rbf")
    pipeline = build_preprocessing_pipeline(model=model)

    best_estimator = fine_tune_model(
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        param_grid=get_param_distributions_svr()
    )

    metrics = evaluate_model(
        best_estimator,
        test_data=(X_test, y_test)
    )

    logging.info(
        "Final Test Metrics | R2: %.4f | RMSE: %.4f",
        metrics["r2"],
        metrics["rmse"]
    )

    return metrics


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Pipeline execution failed")