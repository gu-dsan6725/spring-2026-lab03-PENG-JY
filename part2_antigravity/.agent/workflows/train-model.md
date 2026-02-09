# Train Model Workflow

Train an XGBoost classification model on the prepared Wine dataset.

## Prerequisites

Run the EDA and feature engineering steps first to generate the train/test splits.

## Steps

1. Load the prepared train/test splits from `output/` (parquet files).
2. Parse CLI arguments (support `--debug` for verbose logging).
3. Perform Hyperparameter Tuning using `RandomizedSearchCV` (n_iter=20) to find the best `XGBClassifier` parameters.
4. Train the best model on the full training set.
5. Generate predictions on the test set.
6. Compute evaluation metrics:
    - Overall Accuracy, Precision, Recall, F1-Score (Weighted)
    - Per-Class Precision/Recall breakdown
7. Create diagnostic plots:
    - Confusion Matrix Heatmap
    - Feature Importance Plot
8. Save the trained model as `output/xgboost_model.joblib`.
9. Write a comprehensive evaluation report to `output/evaluation_report.md`.

## Requirements

- Use polars for data loading.
- Follow the coding standards in `.agent/rules/code-style-guide.md`.
- Save the training script as `part2_antigravity/src/03_xgboost_model.py`.
- After writing the file, run `uv run ruff check --fix` and `uv run python -m py_compile`.
