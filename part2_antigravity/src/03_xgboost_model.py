import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from xgboost import XGBClassifier


# Configure logging
def _setup_logging(debug: bool = False) -> None:
    """Configure logging level."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
        force=True,
    )


# Constants
OUTPUT_DIR: Path = Path("part2_antigravity/output")
MODEL_PATH: Path = OUTPUT_DIR / "xgboost_model.joblib"
REPORT_PATH: Path = OUTPUT_DIR / "evaluation_report.md"
CV_FOLDS: int = 5
RANDOM_STATE: int = 42


def _ensure_output_dir() -> None:
    """Ensure the output directory exists."""
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
    """Load train and test data from parquet files."""
    X_train = pl.read_parquet(OUTPUT_DIR / "X_train.parquet")
    X_test = pl.read_parquet(OUTPUT_DIR / "X_test.parquet")
    # Load targets and convert to Series
    y_train = pl.read_parquet(OUTPUT_DIR / "y_train.parquet")["target"]
    y_test = pl.read_parquet(OUTPUT_DIR / "y_test.parquet")["target"]

    logging.info(f"Loaded data: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test


def _optimize_hyperparameters(X_train: pl.DataFrame, y_train: pl.Series) -> XGBClassifier:
    """Perform hyperparameter tuning using RandomizedSearchCV."""
    logging.info("Starting hyperparameter tuning...")

    base_model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        n_jobs=-1,
    )

    param_dist = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 4, 5, 6],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=20,
        cv=CV_FOLDS,
        scoring="accuracy",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train.to_numpy(), y_train.to_numpy())

    logging.info(f"Best parameters: {search.best_params_}")
    logging.info(f"Best CV score: {search.best_score_:.4f}")

    return search.best_estimator_


def _train_and_evaluate_cv(
    X_train: pl.DataFrame, y_train: pl.Series
) -> tuple[XGBClassifier, float, float]:
    """Train XGBoost model and perform 5-fold CV."""

    # 1. Hyperparameter Tuning
    model = _optimize_hyperparameters(X_train, y_train)

    # 2. Verify with Cross-Validation (on best model)
    logging.info(f"Running {CV_FOLDS}-fold Cross-Validation on best model...")
    # XGBoost accepts Polars DataFrame directly in recent versions, or we convert to numpy/pandas
    # For safety with scikit-learn CV, convert to numpy valid
    cv_scores = cross_val_score(
        model, X_train.to_numpy(), y_train.to_numpy(), cv=CV_FOLDS, scoring="accuracy"
    )

    mean_cv_score = cv_scores.mean()
    std_cv_score = cv_scores.std()
    logging.info(f"CV Accuracy: {mean_cv_score:.4f} (+/- {std_cv_score:.4f})")

    # The model is already trained (best_estimator_ from RandomizedSearchCV)
    # No need to call model.fit again on the full training set unless we want to retrain
    # with the best params on the full dataset without CV.
    # For simplicity, we'll use the best_estimator_ as is.

    return model, mean_cv_score, std_cv_score


def _evaluate_model(
    model: XGBClassifier, X_test: pl.DataFrame, y_test: pl.Series
) -> dict[str, float]:
    """Evaluate model on test set."""
    logging.info("Evaluating on test set...")
    y_pred = model.predict(X_test.to_numpy())
    y_true = y_test.to_numpy()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }

    # Per-class report
    cls_report = classification_report(y_true, y_pred, output_dict=True)
    metrics["classification_report"] = cls_report

    logging.info("Test Set Metrics:")
    logging.info(json.dumps(metrics, indent=2))

    # Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.close()

    return metrics


def _plot_feature_importance(model: XGBClassifier, feature_names: list[str]) -> None:
    """Plot feature importance."""
    importances = model.feature_importances_

    # Create DataFrame for plotting
    fi_df = pl.DataFrame({"feature": feature_names, "importance": importances}).sort(
        "importance", descending=True
    )

    plt.figure(figsize=(10, 8))
    sns.barplot(data=fi_df.to_pandas(), x="importance", y="feature", palette="viridis")
    plt.title("Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png")
    plt.close()

    logging.info("Saved feature importance plot.")


def _save_artifacts(model: XGBClassifier, metrics: dict, cv_mean: float, cv_std: float) -> None:
    """Save model and evaluation report."""
    # Save model
    joblib.dump(model, MODEL_PATH)
    logging.info(f"Saved model to {MODEL_PATH}")

    # Write report
    cls_report = metrics.get("classification_report", {})

    # Format per-class table
    per_class_table = "| Class | Precision | Recall | F1-Score | Support |\n|---|---|---|---|---|\n"
    for cls in ["0", "1", "2"]:
        # Ensure the class key exists and is not 'accuracy', 'macro avg', 'weighted avg'
        if cls in cls_report and isinstance(cls_report[cls], dict):
            row = cls_report[cls]
            per_class_table += (
                f"| {cls} | {row['precision']:.4f} | {row['recall']:.4f} | "
                f"{row['f1-score']:.4f} | {row['support']} |\n"
            )

    report_content = f"""# XGBoost Model Evaluation Report

## Model Performance

### Cross-Validation (Training Set)
- **Mean Accuracy**: {cv_mean:.4f} (+/- {cv_std:.4f})
- **Folds**: {CV_FOLDS}

### Test Set Metrics
- **Accuracy**: {metrics['accuracy']:.4f}
- **Precision (Weighted)**: {metrics['precision']:.4f}
- **Recall (Weighted)**: {metrics['recall']:.4f}
- **F1 Score (Weighted)**: {metrics['f1']:.4f}

### Per-Class Metrics
{per_class_table}

## Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

## Feature Importance
![Feature Importance](feature_importance.png)

## Recommendations
- The model achieves strong performance with an accuracy of {metrics['accuracy']:.2%}
  on the test set.
- Check feature importance to understand key drivers (e.g., alcohol_to_malic ratio, proline).
"""

    with open(REPORT_PATH, "w") as f:
        f.write(report_content)

    logging.info(f"Saved evaluation report to {REPORT_PATH}")


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train XGBoost model on Wine dataset.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = _parse_args()
    _setup_logging(args.debug)

    _ensure_output_dir()

    X_train, X_test, y_train, y_test = _load_data()

    model, cv_mean, cv_std = _train_and_evaluate_cv(X_train, y_train)
    metrics = _evaluate_model(model, X_test, y_test)

    _plot_feature_importance(model, X_train.columns)
    _save_artifacts(model, metrics, cv_mean, cv_std)

    logging.info("Model training and evaluation completed successfully.")


if __name__ == "__main__":
    main()
