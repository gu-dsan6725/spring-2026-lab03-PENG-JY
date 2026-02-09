"""XGBoost Classification Model for UCI Wine Dataset."""

import json
import logging
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

INPUT_DIR: Path = Path("output")
OUTPUT_DIR: Path = Path("output")
RANDOM_STATE: int = 42
N_CV_FOLDS: int = 5
FIGURE_SIZE: tuple[int, int] = (10, 8)


def load_train_test_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load train and test datasets from parquet files."""
    logging.info("Loading train and test datasets")

    train_path = INPUT_DIR / "train_data.parquet"
    test_path = INPUT_DIR / "test_data.parquet"

    train_df = pl.read_parquet(train_path)
    test_df = pl.read_parquet(test_path)

    logging.info(f"Train set: {len(train_df)} samples")
    logging.info(f"Test set: {len(test_df)} samples")

    return train_df, test_df


def train_xgboost_model(
    X_train: pl.DataFrame,
    y_train: pl.Series,
) -> xgb.XGBClassifier:
    """Train XGBoost classifier with cross-validation."""
    logging.info("Training XGBoost classifier")

    model = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
    )

    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()

    logging.info(f"Performing {N_CV_FOLDS}-fold cross-validation")
    cv_scores = cross_val_score(
        model,
        X_train_np,
        y_train_np,
        cv=N_CV_FOLDS,
        scoring="accuracy",
    )

    logging.info(
        f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
    )

    model.fit(X_train_np, y_train_np)
    logging.info("Model training completed")

    return model


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: pl.DataFrame,
    y_test: pl.Series,
) -> dict[str, float | dict]:
    """Evaluate model performance on test set."""
    logging.info("Evaluating model on test set")

    X_test_np = X_test.to_numpy()
    y_test_np = y_test.to_numpy()

    y_pred = model.predict(X_test_np)

    accuracy = accuracy_score(y_test_np, y_pred)
    precision = precision_score(y_test_np, y_pred, average="weighted")
    recall = recall_score(y_test_np, y_pred, average="weighted")
    f1 = f1_score(y_test_np, y_pred, average="weighted")

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }

    logging.info(f"Test Accuracy: {accuracy:.4f}")
    logging.info(f"Test Precision: {precision:.4f}")
    logging.info(f"Test Recall: {recall:.4f}")
    logging.info(f"Test F1-Score: {f1:.4f}")

    class_report = classification_report(
        y_test_np,
        y_pred,
        output_dict=True,
    )
    metrics["classification_report"] = class_report

    return metrics


def plot_confusion_matrix(
    model: xgb.XGBClassifier,
    X_test: pl.DataFrame,
    y_test: pl.Series,
) -> None:
    """Generate and save confusion matrix heatmap."""
    logging.info("Generating confusion matrix")

    X_test_np = X_test.to_numpy()
    y_test_np = y_test.to_numpy()
    y_pred = model.predict(X_test_np)

    cm = confusion_matrix(y_test_np, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Class 0", "Class 1", "Class 2"],
        yticklabels=["Class 0", "Class 1", "Class 2"],
    )
    plt.title("Confusion Matrix", fontsize=14)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    logging.info(f"Saved confusion matrix to {output_path}")


def plot_feature_importance(
    model: xgb.XGBClassifier,
    feature_names: list[str],
) -> None:
    """Generate and save feature importance plot."""
    logging.info("Generating feature importance plot")

    importances = model.feature_importances_
    indices = importances.argsort()[::-1][:15]

    plt.figure(figsize=FIGURE_SIZE)
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(
        range(len(indices)),
        [feature_names[i] for i in indices],
        rotation=45,
        ha="right",
    )
    plt.title("Top 15 Feature Importances", fontsize=14)
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Importance", fontsize=12)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "feature_importance.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    logging.info(f"Saved feature importance plot to {output_path}")


def save_model_and_results(
    model: xgb.XGBClassifier,
    metrics: dict[str, float | dict],
) -> None:
    """Save trained model and evaluation metrics."""
    logging.info("Saving model and evaluation results")

    model_path = OUTPUT_DIR / "xgboost_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logging.info(f"Saved model to {model_path}")

    metrics_path = OUTPUT_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logging.info(f"Saved metrics to {metrics_path}")


def main() -> None:
    """Execute XGBoost classification pipeline."""
    start_time = time.time()
    logging.info("Starting XGBoost classification pipeline")

    OUTPUT_DIR.mkdir(exist_ok=True)

    train_df, test_df = load_train_test_data()

    feature_cols = [col for col in train_df.columns if col != "target"]
    X_train = train_df.select(feature_cols)
    y_train = train_df["target"]
    X_test = test_df.select(feature_cols)
    y_test = test_df["target"]

    model = train_xgboost_model(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    plot_confusion_matrix(model, X_test, y_test)
    plot_feature_importance(model, feature_cols)

    save_model_and_results(model, metrics)

    elapsed = time.time() - start_time
    logging.info(
        f"XGBoost classification pipeline completed in {elapsed:.2f} seconds"
    )


if __name__ == "__main__":
    main()
