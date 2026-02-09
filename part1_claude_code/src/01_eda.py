"""Exploratory Data Analysis for UCI Wine Dataset."""

import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.datasets import load_wine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_DIR: Path = Path("output")
FIGURE_SIZE: tuple[int, int] = (12, 8)
RANDOM_STATE: int = 42


def _detect_outliers_iqr(
    df: pl.DataFrame,
    column: str,
) -> dict[str, int | float]:
    """Detect outliers using IQR method for a single column."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df.filter(
        (pl.col(column) < lower_bound) | (pl.col(column) > upper_bound)
    )
    return {
        "column": column,
        "n_outliers": len(outliers),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
    }


def load_wine_data() -> pl.DataFrame:
    """Load UCI Wine dataset and convert to polars DataFrame."""
    logging.info("Loading UCI Wine dataset")
    wine = load_wine()
    df = pl.DataFrame(
        data=wine.data,
        schema=wine.feature_names,
    )
    df = df.with_columns(pl.Series("target", wine.target))
    logging.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    return df


def compute_summary_statistics(
    df: pl.DataFrame,
) -> dict[str, dict[str, float]]:
    """Compute summary statistics for all numeric columns."""
    logging.info("Computing summary statistics")
    stats = {}
    numeric_cols = [col for col in df.columns if col != "target"]

    for col in numeric_cols:
        stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
        }

    logging.info(f"Computed statistics for {len(stats)} features")
    return stats


def plot_feature_distributions(
    df: pl.DataFrame,
) -> None:
    """Generate distribution plots for all features."""
    logging.info("Generating feature distribution plots")
    feature_cols = [col for col in df.columns if col != "target"]
    n_features = len(feature_cols)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten()

    for idx, col in enumerate(feature_cols):
        axes[idx].hist(df[col].to_numpy(), bins=30, edgecolor="black")
        axes[idx].set_title(col, fontsize=10)
        axes[idx].set_xlabel("Value")
        axes[idx].set_ylabel("Frequency")

    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "feature_distributions.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info(f"Saved feature distributions to {output_path}")


def plot_correlation_matrix(
    df: pl.DataFrame,
) -> None:
    """Generate correlation heatmap for all features."""
    logging.info("Generating correlation heatmap")
    feature_cols = [col for col in df.columns if col != "target"]
    corr_matrix = df.select(feature_cols).to_pandas().corr()

    plt.figure(figsize=FIGURE_SIZE)
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
    )
    plt.title("Feature Correlation Matrix", fontsize=14)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "correlation_matrix.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info(f"Saved correlation matrix to {output_path}")


def check_class_balance(
    df: pl.DataFrame,
) -> dict[str, int]:
    """Check class distribution in the target variable."""
    logging.info("Checking class balance")
    class_counts = df.group_by("target").agg(pl.count()).sort("target")
    balance = {
        f"class_{int(row[0])}": int(row[1])
        for row in class_counts.iter_rows()
    }
    logging.info(f"Class distribution: {json.dumps(balance, indent=2)}")
    return balance


def detect_outliers(
    df: pl.DataFrame,
) -> list[dict[str, int | float]]:
    """Detect outliers for all numeric features using IQR method."""
    logging.info("Detecting outliers using IQR method")
    feature_cols = [col for col in df.columns if col != "target"]
    outlier_info = [_detect_outliers_iqr(df, col) for col in feature_cols]

    total_outliers = sum(info["n_outliers"] for info in outlier_info)
    logging.info(f"Detected {total_outliers} total outliers across all features")
    return outlier_info


def main() -> None:
    """Execute exploratory data analysis pipeline."""
    start_time = time.time()
    logging.info("Starting EDA pipeline")

    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_wine_data()

    stats = compute_summary_statistics(df)
    stats_path = OUTPUT_DIR / "summary_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logging.info(f"Saved summary statistics to {stats_path}")

    plot_feature_distributions(df)
    plot_correlation_matrix(df)

    class_balance = check_class_balance(df)
    balance_path = OUTPUT_DIR / "class_balance.json"
    with open(balance_path, "w") as f:
        json.dump(class_balance, f, indent=2)
    logging.info(f"Saved class balance to {balance_path}")

    outliers = detect_outliers(df)
    outliers_path = OUTPUT_DIR / "outlier_analysis.json"
    with open(outliers_path, "w") as f:
        json.dump(outliers, f, indent=2, default=str)
    logging.info(f"Saved outlier analysis to {outliers_path}")

    df.write_parquet(OUTPUT_DIR / "wine_data.parquet")
    logging.info("Saved raw data to parquet")

    elapsed = time.time() - start_time
    logging.info(f"EDA pipeline completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
