import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.datasets import load_wine


# Configure logging
def _setup_logging(debug: bool = False) -> None:
    """Configure logging level."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
        force=True,  # Reset any existing config
    )


# Constants
OUTPUT_DIR: Path = Path("part2_antigravity/output")
IQR_MULTIPLIER: float = 1.5


def _ensure_output_dir() -> None:
    """Ensure the output directory exists."""
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created output directory: {OUTPUT_DIR}")


def _load_data() -> pl.DataFrame:
    """Load the Wine dataset and convert to Polars DataFrame."""
    wine = load_wine()
    data = pl.DataFrame(wine.data, schema=wine.feature_names)
    target = pl.Series("target", wine.target)

    # Add target to dataframe
    df = data.with_columns(target)

    logging.info(f"Loaded Wine dataset with shape: {df.shape}")
    return df


def _compute_stats(df: pl.DataFrame) -> None:
    """Compute and log basic statistics."""
    stats = df.describe()
    logging.info("Dataset Statistics:")
    # Convert to strict dict for json serialization to avoid issues with polars types
    stats_dict = stats.to_dict(as_series=False)
    logging.info(json.dumps(str(stats_dict), indent=2))

    # Check for missing values
    null_counts = df.null_count()
    logging.info("Missing Value Counts:")
    logging.info(json.dumps(null_counts.row(0, named=True), indent=2))


def _check_class_balance(df: pl.DataFrame) -> None:
    """Check and log class balance."""
    balance = df.group_by("target").len().sort("target")
    logging.info("Class Balance:")
    logging.info(json.dumps(balance.to_dicts(), indent=2))

    # Plot class balance
    plt.figure(figsize=(8, 6))
    sns.barplot(x=balance["target"], y=balance["len"])
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    output_path = OUTPUT_DIR / "class_balance.png"
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Saved class balance plot to {output_path}")


def _detect_outliers(df: pl.DataFrame, features: list[str]) -> None:
    """Detect outliers using IQR method."""
    outlier_report = {}

    for feature in features:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (IQR_MULTIPLIER * iqr)
        upper_bound = q3 + (IQR_MULTIPLIER * iqr)

        outliers = df.filter((pl.col(feature) < lower_bound) | (pl.col(feature) > upper_bound))
        count = outliers.height
        outlier_report[feature] = {
            "count": count,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }

    logging.info("Outlier Detection Report (IQR method):")
    logging.info(json.dumps(outlier_report, indent=2, default=str))


def _plot_distributions(df: pl.DataFrame, features: list[str]) -> None:
    """Generate distribution plots for each feature."""
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df.to_pandas(), x=feature, kde=True, hue="target", palette="viridis")
        safe_feature = feature.replace("/", "_")
        plt.title(f"Distribution of {feature}")
        output_path = OUTPUT_DIR / f"dist_{safe_feature}.png"
        plt.savefig(output_path)
        plt.close()

    logging.info(f"Saved distribution plots for {len(features)} features")


def _plot_correlation_heatmap(df: pl.DataFrame) -> None:
    """Generate correlation heatmap."""
    # Compute correlation matrix
    corr_matrix = df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix.to_pandas(),  # seaborn needs pandas or numpy
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=df.columns,
        yticklabels=df.columns,
    )
    plt.title("Feature Correlation Heatmap")
    output_path = OUTPUT_DIR / "correlation_heatmap.png"
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Saved correlation heatmap to {output_path}")


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Perform EDA on Wine dataset.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = _parse_args()
    _setup_logging(args.debug)

    _ensure_output_dir()

    df = _load_data()
    feature_cols = [c for c in df.columns if c != "target"]

    _compute_stats(df)
    _check_class_balance(df)
    _detect_outliers(df, feature_cols)

    # Visualization
    _plot_distributions(df, feature_cols)
    _plot_correlation_heatmap(df)

    logging.info("EDA completed successfully.")


if __name__ == "__main__":
    main()
