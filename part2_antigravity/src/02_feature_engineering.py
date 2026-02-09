import argparse
import logging
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


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
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2


def _ensure_output_dir() -> None:
    """Ensure the output directory exists."""
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_data() -> tuple[pl.DataFrame, pl.Series]:
    """Load the Wine dataset."""
    wine = load_wine()
    X = pl.DataFrame(wine.data, schema=wine.feature_names)
    y = pl.Series("target", wine.target)
    logging.info(f"Loaded dataset: X shape={X.shape}, y shape={y.shape}")
    return X, y


def _create_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    """Create derived features using Polars expressions."""
    logging.info("Creating derived features...")

    # alcohol_to_malic: alcohol / malic_acid
    # magnesium_richness: magnesium / ash
    # proline_log: log1p(proline)

    df_derived = df.with_columns(
        [
            (pl.col("alcohol") / pl.col("malic_acid")).alias("alcohol_to_malic"),
            (pl.col("magnesium") / pl.col("ash")).alias("magnesium_richness"),
            pl.col("proline").log1p().alias("proline_log"),
        ]
    )

    logging.info(f"Added 3 derived features. New shape: {df_derived.shape}")
    return df_derived


def _split_data(X: pl.DataFrame, y: pl.Series) -> tuple:
    """Split data into train and test sets using Stratified ShuffleSplit."""
    logging.info(f"Splitting data with test_size={TEST_SIZE}, random_state={RANDOM_STATE}")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # SSS expects numpy arrays or pandas, but works with simple iterables for indices
    # We pass zeros just to get indices based on y
    train_idx, test_idx = next(sss.split(np.zeros(len(y)), y))

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    logging.info(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    logging.info(f"Test shapes: X={X_test.shape}, y={y_test.shape}")

    return X_train, X_test, y_train, y_test


def _scale_features(
    X_train: pl.DataFrame, X_test: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Scale features using StandardScaler."""
    logging.info("Scaling features with StandardScaler...")

    scaler = StandardScaler()

    # Convert to numpy for sklearn
    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()

    # Fit on train, transform on both
    X_train_scaled_np = scaler.fit_transform(X_train_np)
    X_test_scaled_np = scaler.transform(X_test_np)

    # Convert back to Polars to preserve column names
    X_train_scaled = pl.DataFrame(X_train_scaled_np, schema=X_train.columns)
    X_test_scaled = pl.DataFrame(X_test_scaled_np, schema=X_test.columns)

    return X_train_scaled, X_test_scaled


def _save_data(
    X_train: pl.DataFrame,
    X_test: pl.DataFrame,
    y_train: pl.Series,
    y_test: pl.Series,
) -> None:
    """Save processed data to parquet."""
    X_train.write_parquet(OUTPUT_DIR / "X_train.parquet")
    X_test.write_parquet(OUTPUT_DIR / "X_test.parquet")

    # Save targets as DataFrame to keep it simple for parquet
    pl.DataFrame([y_train]).write_parquet(OUTPUT_DIR / "y_train.parquet")
    pl.DataFrame([y_test]).write_parquet(OUTPUT_DIR / "y_test.parquet")

    logging.info(f"Saved processed data to {OUTPUT_DIR}")


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Perform feature engineering on Wine dataset.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = _parse_args()
    _setup_logging(args.debug)

    _ensure_output_dir()

    X, y = _load_data()
    X = _create_derived_features(X)

    X_train, X_test, y_train, y_test = _split_data(X, y)
    X_train, X_test = _scale_features(X_train, X_test)

    _save_data(X_train, X_test, y_train, y_test)
    logging.info("Feature engineering completed successfully.")


if __name__ == "__main__":
    main()
