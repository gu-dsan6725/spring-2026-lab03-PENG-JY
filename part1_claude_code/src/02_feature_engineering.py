"""Feature Engineering for UCI Wine Dataset."""

import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

INPUT_DIR: Path = Path("output")
OUTPUT_DIR: Path = Path("output")
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42


def _create_ratio_feature(
    df: pl.DataFrame,
    numerator: str,
    denominator: str,
    name: str,
) -> pl.DataFrame:
    """Create a ratio feature from two columns."""
    return df.with_columns(
        (pl.col(numerator) / pl.col(denominator)).alias(name)
    )


def _replace_infinite_values(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Replace infinite values with column median."""
    for col in df.columns:
        if df[col].dtype in [pl.Float64, pl.Float32]:
            col_median = df.filter(pl.col(col).is_finite())[col].median()
            df = df.with_columns(
                pl.when(pl.col(col).is_infinite())
                .then(col_median)
                .otherwise(pl.col(col))
                .alias(col)
            )
    return df


def load_wine_data() -> pl.DataFrame:
    """Load wine data from parquet file."""
    logging.info("Loading wine data from parquet")
    input_path = INPUT_DIR / "wine_data.parquet"
    df = pl.read_parquet(input_path)
    logging.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    return df


def create_derived_features(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Create derived features from existing wine features."""
    logging.info("Creating derived features")

    df = _create_ratio_feature(
        df,
        "alcohol",
        "malic_acid",
        "alcohol_malic_ratio",
    )

    df = _create_ratio_feature(
        df,
        "total_phenols",
        "flavanoids",
        "phenol_flavanoid_ratio",
    )

    df = df.with_columns(
        (pl.col("ash") * pl.col("alcalinity_of_ash")).alias(
            "ash_alcalinity_interaction"
        )
    )

    df = df.with_columns(
        (
            pl.col("color_intensity") / pl.col("od280/od315_of_diluted_wines")
        ).alias("color_dilution_ratio")
    )

    df = df.with_columns(
        (pl.col("magnesium") ** 2).alias("magnesium_squared")
    )

    df = _replace_infinite_values(df)

    n_derived = 5
    logging.info(f"Created {n_derived} derived features")
    return df


def scale_features(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, StandardScaler]:
    """Apply standard scaling to feature columns."""
    logging.info("Applying standard scaling to features")

    feature_cols = [col for col in df.columns if col != "target"]
    target_col = df["target"].to_numpy()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.select(feature_cols).to_numpy())

    scaled_df = pl.DataFrame(
        scaled_features,
        schema=feature_cols,
    )
    scaled_df = scaled_df.with_columns(pl.Series("target", target_col))

    logging.info(f"Scaled {len(feature_cols)} features")
    return scaled_df, scaler


def split_train_test(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray, np.ndarray]:
    """Create stratified train/test split."""
    logging.info(
        f"Creating stratified train/test split with test_size={TEST_SIZE}"
    )

    feature_cols = [col for col in df.columns if col != "target"]
    X = df.select(feature_cols).to_numpy()
    y = df["target"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_df = pl.DataFrame(
        X_train,
        schema=feature_cols,
    )
    train_df = train_df.with_columns(pl.Series("target", y_train))

    test_df = pl.DataFrame(
        X_test,
        schema=feature_cols,
    )
    test_df = test_df.with_columns(pl.Series("target", y_test))

    logging.info(f"Train set: {len(train_df)} samples")
    logging.info(f"Test set: {len(test_df)} samples")

    return train_df, test_df, y_train, y_test


def save_processed_data(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
) -> None:
    """Save processed train and test datasets."""
    logging.info("Saving processed datasets")

    train_path = OUTPUT_DIR / "train_data.parquet"
    test_path = OUTPUT_DIR / "test_data.parquet"

    train_df.write_parquet(train_path)
    test_df.write_parquet(test_path)

    logging.info(f"Saved train data to {train_path}")
    logging.info(f"Saved test data to {test_path}")


def main() -> None:
    """Execute feature engineering pipeline."""
    start_time = time.time()
    logging.info("Starting feature engineering pipeline")

    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_wine_data()

    df = create_derived_features(df)

    scaled_df, scaler = scale_features(df)

    train_df, test_df, y_train, y_test = split_train_test(scaled_df)

    save_processed_data(train_df, test_df)

    logging.info(f"Feature names: {[col for col in train_df.columns if col != 'target']}")

    elapsed = time.time() - start_time
    logging.info(f"Feature engineering completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
