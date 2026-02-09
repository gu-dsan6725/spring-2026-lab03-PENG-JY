# Implementation Plan - Wine Classification Pipeline

This plan outlines the steps to build a complete ML pipeline for the UCI Wine dataset using `sklearn.datasets.load_wine()`.

## Goal Description
Build a robust XGBoost classification pipeline to classify wine cultivars (3 classes) based on 13 chemical features. The pipeline will include EDA, automated feature engineering, cross-validated training, and a comprehensive evaluation report.

## User Review Required
> [!IMPORTANT]
> The plan includes creating specific derived features (`alcohol_to_malic`, `magnesium_richness`, `proline_interaction`). Please review if these domain-specific features are acceptable.

## Proposed Changes

### Scripts (`part2_antigravity/src/`)

#### [NEW] [01_eda.py](file:///home/ubuntu/dsan6725/spring-2026-lab03-PENG-JY/part2_antigravity/src/01_eda.py)
**Goal**: Analyze dataset structure and quality.
- **Input**: `sklearn.datasets.load_wine()`
- **Steps**:
    1. Load data into Polars DataFrame.
    2. Compute basic stats (mean, std, min, max, nulls).
    3. Check class balance (target counts).
    4. Detect outliers using IQR (1.5*IQR rule) for key features.
    5. **Plots** (saved to `output/`):
        - `dist_*.png`: Histograms for all 13 features.
        - `correlation_heatmap.png`: Seaborn heatmap of feature correlations.
        - `pairplot_key_features.png`: Pairplot of top correlation features.

#### [NEW] [02_feature_engineering.py](file:///home/ubuntu/dsan6725/spring-2026-lab03-PENG-JY/part2_antigravity/src/02_feature_engineering.py)
**Goal**: Prepare data for modeling.
- **Input**: `sklearn.datasets.load_wine()`
- **Derived Features**:
    1. `alcohol_to_malic`: `alcohol` / `malic_acid` (ratio of strength to acidity)
    2. `magnesium_richness`: `magnesium` / `ash` (mineral composition)
    3. `proline_log`: `log1p(proline)` (normalize skew in proline)
- **Transformations**:
    - StandardScaler for all features.
- **Splitting**:
    - Stratified ShuffleSplit (80% train, 20% test, random_state=42).
- **Output**:
    - `output/X_train.parquet`, `output/X_test.parquet`, `output/y_train.parquet`, `output/y_test.parquet` (saved using Polars).

#### [NEW] [03_xgboost_model.py](file:///home/ubuntu/dsan6725/spring-2026-lab03-PENG-JY/part2_antigravity/src/03_xgboost_model.py)
**Goal**: Train and evaluate model.
- **Input**: Parquet files from Step 2.
- **Model**: `XGBClassifier` (objective='multi:softprob', num_class=3).
- **Validation**:
    - **5-Fold Cross-Validation** on training set to estimate stable performance.
- **Hyperparameters**: Use defaults or minimal tuning (n_estimators=100, learning_rate=0.1, max_depth=3) unless CV suggests otherwise.
- **Evaluation**:
    - Key Metrics: Accuracy, Weighted Precision, Weighted Recall, Weighted F1-Score.
    - **Plots**:
        - `confusion_matrix.png`: Heatmap of true vs predicted labels.
        - `feature_importance.png`: Top 10 important features (gain).
- **Output**:
    - `output/xgboost_model.joblib`: Saved model.
    - `output/evaluation_report.md`: Markdown summary of metrics and recommendations.

## Verification Plan

### Automated Verification
- Run `uv run python part2_antigravity/src/01_eda.py` -> Check `output/` for plots.
- Run `uv run python part2_antigravity/src/02_feature_engineering.py` -> Check `output/*.parquet`.
- Run `uv run python part2_antigravity/src/03_xgboost_model.py` -> Check `output/evaluation_report.md` and score > 0.90.
