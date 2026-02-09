Create a detailed implementation plan for building a Wine classification pipeline: $ARGUMENTS

## Instructions

Derive a short kebab-case feature name from the task description (e.g., "wine-classification-pipeline", "wine-eda", "wine-feature-engineering").

Write the plan to `.scratchpad/<feature-name>/plan.md` with the following structure:

---

# Implementation Plan: Wine Classification Pipeline

## 1. Objective

Brief description of what will be built for the Wine classification task. Include:
- Goal: Classify wines into 3 classes using UCI Wine dataset
- Dataset: 178 samples, 13 chemical features, 3 target classes
- Deliverables: EDA, feature engineering, XGBoost classifier, evaluation report

## 2. Architecture

### Data Flow
```
UCI Wine Dataset (load_wine)
    ↓
Exploratory Data Analysis (01_eda.py)
    ↓
Feature Engineering (02_feature_engineering.py)
    ↓
XGBoost Classification Model (03_xgboost_model.py)
    ↓
Evaluation & Reporting
```

### Component Responsibilities
- **01_eda.py**: Load wine data, compute statistics, visualize distributions, check class balance, detect outliers
- **02_feature_engineering.py**: Create derived features, scale features, stratified train/test split
- **03_xgboost_model.py**: Train XGBoost classifier, 5-fold CV, compute metrics, generate visualizations

### Technology Stack
- **Data manipulation**: polars (NOT pandas)
- **ML framework**: XGBoost, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Package manager**: uv (NOT pip)
- **Code quality**: ruff for linting and formatting

## 3. File Structure

```
part1_claude_code/
├── src/
│   ├── 01_eda.py                    # Exploratory data analysis
│   ├── 02_feature_engineering.py   # Feature engineering and scaling
│   └── 03_xgboost_model.py         # Model training and evaluation
├── output/                          # All output artifacts
│   ├── wine_data.parquet
│   ├── train_data.parquet
│   ├── test_data.parquet
│   ├── summary_statistics.json
│   ├── class_balance.json
│   ├── outlier_analysis.json
│   ├── feature_distributions.png
│   ├── correlation_matrix.png
│   ├── xgboost_model.pkl
│   ├── evaluation_metrics.json
│   ├── confusion_matrix.png
│   └── feature_importance.png
└── CLAUDE.md                        # Wine-specific coding standards
```

## 4. Implementation Steps

### Step 1: Exploratory Data Analysis (01_eda.py)

**File**: `part1_claude_code/src/01_eda.py`

**Functions to implement**:
1. `load_wine_data() -> pl.DataFrame`
   - Load UCI Wine dataset using `sklearn.datasets.load_wine()`
   - Convert to polars DataFrame with proper column names
   - Add target column

2. `compute_summary_statistics(df: pl.DataFrame) -> dict`
   - Calculate mean, std, min, max, median for each feature
   - Save to `output/summary_statistics.json`

3. `plot_feature_distributions(df: pl.DataFrame) -> None`
   - Create histograms for all 13 chemical features
   - Save to `output/feature_distributions.png`

4. `plot_correlation_matrix(df: pl.DataFrame) -> None`
   - Generate correlation heatmap
   - Save to `output/correlation_matrix.png`

5. `check_class_balance(df: pl.DataFrame) -> dict`
   - Count samples per wine class (0, 1, 2)
   - Save to `output/class_balance.json`

6. `detect_outliers(df: pl.DataFrame) -> list[dict]`
   - Use IQR method for each feature
   - Save to `output/outlier_analysis.json`

**Dependencies**: None (first script in pipeline)

**Output**: Parquet file + 4 visualizations + 3 JSON files

---

### Step 2: Feature Engineering (02_feature_engineering.py)

**File**: `part1_claude_code/src/02_feature_engineering.py`

**Functions to implement**:
1. `load_wine_data() -> pl.DataFrame`
   - Load from `output/wine_data.parquet`

2. `create_derived_features(df: pl.DataFrame) -> pl.DataFrame`
   - Create at least 3 derived features:
     - Ratio features (e.g., alcohol/malic_acid)
     - Interaction terms (e.g., ash * alcalinity)
     - Polynomial features (e.g., magnesium^2)
   - Handle infinite values (replace with median)

3. `scale_features(df: pl.DataFrame) -> tuple[pl.DataFrame, StandardScaler]`
   - Apply StandardScaler to all features
   - Return scaled DataFrame and fitted scaler

4. `split_train_test(df: pl.DataFrame) -> tuple`
   - Stratified split (80/20) with random_state=42
   - Maintain class balance in both sets

5. `save_processed_data(train_df, test_df) -> None`
   - Save to parquet files

**Dependencies**: Requires `output/wine_data.parquet` from Step 1

**Output**: train_data.parquet, test_data.parquet

---

### Step 3: XGBoost Classification (03_xgboost_model.py)

**File**: `part1_claude_code/src/03_xgboost_model.py`

**Functions to implement**:
1. `load_train_test_data() -> tuple`
   - Load train and test parquet files

2. `train_xgboost_model(X_train, y_train) -> XGBClassifier`
   - Configure XGBoost for multi-class classification:
     - objective="multi:softmax"
     - num_class=3
     - eval_metric="mlogloss"
   - Perform 5-fold stratified cross-validation
   - Log CV scores (mean ± std)
   - Fit final model on full training set

3. `evaluate_model(model, X_test, y_test) -> dict`
   - Compute metrics:
     - Accuracy
     - Precision (weighted)
     - Recall (weighted)
     - F1-score (weighted)
     - Per-class metrics
   - Save to `output/evaluation_metrics.json`

4. `plot_confusion_matrix(model, X_test, y_test) -> None`
   - Generate confusion matrix heatmap
   - Save to `output/confusion_matrix.png`

5. `plot_feature_importance(model, feature_names) -> None`
   - Extract and visualize top 15 features
   - Save to `output/feature_importance.png`

6. `save_model_and_results(model, metrics) -> None`
   - Save model to `output/xgboost_model.pkl`

**Dependencies**: Requires train_data.parquet and test_data.parquet from Step 2

**Output**: model.pkl + metrics.json + 2 visualizations

---

## 5. Technical Decisions

### Data Manipulation Library
- **Choice**: polars
- **Rationale**: CLAUDE.md mandates polars over pandas
- **Trade-off**: Less familiar API, but faster and more memory-efficient

### Classification Model
- **Choice**: XGBoost Classifier
- **Rationale**:
  - Excellent for small-to-medium tabular data
  - Handles non-linear relationships well
  - Provides feature importance
- **Alternative considered**: Random Forest (less powerful), SVM (no feature importance)

### Cross-Validation Strategy
- **Choice**: Stratified 5-fold CV
- **Rationale**:
  - Stratification maintains class balance in each fold
  - 5 folds is standard for datasets of this size (178 samples)
- **Trade-off**: More computation vs better generalization estimate

### Feature Scaling
- **Choice**: StandardScaler
- **Rationale**:
  - Required for many features to be on same scale
  - Wine chemical properties have different units and ranges
- **Alternative considered**: MinMaxScaler (less robust to outliers)

### Train/Test Split
- **Choice**: 80/20 stratified split with random_state=42
- **Rationale**:
  - Standard split ratio for small datasets
  - Stratification ensures balanced class distribution
  - Fixed seed for reproducibility

## 6. Testing Strategy

### Automated Testing (Hooks)
- PostToolUse hook runs `ruff check --fix` on every Python file
- PostToolUse hook runs `py_compile` to verify syntax
- Ensures code quality without manual intervention

### Manual Verification
1. Run each script in sequence and verify outputs exist
2. Check log messages for expected format and content
3. Verify parquet files contain expected columns and row counts
4. Inspect visualizations for correctness
5. Validate metrics are in expected ranges (accuracy 70-100%)

### Data Validation
- Verify class balance in train/test splits
- Check for data leakage between train and test
- Ensure no NaN or infinite values after feature engineering
- Confirm feature scaling worked (mean≈0, std≈1)

## 7. Expected Output

### Files to be created
- `src/01_eda.py` (200-250 lines)
- `src/02_feature_engineering.py` (150-200 lines)
- `src/03_xgboost_model.py` (200-250 lines)

### Artifacts in output/
- 3 parquet data files (wine_data, train_data, test_data)
- 5 JSON files (statistics, balance, outliers, metrics, etc.)
- 4 PNG visualizations (distributions, correlation, confusion matrix, feature importance)
- 1 trained model file (.pkl)

### Expected Performance
- Accuracy: 85-100% (Wine dataset is relatively easy to classify)
- CV score variance: Low (< 5%) indicating stable model
- Class balance: Roughly equal in train/test splits
- No errors or warnings during execution

## 8. Dependencies Between Steps

```
Step 1 (EDA)
    ↓ (produces wine_data.parquet)
Step 2 (Feature Engineering)
    ↓ (produces train_data.parquet, test_data.parquet)
Step 3 (XGBoost Model)
```

**Critical path**: Must execute in order (1 → 2 → 3)

## 9. Compliance with CLAUDE.md

All code must follow these standards:
- ✅ Use polars for data manipulation
- ✅ Type annotations for all function parameters
- ✅ Private functions prefixed with `_` and placed at top of file
- ✅ Constants declared at file top with type annotations
- ✅ Functions under 50 lines
- ✅ Prescribed logging format
- ✅ Multi-line imports
- ✅ Stratified splits with random_state=42
- ✅ Classification metrics (not regression)
- ✅ XGBoost with objective="multi:softmax" and num_class=3

## 10. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Infinite values from feature ratios | Replace with median using `_replace_infinite_values()` |
| Class imbalance affecting metrics | Use stratified splits and weighted metrics |
| Overfitting on small dataset | Use 5-fold CV and monitor train/test gap |
| Hook failures blocking progress | Scripts include proper error handling |
| Missing dependencies | Use `uv` for reproducible environment |

---

## Next Steps

After reviewing this plan:
1. ✅ Approve the plan as-is, OR
2. 🔄 Request modifications to approach, OR
3. ❌ Reject and request a different approach

Once approved, implementation will proceed in the order specified above.

---

**Note**: This plan follows the Wine Classification requirements in `part1_claude_code/CLAUDE.md`. All code will be automatically checked by hooks for compliance with coding standards.
