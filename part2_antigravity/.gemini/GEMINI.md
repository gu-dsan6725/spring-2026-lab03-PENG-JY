# ML Project - Antigravity Rules

## Project Overview
This project performs exploratory data analysis and builds an XGBoost model for the **UCI Wine Classification** task (`sklearn.datasets.load_wine()`).

## Project Requirements

### 1. Exploratory Data Analysis (EDA)
- **Profile**: Summary statistics, missing values, data types.
- **Plots**: Distribution plots (histograms/KDE) for *all* features, Correlation Heatmap, Class Balance plot.
- **Analysis**: Check for outliers using IQR.

### 2. Feature Engineering
- **Derived Features**: Must create at least 3 domain-specific features (e.g., alcohol/malic ratio).
- **Scaling**: Use `StandardScaler`.
- **Splitting**: Use `StratifiedShuffleSplit` (80% train / 20% test).

### 3. Model Training
- **Algorithm**: `XGBoost` Classifier (`XGBClassifier`).
- **Validation**: 5-Fold Cross-Validation on training set.
- **Optimization**: `RandomizedSearchCV` (n_iter=20).

### 4. Reporting
- **Metrics**: Accuracy, Precision, Recall, F1-Score (Weighted & Per-Class).
- **Plots**: Confusion Matrix, Feature Importance.
- **Report**: Save comprehensive markdown report to `output/evaluation_report.md`.

## Coding Standards

### Language and Tools
- Use Python 3.11+
- Use `uv` for package management (never pip)
- Use `polars` for data manipulation (not pandas)
- Use `ruff` for linting and formatting
- Use `pytest` for testing

### Code Style
- Use type annotations for all function parameters (one parameter per line)
- All private functions must start with underscore (`_`) and be placed at the top of the file
- Public functions follow after private functions
- Functions should be no more than 30-50 lines
- Two blank lines between function definitions
- Use multi-line imports

### Logging
Always use this logging configuration:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
```

### Constants
- Do not hard-code constants inside functions
- Declare constants at the top of the file with type annotations

### After Writing Python Files
- Always run `uv run ruff check --fix <filename>` after writing Python files
- Always run `uv run python -m py_compile <filename>` to verify syntax

### Output
- Save plots to the `output/` directory
- Use `logging.info()` for progress messages
- Pretty-print dictionaries in log messages using `json.dumps(data, indent=2, default=str)`
