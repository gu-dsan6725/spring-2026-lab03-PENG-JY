# Wine Classification ML Project

## Project Overview
This project performs exploratory data analysis and builds an XGBoost classification model on the UCI Wine dataset from scikit-learn. The dataset contains 178 wine samples with 13 chemical features (alcohol, malic acid, ash, etc.) across 3 wine classes. The goal is to classify wines into the correct class based on their chemical properties.

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

## Wine Classification Specific Requirements

### Dataset
- Use `sklearn.datasets.load_wine()` to load the UCI Wine dataset
- Dataset has 178 samples, 13 features, and 3 target classes (0, 1, 2)
- Features include: alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins, color_intensity, hue, od280/od315_of_diluted_wines, proline

### Classification Requirements
- Use **stratified train/test split** to maintain class balance (test_size=0.2, random_state=42)
- Use **stratified K-fold cross-validation** (5 folds) for model training
- Always set random_state=42 for reproducibility

### Evaluation Metrics
For classification tasks, always report:
- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted precision across all classes
- **Recall**: Weighted recall across all classes
- **F1-Score**: Weighted F1-score across all classes
- **Confusion Matrix**: Visualize as a heatmap with class labels
- **Per-class metrics**: Include precision, recall, and F1 for each wine class
- **Cross-validation scores**: Report mean and standard deviation

### Feature Engineering
- Create at least 3 derived features from the existing chemical properties
- Examples: ratios between features, polynomial features, interaction terms
- Apply StandardScaler to all features before model training
- Handle any infinite or NaN values that result from feature engineering

### Model Requirements
- Use XGBoost classifier (`xgb.XGBClassifier`)
- Set objective="multi:softmax" for multi-class classification
- Set num_class=3 for the three wine classes
- Use eval_metric="mlogloss" for multi-class log loss
- Generate feature importance plot and save to output/
