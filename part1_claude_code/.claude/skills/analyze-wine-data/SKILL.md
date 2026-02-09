---
name: analyze-wine-data
description: Perform exploratory data analysis on the UCI Wine dataset. Use when asked to explore, profile, or analyze wine data.
argument-hint: [optional dataset path]
---

# Wine Dataset Analysis Skill

When performing exploratory data analysis on the UCI Wine dataset, follow these steps:

## 1. Load Wine Dataset
- Use `sklearn.datasets.load_wine()` to load the UCI Wine dataset
- Convert to polars DataFrame with feature names as columns
- Add target column with wine class labels (0, 1, 2)
- Log dataset dimensions: 178 samples, 13 features, 3 classes

## 2. Dataset Overview
- Print first few rows to inspect data structure
- List all feature names (alcohol, malic_acid, ash, etc.)
- Verify target has exactly 3 unique classes

## 3. Summary Statistics
- Compute mean, median, std, min, max for each chemical feature
- Use polars operations (not pandas)
- Save statistics to `output/summary_statistics.json`
- Log key statistics using the project's logging format

## 4. Class Balance Analysis
- Count samples per wine class (0, 1, 2)
- Calculate percentage distribution
- Create a bar plot showing class distribution
- Save to `output/wine_class_distribution.png`
- Report if dataset is balanced or imbalanced

## 5. Missing Values Check
- Check for missing values in each column
- Report count and percentage per column
- Log whether dataset has any missing data

## 6. Feature Distributions
- Generate histogram plots for all 13 chemical features
- Use 4 columns layout with appropriate rows
- Highlight any skewed distributions
- Save to `output/wine_feature_distributions.png`

## 7. Correlation Analysis
- Create correlation matrix for all features
- Generate heatmap using seaborn with 'coolwarm' colormap
- Identify highly correlated feature pairs (|r| > 0.7)
- Save to `output/wine_correlation_matrix.png`
- Log top 5 most correlated feature pairs

## 8. Outlier Detection
- Use IQR method for each chemical feature
- Count outliers per feature
- Calculate percentage of samples with outliers
- Save outlier analysis to `output/wine_outlier_analysis.json`
- Log features with most outliers

## 9. Chemical Feature Insights
- Analyze which features best discriminate between wine classes
- Create box plots showing feature distribution per class
- Identify features with highest separation between classes
- Save to `output/wine_features_by_class.png`

## 10. Summary Report
- Log a summary of key findings:
  - Dataset balance status
  - Features with highest variance
  - Most correlated features
  - Features with most outliers
  - Recommendations for feature engineering

## Technical Requirements
- Use polars (not pandas) for all data manipulation
- Follow CLAUDE.md coding standards
- Use the prescribed logging format
- Save all plots to `output/` directory with 150 dpi
- Pretty-print JSON outputs with indent=2

## Output Files
- `output/summary_statistics.json`
- `output/wine_class_distribution.png`
- `output/wine_feature_distributions.png`
- `output/wine_correlation_matrix.png`
- `output/wine_outlier_analysis.json`
- `output/wine_features_by_class.png`
