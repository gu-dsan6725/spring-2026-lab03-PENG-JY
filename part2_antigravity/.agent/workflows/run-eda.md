# Run EDA Workflow

Perform exploratory data analysis on the Wine dataset.

## Steps

1. Load the dataset from `sklearn.datasets.load_wine()` and convert to a polars DataFrame.
2. Compute summary statistics (mean, median, std, min, max) for each feature.
3. Check for missing values and report findings.
4. Check the class balance of the target variable (3 wine classes).
5. Generate distribution histograms for each feature using matplotlib.
6. Create a correlation matrix heatmap using seaborn.
7. Identify outliers using the IQR method.
8. Save all plots to the `output/` directory.
9. Log a summary of findings.

## Requirements

- Use polars (not pandas) for all data manipulation.
- Follow the coding standards in `.agent/rules/code-style-guide.md`.
- Save the EDA script as `part2_antigravity/src/01_eda.py`.
- After writing the file, run `uv run ruff check --fix` and `uv run python -m py_compile`.
