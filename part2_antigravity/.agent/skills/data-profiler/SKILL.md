---
name: data-profiler
description: Profile a dataset by computing statistics, data types, distributions, and quality metrics. Use when asked to profile, describe, or summarize a dataset.
---

When profiling a dataset, generate a comprehensive data profile:

1. **Basic Info**: Number of rows, columns, memory usage
2. **Column Types**: Data type for each column, count of numeric vs categorical
3. **Summary Statistics**: Mean, median, std, min, max, quartiles for numeric columns
4. **Missing Values**: Count and percentage of nulls per column
5. **Unique Values**: Count of unique values per column
6. **Distribution Shape**: Skewness and kurtosis for numeric columns
7. **Outlier Detection**: Count of outliers per column using the IQR method
8. **Correlation**: Top 5 feature pairs AND a full **Correlation Heatmap**.
9. **Class Balance**: Count and plot of target classes (for classification tasks).

Save the profile as a markdown report to `output/data_profile.md`.

Use polars for data manipulation. Follow the project's coding standards.
