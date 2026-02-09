---
name: generate-report
description: Generate a comprehensive markdown report for a trained ML model. Use when asked to create a report, summarize model results, or document model performance.
argument-hint: [model-output-dir]
---

Generate a comprehensive model evaluation report by following these steps:

1. **Locate artifacts** in the directory specified by $ARGUMENTS (default: `output/`).
   Read `evaluation_report.md` if it exists to gather existing metrics.

2. **Load the trained model** (look for `.joblib` or `.pkl` files) and extract:
   - Model type and hyperparameters
   - Feature names and count
   - Feature importance scores (if the model supports them)

3. **Load test data** (look for `x_test.parquet` and `y_test.parquet`) and compute:
   - Prediction distribution statistics
   - Error distribution statistics
   - Any metrics not already in the evaluation report

4. **Fill in the report template** from [templates/report_template.md](templates/report_template.md).
   Replace each placeholder section with actual data:
   - Executive Summary: 2-3 sentence overview of model performance
   - Dataset Overview: number of samples, features, target variable description
   - Model Configuration: model type, all hyperparameters in a table
   - Performance Metrics: all computed metrics in a table
   - Feature Importance: top 5 features ranked by importance with scores
   - Recommendations: at least 3 actionable suggestions for improvement

5. **Save the completed report** to `output/full_report.md`.

6. **Save the report generation code** as a standalone Python script.
   - Collect all the code used in steps 1-5 into a single `.py` file.
   - The script should be runnable independently (include all imports, logging config, and a `main()` function).
   - Ask the user where they would like to save the script (suggest a default like `scripts/generate_report.py`).

7. **Log a summary** of what was generated using the project's logging format.

Follow the coding standards in CLAUDE.md. Use polars for any data loading.
