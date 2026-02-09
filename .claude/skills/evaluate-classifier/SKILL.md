---
name: evaluate-classifier
description: Evaluate a trained Wine classification model and generate a comprehensive performance report. Use when asked to evaluate, assess, or analyze classifier performance.
argument-hint: [optional model path]
---

# Wine Classifier Evaluation Skill

When evaluating a trained Wine classification model, follow these steps:

## 1. Load Trained Model
- Look for model files in `output/` directory (.pkl, .joblib files)
- If $ARGUMENTS specifies a path, use that
- Load using pickle or joblib
- Verify model type is a classifier (not regressor)
- Log model type and configuration

## 2. Load Test Data
- Load test dataset from `output/test_data.parquet`
- Separate features and target (3 wine classes: 0, 1, 2)
- Verify data shape matches training expectations
- Log number of test samples

## 3. Generate Predictions
- Use model.predict() to get class predictions
- Use model.predict_proba() to get class probabilities (if available)
- Verify predictions are in range [0, 1, 2]
- Log prediction distribution

## 4. Compute Classification Metrics
Calculate and log the following metrics:

### Overall Metrics
- **Accuracy**: Overall classification accuracy
- **Macro-averaged Precision**: Average precision across classes (unweighted)
- **Macro-averaged Recall**: Average recall across classes (unweighted)
- **Macro-averaged F1-Score**: Average F1 across classes (unweighted)
- **Weighted-averaged Precision**: Weighted by class support
- **Weighted-averaged Recall**: Weighted by class support
- **Weighted-averaged F1-Score**: Weighted by class support

### Per-Class Metrics
For each wine class (0, 1, 2):
- Precision for class i
- Recall for class i
- F1-score for class i
- Support (number of samples in class i)

Save all metrics to `output/classifier_metrics.json`

## 5. Confusion Matrix Analysis
- Generate confusion matrix using sklearn.metrics.confusion_matrix
- Create heatmap visualization:
  - Use seaborn with 'Blues' colormap
  - Annotate with actual counts
  - Add class labels: "Class 0", "Class 1", "Class 2"
  - Title: "Wine Classification Confusion Matrix"
- Save to `output/wine_confusion_matrix.png`
- Analyze and log:
  - Which classes are most confused
  - Classification accuracy per class
  - Common misclassification patterns

## 6. Cross-Validation Scores (if available)
- If CV scores exist in output/, load them
- Report mean and standard deviation
- Compare test score to CV mean
- Assess if model generalizes well

## 7. Feature Importance Analysis
- Extract feature importances from model (if tree-based)
- Or extract coefficients (if linear model)
- Rank features by importance
- Create horizontal bar chart showing top 15 features
- Save to `output/wine_feature_importance.png`
- Log top 5 most important chemical features

## 8. Classification Report
- Generate sklearn classification_report
- Format as markdown table
- Include precision, recall, f1-score, support for each class
- Save to `output/wine_classification_report.md`

## 9. Probability Calibration Analysis (if applicable)
- If model supports predict_proba:
  - Analyze prediction confidence distribution
  - Create histogram of prediction probabilities
  - Check for overconfident or underconfident predictions
  - Save to `output/wine_prediction_confidence.png`

## 10. Error Analysis
- Identify misclassified samples
- Analyze characteristics of misclassified wines:
  - Which features differ most from correctly classified?
  - Are misclassifications systematic or random?
- Create visualization of errors by feature
- Save to `output/wine_error_analysis.png`

## 11. Performance Summary
Generate a comprehensive evaluation report with:

### Executive Summary (2-3 sentences)
- Overall model performance
- Best and worst performing classes
- Key strengths and weaknesses

### Detailed Findings
- Metrics table (all computed metrics)
- Confusion matrix interpretation
- Feature importance insights
- Error patterns identified

### Recommendations (at least 3)
Examples:
- Which features to engineer further
- Which classes need more training data
- Hyperparameter tuning suggestions
- Data collection recommendations

Save to `output/wine_evaluation_report.md`

## 12. Comparison with Baseline
- Calculate baseline accuracy (majority class classifier)
- Calculate baseline using stratified random guessing
- Report improvement over baseline
- Log relative performance gain

## Technical Requirements
- Use polars for data loading (not pandas)
- Follow CLAUDE.md coding standards
- Use prescribed logging format
- Save all plots with 150 dpi to `output/`
- Pretty-print metrics using json.dumps(indent=2)

## Output Files
- `output/classifier_metrics.json`
- `output/wine_confusion_matrix.png`
- `output/wine_feature_importance.png`
- `output/wine_classification_report.md`
- `output/wine_prediction_confidence.png`
- `output/wine_error_analysis.png`
- `output/wine_evaluation_report.md`

## Success Criteria
- All metrics computed without errors
- Visualizations clearly show model performance
- Report provides actionable insights
- Recommendations are specific to Wine classification problem
