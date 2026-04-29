# Model Card: Civic Complaint Triage Model

## Project name
Civic Complaint Triage Model

## Intended use
This model is a prototype triage signal to help analysts prioritize review of Philadelphia L&I complaints. It is designed for decision support, not automation.

## Not intended use
- Automated enforcement or service denial
- Use as a final decision maker
- Use outside the Philadelphia L&I complaint context without validation

## Dataset
Public Philadelphia L&I complaint data (CSV):
https://phl.carto.com/api/v2/sql?q=SELECT+*,+ST_Y(the_geom)+AS+lat,+ST_X(the_geom)+AS+lng+FROM+complaints&filename=complaints&format=csv&skipfields=cartodb_id

## Target definition
`delayed_30 = 1` if `days_to_resolution` is greater than 30 days, otherwise `0`.

## Features used
- Complaint type, source, ZIP code, and location (lat/lng when available)
- Submission time features (month, day of week, weekend)
- Aggregate workload features (counts by type and ZIP)
- See `reports/selected_features.txt` for the final list used in the latest run

## Models trained
- Majority-class baseline (predicts the most common class only)
- Logistic regression (baseline ML model)
- Random forest (main model)

## Evaluation metrics
- Accuracy, precision, recall, F1
- ROC-AUC (for models with `predict_proba`)
- Confusion counts (TP/FP/TN/FN)
- Predicted vs actual delayed rate
- Threshold sweep for F1 optimization
- Balanced-rate threshold to align predicted and actual delayed rates

## Key results (latest run, primary target: delayed_30)
- Majority baseline: accuracy 0.558, precision 0.558, recall 1.000, F1 0.717, ROC-AUC N/A
- Logistic regression: accuracy 0.601, precision 0.590, recall 0.930, F1 0.722, ROC-AUC 0.654
- Random forest (default threshold): accuracy 0.601, precision 0.586, recall 0.978, F1 0.733, ROC-AUC 0.690
- Random forest (balanced-rate threshold 0.7): precision 0.685, recall 0.719, F1 0.702
- Actual delayed rate: 0.558; predicted delayed rate at default threshold: 0.932
- Predicted delayed rate at balanced-rate threshold: 0.586

## Risks and limitations
- High recall with low precision means many non-delayed complaints are flagged at the default threshold.
- Complaint volume can reflect reporting behavior rather than true need.
- Data quality and missing fields may bias results.
- This model is a prototype triage signal, not an automated decision system.

## Bias and fairness concerns
- Complaint patterns vary by neighborhood and reporting access.
- Location-based features may reflect systemic reporting differences.
- No fairness evaluation has been performed yet.

## Recommended next steps
- Add fairness checks by region and ZIP
- Compare balanced-rate thresholds across targets and time slices
- Calibrate probability thresholds to better balance precision/recall
- Test temporal cross-validation or holdout by year
- Review top features for potential leakage or bias
