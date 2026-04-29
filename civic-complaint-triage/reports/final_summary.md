# Final Summary: Civic Complaint Triage Model

## Problem statement
Build a beginner-friendly model to predict whether a Philadelphia L&I complaint is likely to remain unresolved beyond 30 days.

## Dataset source
Public Philadelphia L&I complaint data (CSV):
https://phl.carto.com/api/v2/sql?q=SELECT+*,+ST_Y(the_geom)+AS+lat,+ST_X(the_geom)+AS+lng+FROM+complaints&filename=complaints&format=csv&skipfields=cartodb_id

## Cleaning summary
- Parsed open and close dates to compute `days_to_resolution`
- Removed missing and negative resolution intervals
- Created `delayed_30` target based on a fixed 30-day threshold
- Standardized common fields (type, source, ZIP, lat/lng)

## Feature engineering summary
- Time features: month, day of week, weekend
- Workload features: counts by complaint type and ZIP
- Leakage filters removed post-outcome fields

## Model comparison summary
- Majority-class baseline for context
- Logistic regression baseline
- Random forest as main model
- Random forest provides the strongest overall results for the 30-day target

## Threshold tuning summary
- Threshold sweep from 0.1 to 0.9
- Best-F1 threshold recorded per model in `reports/model_metrics_delayed_30.csv`
- Balanced-rate threshold reported to align predicted delayed rate with the actual delayed rate

## Key results (primary target: delayed_30)
- Random forest (default threshold): accuracy 0.601, precision 0.586, recall 0.978, F1 0.733, ROC-AUC 0.690
- Actual delayed rate: 0.558; predicted delayed rate at default threshold: 0.932
- Balanced-rate threshold (0.7): precision 0.685, recall 0.719, F1 0.702
- Balanced-rate predicted delayed rate: 0.586

## Feature importance summary
- Random forest importance saved in `reports/random_forest_feature_importance.csv`
- Top 15 features chart saved in `reports/figures/random_forest_top_features.png`

## Limitations
- High recall but modest precision suggests over-flagging at the default threshold
- Complaint data may be incomplete or inconsistent
- Reporting behavior can bias outcomes
- Prototype triage signal only, not an automated decision system

## Resume-ready project summary
Built an end-to-end ML pipeline using public Philadelphia L&I complaint data to predict complaints likely to remain unresolved beyond 30 days. Engineered temporal, geographic, and workload features, compared baseline vs ML models, and evaluated performance with ROC-AUC, confusion counts, threshold tuning, and feature importance. Added balanced-rate thresholding to align predicted delayed rates with observed outcomes for a more conservative triage signal.
