# Civic Complaint Triage Model

## Problem statement
Create a beginner-friendly model that predicts whether a Philadelphia Licenses and Inspections complaint is likely to have a delayed resolution.

## Why this matters
City agencies receive large volumes of complaints. A simple triage signal can help prioritize follow-up and improve response planning.

## Dataset
Public Philadelphia L&I complaint data (CSV):
https://phl.carto.com/api/v2/sql?q=SELECT+*,+ST_Y(the_geom)+AS+lat,+ST_X(the_geom)+AS+lng+FROM+complaints&filename=complaints&format=csv&skipfields=cartodb_id

## What the model predicts
The model predicts a binary label called `delayed_30` for each complaint.

## How delayed is defined
`delayed_30 = 1` if `days_to_resolution` is greater than 30 days, otherwise `0`.

## Project structure
```
civic-complaint-triage/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_load_and_clean.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   └── 03_modeling.ipynb
├── reports/
│   └── figures/
├── models/
└── src/
    ├── download_data.py
    ├── clean_data.py
    ├── features.py
    ├── train_model.py
    └── evaluate_model.py
```

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Python version recommendation: 3.11+ (tested with 3.13).

## How to run
```powershell
python src/download_data.py
python src/clean_data.py
python src/features.py
python src/train_model.py --target delayed
python src/evaluate_model.py --target delayed
python src/train_model.py --target delayed_30
python src/evaluate_model.py --target delayed_30
python src/train_model.py --target delayed_top25
python src/evaluate_model.py --target delayed_top25
python src/target_analysis.py
python src/compare_targets.py
```

Fast target comparison (sample + smaller models):
```powershell
python src/train_model.py --target delayed --fast --sample-size 50000
python src/evaluate_model.py --target delayed
python src/train_model.py --target delayed_30 --fast --sample-size 50000
python src/evaluate_model.py --target delayed_30
python src/train_model.py --target delayed_top25 --fast --sample-size 50000
python src/evaluate_model.py --target delayed_top25
python src/target_analysis.py
python src/compare_targets.py
```

PowerShell shortcut:
```powershell
.\run_pipeline.ps1
```

## Methods
- Basic data cleaning and feature engineering
- Majority-class baseline (predicts the most common class only)
- Logistic regression baseline
- Random forest model
- Evaluation with accuracy, precision, recall, F1-score, ROC-AUC, confusion counts, and threshold analysis

## Results
- Full metrics saved to `reports/model_metrics_delayed_30.csv` (includes ROC-AUC, confusion counts, baseline comparison, and predicted vs actual delayed rate)
- Threshold sweep saved to `reports/threshold_analysis_delayed_30.csv`
- Plots saved in `reports/figures/`

Latest results (time-based split, 80/20) for the 30-day target:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Actual delayed rate | Predicted delayed rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| majority_baseline | 0.558 | 0.558 | 1.000 | 0.717 | N/A | 0.558 | 1.000 |
| logistic_regression | 0.601 | 0.590 | 0.930 | 0.722 | 0.654 | 0.558 | 0.879 |
| random_forest | 0.601 | 0.586 | 0.978 | 0.733 | 0.690 | 0.558 | 0.932 |

Random forest threshold summary (primary target: delayed_30):
- Default threshold (0.5): accuracy 0.601, precision 0.586, recall 0.978, F1 0.733, ROC-AUC 0.690
- Balanced-rate threshold (0.7): precision 0.685, recall 0.719, F1 0.702
- Actual delayed rate: 0.558; balanced-rate predicted delayed rate: 0.586

Plain-English interpretation:
- At the default threshold, the model is a high-recall screening tool that over-flags non-delayed complaints.
- At the balanced-rate threshold, the model provides a more realistic triage signal by aligning predicted and actual delayed rates.
- This remains a prototype decision-support signal, not an automated decision system.

## Target Definition Sensitivity
Because "delayed" is not explicitly labeled, the project tests multiple delay definitions:
- Median target: useful balanced experiment, but less interpretable than the 30-day target.
- 30-day target: best primary target because it is easy to explain and has the strongest overall results.
- Top-25% target: performed worse due to class imbalance and should remain a sensitivity test, not the primary model.

Run `python src/target_analysis.py` to compare delayed rates and `python src/compare_targets.py` to compare model metrics across targets.

Interpretation guidance:
- Baseline comparison matters: if ML does not beat the majority baseline on F1/ROC-AUC, it is not adding value.
- High recall can still be misleading when precision is low because many non-delayed complaints are flagged.
- Use the predicted vs actual delayed rate to check if the model overpredicts delays.
- Confusion counts (TP/FP/TN/FN) show tradeoffs between catching delayed cases and false alarms.

Threshold analysis:
- `reports/threshold_analysis_delayed_30.csv` evaluates thresholds from 0.1 to 0.9.
- The best-F1 threshold is reported per model in `reports/model_metrics_delayed_30.csv`.
- Best-F1 thresholds maximize F1 but can still overpredict delayed complaints.
- Balanced-rate thresholds are more conservative by matching predicted delayed rate to the actual delayed rate.
- Both are reported because triage systems involve tradeoffs.

Feature importance:
- Random forest feature importance saved to `reports/random_forest_feature_importance.csv`.
- Top features chart saved to `reports/figures/random_forest_top_features.png`.

## Geographic Error Analysis
This project includes a basic ZIP-level error analysis to check whether model errors vary across high-volume ZIP codes. It is not a fairness audit and should not be used to rank neighborhoods or allocate services automatically. Complaint data can reflect reporting behavior and service patterns, so geographic comparisons require caution. The goal is to see whether model error rates differ across the ZIP codes with the most complaints.

To check whether model behavior varied across high-volume ZIP codes, I compared actual and predicted delayed rates for the top 10 ZIP codes in the test set using the delayed_30 random forest model at the balanced threshold of 0.7.

The model’s predicted delayed rates were reasonably close to observed rates for several ZIP codes, but there were visible differences. For example, ZIP 19132 had an observed delayed rate of about 0.59 but a predicted delayed rate of about 0.73, while ZIP 19146 had an observed delayed rate of about 0.54 but a predicted delayed rate of about 0.38. This suggests that the model may overestimate delay risk in some areas and underestimate it in others.

This is a basic geographic error analysis, not a full fairness audit. Complaint data may reflect differences in reporting behavior, workload, property conditions, and city service patterns. These results should not be used to rank neighborhoods or allocate services automatically.

This model is a prototype triage signal, not an automated decision system.

## Scoring a New Complaint
Use the scoring script to estimate the probability that a new complaint remains unresolved beyond 30 days. The score is a triage signal, not an automated decision.

```powershell
python src/score_complaint.py --input examples/sample_complaint.json
```

The script prints a probability, default and balanced threshold predictions, and a risk band (low, moderate, high). This should not be used to rank neighborhoods or allocate services automatically.

## How to interpret the results
- Accuracy: overall fraction of correct predictions, which can be misleading with class imbalance.
- Precision: when the model predicts delayed, how often it is correct. Low precision means many false alarms.
- Recall: how many truly delayed complaints the model catches. High recall can still be misleading if precision is low.
- F1: balances precision and recall to summarize tradeoffs.
- ROC-AUC: ability to rank delayed vs not delayed across thresholds.
- Confusion matrix: shows true/false positives and negatives to explain errors.
- Threshold tuning: adjusts the probability cutoff to trade recall for precision (and vice versa).

## Reproducibility notes
- Python version: 3.11+ recommended (tested with 3.13)
- Setup: see `Setup` section above
- Full run:
    - `python src/download_data.py`
    - `python src/clean_data.py`
    - `python src/features.py`
    - `python src/train_model.py`
    - `python src/evaluate_model.py`
- Expected outputs:
    - `data/raw/complaints.csv`
    - `data/processed/complaints_cleaned.csv`
    - `data/processed/complaints_features.csv`
    - `models/logistic_regression.pkl`
    - `models/random_forest.pkl`
    - `reports/model_metrics_delayed.csv`
    - `reports/model_metrics_delayed.csv`
    - `reports/model_metrics_delayed_30.csv`
    - `reports/model_metrics_delayed_top25.csv`
    - `reports/target_definition_analysis.csv`
    - `reports/target_model_comparison.csv`
    - `reports/threshold_analysis.csv`
    - `reports/figures/*`
    - `reports/random_forest_feature_importance.csv`
    - `reports/selected_features.txt`
- Troubleshooting: if models do not exist, run `train_model.py` before `evaluate_model.py`.

## Limitations
- This is a decision-support prototype.
- Public complaint data may have missing or inconsistent records.
- Complaint volume may reflect reporting behavior, not only actual need.
- The model should not be used for automated enforcement or service denial.

## Future improvements
- Add fairness checks by region
- Add richer geospatial analysis
- Test a 30-day delay threshold
- Add a simple Streamlit dashboard
- Use more advanced models

