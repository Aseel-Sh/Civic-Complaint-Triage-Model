# Civic Complaint Triage Model

## Problem statement
Create a beginner-friendly model that predicts whether a Philadelphia Licenses and Inspections complaint is likely to have a delayed resolution.

## Why this matters
City agencies receive large volumes of complaints. A simple triage signal can help prioritize follow-up and improve response planning.

## Dataset
Public Philadelphia L&I complaint data (CSV):
https://phl.carto.com/api/v2/sql?q=SELECT+*,+ST_Y(the_geom)+AS+lat,+ST_X(the_geom)+AS+lng+FROM+complaints&filename=complaints&format=csv&skipfields=cartodb_id

## What the model predicts
The model predicts a binary label called `delayed` for each complaint.

## How delayed is defined
`delayed = 1` if `days_to_resolution` is greater than the median resolution time in the cleaned dataset, otherwise `0`.

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

## How to run
```powershell
python src/download_data.py
python src/clean_data.py
python src/features.py
python src/train_model.py
python src/evaluate_model.py
```

## Methods
- Basic data cleaning and feature engineering
- Logistic regression baseline
- Random forest model
- Evaluation with accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices

## Results
- Metrics will be saved to `reports/model_metrics.csv`
- Plots will be saved in `reports/figures/`

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

## Resume bullets
**Civic Complaint Triage Model | Python, pandas, scikit-learn, Matplotlib**
- Built a machine learning pipeline using public Philadelphia inspection complaint data to predict delayed complaint resolution from complaint type, location, and submission timing.
- Engineered temporal, geographic, and workload features, compared logistic regression and random forest models, and evaluated performance using precision, recall, F1-score, ROC-AUC, and feature importance.
