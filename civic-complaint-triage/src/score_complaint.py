from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_model import load_feature_data, select_feature_columns


def _normalize_zip(value) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return str(value)

    text = str(value).strip()
    if text == "":
        return None
    if text.isdigit():
        return text

    try:
        as_float = float(text)
    except ValueError:
        return text

    if float(as_float).is_integer():
        return str(int(as_float))
    return text


def _load_input(path: Path | None) -> dict:
    if path is None:
        return {
            "complaint_type": "VACANT LOTS",
            "complaint_source": "311",
            "zip_code": "19134",
            "submitted_month": 6,
            "submitted_dayofweek": 2,
            "is_weekend": 0,
            "lat": 39.9922,
            "lng": -75.0896,
        }

    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found at {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _default_value(series: pd.Series):
    if series.dropna().empty:
        return 0

    if pd.api.types.is_bool_dtype(series):
        return int(series.mode(dropna=True).iloc[0])

    if pd.api.types.is_numeric_dtype(series):
        return float(series.median(skipna=True))

    mode = series.dropna().mode()
    if not mode.empty:
        return mode.iloc[0]
    return "unknown"


def _build_defaults(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    defaults = {}
    for col in feature_cols:
        defaults[col] = _default_value(df[col]) if col in df.columns else 0
    return defaults


def _build_aggregate_maps(df: pd.DataFrame) -> dict:
    df = df.copy()
    if "zip_code" in df.columns:
        df["zip_code_clean"] = df["zip_code"].apply(_normalize_zip)
    else:
        df["zip_code_clean"] = None

    maps = {
        "complaint_type_total_count": None,
        "zip_total_complaints": None,
        "zip_type_complaint_count": None,
    }

    if "complaint_type" in df.columns and "complaint_type_total_count" in df.columns:
        maps["complaint_type_total_count"] = (
            df.groupby("complaint_type")["complaint_type_total_count"].median()
        )

    if "zip_code_clean" in df.columns and "zip_total_complaints" in df.columns:
        maps["zip_total_complaints"] = (
            df.groupby("zip_code_clean")["zip_total_complaints"].median()
        )

    if (
        "zip_code_clean" in df.columns
        and "complaint_type" in df.columns
        and "zip_type_complaint_count" in df.columns
    ):
        maps["zip_type_complaint_count"] = (
            df.groupby(["zip_code_clean", "complaint_type"])[
                "zip_type_complaint_count"
            ].median()
        )

    return maps


def _apply_input(
    row: dict, user_input: dict, feature_cols: list[str]
) -> dict:
    cleaned = {}
    for key, value in user_input.items():
        if key not in feature_cols:
            continue
        if key == "zip_code":
            cleaned[key] = _normalize_zip(value)
        else:
            cleaned[key] = value

    row.update(cleaned)
    return row


def _fill_aggregate_features(
    row: dict,
    defaults: dict,
    maps: dict,
    warnings: list[str],
) -> None:
    complaint_type = row.get("complaint_type")
    zip_code = _normalize_zip(row.get("zip_code"))

    if "complaint_type_total_count" in row:
        lookup = maps.get("complaint_type_total_count")
        if complaint_type and lookup is not None and complaint_type in lookup:
            row["complaint_type_total_count"] = float(lookup[complaint_type])
        else:
            row["complaint_type_total_count"] = defaults.get(
                "complaint_type_total_count", 0
            )
            warnings.append("complaint_type_total_count fallback used")

    if "zip_total_complaints" in row:
        lookup = maps.get("zip_total_complaints")
        if zip_code and lookup is not None and zip_code in lookup:
            row["zip_total_complaints"] = float(lookup[zip_code])
        else:
            row["zip_total_complaints"] = defaults.get("zip_total_complaints", 0)
            warnings.append("zip_total_complaints fallback used")

    if "zip_type_complaint_count" in row:
        lookup = maps.get("zip_type_complaint_count")
        key = (zip_code, complaint_type)
        if zip_code and complaint_type and lookup is not None and key in lookup:
            row["zip_type_complaint_count"] = float(lookup[key])
        else:
            row["zip_type_complaint_count"] = defaults.get(
                "zip_type_complaint_count", 0
            )
            warnings.append("zip_type_complaint_count fallback used")


def _risk_band(probability: float) -> str:
    if probability < 0.4:
        return "Low"
    if probability < 0.7:
        return "Moderate"
    return "High"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a complaint for 30-day delay risk.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a JSON file with complaint fields.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "random_forest_delayed_30.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train delayed_30 first."
        )

    df = load_feature_data()
    feature_cols = select_feature_columns(df, "delayed_30")

    defaults = _build_defaults(df, feature_cols)
    maps = _build_aggregate_maps(df)

    user_input = _load_input(Path(args.input)) if args.input else _load_input(None)
    row = defaults.copy()
    row = _apply_input(row, user_input, feature_cols)

    warnings = []
    _fill_aggregate_features(row, defaults, maps, warnings)

    feature_row = pd.DataFrame([row], columns=feature_cols)

    model = joblib.load(model_path)
    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model does not support predict_proba.")

    probability = float(model.predict_proba(feature_row)[:, 1][0])
    default_pred = "Delayed" if probability >= 0.5 else "Not delayed"
    balanced_pred = "Delayed" if probability >= 0.7 else "Not delayed"
    band = _risk_band(probability)

    print(f"30-Day Delay Risk Score: {probability:.2f}")
    print(f"Default threshold prediction: {default_pred}")
    print(f"Balanced threshold prediction: {balanced_pred}")
    print(f"Risk band: {band}")
    print(
        "Interpretation: This complaint has {band} delay risk. "
        "It should be treated as a triage signal, not an automated decision."
        .format(band=band.lower())
    )

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")

    output = {
        "input": user_input,
        "risk_score": round(probability, 4),
        "default_threshold": 0.5,
        "balanced_threshold": 0.7,
        "default_prediction": default_pred,
        "balanced_prediction": balanced_pred,
        "risk_band": band,
        "interpretation": (
            f"This complaint has {band.lower()} delay risk. "
            "It should be treated as a triage signal, not an automated decision."
        ),
    }

    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "scored_sample_complaint.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(f"\nSaved scored output to {output_path}")


if __name__ == "__main__":
    main()
