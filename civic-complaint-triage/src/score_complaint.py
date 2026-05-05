from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_model import SAFE_FEATURES, load_feature_data, get_feature_columns


INPUT_FIELDS = [
    "complaint_type",
    "complaint_source",
    "zip_code",
    "council_district",
    "lat",
    "lng",
    "submitted_month",
    "submitted_dayofweek",
    "is_weekend",
]


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


def _normalize_complaint_type(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    return text.upper()


def _normalize_complaint_source(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    return text.upper()


def _load_input(path: Path | None) -> dict:
    if path is None:
        return {
            "complaint_type": "VACANT LOTS (CLIP)",
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


def _validate_input_keys(user_input: dict) -> None:
    extras = sorted({key for key in user_input.keys()} - set(INPUT_FIELDS))
    if extras:
        raise ValueError(
            "Input contains unsupported fields. Remove these keys: "
            + ", ".join(extras)
        )


def _validate_input_columns(input_df: pd.DataFrame) -> None:
    extras = sorted(set(input_df.columns) - set(INPUT_FIELDS))
    if extras:
        raise ValueError(
            "Batch input contains unsupported fields. Remove these columns: "
            + ", ".join(extras)
        )

    if len(input_df.columns) == 0:
        raise ValueError(
            "Batch input has no supported fields. Provide scoring-safe columns."
        )


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

    if "complaint_type" in df.columns:
        df["complaint_type_clean"] = df["complaint_type"].apply(
            _normalize_complaint_type
        )
    else:
        df["complaint_type_clean"] = None

    maps = {
        "complaint_type_total_count": None,
        "zip_total_complaints": None,
        "zip_type_complaint_count": None,
    }

    if (
        "complaint_type_clean" in df.columns
        and "complaint_type_total_count" in df.columns
    ):
        grouped = df.groupby("complaint_type_clean")["complaint_type_total_count"]
        if pd.api.types.is_numeric_dtype(df["complaint_type_total_count"]):
            maps["complaint_type_total_count"] = grouped.median()
        else:
            maps["complaint_type_total_count"] = grouped.apply(
                lambda s: s.dropna().iloc[0] if not s.dropna().empty else np.nan
            )

    if "zip_code_clean" in df.columns and "zip_total_complaints" in df.columns:
        grouped = df.groupby("zip_code_clean")["zip_total_complaints"]
        if pd.api.types.is_numeric_dtype(df["zip_total_complaints"]):
            maps["zip_total_complaints"] = grouped.median()
        else:
            maps["zip_total_complaints"] = grouped.apply(
                lambda s: s.dropna().iloc[0] if not s.dropna().empty else np.nan
            )

    if (
        "zip_code_clean" in df.columns
        and "complaint_type_clean" in df.columns
        and "zip_type_complaint_count" in df.columns
    ):
        grouped = df.groupby(
            ["zip_code_clean", "complaint_type_clean"]
        )["zip_type_complaint_count"]
        if pd.api.types.is_numeric_dtype(df["zip_type_complaint_count"]):
            maps["zip_type_complaint_count"] = grouped.median()
        else:
            maps["zip_type_complaint_count"] = grouped.apply(
                lambda s: s.dropna().iloc[0] if not s.dropna().empty else np.nan
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
        elif key == "complaint_type":
            cleaned[key] = _normalize_complaint_type(value)
        elif key == "complaint_source":
            cleaned[key] = _normalize_complaint_source(value)
        else:
            cleaned[key] = value

    row.update(cleaned)
    return row


def _suggest_complaint_types(lookup, complaint_type: str) -> list[str]:
    if lookup is None:
        return []
    if not complaint_type:
        return []
    first_word = complaint_type.split()[0]
    suggestions = [
        value for value in lookup.index if first_word in str(value).split()
    ]
    return suggestions[:5]


def _fill_aggregate_features(
    row: dict,
    defaults: dict,
    maps: dict,
    warnings: list[str],
) -> None:
    complaint_type = _normalize_complaint_type(row.get("complaint_type"))
    zip_code = _normalize_zip(row.get("zip_code"))

    if "complaint_type_total_count" in row:
        lookup = maps.get("complaint_type_total_count")
        if complaint_type and lookup is not None and complaint_type in lookup.index:
            row["complaint_type_total_count"] = float(lookup[complaint_type])
            print("Looked up complaint_type_total_count from historical data")
        else:
            row["complaint_type_total_count"] = defaults.get(
                "complaint_type_total_count", 0
            )
            suggestions = _suggest_complaint_types(lookup, complaint_type)
            if suggestions:
                warnings.append(
                    "complaint_type_total_count fallback used; "
                    "closest complaint_type examples: "
                    f"{', '.join(suggestions)}"
                )
            else:
                warnings.append(
                    "complaint_type_total_count fallback used; "
                    "no historical match for complaint_type"
                )

    if "zip_total_complaints" in row:
        lookup = maps.get("zip_total_complaints")
        if zip_code and lookup is not None and zip_code in lookup.index:
            row["zip_total_complaints"] = float(lookup[zip_code])
            print("Looked up zip_total_complaints from historical data")
        else:
            row["zip_total_complaints"] = defaults.get("zip_total_complaints", 0)
            warnings.append(
                "zip_total_complaints fallback used; no historical match for zip_code"
            )

    if "zip_type_complaint_count" in row:
        lookup = maps.get("zip_type_complaint_count")
        key = (zip_code, complaint_type)
        if (
            zip_code
            and complaint_type
            and lookup is not None
            and key in lookup.index
        ):
            row["zip_type_complaint_count"] = float(lookup[key])
            print("Looked up zip_type_complaint_count from historical data")
        else:
            row["zip_type_complaint_count"] = defaults.get(
                "zip_type_complaint_count", 0
            )
            suggestions = _suggest_complaint_types(
                maps.get("complaint_type_total_count"), complaint_type
            )
            if suggestions:
                warnings.append(
                    "zip_type_complaint_count fallback used; "
                    "closest complaint_type examples: "
                    f"{', '.join(suggestions)}"
                )
            else:
                warnings.append(
                    "zip_type_complaint_count fallback used; "
                    "no historical match for zip_code + complaint_type"
                )


def _risk_band(probability: float) -> str:
    if probability < 0.4:
        return "Low"
    if probability < 0.7:
        return "Moderate"
    return "High"


def _score_row(
    model,
    user_input: dict,
    feature_cols: list[str],
    defaults: dict,
    maps: dict,
    *,
    balanced_threshold: float,
) -> tuple[float, str, str, str, str, list[str]]:
    row = defaults.copy()
    row = _apply_input(row, user_input, feature_cols)

    warnings = []
    _fill_aggregate_features(row, defaults, maps, warnings)

    feature_row = pd.DataFrame([row], columns=feature_cols)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model does not support predict_proba.")

    probability = float(model.predict_proba(feature_row)[:, 1][0])
    default_pred = "Delayed" if probability >= 0.5 else "Not delayed"
    balanced_pred = (
        "Delayed" if probability >= balanced_threshold else "Not delayed"
    )
    band = _risk_band(probability)
    interpretation = (
        f"This complaint has {band.lower()} delay risk. "
        "It should be treated as a triage signal, not an automated decision."
    )

    return (
        probability,
        default_pred,
        balanced_pred,
        band,
        interpretation,
        warnings,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a complaint for 30-day delay risk.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a JSON file with complaint fields.",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Path to a CSV file with complaint fields (batch scoring).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save batch scoring output CSV.",
    )
    args = parser.parse_args()

    if args.batch and args.input:
        raise ValueError("Use either --input or --batch, not both.")
    if args.batch and not args.output:
        raise ValueError("--output is required when using --batch.")

    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "random_forest_delayed_30.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train delayed_30 first."
        )

    df = load_feature_data()
    feature_cols = get_feature_columns(df, "delayed_30", project_root / "reports")

    defaults = _build_defaults(df, feature_cols)
    maps = _build_aggregate_maps(df)

    model = joblib.load(model_path)

    if args.batch:
        batch_path = Path(args.batch)
        if not batch_path.exists():
            raise FileNotFoundError(f"Batch CSV not found at {batch_path}")
        batch_df = pd.read_csv(batch_path)
        _validate_input_columns(batch_df)

        output_rows = []
        for index, record in batch_df.iterrows():
            user_input = record.to_dict()
            _validate_input_keys(user_input)
            (
                probability,
                default_pred,
                balanced_pred,
                band,
                interpretation,
                warnings,
            ) = _score_row(
                model,
                user_input,
                feature_cols,
                defaults,
                maps,
                balanced_threshold=0.7,
            )

            if warnings:
                print(f"\nWarnings for row {index}:")
                for warning in warnings:
                    print(f"- {warning}")

            output_row = {col: user_input.get(col) for col in batch_df.columns}
            output_row.update(
                {
                    "delay_probability": round(probability, 6),
                    "default_prediction": default_pred,
                    "balanced_prediction": balanced_pred,
                    "risk_band": band,
                    "interpretation": interpretation,
                }
            )
            output_rows.append(output_row)

        output_df = pd.DataFrame(output_rows)
        output_df = output_df.sort_values(
            "delay_probability", ascending=False
        )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        print(f"Saved batch scoring output to {output_path}")
        return

    user_input = _load_input(Path(args.input)) if args.input else _load_input(None)
    _validate_input_keys(user_input)
    (
        probability,
        default_pred,
        balanced_pred,
        band,
        interpretation,
        warnings,
    ) = _score_row(
        model,
        user_input,
        feature_cols,
        defaults,
        maps,
        balanced_threshold=0.7,
    )

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
        "interpretation": interpretation,
    }

    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "scored_sample_complaint.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(f"\nSaved scored output to {output_path}")


if __name__ == "__main__":
    main()
