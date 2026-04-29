from __future__ import annotations

from pathlib import Path

import pandas as pd


def add_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    if "complaint_type" in df.columns:
        df["complaint_type_total_count"] = df.groupby("complaint_type")[
            "complaint_type"
        ].transform("count")

    if "zip_code" in df.columns:
        df["zip_total_complaints"] = df.groupby("zip_code")["zip_code"].transform(
            "count"
        )

    if "zip_code" in df.columns and "complaint_type" in df.columns:
        df["zip_type_complaint_count"] = df.groupby(["zip_code", "complaint_type"])[
            "complaint_type"
        ].transform("count")

    return df


def build_features() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    cleaned_path = project_root / "data" / "processed" / "complaints_cleaned.csv"
    features_path = project_root / "data" / "processed" / "complaints_features.csv"
    features_path.parent.mkdir(parents=True, exist_ok=True)

    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at {cleaned_path}. Run clean_data.py first."
        )

    df = pd.read_csv(cleaned_path, low_memory=False)
    df = add_aggregate_features(df)
    df.to_csv(features_path, index=False)

    print(f"Saved feature data to {features_path}")
    print(f"Feature dataset shape: {df.shape}")

    return df


def main() -> None:
    build_features()


if __name__ == "__main__":
    main()
