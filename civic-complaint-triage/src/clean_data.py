from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _find_column(columns: list[str], keywords: list[str]) -> str | None:
    for key in keywords:
        for col in columns:
            if key in col:
                return col
    return None


def _infer_date_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    original_cols = list(df.columns)
    lower_cols = [col.lower() for col in original_cols]

    opened_keywords = [
        "opened",
        "created",
        "received",
        "requested",
        "submitted",
        "start",
        "request",
    ]
    closed_keywords = [
        "closed",
        "resolved",
        "completed",
        "final",
        "resolution",
        "finished",
    ]

    opened_col_lower = _find_column(lower_cols, opened_keywords)
    closed_col_lower = _find_column(lower_cols, closed_keywords)

    if opened_col_lower:
        opened_col = original_cols[lower_cols.index(opened_col_lower)]
    else:
        opened_col = None

    if closed_col_lower:
        closed_col = original_cols[lower_cols.index(closed_col_lower)]
    else:
        closed_col = None

    if not opened_col or not closed_col:
        date_like = [
            original_cols[i]
            for i, col in enumerate(lower_cols)
            if "date" in col or "time" in col
        ]
        if not opened_col and len(date_like) > 0:
            opened_col = date_like[0]
        if not closed_col and len(date_like) > 1:
            closed_col = date_like[1]

    return opened_col, closed_col


def clean_data() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    raw_path = project_root / "data" / "raw" / "complaints.csv"
    processed_path = project_root / "data" / "processed" / "complaints_cleaned.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. Run download_data.py first."
        )

    df = pd.read_csv(raw_path, low_memory=False)
    rows_before = len(df)

    opened_col, closed_col = _infer_date_columns(df)
    if not opened_col or not closed_col:
        raise ValueError(
            "Could not identify opened and closed date columns. "
            "Please inspect the raw data columns."
        )

    print(f"Opened date column: {opened_col}")
    print(f"Closed date column: {closed_col}")

    df["opened_date"] = pd.to_datetime(df[opened_col], errors="coerce")
    df["closed_date"] = pd.to_datetime(df[closed_col], errors="coerce")

    df["days_to_resolution"] = (df["closed_date"] - df["opened_date"]).dt.days
    df = df[df["days_to_resolution"].notna()]
    df = df[df["days_to_resolution"] >= 0]

    median_days = df["days_to_resolution"].median()
    top25_threshold = df["days_to_resolution"].quantile(0.75)
    df["delayed"] = (df["days_to_resolution"] > median_days).astype(int)
    df["delayed_30"] = (df["days_to_resolution"] > 30).astype(int)
    df["delayed_top25"] = (df["days_to_resolution"] >= top25_threshold).astype(int)

    df["submitted_month"] = df["opened_date"].dt.month
    df["submitted_dayofweek"] = df["opened_date"].dt.dayofweek
    df["is_weekend"] = df["submitted_dayofweek"].isin([5, 6]).astype(int)

    lower_cols = [col.lower() for col in df.columns]
    original_cols = list(df.columns)

    complaint_type_col = _find_column(
        lower_cols,
        ["complaint_type", "complaint", "type", "category", "subject", "issue"],
    )
    complaint_source_col = _find_column(
        lower_cols,
        ["source", "channel", "method", "intake", "submitted_via"],
    )
    zip_col = _find_column(lower_cols, ["zip", "zipcode", "postal"])
    lat_col = _find_column(lower_cols, ["lat", "latitude"])
    lng_col = _find_column(lower_cols, ["lng", "longitude", "lon", "long"])

    if complaint_type_col:
        original = original_cols[lower_cols.index(complaint_type_col)]
        if "complaint_type" not in df.columns:
            df["complaint_type"] = df[original]
    if complaint_source_col:
        original = original_cols[lower_cols.index(complaint_source_col)]
        if "complaint_source" not in df.columns:
            df["complaint_source"] = df[original]
    if zip_col:
        original = original_cols[lower_cols.index(zip_col)]
        if "zip_code" not in df.columns:
            df["zip_code"] = (
                df[original].astype(str).str.extract(r"(\d{5})")[0]
            )
    if lat_col:
        original = original_cols[lower_cols.index(lat_col)]
        if "lat" not in df.columns:
            df["lat"] = df[original]
    if lng_col:
        original = original_cols[lower_cols.index(lng_col)]
        if "lng" not in df.columns:
            df["lng"] = df[original]

    rows_after = len(df)
    delayed_rate = df["delayed"].mean()

    df.to_csv(processed_path, index=False)

    print(f"Rows before cleaning: {rows_before}")
    print(f"Rows after cleaning: {rows_after}")
    print(f"Median days to resolution: {median_days:.2f}")
    print(f"Delayed class balance: {delayed_rate:.3f}")
    print(f"Saved cleaned data to {processed_path}")

    return df


def main() -> None:
    clean_data()


if __name__ == "__main__":
    main()
