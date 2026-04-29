from __future__ import annotations

from pathlib import Path

import pandas as pd


def analyze_targets() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    cleaned_path = project_root / "data" / "processed" / "complaints_cleaned.csv"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at {cleaned_path}. Run clean_data.py first."
        )

    df = pd.read_csv(cleaned_path, low_memory=False)
    if "days_to_resolution" not in df.columns:
        raise ValueError("days_to_resolution column not found in cleaned data.")

    median_threshold = df["days_to_resolution"].median()
    top25_threshold = df["days_to_resolution"].quantile(0.75)

    targets = [
        ("delayed", "median", median_threshold),
        ("delayed_30", "30_days", 30),
        ("delayed_top25", "top25", top25_threshold),
    ]

    rows = []
    for target_col, label, threshold in targets:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")
        delayed_count = int(df[target_col].sum())
        non_delayed_count = int((df[target_col] == 0).sum())
        delayed_rate = float(df[target_col].mean())
        rows.append(
            {
                "target": target_col,
                "definition": label,
                "threshold": float(threshold),
                "delayed_count": delayed_count,
                "non_delayed_count": non_delayed_count,
                "delayed_rate": delayed_rate,
            }
        )

        print(
            f"Target: {target_col} | threshold={threshold:.2f} | "
            f"delayed={delayed_count} | non_delayed={non_delayed_count} | "
            f"rate={delayed_rate:.3f}"
        )

    analysis_df = pd.DataFrame(rows)
    analysis_df.to_csv(reports_dir / "target_definition_analysis.csv", index=False)
    print("Saved target definition analysis to reports/target_definition_analysis.csv")

    return analysis_df


def main() -> None:
    analyze_targets()


if __name__ == "__main__":
    main()
