from __future__ import annotations

from pathlib import Path

import pandas as pd


def compare_targets() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    reports_dir = project_root / "reports"
    targets = ["delayed", "delayed_30", "delayed_top25"]

    rows = []
    for target in targets:
        metrics_path = reports_dir / f"model_metrics_{target}.csv"
        if not metrics_path.exists():
            print(f"Missing metrics file: {metrics_path}")
            continue

        metrics_df = pd.read_csv(metrics_path, low_memory=False)
        for _, row in metrics_df.iterrows():
            rows.append(
                {
                    "target": target,
                    "delayed_rate": row.get("actual_delayed_rate"),
                    "model": row.get("model"),
                    "accuracy": row.get("accuracy"),
                    "precision": row.get("precision"),
                    "recall": row.get("recall"),
                    "f1": row.get("f1"),
                    "roc_auc": row.get("roc_auc"),
                    "actual_delayed_rate": row.get("actual_delayed_rate"),
                    "predicted_delayed_rate": row.get("predicted_delayed_rate"),
                }
            )

    comparison_df = pd.DataFrame(rows)
    output_path = reports_dir / "target_model_comparison.csv"
    comparison_df.to_csv(output_path, index=False)
    print(f"Saved target comparison to {output_path}")

    return comparison_df


def main() -> None:
    compare_targets()


if __name__ == "__main__":
    main()
