from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from train_model import load_feature_data, select_feature_columns, split_data


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _zip_metrics(group: pd.DataFrame) -> dict:
    y_true = group["y_true"].astype(int)
    y_pred = group["y_pred"].astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    actual_rate = float(y_true.mean()) if len(y_true) else 0.0
    predicted_rate = float(y_pred.mean()) if len(y_pred) else 0.0

    return {
        "complaint_count": int(len(group)),
        "actual_delayed_rate": actual_rate,
        "predicted_delayed_rate": predicted_rate,
        "false_positive_rate": _safe_rate(fp, fp + tn),
        "false_negative_rate": _safe_rate(fn, fn + tp),
        "precision": _safe_rate(tp, tp + fp),
        "recall": _safe_rate(tp, tp + fn),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    target_column = "delayed_30"
    threshold = 0.7

    df = load_feature_data()
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in data. "
            "Run clean_data.py and features.py first."
        )

    if "zip_code" not in df.columns:
        print(
            "zip_code column not found. Geographic error analysis cannot run."
        )
        return

    # Reuse the same feature selection and time-based split as evaluation.
    feature_cols = select_feature_columns(df, target_column)
    X_train, X_test, y_train, y_test, split_method = split_data(
        df, feature_cols, target_column
    )

    print("Zip error analysis for delayed_30")
    print(f"Split method: {split_method}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Balanced threshold: {threshold}")

    models_dir = project_root / "models"
    model_path = models_dir / "random_forest_delayed_30.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train delayed_30 first."
        )

    model = joblib.load(model_path)

    # Use the balanced-rate threshold to build conservative predictions.
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    results = X_test[["zip_code"]].copy()
    results["y_true"] = y_test.to_numpy()
    results["y_pred"] = preds

    # Focus on the top 10 ZIP codes by volume in the test set.
    top_zips = (
        results["zip_code"]
        .dropna()
        .astype(str)
        .value_counts()
        .head(10)
        .index
    )

    if len(top_zips) == 0:
        print("No ZIP codes available after filtering missing values.")
        return

    filtered = results[results["zip_code"].astype(str).isin(top_zips)]

    summary_rows = []
    for zip_code, group in filtered.groupby("zip_code"):
        metrics = _zip_metrics(group)
        metrics["zip_code"] = str(zip_code)
        summary_rows.append(metrics)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        "complaint_count", ascending=False
    )

    output_path = reports_dir / "zip_error_analysis_delayed_30.csv"
    summary_df.to_csv(output_path, index=False)

    print("\nTop ZIP codes by complaint volume (test set):")
    for _, row in summary_df.iterrows():
        print(
            "ZIP {zip_code}: count={count}, actual={actual:.3f}, "
            "predicted={predicted:.3f}, precision={precision:.3f}, "
            "recall={recall:.3f}".format(
                zip_code=row["zip_code"],
                count=int(row["complaint_count"]),
                actual=row["actual_delayed_rate"],
                predicted=row["predicted_delayed_rate"],
                precision=row["precision"],
                recall=row["recall"],
            )
        )

    print(f"\nSaved ZIP error analysis to {output_path}")

    chart_df = summary_df.sort_values("complaint_count", ascending=False)
    x = np.arange(len(chart_df))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width / 2,
        chart_df["actual_delayed_rate"],
        width,
        label="Actual rate",
        color="#4C78A8",
    )
    ax.bar(
        x + width / 2,
        chart_df["predicted_delayed_rate"],
        width,
        label="Predicted rate",
        color="#F58518",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(chart_df["zip_code"].astype(str), rotation=45, ha="right")
    ax.set_ylabel("Delayed rate")
    ax.set_xlabel("ZIP code")
    ax.set_title("Actual vs Predicted Delayed Rate by ZIP (Top 10)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    chart_path = figures_dir / "zip_actual_vs_predicted_delay_rate.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    print(f"Saved chart to {chart_path}")


if __name__ == "__main__":
    main()
