from __future__ import annotations

from pathlib import Path

import argparse

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from train_model import load_feature_data, select_feature_columns, split_data


def _confusion_counts(y_true: pd.Series, y_pred: pd.Series) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _evaluate_predictions(
    y_true: pd.Series, y_pred: pd.Series, *, roc_auc: float | None
) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "actual_delayed_rate": float(y_true.mean()),
        "predicted_delayed_rate": float(y_pred.mean()),
    }
    metrics.update(_confusion_counts(y_true, y_pred))
    return metrics


def _print_interpretation(metrics: dict) -> None:
    if metrics["predicted_delayed_rate"] > metrics["actual_delayed_rate"]:
        print(
            "Interpretation: The model catches many delayed complaints, "
            "but it also flags many non-delayed complaints as delayed."
        )
    else:
        print(
            "Interpretation: The model is conservative and may miss some delayed "
            "complaints to avoid false alarms."
        )


def _evaluate_thresholds(
    y_true: pd.Series,
    y_score: pd.Series,
    model_name: str,
    target_column: str,
) -> tuple[pd.DataFrame, dict, dict]:
    # Threshold controls the cutoff for predicting delayed; lower means more positives.
    rows = []
    best = {
        "best_f1_threshold": None,
        "best_f1": None,
        "best_f1_precision": None,
        "best_f1_recall": None,
        "best_f1_predicted_delayed_rate": None,
    }
    balanced = {
        "balanced_rate_threshold": None,
        "balanced_rate_precision": None,
        "balanced_rate_recall": None,
        "balanced_rate_f1": None,
        "balanced_rate_predicted_delayed_rate": None,
    }
    actual_rate = float(y_true.mean())
    closest_diff = None

    for threshold in [round(x, 1) for x in list(np.arange(0.1, 1.0, 0.1))]:
        preds = (y_score >= threshold).astype(int)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        predicted_rate = float(preds.mean())
        rate_diff = abs(predicted_rate - actual_rate)
        rows.append(
            {
                "target": target_column,
                "model": model_name,
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted_delayed_rate": predicted_rate,
                "actual_delayed_rate": actual_rate,
                "difference_from_actual_rate": rate_diff,
            }
        )

        if best["best_f1"] is None or f1 > best["best_f1"]:
            best.update(
                {
                    "best_f1_threshold": threshold,
                    "best_f1": f1,
                    "best_f1_precision": precision,
                    "best_f1_recall": recall,
                    "best_f1_predicted_delayed_rate": predicted_rate,
                }
            )

        if closest_diff is None or rate_diff < closest_diff:
            closest_diff = rate_diff
            balanced.update(
                {
                    "balanced_rate_threshold": threshold,
                    "balanced_rate_precision": precision,
                    "balanced_rate_recall": recall,
                    "balanced_rate_f1": f1,
                    "balanced_rate_predicted_delayed_rate": predicted_rate,
                }
            )

    return pd.DataFrame(rows), best, balanced


def _plot_thresholds(
    threshold_df: pd.DataFrame, figures_dir: Path
) -> None:
    models = threshold_df["model"].unique().tolist()
    fig, axes = plt.subplots(1, len(models), figsize=(12, 4), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        data = threshold_df[threshold_df["model"] == model_name]
        ax.plot(data["threshold"], data["precision"], label="Precision")
        ax.plot(data["threshold"], data["recall"], label="Recall")
        ax.plot(data["threshold"], data["f1"], label="F1")
        ax.set_title(model_name.replace("_", " ").title())
        ax.set_xlabel("Threshold")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(figures_dir / "threshold_precision_recall_f1.png", dpi=150)
    plt.close(fig)


def _plot_predicted_vs_actual_rate(
    threshold_df: pd.DataFrame, figures_dir: Path, target_column: str
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for model_name in threshold_df["model"].unique():
        data = threshold_df[threshold_df["model"] == model_name]
        ax.plot(
            data["threshold"],
            data["predicted_delayed_rate"],
            label=model_name.replace("_", " ").title(),
        )

    actual_rate = threshold_df["actual_delayed_rate"].iloc[0]
    ax.axhline(actual_rate, color="#444444", linestyle="--", label="Actual rate")
    ax.set_title("Predicted vs Actual Delayed Rate")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Predicted delayed rate")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figures_dir / f"threshold_predicted_vs_actual_rate_{target_column}.png",
        dpi=150,
    )
    plt.close(fig)


def _plot_confusion_matrix(cm, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not delayed", "Delayed"])
    ax.set_yticklabels(["Not delayed", "Delayed"])

    for (i, j), value in zip(
        [(0, 0), (0, 1), (1, 0), (1, 1)], cm.flatten()
    ):
        ax.text(j, i, str(value), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _suffix_for_target(target_column: str) -> str:
    return "" if target_column == "delayed" else f"_{target_column}"


def evaluate_models(target_column: str) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    suffix = _suffix_for_target(target_column)

    df = load_feature_data()
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")
    feature_cols = select_feature_columns(df, target_column)
    X_train, X_test, y_train, y_test, split_method = split_data(
        df, feature_cols, target_column
    )

    print(f"Evaluation split method: {split_method}")
    print(f"Testing rows: {len(X_test)}")

    models = {
        "logistic_regression": joblib.load(
            models_dir / f"logistic_regression_{target_column}.pkl"
        ),
        "random_forest": joblib.load(
            models_dir / f"random_forest_{target_column}.pkl"
        ),
    }

    metrics_rows = []
    threshold_rows = []

    majority_class = int(y_test.value_counts().idxmax())
    baseline_preds = pd.Series([majority_class] * len(y_test), index=y_test.index)
    baseline_metrics = _evaluate_predictions(
        y_test, baseline_preds, roc_auc=None
    )
    baseline_metrics.update(
        {
            "model": "majority_baseline",
            "best_f1_threshold": None,
            "best_f1": None,
            "best_f1_precision": None,
            "best_f1_recall": None,
            "best_f1_predicted_delayed_rate": None,
            "balanced_rate_threshold": None,
            "balanced_rate_precision": None,
            "balanced_rate_recall": None,
            "balanced_rate_f1": None,
            "balanced_rate_predicted_delayed_rate": None,
        }
    )
    print("\nModel: majority_baseline")
    print(
        "Baseline predicts the majority class only; "
        "use it to judge if ML adds value."
    )
    print(
        f"Predicted delayed rate: {baseline_metrics['predicted_delayed_rate']:.3f}; "
        f"Actual delayed rate: {baseline_metrics['actual_delayed_rate']:.3f}"
    )
    _print_interpretation(baseline_metrics)
    metrics_rows.append(baseline_metrics)

    for name, model in models.items():
        print(f"\nModel: {name}")
        preds = model.predict(X_test)
        roc_auc = None
        balanced_summary = None
        threshold_summary = {
            "best_f1_threshold": None,
            "best_f1": None,
            "best_f1_precision": None,
            "best_f1_recall": None,
            "best_f1_predicted_delayed_rate": None,
        }

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, proba)
            threshold_df, threshold_summary, balanced_summary = _evaluate_thresholds(
                y_test, proba, name, target_column
            )
            threshold_rows.append(threshold_df)
            print(
                f"Best F1 threshold for {name}: "
                f"{threshold_summary['best_f1_threshold']}"
            )

        metrics = _evaluate_predictions(y_test, preds, roc_auc=roc_auc)
        metrics.update({"model": name})
        metrics.update(threshold_summary)
        if balanced_summary is not None:
            metrics.update(balanced_summary)
        metrics_rows.append(metrics)

        print(classification_report(y_test, preds, zero_division=0))
        if metrics["roc_auc"] is not None:
            print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
        print(
            f"Predicted delayed rate: {metrics['predicted_delayed_rate']:.3f}; "
            f"Actual delayed rate: {metrics['actual_delayed_rate']:.3f}"
        )
        _print_interpretation(metrics)

        cm = confusion_matrix(y_test, preds)
        _plot_confusion_matrix(
            cm,
            title=f"Confusion Matrix: {name}",
            path=figures_dir / f"confusion_matrix_{name}{suffix}.png",
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(
        reports_dir / f"model_metrics_{target_column}.csv", index=False
    )

    if threshold_rows:
        threshold_df = pd.concat(threshold_rows, ignore_index=True)
        threshold_df.to_csv(
            reports_dir / f"threshold_analysis{suffix}.csv", index=False
        )
        _plot_thresholds(threshold_df, figures_dir)
        _plot_predicted_vs_actual_rate(threshold_df, figures_dir, target_column)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(metrics_df["model"], metrics_df["f1"], color="#4C78A8")
    ax.set_title("Model Comparison (F1)")
    ax.set_xlabel("Model")
    ax.set_ylabel("F1 Score")
    fig.tight_layout()
    fig.savefig(figures_dir / f"model_comparison{suffix}.png", dpi=150)
    plt.close(fig)

    print(f"Saved metrics and figures to {reports_dir}")

    return metrics_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models for a target.")
    parser.add_argument(
        "--target",
        default="delayed",
        help="Target column to evaluate (default: delayed).",
    )
    args = parser.parse_args()
    evaluate_models(args.target)


if __name__ == "__main__":
    main()
