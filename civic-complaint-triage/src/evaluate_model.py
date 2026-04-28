from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
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


def _evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, proba)
    else:
        metrics["roc_auc"] = None

    print(classification_report(y_test, preds, zero_division=0))

    return metrics


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


def evaluate_models() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_data()
    feature_cols = select_feature_columns(df)
    X_train, X_test, y_train, y_test, split_method = split_data(df, feature_cols)

    print(f"Evaluation split method: {split_method}")
    print(f"Testing rows: {len(X_test)}")

    models = {
        "logistic_regression": joblib.load(models_dir / "logistic_regression.pkl"),
        "random_forest": joblib.load(models_dir / "random_forest.pkl"),
    }

    metrics_rows = []
    for name, model in models.items():
        print(f"\nModel: {name}")
        metrics = _evaluate(model, X_test, y_test)
        metrics["model"] = name
        metrics_rows.append(metrics)

        cm = confusion_matrix(y_test, model.predict(X_test))
        _plot_confusion_matrix(
            cm,
            title=f"Confusion Matrix: {name}",
            path=figures_dir / f"confusion_matrix_{name}.png",
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(reports_dir / "model_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(metrics_df["model"], metrics_df["f1"], color="#4C78A8")
    ax.set_title("Model Comparison (F1)")
    ax.set_xlabel("Model")
    ax.set_ylabel("F1 Score")
    fig.tight_layout()
    fig.savefig(figures_dir / "model_comparison.png", dpi=150)
    plt.close(fig)

    print(f"Saved metrics and figures to {reports_dir}")

    return metrics_df


def main() -> None:
    evaluate_models()


if __name__ == "__main__":
    main()
