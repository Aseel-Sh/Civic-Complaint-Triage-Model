from __future__ import annotations

from pathlib import Path

import argparse

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier

from preprocess_utils import to_string_array


SAFE_FEATURES = [
    "complaint_type",
    "complaint_source",
    "zip_code",
    "council_district",
    "lat",
    "lng",
    "submitted_month",
    "submitted_dayofweek",
    "is_weekend",
    "complaint_type_total_count",
    "zip_total_complaints",
    "zip_type_complaint_count",
]


def load_feature_data() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    features_path = project_root / "data" / "processed" / "complaints_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature data not found at {features_path}. Run features.py first."
        )
    return pd.read_csv(features_path, low_memory=False)


def _save_selected_features(
    features: list[str], reports_dir: Path, target_column: str
) -> None:
    suffix = f"_{target_column}" if target_column else ""
    output_path = reports_dir / f"selected_features{suffix}.txt"
    output_path.write_text("\n".join(features))


def _load_selected_features(
    reports_dir: Path, target_column: str
) -> list[str]:
    suffix = f"_{target_column}" if target_column else ""
    input_path = reports_dir / f"selected_features{suffix}.txt"
    if not input_path.exists():
        return []
    return [
        line.strip()
        for line in input_path.read_text().splitlines()
        if line.strip()
    ]


def select_feature_columns(df: pd.DataFrame, target_column: str) -> list[str]:
    # The model only uses fields that would be available near complaint submission
    # time or derived from historical complaint data.
    return [
        col for col in SAFE_FEATURES if col in df.columns and col != target_column
    ]


def get_feature_columns(
    df: pd.DataFrame, target_column: str, reports_dir: Path | None = None
) -> list[str]:
    base = select_feature_columns(df, target_column)
    if reports_dir is None:
        return base

    selected = _load_selected_features(reports_dir, target_column)
    if not selected:
        return base

    return [
        col
        for col in selected
        if col in SAFE_FEATURES and col in df.columns and col != target_column
    ]


def split_data(
    df: pd.DataFrame, feature_cols: list[str], target_column: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    if "opened_date" in df.columns:
        opened_date = pd.to_datetime(df["opened_date"], errors="coerce")
        if opened_date.notna().any():
            df = df.copy()
            df["_opened_date"] = opened_date
            df = df.sort_values("_opened_date")
            split_index = int(len(df) * 0.8)
            train_df = df.iloc[:split_index]
            test_df = df.iloc[split_index:]
            X_train = train_df[feature_cols]
            X_test = test_df[feature_cols]
            y_train = train_df[target_column].astype(int)
            y_test = test_df[target_column].astype(int)
            return X_train, X_test, y_train, y_test, "time"

    X = df[feature_cols]
    y = df[target_column].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, "random"


def build_preprocessor(X: pd.DataFrame, *, sparse_output: bool) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns
    categorical_features = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    if len(numeric_features) == 0 and len(categorical_features) == 0:
        raise ValueError("No usable features found after preprocessing checks.")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "to_string",
                FunctionTransformer(to_string_array),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=50,
                    max_categories=50,
                    sparse_output=sparse_output,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )


def _get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        names = preprocessor.get_feature_names_out()
    except AttributeError:
        return []

    cleaned = []
    for name in names:
        cleaned.append(
            name.replace("numeric__", "")
            .replace("categorical__", "")
            .replace("onehot__", "")
        )
    return cleaned


def _save_random_forest_importance(
    model: Pipeline, reports_dir: Path, figures_dir: Path, suffix: str = ""
) -> None:
    preprocessor = model.named_steps.get("preprocess")
    estimator = model.named_steps.get("model")
    if preprocessor is None or estimator is None:
        return

    feature_names = _get_feature_names(preprocessor)
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None or len(feature_names) != len(importances):
        return

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(
        reports_dir / f"random_forest_feature_importance{suffix}.csv", index=False
    )

    top = importance_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["importance"], color="#4C78A8")
    ax.set_title("Random Forest Top 15 Features")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(figures_dir / f"random_forest_top_features{suffix}.png", dpi=150)
    plt.close(fig)


def compute_metrics(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
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

    return metrics


def train_models(target_column: str, sample_size: int | None, fast: bool) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_data()
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")
    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("--sample-size must be a positive integer.")
        sample_count = min(sample_size, len(df))
        df = df.sample(n=sample_count, random_state=42)
        print(f"Using sample size: {len(df)} rows")
    feature_cols = get_feature_columns(df, target_column, reports_dir)
    X_train, X_test, y_train, y_test, split_method = split_data(
        df, feature_cols, target_column
    )

    print(f"Train/test split method: {split_method}")
    print(f"Training rows: {len(X_train)}, Testing rows: {len(X_test)}")
    print(f"Features used: {len(feature_cols)}")
    print(f"Selected features for target '{target_column}':")
    for feature in feature_cols:
        print(f"- {feature}")
    _save_selected_features(feature_cols, reports_dir, target_column)

    preprocessor_sparse = build_preprocessor(X_train, sparse_output=True)
    preprocessor_dense = build_preprocessor(X_train, sparse_output=False)

    if fast:
        logistic_params = {"max_iter": 1000, "class_weight": "balanced"}
        rf_params = {
            "n_estimators": 50,
            "max_depth": 8,
            "min_samples_leaf": 20,
            "random_state": 42,
            "class_weight": "balanced",
            "n_jobs": -1,
        }
    else:
        logistic_params = {
            "max_iter": 3000,
            "class_weight": "balanced",
            "solver": "saga",
        }
        rf_params = {
            "n_estimators": 200,
            "max_depth": 12,
            "random_state": 42,
            "class_weight": "balanced",
        }

    logistic_model = Pipeline(
        steps=[
            ("preprocess", preprocessor_sparse),
            ("model", LogisticRegression(**logistic_params)),
        ]
    )
    # Logistic regression is a baseline; random forest is the main model.
    rf_model = Pipeline(
        steps=[
            ("preprocess", preprocessor_dense),
            ("model", RandomForestClassifier(**rf_params)),
        ]
    )

    logistic_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    suffix = f"_{target_column}" if target_column else ""
    _save_random_forest_importance(rf_model, reports_dir, figures_dir, suffix)

    joblib.dump(
        logistic_model, models_dir / f"logistic_regression_{target_column}.pkl"
    )
    joblib.dump(
        rf_model, models_dir / f"random_forest_{target_column}.pkl"
    )

    metrics_rows = []
    for name, model in [
        ("logistic_regression", logistic_model),
        ("random_forest", rf_model),
    ]:
        metrics = compute_metrics(model, X_test, y_test)
        metrics["model"] = name
        metrics_rows.append(metrics)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(
        reports_dir / f"model_metrics_{target_column}.csv", index=False
    )

    print("Saved trained models and metrics.")

    return metrics_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train models for a target.")
    parser.add_argument(
        "--target",
        default="delayed",
        help="Target column to predict (default: delayed).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional number of rows to sample for faster training.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use smaller, faster model settings.",
    )
    args = parser.parse_args()
    train_models(args.target, args.sample_size, args.fast)


if __name__ == "__main__":
    main()
