from __future__ import annotations

from pathlib import Path

import joblib
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


def load_feature_data() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    features_path = project_root / "data" / "processed" / "complaints_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature data not found at {features_path}. Run features.py first."
        )
    return pd.read_csv(features_path)


def _is_leakage_column(column: str) -> bool:
    lower = column.lower()
    leak_keywords = [
        "days_to_resolution",
        "closed",
        "resolved",
        "resolution",
        "outcome",
        "final",
        "completed",
        "status",
    ]
    return any(keyword in lower for keyword in leak_keywords)


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"delayed"}
    candidate_cols = [col for col in df.columns if col not in exclude]

    candidate_cols = [col for col in candidate_cols if not _is_leakage_column(col)]
    candidate_cols = [col for col in candidate_cols if not col.lower().endswith("_id")]
    candidate_cols = [
        col for col in candidate_cols if col.lower() not in {"id", "cartodb_id"}
    ]

    datetime_cols = df[candidate_cols].select_dtypes(include=["datetime64[ns]"]).columns
    candidate_cols = [col for col in candidate_cols if col not in datetime_cols]

    if "opened_date" in candidate_cols:
        candidate_cols.remove("opened_date")

    # Drop extremely high-cardinality categorical columns to keep feature size manageable.
    high_cardinality = []
    protected_categoricals = {"complaint_type", "complaint_source", "zip_code"}
    for col in candidate_cols:
        if df[col].dtype == "object" or str(df[col].dtype).startswith("string"):
            unique_count = df[col].nunique(dropna=True)
            if unique_count > 5000 and col not in protected_categoricals:
                high_cardinality.append(col)

    if high_cardinality:
        print("Dropping high-cardinality categorical columns:")
        for col in high_cardinality:
            print(f"- {col}")
        candidate_cols = [col for col in candidate_cols if col not in high_cardinality]

    return candidate_cols


def split_data(
    df: pd.DataFrame, feature_cols: list[str]
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
            y_train = train_df["delayed"].astype(int)
            y_test = test_df["delayed"].astype(int)
            return X_train, X_test, y_train, y_test, "time"

    X = df[feature_cols]
    y = df["delayed"].astype(int)
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


def train_models() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_data()
    feature_cols = select_feature_columns(df)
    X_train, X_test, y_train, y_test, split_method = split_data(df, feature_cols)

    print(f"Train/test split method: {split_method}")
    print(f"Training rows: {len(X_train)}, Testing rows: {len(X_test)}")
    print(f"Features used: {len(feature_cols)}")

    preprocessor_sparse = build_preprocessor(X_train, sparse_output=True)
    preprocessor_dense = build_preprocessor(X_train, sparse_output=False)

    logistic_model = Pipeline(
        steps=[
            ("preprocess", preprocessor_sparse),
            (
                "model",
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", solver="saga", n_jobs=-1
                ),
            ),
        ]
    )
    rf_model = Pipeline(
        steps=[
            ("preprocess", preprocessor_dense),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    logistic_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    joblib.dump(logistic_model, models_dir / "logistic_regression.pkl")
    joblib.dump(rf_model, models_dir / "random_forest.pkl")

    metrics_rows = []
    for name, model in [
        ("logistic_regression", logistic_model),
        ("random_forest", rf_model),
    ]:
        metrics = compute_metrics(model, X_test, y_test)
        metrics["model"] = name
        metrics_rows.append(metrics)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(reports_dir / "model_metrics.csv", index=False)

    print("Saved trained models and metrics.")

    return metrics_df


def main() -> None:
    train_models()


if __name__ == "__main__":
    main()
