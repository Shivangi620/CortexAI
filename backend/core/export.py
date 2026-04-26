import json
import os
import time
import zipfile

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from infra.launch_origin import parse_launch_origin
from infra.storage import get_run_dir, resolve_model_path


def build_export_bundle_filename(job_id: str, launch_origin: dict | None = None) -> str:
    launch_origin = launch_origin or {}
    prefix = "drift_reopen" if launch_origin.get("launch_source") == "drift_recommendation" else "manual"
    return f"{prefix}_automl_export_{job_id[:8]}.zip"


def _safe_json_load(path: str):
    try:
        if path and os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _display_metric(metric_name: str, score_val):
    try:
        score_float = float(score_val)
    except (TypeError, ValueError):
        return "N/A"

    metric_lower = (metric_name or "").lower()
    if "r²" in metric_name or metric_lower in {"r2", "r2 score"}:
        return f"{score_float / 100:.3f}"
    if "rmse" in metric_lower or "mse" in metric_lower or "mae" in metric_lower:
        return f"{score_float:.4f}"
    return f"{score_float:.1f}%"


def _model_import_and_init(best_model_name: str, is_clf: bool):
    if "LightGBM" in best_model_name:
        model_class = "LGBMClassifier" if is_clf else "LGBMRegressor"
        return (
            f"from lightgbm import {model_class}",
            f"{model_class}(random_state=42, verbose=-1)",
        )
    if "XGBoost" in best_model_name or "XGB" in best_model_name:
        model_class = "XGBClassifier" if is_clf else "XGBRegressor"
        if is_clf:
            return (
                f"from xgboost import {model_class}",
                f"{model_class}(random_state=42, n_estimators=150, eval_metric='logloss')",
            )
        return (
            f"from xgboost import {model_class}",
            f"{model_class}(random_state=42, n_estimators=150)",
        )
    if "Forest" in best_model_name:
        model_class = "RandomForestClassifier" if is_clf else "RandomForestRegressor"
        return (
            f"from sklearn.ensemble import {model_class}",
            f"{model_class}(n_estimators=200, random_state=42)",
        )
    if "Extra Trees" in best_model_name:
        model_class = "ExtraTreesClassifier" if is_clf else "ExtraTreesRegressor"
        return (
            f"from sklearn.ensemble import {model_class}",
            f"{model_class}(n_estimators=200, random_state=42)",
        )
    if "Hist Gradient Boosting" in best_model_name:
        model_class = (
            "HistGradientBoostingClassifier"
            if is_clf
            else "HistGradientBoostingRegressor"
        )
        return (
            f"from sklearn.ensemble import {model_class}",
            f"{model_class}(random_state=42)",
        )
    if "Decision Tree" in best_model_name:
        model_class = "DecisionTreeClassifier" if is_clf else "DecisionTreeRegressor"
        return (
            f"from sklearn.tree import {model_class}",
            f"{model_class}(max_depth=8, random_state=42)",
        )
    if "Logistic" in best_model_name:
        return (
            "from sklearn.linear_model import LogisticRegression",
            "LogisticRegression(max_iter=1000, random_state=42)",
        )
    if "Linear" in best_model_name:
        return (
            "from sklearn.linear_model import LinearRegression",
            "LinearRegression()",
        )
    if "Ridge" in best_model_name:
        return (
            "from sklearn.linear_model import Ridge",
            "Ridge(alpha=1.0, random_state=42)",
        )
    if "ElasticNet" in best_model_name:
        return (
            "from sklearn.linear_model import ElasticNet",
            "ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)",
        )
    if "MLP" in best_model_name:
        model_class = "MLPClassifier" if is_clf else "MLPRegressor"
        return (
            f"from sklearn.neural_network import {model_class}",
            f"{model_class}(hidden_layer_sizes=(64, 32), max_iter=250, early_stopping=True, random_state=42)",
        )
    if "SVM" in best_model_name or "SVC" in best_model_name:
        model_class = "SVC" if is_clf else "SVR"
        if is_clf:
            return (
                f"from sklearn.svm import {model_class}",
                f"{model_class}(probability=True)",
            )
        return (
            f"from sklearn.svm import {model_class}",
            f"{model_class}()",
        )
    return (
        "# Replace with your preferred estimator import",
        "# Replace with your preferred estimator init",
    )


def _training_script_content(
    best_model_name: str,
    target_name: str,
    feature_names,
    is_clf: bool,
    metric_name: str,
    execution_profile: dict,
    preprocessor_name: str,
):
    model_import, model_init = _model_import_and_init(best_model_name, is_clf)
    selected_features_literal = json.dumps(feature_names or [])
    is_clf_literal = "True" if is_clf else "False"

    return f'''"""
End-to-End Training + Inference Script
Generated by AutoML Studio

This script mirrors the exported pipeline as closely as possible:
1. Load dataset
2. Clean and normalize data
3. Select features and target
4. Split train/test
5. Build preprocessing pipeline
6. Train model
7. Evaluate
8. Save reusable bundle
9. Predict on new records
"""

import json
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler

try:
    from category_encoders import TargetEncoder
except ImportError:
    TargetEncoder = None

{model_import}


TARGET_COLUMN = "{target_name}"
SELECTED_FEATURES = {selected_features_literal}
IS_CLASSIFICATION = {is_clf_literal}
PRIMARY_METRIC = "{metric_name}"
EXPORT_PREPROCESSOR = "{preprocessor_name}"
EXECUTION_PROFILE = {json.dumps(execution_profile or {{}}, indent=4)}
SANITIZATION_POLICY = {{
    "drop_duplicates": True,
    "drop_invalid_target_rows": True,
    "drop_constant_columns": True,
    "drop_gt_90pct_missing_columns": True,
    "normalize_null_like_tokens": ["??", "nan", "none", "null", "unknown", "na", "n/a", "invalid", "?"],
}}


class OutlierClipper:
    def __init__(self, factor=3.0):
        self.factor = factor

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.clip(lower=self.lower_, upper=self.upper_, axis=1).values

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else [], dtype=object)


class SkewTransformer:
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.skewed_cols_ = X.columns[X.skew(numeric_only=True).abs() > 0.75].tolist()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in self.skewed_cols_:
            X[col] = np.log1p(np.maximum(X[col], 0))
        return X.values

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else [], dtype=object)


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df is None or df.empty:
        raise ValueError("Dataset is empty or unreadable.")
    return df


def basic_cleaning(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    df = df.copy()
    placeholders = SANITIZATION_POLICY["normalize_null_like_tokens"]

    df.columns = [str(col).strip().replace("\\n", " ") for col in df.columns]

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace(placeholders, np.nan)
            df[col] = df[col].astype(str).str.strip().replace("nan", np.nan)
            numeric_try = pd.to_numeric(df[col], errors="coerce")
            if float(numeric_try.notna().mean()) >= 0.85:
                df[col] = numeric_try

    if SANITIZATION_POLICY["drop_duplicates"]:
        df = df.drop_duplicates()
    if SANITIZATION_POLICY["drop_invalid_target_rows"]:
        df = df.dropna(subset=[target_column])

    high_missing_cols = [c for c in df.columns if c != target_column and df[c].isna().mean() > 0.9]
    if SANITIZATION_POLICY["drop_gt_90pct_missing_columns"] and high_missing_cols:
        df = df.drop(columns=high_missing_cols)

    constant_cols = [c for c in df.columns if c != target_column and df[c].nunique(dropna=False) <= 1]
    if SANITIZATION_POLICY["drop_constant_columns"] and constant_cols:
        df = df.drop(columns=constant_cols)

    return df


def select_training_columns(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    requested = [c for c in SELECTED_FEATURES if c in df.columns and c != target_column]
    if requested:
        keep_cols = requested + [target_column]
        return df[keep_cols].copy()
    return df.copy()


def build_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if EXPORT_PREPROCESSOR == "full_column_transformer":
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("skew_fix", SkewTransformer()),
                ("outlier_clipper", OutlierClipper()),
                ("interactions", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

    cat_steps = [("imputer", SimpleImputer(strategy="constant", fill_value="missing"))]
    if TargetEncoder is not None and categorical_cols:
        cat_steps.append(("target_encoder", TargetEncoder()))
    categorical_transformer = Pipeline(steps=cat_steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


def build_model():
    return {model_init}


def prepare_target(y: pd.Series):
    encoder = None
    if IS_CLASSIFICATION:
        encoder = LabelEncoder()
        y = pd.Series(encoder.fit_transform(y.astype(str)), index=y.index)
    return y, encoder


def train_pipeline(df: pd.DataFrame):
    df = basic_cleaning(df, TARGET_COLUMN)
    df = select_training_columns(df, TARGET_COLUMN)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    y, label_encoder = prepare_target(y)

    split_kwargs = {{"test_size": 0.2, "random_state": 42}}
    if IS_CLASSIFICATION and y.nunique() > 1:
        split_kwargs["stratify"] = y

    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)
    model = build_model()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)

    metrics = evaluate_pipeline(pipeline, X_test, y_test)
    bundle = {{
        "model": pipeline,
        "features": X.columns.tolist(),
        "target": TARGET_COLUMN,
        "is_classification": IS_CLASSIFICATION,
        "label_encoder_classes": label_encoder.classes_.tolist() if label_encoder is not None else None,
        "metadata": {{
            "model_type": "{best_model_name}",
            "metric_name": PRIMARY_METRIC,
            "preprocessor": EXPORT_PREPROCESSOR,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "execution_profile": EXECUTION_PROFILE,
        }},
    }}
    return bundle, metrics, X_test


def evaluate_pipeline(pipeline, X_test, y_test):
    preds = pipeline.predict(X_test)
    if IS_CLASSIFICATION:
        return {{
            "accuracy": round(float(accuracy_score(y_test, preds)) * 100, 2),
            "precision": round(float(precision_score(y_test, preds, average="weighted", zero_division=0)) * 100, 2),
            "recall": round(float(recall_score(y_test, preds, average="weighted", zero_division=0)) * 100, 2),
            "f1": round(float(f1_score(y_test, preds, average="weighted", zero_division=0)) * 100, 2),
        }}
    return {{
        "r2": round(float(r2_score(y_test, preds)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
        "mae": round(float(mean_absolute_error(y_test, preds)), 4),
    }}


def save_bundle(bundle, output_path="model.pkl"):
    joblib.dump(bundle, output_path)


def predict_from_bundle(bundle_path: str, records):
    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    features = bundle["features"]
    frame = pd.DataFrame(records)
    frame = frame[features]
    return model.predict(frame)


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "dataset.csv"
    bundle, metrics, X_test = train_pipeline(load_dataset(data_path))
    save_bundle(bundle, "model.pkl")
    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    if len(X_test):
        sample = X_test.head(1)
        pred = predict_from_bundle("model.pkl", sample.to_dict(orient="records"))
        print("Sample prediction:", pred[0])
'''


def _inference_script_content(feature_names, target_name, best_model_name):
    example_features = {
        f: "..." for f in (feature_names or [])[: min(6, len(feature_names or []))]
    }
    return f'''"""
Inference helpers for the exported AutoML model.
"""

import joblib
import pandas as pd


raw_obj = joblib.load("model.pkl")
if isinstance(raw_obj, dict) and "model" in raw_obj:
    bundle = raw_obj
    model = bundle["model"]
    FEATURE_NAMES = bundle.get("features", {feature_names})
    TARGET_NAME = bundle.get("target", "{target_name}")
    MODEL_TYPE = bundle.get("metadata", {{}}).get("model_type", "{best_model_name}")
else:
    bundle = {{"model": raw_obj, "features": {feature_names}, "target": "{target_name}", "metadata": {{"model_type": "{best_model_name}"}}}}
    model = raw_obj
    FEATURE_NAMES = {feature_names}
    TARGET_NAME = "{target_name}"
    MODEL_TYPE = "{best_model_name}"


def _normalize_record(record: dict):
    cleaned = {{}}
    for feature in FEATURE_NAMES:
        value = record.get(feature)
        if isinstance(value, str) and value.strip().lower() in {{"", "nan", "null", "none", "n/a", "na", "unknown", "?"}}:
            value = None
        cleaned[feature] = value
    return cleaned


def predict_one(record: dict):
    frame = pd.DataFrame([_normalize_record(record)], columns=FEATURE_NAMES)
    pred = model.predict(frame)[0]
    return float(pred) if hasattr(pred, "item") else pred


def predict_many(records):
    frame = pd.DataFrame([_normalize_record(record) for record in records], columns=FEATURE_NAMES)
    preds = model.predict(frame)
    return [float(p) if hasattr(p, "item") else p for p in preds]


def predict_proba_one(record: dict):
    if not hasattr(model, "predict_proba"):
        return None
    frame = pd.DataFrame([_normalize_record(record)], columns=FEATURE_NAMES)
    proba = model.predict_proba(frame)[0]
    labels = list(getattr(model, "classes_", range(len(proba))))
    return {{
        str(label): float(score)
        for label, score in zip(labels, proba)
    }}


if __name__ == "__main__":
    print("Loaded", MODEL_TYPE)
    print("Expected target:", TARGET_NAME)
    print("Expected features:", FEATURE_NAMES)
    print("Example payload:", {example_features})
'''


def _api_script_content(feature_names, target_name, best_model_name):
    return f'''"""
Minimal FastAPI wrapper for the exported model.
"""

import joblib
import os
import pandas as pd
import sys
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


FEATURE_NAMES = {feature_names}
TARGET_NAME = "{target_name}"
MODEL_TYPE = "{best_model_name}"
MODEL = joblib.load("model.pkl")


class PredictRequest(BaseModel):
    features: dict


app = FastAPI(title=f"Exported {{MODEL_TYPE}} API")


def _normalize(features: dict):
    cleaned = {{}}
    for feature in FEATURE_NAMES:
        value = features.get(feature)
        if isinstance(value, str) and value.strip().lower() in {{"", "nan", "null", "none", "n/a", "na", "unknown", "?"}}:
            value = None
        cleaned[feature] = value
    return cleaned


@app.get("/health")
def health():
    return {{"status": "ok", "model": MODEL_TYPE, "target": TARGET_NAME}}


@app.post("/predict")
def predict(req: PredictRequest):
    frame = pd.DataFrame([_normalize(req.features)], columns=FEATURE_NAMES)
    pred = MODEL.predict(frame)[0]
    payload = {{
        "prediction": float(pred) if hasattr(pred, "item") else pred,
        "feature_names": FEATURE_NAMES,
    }}
    if hasattr(MODEL, "predict_proba"):
        try:
            proba = MODEL.predict_proba(frame)[0]
            labels = list(getattr(MODEL, "classes_", range(len(proba))))
            payload["probabilities"] = {{
                str(label): float(score)
                for label, score in zip(labels, proba)
            }}
        except Exception:
            pass
    return payload
'''


def _requirements_content(best_model_name: str, preprocessor_name: str):
    deps = [
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
    ]

    if preprocessor_name in {"lite_column_transformer", "full_column_transformer"}:
        deps.append("category-encoders>=2.6.0")
    if "LightGBM" in best_model_name:
        deps.append("lightgbm>=4.0.0")
    if "XGBoost" in best_model_name or "XGB" in best_model_name:
        deps.append("xgboost>=2.0.0")

    deps.extend(["shap>=0.44.0", "optuna>=3.0.0"])
    deps.extend(["fastapi>=0.110.0", "uvicorn>=0.29.0"])
    return "\n".join(dict.fromkeys(deps)) + "\n"


def _windows_runner_bat_content():
    return r'''@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

if not exist ".venv" (
  py -m venv .venv
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements.txt

if "%~1"=="" (
  set DATASET=training_dataset.csv
) else (
  set DATASET=%~1
)

python training.py "%DATASET%"
if errorlevel 1 (
  echo Training failed.
  exit /b 1
)

echo Export bundle is ready. Use inference.py for local predictions.
endlocal
'''


def _explain_script_content(feature_names):
    feature_names_literal = json.dumps(feature_names or [])
    return '''"""
Optional local explanation script.
Produces top feature importances for a CSV input.
Usage:
    python explain.py dataset.csv
"""

import json
import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import shap
except ImportError:
    shap = None

raw_obj = joblib.load("model.pkl")
if isinstance(raw_obj, dict) and "model" in raw_obj:
    bundle = raw_obj
    model = bundle["model"]
    FEATURE_NAMES = bundle.get("features", [])
else:
    bundle = {"model": raw_obj, "features": __FEATURE_NAMES__}
    model = raw_obj
    FEATURE_NAMES = __FEATURE_NAMES__


def main():
    if shap is None:
        print("shap is not installed. Run: pip install shap")
        return

    input_path = sys.argv[1] if len(sys.argv) > 1 else "dataset.csv"
    df = pd.read_csv(input_path)
    frame = df[FEATURE_NAMES].head(50)

    if hasattr(model, "preprocess") and hasattr(model, "get_underlying_model"):
        transformed = model.preprocess(frame)
        estimator = model.get_underlying_model()
        names = model.get_feature_names_out()
    elif hasattr(model, "preprocess") and hasattr(model, "model_"):
        transformed = model.preprocess(frame)
        estimator = model.model_
        names = model.get_feature_names_out()
    else:
        pre = model.named_steps["preprocessor"]
        estimator = model.named_steps["model"]
        transformed = pre.transform(frame)
        names = pre.get_feature_names_out()

    explainer = shap.Explainer(estimator, transformed)
    shap_values = explainer(transformed)
    values = np.abs(shap_values.values).mean(axis=0)
    if len(values.shape) > 1:
        values = values.mean(axis=1)

    rows = [
        {"feature": str(name), "importance": round(float(val), 6)}
        for name, val in sorted(zip(names, values), key=lambda x: x[1], reverse=True)[:20]
    ]
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
'''.replace("__FEATURE_NAMES__", feature_names_literal)


def _feature_dictionary(schema_data: dict, feature_names, profile: dict):
    schema = (schema_data or {}).get("schema", {})
    col_stats = (profile or {}).get("column_stats", {})
    entries = []
    for name in feature_names or []:
        stats = col_stats.get(name, {}) if isinstance(col_stats, dict) else {}
        spec = schema.get(name, {}) if isinstance(schema, dict) else {}
        entries.append(
            {
                "name": name,
                "dtype": spec.get("type") or stats.get("dtype", "unknown"),
                "missing_count": spec.get("missing", stats.get("missing", 0)),
                "missing_pct": stats.get("missing_pct", 0),
                "semantic_type": stats.get("semantic_type", "unknown"),
                "notes": "Required model input field; preserve training-time meaning and units.",
            }
        )
    return json.dumps(entries, indent=2)


def _pipeline_steps_content(results: dict, model_metadata: dict) -> str:
    sanitizer = results.get("sanitizer_report", {}) or {}
    optimized = (results.get("performance_metrics", {}) or {}).get(
        "optimized_metric", {}
    )
    payload = {
        "artifact_type": "end_to_end_raw_tabular_pipeline",
        "load_contract": (
            "Load model.pkl from the bundle root. The bundle includes the custom "
            "services/ source package required by joblib to import the pipeline classes."
        ),
        "task_type": model_metadata.get("task_type") or results.get("task_detection", {}).get("task_type"),
        "target": results.get("target"),
        "optimized_metric": optimized,
        "raw_feature_names": results.get("feature_names", []),
        "derived_feature_names": results.get("derived_feature_names", []),
        "preprocessing": {
            "custom_feature_engineer": "services.training.inference.DeterministicFeatureEngineer",
            "sanitizer": "services.data_sanitizer.sanitize_dataframe",
            "preprocessor": model_metadata.get("preprocessor"),
            "pca_applied": model_metadata.get("pca_applied"),
            "pca_components_used": model_metadata.get("pca_components_used"),
            "class_labels": model_metadata.get("class_labels", []),
        },
        "cleaning_policy_observed": {
            "rows_before": sanitizer.get("rows_before"),
            "rows_after": sanitizer.get("rows_after"),
            "duplicate_rows_removed": sanitizer.get("duplicate_rows_removed"),
            "dropped_target_rows": sanitizer.get("dropped_target_rows"),
            "numeric_coercions": sanitizer.get("numeric_coercions", []),
            "datetime_columns": sanitizer.get("datetime_columns", []),
            "categorical_columns": sanitizer.get("categorical_columns", []),
            "empty_string_cells_cleaned": sanitizer.get("empty_string_cells_cleaned"),
        },
        "quality_warnings": results.get("warnings", []),
        "validation_summary": results.get("validation_summary", {}),
    }
    return json.dumps(payload, indent=2, default=str)


def _source_manifest_content() -> str:
    return """# Pipeline Source Manifest

The exported `model.pkl` is the real trained inference artifact. It contains custom
classes for raw-input cleaning, deterministic feature handling, preprocessing, and
label decoding. Those classes are included in this bundle so `joblib.load("model.pkl")`
works outside the AutoML Studio repository.

Included source files:
- `services/data_sanitizer.py`: null normalization, type coercion, target validation.
- `services/training/preprocessing.py`: sklearn preprocessing factories and custom transformers.
- `services/training/inference.py`: raw tabular inference artifact and deterministic feature engineering.

Keep these files beside `model.pkl` unless you package them into your own Python
environment. The prediction helpers already add the bundle root to `sys.path`.
"""


def _readme_content(
    best_model_name: str,
    metric_name: str,
    score_val,
    target_name: str,
    feature_names,
    execution_profile: dict,
    tested_models,
    validation_summary: dict,
    warnings: list,
    training_date: str,
    risks: list,
    intended_use: str,
    launch_origin: dict,
):
    display_score = _display_metric(metric_name, score_val)
    feature_list = (
        "\n".join(f"- `{f}`" for f in (feature_names or [])) or "- *(not available)*"
    )
    tested_lines = []
    for item in tested_models[:10] if isinstance(tested_models, list) else []:
        line = (
            f"- `{item.get('model', 'Unknown')}` | status={item.get('status', 'n/a')}"
        )
        if item.get("sweep_score") is not None:
            line += f" | sweep={item.get('sweep_score')}%"
        if item.get("best_cv_score") is not None:
            line += f" | best_cv={item.get('best_cv_score')}%"
        if item.get("holdout_score") is not None:
            line += f" | holdout={item.get('holdout_score')}%"
        tested_lines.append(line)
    tested_block = "\n".join(tested_lines) if tested_lines else "- *(not available)*"
    validation_summary = validation_summary or {}
    warning_lines = []
    for item in warnings[:5] if isinstance(warnings, list) else []:
        warning_lines.append(
            f"- `{item.get('type', 'warning')}` ({item.get('severity', 'info')}): {item.get('message', '')}"
        )
    warning_block = "\n".join(warning_lines) if warning_lines else "- No major training warnings were triggered."
    launch_origin = launch_origin or {}
    launch_context = launch_origin.get("launch_context", {}) or {}
    origin_note = (
        f"Reopened from drift monitoring via parent run `{(launch_context.get('parent_job_id') or launch_context.get('source_job_id') or '—')}`."
        if launch_origin.get("launch_source") == "drift_recommendation"
        else "Directly launched from the studio."
    )

    return f"""# AutoML Export Bundle

## Summary
| Field | Value |
|---|---|
| Best model | {best_model_name} |
| Target | {target_name} |
| Metric | {metric_name} |
| Score | {display_score} |
| Launch origin | {launch_origin.get("launch_label", "Manual")} |

## Operational Origin
- Launch type: `{launch_origin.get("launch_label", "Manual")}`
- Context: {origin_note}
- Recommended lane: `{launch_context.get("recommended_goal", "—")} / {launch_context.get("recommended_mode", "—")}`

## Validation Drift
| Field | Value |
|---|---|
| Status | {validation_summary.get("status", "n/a")} |
| CV score | {validation_summary.get("cv_score", "n/a")} |
| Holdout score | {validation_summary.get("holdout_score", "n/a")} |
| Absolute gap | {validation_summary.get("absolute_gap_display", "n/a")} |
| Relative gap | {validation_summary.get("absolute_gap_ratio", "n/a")} |

## Execution Profile
```json
{json.dumps(execution_profile or {{}}, indent=2)}
```

## Quality Warnings
{warning_block}

## Exported Files
- `model.pkl`: real trained end-to-end inference artifact, including cleaning, feature handling, preprocessing, encoding, and the fitted estimator
- `services/`: source package required to load the custom artifact on another machine
- `pipeline_steps.json`: machine-readable record of the cleaning, preprocessing, target, metric, and validation decisions from the run
- `training.py`: reproducible retraining script for the same data contract
- `inference.py`: small prediction helper that loads the real artifact
- `api.py`: optional FastAPI wrapper around the exported model
- `explain.py`: optional local explanation helper
- `run_training_windows.bat`: Windows launcher for retraining the exported bundle
- `model_metadata.json`: saved model metadata from the run
- `metrics.json`: result payload from the run
- `schema.json`: saved data contract / schema snapshot when available

## Feature List
{feature_list}

## Tested Models
{tested_block}

## Model Card
| Field | Value |
|---|---|
| Training date | {training_date} |
| Intended use | {intended_use} |
| Key risks | {"; ".join(risks) if risks else "Model quality depends on data drift, missing-value patterns, and input consistency."} |

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Linux/macOS retrain: `python training.py path/to/dataset.csv`
3. Windows retrain: double-click `run_training_windows.bat` or run `run_training_windows.bat path\\to\\dataset.csv`
4. Load the trained artifact in Python with `joblib.load("model.pkl")`
5. Optional local serving: `uvicorn api:app --host 0.0.0.0 --port 8000`

## Production Note
For prediction, prefer `model.pkl`, `inference.py`, or `api.py`. The saved artifact is the source of truth because it contains the fitted cleaning/preprocessing/encoding/model state. `training.py` is included for retraining and auditability.
"""


def create_export_bundle(job_id: str, results: dict) -> str:
    """
    Creates a ZIP bundle containing the trained model and reproducible code.
    """
    export_dir = os.path.join("tmp", "exports")
    os.makedirs(export_dir, exist_ok=True)

    run_dir = get_run_dir(job_id)
    metrics_path = os.path.join(run_dir, "logs", "metrics.json")
    schema_path = os.path.join(run_dir, "data", "schema.json")
    metadata_path = os.path.join(run_dir, "artifacts", "model_metadata.json")
    model_path = resolve_model_path(job_id)

    model_metadata = _safe_json_load(metadata_path)
    saved_metrics = _safe_json_load(metrics_path)
    schema_data = _safe_json_load(schema_path)

    merged_results = dict(saved_metrics) if isinstance(saved_metrics, dict) else {}
    merged_results.update(results or {})

    feature_names = merged_results.get("feature_names", []) or model_metadata.get(
        "feature_names", []
    )
    target_name = merged_results.get("target", "target")
    best_model_name = merged_results.get("best_model", "Unknown")
    is_clf = bool(merged_results.get("is_classification", True))
    metric_name = merged_results.get("metric_name", "Score")
    score_val = merged_results.get("score", "N/A")
    execution_profile = merged_results.get("execution_profile", {})
    tested_models = merged_results.get("tested_models", [])
    validation_summary = merged_results.get("validation_summary", {})
    warnings = merged_results.get("warnings", [])
    preprocessor_name = model_metadata.get("preprocessor", "lite_column_transformer")
    training_date = time.strftime(
        "%Y-%m-%d", time.localtime(model_metadata.get("timestamp", time.time()))
    )

    profile = {}
    dataset_path = ""
    job_snapshot = {}
    try:
        from infra.database import db_session, JobModel, DatasetModel

        with db_session() as db:
            job = db.query(JobModel).filter(JobModel.id == job_id).first()
            if job:
                try:
                    job_snapshot = {
                        "story": job.story,
                        "insights": (
                            json.loads(job.insights_json) if job.insights_json else {}
                        ),
                        "reasoning": (
                            json.loads(job.reasoning_json) if job.reasoning_json else []
                        ),
                        "params": (
                            json.loads(job.params_json) if job.params_json else {}
                        ),
                    }
                except Exception:
                    job_snapshot = {
                        "story": job.story,
                        "insights": {},
                        "reasoning": [],
                        "params": {},
                    }
                dataset = (
                    db.query(DatasetModel)
                    .filter(DatasetModel.id == job.dataset_id)
                    .first()
                )
                if dataset:
                    dataset_path = dataset.file_path or ""
                    try:
                        profile = (
                            json.loads(dataset.profile_json)
                            if dataset.profile_json
                            else {}
                        )
                    except Exception:
                        profile = {}
    except Exception:
        profile = {}
        dataset_path = ""

    launch_origin = parse_launch_origin(job_snapshot.get("params", {}))

    risks = []
    execution_risks = []
    if execution_profile.get("use_full_preprocessor"):
        execution_risks.append(
            "Full-mode preprocessing may be more sensitive to schema drift and dependency mismatch."
        )
    if (profile or {}).get("missing_pct", 0):
        execution_risks.append(
            "Input data with different missing-value patterns may degrade performance."
        )
    if (profile or {}).get("imbalance") == "High ⚠️":
        execution_risks.append(
            "Class imbalance in production can affect recall and calibration."
        )
    risks = execution_risks[:3]

    zip_path = os.path.join(
        export_dir,
        build_export_bundle_filename(job_id, launch_origin),
    )

    temp_files = {
        "training.py": _training_script_content(
            best_model_name=best_model_name,
            target_name=target_name,
            feature_names=feature_names,
            is_clf=is_clf,
            metric_name=metric_name,
            execution_profile=execution_profile,
            preprocessor_name=preprocessor_name,
        ),
        "inference.py": _inference_script_content(
            feature_names, target_name, best_model_name
        ),
        "api.py": _api_script_content(feature_names, target_name, best_model_name),
        "requirements.txt": _requirements_content(best_model_name, preprocessor_name),
        "run_training_windows.bat": _windows_runner_bat_content(),
        "sample_input.json": json.dumps({feature: None for feature in feature_names}, indent=2),
        "sample_output.json": json.dumps({"prediction": None, "target": target_name}, indent=2),
        "README.md": _readme_content(
            best_model_name=best_model_name,
            metric_name=metric_name,
            score_val=score_val,
            target_name=target_name,
            feature_names=feature_names,
            execution_profile=execution_profile,
            tested_models=tested_models,
            validation_summary=validation_summary,
            warnings=warnings,
            training_date=training_date,
            risks=risks,
            intended_use="Batch or API inference on datasets with the same feature semantics and preprocessing assumptions as training.",
            launch_origin=launch_origin,
        ),
        "import_bundle.json": json.dumps(
            {
                "format": "automl_studio_export_bundle_v2",
                "job_id": job_id,
                "launch_origin": launch_origin,
                "profile": profile,
                "results": merged_results,
                "story": job_snapshot.get("story"),
                "insights": job_snapshot.get("insights", {}),
                "reasoning": job_snapshot.get("reasoning", []),
                "params": job_snapshot.get("params", {}),
            },
            indent=2,
        ),
    }

    generated_paths = []
    for filename, content in temp_files.items():
        path = os.path.join("tmp", f"{job_id}_{filename}")
        with open(path, "w") as f:
            f.write(content)
        generated_paths.append((path, filename))

    with zipfile.ZipFile(zip_path, "w") as zipf:
        if model_path and os.path.exists(model_path):
            zipf.write(model_path, arcname="model.pkl")

        if dataset_path and os.path.exists(dataset_path):
            zipf.write(dataset_path, arcname="training_dataset.csv")

        source_files = [
            ("backend/services/__init__.py", "services/__init__.py"),
            ("backend/services/data_sanitizer.py", "services/data_sanitizer.py"),
            ("backend/services/training/__init__.py", "services/training/__init__.py"),
            ("backend/services/training/preprocessing.py", "services/training/preprocessing.py"),
            ("backend/services/training/inference.py", "services/training/inference.py"),
        ]
        for src, arcname in source_files:
            if os.path.exists(src):
                zipf.write(src, arcname=arcname)

        for path, arcname in generated_paths:
            zipf.write(path, arcname=arcname)

    return zip_path


def cleanup_old_exports(max_age_hours: int = 24):
    """Remove tmp/ artifacts older than max_age_hours. Call periodically."""
    now = time.time()
    for folder in ["tmp", "tmp/exports"]:
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            if os.path.isfile(fpath):
                age_hours = (now - os.path.getmtime(fpath)) / 3600
                if age_hours > max_age_hours:
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
