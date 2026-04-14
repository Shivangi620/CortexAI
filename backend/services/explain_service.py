"""
services/explain_service.py
Feature 3: Enhanced SHAP explainability — global and local (per-prediction).
"""
from __future__ import annotations
import pandas as pd
import os
import json
import numpy as np
from typing import Dict, Any, List


def get_global_shap(job_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
    shap_summary = results.get("shap_summary", {})
    if not isinstance(shap_summary, dict) or not shap_summary:
        return {"error": "No SHAP data available. Re-train with Balanced or Full mode."}

    try:
        total = sum(abs(float(v)) for v in shap_summary.values()) or 1
    except Exception:
        total = 1

    ranked = []
    for feat, importance in sorted(shap_summary.items(), key=lambda x: abs(float(x[1])), reverse=True):
        try:
            val = float(importance)
        except Exception:
            val = 0.0

        ranked.append({
            "feature": feat,
            "importance": round(val, 6),
            "importance_pct": round(abs(val) / total * 100, 1),
        })

    return {
        "job_id": job_id,
        "feature_importance": ranked,
        "best_model": results.get("best_model"),
        "top_feature": ranked[0]["feature"] if ranked else None,
    }


def explain_local(
    job_id: str,
    results: Dict[str, Any],
    features: Dict[str, Any],
) -> Dict[str, Any]:
    import joblib

    try:
        import shap
    except Exception:
        shap = None

    from infra.storage import resolve_model_path

    model_path = resolve_model_path(job_id) or results.get("model_path")
    if not model_path or not os.path.exists(model_path):
        return {"error": "Model file not found"}

    expected_features: List[str] = results.get("feature_names", []) or []
    is_clf: bool = bool(results.get("is_classification", True))

    try:
        pipeline = joblib.load(model_path)
    except Exception as e:
        return {"error": f"Failed to load model: {e}"}

    try:
        if expected_features:
            missing = [f for f in expected_features if f not in features]
            if missing:
                return {"error": f"Missing features: {missing}"}
            row = {f: features.get(f) for f in expected_features}
        else:
            row = dict(features or {})

        input_df = pd.DataFrame([row])

        pred = pipeline.predict(input_df)[0]
        prediction = float(pred) if hasattr(pred, "item") else pred

        confidence = None
        if is_clf and hasattr(pipeline, "predict_proba"):
            try:
                proba = pipeline.predict_proba(input_df)
                confidence = round(float(max(proba[0])) * 100, 1)
            except Exception:
                confidence = None

        pre = None
        model = pipeline

        if hasattr(pipeline, "named_steps"):
            try:
                steps = list(pipeline.named_steps.keys())
                model = pipeline.named_steps[steps[-1]]
                if len(steps) > 1:
                    pre = pipeline.named_steps[steps[0]]
            except Exception:
                model = pipeline
                pre = None

        try:
            X_proc = pre.transform(input_df) if pre is not None else input_df.values
        except Exception:
            X_proc = input_df.values

        contributions = {}
        base_value = 0.0

        if shap is not None:
            try:
                explainer = shap.Explainer(model, X_proc)
                shap_vals = explainer(X_proc)

                raw_vals = getattr(shap_vals, "values", None)
                if raw_vals is not None:
                    if len(raw_vals.shape) == 3:
                        raw_vals = raw_vals[:, :, 0]
                    vals = raw_vals[0]

                    try:
                        feat_names = pre.get_feature_names_out() if pre else expected_features
                    except Exception:
                        feat_names = [f"f{i}" for i in range(len(vals))]

                    for i, name in enumerate(feat_names[:len(vals)]):
                        short = str(name).split("__")[-1]
                        try:
                            contributions[short] = round(float(vals[i]), 6)
                        except Exception:
                            contributions[short] = 0.0

                try:
                    base_val = getattr(shap_vals, "base_values", None)
                    if base_val is not None:
                        base_value = float(base_val[0])
                except Exception:
                    base_value = 0.0

            except Exception:
                contributions = dict(results.get("shap_summary") or {})
                base_value = 0.0
        else:
            contributions = dict(results.get("shap_summary") or {})
            base_value = 0.0

        return {
            "prediction": prediction,
            "confidence_pct": confidence,
            "base_value": round(base_value, 6),
            "feature_contributions": contributions,
            "input_features": row,
            "is_classification": is_clf,
        }

    except Exception as e:
        return {"error": str(e)}


def get_feature_lineage(job_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
    import joblib
    from infra.storage import resolve_model_path

    model_path = resolve_model_path(job_id) or results.get("model_path")
    if not model_path or not os.path.exists(model_path):
        return {"error": "Model file not found"}

    try:
        pipeline = joblib.load(model_path)
        preprocessor = pipeline.named_steps.get("preprocessor") if hasattr(pipeline, "named_steps") else None
        if preprocessor is None or not hasattr(preprocessor, "get_feature_names_out"):
            return {"error": "Preprocessor lineage not available"}

        transformed = preprocessor.get_feature_names_out()
        lineage = []
        for name in transformed:
            name_str = str(name)
            raw_name = name_str.split("__")[-1]
            lineage.append(
                {
                    "raw_feature": raw_name,
                    "transformed_feature": name_str,
                    "transform_group": name_str.split("__")[0] if "__" in name_str else "derived",
                }
            )
        return {"lineage": lineage, "count": len(lineage)}
    except Exception as e:
        return {"error": str(e)}


def get_calibration_report(job_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
    import joblib
    from sklearn.calibration import calibration_curve
    from sklearn.preprocessing import LabelEncoder
    from infra.database import get_db, JobModel, DatasetModel
    from infra.storage import resolve_model_path
    from core.file_loader import load_dataframe

    if not results.get("is_classification"):
        return {"error": "Calibration report is only available for classification models."}

    model_path = resolve_model_path(job_id) or results.get("model_path")
    if not model_path or not os.path.exists(model_path):
        return {"error": "Model file not found"}

    try:
        pipeline = joblib.load(model_path)
    except Exception as e:
        return {"error": f"Failed to load model: {e}"}

    try:
        with get_db() as db:
            job = db.query(JobModel).filter(JobModel.id == job_id).first()
            dataset = db.query(DatasetModel).filter(DatasetModel.id == job.dataset_id).first() if job else None
            profile = json.loads(dataset.profile_json) if dataset and dataset.profile_json else {}
            df = load_dataframe(filepath=dataset.file_path) if dataset and dataset.file_path else None
    except Exception as e:
        return {"error": f"Failed to load dataset: {e}"}

    if df is None or df.empty:
        return {"error": "Dataset not available for calibration analysis"}

    target = results.get("target") or profile.get("suggested_target")
    feature_names = [f for f in (results.get("feature_names") or []) if f in df.columns]
    if not target or target not in df.columns or not feature_names:
        return {"error": "Missing target or feature metadata"}

    work_df = df[feature_names + [target]].dropna(subset=[target]).copy()
    if len(work_df) < 10:
        return {"error": "Not enough labeled data for calibration"}

    try:
        y = LabelEncoder().fit_transform(work_df[target].astype(str))
        if len(set(y)) != 2 or not hasattr(pipeline, "predict_proba"):
            return {"error": "Calibration report currently supports binary classifiers with probabilities."}
        probs = pipeline.predict_proba(work_df[feature_names])[:, 1]
        frac_pos, mean_pred = calibration_curve(y, probs, n_bins=8, strategy="uniform")
        brier = float(np.mean((probs - y) ** 2))
        return {
            "brier_score": round(brier, 4),
            "bins": [
                {
                    "mean_predicted": round(float(mp), 4),
                    "fraction_positive": round(float(fp), 4),
                }
                for mp, fp in zip(mean_pred, frac_pos)
            ],
        }
    except Exception as e:
        return {"error": str(e)}


def get_threshold_tuning(job_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
    import joblib
    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.preprocessing import LabelEncoder
    from infra.database import get_db, JobModel, DatasetModel
    from infra.storage import resolve_model_path
    from core.file_loader import load_dataframe

    if not results.get("is_classification"):
        return {"error": "Threshold tuning is only available for classification models."}

    model_path = resolve_model_path(job_id) or results.get("model_path")
    if not model_path or not os.path.exists(model_path):
        return {"error": "Model file not found"}

    try:
        pipeline = joblib.load(model_path)
    except Exception as e:
        return {"error": f"Failed to load model: {e}"}

    try:
        with get_db() as db:
            job = db.query(JobModel).filter(JobModel.id == job_id).first()
            dataset = db.query(DatasetModel).filter(DatasetModel.id == job.dataset_id).first() if job else None
            profile = json.loads(dataset.profile_json) if dataset and dataset.profile_json else {}
            df = load_dataframe(filepath=dataset.file_path) if dataset and dataset.file_path else None
    except Exception as e:
        return {"error": f"Failed to load dataset: {e}"}

    if df is None or df.empty:
        return {"error": "Dataset not available for threshold tuning"}

    target = results.get("target") or profile.get("suggested_target")
    feature_names = [f for f in (results.get("feature_names") or []) if f in df.columns]
    if not target or target not in df.columns or not feature_names:
        return {"error": "Missing target or feature metadata"}

    work_df = df[feature_names + [target]].dropna(subset=[target]).copy()
    if len(work_df) < 10:
        return {"error": "Not enough labeled data for threshold tuning"}

    try:
        y = LabelEncoder().fit_transform(work_df[target].astype(str))
        if len(set(y)) != 2 or not hasattr(pipeline, "predict_proba"):
            return {"error": "Threshold tuning currently supports binary classifiers with probabilities."}
        probs = pipeline.predict_proba(work_df[feature_names])[:, 1]

        rows = []
        best = None
        for threshold in [round(x, 2) for x in np.arange(0.1, 0.91, 0.05)]:
            preds = (probs >= threshold).astype(int)
            row = {
                "threshold": threshold,
                "precision": round(float(precision_score(y, preds, zero_division=0)) * 100, 2),
                "recall": round(float(recall_score(y, preds, zero_division=0)) * 100, 2),
                "f1": round(float(f1_score(y, preds, zero_division=0)) * 100, 2),
            }
            rows.append(row)
            if best is None or row["f1"] > best["f1"]:
                best = row
        return {"best_threshold": best, "thresholds": rows}
    except Exception as e:
        return {"error": str(e)}
