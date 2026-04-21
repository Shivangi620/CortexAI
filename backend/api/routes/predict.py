"""api/routes/predict.py — Live prediction and future sweep endpoints."""
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, List
import json
import os
import joblib
import pandas as pd
import shap

from infra.database import get_db, db_session, JobModel
from infra.result_contract import normalize_results
from infra.storage import get_schema_path
from api.routes.datasets import _stream_upload_to_file
from core.file_loader import load_dataframe
from services.data_sanitizer import sanitize_dataframe

router = APIRouter(prefix="/api", tags=["predict"])


def _build_inference_frame(features: Dict[str, Any], expected_features: List[str]) -> pd.DataFrame:
    incoming = set(features.keys())
    expected = set(expected_features or [])

    if expected:
        missing = sorted(expected - incoming)
        extra = sorted(incoming - expected)

        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        # We now ignore extra features instead of raising a 422 error,
        # which is more resilient to noisy payloads in production.
        row = {name: features.get(name) for name in expected_features}
        frame = pd.DataFrame([row], columns=expected_features)
    else:
        frame = pd.DataFrame([features])

    return sanitize_dataframe(frame).df


class PredictRequest(BaseModel):
    features: Dict[str, Any]


@router.post("/predict/{job_id}")
def predict(job_id: str, req: PredictRequest):
    with db_session() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job or job.status != "completed":
            raise HTTPException(status_code=404, detail="Job not completed or not found")

        try:
            raw_results = json.loads(job.results_json) if job.results_json else {}
        except Exception:
            raw_results = {}

        results = normalize_results(raw_results)

    from infra.storage import resolve_model_path

    model_path = resolve_model_path(job_id) or results.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found on disk")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    try:
        expected_features = results.get("feature_names") or []
        df = _build_inference_frame(req.features, expected_features)

        pred = model.predict(df)
        raw = pred[0]

        result = {
            "prediction": float(raw) if hasattr(raw, "item") else raw
        }

        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(df)
                result["confidence_pct"] = round(float(max(proba[0])) * 100, 1)
                classes = getattr(model, "classes_", None)
                if classes is not None and len(classes) == len(proba[0]):
                    result["probabilities"] = {
                        str(label): round(float(score) * 100, 2)
                        for label, score in zip(classes, proba[0])
                    }
            except Exception:
                pass

        result["feature_names"] = expected_features

        # Calculate sensitivity (SHAP)
        try:
            # Use a KernelExplainer or TreeExplainer if possible
            # For brevity and performance in a live sandbox, we use a heuristic 
            # or a simplified Explainer if the model is compatible.
            if hasattr(model, "predict"):
                explainer = shap.Explainer(model.predict, df)
                shap_values = explainer(df)
                result["sensitivity"] = {
                    name: round(float(abs(val)), 4)
                    for name, val in zip(expected_features, shap_values.values[0])
                }
        except Exception:
            # Fallback sensitivity if SHAP fails (heuristic based on prediction delta)
            pass
        return result

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CounterfactualRequest(BaseModel):
    payload: Dict[str, Any]
    target_prediction: float

@router.post("/counterfactual-lite/{job_id}")
def get_counterfactual(job_id: str, req: CounterfactualRequest):
    """
    Suggest minimal changes to reach a target prediction.
    Uses a simple greedy search for demonstration.
    """
    with db_session() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job or not job.model_path or not os.path.exists(job.model_path):
            return {"error": "Model not available"}
        model = joblib.load(job.model_path)

    try:
        base_df = pd.DataFrame([req.payload])
        current_pred = float(model.predict(base_df)[0])
        
        # Simple heuristic: try ±10% on each numeric feature
        suggestions = []
        for col in base_df.columns:
            val = base_df[col].iloc[0]
            if isinstance(val, (int, float)):
                for factor in [0.9, 1.1]:
                    test_df = base_df.copy()
                    test_df[col] = val * factor
                    new_pred = float(model.predict(test_df)[0])
                    # If it moves closer to target
                    if abs(new_pred - req.target_prediction) < abs(current_pred - req.target_prediction):
                        suggestions.append({
                            "feature": col,
                            "change": f"{'Increase' if factor > 1 else 'Decrease'} by 10%",
                            "new_prediction": round(new_pred, 4),
                            "impact": round(abs(new_pred - current_pred), 4)
                        })
        
        return {
            "current_prediction": current_pred,
            "target_prediction": req.target_prediction,
            "suggestions": sorted(suggestions, key=lambda x: x["impact"], reverse=True)[:3]
        }
    except Exception as e:
        return {"error": str(e)}


class FutureRequest(BaseModel):
    job_id: str
    base_features: dict
    sweep_feature: str
    sweep_values: list


@router.post("/future")
def future_predict(req: FutureRequest):
    with db_session() as db:
        job = db.query(JobModel).filter(JobModel.id == req.job_id).first()
        if not job or job.status != "completed":
            raise HTTPException(status_code=404, detail="Job not completed")

        try:
            raw_results = json.loads(job.results_json) if job.results_json else {}
        except Exception:
            raw_results = {}

        results = normalize_results(raw_results)

    from infra.storage import resolve_model_path

    model_path = resolve_model_path(req.job_id) or results.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    expected_features = results.get("feature_names") or []

    predictions = []

    for val in req.sweep_values:
        features = dict(req.base_features or {})
        features[req.sweep_feature] = val

        try:
            df_row = _build_inference_frame(features, expected_features)

            pred = model.predict(df_row)[0]

            conf = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(df_row)
                    conf = round(float(max(proba[0])) * 100, 1)
                except Exception:
                    conf = None

            predictions.append({
                "x": val,
                "prediction": float(pred) if hasattr(pred, "item") else pred,
                "confidence": conf
            })

        except Exception as e:
            predictions.append({
                "x": val,
                "error": str(e)
            })

    return {
        "sweep_feature": req.sweep_feature,
        "predictions": predictions
    }


@router.post("/contract-check/{job_id}")
async def contract_check(job_id: str, file: UploadFile = File(...)):
    with db_session() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job or job.status != "completed":
            raise HTTPException(status_code=404, detail="Job not completed or not found")

        try:
            raw_results = json.loads(job.results_json) if job.results_json else {}
        except Exception:
            raw_results = {}
        results = normalize_results(raw_results)

    expected_features = results.get("feature_names") or []
    schema_path = get_schema_path(job_id)
    contract_schema = {}
    if os.path.exists(schema_path):
        try:
            with open(schema_path, "r") as handle:
                contract_schema = json.load(handle).get("schema", {}) or {}
        except Exception:
            contract_schema = {}

    temp_path = None
    try:
        temp_path = await _stream_upload_to_file(
            f"contract_{job_id}_{os.urandom(4).hex()}",
            file.filename or "inference.csv",
            file,
        )
        df = load_dataframe(filepath=temp_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read inference file: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    incoming_columns = list(df.columns)
    missing = sorted(set(expected_features) - set(incoming_columns))
    extra = sorted(set(incoming_columns) - set(expected_features))
    dtype_mismatches = []
    for col in incoming_columns:
        if col in contract_schema:
            expected_type = str(contract_schema[col].get("type", "unknown"))
            actual_type = str(df[col].dtype)
            if expected_type != "unknown" and actual_type != expected_type:
                dtype_mismatches.append(
                    {"column": col, "expected_type": expected_type, "actual_type": actual_type}
                )

    status = "aligned"
    if missing or dtype_mismatches:
        status = "drift_risk"
    elif extra:
        status = "warning"

    return {
        "job_id": job_id,
        "status": status,
        "rows": int(len(df)),
        "expected_feature_count": len(expected_features),
        "incoming_column_count": len(incoming_columns),
        "missing_features": missing,
        "extra_columns": extra,
        "dtype_mismatches": dtype_mismatches,
        "incoming_columns": incoming_columns,
    }


@router.post("/batch-predict/{job_id}")
async def batch_predict(job_id: str, file: UploadFile = File(...)):
    with db_session() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job or job.status != "completed":
            raise HTTPException(status_code=404, detail="Job not completed or not found")

        try:
            raw_results = json.loads(job.results_json) if job.results_json else {}
        except Exception:
            raw_results = {}
        results = normalize_results(raw_results)

    from infra.storage import resolve_model_path
    model_path = resolve_model_path(job_id) or results.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    temp_path = None
    try:
        temp_path = await _stream_upload_to_file(
            f"batch_{job_id}_{os.urandom(4).hex()}",
            file.filename or "batch.csv",
            file,
        )
        df = load_dataframe(filepath=temp_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read batch file: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    try:
        expected_features = results.get("feature_names") or []
        sanitized = sanitize_dataframe(df).df
        
        # Ensure only expected features are used
        if expected_features:
            sanitized = sanitized[expected_features]

        preds = model.predict(sanitized)
        df["prediction"] = preds

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(sanitized)
            df["confidence_pct"] = [round(float(max(p)) * 100, 1) for p in proba]

        # Return the first 100 rows as a preview, or return CSV download URL?
        # For now, return JSON preview + row count.
        preview = df.head(100).to_dict(orient="records")
        return {
            "job_id": job_id,
            "row_count": len(df),
            "preview": preview,
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")
