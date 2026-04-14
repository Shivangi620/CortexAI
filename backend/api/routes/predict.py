"""api/routes/predict.py — Live prediction and future sweep endpoints."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import json
import os
import joblib
import pandas as pd

from infra.database import get_db, JobModel
from infra.result_contract import normalize_results

router = APIRouter(prefix="/api", tags=["predict"])


def _build_inference_frame(features: Dict[str, Any], expected_features: List[str]) -> pd.DataFrame:
    incoming = set(features.keys())
    expected = set(expected_features or [])

    if expected:
        missing = sorted(expected - incoming)
        extra = sorted(incoming - expected)

        if missing:
            raise ValueError(f"Missing required features: {missing}")
        if extra:
            raise ValueError(f"Unexpected features: {extra}")

        row = {name: features.get(name) for name in expected_features}
        return pd.DataFrame([row], columns=expected_features)

    return pd.DataFrame([features])


class PredictRequest(BaseModel):
    features: Dict[str, Any]


@router.post("/predict/{job_id}")
def predict(job_id: str, req: PredictRequest):
    with get_db() as db:
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
            except Exception:
                pass

        result["feature_names"] = expected_features
        return result

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class FutureRequest(BaseModel):
    job_id: str
    base_features: dict
    sweep_feature: str
    sweep_values: list


@router.post("/future")
def future_predict(req: FutureRequest):
    with get_db() as db:
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