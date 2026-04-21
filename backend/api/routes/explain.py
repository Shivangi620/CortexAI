"""api/routes/explain.py — SHAP explainability endpoints (Feature 3)."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json

from infra.database import get_db, db_session, JobModel
from infra.result_contract import normalize_results

router = APIRouter(prefix="/api", tags=["explainability"])


class ExplainRequest(BaseModel):
    features: Optional[Dict[str, Any]] = None
    payload: Optional[Dict[str, Any]] = None
    target_prediction: Optional[float] = None


@router.get("/shap/{job_id}")
def get_shap_summary(job_id: str):
    """Global SHAP feature importance for a completed job."""
    with db_session() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if not job.results_json:
            return {
                "status": "pending",
                "message": f"Job is {job.status}. Results not yet available.",
            }

        try:
            results = normalize_results(json.loads(job.results_json))
        except Exception:
            results = {}

    from services.explain_service import get_global_shap

    return get_global_shap(job_id, results)


@router.get("/permutation/{job_id}")
def get_permutation_summary(job_id: str):
    with db_session() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if not job.results_json:
            return {
                "status": "pending",
                "message": f"Job is {job.status}. Results not yet available.",
            }

        try:
            results = normalize_results(json.loads(job.results_json))
        except Exception:
            results = {}

    ranking = results.get("permutation_importance") or {}
    return {
        "job_id": job_id,
        "feature_importance": [
            {"feature": key, "importance": value} for key, value in ranking.items()
        ],
    }


@router.post("/explain/{job_id}")
def explain_prediction(job_id: str, req: ExplainRequest):
    """
    Feature 3: Per-prediction local SHAP explanation.
    Returns feature contributions, base value, and the prediction.
    """
    with db_session() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != "completed":
            return {
                "status": "pending",
                "message": f"Job is {job.status}. Results not yet available.",
            }

        try:
            results = normalize_results(
                json.loads(job.results_json) if job.results_json else {}
            )
        except Exception:
            results = {}

    if not req.features:
        return {
            "error": "Invalid request",
            "message": "Request must include 'features' dictionary with feature values.",
        }

    from services.explain_service import explain_local

    return explain_local(job_id, results, req.features)


@router.get("/pipeline/{job_id}")
def pipeline_graph(job_id: str):
    with db_session() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if not job.results_json:
            return {
                "status": "pending",
                "message": f"Job is {job.status}. Results not yet available.",
            }

        from infra.database import DatasetModel

        dataset = (
            db.query(DatasetModel).filter(DatasetModel.id == job.dataset_id).first()
        )

        try:
            profile = (
                json.loads(dataset.profile_json)
                if dataset and dataset.profile_json
                else {}
            )
        except Exception:
            profile = {}

        try:
            results = normalize_results(json.loads(job.results_json))
        except Exception:
            results = {}

    from core.debugger import generate_pipeline_graph

    return {"mermaid": generate_pipeline_graph(profile, results)}


@router.get("/lineage/{job_id}")
def feature_lineage(job_id: str):
    with db_session() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if not job.results_json:
            return {
                "status": "pending",
                "message": f"Job is {job.status}. Results not yet available.",
            }
        try:
            results = normalize_results(json.loads(job.results_json))
        except Exception:
            results = {}

    from services.explain_service import get_feature_lineage

    return get_feature_lineage(job_id, results)


@router.get("/calibration/{job_id}")
def calibration_report(job_id: str):
    with db_session() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if not job.results_json:
            return {
                "status": "pending",
                "message": f"Job is {job.status}. Results not yet available.",
            }
        try:
            results = normalize_results(json.loads(job.results_json))
        except Exception:
            results = {}

    from services.explain_service import get_calibration_report

    return get_calibration_report(job_id, results)


@router.get("/thresholds/{job_id}")
def threshold_report(job_id: str):
    with db_session() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if not job.results_json:
            return {
                "status": "pending",
                "message": f"Job is {job.status}. Results not yet available.",
            }
        try:
            results = normalize_results(json.loads(job.results_json))
        except Exception:
            results = {}

    from services.explain_service import get_threshold_tuning

    return get_threshold_tuning(job_id, results)


@router.post("/counterfactual/{job_id}")
def counterfactual(job_id: str, req: ExplainRequest):
    with db_session() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if not job.results_json:
            return {
                "status": "pending",
                "message": f"Job is {job.status}. Results not yet available.",
            }
        try:
            results = normalize_results(json.loads(job.results_json))
        except Exception:
            results = {}

    features = req.features or req.payload
    if not features:
        return {
            "error": "Invalid request",
            "message": "Request must include a feature dictionary in 'features' or 'payload'.",
        }

    from services.explain_service import generate_counterfactual

    return generate_counterfactual(
        job_id,
        results,
        features,
        target_prediction=req.target_prediction,
    )


@router.get("/trust/{job_id}")
def trust_heatmap(job_id: str):
    with db_session() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if not job.results_json:
            return {
                "status": "pending",
                "message": f"Job is {job.status}. Results not yet available.",
            }
        dataset_id = job.dataset_id
        try:
            results = normalize_results(json.loads(job.results_json))
        except Exception:
            results = {}

    from services.studio_service import build_trust_heatmap

    return build_trust_heatmap(dataset_id, results)
