"""api/routes/explain.py — SHAP explainability endpoints (Feature 3)."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import json

from infra.database import get_db, JobModel
from infra.result_contract import normalize_results

router = APIRouter(prefix="/api", tags=["explainability"])


class ExplainRequest(BaseModel):
    features: Dict[str, Any]


@router.get("/shap/{job_id}")
def get_shap_summary(job_id: str):
    """Global SHAP feature importance for a completed job."""
    with get_db() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job or not job.results_json:
            raise HTTPException(status_code=404, detail="Job not found or not completed")

        try:
            results = normalize_results(json.loads(job.results_json))
        except Exception:
            results = {}

    from services.explain_service import get_global_shap
    return get_global_shap(job_id, results)


@router.post("/explain/{job_id}")
def explain_prediction(job_id: str, req: ExplainRequest):
    """
    Feature 3: Per-prediction local SHAP explanation.
    Returns feature contributions, base value, and the prediction.
    """
    with get_db() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job or job.status != "completed":
            raise HTTPException(status_code=404, detail="Job not completed")

        try:
            results = normalize_results(json.loads(job.results_json) if job.results_json else {})
        except Exception:
            results = {}

    from services.explain_service import explain_local
    return explain_local(job_id, results, req.features)


@router.get("/pipeline/{job_id}")
def pipeline_graph(job_id: str):
    with get_db() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job or not job.results_json:
            return {"error": "Job not completed"}

        from infra.database import DatasetModel

        dataset = db.query(DatasetModel).filter(DatasetModel.id == job.dataset_id).first()

        try:
            profile = json.loads(dataset.profile_json) if dataset and dataset.profile_json else {}
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
    with get_db() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job or not job.results_json:
            return {"error": "Job not completed"}
        try:
            results = normalize_results(json.loads(job.results_json))
        except Exception:
            results = {}

    from services.explain_service import get_feature_lineage
    return get_feature_lineage(job_id, results)


@router.get("/calibration/{job_id}")
def calibration_report(job_id: str):
    with get_db() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job or not job.results_json:
            return {"error": "Job not completed"}
        try:
            results = normalize_results(json.loads(job.results_json))
        except Exception:
            results = {}

    from services.explain_service import get_calibration_report
    return get_calibration_report(job_id, results)


@router.get("/thresholds/{job_id}")
def threshold_report(job_id: str):
    with get_db() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job or not job.results_json:
            return {"error": "Job not completed"}
        try:
            results = normalize_results(json.loads(job.results_json))
        except Exception:
            results = {}

    from services.explain_service import get_threshold_tuning
    return get_threshold_tuning(job_id, results)
