"""api/routes/experiments.py — Experiment tracking dashboard (Feature 1)."""
from fastapi import APIRouter, Query
from typing import Optional
import json

from infra.database import get_db, ExperimentRun

router = APIRouter(prefix="/api", tags=["experiments"])


@router.get("/experiments")
def list_experiments(limit: int = 50, task_type: Optional[str] = None):
    """Return all experiment runs, most recent first."""
    with get_db() as db:
        q = db.query(ExperimentRun).order_by(ExperimentRun.created_at.desc())
        if task_type:
            q = q.filter(ExperimentRun.task_type == task_type)
        runs = q.limit(limit).all()

        result = []
        for r in runs:
            try:
                hyperparams = json.loads(r.hyperparams_json) if r.hyperparams_json else {}
            except Exception:
                hyperparams = {}

            try:
                metrics = json.loads(r.metrics_json) if r.metrics_json else {}
            except Exception:
                metrics = {}

            try:
                leaderboard = json.loads(r.leaderboard_json) if r.leaderboard_json else []
            except Exception:
                leaderboard = []

            result.append({
                "id": r.id,
                "job_id": r.job_id,
                "dataset_id": r.dataset_id,
                "model_name": r.model_name,
                "metric_name": r.metric_name,
                "score": r.score,
                "task_type": r.task_type,
                "mode": r.mode,
                "goal": r.goal,
                "feature_count": r.feature_count,
                "row_count": r.row_count,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "hyperparams": hyperparams,
                "metrics": metrics,
                "leaderboard": leaderboard,
            })
        return result


@router.get("/experiments/compare")
def compare_experiments(ids: str = Query(..., description="Comma-separated experiment run IDs")):
    """Side-by-side comparison of multiple experiment runs."""
    id_list = [i.strip() for i in ids.split(",") if i.strip()]

    with get_db() as db:
        runs = db.query(ExperimentRun).filter(ExperimentRun.id.in_(id_list)).all()

        comparison = []
        for r in runs:
            try:
                hyperparams = json.loads(r.hyperparams_json) if r.hyperparams_json else {}
            except Exception:
                hyperparams = {}

            try:
                metrics = json.loads(r.metrics_json) if r.metrics_json else {}
            except Exception:
                metrics = {}

            comparison.append({
                "id": r.id,
                "job_id": r.job_id,
                "model_name": r.model_name,
                "metric_name": r.metric_name,
                "score": _safe_float(r.score),
                "task_type": r.task_type,
                "mode": r.mode,
                "goal": r.goal,
                "feature_count": r.feature_count,
                "row_count": r.row_count,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "hyperparams": hyperparams,
                "metrics": metrics,
            })

    return {"comparison": comparison, "count": len(comparison)}


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@router.get("/experiments/{run_id}")
def get_experiment(run_id: str):
    """Get a single experiment run by ID."""
    with get_db() as db:
        r = db.query(ExperimentRun).filter(ExperimentRun.id == run_id).first()
        if not r:
            return {"error": "Experiment not found"}

        try:
            hyperparams = json.loads(r.hyperparams_json) if r.hyperparams_json else {}
        except Exception:
            hyperparams = {}

        try:
            metrics = json.loads(r.metrics_json) if r.metrics_json else {}
        except Exception:
            metrics = {}

        try:
            leaderboard = json.loads(r.leaderboard_json) if r.leaderboard_json else []
        except Exception:
            leaderboard = []

        return {
            "id": r.id,
            "job_id": r.job_id,
            "model_name": r.model_name,
            "metric_name": r.metric_name,
            "score": r.score,
            "task_type": r.task_type,
            "mode": r.mode,
            "goal": r.goal,
            "feature_count": r.feature_count,
            "row_count": r.row_count,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "hyperparams": hyperparams,
            "metrics": metrics,
            "leaderboard": leaderboard,
        }
