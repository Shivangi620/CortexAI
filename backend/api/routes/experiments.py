"""api/routes/experiments.py — Experiment tracking dashboard (Feature 1)."""
from fastapi import APIRouter, Query
from typing import Optional
import json
import math
from pydantic import BaseModel

from infra.database import get_db, ExperimentRun, ModelRegistryEntry, TeamNote
from infra.result_contract import sanitize_for_json
from services.studio_service import experiment_diff

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
                "score": _safe_float(r.score),
                "task_type": r.task_type,
                "mode": r.mode,
                "goal": r.goal,
                "feature_count": r.feature_count,
                "row_count": r.row_count,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "hyperparams": sanitize_for_json(hyperparams),
                "metrics": sanitize_for_json(metrics),
                "leaderboard": sanitize_for_json(leaderboard),
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
                "hyperparams": sanitize_for_json(hyperparams),
                "metrics": sanitize_for_json(metrics),
            })

    return {"comparison": comparison, "count": len(comparison)}


@router.get("/experiments/diff")
def diff_experiments(run_a: str = Query(...), run_b: str = Query(...)):
    return experiment_diff(run_a, run_b)


def _safe_float(value):
    try:
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
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
            "score": _safe_float(r.score),
            "task_type": r.task_type,
            "mode": r.mode,
            "goal": r.goal,
            "feature_count": r.feature_count,
            "row_count": r.row_count,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "hyperparams": sanitize_for_json(hyperparams),
            "metrics": sanitize_for_json(metrics),
            "leaderboard": sanitize_for_json(leaderboard),
        }


@router.get("/experiments/{run_id}/registry")
def get_registry(run_id: str):
    with get_db() as db:
        row = db.query(ModelRegistryEntry).filter(ModelRegistryEntry.run_id == run_id).first()
        if not row:
            return {"run_id": run_id, "label": None, "note": None}
        return {"run_id": run_id, "label": row.label, "note": row.note}


class RegistryRequest(BaseModel):
    label: str
    note: Optional[str] = None


@router.post("/experiments/{run_id}/registry")
def save_registry(run_id: str, req: RegistryRequest):
    with get_db() as db:
        row = db.query(ModelRegistryEntry).filter(ModelRegistryEntry.run_id == run_id).first()
        if row:
            row.label = req.label
            row.note = req.note
        else:
            db.add(ModelRegistryEntry(run_id=run_id, label=req.label, note=req.note))
    return {"run_id": run_id, "label": req.label, "note": req.note}


@router.get("/notes/{entity_type}/{entity_id}")
def get_notes(entity_type: str, entity_id: str):
    with get_db() as db:
        rows = (
            db.query(TeamNote)
            .filter(TeamNote.entity_type == entity_type, TeamNote.entity_id == entity_id)
            .order_by(TeamNote.created_at.desc())
            .all()
        )
        return {
            "notes": [
                {
                    "id": row.id,
                    "note": row.note,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]
        }


class TeamNoteRequest(BaseModel):
    note: str


@router.post("/notes/{entity_type}/{entity_id}")
def add_note(entity_type: str, entity_id: str, req: TeamNoteRequest):
    text = (req.note or "").strip()
    if not text:
        return {"error": "Note cannot be empty."}
    with get_db() as db:
        db.add(TeamNote(entity_type=entity_type, entity_id=entity_id, note=text))
    return {"ok": True}
