"""api/routes/experiments.py — Experiment tracking dashboard (Feature 1)."""
from fastapi import APIRouter, Query
from typing import Optional
import json
import math
from pydantic import BaseModel

from infra.database import get_db, ExperimentRun, ModelRegistryEntry, TeamNote, WorkspaceModel, JobModel
from infra.result_contract import sanitize_for_json
from services.studio_service import experiment_diff, list_notifications

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
                "dataset_name": r.dataset_name,
                "workspace_id": r.workspace_id,
                "workspace_name": r.workspace_name,
                "model_name": r.model_name,
                "metric_name": r.metric_name,
                "score": _safe_float(r.score),
                "task_type": r.task_type,
                "mode": r.mode,
                "goal": r.goal,
                "feature_count": r.feature_count,
                "row_count": r.row_count,
                "preset_name": r.preset_name,
                "summary_text": r.summary_text,
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
                "dataset_name": r.dataset_name,
                "workspace_name": r.workspace_name,
                "model_name": r.model_name,
                "metric_name": r.metric_name,
                "score": _safe_float(r.score),
                "task_type": r.task_type,
                "mode": r.mode,
                "goal": r.goal,
                "feature_count": r.feature_count,
                "row_count": r.row_count,
                "preset_name": r.preset_name,
                "summary_text": r.summary_text,
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
            "dataset_id": r.dataset_id,
            "dataset_name": r.dataset_name,
            "workspace_id": r.workspace_id,
            "workspace_name": r.workspace_name,
            "model_name": r.model_name,
            "metric_name": r.metric_name,
            "score": _safe_float(r.score),
            "task_type": r.task_type,
            "mode": r.mode,
            "goal": r.goal,
            "feature_count": r.feature_count,
            "row_count": r.row_count,
            "preset_name": r.preset_name,
            "summary_text": r.summary_text,
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


class WorkspaceRequest(BaseModel):
    name: str
    dataset_id: Optional[str] = None
    last_job_id: Optional[str] = None
    settings: dict = {}


@router.get("/workspaces")
def list_workspaces():
    with get_db() as db:
        rows = db.query(WorkspaceModel).order_by(WorkspaceModel.updated_at.desc()).all()
        return {
            "workspaces": [
                {
                    "id": row.id,
                    "name": row.name,
                    "dataset_id": row.dataset_id,
                    "last_job_id": row.last_job_id,
                    "last_run_id": row.last_run_id,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                }
                for row in rows
            ]
        }


@router.post("/workspaces")
def create_workspace(req: WorkspaceRequest):
    name = (req.name or "").strip()
    if not name:
        return {"error": "Workspace name is required"}
    with get_db() as db:
        row = db.query(WorkspaceModel).filter(WorkspaceModel.name == name).first()
        if row:
            row.dataset_id = req.dataset_id or row.dataset_id
            row.last_job_id = req.last_job_id or row.last_job_id
            row.settings_json = json.dumps(req.settings or {})
        else:
            row = WorkspaceModel(
                name=name,
                dataset_id=req.dataset_id,
                last_job_id=req.last_job_id,
                settings_json=json.dumps(req.settings or {}),
                reports_json=json.dumps({}),
            )
            db.add(row)
        db.commit()
        return {"id": row.id, "name": row.name}


@router.get("/workspaces/resume")
def resume_last_completed_run(workspace_id: Optional[str] = None):
    with get_db() as db:
        if workspace_id:
            workspace = db.query(WorkspaceModel).filter(WorkspaceModel.id == workspace_id).first()
            if workspace and workspace.last_job_id:
                job = db.query(JobModel).filter(JobModel.id == workspace.last_job_id, JobModel.status == "completed").first()
                if job:
                    return {
                        "workspace_id": workspace.id,
                        "workspace_name": workspace.name,
                        "job_id": job.id,
                        "dataset_id": job.dataset_id,
                    }
        latest = (
            db.query(JobModel)
            .filter(JobModel.status == "completed")
            .order_by(JobModel.created_at.desc())
            .first()
        )
        if not latest:
            return {"error": "No completed run found"}
        return {"job_id": latest.id, "dataset_id": latest.dataset_id}


@router.get("/notifications")
def get_notifications():
    return list_notifications()
