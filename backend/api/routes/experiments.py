"""api/routes/experiments.py — Experiment tracking dashboard (Feature 1)."""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import json
import math
from pydantic import BaseModel

from infra.database import db_session, ExperimentRun, TeamNote, WorkspaceModel, JobModel
from infra.launch_origin import parse_launch_origin
from infra.result_contract import sanitize_for_json
from services.studio_service import experiment_diff, list_notifications

router = APIRouter(prefix="/api", tags=["experiments"])


def _safe_float(value):
    try:
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    except (TypeError, ValueError):
        return None

def _resolve_experiment_runs(db, raw_ids):
    cleaned_ids = [item.strip() for item in raw_ids if item and item.strip()]
    if not cleaned_ids:
        raise HTTPException(status_code=422, detail="At least one run identifier is required.")

    rows = db.query(ExperimentRun).filter(ExperimentRun.id.in_(cleaned_ids)).all()
    found_ids = {row.id for row in rows}
    missing = [item for item in cleaned_ids if item not in found_ids]

    if missing:
        job_rows = db.query(ExperimentRun).filter(ExperimentRun.job_id.in_(missing)).all()
        for row in job_rows:
            if row.id not in found_ids:
                rows.append(row)
                found_ids.add(row.id)

    unresolved = [item for item in cleaned_ids if item not in found_ids and item not in {row.job_id for row in rows}]
    return rows, unresolved


@router.get("/experiments")
def list_experiments(limit: int = 50, task_type: Optional[str] = None):
    """Return all experiment runs, most recent first."""
    with db_session() as db:
        q = db.query(ExperimentRun).order_by(ExperimentRun.created_at.desc())
        if task_type:
            q = q.filter(ExperimentRun.task_type == task_type)
        runs = q.limit(limit).all()
        job_ids = [run.job_id for run in runs if run.job_id]
        job_rows = db.query(JobModel).filter(JobModel.id.in_(job_ids)).all() if job_ids else []
        job_params_map = {}
        for job in job_rows:
            try:
                job_params_map[job.id] = json.loads(job.params_json) if job.params_json else {}
            except Exception:
                job_params_map[job.id] = {}

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
            launch_origin = parse_launch_origin(job_params_map.get(r.job_id))

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
                "launch_source": launch_origin["launch_source"],
                "launch_label": launch_origin["launch_label"],
                "hyperparams": sanitize_for_json(hyperparams),
                "metrics": sanitize_for_json(metrics),
                "leaderboard": sanitize_for_json(leaderboard),
            })
        return result


@router.get("/experiments/compare")
def compare_experiments(ids: str = Query(..., description="Comma-separated experiment run IDs")):
    """Side-by-side comparison of multiple experiment runs."""
    id_list = [i.strip() for i in ids.split(",") if i.strip()]

    with db_session() as db:
        runs, unresolved = _resolve_experiment_runs(db, id_list)
        if not runs:
            raise HTTPException(status_code=404, detail="No matching experiment runs were found.")
        job_ids = [run.job_id for run in runs if run.job_id]
        job_rows = db.query(JobModel).filter(JobModel.id.in_(job_ids)).all() if job_ids else []
        job_params_map = {}
        for job in job_rows:
            try:
                job_params_map[job.id] = json.loads(job.params_json) if job.params_json else {}
            except Exception:
                job_params_map[job.id] = {}

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
            launch_origin = parse_launch_origin(job_params_map.get(r.job_id))

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
                "launch_source": launch_origin["launch_source"],
                "launch_label": launch_origin["launch_label"],
                "hyperparams": sanitize_for_json(hyperparams),
                "metrics": sanitize_for_json(metrics),
            })

    return {"comparison": comparison, "count": len(comparison), "unresolved": unresolved}


@router.get("/experiments/diff")
def diff_experiments(run_a: str = Query(...), run_b: str = Query(...)):
    return experiment_diff(run_a, run_b)


@router.get("/experiments/{run_id}")
def get_experiment(run_id: str):
    """Get a single experiment run by ID."""
    with db_session() as db:
        r = db.query(ExperimentRun).filter(ExperimentRun.id == run_id).first()
        if not r:
            raise HTTPException(status_code=404, detail="Experiment not found")
        job = db.query(JobModel).filter(JobModel.id == r.job_id).first() if r.job_id else None
        try:
            job_params = json.loads(job.params_json) if job and job.params_json else {}
        except Exception:
            job_params = {}
        launch_origin = parse_launch_origin(job_params)

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
            "launch_source": launch_origin["launch_source"],
            "launch_label": launch_origin["launch_label"],
            "hyperparams": sanitize_for_json(hyperparams),
            "metrics": sanitize_for_json(metrics),
            "leaderboard": sanitize_for_json(leaderboard),
        }


@router.get("/notes/{entity_type}/{entity_id}")
def get_notes(entity_type: str, entity_id: str):
    with db_session() as db:
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
        raise HTTPException(status_code=422, detail="Note cannot be empty.")
    with db_session() as db:
        db.add(TeamNote(entity_type=entity_type, entity_id=entity_id, note=text))
        db.commit()
    return {"ok": True}


class WorkspaceRequest(BaseModel):
    name: str
    dataset_id: Optional[str] = None
    last_job_id: Optional[str] = None
    settings: dict = {}


@router.get("/workspaces")
def list_workspaces():
    with db_session() as db:
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
        raise HTTPException(status_code=422, detail="Workspace name is required")
    with db_session() as db:
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
    with db_session() as db:
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
            raise HTTPException(status_code=404, detail="No completed run found")
        return {"job_id": latest.id, "dataset_id": latest.dataset_id}


@router.get("/notifications")
def get_notifications():
    return list_notifications()
