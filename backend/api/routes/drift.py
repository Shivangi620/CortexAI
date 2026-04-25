import csv
from datetime import datetime, timedelta, UTC
import json
import os
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from infra.database import DriftCheck, DriftSchedule, DatasetModel, JobModel, db_session
from infra.result_contract import normalize_results

csv.field_size_limit(int(1e9))

router = APIRouter(prefix="/api", tags=["drift"])


def _resolve_retrain_launch_config(job_params: dict, goal_override: str = "", mode_override: str = "") -> dict:
    return {
        "goal": (goal_override or "").strip() or job_params.get("goal", "Balanced"),
        "mode": (mode_override or "").strip() or job_params.get("mode", "Balanced"),
    }


def _resolve_retrain_launch_context(
    launch_context_json: str = "",
    *,
    parent_job_id: str,
    launch_config: dict,
) -> dict:
    base_context = {
        "source": "drift_recommendation",
        "parent_job_id": parent_job_id,
        "recommended_goal": launch_config.get("goal", "Balanced"),
        "recommended_mode": launch_config.get("mode", "Balanced"),
    }
    try:
        payload = json.loads(launch_context_json) if launch_context_json else {}
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return {**base_context, **payload}


@router.post("/drift/{job_id}")
async def detect_drift(job_id: str, file: UploadFile = File(...)):
    """
    Feature 6: Drift dashboard.
    Upload a new CSV; compare its feature distributions against training baseline.
    """
    with db_session() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job or job.status != "completed":
            raise HTTPException(status_code=404, detail="Job not completed")
        schedule = db.query(DriftSchedule).filter(DriftSchedule.job_id == job_id).first()

        try:
            results = normalize_results(json.loads(job.results_json) if job.results_json else {})
        except Exception:
            results = {}
        try:
            params = json.loads(job.params_json) if job.params_json else {}
        except Exception:
            params = {}

    # Save uploaded file temporarily
    tmp_path = f"tmp/drift_{job_id}_{os.urandom(4).hex()}.csv"
    os.makedirs("tmp", exist_ok=True)

    try:
        with open(tmp_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    from core.file_loader import load_dataframe
    from services.drift_service import get_drift_dashboard

    try:
        current_df = load_dataframe(filepath=tmp_path)
        if current_df is None or current_df.empty:
            raise HTTPException(status_code=422, detail="Uploaded dataset is empty or unreadable")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read file: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    baseline_json = results.get("drift_baseline_json")
    baseline_stats = None

    if baseline_json:
        try:
            baseline_stats = json.loads(baseline_json)
        except Exception:
            baseline_stats = None

    if baseline_stats is None:
        from core.drift_detector import DriftDetector

        detector = DriftDetector(job_id)
        if os.path.exists(detector.baseline_path):
            try:
                with open(detector.baseline_path, "r") as f:
                    baseline_stats = json.load(f)
            except Exception:
                baseline_stats = None

    if baseline_stats is None:
        raise HTTPException(
            status_code=400,
            detail="No drift baseline found for this job. Re-train to generate one."
        )

    feature_names = results.get("feature_names") or []

    warning_threshold = None
    critical_threshold = None
    if schedule:
        try:
            warning_threshold = float(schedule.warning_threshold or 0.1)
        except Exception:
            warning_threshold = 0.1
        try:
            critical_threshold = float(schedule.critical_threshold or 0.2)
        except Exception:
            critical_threshold = 0.2

    report = get_drift_dashboard(
        current_df,
        baseline_stats,
        feature_names,
        target_name=results.get("target"),
        warning_threshold=warning_threshold,
        critical_threshold=critical_threshold,
        task_type="classification" if results.get("is_classification") else "regression",
        current_model=results.get("best_model") or "",
        metric_name=results.get("metric_name") or "",
        current_score=results.get("score"),
        current_validation_gap=(results.get("validation_summary") or {}).get("absolute_gap_display"),
        goal=params.get("goal") or "",
        mode=params.get("mode") or "",
    )

    try:
        with db_session() as db:
            job = db.query(JobModel).filter(JobModel.id == job_id).first()
            schedule_row = db.query(DriftSchedule).filter(DriftSchedule.job_id == job_id).first()
            db.add(
                DriftCheck(
                    job_id=job_id,
                    dataset_id=job.dataset_id if job else None,
                    uploaded_name=file.filename or "drift.csv",
                    status=report.get("overall_status"),
                    drift_score_pct=str(report.get("drift_score_pct", 0)),
                    report_json=json.dumps(report),
                )
            )
            if schedule_row:
                schedule_row.last_alert_status = report.get("alert_level")
                schedule_row.last_alert_summary = json.dumps(report.get("alert_summary") or {})
            db.commit()
    except Exception:
        pass

    return report


@router.get("/drift/{job_id}/history")
def drift_history(job_id: str):
    with db_session() as db:
        rows = (
            db.query(DriftCheck)
            .filter(DriftCheck.job_id == job_id)
            .order_by(DriftCheck.created_at.desc())
            .limit(50)
            .all()
        )

        history = []
        for row in rows:
            history.append(
                {
                    "id": row.id,
                    "uploaded_name": row.uploaded_name,
                    "status": row.status,
                    "drift_score_pct": float(row.drift_score_pct) if row.drift_score_pct else 0.0,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
            )
        return {"history": history}


@router.get("/drift/{job_id}/feature-timeline")
def drift_feature_timeline(job_id: str, feature: str | None = None):
    with db_session() as db:
        rows = (
            db.query(DriftCheck)
            .filter(DriftCheck.job_id == job_id)
            .order_by(DriftCheck.created_at.asc())
            .limit(100)
            .all()
        )

    timeline = []
    feature_set = set()
    for row in rows:
        try:
            report = json.loads(row.report_json) if row.report_json else {}
        except Exception:
            report = {}

        for item in report.get("feature_drift", []) or []:
            feature_name = item.get("feature")
            if not feature_name:
                continue
            feature_set.add(feature_name)
            if feature and feature_name != feature:
                continue
            timeline.append(
                {
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "uploaded_name": row.uploaded_name,
                    "feature": feature_name,
                    "psi": item.get("psi"),
                    "ks_p_value": item.get("ks_p_value"),
                    "severity": item.get("severity"),
                    "drift_detected": item.get("drift_detected", False),
                    "current_mean": item.get("current_mean"),
                    "baseline_mean": item.get("baseline_mean"),
                }
            )

    return {
        "job_id": job_id,
        "features": sorted(feature_set),
        "timeline": timeline,
    }


@router.get("/drift/{job_id}/schedule")
def get_drift_schedule(job_id: str):
    with db_session() as db:
        row = db.query(DriftSchedule).filter(DriftSchedule.job_id == job_id).first()
        latest_check = (
            db.query(DriftCheck)
            .filter(DriftCheck.job_id == job_id)
            .order_by(DriftCheck.created_at.desc())
            .first()
        )
        if not row:
            frequency_days = 7
            enabled = True
            warning_threshold = 0.1
            critical_threshold = 0.2
            last_alert_status = None
            last_alert_summary = {}
        else:
            frequency_days = int(row.frequency_days or 7)
            enabled = str(row.enabled).lower() == "true"
            try:
                warning_threshold = float(row.warning_threshold or 0.1)
            except Exception:
                warning_threshold = 0.1
            try:
                critical_threshold = float(row.critical_threshold or 0.2)
            except Exception:
                critical_threshold = 0.2
            last_alert_status = row.last_alert_status
            try:
                last_alert_summary = json.loads(row.last_alert_summary) if row.last_alert_summary else {}
            except Exception:
                last_alert_summary = {}

        next_due_at = None
        due_now = False
        if latest_check and latest_check.created_at:
            created_at = latest_check.created_at
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at)
                except Exception:
                    created_at = datetime.now(UTC)
            
            next_due = created_at + timedelta(days=frequency_days)
            next_due_at = next_due.isoformat()
            if created_at.tzinfo is None:
                next_due_cmp = next_due.replace(tzinfo=UTC)
            else:
                next_due_cmp = next_due.astimezone(UTC)
            due_now = enabled and next_due_cmp <= datetime.now(UTC)
        elif enabled:
            due_now = True

        return {
            "job_id": job_id,
            "enabled": enabled,
            "frequency_days": frequency_days,
            "last_check_at": latest_check.created_at.isoformat() if latest_check and latest_check.created_at else None,
            "next_due_at": next_due_at,
            "due_now": due_now,
            "warning_threshold": warning_threshold,
            "critical_threshold": critical_threshold,
            "last_alert_status": last_alert_status,
            "last_alert_summary": last_alert_summary,
        }


@router.post("/drift/{job_id}/schedule")
def save_drift_schedule(
    job_id: str,
    enabled: bool = True,
    frequency_days: int = 7,
    warning_threshold: float = 0.1,
    critical_threshold: float = 0.2,
):
    frequency_days = max(1, int(frequency_days or 7))
    warning_threshold = max(0.01, float(warning_threshold or 0.1))
    critical_threshold = max(warning_threshold, float(critical_threshold or 0.2))
    with db_session() as db:
        row = db.query(DriftSchedule).filter(DriftSchedule.job_id == job_id).first()
        if row:
            row.enabled = str(bool(enabled)).lower()
            row.frequency_days = str(frequency_days)
            row.warning_threshold = str(warning_threshold)
            row.critical_threshold = str(critical_threshold)
        else:
            db.add(
                DriftSchedule(
                    job_id=job_id,
                    enabled=str(bool(enabled)).lower(),
                    frequency_days=str(frequency_days),
                    warning_threshold=str(warning_threshold),
                    critical_threshold=str(critical_threshold),
                )
            )
        db.commit()
    return {
        "job_id": job_id,
        "enabled": enabled,
        "frequency_days": frequency_days,
        "warning_threshold": warning_threshold,
        "critical_threshold": critical_threshold,
    }


@router.post("/drift/{job_id}/retrain")
async def retrain_on_drift_dataset(
    job_id: str,
    file: UploadFile = File(...),
    goal_override: str = Form(""),
    mode_override: str = Form(""),
    launch_context_json: str = Form(""),
):
    from core.file_loader import load_dataframe
    from core.data_profiler import profile_dataset
    from api.routes.training import TrainRequest, start_training

    with db_session() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Original job not found")
        parent_dataset_id = job.dataset_id

        try:
            job_params = json.loads(job.params_json) if job.params_json else {}
        except Exception:
            job_params = {}

    new_dataset_id = str(uuid4())
    file_path = f"tmp/{new_dataset_id}.csv"
    os.makedirs("tmp", exist_ok=True)

    try:
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)
        df = load_dataframe(filepath=file_path)
        if df is None or df.empty:
            raise HTTPException(status_code=422, detail="Uploaded dataset is empty or unreadable")
        profile = profile_dataset(df)
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=422, detail=f"Could not prepare retraining dataset: {e}")

    try:
        profile_json = json.dumps(profile)
    except Exception:
        profile_json = json.dumps({})

    with db_session() as db:
        db.add(
            DatasetModel(
                id=new_dataset_id,
                file_path=file_path,
                profile_json=profile_json,
                parent_dataset_id=parent_dataset_id,
                source_type="drift_retrain",
            )
        )
        db.commit()

    launch_config = _resolve_retrain_launch_config(
        job_params,
        goal_override=goal_override,
        mode_override=mode_override,
    )
    launch_context = _resolve_retrain_launch_context(
        launch_context_json,
        parent_job_id=job_id,
        launch_config=launch_config,
    )

    req = TrainRequest(
        dataset_id=new_dataset_id,
        target_column=job_params.get("target_column", ""),
        goal=launch_config["goal"],
        mode=launch_config["mode"],
        eval_metric=job_params.get("eval_metric", ""),
        selected_features=job_params.get("selected_features", []),
        handle_imbalance=bool(job_params.get("handle_imbalance", False)),
        auto_clean=bool(job_params.get("auto_clean", True)),
        cv_folds=int(job_params.get("cv_folds", 0) or 0),
        export_model=bool(job_params.get("export_model", True)),
        export_code=bool(job_params.get("export_code", True)),
        export_report=bool(job_params.get("export_report", True)),
    )
    train_resp = start_training(req)
    new_job_id = train_resp.get("job_id")
    if new_job_id:
        try:
            with db_session() as db:
                new_job = db.query(JobModel).filter(JobModel.id == new_job_id).first()
                if new_job:
                    try:
                        params = json.loads(new_job.params_json) if new_job.params_json else {}
                    except Exception:
                        params = {}
                    params["launch_context"] = launch_context
                    new_job.params_json = json.dumps(params)
                    db.commit()
        except Exception:
            pass
    return {
        "dataset_id": new_dataset_id,
        "profile": profile,
        "job_id": new_job_id,
        "goal": launch_config["goal"],
        "mode": launch_config["mode"],
        "launch_context": launch_context,
    }
