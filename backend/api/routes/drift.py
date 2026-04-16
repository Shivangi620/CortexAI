import csv
from datetime import datetime, timedelta
import json
import os
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from infra.database import DriftCheck, DriftSchedule, DatasetModel, JobModel, get_db
from infra.result_contract import normalize_results

csv.field_size_limit(int(1e9))

router = APIRouter(prefix="/api", tags=["drift"])


@router.post("/drift/{job_id}")
async def detect_drift(job_id: str, file: UploadFile = File(...)):
    """
    Feature 6: Drift dashboard.
    Upload a new CSV; compare its feature distributions against training baseline.
    """
    with get_db() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job or job.status != "completed":
            raise HTTPException(status_code=404, detail="Job not completed")

        try:
            results = normalize_results(json.loads(job.results_json) if job.results_json else {})
        except Exception:
            results = {}

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

    report = get_drift_dashboard(current_df, baseline_stats, feature_names)

    try:
        with get_db() as db:
            job = db.query(JobModel).filter(JobModel.id == job_id).first()
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
    except Exception:
        pass

    return report


@router.get("/drift/{job_id}/history")
def drift_history(job_id: str):
    with get_db() as db:
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
    with get_db() as db:
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
    with get_db() as db:
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
        else:
            frequency_days = int(row.frequency_days or 7)
            enabled = str(row.enabled).lower() == "true"

        next_due_at = None
        due_now = False
        if latest_check and latest_check.created_at:
            next_due = latest_check.created_at + timedelta(days=frequency_days)
            next_due_at = next_due.isoformat()
            due_now = enabled and next_due <= datetime.utcnow()
        elif enabled:
            due_now = True

        return {
            "job_id": job_id,
            "enabled": enabled,
            "frequency_days": frequency_days,
            "last_check_at": latest_check.created_at.isoformat() if latest_check and latest_check.created_at else None,
            "next_due_at": next_due_at,
            "due_now": due_now,
        }


@router.post("/drift/{job_id}/schedule")
def save_drift_schedule(job_id: str, enabled: bool = True, frequency_days: int = 7):
    frequency_days = max(1, int(frequency_days or 7))
    with get_db() as db:
        row = db.query(DriftSchedule).filter(DriftSchedule.job_id == job_id).first()
        if row:
            row.enabled = str(bool(enabled)).lower()
            row.frequency_days = str(frequency_days)
        else:
            db.add(
                DriftSchedule(
                    job_id=job_id,
                    enabled=str(bool(enabled)).lower(),
                    frequency_days=str(frequency_days),
                )
            )
    return {"job_id": job_id, "enabled": enabled, "frequency_days": frequency_days}


@router.post("/drift/{job_id}/retrain")
async def retrain_on_drift_dataset(job_id: str, file: UploadFile = File(...)):
    from core.file_loader import load_dataframe
    from core.data_profiler import profile_dataset
    from api.routes.training import TrainRequest, start_training

    with get_db() as db:
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

    with get_db() as db:
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

    req = TrainRequest(
        dataset_id=new_dataset_id,
        target_column=job_params.get("target_column", ""),
        goal=job_params.get("goal", "Balanced"),
        mode=job_params.get("mode", "Balanced"),
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
    return {
        "dataset_id": new_dataset_id,
        "profile": profile,
        "job_id": train_resp.get("job_id"),
    }
