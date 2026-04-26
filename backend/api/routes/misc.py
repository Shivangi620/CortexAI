"""api/routes/misc.py — Chat, recommendations, resume, export, zeroshot, meta."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List
import json
import os

from infra.database import db_session, DatasetModel, JobModel
from infra.launch_origin import parse_launch_origin
from infra.result_contract import normalize_results, sanitize_for_json
from services.studio_service import narrate_experiment, synthetic_data_judge

router = APIRouter(prefix="/api", tags=["misc"])


# ── Chat ───────────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    job_id: str
    prompt: str


@router.post("/chat")
def chat_endpoint(req: ChatRequest):
    with db_session() as db:
        job = db.query(JobModel).filter(JobModel.id == req.job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        try:
            results = (
                normalize_results(json.loads(job.results_json))
                if job.results_json
                else None
            )
        except Exception:
            results = None

        context = {
            "status": job.status,
            "results": results,
        }

    from core.insights import chat_with_model

    return {"response": chat_with_model(req.prompt, context)}


# ── NL→ML intent parser ────────────────────────────────────────────────────────


class NLIntentRequest(BaseModel):
    prompt: str
    dataset_id: str = ""


@router.post("/nl/intent")
def parse_nl_intent(req: NLIntentRequest):
    from core.insights import parse_nl_intent as _parse

    profile = {}
    if req.dataset_id:
        with db_session() as db:
            ds = (
                db.query(DatasetModel).filter(DatasetModel.id == req.dataset_id).first()
            )
            if ds:
                try:
                    profile = json.loads(ds.profile_json)
                except Exception:
                    profile = {}

    return _parse(req.prompt, profile)


# ── Recommendations ────────────────────────────────────────────────────────────


@router.get("/recommend/{job_id}")
def get_recommendations(job_id: str):
    with db_session() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job or not job.results_json:
            raise HTTPException(
                status_code=404, detail="Job not found or not completed"
            )

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

    from core.recommendations import generate_recommendations

    return {"recommendations": generate_recommendations(profile, results)}


# ── Export ─────────────────────────────────────────────────────────────────────


@router.get("/export/{job_id}")
def export_model(job_id: str):
    with db_session() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != "completed":
            raise HTTPException(
                status_code=409, detail=f"Job not completed (status: {job.status})"
            )
        if not job.results_json:
            raise HTTPException(status_code=400, detail="No results available")

        try:
            results = normalize_results(json.loads(job.results_json))
        except Exception:
            results = {}

    from core.export import (
        build_export_bundle_filename,
        create_export_bundle,
        cleanup_old_exports,
    )

    cleanup_old_exports(max_age_hours=24)

    export_path = create_export_bundle(job_id, results)

    if not export_path or not isinstance(export_path, str):
        raise HTTPException(status_code=500, detail="Export failed")

    try:
        params = json.loads(job.params_json) if job.params_json else {}
    except Exception:
        params = {}
    launch_origin = parse_launch_origin(params)

    return FileResponse(
        path=export_path,
        filename=build_export_bundle_filename(job_id, launch_origin),
        media_type="application/zip",
    )


# ── Synthetic Data ─────────────────────────────────────────────────────────────


@router.post("/synthetic/{dataset_id}")
def synthetic_expand(dataset_id: str, n_rows: int = None):
    """
    Generate synthetic data by expanding the dataset with statistically similar rows.

    Query parameters:
    - n_rows (optional): Number of synthetic rows to generate. If not provided, auto-calculates.

    Returns:
    - new_dataset_id: ID of the synthetic dataset
    - original_rows: Number of original rows
    - synthetic_rows_added: Number of synthetic rows added
    - total_rows: Total rows in expanded dataset
    - preview: First 5 synthetic rows
    """
    from uuid import uuid4
    from core.synthetic import generate_synthetic, suggest_expansion_size
    from core.data_profiler import profile_dataset
    from core.file_loader import load_dataframe

    with db_session() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        file_path = dataset.file_path

        try:
            profile = sanitize_for_json(json.loads(dataset.profile_json))
        except Exception:
            profile = {}

    try:
        df = load_dataframe(filepath=file_path)
        if df is None or df.empty:
            return {"error": "Dataset is empty or unreadable"}
    except Exception as e:
        return {"error": f"Failed to load dataset: {e}"}

    original_rows = len(df)

    # Validate and convert n_rows parameter
    try:
        requested_n = int(n_rows) if n_rows else None
    except (ValueError, TypeError):
        requested_n = None

    recommended_n = suggest_expansion_size(original_rows)
    n_new = max(1, requested_n or recommended_n)

    adjustment_note = None
    if requested_n and requested_n > recommended_n * 4:
        adjustment_note = (
            f"Requested {requested_n} rows. This is much larger than the recommended "
            f"{recommended_n} rows. Validate the augmented dataset carefully before retraining."
        )

    try:
        expanded_df, synthetic_only = generate_synthetic(df, n_new)
    except (ValueError, Exception) as e:
        return {"error": f"Failed to generate synthetic data: {str(e)}"}

    new_id = str(uuid4())
    os.makedirs("tmp", exist_ok=True)
    new_path = f"tmp/{new_id}.csv"
    actual_added = len(synthetic_only)

    try:
        expanded_df.to_csv(new_path, index=False)
    except Exception as e:
        return {"error": f"Failed to save synthetic dataset: {e}"}

    try:
        new_profile = sanitize_for_json(profile_dataset(expanded_df))
    except Exception:
        new_profile = profile.copy()
        new_profile.update(
            {
                "rows": len(expanded_df),
                "columns": list(expanded_df.columns),
            }
        )

    new_profile.update({"synthetic_added": actual_added, "original_rows": original_rows})

    try:
        profile_json = json.dumps(new_profile)
    except Exception:
        profile_json = json.dumps({})

    with db_session() as db:
        db.add(
            DatasetModel(
                id=new_id,
                file_path=new_path,
                profile_json=profile_json,
                parent_dataset_id=dataset_id,
                source_type="synthetic",
            )
        )
        db.commit()

    profile_diff = {}
    for key in ("rows", "cols", "missing_pct"):
        try:
            before = profile.get(key)
            after = new_profile.get(key)
            if before is not None and after is not None:
                profile_diff[key] = round(float(after) - float(before), 2)
        except Exception:
            continue

    return sanitize_for_json({
        "dataset_id": dataset_id,
        "new_dataset_id": new_id,
        "original_rows": original_rows,
        "recommended_rows": recommended_n,
        "requested_rows": requested_n,
        "generation_mode": "manual" if requested_n else "auto",
        "synthetic_rows_added": actual_added,
        "total_rows": len(expanded_df),
        "augmentation_ratio": round(len(synthetic_only) / max(original_rows, 1), 2),
        "adjustment_note": adjustment_note,
        "original_profile": profile,
        "profile": new_profile,
        "profile_diff": profile_diff,
        "preview": json.loads(synthetic_only.head(5).to_json(orient="records")),
    })


@router.get("/synthetic/judge/{dataset_id}")
def synthetic_judge(dataset_id: str):
    with db_session() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        return {"error": "Dataset not found"}
    return synthetic_data_judge(dataset_id)


# ── Playground Quick Train ─────────────────────────────────────────────────────


class PlaygroundRequest(BaseModel):
    dataset_id: str
    target_column: str
    selected_features: List[str] = Field(default_factory=list)
    selected_models: List[str] = Field(default_factory=list)


@router.post("/quicktrain")
def playground_train(req: PlaygroundRequest):
    with db_session() as db:
        dataset = (
            db.query(DatasetModel).filter(DatasetModel.id == req.dataset_id).first()
        )
        if not dataset:
            return {"error": "Dataset not found"}

        file_path = dataset.file_path

    from core.file_loader import load_dataframe
    from core.playground import quick_train

    try:
        df = load_dataframe(filepath=file_path)
        if df is None or df.empty:
            raise HTTPException(
                status_code=422, detail="Dataset is empty or unreadable"
            )
    except Exception as e:
        raise HTTPException(
            status_code=422, detail=f"Could not reload dataset: {e}"
        ) from e

    try:
        return quick_train(
            df, req.target_column, req.selected_features, req.selected_models
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


# ── Zero-Shot + Meta ───────────────────────────────────────────────────────────


@router.get("/zeroshot/{dataset_id}")
def zero_shot(dataset_id: str):
    with db_session() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        try:
            profile = json.loads(dataset.profile_json)
        except Exception:
            profile = {}

    from core.meta_learning import zero_shot_recommend

    return zero_shot_recommend(profile)


@router.get("/meta/insights/{dataset_id}")
def cross_dataset_insights(dataset_id: str):
    with db_session() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        try:
            profile = json.loads(dataset.profile_json)
        except Exception:
            profile = {}

    from core.meta_learning import get_cross_dataset_insights

    return get_cross_dataset_insights(profile)


@router.get("/meta/status")
def meta_status():
    try:
        from core.meta_learning import meta_engine

        return {
            "is_trained": bool(meta_engine.is_trained),
            "min_records": meta_engine.min_records,
            "validation_error": round(float(meta_engine.val_error), 4),
            "backend": "lightgbm" if meta_engine.model is not None else "heuristics",
        }
    except Exception as exc:
        return {
            "is_trained": False,
            "min_records": 0,
            "validation_error": None,
            "backend": "heuristics",
            "warning": str(exc),
        }


@router.get("/narrate/{job_id}")
def narrate_job(job_id: str):
    with db_session() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        dataset_id = job.dataset_id
        story = job.story
        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()

        try:
            profile = (
                json.loads(dataset.profile_json)
                if dataset and dataset.profile_json
                else {}
            )
        except Exception:
            profile = {}

        try:
            results = (
                normalize_results(json.loads(job.results_json))
                if job.results_json
                else {}
            )
        except Exception:
            results = {}

    return narrate_experiment(profile, results, story)
