"""api/routes/misc.py — Chat, recommendations, resume, export, zeroshot, meta."""
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List
import json
import os

from infra.database import get_db, DatasetModel, JobModel
from infra.result_contract import normalize_results
from services.studio_service import narrate_experiment, synthetic_data_judge

router = APIRouter(prefix="/api", tags=["misc"])


# ── Chat ───────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    job_id: str
    prompt: str


@router.post("/chat")
def chat_endpoint(req: ChatRequest):
    with get_db() as db:
        job = db.query(JobModel).filter(JobModel.id == req.job_id).first()
        if not job:
            return {"error": "Job not found"}

        try:
            results = normalize_results(json.loads(job.results_json)) if job.results_json else None
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
        with get_db() as db:
            ds = db.query(DatasetModel).filter(DatasetModel.id == req.dataset_id).first()
            if ds:
                try:
                    profile = json.loads(ds.profile_json)
                except Exception:
                    profile = {}

    return _parse(req.prompt, profile)


# ── Recommendations ────────────────────────────────────────────────────────────

@router.get("/recommend/{job_id}")
def get_recommendations(job_id: str):
    with get_db() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job or not job.results_json:
            return {"error": "Job not found or not completed"}

        dataset = db.query(DatasetModel).filter(DatasetModel.id == job.dataset_id).first()

        try:
            profile = json.loads(dataset.profile_json) if dataset and dataset.profile_json else {}
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
    with get_db() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != "completed":
            raise HTTPException(status_code=409, detail=f"Job not completed (status: {job.status})")
        if not job.results_json:
            raise HTTPException(status_code=400, detail="No results available")

        try:
            results = normalize_results(json.loads(job.results_json))
        except Exception:
            results = {}

    from core.export import create_export_bundle, cleanup_old_exports

    cleanup_old_exports(max_age_hours=24)

    export_path = create_export_bundle(job_id, results)

    if not export_path or not isinstance(export_path, str):
        raise HTTPException(status_code=500, detail="Export failed")

    return FileResponse(path=export_path, filename="automl_export.zip", media_type="application/zip")


# ── Synthetic Data ─────────────────────────────────────────────────────────────

@router.post("/synthetic/{dataset_id}")
def synthetic_expand(dataset_id: str, n_rows: int = None):
    import pandas as pd
    from uuid import uuid4
    from core.synthetic import generate_synthetic, suggest_expansion_size
    from core.data_profiler import profile_dataset
    from core.file_loader import load_dataframe

    with get_db() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if not dataset:
            return {"error": "Dataset not found"}

        file_path = dataset.file_path

        try:
            profile = json.loads(dataset.profile_json)
        except Exception:
            profile = {}

    try:
        df = load_dataframe(filepath=file_path)
        if df is None or df.empty:
            return {"error": "Dataset is empty or unreadable"}
    except Exception as e:
        return {"error": f"Failed to load dataset: {e}"}

    original_rows = len(df)
    recommended_n = suggest_expansion_size(original_rows)
    requested_n = int(n_rows or recommended_n)
    n_new = max(1, requested_n)
    adjustment_note = None
    if requested_n > recommended_n * 4:
        adjustment_note = (
            f"Requested {requested_n} rows. This is much larger than the recommended "
            f"{recommended_n} rows, so validate the augmented dataset carefully before retraining."
        )

    expanded_df, synthetic_only = generate_synthetic(df, n_new)

    new_id = str(uuid4())
    new_path = f"tmp/{new_id}.csv"

    try:
        expanded_df.to_csv(new_path, index=False)
    except Exception as e:
        return {"error": f"Failed to save synthetic dataset: {e}"}

    try:
        new_profile = profile_dataset(expanded_df)
    except Exception:
        new_profile = profile.copy()
        new_profile.update({
            "rows": len(expanded_df),
            "columns": list(expanded_df.columns),
        })

    new_profile.update({
        "synthetic_added": n_new,
        "original_rows": original_rows
    })

    try:
        profile_json = json.dumps(new_profile)
    except Exception:
        profile_json = json.dumps({})

    with get_db() as db:
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

    return {
        "new_dataset_id": new_id,
        "original_rows": original_rows,
        "recommended_rows": recommended_n,
        "requested_rows": requested_n,
        "synthetic_rows_added": n_new,
        "total_rows": len(expanded_df),
        "augmentation_ratio": round(n_new / max(original_rows, 1), 2),
        "adjustment_note": adjustment_note,
        "original_profile": profile,
        "profile": new_profile,
        "profile_diff": profile_diff,
        "preview": json.loads(synthetic_only.head(5).to_json(orient="records")),
    }


@router.get("/synthetic/judge/{dataset_id}")
def synthetic_judge(dataset_id: str):
    return synthetic_data_judge(dataset_id)


# ── Playground Quick Train ─────────────────────────────────────────────────────


class PlaygroundRequest(BaseModel):
    dataset_id: str
    target_column: str
    selected_features: List[str] = Field(default_factory=list)
    selected_models: List[str] = Field(default_factory=list)


@router.post("/quicktrain")
def playground_train(req: PlaygroundRequest):
    with get_db() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == req.dataset_id).first()
        if not dataset:
            return {"error": "Dataset not found"}

        file_path = dataset.file_path

    from core.file_loader import load_dataframe
    from core.playground import quick_train

    try:
        df = load_dataframe(filepath=file_path)
        if df is None or df.empty:
            return {"error": "Dataset is empty or unreadable"}
    except Exception as e:
        return {"error": f"Could not reload dataset: {e}"}

    return quick_train(df, req.target_column, req.selected_features, req.selected_models)


# ── Zero-Shot + Meta ───────────────────────────────────────────────────────────

@router.get("/zeroshot/{dataset_id}")
def zero_shot(dataset_id: str):
    with get_db() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if not dataset:
            return {"error": "Dataset not found"}

        try:
            profile = json.loads(dataset.profile_json)
        except Exception:
            profile = {}

    from core.meta_learning import zero_shot_recommend
    return zero_shot_recommend(profile)


@router.get("/meta/insights/{dataset_id}")
def cross_dataset_insights(dataset_id: str):
    with get_db() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if not dataset:
            return {"error": "Dataset not found"}

        try:
            profile = json.loads(dataset.profile_json)
        except Exception:
            profile = {}

    from core.meta_learning import get_cross_dataset_insights
    return get_cross_dataset_insights(profile)


@router.get("/meta/status")
def meta_status():
    from core.meta_learning import meta_engine

    return {
        "is_trained": bool(meta_engine.is_trained),
        "min_records": meta_engine.min_records,
        "validation_error": round(float(meta_engine.val_error), 4),
        "backend": "lightgbm" if meta_engine.model is not None else "heuristics",
    }


@router.get("/narrate/{job_id}")
def narrate_job(job_id: str):
    with get_db() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job:
            return {"error": "Job not found"}
        dataset_id = job.dataset_id
        story = job.story
        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()

        try:
            profile = json.loads(dataset.profile_json) if dataset and dataset.profile_json else {}
        except Exception:
            profile = {}

        try:
            results = normalize_results(json.loads(job.results_json)) if job.results_json else {}
        except Exception:
            results = {}

    return narrate_experiment(profile, results, story)
