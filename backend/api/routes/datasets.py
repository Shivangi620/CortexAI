"""api/routes/datasets.py — Dataset upload, profile, health, detect, leakage."""
from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
import json
import os
import csv
import zipfile
from uuid import uuid4
from typing import Optional
import pandas as pd
from sqlalchemy import create_engine

from infra.database import get_db, DatasetModel
from infra.result_contract import normalize_results
from infra.storage import get_run_dir, get_model_path, get_metrics_path, get_schema_path
from core.data_profiler import profile_dataset
from core.health_score import compute_health_score
from core.file_loader import SUPPORTED_EXTENSIONS, load_dataframe
from services.training.preprocessing import auto_clean_data
from services.studio_service import (
    build_lineage_graph,
    get_workspace_snapshot,
    list_datasets,
    merge_preview,
)

csv.field_size_limit(int(1e9))

router = APIRouter(prefix="/api", tags=["datasets"])


def _save_uploaded_file(dataset_id: str, filename: str, payload: bytes) -> str:
    ext = os.path.splitext(filename or "")[1].lower() or ".csv"
    file_path = os.path.join("tmp", f"{dataset_id}{ext}")
    with open(file_path, "wb") as buffer:
        buffer.write(payload)
    return file_path


async def _stream_upload_to_file(dataset_id: str, filename: str, upload: UploadFile) -> str:
    ext = os.path.splitext(filename or "")[1].lower() or ".csv"
    file_path = os.path.join("tmp", f"{dataset_id}{ext}")
    os.makedirs("tmp", exist_ok=True)

    try:
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)
    except Exception:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    finally:
        await upload.close()

    return file_path


def _persist_dataframe_as_csv(dataset_id: str, df):
    csv_path = os.path.join("tmp", f"{dataset_id}.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def _build_dataset_response(dataset_id: str, df, source_type: str = "upload"):
    file_path = _persist_dataframe_as_csv(dataset_id, df)
    profile = profile_dataset(df)

    try:
        profile_json = json.dumps(profile)
    except Exception:
        profile_json = json.dumps({})

    with get_db() as db:
        db.add(
            DatasetModel(
                id=dataset_id,
                file_path=file_path,
                profile_json=profile_json,
                source_type=source_type,
            )
        )
        db.commit()

    preview_records = json.loads(df.head(8).to_json(orient="records", date_format="iso"))
    return {
        "dataset_id": dataset_id,
        "profile": profile,
        "preview_records": preview_records,
        "ingest_summary": {
            "source_type": source_type,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": list(df.columns),
        },
    }


def _pick_dataset_member(names):
    preferred = ["training_dataset.csv", "data/data.csv"]
    for name in preferred:
        if name in names:
            return name

    artifact_names = {
        "model.pkl",
        "artifacts/model.pkl",
        "model_metadata.json",
        "artifacts/model_metadata.json",
        "metrics.json",
        "logs/metrics.json",
        "schema.json",
        "data/schema.json",
        "import_bundle.json",
        "export_manifest.json",
        "feature_dictionary.json",
        "sample_input.json",
        "sample_output.json",
        "api.py",
        "batch_predict.py",
        "inference.py",
        "training.py",
        "explain.py",
        "requirements.txt",
        "README.md",
        "curl_examples.md",
    }
    preferred_exts = {
        ".csv", ".tsv", ".txt", ".dat", ".tab", ".log",
        ".xlsx", ".xls", ".xlsm", ".ods",
        ".json", ".jsonl", ".ndjson",
        ".parquet", ".feather", ".arrow", ".orc",
        ".dta", ".sas7bdat", ".sav", ".xpt",
        ".xml", ".html", ".htm",
        ".db", ".sqlite", ".sqlite3",
    }

    candidates = []
    for name in names:
        normalized = name.strip("/")
        ext = os.path.splitext(name)[1].lower()
        if name.endswith("/") or normalized in artifact_names:
            continue
        if ext in preferred_exts:
            candidates.append(name)

    if candidates:
        return candidates[0]

    for name in names:
        normalized = name.strip("/")
        ext = os.path.splitext(name)[1].lower()
        if name.endswith("/") or normalized in artifact_names:
            continue
        if ext in SUPPORTED_EXTENSIONS:
            return name

    return None


def _extract_bundle(zip_path: str):
    payload = {
        "dataset_member": None,
        "import_bundle": {},
        "artifacts": {},
    }

    with zipfile.ZipFile(zip_path) as zipf:
        names = zipf.namelist()
        payload["dataset_member"] = _pick_dataset_member(names)

        if "import_bundle.json" in names:
            try:
                payload["import_bundle"] = json.loads(zipf.read("import_bundle.json").decode("utf-8"))
            except Exception:
                payload["import_bundle"] = {}

        for archive_name in ("model.pkl", "model_metadata.json", "metrics.json", "schema.json"):
            if archive_name in names:
                payload["artifacts"][archive_name] = zipf.read(archive_name)

    return payload


def _restore_imported_job(dataset_id: str, bundle_payload: dict):
    import_bundle = bundle_payload.get("import_bundle") or {}
    artifacts = bundle_payload.get("artifacts") or {}
    raw_results = import_bundle.get("results")

    if not raw_results and artifacts.get("metrics.json"):
        try:
            raw_results = json.loads(artifacts["metrics.json"].decode("utf-8"))
        except Exception:
            raw_results = {}

    if not raw_results and not artifacts:
        return None

    results = normalize_results(raw_results)

    job_id = str(uuid4())
    run_dir = get_run_dir(job_id)
    model_path = None

    for archive_name, content in artifacts.items():
        if archive_name == "model.pkl":
            target_path = get_model_path(job_id)
            model_path = target_path
        elif archive_name == "metrics.json":
            target_path = get_metrics_path(job_id)
        elif archive_name == "schema.json":
            target_path = get_schema_path(job_id)
        elif archive_name == "model_metadata.json":
            target_path = os.path.join(run_dir, "artifacts", "model_metadata.json")
        else:
            continue

        with open(target_path, "wb") as handle:
            handle.write(content)

    if "metrics.json" not in artifacts:
        with open(get_metrics_path(job_id), "w") as handle:
            json.dump(results, handle, indent=2)

    with get_db() as db:
        from infra.database import JobModel

        db.add(
            JobModel(
                id=job_id,
                dataset_id=dataset_id,
                status="completed",
                history_json=json.dumps([{"time": "Import", "metric": "Bundle restored", "kind": "import"}]),
                results_json=json.dumps(results),
                model_path=model_path,
                insights_json=json.dumps(import_bundle.get("insights", {})),
                reasoning_json=json.dumps(import_bundle.get("reasoning", ["Imported from export bundle."])),
                story=import_bundle.get("story"),
                params_json=json.dumps(import_bundle.get("params", {})),
            )
        )
        db.commit()

    return job_id


# ── Upload ─────────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    pdf_mode: str = Form("text"),
):
    dataset_id = str(uuid4())
    os.makedirs("tmp", exist_ok=True)
    filename = file.filename or "upload.csv"

    try:
        uploaded_path = await _stream_upload_to_file(dataset_id, filename, file)
    except Exception as e:
        return {"error": f"Failed to save upload: {e}"}

    bundle_payload = {}
    source_path = uploaded_path
    source_name = filename

    if filename.lower().endswith(".zip"):
        try:
            bundle_payload = _extract_bundle(uploaded_path)
        except Exception as e:
            if os.path.exists(uploaded_path):
                os.remove(uploaded_path)
            return {"error": f"Could not read export bundle: {e}"}

        dataset_member = bundle_payload.get("dataset_member")
        if not dataset_member:
            if os.path.exists(uploaded_path):
                os.remove(uploaded_path)
            return {"error": "ZIP archive does not include a reusable dataset. Export a fresh bundle and try again."}

        source_name = os.path.basename(dataset_member)
        source_path = os.path.join("tmp", f"{dataset_id}{os.path.splitext(source_name)[1].lower() or '.csv'}")
        try:
            with zipfile.ZipFile(uploaded_path) as zipf:
                with open(source_path, "wb") as extracted:
                    extracted.write(zipf.read(dataset_member))
        except Exception as e:
            if os.path.exists(uploaded_path):
                os.remove(uploaded_path)
            return {"error": f"Failed to extract dataset from bundle: {e}"}

    try:
        df = load_dataframe(filepath=source_path, pdf_mode=pdf_mode)
    except Exception as e:
        for path in {uploaded_path, source_path}:
            if path and os.path.exists(path):
                os.remove(path)
        return {"error": f"Could not read dataset: {str(e)}"}

    if df is None or df.empty:
        for path in {uploaded_path, source_path}:
            if path and os.path.exists(path):
                os.remove(path)
        return {"error": "Uploaded dataset is empty or invalid"}

    source_type = "upload"
    ext = os.path.splitext(source_name or filename or "")[1].lower()
    if ext == ".pdf":
        source_type = f"pdf_{(pdf_mode or 'text').lower()}"
    elif ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}:
        source_type = "image_ocr"
    elif ext in {".db", ".sqlite", ".sqlite3"}:
        source_type = "database_import"

    try:
        response = _build_dataset_response(dataset_id, df, source_type=source_type)
    except Exception as e:
        for path in {uploaded_path, source_path}:
            if path and os.path.exists(path):
                os.remove(path)
        return {"error": f"Failed to normalize dataset as CSV: {e}"}

    for path in {uploaded_path, source_path}:
        persisted_path = os.path.join("tmp", f"{dataset_id}.csv")
        if path and path != persisted_path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

    imported_job_id = None
    if bundle_payload:
        try:
            imported_job_id = _restore_imported_job(dataset_id, bundle_payload)
        except Exception:
            imported_job_id = None

    if imported_job_id:
        response["imported_job_id"] = imported_job_id
        response["imported_bundle"] = True
    return response


class SourceImportRequest(BaseModel):
    source_type: str
    connection_uri: str
    query: str


@router.post("/import-source")
def import_from_source(req: SourceImportRequest):
    try:
        engine = create_engine(req.connection_uri)
        with engine.connect() as conn:
            df = pd.read_sql_query(req.query, conn)
    except Exception as e:
        return {"error": f"Failed to import from {req.source_type}: {e}"}

    if df is None or df.empty:
        return {"error": "The source query returned no rows."}

    dataset_id = str(uuid4())
    try:
        return _build_dataset_response(dataset_id, df, source_type=f"connector_{req.source_type}")
    except Exception as e:
        return {"error": f"Failed to save imported dataset: {e}"}


class OCRReviewRequest(BaseModel):
    text: str


@router.post("/dataset/{dataset_id}/ocr-review")
def create_dataset_from_ocr_review(dataset_id: str, req: OCRReviewRequest):
    reviewed_text = (req.text or "").strip()
    if not reviewed_text:
        return {"error": "Edited OCR text is empty."}

    lines = [line.strip() for line in reviewed_text.splitlines() if line.strip()]
    if not lines:
        return {"error": "No usable lines found in OCR text."}

    df = pd.DataFrame(
        {
            "source_dataset_id": dataset_id,
            "line_number": range(1, len(lines) + 1),
            "text": lines,
            "text_length": [len(line) for line in lines],
        }
    )

    new_dataset_id = str(uuid4())
    try:
        response = _build_dataset_response(new_dataset_id, df, source_type="ocr_review")
    except Exception as e:
        return {"error": f"Failed to create reviewed OCR dataset: {e}"}

    with get_db() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == new_dataset_id).first()
        if dataset:
            dataset.parent_dataset_id = dataset_id
    return response


class RepairPreviewRequest(BaseModel):
    dataset_id: str
    target_column: Optional[str] = None


@router.post("/repair-preview")
def repair_preview(req: RepairPreviewRequest):
    with get_db() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == req.dataset_id).first()
        if not dataset:
            return {"error": "Dataset not found"}
        file_path = dataset.file_path

    try:
        df = load_dataframe(filepath=file_path)
    except Exception as e:
        return {"error": f"Could not load dataset: {e}"}

    target_col = req.target_column or (df.columns[-1] if len(df.columns) else None)
    if not target_col:
        return {"error": "Could not determine a target column for repair preview."}

    repaired_df, repair_logs = auto_clean_data(df, target_col)
    return {
        "dataset_id": req.dataset_id,
        "target_column": target_col,
        "before_rows": int(len(df)),
        "after_rows": int(len(repaired_df)),
        "before_columns": int(len(df.columns)),
        "after_columns": int(len(repaired_df.columns)),
        "logs": repair_logs,
        "preview_records": json.loads(repaired_df.head(8).to_json(orient="records", date_format="iso")),
    }


class RepairApplyRequest(BaseModel):
    dataset_id: str
    target_column: Optional[str] = None


@router.post("/repair-apply")
def repair_apply(req: RepairApplyRequest):
    with get_db() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == req.dataset_id).first()
        if not dataset:
            return {"error": "Dataset not found"}
        file_path = dataset.file_path

    try:
        df = load_dataframe(filepath=file_path)
    except Exception as e:
        return {"error": f"Could not load dataset: {e}"}

    target_col = req.target_column or (df.columns[-1] if len(df.columns) else None)
    if not target_col:
        return {"error": "Could not determine a target column for repair."}

    repaired_df, repair_logs = auto_clean_data(df, target_col)
    new_dataset_id = str(uuid4())
    try:
        response = _build_dataset_response(new_dataset_id, repaired_df, source_type="repaired")
    except Exception as e:
        return {"error": f"Failed to save repaired dataset: {e}"}

    with get_db() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == new_dataset_id).first()
        if dataset:
            dataset.parent_dataset_id = req.dataset_id

    response["repair_logs"] = repair_logs
    return response


class MergeStudioRequest(BaseModel):
    left_dataset_id: str
    right_dataset_id: str
    join_key_left: str
    join_key_right: str
    join_type: str = "inner"


@router.post("/merge-studio")
def merge_studio(req: MergeStudioRequest):
    with get_db() as db:
        left_dataset = db.query(DatasetModel).filter(DatasetModel.id == req.left_dataset_id).first()
        right_dataset = db.query(DatasetModel).filter(DatasetModel.id == req.right_dataset_id).first()
        if not left_dataset or not right_dataset:
            return {"error": "One or both datasets were not found."}
        left_path = left_dataset.file_path
        right_path = right_dataset.file_path

    try:
        left_df = load_dataframe(filepath=left_path)
        right_df = load_dataframe(filepath=right_path)
    except Exception as e:
        return {"error": f"Could not load datasets for merge: {e}"}

    if req.join_key_left not in left_df.columns:
        return {"error": f"Left join key '{req.join_key_left}' is not present in the left dataset."}
    if req.join_key_right not in right_df.columns:
        return {"error": f"Right join key '{req.join_key_right}' is not present in the right dataset."}

    join_type = req.join_type if req.join_type in {"inner", "left", "right", "outer"} else "inner"
    left_dtype = str(left_df[req.join_key_left].dtype)
    right_dtype = str(right_df[req.join_key_right].dtype)
    key_coerced = False
    if left_dtype != right_dtype:
        left_df = left_df.copy()
        right_df = right_df.copy()
        left_df[req.join_key_left] = left_df[req.join_key_left].astype("string")
        right_df[req.join_key_right] = right_df[req.join_key_right].astype("string")
        key_coerced = True

    merged_df = pd.merge(
        left_df,
        right_df,
        left_on=req.join_key_left,
        right_on=req.join_key_right,
        how=join_type,
        suffixes=("_left", "_right"),
    )
    if merged_df.empty:
        return {"error": "Merge produced no rows. Check your join keys and join type."}

    new_dataset_id = str(uuid4())
    try:
        response = _build_dataset_response(new_dataset_id, merged_df, source_type="merge_studio")
    except Exception as e:
        return {"error": f"Failed to save merged dataset: {e}"}

    with get_db() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == new_dataset_id).first()
        if dataset:
            dataset.parent_dataset_id = req.left_dataset_id

    response["merge_summary"] = {
        "join_type": join_type,
        "left_rows": int(len(left_df)),
        "right_rows": int(len(right_df)),
        "merged_rows": int(len(merged_df)),
        "left_key": req.join_key_left,
        "right_key": req.join_key_right,
        "join_key_coerced_to_string": key_coerced,
        "left_key_dtype": left_dtype,
        "right_key_dtype": right_dtype,
    }
    return response


@router.post("/merge-studio/preview")
def merge_studio_preview(req: MergeStudioRequest):
    with get_db() as db:
        left_dataset = db.query(DatasetModel).filter(DatasetModel.id == req.left_dataset_id).first()
        right_dataset = db.query(DatasetModel).filter(DatasetModel.id == req.right_dataset_id).first()
        if not left_dataset or not right_dataset:
            return {"error": "One or both datasets were not found."}

    try:
        left_df = load_dataframe(filepath=left_dataset.file_path)
        right_df = load_dataframe(filepath=right_dataset.file_path)
    except Exception as e:
        return {"error": f"Could not load datasets for preview: {e}"}

    if req.join_key_left not in left_df.columns:
        return {"error": f"Left join key '{req.join_key_left}' is not present in the left dataset."}
    if req.join_key_right not in right_df.columns:
        return {"error": f"Right join key '{req.join_key_right}' is not present in the right dataset."}

    try:
        preview = merge_preview(
            left_df=left_df,
            right_df=right_df,
            left_key=req.join_key_left,
            right_key=req.join_key_right,
            join_type=req.join_type,
        )
    except Exception as e:
        return {"error": f"Could not build merge preview: {e}"}
    preview["join_type"] = req.join_type
    preview["left_dataset_id"] = req.left_dataset_id
    preview["right_dataset_id"] = req.right_dataset_id
    return preview


# ── Dataset info ───────────────────────────────────────────────────────────────

@router.get("/dataset/{dataset_id}")
def get_dataset_info(dataset_id: str):
    with get_db() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if not dataset:
            return {"error": "Dataset not found"}
        try:
            return json.loads(dataset.profile_json)
        except Exception:
            return {}


@router.get("/datasets")
def get_datasets(limit: int = 100):
    return {"datasets": list_datasets(limit=limit)}


@router.get("/workspace/latest")
def get_latest_workspace():
    return get_workspace_snapshot()


@router.get("/workspace/restore")
def restore_workspace(dataset_id: Optional[str] = None, job_id: Optional[str] = None):
    snapshot = get_workspace_snapshot(dataset_id=dataset_id, job_id=job_id)
    if not snapshot.get("dataset") and not snapshot.get("job"):
        return {"error": "No persisted workspace found"}
    return snapshot


# ── Health score ───────────────────────────────────────────────────────────────

@router.get("/health/{dataset_id}")
def get_health_score(dataset_id: str):
    with get_db() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if not dataset:
            return {"error": "Dataset not found"}
        try:
            profile = json.loads(dataset.profile_json)
        except Exception:
            profile = {}
    return compute_health_score(profile)


# ── Auto Problem Type Detection ────────────────────────────────────────────────

class DetectRequest(BaseModel):
    dataset_id: str
    target_column: Optional[str] = None


@router.post("/detect")
def detect_problem_type(req: DetectRequest):
    if req.target_column:
        req.target_column = req.target_column.strip()

    with get_db() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == req.dataset_id).first()
        if not dataset:
            return {"error": "Dataset not found"}
        try:
            profile = json.loads(dataset.profile_json)
        except Exception:
            profile = {}
        file_path = dataset.file_path

    try:
        df = load_dataframe(filepath=file_path)
        if df is None or df.empty:
            return {"error": "Dataset is empty or unreadable"}
    except Exception as e:
        return {"error": f"Could not load dataset: {e}"}

    from services.profiling_service import detect_problem_type as _detect
    result = _detect(df, profile, req.target_column)
    del df
    return result


# ── Leakage Detector ───────────────────────────────────────────────────────────

@router.get("/leakage/{dataset_id}")
def get_leakage_report(dataset_id: str, target_column: Optional[str] = None):
    if target_column:
        target_column = target_column.strip()

    with get_db() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if not dataset:
            return {"error": "Dataset not found"}
        try:
            profile = json.loads(dataset.profile_json)
        except Exception:
            profile = {}
        file_path = dataset.file_path

    try:
        df = load_dataframe(filepath=file_path)
        if df is None or df.empty:
            return {"error": "Dataset is empty or unreadable"}
    except Exception as e:
        return {"error": f"Could not load dataset: {e}"}

    if not target_column:
        target_column = profile.get("suggested_target") or (df.columns[-1] if len(df.columns) else None)

    if not target_column:
        return {"error": "Could not determine target column"}

    from services.leakage_service import run_leakage_report
    report = run_leakage_report(df, target_column)
    del df
    return report


@router.get("/dataset/{dataset_id}/timeline")
def get_dataset_timeline(dataset_id: str):
    with get_db() as db:
        current = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if not current:
            return {"error": "Dataset not found"}

        lineage = []
        cursor = current
        seen = set()

        while cursor and cursor.id not in seen:
            seen.add(cursor.id)
            try:
                profile = json.loads(cursor.profile_json) if cursor.profile_json else {}
            except Exception:
                profile = {}

            lineage.append(
                {
                    "dataset_id": cursor.id,
                    "parent_dataset_id": cursor.parent_dataset_id,
                    "source_type": cursor.source_type or "unknown",
                    "created_at": cursor.created_at.isoformat() if cursor.created_at else None,
                    "rows": profile.get("rows"),
                    "cols": profile.get("cols"),
                    "missing_pct": profile.get("missing_pct"),
                    "imbalance": profile.get("imbalance"),
                }
            )

            if not cursor.parent_dataset_id:
                break
            cursor = db.query(DatasetModel).filter(DatasetModel.id == cursor.parent_dataset_id).first()

        before = lineage[-1] if lineage else None
        after = lineage[0] if lineage else None
        diff = {}
        if before and after:
            for key in ("rows", "cols", "missing_pct"):
                try:
                    if before.get(key) is not None and after.get(key) is not None:
                        diff[key] = round(float(after[key]) - float(before[key]), 2)
                except Exception:
                    continue

        return {"timeline": lineage, "profile_diff": diff}


@router.get("/dataset/{dataset_id}/lineage-graph")
def get_dataset_lineage_graph(dataset_id: str):
    return build_lineage_graph(dataset_id)
