"""api/routes/datasets.py — Dataset upload, profile, health, detect, leakage."""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import json
import os
import csv
import tempfile
import zipfile
from uuid import uuid4
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import httpx
import pandas as pd
from sqlalchemy import create_engine

from infra.database import db_session, DatasetModel
from infra.result_contract import normalize_results, sanitize_for_json
from infra.storage import get_run_dir, get_model_path, get_metrics_path, get_schema_path
from core.data_profiler import profile_dataset
from core.health_score import compute_health_score
from core.file_loader import SUPPORTED_EXTENSIONS, load_dataframe
from services.data_sanitizer import sanitize_dataframe
from services.studio_service import (
    build_lineage_graph,
    compare_dataset_versions,
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


def _json_loads_safe(value: str | None, default: Any):
    try:
        return json.loads(value) if value else default
    except Exception:
        return default


def _build_dataset_response(
    dataset_id: str,
    df,
    source_type: str = "upload",
    display_name: str | None = None,
    parent_dataset_id: str | None = None,
):
    sanitized = sanitize_dataframe(df, dataset_name=display_name)
    df = sanitized.df
    file_path = _persist_dataframe_as_csv(dataset_id, df)
    profile = sanitize_for_json(profile_dataset(df))
    profile["sanitizer"] = sanitized.report
    profile["sanitizer_logs"] = sanitized.logs

    try:
        profile_json = json.dumps(profile)
    except Exception:
        profile_json = json.dumps({})

    with db_session() as db:
        db.add(
            DatasetModel(
                id=dataset_id,
                file_path=file_path,
                profile_json=profile_json,
                source_type=source_type,
                display_name=display_name,
                parent_dataset_id=parent_dataset_id,
            )
        )
        db.commit()

    preview_records = sanitize_for_json(json.loads(df.head(8).to_json(orient="records", date_format="iso")))
    return sanitize_for_json({
        "dataset_id": dataset_id,
        "profile": profile,
        "preview_records": preview_records,
        "ingest_summary": {
            "source_type": source_type,
            "display_name": display_name,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": list(df.columns),
        },
        "sanitizer": sanitized.report,
        "sanitizer_logs": sanitized.logs,
    })


def _pick_dataset_member(names, archive_member: str | None = None):
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

    if archive_member:
        normalized_member = archive_member.strip("/")
        for candidate in candidates:
            if candidate.strip("/") == normalized_member:
                return candidate
        for name in names:
            if name.strip("/") == normalized_member:
                return name
        raise ValueError(
            f"Archive member '{archive_member}' was not found. Available candidates: {', '.join(candidates or names)}"
        )

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


def _public_google_drive_download_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    if "drive.google.com" not in parsed.netloc:
        return raw_url

    query = parse_qs(parsed.query or "")
    file_id = query.get("id", [None])[0]
    if not file_id:
        parts = [part for part in parsed.path.split("/") if part]
        try:
            idx = parts.index("d")
            file_id = parts[idx + 1]
        except Exception:
            file_id = None

    if not file_id:
        return raw_url
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _dropbox_download_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    if "dropbox.com" not in parsed.netloc:
        return raw_url
    query = parse_qs(parsed.query or "")
    query["dl"] = ["1"]
    flattened = [(key, value) for key, values in query.items() for value in values]
    return parsed._replace(query="&".join(f"{key}={value}" for key, value in flattened)).geturl()


def _onedrive_download_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    host = parsed.netloc.lower()
    if "onedrive.live.com" not in host and "sharepoint.com" not in host and "1drv.ms" not in host:
        return raw_url

    url = raw_url
    if "download=1" not in url:
        url += "&download=1" if "?" in url else "?download=1"
    return url


def _normalize_remote_url(raw_url: str, source_type: str) -> str:
    source_norm = str(source_type or "").strip().lower().replace(" ", "_")
    if source_norm == "google_drive":
        return _public_google_drive_download_url(raw_url)
    if source_norm == "dropbox":
        return _dropbox_download_url(raw_url)
    if source_norm in {"onedrive", "sharepoint"}:
        return _onedrive_download_url(raw_url)
    return raw_url


def _download_remote_payload(url: str, method: str = "GET", headers: dict | None = None, body: Any = None):
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        response = client.request(method.upper(), url, headers=headers or {}, json=body)
        response.raise_for_status()
        return response


def _temp_file_from_response(response: httpx.Response, hint: str = "remote") -> str:
    parsed = urlparse(str(response.request.url))
    ext = os.path.splitext(parsed.path or "")[1].lower()
    if not ext:
        content_type = (response.headers.get("content-type") or "").lower()
        if "json" in content_type:
            ext = ".json"
        elif "csv" in content_type or "text/plain" in content_type:
            ext = ".csv"
        elif "pdf" in content_type:
            ext = ".pdf"
        elif "html" in content_type:
            ext = ".html"
    ext = ext or ".csv"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext, prefix=f"{hint}_") as handle:
        handle.write(response.content)
        return handle.name


def _load_remote_dataframe(
    url: str,
    source_type: str,
    method: str = "GET",
    headers: dict | None = None,
    body: Any = None,
    sheet_name: str | None = None,
    sqlite_table: str | None = None,
    archive_member: str | None = None,
):
    resolved_url = _normalize_remote_url(url, source_type)
    response = _download_remote_payload(resolved_url, method=method, headers=headers, body=body)

    if "json" in (response.headers.get("content-type") or "").lower():
        try:
            payload = response.json()
        except Exception:
            payload = None
        if isinstance(payload, list):
            return pd.json_normalize(payload), {}
        if isinstance(payload, dict):
            if isinstance(payload.get("data"), list):
                return pd.json_normalize(payload.get("data")), {}
            return pd.json_normalize([payload]), {}

    temp_path = _temp_file_from_response(response, hint=source_type.replace(" ", "_"))

    try:
        if temp_path.lower().endswith(".zip"):
            bundle_payload = _extract_bundle(temp_path)
            dataset_member = _pick_dataset_member(
                zipfile.ZipFile(temp_path).namelist(),
                archive_member=archive_member,
            )
            if not dataset_member:
                raise ValueError("ZIP source did not contain a supported dataset file.")
            extracted_ext = os.path.splitext(dataset_member)[1].lower() or ".csv"
            extracted_path = os.path.join(
                tempfile.gettempdir(), f"{uuid4().hex}{extracted_ext}"
            )
            with zipfile.ZipFile(temp_path) as archive:
                with open(extracted_path, "wb") as extracted:
                    extracted.write(archive.read(dataset_member))
            try:
                df = load_dataframe(
                    filepath=extracted_path,
                    sheet_name=sheet_name,
                    sqlite_table=sqlite_table,
                )
            finally:
                if os.path.exists(extracted_path):
                    os.remove(extracted_path)
            return df, bundle_payload

        return (
            load_dataframe(
                filepath=temp_path,
                sheet_name=sheet_name,
                sqlite_table=sqlite_table,
            ),
            {},
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _load_s3_dataframe(
    connection_uri: str,
    sheet_name: str | None = None,
    sqlite_table: str | None = None,
):
    parsed = urlparse(connection_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError("S3 import expects a URI like s3://bucket/path/to/file.csv")

    try:
        import boto3
    except ImportError as exc:
        raise ValueError("S3 import requires the optional 'boto3' package.") from exc

    suffix = os.path.splitext(parsed.path)[1].lower() or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        temp_path = handle.name

    try:
        boto3.client("s3").download_file(parsed.netloc, parsed.path.lstrip("/"), temp_path)
        return load_dataframe(
            filepath=temp_path,
            sheet_name=sheet_name,
            sqlite_table=sqlite_table,
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _load_gcs_dataframe(
    connection_uri: str,
    sheet_name: str | None = None,
    sqlite_table: str | None = None,
):
    parsed = urlparse(connection_uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path:
        raise ValueError("GCS import expects a URI like gs://bucket/path/to/file.csv")

    try:
        from google.cloud import storage
    except ImportError as exc:
        raise ValueError("GCS import requires the optional 'google-cloud-storage' package.") from exc

    suffix = os.path.splitext(parsed.path)[1].lower() or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        temp_path = handle.name

    try:
        client = storage.Client()
        bucket = client.bucket(parsed.netloc)
        blob = bucket.blob(parsed.path.lstrip("/"))
        blob.download_to_filename(temp_path)
        return load_dataframe(filepath=temp_path, sheet_name=sheet_name, sqlite_table=sqlite_table)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _load_azure_blob_dataframe(
    connection_uri: str,
    sheet_name: str | None = None,
    sqlite_table: str | None = None,
):
    parsed = urlparse(connection_uri)
    if parsed.scheme not in {"az", "azure"} or not parsed.netloc or not parsed.path:
        raise ValueError("Azure Blob import expects a URI like az://container/path/to/file.csv")

    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    if not conn_str:
        raise ValueError("Azure Blob import requires AZURE_STORAGE_CONNECTION_STRING to be set.")

    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError as exc:
        raise ValueError("Azure Blob import requires the optional 'azure-storage-blob' package.") from exc

    suffix = os.path.splitext(parsed.path)[1].lower() or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        temp_path = handle.name

    try:
        client = BlobServiceClient.from_connection_string(conn_str)
        blob_client = client.get_blob_client(container=parsed.netloc, blob=parsed.path.lstrip("/"))
        with open(temp_path, "wb") as output:
            output.write(blob_client.download_blob().readall())
        return load_dataframe(filepath=temp_path, sheet_name=sheet_name, sqlite_table=sqlite_table)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _inspect_local_source(filepath: str, display_name: str = ""):
    name = display_name or os.path.basename(filepath)
    ext = os.path.splitext(name)[1].lower()
    payload: dict[str, Any] = {
        "filename": name,
        "extension": ext,
        "zip_members": [],
        "excel_sheets": [],
        "sqlite_tables": [],
        "recommended": {},
    }

    try:
        if ext == ".zip":
            with zipfile.ZipFile(filepath) as archive:
                names = [name for name in archive.namelist() if not name.endswith("/")]
                payload["zip_members"] = names
                recommended = _pick_dataset_member(names)
                if recommended:
                    payload["recommended"]["archive_member"] = recommended
        elif ext in {".xlsx", ".xls", ".xlsm", ".ods"}:
            if ext in {".xlsx", ".xlsm"}:
                engine = "openpyxl"
            elif ext == ".xls":
                engine = "xlrd"
            else:
                engine = "odf"
            workbook = pd.ExcelFile(filepath, engine=engine)
            payload["excel_sheets"] = workbook.sheet_names
            if workbook.sheet_names:
                payload["recommended"]["sheet_name"] = workbook.sheet_names[0]
        elif ext in {".db", ".sqlite", ".sqlite3"}:
            with sqlite3.connect(filepath) as conn:
                rows = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
            payload["sqlite_tables"] = [row[0] for row in rows]
            if payload["sqlite_tables"]:
                payload["recommended"]["sqlite_table"] = payload["sqlite_tables"][0]
    except Exception as exc:
        payload["warning"] = str(exc)

    return payload


def _inspect_remote_source(
    url: str,
    source_type: str,
    method: str = "GET",
    headers: dict | None = None,
    body: Any = None,
):
    resolved_url = _normalize_remote_url(url, source_type)
    response = _download_remote_payload(resolved_url, method=method, headers=headers, body=body)
    temp_path = _temp_file_from_response(response, hint="inspect")
    try:
        return _inspect_local_source(temp_path, display_name=os.path.basename(urlparse(url).path) or "remote_source")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


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

    with db_session() as db:
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
    sheet_name: str = Form(""),
    sqlite_table: str = Form(""),
    archive_member: str = Form(""),
    text_chunk_size: int = Form(0),
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
            return {"error": f"Could not read ZIP archive: {e}"}

        try:
            with zipfile.ZipFile(uploaded_path) as zipf:
                dataset_member = _pick_dataset_member(
                    zipf.namelist(),
                    archive_member=archive_member or None,
                )
        except Exception as e:
            if os.path.exists(uploaded_path):
                os.remove(uploaded_path)
            return {"error": f"Could not inspect ZIP archive: {e}"}
        if not dataset_member:
            if os.path.exists(uploaded_path):
                os.remove(uploaded_path)
            return {
                "error": "ZIP archive does not include a supported dataset file. Upload a CSV, Excel, JSON, Parquet, SQLite, image, PDF, or text document inside the archive."
            }

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
        df = load_dataframe(
            filepath=source_path,
            pdf_mode=pdf_mode,
            sheet_name=sheet_name or None,
            sqlite_table=sqlite_table or None,
            text_chunk_size=text_chunk_size,
        )
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
    elif filename.lower().endswith(".zip"):
        source_type = "zip_upload"

    try:
        response = _build_dataset_response(
            dataset_id,
            df,
            source_type=source_type,
            display_name=filename,
        )
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
            imported_job_id = _restore_imported_job(
                dataset_id,
                bundle_payload,
            )
        except Exception:
            imported_job_id = None

    if imported_job_id:
        response["imported_job_id"] = imported_job_id
        response["imported_bundle"] = True
    return response


class SourceImportRequest(BaseModel):
    source_type: str
    connection_uri: str = ""
    query: str = ""
    http_method: str = "GET"
    headers_json: str = ""
    body_json: str = ""
    sheet_name: str = ""
    sqlite_table: str = ""
    archive_member: str = ""


class SourceInspectRequest(BaseModel):
    source_type: str = "url"
    connection_uri: str = ""
    http_method: str = "GET"
    headers_json: str = ""
    body_json: str = ""


@router.post("/inspect-upload")
async def inspect_upload(
    file: UploadFile = File(...),
):
    temp_id = str(uuid4())
    filename = file.filename or "upload.csv"
    try:
        uploaded_path = await _stream_upload_to_file(temp_id, filename, file)
        return _inspect_local_source(uploaded_path, display_name=filename)
    except Exception as e:
        return {"error": f"Failed to inspect upload: {e}"}
    finally:
        temp_path = os.path.join("tmp", f"{temp_id}{os.path.splitext(filename or '')[1].lower() or '.csv'}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@router.post("/inspect-source")
def inspect_source(req: SourceInspectRequest):
    source_type = str(req.source_type or "").strip().lower().replace(" ", "_")
    try:
        if source_type in {"s3", "gcs", "azure_blob"}:
            parsed = urlparse(req.connection_uri)
            ext = os.path.splitext(parsed.path or "")[1].lower()
            return {
                "filename": os.path.basename(parsed.path or "") or "remote_source",
                "extension": ext,
                "zip_members": [],
                "excel_sheets": [],
                "sqlite_tables": [],
                "recommended": {},
            }
        headers = _json_loads_safe(req.headers_json, {})
        body = _json_loads_safe(req.body_json, {})
        return _inspect_remote_source(
            req.connection_uri,
            source_type=source_type,
            method=req.http_method or ("POST" if body else "GET"),
            headers=headers if isinstance(headers, dict) else {},
            body=body if isinstance(body, (dict, list)) else {},
        )
    except Exception as e:
        return {"error": f"Failed to inspect source: {e}"}


@router.post("/import-source")
def import_from_source(req: SourceImportRequest):
    source_type = str(req.source_type or "").strip().lower()
    try:
        if source_type in {"postgresql", "mysql", "snowflake", "bigquery", "sql", "database"}:
            engine = create_engine(req.connection_uri)
            with engine.connect() as conn:
                df = pd.read_sql_query(req.query, conn)
        elif source_type in {
            "csv url",
            "csv_url",
            "url",
            "google_drive",
            "rest api",
            "rest_api",
            "api",
            "dropbox",
            "onedrive",
            "sharepoint",
        }:
            headers = _json_loads_safe(req.headers_json, {})
            body = _json_loads_safe(req.body_json, {})
            df, _ = _load_remote_dataframe(
                req.connection_uri,
                source_type=source_type.replace(" ", "_"),
                method=req.http_method or ("POST" if body else "GET"),
                headers=headers if isinstance(headers, dict) else {},
                body=body if isinstance(body, (dict, list)) else {},
                sheet_name=req.sheet_name or None,
                sqlite_table=req.sqlite_table or None,
                archive_member=req.archive_member or None,
            )
        elif source_type == "s3":
            df = _load_s3_dataframe(
                req.connection_uri,
                sheet_name=req.sheet_name or None,
                sqlite_table=req.sqlite_table or None,
            )
        elif source_type == "gcs":
            df = _load_gcs_dataframe(
                req.connection_uri,
                sheet_name=req.sheet_name or None,
                sqlite_table=req.sqlite_table or None,
            )
        elif source_type == "azure_blob":
            df = _load_azure_blob_dataframe(
                req.connection_uri,
                sheet_name=req.sheet_name or None,
                sqlite_table=req.sqlite_table or None,
            )
        else:
            raise ValueError(
                "Unsupported source type. Use PostgreSQL, MySQL, Snowflake, BigQuery, CSV URL, Google Drive, Dropbox, OneDrive, SharePoint, REST API, S3, GCS, or Azure Blob."
            )
    except Exception as e:
        return {"error": f"Failed to import from {req.source_type}: {e}"}

    if df is None or df.empty:
        return {"error": "The source query returned no rows."}

    dataset_id = str(uuid4())
    try:
        return _build_dataset_response(
            dataset_id,
            df,
            source_type=f"connector_{source_type or req.source_type}",
            display_name=f"{source_type or req.source_type}_import",
        )
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
        response = _build_dataset_response(
            new_dataset_id,
            df,
            source_type="ocr_review",
            display_name="ocr_review.csv",
            parent_dataset_id=dataset_id,
        )
    except Exception as e:
        return {"error": f"Failed to create reviewed OCR dataset: {e}"}

    with db_session() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == new_dataset_id).first()
        if dataset:
            dataset.parent_dataset_id = dataset_id
            db.commit()
    return response


class RepairPreviewRequest(BaseModel):
    dataset_id: str
    target_column: Optional[str] = None


@router.post("/repair-preview")
def repair_preview(req: RepairPreviewRequest):
    from services.training.preprocessing import auto_clean_data

    with db_session() as db:
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
    from services.training.preprocessing import auto_clean_data

    with db_session() as db:
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
        response = _build_dataset_response(
            new_dataset_id,
            repaired_df,
            source_type="repaired",
            display_name="repaired_dataset.csv",
            parent_dataset_id=req.dataset_id,
        )
    except Exception as e:
        return {"error": f"Failed to save repaired dataset: {e}"}

    with db_session() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == new_dataset_id).first()
        if dataset:
            dataset.parent_dataset_id = req.dataset_id
            db.commit()

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
    with db_session() as db:
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
        response = _build_dataset_response(
            new_dataset_id,
            merged_df,
            source_type="merge_studio",
            display_name="merged_dataset.csv",
        )
    except Exception as e:
        return {"error": f"Failed to save merged dataset: {e}"}

    with db_session() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == new_dataset_id).first()
        if dataset:
            dataset.parent_dataset_id = req.left_dataset_id
            db.commit()

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
    with db_session() as db:
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
    with db_session() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if not dataset:
            return {"error": "Dataset not found"}
        try:
            return json.loads(dataset.profile_json)
        except Exception:
            return {}


@router.get("/datasets")
def get_datasets(limit: int = 100, include_archived: bool = False):
    return {"datasets": list_datasets(limit=limit, include_archived=include_archived)}


@router.post("/dataset/{dataset_id}/archive")
def archive_dataset(dataset_id: str):
    with db_session() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if not dataset:
            return {"error": "Dataset not found"}
        try:
            profile = json.loads(dataset.profile_json) if dataset.profile_json else {}
        except Exception:
            profile = {}
        profile["archived"] = True
        dataset.profile_json = json.dumps(profile)
        if not str(dataset.source_type or "").startswith("archived:"):
            dataset.source_type = f"archived:{dataset.source_type or 'dataset'}"
        db.commit()
    return {"ok": True, "dataset_id": dataset_id}


@router.post("/dataset/{dataset_id}/unarchive")
def unarchive_dataset(dataset_id: str):
    with db_session() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if not dataset:
            return {"error": "Dataset not found"}
        try:
            profile = json.loads(dataset.profile_json) if dataset.profile_json else {}
        except Exception:
            profile = {}
        profile["archived"] = False
        dataset.profile_json = json.dumps(profile)
        if str(dataset.source_type or "").startswith("archived:"):
            dataset.source_type = str(dataset.source_type).split("archived:", 1)[-1] or "dataset"
        db.commit()
    return {"ok": True, "dataset_id": dataset_id}


@router.delete("/dataset/{dataset_id}")
def delete_dataset(dataset_id: str):
    from infra.database import JobModel, ExperimentRun, DriftCheck, DriftSchedule, WorkspaceModel, NotificationModel

    with db_session() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if not dataset:
            return {"error": "Dataset not found"}
        file_path = dataset.file_path

        jobs = db.query(JobModel).filter(JobModel.dataset_id == dataset_id).all()
        job_ids = [job.id for job in jobs]

        if job_ids:
            db.query(ExperimentRun).filter(ExperimentRun.dataset_id == dataset_id).delete(synchronize_session=False)
            db.query(DriftCheck).filter(DriftCheck.dataset_id == dataset_id).delete(synchronize_session=False)
            for job_id in job_ids:
                db.query(DriftSchedule).filter(DriftSchedule.job_id == job_id).delete(synchronize_session=False)
                db.query(NotificationModel).filter(NotificationModel.entity_id == job_id).delete(synchronize_session=False)
            db.query(JobModel).filter(JobModel.dataset_id == dataset_id).delete(synchronize_session=False)

        workspaces = db.query(WorkspaceModel).filter(WorkspaceModel.dataset_id == dataset_id).all()
        for workspace in workspaces:
            workspace.dataset_id = None
            if workspace.last_job_id in job_ids:
                workspace.last_job_id = None

        db.delete(dataset)
        db.commit()

    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass

    return {"ok": True, "dataset_id": dataset_id, "deleted_jobs": len(job_ids)}


@router.get("/dataset/{dataset_id}/versions")
def dataset_versions(dataset_id: str, target_column: Optional[str] = None):
    with db_session() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        return {"error": "Dataset not found"}
    return compare_dataset_versions(dataset_id, target=target_column)


@router.get("/workspace/latest")
def get_latest_workspace():
    return get_workspace_snapshot()


@router.get("/workspace/restore")
def restore_workspace(dataset_id: Optional[str] = None, job_id: Optional[str] = None):
    snapshot = get_workspace_snapshot(dataset_id=dataset_id, job_id=job_id)
    if not snapshot.get("dataset") and not snapshot.get("job"):
        raise HTTPException(status_code=404, detail="No persisted workspace found")
    return snapshot


# ── Health score ───────────────────────────────────────────────────────────────

@router.get("/health/{dataset_id}")
def get_health_score(dataset_id: str):
    with db_session() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
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

    with db_session() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == req.dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        try:
            profile = json.loads(dataset.profile_json)
        except Exception:
            profile = {}
        file_path = dataset.file_path

    try:
        df = load_dataframe(filepath=file_path)
        if df is None or df.empty:
            raise HTTPException(status_code=422, detail="Dataset is empty or unreadable")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not load dataset: {e}") from e

    from services.profiling_service import detect_problem_type as _detect
    result = _detect(df, profile, req.target_column)
    del df
    return result


# ── Leakage Detector ───────────────────────────────────────────────────────────

@router.get("/leakage/{dataset_id}")
def get_leakage_report(dataset_id: str, target_column: Optional[str] = None):
    if target_column:
        target_column = target_column.strip()

    with db_session() as db:
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
    with db_session() as db:
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
    with db_session() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        return {"error": "Dataset not found"}
    return build_lineage_graph(dataset_id)
