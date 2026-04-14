"""
core/storage.py — Structured file storage for AutoML Studio.
"""
import os
import time
import json
import shutil
import hashlib
import pandas as pd
from typing import Optional, Dict, Any
from infra.result_contract import sanitize_for_json


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(BASE_DIR, "runs")
TMP_DIR = os.path.join(BASE_DIR, "tmp")


def get_run_dir(run_id: str) -> str:
    path = os.path.join(RUNS_DIR, run_id)
    try:
        os.makedirs(os.path.join(path, "artifacts"), exist_ok=True)
        os.makedirs(os.path.join(path, "data"), exist_ok=True)
        os.makedirs(os.path.join(path, "logs"), exist_ok=True)
    except Exception:
        pass
    return path


def get_model_path(run_id: str) -> str:
    return os.path.join(get_run_dir(run_id), "artifacts", "model.pkl")


def get_metrics_path(run_id: str) -> str:
    return os.path.join(get_run_dir(run_id), "logs", "metrics.json")


def get_data_path(run_id: str) -> str:
    return os.path.join(get_run_dir(run_id), "data", "data.csv")


def get_schema_path(run_id: str) -> str:
    return os.path.join(get_run_dir(run_id), "data", "schema.json")


def get_shap_path(run_id: str) -> str:
    return os.path.join(get_run_dir(run_id), "logs", "shap_values.json")


def get_report_path(run_id: str) -> str:
    return os.path.join(get_run_dir(run_id), "logs", "report.pdf")


def save_metrics(run_id: str, metrics: dict) -> None:
    path = get_metrics_path(run_id)
    try:
        with open(path, "w") as f:
            json.dump(sanitize_for_json(metrics), f, indent=2, allow_nan=False)
    except Exception:
        pass


def load_metrics(run_id: str) -> Optional[dict]:
    path = get_metrics_path(run_id)
    try:
        if not os.path.exists(path):
            old_path = os.path.join(get_run_dir(run_id), "metrics.json")
            if os.path.exists(old_path):
                with open(old_path) as f:
                    return json.load(f)
            return None
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


class DataContract:
    @staticmethod
    def generate_schema(df: pd.DataFrame) -> Dict[str, Any]:
        schema = {}
        for col in df.columns:
            try:
                dtype = str(df[col].dtype)
                schema[col] = {
                    "type": dtype,
                    "missing": int(df[col].isna().sum())
                }
            except Exception:
                schema[col] = {"type": "unknown", "missing": 0}

        try:
            df_hash = hashlib.sha256(
                pd.util.hash_pandas_object(df, index=True).values
            ).hexdigest()
        except Exception:
            df_hash = ""

        return {
            "schema": schema,
            "hash": df_hash,
            "rows": int(len(df)),
            "cols": int(len(df.columns))
        }

    @staticmethod
    def save_contract(run_id: str, df: pd.DataFrame):
        try:
            schema_data = DataContract.generate_schema(df)
            path = get_schema_path(run_id)
            with open(path, "w") as f:
                json.dump(schema_data, f, indent=2)
        except Exception:
            pass


class ModelRegistry:
    @staticmethod
    def save_model(run_id: str, model_obj: Any, metadata: Dict[str, Any]):
        import joblib

        try:
            path = get_model_path(run_id)
            joblib.dump(model_obj, path)

            meta_path = os.path.join(get_run_dir(run_id), "artifacts", "model_metadata.json")
            metadata = dict(metadata or {})
            metadata["timestamp"] = time.time()
            metadata["run_id"] = run_id

            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception:
            pass

    @staticmethod
    def load_model(run_id: str) -> Optional[Any]:
        import joblib

        path = get_model_path(run_id)
        try:
            if not os.path.exists(path):
                old_path = os.path.join(get_run_dir(run_id), "model.pkl")
                if os.path.exists(old_path):
                    return joblib.load(old_path)
                return None
            return joblib.load(path)
        except Exception:
            return None


def get_legacy_model_path(job_id: str) -> str:
    return os.path.join(TMP_DIR, f"{job_id}_model.pkl")


def resolve_model_path(job_id: str) -> Optional[str]:
    try:
        new_path = get_model_path(job_id)
        if os.path.exists(new_path):
            return new_path

        legacy_dir = os.path.join(get_run_dir(job_id), "model.pkl")
        if os.path.exists(legacy_dir):
            return legacy_dir

        legacy = get_legacy_model_path(job_id)
        if os.path.exists(legacy):
            return legacy

    except Exception:
        pass

    return None


def resolve_features_path(job_id: str) -> Optional[Any]:
    try:
        meta_path = os.path.join(get_run_dir(job_id), "artifacts", "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                if isinstance(meta, dict) and "feature_names" in meta:
                    return meta.get("feature_names")
    except Exception:
        pass

    return None


def cleanup_old_runs(days: int = 7) -> int:
    removed = 0
    cutoff = time.time() - days * 86400

    try:
        if os.path.exists(RUNS_DIR):
            for entry in os.listdir(RUNS_DIR):
                path = os.path.join(RUNS_DIR, entry)
                if os.path.isdir(path):
                    try:
                        if os.path.getmtime(path) < cutoff:
                            shutil.rmtree(path, ignore_errors=True)
                            removed += 1
                    except Exception:
                        continue

        tmp_cutoff = time.time() - 86400
        if os.path.exists(TMP_DIR):
            for fname in os.listdir(TMP_DIR):
                fpath = os.path.join(TMP_DIR, fname)
                if os.path.isfile(fpath):
                    try:
                        if os.path.getmtime(fpath) < tmp_cutoff:
                            os.remove(fpath)
                            removed += 1
                    except Exception:
                        continue
    except Exception:
        pass

    return removed


try:
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
except Exception:
    pass
