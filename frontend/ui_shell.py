import os
import json
from typing import Any, Iterable, Optional

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("AUTOML_API_URL", "http://127.0.0.1:8000/api").rstrip("/")

SESSION_DEFAULTS = {
    "dataset_id": None,
    "profile": None,
    "job_id": None,
    "auto_detect": None,
    "last_analyzed_file": None,
    "upload_preview_records": [],
    "upload_ingest_summary": {},
    "_workspace_restored": False,
    "_workspace_bootstrapped": False,
    "_workspace_cleared": False,
}


def load_css() -> None:
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
    try:
        with open(css_path, encoding="utf-8") as file:
            st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass


def ensure_session_state() -> None:
    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default
    if not st.session_state.get("_workspace_bootstrapped"):
        restore_workspace_state()
        st.session_state["_workspace_bootstrapped"] = True


def _query_param_value(name: str) -> Optional[str]:
    try:
        value = st.query_params.get(name)
    except Exception:
        return None
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def get_query_param(name: str) -> Optional[str]:
    return _query_param_value(name)


def sync_query_params(**updates: Any) -> None:
    try:
        merged = dict(st.query_params)
        for key, value in updates.items():
            if value in (None, "", [], {}):
                merged.pop(key, None)
            else:
                merged[key] = str(value)
        st.query_params.clear()
        st.query_params.update(merged)
    except Exception:
        pass


def sync_workspace_query_params(**extra: Any) -> None:
    sync_query_params(
        dataset_id=st.session_state.get("dataset_id"),
        job_id=st.session_state.get("job_id"),
        fresh=None,
        **extra,
    )


def clear_workspace_state() -> None:
    st.session_state["dataset_id"] = None
    st.session_state["profile"] = None
    st.session_state["job_id"] = None
    st.session_state["auto_detect"] = None
    st.session_state["last_analyzed_file"] = None
    st.session_state["upload_preview_records"] = []
    st.session_state["upload_ingest_summary"] = {}
    st.session_state["_workspace_restored"] = False
    st.session_state["_workspace_cleared"] = True
    sync_query_params(dataset_id=None, job_id=None, fresh="1")


def restore_workspace_state() -> None:
    if (_query_param_value("fresh") or "").lower() in {"1", "true", "yes"}:
        st.session_state["_workspace_restored"] = False
        st.session_state["_workspace_cleared"] = True
        return

    dataset_id = _query_param_value("dataset_id")
    job_id = _query_param_value("job_id")
    path = (
        f"/workspace/restore?dataset_id={dataset_id or ''}&job_id={job_id or ''}"
        if dataset_id or job_id
        else "/workspace/latest"
    )
    payload = api_json(path, timeout=10)
    if not isinstance(payload, dict) or payload.get("error"):
        return

    dataset = payload.get("dataset") or {}
    job = payload.get("job") or {}
    if dataset:
        st.session_state["dataset_id"] = dataset.get("id")
        st.session_state["profile"] = dataset.get("profile") or st.session_state.get("profile")
        st.session_state["upload_preview_records"] = dataset.get("preview_records") or []
        st.session_state["upload_ingest_summary"] = dataset.get("ingest_summary") or {}
        st.session_state["auto_detect"] = dataset.get("auto_detect")
    if job and job.get("id"):
        st.session_state["job_id"] = job.get("id")

    st.session_state["_workspace_restored"] = bool(dataset or job)
    if dataset or job:
        sync_workspace_query_params()


def api_json(path: str, timeout: int = 10):
    try:
        response = requests.get(f"{API_URL}{path}", timeout=timeout)
        if response.status_code == 200:
            return response.json()
        try:
            payload = response.json()
            detail = payload.get("detail") or payload.get("error")
        except Exception:
            detail = None
        return {"error": detail or f"HTTP {response.status_code}"}
    except Exception as exc:
        return {"error": str(exc)}


def _serialize_cell(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict, set)):
        try:
            return json.dumps(value, ensure_ascii=True, default=str)
        except Exception:
            return str(value)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def prepare_dataframe_for_display(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = pd.DataFrame(data)

    if df.empty:
        return df

    for col in df.columns:
        series = df[col]
        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        ):
            # Display tables are safer when every loose/object column is normalized
            # into a single scalar-friendly representation before Streamlit Arrow conversion.
            df[col] = series.map(_serialize_cell)
    return df


def render_safe_dataframe(data: Any, **kwargs: Any) -> None:
    st.dataframe(prepare_dataframe_for_display(data), **kwargs)


def fetch_backend_overview() -> dict:
    jobs = api_json("/jobs", timeout=5)
    if not isinstance(jobs, list):
        return {
            "backend_ok": False,
            "total": 0,
            "completed": 0,
            "running": 0,
            "failed": 0,
        }

    return {
        "backend_ok": True,
        "total": len(jobs),
        "completed": sum(1 for job in jobs if job.get("status") == "completed"),
        "running": sum(1 for job in jobs if job.get("status") == "training"),
        "failed": sum(1 for job in jobs if job.get("status") == "failed"),
    }


def render_page_shell(
    title: str,
    eyebrow: str,
    description: str,
    stats: Optional[Iterable[tuple[str, object]]] = None,
    accent: str = "default",
) -> None:
    stat_markup = ""
    for label, value in list(stats or [])[:4]:
        stat_markup += (
            '<div class="hero-stat">'
            f'<span>{label}</span><strong>{value}</strong>'
            "</div>"
        )

    accent_labels = {
        "default": "Core Mode",
        "analysis": "Analysis Mode",
        "lab": "Training Mode",
        "results": "Results Mode",
    }
    accent_label = accent_labels.get(accent, "Core Mode")

    st.markdown(
        f"""
        <section class="page-hero page-hero--{accent}">
            <div class="page-hero__grid"></div>
            <div class="page-hero__glow page-hero__glow--one"></div>
            <div class="page-hero__glow page-hero__glow--two"></div>
            <div class="page-hero__scanline"></div>
            <div class="page-hero__orbit"></div>
            <div class="page-hero__copy">
                <div class="page-hero__eyebrow">{eyebrow}</div>
                <div class="page-hero__badge">
                    <span class="page-hero__badge-dot"></span>
                    {accent_label}
                </div>
                <h1 class="page-hero__title">{title}</h1>
                <p class="page-hero__desc">{description}</p>
            </div>
            <div class="page-hero__stats">{stat_markup}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_workspace_banner() -> None:
    profile = st.session_state.get("profile") or {}
    dataset_id = st.session_state.get("dataset_id")
    job_id = st.session_state.get("job_id")

    rows = profile.get("rows", "—")
    cols = profile.get("cols") or len(profile.get("columns", []) or [])
    target = profile.get("suggested_target", "Not detected")
    job_display = job_id[:8] if job_id else "No run"
    dataset_display = dataset_id[:8] if isinstance(dataset_id, str) else (dataset_id or "No dataset")

    st.markdown(
        f"""
        <div class="workspace-banner">
            <div class="workspace-pill">
                <span>Dataset</span>
                <strong>{dataset_display}</strong>
            </div>
            <div class="workspace-pill">
                <span>Rows</span>
                <strong>{rows}</strong>
            </div>
            <div class="workspace-pill">
                <span>Columns</span>
                <strong>{cols}</strong>
            </div>
            <div class="workspace-pill">
                <span>Target</span>
                <strong>{target}</strong>
            </div>
            <div class="workspace-pill">
                <span>Active Run</span>
                <strong>{job_display}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(label: str, title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="section-intro">
            <div class="section-intro__label">{label}</div>
            <div class="section-intro__title">{title}</div>
            <div class="section-intro__text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_backend_notice(backend_ok: bool) -> None:
    state = "Connected" if backend_ok else "Offline"
    theme = "success" if backend_ok else "danger"
    text = (
        "Backend services are reachable. Uploads, training, and reports are available."
        if backend_ok
        else "Backend services are not reachable right now. Start the API on port 8000 to unlock the workflow."
    )
    st.markdown(
        f"""
        <div class="inline-notice inline-notice--{theme}">
            <strong>Backend {state}</strong>
            <span>{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
