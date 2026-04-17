import os
from typing import Iterable, Optional

import requests
import streamlit as st

API_URL = "http://localhost:7860/api"

SESSION_DEFAULTS = {
    "dataset_id": None,
    "profile": None,
    "job_id": None,
    "auto_detect": None,
    "last_analyzed_file": None,
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
