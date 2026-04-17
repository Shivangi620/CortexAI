import streamlit as st
import requests
import time
from ui_utils import format_metric_value
from ui_shell import (
    ensure_session_state,
    load_css,
    render_page_shell,
    render_section_intro,
    render_workspace_banner,
)

API_URL = "http://localhost:8000/api"

st.set_page_config(page_title="Live Training & Results", page_icon="⚡", layout="wide")

load_css()
ensure_session_state()

if not st.session_state.get('job_id'):
    st.info("👈 Start a training job from the Home page first.")
    st.stop()

job_id = st.session_state['job_id']

# ── Fetch current status ──────────────────────────────────────────────────────
try:
    res = requests.get(f"{API_URL}/status/{job_id}", timeout=5)
    job_data = res.json() if res.status_code == 200 else {"error": f"HTTP {res.status_code}"}
except Exception:
    st.error("Cannot reach backend.")
    st.stop()

if job_data.get("error") and job_data.get("status") != "failed":
    st.error(f"Backend status error: {job_data.get('error')}")

status = job_data.get('status', 'unknown')
render_page_shell(
    title="Training Lab",
    eyebrow="Live Execution",
    description="Monitor the active run, watch score checkpoints, and move into the results workspace as soon as the winning model lands.",
    stats=[
        ("Run", job_id[:8]),
        ("Status", status.title()),
        ("History Events", len(job_data.get("history", []) or [])),
        ("Reasoning Notes", len(job_data.get("reasoning", []) or [])),
    ],
    accent="lab",
)
render_workspace_banner()
render_section_intro(
    "Run Monitor",
    "The lab keeps the training log and score trajectory visible while the backend works.",
    "When the run completes, this page turns into a launchpad straight into the deeper results console.",
)

# ── Live polling (replaces asyncio.run — Streamlit-safe) ──────────────────────
if status == "training":
    st.info("⏳ Training in progress... page refreshes automatically.")

    history = job_data.get('history', [])
    reasoning = job_data.get('reasoning', [])

    col_log, col_chart = st.columns([1, 1])

    with col_log:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("### 📋 Training Log")
        if not history:
            st.write("Warming up the engine...")
        for entry in history:
            name = entry.get("time", "?")
            metric_v = entry.get("metric")
            icon = "⭐" if name == "Final" else "👉"
            metric_txt = f"{metric_v}%" if isinstance(metric_v, (int, float)) else str(metric_v)
            st.write(f"{icon} `[{name}]` → **{metric_txt}**")
        if reasoning:
            st.markdown("### 🧠 Live Reasoning")
            for line in reasoning[-12:]:
                st.write(f"▫️ {line}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_chart:
        if len(history) > 1:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.markdown("### 📈 Live Metric Chart")
            chart_data = {e['time']: e['metric'] for e in history if isinstance(e['metric'], (int, float))}
            if chart_data:
                import pandas as pd
                chart_df = pd.DataFrame({"Score %": list(chart_data.values())},
                                         index=list(chart_data.keys()))
                st.line_chart(chart_df)
            st.markdown('</div>', unsafe_allow_html=True)

    # Rerun every 2s — standard Streamlit polling pattern (no asyncio needed)
    time.sleep(2)
    st.rerun()

    # ── Results display ───────────────────────────────────────────────────────────
elif status == 'completed':
    st.success("🎉 Training Completed!")
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🥇 Winner Summary")
    results = job_data.get('results', {})
    best_model = results.get('best_model', 'Unknown')
    score = results.get('score', 0)
    m_name = results.get('metric_name', 'Score')
    
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.metric("Best Model", best_model)
    with col_w2:
        st.metric(m_name, format_metric_value(m_name, score))

    exec_profile = results.get("execution_profile", {})
    if exec_profile:
        st.caption(
            f"Mode profile: sweep_size={exec_profile.get('sweep_size')}, "
            f"top_k={exec_profile.get('top_k')}, optuna={exec_profile.get('run_optuna')}, "
            f"trials={exec_profile.get('n_trials')}"
        )
    
    st.markdown("---")
    st.write("Training is finished. Deep insights, SHAP values, and the prediction playground are now available in the Results Console.")
    
    if st.button("🔍 View Deep Analysis in Results Console", type="primary", width="stretch"):
        st.switch_page("pages/4_Results_Console.py")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif status == 'failed':
    st.error(f"❌ Training failed: {job_data.get('error')}")
else:
    st.warning(f"Unknown job status: `{status}`")
