"""
Legacy Streamlit Drift Monitor page for CODIN compatibility workflows.

The primary product UI is now the React studio served by FastAPI.
"""

import json
import pandas as pd
import requests
import streamlit as st
from ui_shell import (
    API_URL,
    ensure_session_state,
    load_css,
    render_focus_strip,
    render_page_shell,
    render_safe_dataframe,
    render_section_intro,
    render_workspace_banner,
    sync_workspace_query_params,
)

st.set_page_config(page_title="Legacy Drift Monitor - CODIN", page_icon="📉", layout="wide")


def get_json(path: str, timeout: int = 20):
    try:
        res = requests.get(f"{API_URL}{path}", timeout=timeout)
        if res.status_code == 200:
            return res.json()
        return {"error": f"HTTP {res.status_code}"}
    except Exception as e:
        return {"error": str(e)}


load_css()
ensure_session_state()

if not st.session_state.get("job_id"):
    st.info("👈 Train a model on the Home page first.")
    st.stop()

job_id = st.session_state["job_id"]
dataset_id = st.session_state.get("dataset_id")
render_page_shell(
    title="Drift Monitor",
    eyebrow="Production Stability",
    description="Check live data against the saved training baseline, record drift history, and trigger retraining without rebuilding the workflow manually.",
    stats=[
        ("Run", job_id[:8]),
        ("Dataset", (st.session_state.get("dataset_id") or "—")[:8] if st.session_state.get("dataset_id") else "—"),
    ],
    accent="analysis",
)
render_workspace_banner()
st.info(f"Primary React studio: {API_URL}/monitoring")
render_section_intro(
    "Shift Detection",
    "This page pairs detection, cadence management, history, and retraining so drift handling stays operational instead of fragmented.",
    "Upload a fresh batch for analysis, save review cadence, then promote drifted data into a retraining job when needed.",
)
render_focus_strip(
    [
        ("Data Quality", "Preprocessing choices affect drift sensitivity and retraining stability."),
        ("Evaluation Discipline", "A saved baseline makes train-test and production comparisons meaningful."),
        ("Model Health", "Feature engineering and interpretability are easier to trust when drift is monitored continuously."),
    ]
)

if dataset_id:
    version_payload = get_json(f"/dataset/{dataset_id}/versions")
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🧾 Current Dataset vs Previous Version")
    if version_payload.get("error"):
        st.caption("Version comparison will appear once this workspace has an earlier dataset to compare against.")
    else:
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Current Rows", version_payload.get("current_rows", "—"))
        v2.metric("Previous Rows", version_payload.get("previous_rows", "—"))
        v3.metric("Row Delta", version_payload.get("row_delta", "—"))
        v4.metric("Added Columns", len(version_payload.get("added_columns", []) or []))
    st.markdown("</div>", unsafe_allow_html=True)

schedule_payload = get_json(f"/drift/{job_id}/schedule")
drift_data = st.session_state.get("drift_result") or {}
retrain_recommendation = drift_data.get("retrain_recommendation", {}) if isinstance(drift_data, dict) else {}

upload_col, retrain_col = st.columns(2)

with upload_col:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 📂 Drift Analysis")
    drift_file = st.file_uploader("Upload CSV for drift analysis", type=["csv"], key="drift_monitor_file")
    if drift_file and st.button("🔍 Run Drift Analysis", type="primary", width="stretch"):
        with st.spinner("Analysing feature distributions..."):
            try:
                resp = requests.post(
                    f"{API_URL}/drift/{job_id}",
                    files={"file": (drift_file.name, drift_file.getvalue(), "text/csv")},
                    timeout=60,
                )
                st.session_state["drift_result"] = resp.json()
            except Exception as e:
                st.session_state["drift_result"] = {"error": str(e)}
    st.markdown("</div>", unsafe_allow_html=True)

with retrain_col:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🔁 One-Click Retrain")
    if retrain_recommendation:
        recommended_goal = retrain_recommendation.get("recommended_goal", "Balanced")
        recommended_mode = retrain_recommendation.get("recommended_mode", "Balanced")
        current_model = retrain_recommendation.get("current_model", "—")
        historical_winner = retrain_recommendation.get("historical_common_winner", "—")
        st.info(
            f"Recommended lane: {recommended_goal} / {recommended_mode}\n\n"
            f"Current model: {current_model} • Historical winner: {historical_winner}"
        )
        if retrain_recommendation.get("message"):
            st.caption(retrain_recommendation["message"])
        candidate_models = retrain_recommendation.get("candidate_models", []) or []
        if candidate_models:
            st.caption(f"Challenger set: {', '.join(candidate_models[:6])}")
    else:
        st.caption("Run a drift analysis first to unlock the recommended retrain lane and challenger set.")
    retrain_file = st.file_uploader("Upload drifted CSV to retrain", type=["csv"], key="drift_retrain_file")
    if retrain_file and st.button("🚀 Retrain On Drifted Dataset", width="stretch"):
        with st.spinner("Creating dataset version and starting training..."):
            try:
                launch_context = {
                    "source": "drift_recommendation",
                    "parent_job_id": job_id,
                    "recommended_goal": retrain_recommendation.get("recommended_goal"),
                    "recommended_mode": retrain_recommendation.get("recommended_mode"),
                    "message": retrain_recommendation.get("message"),
                    "current_model": retrain_recommendation.get("current_model"),
                    "historical_winner": retrain_recommendation.get("historical_common_winner"),
                    "candidate_models": retrain_recommendation.get("candidate_models", []) or [],
                }
                resp = requests.post(
                    f"{API_URL}/drift/{job_id}/retrain",
                    files={"file": (retrain_file.name, retrain_file.getvalue(), "text/csv")},
                    data={
                        "goal_override": retrain_recommendation.get("recommended_goal", ""),
                        "mode_override": retrain_recommendation.get("recommended_mode", ""),
                        "launch_context_json": json.dumps(launch_context),
                    },
                    timeout=120,
                )
                retrain_payload = resp.json()
            except Exception as e:
                retrain_payload = {"error": str(e)}

        if retrain_payload.get("error"):
            st.error(retrain_payload["error"])
        else:
            st.session_state["dataset_id"] = retrain_payload.get("dataset_id")
            if retrain_payload.get("profile"):
                st.session_state["profile"] = retrain_payload["profile"]
            st.session_state["job_id"] = retrain_payload.get("job_id")
            sync_workspace_query_params()
            if retrain_recommendation:
                st.success(
                    f"Started retraining job `{retrain_payload.get('job_id', '')[:8]}` "
                    f"using the recommended lane {retrain_recommendation.get('recommended_goal', 'Balanced')} / "
                    f"{retrain_recommendation.get('recommended_mode', 'Balanced')}."
                )
            else:
                st.success(f"Started retraining job `{retrain_payload.get('job_id', '')[:8]}` on the drifted dataset.")
            st.page_link("pages/3_Training_Lab.py", label="Open Live Training", icon="🧪")
    st.markdown("</div>", unsafe_allow_html=True)

if drift_data:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    if drift_data.get("error"):
        st.error(f"❌ {drift_data['error']}")
    else:
        overall = drift_data.get("overall_status", "")
        drift_score = drift_data.get("drift_score_pct", 0)
        drifted = drift_data.get("drifted_features", [])
        critical = drift_data.get("critical_features", [])
        alert_msg = drift_data.get("alert_message", "")
        feature_drift = drift_data.get("feature_drift", [])
        total = drift_data.get("total_features_checked", 0)
        drifted_count = drift_data.get("drifted_count", 0)

        if "Critical" in overall:
            st.error(f"### {overall}")
        elif "Moderate" in overall:
            st.warning(f"### {overall}")
        else:
            st.success(f"### {overall}")

        if alert_msg:
            st.markdown(f"> ⚠️ **{alert_msg}**")

        alert_summary = drift_data.get("alert_summary", {}) or {}
        thresholds = drift_data.get("thresholds", {}) or {}
        if alert_summary:
            st.markdown("### 🚨 Drift Alert Summary")
            alert_cols = st.columns(3)
            alert_cols[0].metric("Alert Level", alert_summary.get("level", "—").title())
            alert_cols[1].metric("Warning PSI", thresholds.get("warning_psi", "—"))
            alert_cols[2].metric("Critical PSI", thresholds.get("critical_psi", "—"))
            st.caption(alert_summary.get("recommended_action", ""))

        if retrain_recommendation:
            st.markdown("### 🧭 Retrain Recommendation")
            rec1, rec2, rec3, rec4 = st.columns(4)
            rec1.metric("Goal", retrain_recommendation.get("recommended_goal", "—"))
            rec2.metric("Mode", retrain_recommendation.get("recommended_mode", "—"))
            rec3.metric("Current Winner", retrain_recommendation.get("current_model", "—"))
            rec4.metric("Historical Winner", retrain_recommendation.get("historical_common_winner", "—"))
            if retrain_recommendation.get("message"):
                st.caption(retrain_recommendation["message"])
            candidate_models = retrain_recommendation.get("candidate_models", []) or []
            if candidate_models:
                st.write("Challenger models:", ", ".join(candidate_models[:8]))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Drift Score", f"{drift_score}%")
        c2.metric("Features Checked", total)
        c3.metric("Drifted Features", drifted_count)
        c4.metric("Critical Features", len(critical))

        if feature_drift:
            rows = []
            for item in feature_drift:
                rows.append(
                    {
                        "Feature": item.get("feature", ""),
                        "Status": item.get("severity", "Stable"),
                        "PSI": item.get("psi"),
                        "KS p-value": item.get("ks_p_value"),
                        "Current Mean": item.get("current_mean"),
                        "Baseline Mean": item.get("baseline_mean"),
                    }
                )
            feature_df = pd.DataFrame(rows)
            render_safe_dataframe(feature_df, width="stretch", hide_index=True)
            if not feature_df.empty and feature_df["PSI"].notna().any():
                psi_chart = feature_df[["Feature", "PSI"]].dropna().set_index("Feature")
                st.bar_chart(psi_chart)

        if critical:
            st.error(f"Critical drift features: {', '.join(critical)}")
        elif drifted:
            st.warning(f"Drifted features: {', '.join(drifted[:10])}")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown("### ⏱ Drift Check Cadence")
cad1, cad2, cad3 = st.columns([1, 1, 1])
enabled_default = bool(schedule_payload.get("enabled", True)) if not schedule_payload.get("error") else True
freq_default = int(schedule_payload.get("frequency_days", 7)) if not schedule_payload.get("error") else 7
warning_default = float(schedule_payload.get("warning_threshold", 0.1)) if not schedule_payload.get("error") else 0.1
critical_default = float(schedule_payload.get("critical_threshold", 0.2)) if not schedule_payload.get("error") else 0.2
enabled_value = cad1.toggle("Enable saved cadence", value=enabled_default)
freq_value = cad2.selectbox("Check every", options=[1, 3, 7, 14, 30], index=[1, 3, 7, 14, 30].index(freq_default) if freq_default in [1, 3, 7, 14, 30] else 2)
if not schedule_payload.get("error"):
    if schedule_payload.get("due_now"):
        cad3.warning("A drift review is due now for this job.")
    else:
        cad3.caption(f"Next due: {schedule_payload.get('next_due_at') or 'after first check'}")
else:
    cad3.caption("This saves the preferred review cadence for this job and pairs with the drift history below.")

th1, th2, th3 = st.columns([1, 1, 1])
warning_value = th1.number_input("Warning PSI", min_value=0.01, max_value=1.0, value=warning_default, step=0.01)
critical_value = th2.number_input("Critical PSI", min_value=warning_value, max_value=2.0, value=max(critical_default, warning_value), step=0.01)
last_alert_summary = schedule_payload.get("last_alert_summary", {}) if not schedule_payload.get("error") else {}
if last_alert_summary:
    th3.caption(f"Latest alert: {(schedule_payload.get('last_alert_status') or 'unknown').title()}")
    th3.caption(last_alert_summary.get("message", ""))
else:
    th3.caption("Thresholds apply to new drift checks and saved alert summaries.")

if st.button("💾 Save Drift Cadence", width="stretch"):
    try:
        res = requests.post(
            f"{API_URL}/drift/{job_id}/schedule",
            params={
                "enabled": enabled_value,
                "frequency_days": freq_value,
                "warning_threshold": warning_value,
                "critical_threshold": critical_value,
            },
            timeout=20,
        )
        if res.status_code == 200:
            st.success("Drift cadence saved.")
        else:
            st.error("Could not save drift cadence.")
    except Exception as e:
        st.error(f"Could not save drift cadence: {e}")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown("### 🗃 Drift History")
history_payload = get_json(f"/drift/{job_id}/history")
if history_payload.get("error"):
    st.info("No saved drift history yet.")
else:
    history_df = pd.DataFrame(history_payload.get("history", []))
    if history_df.empty:
        st.info("No drift checks saved for this job yet.")
    else:
        render_safe_dataframe(history_df, width="stretch", hide_index=True)
        status_counts = history_df["status"].value_counts().rename_axis("Status").to_frame("Checks")
        st.bar_chart(status_counts)
st.markdown("</div>", unsafe_allow_html=True)
