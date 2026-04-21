import numpy as np
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

st.set_page_config(page_title="Smart AI Hub - AutoML Studio", page_icon="🤖", layout="wide")

load_css()
ensure_session_state()

dataset_id = st.session_state.get("dataset_id")
job_id = st.session_state.get("job_id")
profile = st.session_state.get("profile") or {}


def _load_jobs():
    try:
        response = requests.get(f"{API_URL}/jobs", timeout=20)
        rows = response.json() if response.status_code == 200 else []
    except Exception:
        rows = []
    return rows if isinstance(rows, list) else []


def _load_job_status(active_job_id: str):
    try:
        response = requests.get(f"{API_URL}/status/{active_job_id}", timeout=20)
        return response.json() if response.status_code == 200 else {}
    except Exception:
        return {}


render_page_shell(
    title="Smart AI Hub",
    eyebrow="AI Utility Command Deck",
    description="Move through ensemble assembly, scenario simulation, and natural-language planning in a cleaner sequence.",
    stats=[
        ("Dataset", dataset_id[:8] if dataset_id else "No dataset"),
        ("Run", job_id[:8] if job_id else "No run"),
        ("Features", len(profile.get("columns", []) or [])),
    ],
    accent="results",
)
render_workspace_banner()
render_section_intro(
    "Smart Flow",
    "Review the workspace, then build, simulate, and brief the model assistant.",
    "The Hub is now arranged as a sequential operator flow so each step has a clear setup area, action, and result surface.",
)
all_jobs = _load_jobs()
completed_jobs = [job for job in all_jobs if job.get("status") == "completed"]
workspace_ready = bool(dataset_id and profile)

summary_cols = st.columns(4)
summary_cols[0].metric("Completed Runs", len(completed_jobs))
summary_cols[1].metric("Dataset Loaded", "Yes" if dataset_id else "No")
summary_cols[2].metric("Active Job", "Ready" if job_id else "Missing")
summary_cols[3].metric("Numeric Features", len(profile.get("num_cols", []) or []))

st.markdown("### Step 1. Workspace Readiness")
readiness_cols = st.columns(3)
readiness_cols[0].markdown(
    f"""
    <div class="glass-panel" style="min-height: 180px;">
        <div class="section-label">DATASET</div>
        <div style="font-size:1.1rem; font-weight:700;">{profile.get('suggested_target') or 'Target not detected'}</div>
        <div style="margin-top:0.65rem; opacity:0.84;">{profile.get('rows', '—')} rows · {len(profile.get('columns', []) or [])} columns</div>
        <div style="margin-top:0.9rem; opacity:0.72;">Use the Dataset DNA page if you need cleanup or schema validation first.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
readiness_cols[1].markdown(
    f"""
    <div class="glass-panel" style="min-height: 180px;">
        <div class="section-label">RUN POOL</div>
        <div style="font-size:1.1rem; font-weight:700;">{len(completed_jobs)} completed jobs available</div>
        <div style="margin-top:0.65rem; opacity:0.84;">Ensemble mode unlocks once at least two completed runs exist.</div>
        <div style="margin-top:0.9rem; opacity:0.72;">Current active run: {job_id[:8] if job_id else 'none'}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
readiness_cols[2].markdown(
    f"""
    <div class="glass-panel" style="min-height: 180px;">
        <div class="section-label">STUDIO STORAGE</div>
        <div style="font-size:1.1rem; font-weight:700;">Local workspace enabled</div>
        <div style="margin-top:0.65rem; opacity:0.84;">Uploads, runs, drift checks, and saved workspace context are stored locally.</div>
        <div style="margin-top:0.9rem; opacity:0.72;">No login required for this environment.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Step 2. Ensemble Builder")
st.caption("Choose two or more completed runs, select the combination strategy, and save the resulting ensemble back into the same private workspace.")
ensemble_panel, ensemble_side = st.columns([1.5, 0.9])

with ensemble_panel:
    if len(completed_jobs) < 2:
        st.warning("At least two completed runs are needed before an ensemble can be created.")
    else:
        job_options = {
            f"{job.get('best_model') or 'Model'} · {job.get('score') or '—'} · {job['id'][:8]}": job["id"]
            for job in completed_jobs
        }
        with st.form("ensemble_form"):
            selected_labels = st.multiselect(
                "Completed runs to combine",
                options=list(job_options.keys()),
                default=list(job_options.keys())[:2],
            )
            strategy = st.radio(
                "Ensemble strategy",
                ["voting", "stacking"],
                horizontal=True,
            )
            submitted = st.form_submit_button("Create Ensemble", type="primary", use_container_width=True)

        if submitted:
            if len(selected_labels) < 2:
                st.error("Select at least two completed runs.")
            else:
                with st.spinner("Building ensemble..."):
                    response = requests.post(
                        f"{API_URL}/ensemble",
                        json={
                            "job_ids": [job_options[label] for label in selected_labels],
                            "strategy": strategy,
                            "dataset_id": dataset_id,
                        },
                        timeout=60,
                    )
                    payload = response.json() if response.status_code == 200 else {"error": f"HTTP {response.status_code}"}
                if payload.get("job_id"):
                    st.session_state["job_id"] = payload["job_id"]
                    sync_workspace_query_params()
                    st.success(f"Ensemble created with {strategy} strategy.")
                    st.page_link("pages/4_Results_Console.py", label="Open Ensemble Results", icon="📊")
                else:
                    st.error(payload.get("error") or "Failed to build ensemble.")
    st.markdown("</div>", unsafe_allow_html=True)

with ensemble_side:
    st.markdown(
        """
        <div class="glass-panel" style="min-height: 245px;">
            <div class="section-label">BUILD NOTES</div>
            <div style="font-size:1.1rem; font-weight:700;">Use when you already have strong base runs</div>
            <div style="margin-top:0.7rem; opacity:0.84;">Voting is lighter and safer. Stacking can outperform it when the source models are genuinely different.</div>
            <div style="margin-top:1rem; opacity:0.72;">Keep runs on the same target and task type to avoid invalid combinations.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Step 3. What-If Simulation")
st.caption("Lock a base scenario, sweep one feature, and inspect how predictions respond before committing to another training cycle.")
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

if not job_id:
    st.info("Load a completed run first so the simulator has a trained model to probe.")
else:
    status_payload = _load_job_status(job_id)
    job_results = status_payload.get("results") or {}
    feature_names = job_results.get("feature_names") or []
    column_stats = profile.get("column_stats") or {}
    numeric_features = [name for name in profile.get("num_cols", []) if name in feature_names]

    if not feature_names:
        st.warning("Feature names are not available for the current run yet.")
    else:
        setup_cols = st.columns([1, 1])
        with setup_cols[0]:
            sweep_feature = st.selectbox("Feature to sweep", feature_names)
        with setup_cols[1]:
            sweep_stats = column_stats.get(sweep_feature, {})
            if sweep_feature in numeric_features:
                min_val = float(sweep_stats.get("min", 0) or 0)
                max_val = float(sweep_stats.get("max", min_val + 10) or (min_val + 10))
                if min_val == max_val:
                    max_val = min_val + 1
                sweep_min, sweep_max = st.slider(
                    f"Sweep range for `{sweep_feature}`",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    step=max((max_val - min_val) / 20, 0.1),
                )
                n_steps = st.slider("Simulation points", min_value=5, max_value=40, value=15)
                sweep_values = [round(float(v), 6) for v in np.linspace(sweep_min, sweep_max, n_steps)]
            else:
                options = sweep_stats.get("top_values", [])
                sweep_values = st.multiselect(
                    f"Values for `{sweep_feature}`",
                    options=options,
                    default=options[: min(5, len(options))],
                )

        st.markdown("#### Base Scenario")
        remaining_features = [name for name in feature_names if name != sweep_feature]
        input_columns = st.columns(min(3, len(remaining_features)) if remaining_features else 1)
        base_features = {}

        for idx, feature in enumerate(remaining_features):
            stats = column_stats.get(feature, {})
            if feature in numeric_features:
                default_value = stats.get("mean", stats.get("min", 0))
            else:
                top_values = stats.get("top_values", [])
                default_value = top_values[0] if top_values else ""

            with input_columns[idx % len(input_columns)]:
                raw_value = st.text_input(feature, value=str(default_value), key=f"smart_hub_{feature}")
                try:
                    base_features[feature] = float(raw_value) if "." in raw_value else int(raw_value)
                except ValueError:
                    base_features[feature] = raw_value

        if st.button("Run Simulation", type="primary", use_container_width=True):
            if not sweep_values:
                st.warning("Choose at least one sweep value.")
            else:
                with st.spinner("Running what-if simulation..."):
                    response = requests.post(
                        f"{API_URL}/future",
                        json={
                            "job_id": job_id,
                            "base_features": base_features,
                            "sweep_feature": sweep_feature,
                            "sweep_values": sweep_values,
                        },
                        timeout=60,
                    )
                    payload = response.json() if response.status_code == 200 else {"error": f"HTTP {response.status_code}"}

                predictions = payload.get("predictions") or []
                valid_rows = [row for row in predictions if "error" not in row]
                error_rows = [row for row in predictions if "error" in row]

                if payload.get("error"):
                    st.error(payload["error"])
                elif valid_rows:
                    sim_df = pd.DataFrame(valid_rows)
                    render_safe_dataframe(sim_df, width="stretch", hide_index=True)

                    chart_df = sim_df.rename(columns={"x": sweep_feature, "prediction": "Prediction"}).set_index(sweep_feature)
                    st.line_chart(chart_df[["Prediction"]])

                    if "confidence" in sim_df.columns and sim_df["confidence"].notna().any():
                        conf_df = sim_df.rename(columns={"x": sweep_feature, "confidence": "Confidence"}).set_index(sweep_feature)
                        st.area_chart(conf_df[["Confidence"]])

                    low_row = sim_df.loc[sim_df["prediction"].idxmin()]
                    high_row = sim_df.loc[sim_df["prediction"].idxmax()]
                    metric_cols = st.columns(2)
                    metric_cols[0].metric("Lowest Prediction", round(float(low_row["prediction"]), 4), f"{sweep_feature}={low_row['x']}")
                    metric_cols[1].metric("Highest Prediction", round(float(high_row["prediction"]), 4), f"{sweep_feature}={high_row['x']}")

                if error_rows:
                    st.warning("Some simulation points failed.")
                    render_safe_dataframe(pd.DataFrame(error_rows), width="stretch", hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("### Step 4. Natural Language ML Planner")
st.caption("Turn a plain-English objective into structured training guidance that stays tied to the currently loaded dataset.")
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

st.session_state.setdefault("nl_chat_history", [])
for msg in st.session_state.nl_chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Example: Predict churn with strong recall and explainability"):
    st.session_state.nl_chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Parsing intent..."):
            try:
                response = requests.post(
                    f"{API_URL}/nl/intent",
                    json={"prompt": prompt, "dataset_id": dataset_id or ""},
                    timeout=30,
                )
                payload = response.json() if response.status_code == 200 else {"error": f"HTTP {response.status_code}"}
            except Exception as exc:
                payload = {"error": str(exc)}

        if payload.get("error"):
            reply = f"Intent parsing failed: {payload['error']}"
        else:
            reply_lines = ["Parsed plan ready for the next training pass."]
            goal = payload.get("goal")
            mode = payload.get("mode")
            target_column = payload.get("target_column")
            if goal:
                reply_lines.append(f"Goal: `{goal}`")
            if mode:
                reply_lines.append(f"Mode: `{mode}`")
            if target_column:
                reply_lines.append(f"Suggested target: `{target_column}`")
            if payload.get("selected_features"):
                reply_lines.append(f"Feature focus: `{', '.join(payload['selected_features'][:8])}`")
            reply = "\n\n".join(reply_lines)

        st.markdown(reply)
        st.session_state.nl_chat_history.append({"role": "assistant", "content": reply})

st.markdown("</div>", unsafe_allow_html=True)
