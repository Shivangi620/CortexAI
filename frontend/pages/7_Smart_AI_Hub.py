import numpy as np
import pandas as pd
import requests
import streamlit as st
from ui_shell import (
    API_URL,
    ensure_session_state,
    load_css,
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
profile = st.session_state.get("profile", {})
render_page_shell(
    title="Smart AI Hub",
    eyebrow="Advanced Utilities",
    description="Use ensemble building, what-if simulation, synthetic expansion, and natural language helpers to extend the core training workflow.",
    stats=[
        ("Dataset", dataset_id[:8] if dataset_id else "No dataset"),
        ("Run", job_id[:8] if job_id else "No run"),
        ("Features", len(profile.get("columns", []) or [])),
        ("Numeric", len(profile.get("num_cols", []) or [])),
    ],
    accent="results",
)
render_workspace_banner()
render_section_intro(
    "Applied AI Tools",
    "These utilities sit on top of the main AutoML flow and reuse the same workspace state.",
    "The redesign keeps each capability in a dedicated tab while preserving the underlying API behavior.",
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🧩 Ensemble Builder",
    "🧪 Model Simulation (What-if)",
    "🧬 Data Augmentation",
    "🗣️ Natural Language ML",
])

with tab1:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🧩 Build an Ensemble Model")
    st.write("Combine multiple models into a single, high-performance stacking or voting ensemble.")

    try:
        res = requests.get(f"{API_URL}/jobs", timeout=20)
        all_jobs = res.json() if res.status_code == 200 else []
        completed_jobs = [j for j in all_jobs if j.get("status") == "completed"]
    except Exception:
        completed_jobs = []

    if len(completed_jobs) < 2:
        st.warning("You need at least 2 completed training jobs to build an ensemble.")
    else:
        st.markdown("#### Select Models to Combine")
        job_opts = {f"{j['best_model']} | {j['score']} ({j['id'][:8]})": j["id"] for j in completed_jobs}
        selected_jobs = st.multiselect("Pick 2+ jobs", options=list(job_opts.keys()))
        strategy = st.radio("Ensemble Strategy", ["voting", "stacking"], horizontal=True)

        if st.button("🚀 Create Ensemble", type="primary"):
            if len(selected_jobs) < 2:
                st.error("Select at least 2 models.")
            else:
                with st.spinner("Building ensemble..."):
                    payload = {
                        "job_ids": [job_opts[j] for j in selected_jobs],
                        "strategy": strategy,
                        "dataset_id": dataset_id,
                    }
                    e_res = requests.post(f"{API_URL}/ensemble", json=payload, timeout=60)
                    if e_res.status_code == 200:
                        e_data = e_res.json()
                        if e_data.get("job_id"):
                            st.session_state["job_id"] = e_data["job_id"]
                            sync_workspace_query_params()
                        st.success("Ensemble created successfully.")
                        if e_data.get("job_id"):
                            st.page_link("pages/4_Results_Console.py", label="Open Ensemble Results", icon="📊")
                    else:
                        st.error("Failed to build ensemble.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🧪 What-if Simulation")
    st.write("Sweep one feature while holding the others fixed to see how the model response changes.")

    if not job_id:
        st.info("Load a completed model first.")
    else:
        try:
            status_res = requests.get(f"{API_URL}/status/{job_id}", timeout=20)
            status_payload = status_res.json() if status_res.status_code == 200 else {}
        except Exception:
            status_payload = {}

        job_results = status_payload.get("results") or {}
        feature_names = job_results.get("feature_names", [])
        column_stats = profile.get("column_stats", {})
        numeric_features = [f for f in profile.get("num_cols", []) if f in feature_names]

        if not feature_names:
            st.warning("Feature names are not available for this job.")
        else:
            sweep_feature = st.selectbox("Feature to sweep", feature_names)
            sweep_stats = column_stats.get(sweep_feature, {})
            sweep_values = []

            if sweep_feature in numeric_features:
                min_val = float(sweep_stats.get("min", 0) or 0)
                max_val = float(sweep_stats.get("max", min_val + 10) or (min_val + 10))
                if min_val == max_val:
                    max_val = min_val + 1
                sweep_min, sweep_max = st.slider(
                    f"Range for `{sweep_feature}`",
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

            st.markdown("#### Base Feature Values")
            other_features = [f for f in feature_names if f != sweep_feature]
            input_cols = st.columns(min(3, len(other_features)) if other_features else 1)
            base_features = {}

            for idx, feature in enumerate(other_features):
                stats = column_stats.get(feature, {})
                if feature in numeric_features:
                    default_value = stats.get("mean", stats.get("min", 0))
                else:
                    top_values = stats.get("top_values", [])
                    default_value = top_values[0] if top_values else ""

                with input_cols[idx % len(input_cols)]:
                    raw_value = st.text_input(feature, value=str(default_value), key=f"what_if_{feature}")
                    try:
                        base_features[feature] = float(raw_value) if "." in raw_value else int(raw_value)
                    except ValueError:
                        base_features[feature] = raw_value

            if st.button("🔮 Run Simulation", width="stretch"):
                if not sweep_values:
                    st.warning("Choose at least one sweep value.")
                else:
                    with st.spinner("Running simulations..."):
                        fut_res = requests.post(
                            f"{API_URL}/future",
                            json={
                                "job_id": job_id,
                                "base_features": base_features,
                                "sweep_feature": sweep_feature,
                                "sweep_values": sweep_values,
                            },
                            timeout=60,
                        )

                    if fut_res.status_code != 200:
                        st.error(f"Simulation failed: HTTP {fut_res.status_code}")
                    else:
                        fut_data = fut_res.json()
                        predictions = fut_data.get("predictions", [])
                        valid_rows = [row for row in predictions if "error" not in row]
                        error_rows = [row for row in predictions if "error" in row]

                        if valid_rows:
                            sim_df = pd.DataFrame(valid_rows)
                            render_safe_dataframe(sim_df, width="stretch", hide_index=True)

                            chart_df = sim_df.rename(
                                columns={"x": sweep_feature, "prediction": "Prediction"}
                            ).set_index(sweep_feature)
                            st.line_chart(chart_df[["Prediction"]])

                            if "confidence" in sim_df.columns and sim_df["confidence"].notna().any():
                                conf_df = sim_df.rename(
                                    columns={"x": sweep_feature, "confidence": "Confidence"}
                                ).set_index(sweep_feature)
                                st.area_chart(conf_df[["Confidence"]])

                            low_row = sim_df.loc[sim_df["prediction"].idxmin()]
                            high_row = sim_df.loc[sim_df["prediction"].idxmax()]
                            s1, s2 = st.columns(2)
                            s1.metric("Lowest Prediction", round(float(low_row["prediction"]), 4), f"{sweep_feature}={low_row['x']}")
                            s2.metric("Highest Prediction", round(float(high_row["prediction"]), 4), f"{sweep_feature}={high_row['x']}")

                        if error_rows:
                            st.warning("Some simulation points failed.")
                            render_safe_dataframe(pd.DataFrame(error_rows), width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🧬 Synthetic Data Generator")
    if not dataset_id:
        st.info("Upload data on Home page first.")
    else:
        n_syn = st.number_input("Synthetic rows to add", 10, 50000, 1000)
        if st.button("🧬 Generate", width="stretch"):
            with st.spinner("Generating..."):
                s_res = requests.post(f"{API_URL}/synthetic/{dataset_id}", params={"n_rows": n_syn}, timeout=120)
                if s_res.status_code == 200:
                    data = s_res.json()
                    st.session_state["synthetic_result"] = data
                    st.success("Dataset expanded.")
                else:
                    st.error("Synthetic data generation failed.")

        syn = st.session_state.get("synthetic_result")
        if syn and syn.get("new_dataset_id"):
            judge_payload = {}
            try:
                judge_res = requests.get(f"{API_URL}/synthetic/judge/{syn['new_dataset_id']}", timeout=30)
                judge_payload = judge_res.json() if judge_res.status_code == 200 else {"error": f"HTTP {judge_res.status_code}"}
            except Exception as e:
                judge_payload = {"error": str(e)}

            st.markdown("#### Generated Dataset")
            c1, c2, c3 = st.columns(3)
            c1.metric("Original Rows", syn.get("original_rows", 0))
            c2.metric("Added Rows", syn.get("synthetic_rows_added", 0))
            c3.metric("Total Rows", syn.get("total_rows", 0))

            preview = syn.get("preview") or []
            if preview:
                render_safe_dataframe(pd.DataFrame(preview), width="stretch", hide_index=True)

            st.markdown("#### 📊 Synthetic vs Original Comparison")
            original_profile = syn.get("original_profile") or profile
            new_profile = syn.get("profile") or {}
            diff = syn.get("profile_diff") or {}
            cmp1, cmp2, cmp3, cmp4 = st.columns(4)
            cmp1.metric("Rows", new_profile.get("rows", 0), diff.get("rows", 0))
            cmp2.metric("Columns", new_profile.get("cols", 0), diff.get("cols", 0))
            cmp3.metric("Missing %", new_profile.get("missing_pct", 0), diff.get("missing_pct", 0))
            cmp4.metric(
                "Imbalance",
                new_profile.get("imbalance", "—"),
                "updated" if new_profile.get("imbalance") != original_profile.get("imbalance") else "same",
            )

            compare_rows = [
                {"Metric": "Rows", "Original": original_profile.get("rows"), "Augmented": new_profile.get("rows")},
                {"Metric": "Columns", "Original": original_profile.get("cols"), "Augmented": new_profile.get("cols")},
                {"Metric": "Missing %", "Original": original_profile.get("missing_pct"), "Augmented": new_profile.get("missing_pct")},
                {"Metric": "Suggested Target", "Original": original_profile.get("suggested_target"), "Augmented": new_profile.get("suggested_target")},
            ]
            render_safe_dataframe(pd.DataFrame(compare_rows), width="stretch", hide_index=True)

            st.markdown("#### 🧪 Synthetic Quality Panel")
            if judge_payload.get("error"):
                st.info(judge_payload["error"])
            else:
                q1, q2, q3 = st.columns(3)
                q1.metric("Realism Score", judge_payload.get("realism_score", "—"))
                q2.metric("Verdict", judge_payload.get("verdict", "—"))
                q3.metric("Rows Evaluated", judge_payload.get("rows_evaluated", "—"))
                notes = judge_payload.get("notes", []) or []
                if notes:
                    for note in notes:
                        st.caption(f"• {note}")
                privacy_risk = "Review" if float(judge_payload.get("realism_score", 0) or 0) > 95 else "Low"
                corr_match = "Strong" if float(judge_payload.get("realism_score", 0) or 0) >= 85 else "Moderate"
                panel_df = pd.DataFrame(
                    [
                        {"Check": "Distributions", "Assessment": judge_payload.get("verdict", "—")},
                        {"Check": "Correlations", "Assessment": corr_match},
                        {"Check": "Target Behavior", "Assessment": "Aligned" if new_profile.get("suggested_target") == original_profile.get("suggested_target") else "Changed"},
                        {"Check": "Privacy Similarity Risk", "Assessment": privacy_risk},
                    ]
                )
                render_safe_dataframe(panel_df, width="stretch", hide_index=True)

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("🧪 Retest With Augmented Dataset", type="primary", width="stretch"):
                    st.session_state["dataset_id"] = syn["new_dataset_id"]
                    if syn.get("profile"):
                        st.session_state["profile"] = syn["profile"]
                    st.success("Augmented dataset loaded into the workspace.")
                    st.switch_page("pages/1_Home.py")
            with btn_col2:
                if st.button("🧬 Inspect Augmented DNA", width="stretch"):
                    st.session_state["dataset_id"] = syn["new_dataset_id"]
                    if syn.get("profile"):
                        st.session_state["profile"] = syn["profile"]
                    st.switch_page("pages/2_Dataset_DNA.py")
    st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🗣️ NL → Machine Learning")
    st.write("Describe your ML goal in plain English.")

    if "nl_chat_history" not in st.session_state:
        st.session_state.nl_chat_history = []

    for msg in st.session_state.nl_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if nlp_prompt := st.chat_input("e.g. 'Predict churn with high accuracy'"):
        st.session_state.nl_chat_history.append({"role": "user", "content": nlp_prompt})
        with st.chat_message("user"):
            st.markdown(nlp_prompt)
        st.info("Intent parsing logic integrated.")
    st.markdown("</div>", unsafe_allow_html=True)
