import os
import time

import pandas as pd
import requests
import streamlit as st

from ui_utils import format_metric_value

API_URL = "http://localhost:8000/api"

st.set_page_config(page_title="Results Console - AutoML Studio", page_icon="📊", layout="wide")


PROCESS_STEPS = [
    {
        "title": "1. Dataset Validation",
        "icon": "🛡️",
        "detail": "Checks target availability, drops columns with extreme missingness, enforces a schema contract, and fits a drift baseline.",
    },
    {
        "title": "2. Cleaning & Feature Prep",
        "icon": "🧹",
        "detail": "Standardizes nulls, removes broken target rows, auto-cleans noisy columns, and generates richer features in stronger modes.",
    },
    {
        "title": "3. Split & Task Detection",
        "icon": "🧠",
        "detail": "Infers classification vs regression, label-encodes targets when needed, then creates train/test splits with stratification when possible.",
    },
    {
        "title": "4. Model Pool Selection",
        "icon": "🎛️",
        "detail": "Chooses a model family and preprocessing stack from the meta-selector based on size, task type, goal, and mode.",
    },
    {
        "title": "5. Exploration Sweep",
        "icon": "🚀",
        "detail": "Runs a fast sweep on a smaller training slice to rank candidate models and capture stability signals before heavier tuning.",
    },
    {
        "title": "6. Deep Optimization",
        "icon": "🔬",
        "detail": "Balanced and Full modes tune the top models with Optuna or cross-validation, then retrain the final production pipeline.",
    },
    {
        "title": "7. Evaluation & Explainability",
        "icon": "📈",
        "detail": "Scores the final model on holdout data, builds the leaderboard, computes SHAP importance when possible, and stores artifacts.",
    },
]


def load_css():
    css_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "style.css",
    )
    try:
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass


def api_json(path: str, timeout: int = 15):
    try:
        res = requests.get(f"{API_URL}{path}", timeout=timeout)
        if res.status_code == 200:
            return res.json()
        try:
            payload = res.json()
            return {
                "error": payload.get("detail")
                or payload.get("error")
                or f"HTTP {res.status_code}"
            }
        except Exception:
            return {"error": f"HTTP {res.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def prepare_download(
    label: str,
    session_key: str,
    path: str,
    file_name: str,
    mime: str,
    timeout: int = 60,
):
    if st.button(label, use_container_width=True):
        try:
            res = requests.get(f"{API_URL}{path}", timeout=timeout)
            if res.status_code == 200:
                st.session_state[session_key] = res.content
            else:
                try:
                    detail = res.json().get("detail", "Download failed.")
                except Exception:
                    detail = "Download failed."
                st.error(detail)
        except Exception as e:
            st.error(f"Download failed: {e}")

    if st.session_state.get(session_key):
        st.download_button(
            f"⬇️ Save {file_name}",
            data=st.session_state[session_key],
            file_name=file_name,
            mime=mime,
            use_container_width=True,
            key=f"download_{session_key}",
        )


def render_status_badge(status: str):
    badge_map = {
        "completed": "completed",
        "training": "training",
        "failed": "failed",
    }
    badge_class = badge_map.get((status or "").lower(), "training")
    st.markdown(
        f'<div class="status-badge {badge_class} dot">{(status or "unknown").title()}</div>',
        unsafe_allow_html=True,
    )


def build_history_frames(history: list):
    hist_df = pd.DataFrame(history or [])
    if hist_df.empty:
        return hist_df, hist_df

    if "metric" in hist_df.columns:
        hist_df["metric_numeric"] = pd.to_numeric(hist_df["metric"], errors="coerce")
    else:
        hist_df["metric_numeric"] = pd.Series(dtype="float64")

    numeric_history = hist_df[hist_df["metric_numeric"].notna()].copy()
    if not numeric_history.empty:
        numeric_history["step_index"] = range(1, len(numeric_history) + 1)
    return hist_df, numeric_history


def build_process_overview(results: dict):
    eda_summary = results.get("eda_summary", {}) or {}
    execution_profile = results.get("execution_profile", {}) or {}
    model_metadata = results.get("model_metadata", {}) or {}

    return [
        ("Rows After Cleaning", eda_summary.get("rows_after_target_cleaning", "—")),
        (
            "Feature Columns",
            eda_summary.get(
                "columns_after_feature_engineering",
                len(results.get("feature_names", []) or []),
            ),
        ),
        ("Numeric Features", eda_summary.get("numeric_features", "—")),
        ("Categorical Features", eda_summary.get("categorical_features", "—")),
        ("Preprocessor", model_metadata.get("preprocessor", "—")),
        ("Sweep Size", execution_profile.get("sweep_size", "—")),
        ("Top-K Models", execution_profile.get("top_k", "—")),
        ("Optuna Trials", execution_profile.get("n_trials", "—")),
    ]


load_css()
st.markdown("""
<div class="top-header">
    <div>
        <div class="header-title">📊 Results Console</div>
        <div class="header-subtitle">
        Model monitoring, insights, and deployment
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
if not st.session_state.get("job_id"):
    st.info("Please load or run a training job to view results.")
    st.page_link("pages/5_Experiment_Tracker.py", label="Go to Experiment Tracker", icon="📚")
    st.stop()

job_id = st.session_state["job_id"]
job_data = api_json(f"/status/{job_id}", timeout=10)

if job_data.get("error") and not job_data.get("status"):
    st.error("Cannot reach backend.")
    st.stop()

status = job_data.get("status")
history = job_data.get("history", []) or []
reasoning = job_data.get("reasoning", []) or []
history_df, numeric_history = build_history_frames(history)

st.markdown('<div class="hero-shell">', unsafe_allow_html=True)
hero_left, hero_right = st.columns([1.7, 1])
with hero_left:
    st.markdown("### Model Mission Control")
    st.write(
        "Track the winning model, inspect the pipeline journey, compare history, and use one consolidated workspace for everything after training."
    )
    hero_strip = st.columns(4)
    hero_strip[0].metric("Job ID", job_id[:8])
    hero_strip[1].metric("History Events", len(history))
    hero_strip[2].metric("Reasoning Notes", len(reasoning))
    hero_strip[3].metric("Numeric Checkpoints", len(numeric_history))
with hero_right:
    st.markdown("### Run Status")
    render_status_badge(status or "unknown")
    if st.button("🔄 Refresh Current Run", use_container_width=True):
        st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

if status != "completed":
    if status == "failed":
        st.error(f"Training failed: {job_data.get('error') or 'Unknown error'}")
    else:
        st.warning(f"This job is currently `{status}`. Live updates are shown below.")

    st.page_link("pages/3_Training_Lab.py", label="Open Training Lab", icon="🧪")

    live_col1, live_col2 = st.columns([1, 1])
    with live_col1:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("### 📋 Live Training Log")
        if history:
            for entry in history[-20:]:
                label = entry.get("time", "Update")
                metric = entry.get("metric", "—")
                st.write(f"• `[{label}]` → **{metric}**")
        else:
            st.write("Warming up the engine...")
        st.markdown('</div>', unsafe_allow_html=True)

    with live_col2:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("### 🧠 Streaming Reasoning")
        if reasoning:
            for line in reasoning[-20:]:
                st.write(f"▫️ {line}")
        else:
            st.write("Detailed reasoning will appear here as the pipeline advances.")
        st.markdown('</div>', unsafe_allow_html=True)

    if not numeric_history.empty:
        chart_df = numeric_history.set_index("step_index")[["metric_numeric"]].rename(
            columns={"metric_numeric": "Score"}
        )
        st.line_chart(chart_df)

    if status != "failed":
        time.sleep(2)
        st.rerun()
    st.stop()

results = job_data.get("results", {}) or {}
insights = job_data.get("insights", {}) or {}
story = job_data.get("story", "No story generated.")
perf_warning = results.get("perf_warning")
suggested_fixes = results.get("suggested_fixes", [])
m_name = results.get("metric_name", "Score")
leaderboard = results.get("leaderboard", []) or []
tested_models = results.get("tested_models", []) or []
feature_names = results.get("feature_names", []) or []
process_overview = build_process_overview(results)
recommendations = api_json(f"/recommend/{job_id}", timeout=20)
jobs_snapshot = api_json("/jobs", timeout=10)

history_tab, performance_tab, analysis_tab, process_tab, playground_tab, coach_tab = st.tabs(
    [
        "🕓 History Lab",
        "🏆 Performance Hub",
        "🧬 Deep Analysis",
        "🧭 Pipeline Map",
        "🧪 Prediction Playground",
        "🤖 Neural Coach",
    ]
)

with history_tab:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    top_left, top_right = st.columns([1, 1])
    with top_left:
        st.markdown("### 🕓 Run History")
        st.caption("One place for this run’s timeline plus your recent experiments.")
    with top_right:
        if st.button("🔄 Reload History", key="reload_history", use_container_width=True):
            st.rerun()

    if history_df.empty:
        st.info("No history recorded for this run yet.")
    else:
        timeline_left, timeline_right = st.columns([1.15, 0.85])
        with timeline_left:
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        with timeline_right:
            if not numeric_history.empty:
                stage_curve = numeric_history.set_index("step_index")[["metric_numeric"]].rename(
                    columns={"metric_numeric": m_name}
                )
                st.markdown("#### Score Evolution")
                st.area_chart(stage_curve)

                stage_mix = (
                    numeric_history.groupby(numeric_history["phase"].fillna("other"))["metric_numeric"]
                    .mean()
                    .rename_axis("Phase")
                    .to_frame("Avg Score")
                )
                st.markdown("#### Average By Phase")
                st.bar_chart(stage_mix)
            else:
                st.info("Numeric checkpoints appear here once models start scoring.")

    st.markdown("---")
    st.markdown("### 🗃 Recent Jobs")
    jobs_list = jobs_snapshot if isinstance(jobs_snapshot, list) else []
    if jobs_list:
        jobs_df = pd.DataFrame(jobs_list)
        keep_cols = [
            c
            for c in ["id", "status", "best_model", "metric_name", "score", "dataset_id", "created_at", "error"]
            if c in jobs_df.columns
        ]
        st.dataframe(jobs_df[keep_cols], use_container_width=True, hide_index=True)

        if "score" in jobs_df.columns:
            score_series = pd.to_numeric(jobs_df["score"], errors="coerce")
            jobs_df = jobs_df.assign(score_numeric=score_series)
            trend = jobs_df[jobs_df["score_numeric"].notna()].head(12).copy()
            if not trend.empty:
                trend.index = [f"Run {i + 1}" for i in range(len(trend))]
                st.line_chart(trend[["score_numeric"]].rename(columns={"score_numeric": "Recent Scores"}))
    else:
        st.info("Recent jobs will appear here once more runs are available.")
    st.markdown("</div>", unsafe_allow_html=True)

with performance_tab:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    if perf_warning:
        st.error(f"### {perf_warning}")
        if suggested_fixes:
            st.markdown(f"#### 🛠 Suggested Fixes to Improve {m_name}")
            for fix in suggested_fixes:
                st.markdown(f"- {fix}")
        st.markdown("---")

    stat_cols = st.columns(4)
    stat_cols[0].metric("Winner", results.get("best_model", "—"))
    stat_cols[1].metric(m_name, format_metric_value(m_name, results.get("score", 0)))
    stat_cols[2].metric("Task", "Classification" if results.get("is_classification") else "Regression")
    stat_cols[3].metric("Features", len(feature_names))

    c1, c2 = st.columns([1.15, 0.85])
    with c1:
        st.markdown("### 🏆 Leaderboard")
        if leaderboard:
            for idx, entry in enumerate(leaderboard):
                medal = ["🥇", "🥈", "🥉"][idx] if idx < 3 else f"{idx + 1}."
                phase = entry.get("phase", "run")
                st.write(
                    f"{medal} {entry['model']} → {format_metric_value(m_name, entry.get('score'))}  ·  `{phase}`"
                )

            lb_df = pd.DataFrame(leaderboard)
            lb_df["score"] = pd.to_numeric(lb_df["score"], errors="coerce")
            st.dataframe(lb_df, use_container_width=True, hide_index=True)
            chart_df = lb_df[["model", "score"]].dropna().set_index("model")
            if not chart_df.empty:
                st.bar_chart(chart_df)

            if "phase" in lb_df.columns:
                phase_df = (
                    lb_df.groupby("phase")["score"]
                    .mean()
                    .rename_axis("Phase")
                    .to_frame("Average Score")
                )
                st.markdown("#### Average Score By Phase")
                st.bar_chart(phase_df)
        else:
            st.info("Leaderboard data not available.")

    with c2:
        st.markdown("### 🎭 Model Story")
        st.info(story)

        st.markdown("### ⚙️ Execution Profile")
        execution_profile = results.get("execution_profile", {})
        if execution_profile:
            st.json(execution_profile, expanded=False)
        else:
            st.info("Execution profile not available.")

        st.markdown("### ✨ Pipeline Snapshot")
        snap_cols = st.columns(2)
        for idx, (label, value) in enumerate(process_overview[:4]):
            snap_cols[idx % 2].metric(label, value)

    if tested_models:
        st.markdown("### 🔎 Model-by-Model Details")
        tested_df = pd.DataFrame(tested_models).copy()
        preferred_cols = [
            "model",
            "status",
            "phase",
            "winner",
            "sweep_score",
            "best_cv_score",
            "holdout_score",
            "stability_std",
            "optimized",
            "optuna_trials",
            "precision",
            "recall",
            "f1",
            "mse",
            "mae",
        ]
        existing_cols = [c for c in preferred_cols if c in tested_df.columns]
        extra_cols = [
            c
            for c in tested_df.columns
            if c not in existing_cols and c not in {"cheap_config", "best_params", "error"}
        ]
        st.dataframe(tested_df[existing_cols + extra_cols], use_container_width=True, hide_index=True)

        score_cols = [c for c in ["sweep_score", "best_cv_score", "holdout_score"] if c in tested_df.columns]
        if score_cols:
            melt_df = tested_df[["model"] + score_cols].copy().set_index("model")
            numeric_cols = melt_df.apply(pd.to_numeric, errors="coerce")
            if not numeric_cols.dropna(how="all").empty:
                st.markdown("### 📡 Model Progression")
                st.line_chart(numeric_cols)

        for row in tested_models:
            with st.expander(f"{row.get('model', 'Model')} configuration"):
                if row.get("cheap_config"):
                    st.write("Sweep config:", row["cheap_config"])
                if row.get("best_params"):
                    st.write("Best tuned params:", row["best_params"])
                if row.get("error"):
                    st.error(row["error"])
    st.markdown("</div>", unsafe_allow_html=True)

with analysis_tab:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    c_sha1, c_sha2 = st.columns([1, 1])
    with c_sha1:
        st.markdown("### 🧬 Feature Importance (SHAP)")
        shap_data = results.get("shap_summary", {}) or {}
        if shap_data:
            shap_df = pd.Series(shap_data).sort_values(ascending=False).rename("Importance")
            st.bar_chart(shap_df)
            st.dataframe(
                shap_df.reset_index().rename(columns={"index": "Feature"}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("SHAP data not available for this model.")

    with c_sha2:
        st.markdown("### 📋 Data Processing Stats")
        eda_summary = results.get("eda_summary", {}) or {}
        if eda_summary:
            eda_cols = st.columns(2)
            for idx, (label, value) in enumerate(eda_summary.items()):
                pretty = label.replace("_", " ").title()
                eda_cols[idx % 2].metric(pretty, value)
        else:
            st.info("EDA summary not available.")

        process_df = pd.DataFrame(process_overview, columns=["Metric", "Value"])
        if not process_df.empty:
            st.markdown("#### Pipeline Metrics Table")
            st.dataframe(process_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    d1, d2 = st.columns([1, 1])
    calibration = (
        api_json(f"/calibration/{job_id}", timeout=20)
        if results.get("is_classification")
        else {"error": "Calibration report is only available for classification jobs."}
    )
    thresholds = (
        api_json(f"/thresholds/{job_id}", timeout=20)
        if results.get("is_classification")
        else {"error": "Threshold tuning is only available for classification jobs."}
    )
    lineage = api_json(f"/lineage/{job_id}", timeout=20)

    with d1:
        st.markdown("### 🎯 Calibration Report")
        if calibration.get("error"):
            st.info(calibration["error"])
        else:
            st.metric("Brier Score", calibration.get("brier_score", "—"))
            bins_df = pd.DataFrame(calibration.get("bins", []))
            if not bins_df.empty:
                st.dataframe(bins_df, use_container_width=True, hide_index=True)
                chart_df = bins_df.rename(
                    columns={
                        "mean_predicted": "Mean Predicted",
                        "fraction_positive": "Actual Positive Rate",
                    }
                )
                st.line_chart(chart_df)

        st.markdown("### 🧮 Threshold Tuner")
        if thresholds.get("error"):
            st.info(thresholds["error"])
        else:
            best_threshold = thresholds.get("best_threshold") or {}
            cols = st.columns(4)
            cols[0].metric("Best Threshold", best_threshold.get("threshold", "—"))
            cols[1].metric("Precision", f"{best_threshold.get('precision', 0)}%")
            cols[2].metric("Recall", f"{best_threshold.get('recall', 0)}%")
            cols[3].metric("F1", f"{best_threshold.get('f1', 0)}%")
            threshold_df = pd.DataFrame(thresholds.get("thresholds", []))
            if not threshold_df.empty:
                st.dataframe(threshold_df, use_container_width=True, hide_index=True)
                threshold_chart = threshold_df.copy()
                for col in ["precision", "recall", "f1"]:
                    if col in threshold_chart.columns:
                        threshold_chart[col] = pd.to_numeric(threshold_chart[col], errors="coerce")
                if "threshold" in threshold_chart.columns:
                    threshold_chart = threshold_chart.set_index("threshold")
                    cols_to_chart = [c for c in ["precision", "recall", "f1"] if c in threshold_chart.columns]
                    if cols_to_chart:
                        st.line_chart(threshold_chart[cols_to_chart])

    with d2:
        st.markdown("### 🧱 Feature Lineage")
        if lineage.get("error"):
            st.info(lineage["error"])
        else:
            st.metric("Transformed Features", lineage.get("count", 0))
            lineage_df = pd.DataFrame(lineage.get("lineage", []))
            if not lineage_df.empty:
                st.dataframe(lineage_df, use_container_width=True, hide_index=True)
                transform_counts = (
                    lineage_df["transform_group"].value_counts().rename_axis("Transform").to_frame("Count")
                )
                st.bar_chart(transform_counts)

        st.markdown("### 💡 Recommendations")
        recommendation_items = recommendations.get("recommendations", [])
        if recommendation_items:
            for item in recommendation_items[:8]:
                st.markdown(f"- {item}")
        else:
            st.info("Recommendations will appear here when available.")
    st.markdown("</div>", unsafe_allow_html=True)

with process_tab:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🧭 How This Project Processes Your Data")
    st.write(
        "This run follows the backend pipeline component by component. The cards below are based on the actual pipeline code, not placeholder product copy."
    )

    stage_cols = st.columns(2)
    for idx, step in enumerate(PROCESS_STEPS):
        with stage_cols[idx % 2]:
            st.markdown(
                f"""
                <div class="process-stage">
                    <div class="process-stage-icon">{step["icon"]}</div>
                    <div>
                        <div class="process-stage-title">{step["title"]}</div>
                        <div class="process-stage-copy">{step["detail"]}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    p1, p2 = st.columns([1, 1])
    with p1:
        st.markdown("### 🧪 What Cleaning Happens")
        cleaning_points = [
            "Drops columns with more than 90% missing values.",
            "Attempts strict numeric parsing on object columns where possible.",
            "Detects and removes suspicious leakage columns early.",
            "Auto-clean mode standardizes placeholders like `n/a`, `null`, `unknown`, and invalid target rows.",
            "Stores a schema contract and drift baseline before modeling.",
            "Supports imbalance handling through balanced weights for compatible classifiers.",
        ]
        for item in cleaning_points:
            st.markdown(f"- {item}")

    with p2:
        st.markdown("### 🏗️ What Gets Produced")
        output_points = [
            "A persisted production model pipeline.",
            "Leaderboard with sweep and holdout signals.",
            "Execution profile with mode-level tuning budget.",
            "SHAP summary for explainability when supported.",
            "Reasoning trail, history checkpoints, and exportable reports.",
            "Artifacts for model bundle, PDF report, and model card export.",
        ]
        for item in output_points:
            st.markdown(f"- {item}")

    if not numeric_history.empty:
        st.markdown("### ⏱️ Pipeline Score Timeline")
        stage_timeline = numeric_history[["time", "metric_numeric"]].copy()
        stage_timeline.index = range(1, len(stage_timeline) + 1)
        st.line_chart(stage_timeline.set_index(stage_timeline.index)[["metric_numeric"]].rename(columns={"metric_numeric": m_name}))
    st.markdown("</div>", unsafe_allow_html=True)

with playground_tab:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🧪 Live Prediction")
    st.caption("The standalone Playground page has been consolidated here so analysis and interaction stay in one place.")

    if not feature_names:
        st.warning("Feature metadata missing.")
    else:
        input_cols = st.columns(3)
        feature_inputs = {}
        for i, feat in enumerate(feature_names):
            with input_cols[i % 3]:
                feature_inputs[feat] = st.text_input(feat, key=f"pred_{feat}")

        if st.button("🔮 Run Prediction", key="predict_run", use_container_width=True):
            processed = {}
            for key, value in feature_inputs.items():
                if value.strip():
                    try:
                        processed[key] = float(value) if "." in value else int(value)
                    except ValueError:
                        processed[key] = value

            if processed:
                try:
                    p_res = requests.post(
                        f"{API_URL}/predict/{job_id}",
                        json={"features": processed},
                        timeout=20,
                    )
                    if p_res.status_code == 200:
                        p_data = p_res.json()
                        st.success(f"🎯 **Predicted: {p_data['prediction']}**")
                        if "probabilities" in p_data:
                            probs_df = pd.DataFrame(
                                list((p_data.get("probabilities") or {}).items()),
                                columns=["Class", "Probability"],
                            )
                            if not probs_df.empty:
                                probs_df["Probability"] = pd.to_numeric(
                                    probs_df["Probability"], errors="coerce"
                                )
                                st.dataframe(probs_df, use_container_width=True, hide_index=True)
                                st.bar_chart(probs_df.set_index("Class"))
                    else:
                        st.error("Prediction failed.")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.info("Add at least one feature value to run a prediction.")
    st.markdown("</div>", unsafe_allow_html=True)

with coach_tab:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    coach_left, coach_right = st.columns([1, 1])
    with coach_left:
        st.markdown("### 🧠 AI Coach Insights")
        st.write(f"💭 *\"{insights.get('eli5', 'Generating ELI5...')}\"*")

        st.markdown("### 🧵 Reasoning Stream")
        if reasoning:
            with st.expander("⚡ Neural Reasoning Decoder", expanded=True):
                for r in reasoning:
                    st.write(f"▫️ {r}")
        else:
            st.info("Reasoning notes are not available.")

    with coach_right:
        st.markdown("### 💬 Model Chat")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about your model..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                try:
                    c_res = requests.post(
                        f"{API_URL}/chat",
                        json={"job_id": job_id, "prompt": prompt},
                        timeout=30,
                    )
                    ans = c_res.json().get("response", "Error")
                except Exception:
                    ans = "Connection error."
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown("### 📦 Export")
e1, e2, e3 = st.columns(3)

with e1:
    prepare_download(
        "📥 Prepare Model + Code Bundle",
        f"export_zip_{job_id}",
        f"/export/{job_id}",
        f"automl_export_{job_id[:8]}.zip",
        "application/zip",
        timeout=120,
    )

with e2:
    prepare_download(
        "📄 Prepare PDF Report",
        f"report_pdf_{job_id}",
        f"/report/{job_id}/pdf",
        f"automl_report_{job_id[:8]}.pdf",
        "application/pdf",
    )

with e3:
    prepare_download(
        "🪪 Prepare HTML Model Card",
        f"model_card_{job_id}",
        f"/report/{job_id}/model-card",
        f"model_card_{job_id[:8]}.html",
        "text/html",
    )

st.markdown("</div>", unsafe_allow_html=True)
