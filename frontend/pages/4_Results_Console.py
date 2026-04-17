import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

from ui_utils import format_metric_value
from ui_shell import (
    ensure_session_state,
    load_css,
    render_page_shell,
    render_section_intro,
    render_workspace_banner,
)

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
    if st.button(label, width="stretch"):
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
            width="stretch",
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


def build_contract_check(feature_names, profile):
    expected = set(feature_names or [])
    actual = set(profile.get("columns") or [])
    if not expected:
        return {"status": "Unknown", "missing": [], "extra": []}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    return {
        "status": "Aligned" if not missing else "Drift Risk",
        "missing": missing,
        "extra": extra,
    }


def build_failure_analysis(results, perf_warning, suggested_fixes, profile):
    items = []
    if perf_warning:
        items.append(perf_warning)
    try:
        score = float(results.get("score") or 0)
    except Exception:
        score = 0
    if score and score < 70:
        items.append("This run is under the usual strong-performance range, so feature signal or data quality is probably limiting the model.")
    try:
        missing_pct = float(profile.get("missing_pct") or 0)
    except Exception:
        missing_pct = 0
    if missing_pct > 15:
        items.append("High missingness likely increased preprocessing pressure and reduced usable signal.")
    if str(profile.get("imbalance") or "").lower() not in {"", "low", "balanced"}:
        items.append("Target imbalance is likely contributing to unstable class performance.")
    items.extend(suggested_fixes[:4])
    return items

load_css()
ensure_session_state()
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

render_page_shell(
    title="Results Console",
    eyebrow="Post-Training Workspace",
    description="Move from score inspection into explainability, recommendations, exports, prediction testing, and run history without leaving the same workspace.",
    stats=[
        ("Run", job_id[:8]),
        ("Status", (status or "unknown").title()),
        ("History", len(history)),
        ("Checkpoints", len(numeric_history)),
    ],
    accent="results",
)
render_workspace_banner()
render_section_intro(
    "Outcome Review",
    "Everything after training is organized into focused tabs so you can validate the model before exporting or deploying it.",
    "The page keeps live-state handling too, so incomplete jobs still surface logs and reasoning instead of sending you somewhere else.",
)

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
    if st.button("🔄 Refresh Current Run", width="stretch"):
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
dataset_id = st.session_state.get("dataset_id")
profile = st.session_state.get("profile") or {}
contract_check = build_contract_check(feature_names, profile)
failure_analysis = build_failure_analysis(results, perf_warning, suggested_fixes, profile)
narrator = api_json(f"/narrate/{job_id}", timeout=20)
trust_heatmap = api_json(f"/trust/{job_id}", timeout=20)

performance_tab, analysis_tab, process_tab, playground_tab, augmentation_tab, coach_tab = st.tabs(
    [
        "🏆 Performance",
        "🧬 Analysis",
        "🧭 Pipeline Map",
        "🧪 Playground",
        "🧬 Augmentation",
        "🤖 AI Coach",
    ]
)

st.markdown(
    f"""
    <div class="command-grid">
        <div class="command-card">
            <span class="command-card__eyebrow">Model</span>
            <div class="command-card__title">{results.get("best_model", "Unknown")}</div>
            <div class="command-card__copy">Best-performing model selected from the sweep and optimization pipeline.</div>
            <div class="command-card__meta">{m_name}: {format_metric_value(m_name, results.get("score", 0))}</div>
        </div>
        <div class="command-card">
            <span class="command-card__eyebrow">Schema</span>
            <div class="command-card__title">{contract_check['status']}</div>
            <div class="command-card__copy">Feature contract alignment between the trained model and the current workspace dataset.</div>
            <div class="command-card__meta">Missing: {len(contract_check['missing'])} · Extra: {len(contract_check['extra'])}</div>
        </div>
        <div class="command-card">
            <span class="command-card__eyebrow">Process</span>
            <div class="command-card__title">{len(process_overview)} tracked signals</div>
            <div class="command-card__copy">Pipeline state, preprocessing footprint, and optimization budget from the current run.</div>
            <div class="command-card__meta">Run {job_id[:8]}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
            st.dataframe(lb_df, width="stretch", hide_index=True)
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

        st.markdown("### 📝 Experiment Narrator")
        if narrator.get("error"):
            st.info("Narrative is not available right now.")
        else:
            st.caption(narrator.get("narrative", "Narrative unavailable."))

        st.markdown("### 🛠 Failure Analyzer")
        if failure_analysis:
            for item in failure_analysis[:6]:
                st.caption(f"• {item}")
        else:
            st.caption("No major warning signals were detected for this run.")

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
        st.dataframe(tested_df[existing_cols + extra_cols], width="stretch", hide_index=True)

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
                width="stretch",
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
            process_df["Value"] = process_df["Value"].map(
                lambda value: value if isinstance(value, (int, float, bool, np.number)) or value is None else str(value)
            )
            st.markdown("#### Pipeline Metrics Table")
            st.dataframe(process_df, width="stretch", hide_index=True)

        if eda_summary.get("pca_applied"):
            st.success(
                f"PCA was applied to the numeric branch using {eda_summary.get('pca_components_used', '—')} components."
            )
        else:
            st.caption("PCA was not needed for this run.")

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
    drift_feature_timeline = api_json(f"/drift/{job_id}/feature-timeline", timeout=20)

    with d1:
        st.markdown("### 🎯 Calibration Report")
        if calibration.get("error"):
            st.info(calibration["error"])
        else:
            st.metric("Brier Score", calibration.get("brier_score", "—"))
            bins_df = pd.DataFrame(calibration.get("bins", []))
            if not bins_df.empty:
                st.dataframe(bins_df, width="stretch", hide_index=True)
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
                st.dataframe(threshold_df, width="stretch", hide_index=True)
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
                st.dataframe(lineage_df, width="stretch", hide_index=True)
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

        st.markdown("### 🔥 Trust Heatmap")
        if trust_heatmap.get("error"):
            st.info(trust_heatmap["error"])
        else:
            trust_df = pd.DataFrame(trust_heatmap.get("rows", []))
            if trust_df.empty:
                st.info("Trust heatmap is still warming up. More historical runs improve this panel.")
            else:
                st.dataframe(trust_df, width="stretch", hide_index=True)
                trust_counts = trust_df["status"].value_counts().rename_axis("Status").to_frame("Features")
                st.bar_chart(trust_counts)

    st.markdown("---")
    st.markdown("### 🌊 Feature Drift Timeline")
    if drift_feature_timeline.get("error"):
        st.info(drift_feature_timeline["error"])
    else:
        timeline_df = pd.DataFrame(drift_feature_timeline.get("timeline", []))
        if timeline_df.empty:
            st.info("No drift checks have been recorded yet. Run drift analysis to build a per-feature timeline.")
        else:
            feature_options = drift_feature_timeline.get("features", [])
            selected_feature = st.selectbox(
                "Select feature for drift timeline",
                options=feature_options,
                key="drift_feature_timeline_select",
            ) if feature_options else None

            if selected_feature:
                filtered_timeline = timeline_df[timeline_df["feature"] == selected_feature].copy()
            else:
                filtered_timeline = timeline_df.copy()

            if not filtered_timeline.empty:
                filtered_timeline["created_at"] = filtered_timeline["created_at"].fillna("unknown")
                display_cols = [
                    c for c in ["created_at", "uploaded_name", "feature", "psi", "ks_p_value", "severity", "drift_detected", "current_mean", "baseline_mean"]
                    if c in filtered_timeline.columns
                ]
                st.dataframe(filtered_timeline[display_cols], width="stretch", hide_index=True)
                if "psi" in filtered_timeline.columns:
                    psi_chart = filtered_timeline[["created_at", "psi"]].copy().set_index("created_at")
                    psi_chart["psi"] = pd.to_numeric(psi_chart["psi"], errors="coerce")
                    if not psi_chart.dropna().empty:
                        st.line_chart(psi_chart.rename(columns={"psi": "PSI"}))
    st.info("Scenario simulator is available in Smart AI Hub and pairs well with this page for controlled what-if testing.")
    st.markdown("</div>", unsafe_allow_html=True)

with process_tab:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🧭 How This Project Processes Your Data")
    st.write(
        "This run follows the backend pipeline component by component. The cards below are based on the actual pipeline code, not placeholder product copy."
    )

    history_labels = [str(entry.get("metric", "")).lower() for entry in history]
    stage_cols = st.columns(2)
    for idx, step in enumerate(PROCESS_STEPS):
        step_status = "Queued"
        if any(keyword in " ".join(history_labels) for keyword in [
            step["title"].split(". ", 1)[-1].split(" & ")[0].lower(),
            step["detail"].split(",")[0].lower()
        ]):
            step_status = "Observed"
        if status == "completed" and idx == len(PROCESS_STEPS) - 1:
            step_status = "Completed"
        with stage_cols[idx % 2]:
            st.markdown(
                f"""
                <div class="process-stage">
                    <div class="process-stage-icon">{step["icon"]}</div>
                    <div>
                        <div class="process-stage-title">{step["title"]}</div>
                        <div class="process-stage-copy">{step["detail"]}</div>
                        <div class="process-stage-copy"><strong>Status:</strong> {step_status}</div>
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
    st.markdown("### 🧪 Scenario Playground")
    st.caption("Use profile-aware defaults and interactive controls to test what happens when feature values move.")

    st.markdown("#### 🧩 Feature Contract Checker")
    if contract_check["status"] == "Aligned":
        st.success("Training schema and current workspace schema look aligned.")
    elif contract_check["status"] == "Drift Risk":
        st.warning("Current workspace schema differs from the training feature contract.")
    else:
        st.info("Feature contract information is limited for this run.")
    if contract_check["missing"]:
        st.caption(f"Missing expected features: {', '.join(contract_check['missing'][:8])}")
    if contract_check["extra"]:
        st.caption(f"Extra workspace columns: {', '.join(contract_check['extra'][:8])}")

    uploaded_contract_file = st.file_uploader(
        "Upload an inference file to validate against the trained contract",
        type=None,
        key="contract_checker_upload",
    )
    if uploaded_contract_file is not None and st.button("🔍 Check Uploaded Inference File", width="stretch", key="run_contract_checker"):
        try:
            contract_res = requests.post(
                f"{API_URL}/contract-check/{job_id}",
                files={"file": (uploaded_contract_file.name, uploaded_contract_file.getvalue(), "application/octet-stream")},
                timeout=120,
            )
            st.session_state["uploaded_contract_check"] = contract_res.json() if contract_res.status_code == 200 else {"error": f"HTTP {contract_res.status_code}"}
        except Exception as e:
            st.session_state["uploaded_contract_check"] = {"error": str(e)}

    uploaded_contract_check = st.session_state.get("uploaded_contract_check")
    if uploaded_contract_check:
        if uploaded_contract_check.get("error"):
            st.error(uploaded_contract_check["error"])
        else:
            uc1, uc2, uc3 = st.columns(3)
            uc1.metric("Status", uploaded_contract_check.get("status", "unknown").replace("_", " ").title())
            uc2.metric("Missing Features", len(uploaded_contract_check.get("missing_features", [])))
            uc3.metric("Dtype Mismatches", len(uploaded_contract_check.get("dtype_mismatches", [])))
            if uploaded_contract_check.get("missing_features"):
                st.caption(f"Missing: {', '.join(uploaded_contract_check['missing_features'][:12])}")
            if uploaded_contract_check.get("extra_columns"):
                st.caption(f"Extra columns: {', '.join(uploaded_contract_check['extra_columns'][:12])}")
            dtype_rows = uploaded_contract_check.get("dtype_mismatches", [])
            if dtype_rows:
                st.dataframe(dtype_rows, width="stretch", hide_index=True)

    if not feature_names:
        st.warning("Feature metadata missing.")
    else:
        feature_inputs = {}
        column_stats = profile.get("column_stats", {}) or {}
        numeric_features = [f for f in feature_names if str(column_stats.get(f, {}).get("dtype", "")).lower() not in {"object", "category", "string"} and "top_values" not in column_stats.get(f, {})]
        categorical_features = [f for f in feature_names if f not in numeric_features]

        st.markdown("#### Prediction Controls")
        input_cols = st.columns(3)
        for i, feat in enumerate(feature_names):
            stats = column_stats.get(feat, {})
            with input_cols[i % 3]:
                if feat in numeric_features:
                    min_val = float(stats.get("min", 0) or 0)
                    max_val = float(stats.get("max", min_val + 1) or (min_val + 1))
                    mean_val = float(stats.get("mean", min_val) or min_val)
                    if min_val == max_val:
                        max_val = min_val + 1
                    feature_inputs[feat] = st.slider(
                        feat,
                        min_value=min_val,
                        max_value=max_val,
                        value=min(max(mean_val, min_val), max_val),
                        key=f"pred_slider_{feat}",
                    )
                else:
                    options = [str(v) for v in (stats.get("top_values") or []) if str(v).strip()]
                    if options:
                        feature_inputs[feat] = st.selectbox(feat, options=options, key=f"pred_select_{feat}")
                    else:
                        feature_inputs[feat] = st.text_input(feat, key=f"pred_text_{feat}")

        if numeric_features:
            sim_feature = st.selectbox("Scenario feature", options=numeric_features, key="scenario_feature")
            sim_stats = column_stats.get(sim_feature, {})
            s_min = float(sim_stats.get("min", 0) or 0)
            s_max = float(sim_stats.get("max", s_min + 1) or (s_min + 1))
            if s_min == s_max:
                s_max = s_min + 1
            sim_steps = st.slider("Scenario steps", min_value=3, max_value=20, value=8, key="scenario_steps")

        if st.button("🔮 Run Prediction", key="predict_run", width="stretch"):
            processed = {}
            for key, value in feature_inputs.items():
                if value is None:
                    continue
                if isinstance(value, str):
                    if not value.strip():
                        continue
                    try:
                        processed[key] = float(value) if "." in value else int(value)
                    except ValueError:
                        processed[key] = value
                else:
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
                                st.dataframe(probs_df, width="stretch", hide_index=True)
                                st.bar_chart(probs_df.set_index("Class"))
                    else:
                        st.error("Prediction failed.")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.info("Add at least one feature value to run a prediction.")

        if results.get("is_classification") and st.button("↔️ Generate Counterfactual", key="counterfactual_run", width="stretch"):
            processed = {}
            for key, value in feature_inputs.items():
                if value is None:
                    continue
                if isinstance(value, str):
                    if not value.strip():
                        continue
                    try:
                        processed[key] = float(value) if "." in value else int(value)
                    except ValueError:
                        processed[key] = value
                else:
                    processed[key] = value
            if processed:
                try:
                    cf_res = requests.post(
                        f"{API_URL}/counterfactual/{job_id}",
                        json={"features": processed},
                        timeout=45,
                    )
                    cf_payload = cf_res.json()
                except Exception as e:
                    cf_payload = {"error": str(e)}

                if cf_payload.get("error"):
                    st.error(cf_payload["error"])
                else:
                    st.markdown("#### Counterfactual Generator")
                    st.caption(cf_payload.get("message", ""))
                    suggestions = cf_payload.get("suggestions", [])
                    if suggestions:
                        st.dataframe(pd.DataFrame(suggestions), width="stretch", hide_index=True)
                    else:
                        st.info("No one-step flip was found for the current feature values.")

        if numeric_features and st.button("📈 Run Scenario Sweep", key="scenario_sweep", width="stretch"):
            sweep_values = [round(float(v), 6) for v in np.linspace(s_min, s_max, sim_steps).tolist()]
            base_features = {k: v for k, v in feature_inputs.items() if k != sim_feature}
            try:
                fut_res = requests.post(
                    f"{API_URL}/future",
                    json={
                        "job_id": job_id,
                        "base_features": base_features,
                        "sweep_feature": sim_feature,
                        "sweep_values": sweep_values,
                    },
                    timeout=60,
                )
                if fut_res.status_code == 200:
                    sim_data = fut_res.json().get("predictions", [])
                    sim_df = pd.DataFrame([row for row in sim_data if "error" not in row])
                    if not sim_df.empty:
                        st.markdown("#### Scenario Results")
                        st.dataframe(sim_df, width="stretch", hide_index=True)
                        chart_df = sim_df.rename(columns={"x": sim_feature, "prediction": "Prediction"}).set_index(sim_feature)
                        st.line_chart(chart_df[["Prediction"]])
                    else:
                        st.info("No valid scenario points were returned.")
                else:
                    st.error("Scenario sweep failed.")
            except Exception as e:
                st.error(f"Scenario sweep failed: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

with augmentation_tab:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🧬 Data Augmentation Lab")
    st.caption("Generate a larger synthetic dataset directly from the current workspace and move it back into the pipeline.")

    if not dataset_id:
        st.info("Load a dataset first to unlock augmentation.")
    else:
        rows_now = int(profile.get("rows") or 0)
        missing_pct = float(profile.get("missing_pct") or 0)
        imbalance_state = str(profile.get("imbalance") or "unknown")
        advisor_lines = []
        if rows_now < 500:
            advisor_lines.append("Augmentation is likely useful because the dataset is still relatively small.")
        if missing_pct > 15:
            advisor_lines.append("Repair the dataset first. High missingness can amplify synthetic noise.")
        if imbalance_state.lower() not in {"low", "balanced"}:
            advisor_lines.append("Validate class balance after augmentation because imbalance is already present.")
        if not advisor_lines:
            advisor_lines.append("Augmentation looks optional here. Use it mainly for experimentation or minority-class support.")

        st.markdown("#### Augmentation Advisor")
        for line in advisor_lines:
            st.caption(f"• {line}")

        aug_left, aug_right = st.columns([0.9, 1.1])
        with aug_left:
            default_rows = min(max(int((profile.get("rows") or 1000) * 0.25), 100), 10000)
            n_syn = st.number_input(
                "Synthetic rows to add",
                min_value=10,
                max_value=50000,
                value=default_rows,
                step=10,
            )
            if st.button("🧬 Generate Augmented Dataset", width="stretch"):
                with st.spinner("Generating synthetic extension..."):
                    try:
                        s_res = requests.post(
                            f"{API_URL}/synthetic/{dataset_id}",
                            params={"n_rows": int(n_syn)},
                            timeout=120,
                        )
                        st.session_state["results_console_synthetic"] = (
                            s_res.json() if s_res.status_code == 200 else {"error": f"HTTP {s_res.status_code}"}
                        )
                    except Exception as e:
                        st.session_state["results_console_synthetic"] = {"error": str(e)}

        with aug_right:
            st.markdown("#### Suggested Next Uses")
            for item in [
                "Boost smaller datasets before retraining.",
                "Compare original vs augmented data health in Dataset DNA.",
                "Launch a second training run and compare it in Experiment Tracker.",
                "Use augmentation before drift recovery when new data is sparse.",
            ]:
                st.markdown(f"- {item}")

        syn = st.session_state.get("results_console_synthetic")
        if syn:
            if syn.get("error"):
                st.error(syn["error"])
            else:
                if syn.get("adjustment_note"):
                    st.info(syn["adjustment_note"])

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Original Rows", syn.get("original_rows", 0))
                s2.metric("Added Rows", syn.get("synthetic_rows_added", 0))
                s3.metric("Total Rows", syn.get("total_rows", 0))
                s4.metric("Augmentation Ratio", f"{syn.get('augmentation_ratio', 0)}x")

                st.caption(
                    f"Recommended addition: {syn.get('recommended_rows', '—')} rows. "
                    f"Requested: {syn.get('requested_rows', '—')}."
                )

                preview = syn.get("preview") or []
                if preview:
                    st.markdown("#### Preview")
                    st.dataframe(pd.DataFrame(preview), width="stretch", hide_index=True)

                compare_rows = [
                    {"Metric": "Rows", "Original": (syn.get("original_profile") or {}).get("rows"), "Augmented": (syn.get("profile") or {}).get("rows")},
                    {"Metric": "Columns", "Original": (syn.get("original_profile") or {}).get("cols"), "Augmented": (syn.get("profile") or {}).get("cols")},
                    {"Metric": "Missing %", "Original": (syn.get("original_profile") or {}).get("missing_pct"), "Augmented": (syn.get("profile") or {}).get("missing_pct")},
                    {"Metric": "Suggested Target", "Original": (syn.get("original_profile") or {}).get("suggested_target"), "Augmented": (syn.get("profile") or {}).get("suggested_target")},
                ]
                st.markdown("#### Original vs Augmented")
                st.dataframe(pd.DataFrame(compare_rows), width="stretch", hide_index=True)

                load_col, inspect_col = st.columns(2)
                with load_col:
                    if st.button("🧪 Load Augmented Dataset For Retraining", width="stretch"):
                        st.session_state["dataset_id"] = syn.get("new_dataset_id")
                        if syn.get("profile"):
                            st.session_state["profile"] = syn["profile"]
                        st.switch_page("pages/1_Home.py")
                with inspect_col:
                    if st.button("🧬 Inspect Augmented Dataset DNA", width="stretch"):
                        st.session_state["dataset_id"] = syn.get("new_dataset_id")
                        if syn.get("profile"):
                            st.session_state["profile"] = syn["profile"]
                        st.switch_page("pages/2_Dataset_DNA.py")

                judge = api_json(f"/synthetic/judge/{syn.get('new_dataset_id')}", timeout=20) if syn.get("new_dataset_id") else {"error": "Missing synthetic dataset id"}
                st.markdown("#### Synthetic Data Judge")
                if judge.get("error"):
                    st.info(judge["error"])
                else:
                    judge_cols = st.columns(3)
                    judge_cols[0].metric("Realism Score", f"{judge.get('realism_score', 0)}")
                    judge_cols[1].metric("Verdict", judge.get("verdict", "—"))
                    judge_cols[2].metric("Rows Evaluated", judge.get("rows_evaluated", 0))
                    for note in judge.get("notes", [])[:6]:
                        st.caption(f"• {note}")
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
        chat_key = f"messages_{job_id}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []

        for message in st.session_state[chat_key]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about your model..."):
            st.session_state[chat_key].append({"role": "user", "content": prompt})
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
                st.session_state[chat_key].append({"role": "assistant", "content": ans})
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
