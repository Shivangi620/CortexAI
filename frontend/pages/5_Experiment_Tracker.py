"""
Legacy Streamlit Experiment Tracker page for CODIN compatibility workflows.

The primary product UI is now the React studio served by FastAPI.
"""

import streamlit as st
import requests
import pandas as pd
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


def api_json(path: str, timeout: int = 10):
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


st.set_page_config(page_title="Legacy Experiment Tracker - CODIN", page_icon="📊", layout="wide")

load_css()
ensure_session_state()
render_page_shell(
    title="Experiment Tracker",
    eyebrow="Run Archive",
    description="Review historical runs, surface top performers, compare configurations, and jump back into the strongest models with less hunting.",
    stats=[
        (
            "Active Run",
            (
                (st.session_state.get("job_id") or "No run")[:8]
                if st.session_state.get("job_id")
                else "No run"
            ),
        ),
        (
            "Dataset",
            (
                (st.session_state.get("dataset_id") or "No dataset")[:8]
                if st.session_state.get("dataset_id")
                else "No dataset"
            ),
        ),
    ],
    accent="analysis",
)
render_workspace_banner()
st.info(f"Primary React studio: {API_URL}/tracking")
render_section_intro(
    "Comparison Deck",
    "The tracker is now framed as a review workspace instead of a plain table dump.",
    "Use the filters, leaderboard, and side-by-side comparison area to understand which runs deserve a closer look.",
)
render_focus_strip(
    [
        (
            "Model Selection",
            "Compare algorithms, hyperparameters, and fold strategy in one place.",
        ),
        (
            "Hyperparameter Tuning",
            "Grid Search and Random Search style decisions are easier to review after runs land.",
        ),
        (
            "Generalization",
            "Use cross-validation history and holdout metrics together before promoting a run.",
        ),
    ]
)

# -- New: Global Leaderboard Section --
with st.expander("🏆 View Global All-Time Leaderboard", expanded=False):
    try:
        l_res = requests.get(f"{API_URL}/leaderboard", timeout=5)
        if l_res.status_code == 200:
            lb_data = l_res.json()
            if lb_data:
                st.markdown("### 🥇 All-Time Performers")
                l_cols = st.columns(min(3, len(lb_data)))
                for i in range(min(3, len(lb_data))):
                    with l_cols[i]:
                        row = lb_data[i]
                        medal = ["🥇", "🥈", "🥉"][i]
                        st.markdown(
                            f"""
                        <div class="glass-panel" style="text-align:center; border-top: 3px solid #00f2fe;">
                            <div style="font-size: 2rem;">{medal}</div>
                            <h5 style="margin:5px 0;">{row['model']}</h5>
                            <div style="font-size: 1.2rem; font-weight: bold; color: #00f2fe;">{row['score']}%</div>
                            <div style="font-size: 0.7rem; opacity:0.7;">{row['task']}</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
    except Exception:
        st.caption("Unable to load global stats.")

# ── Fetch experiments ─────────────────────────────────────────────────────────
try:
    res = requests.get(f"{API_URL}/experiments", timeout=5)
    experiments = res.json() if res.status_code == 200 else []
except Exception:
    st.error("❌ Cannot reach backend.")
    st.stop()

if not experiments:
    st.info(
        "🔬 No experiments yet. Train a model on the Home page to start tracking runs."
    )
    st.stop()

# ── Filters ───────────────────────────────────────────────────────────────────
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    task_filter = st.selectbox(
        "Filter by Task", ["All", "classification", "regression"]
    )
with col_f2:
    mode_filter = st.selectbox("Filter by Mode", ["All", "Fast", "Balanced", "Full"])
with col_f3:
    sort_by = st.selectbox("Sort by", ["Score ↓", "Score ↑", "Date ↓", "Date ↑"])

filtered = experiments
if task_filter != "All":
    filtered = [e for e in filtered if e.get("task_type") == task_filter]
if mode_filter != "All":
    filtered = [e for e in filtered if e.get("mode") == mode_filter]

if sort_by == "Score ↓":
    filtered.sort(key=lambda x: float(x.get("score") or 0), reverse=True)
elif sort_by == "Score ↑":
    filtered.sort(key=lambda x: float(x.get("score") or 0))
elif sort_by == "Date ↑":
    filtered.sort(key=lambda x: x.get("created_at") or "")

st.markdown(f"**{len(filtered)} experiments** found")

st.markdown(
    """
    <div class="tracker-grid">
        <div class="tracker-panel">
            <div class="tracker-panel__eyebrow">Archive</div>
            <div class="tracker-panel__title">Run History Center</div>
            <div class="tracker-panel__copy">Inspect timelines, reasoning streams, and recent performance progression for individual jobs.</div>
        </div>
        <div class="tracker-panel">
            <div class="tracker-panel__eyebrow">Comparison</div>
            <div class="tracker-panel__title">Battle Arena</div>
            <div class="tracker-panel__copy">Compare 2 to 4 models visually, inspect score trade-offs, and review configuration differences in one place.</div>
        </div>
        </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown("### 🕓 Run History Center")
st.caption(
    "History moved here from Results Console so the archive, run inspector, and battle tools all live together."
)

run_lookup = {
    f"{e.get('model_name', '—')} | {(e.get('created_at') or '')[:16]} | {(e.get('id') or '')[:8]}": e
    for e in filtered[:50]
}
selected_run_label = (
    st.selectbox("Inspect a run", options=list(run_lookup.keys()))
    if run_lookup
    else None
)
selected_run = run_lookup.get(selected_run_label, {}) if selected_run_label else {}
selected_job_id = selected_run.get("job_id")

if selected_job_id:
    jump_col, _ = st.columns([0.45, 0.55])
    with jump_col:
        if st.button(
            "Load This Run Into Workspace",
            width="stretch",
            key=f"load_run_{selected_run.get('id')}",
        ):
            st.session_state["job_id"] = selected_job_id
            if selected_run.get("dataset_id"):
                st.session_state["dataset_id"] = selected_run.get("dataset_id")
            sync_workspace_query_params()
            st.success("Workspace updated with the selected run.")

    status_payload = api_json(f"/status/{selected_job_id}", timeout=10)
    notes_payload = api_json(f"/notes/run/{selected_run.get('id')}", timeout=10)
    if status_payload.get("error"):
        st.info(f"Run history is temporarily unavailable: {status_payload['error']}")
    else:
        history = status_payload.get("history", []) or []
        reasoning = status_payload.get("reasoning", []) or []
        history_df = pd.DataFrame(history)

        # Convert metric column to string to avoid Arrow serialization issues with mixed types
        if "metric" in history_df.columns:
            history_df["metric"] = history_df["metric"].astype(str)

        hist_left, hist_right = st.columns([1.1, 0.9])
        with hist_left:
            if history_df.empty:
                st.info("No timeline recorded for this run.")
            else:
                render_safe_dataframe(history_df, width="stretch", hide_index=True)

        with hist_right:
            st.metric("History Events", len(history))
            st.metric("Reasoning Notes", len(reasoning))
            if not history_df.empty and "metric" in history_df.columns:
                history_df["metric_numeric"] = pd.to_numeric(
                    history_df["metric"], errors="coerce"
                )
                curve = history_df[history_df["metric_numeric"].notna()].copy()
                if not curve.empty:
                    curve.index = range(1, len(curve) + 1)
                    st.line_chart(
                        curve.set_index(curve.index)[["metric_numeric"]].rename(
                            columns={"metric_numeric": "Run Score"}
                        )
                    )

            if reasoning:
                with st.expander("Reasoning Stream", expanded=False):
                    for line in reasoning[:25]:
                        st.write(f"• {line}")

        st.markdown("#### 📝 Team Notes")
        new_note = st.text_area(
            "Add note for this run",
            key=f"team_note_{selected_run.get('id')}",
            height=100,
        )
        if st.button(
            "➕ Save Note", width="stretch", key=f"save_note_{selected_run.get('id')}"
        ):
            try:
                note_res = requests.post(
                    f"{API_URL}/notes/run/{selected_run.get('id')}",
                    json={"note": new_note},
                    timeout=15,
                )
                note_data = note_res.json()
                if note_res.status_code == 200 and not note_data.get("error"):
                    st.success("Note saved.")
                else:
                    st.error(note_data.get("error", "Failed to save note."))
            except Exception as e:
                st.error(f"Note save failed: {e}")

        notes = (
            notes_payload.get("notes", []) if isinstance(notes_payload, dict) else []
        )
        if notes:
            for note in notes[:8]:
                st.caption(
                    f"{(note.get('created_at') or '')[:16]} · {note.get('note')}"
                )
else:
    st.info("Choose a run to inspect its history and reasoning stream.")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ── Summary metrics ───────────────────────────────────────────────────────────
if filtered:
    scores = [float(e["score"]) for e in filtered if e.get("score")]
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Total Runs", len(filtered))
    with col_m2:
        st.metric("Best Score", f"{max(scores):.1f}%" if scores else "—")
    with col_m3:
        st.metric("Avg Score", f"{sum(scores)/len(scores):.1f}%" if scores else "—")
    with col_m4:
        models = [e.get("model_name") for e in filtered if e.get("model_name")]
        top_model = max(set(models), key=models.count) if models else "—"
        st.metric("Most Used Model", top_model)

st.markdown("---")

# ── Run table ─────────────────────────────────────────────────────────────────
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown("### 📋 All Runs")

table_rows = []
for e in filtered[:50]:
    score_val = e.get("score")
    try:
        score_display = f"{float(score_val):.1f}%"
    except (TypeError, ValueError):
        score_display = "—"

    table_rows.append(
        {
            "ID": e.get("id", "")[:8] + "...",
            "Model": e.get("model_name", "—"),
            "Score": score_display,
            "Metric": e.get("metric_name", "—"),
            "Task": e.get("task_type", "—"),
            "Mode": e.get("mode", "—"),
            "Goal": e.get("goal", "—"),
            "Features": e.get("feature_count", "—"),
            "Rows": e.get("row_count", "—"),
            "Date": (e.get("created_at") or "")[:16],
            "_id": e.get("id", ""),
        }
    )

df_table = pd.DataFrame(table_rows)
display_df = df_table.drop(columns=["_id"])
render_safe_dataframe(display_df, width="stretch", hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Compare selected ──────────────────────────────────────────────────────────
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown("### 🔬 Compare Experiments")

run_options = {
    f"{e.get('model_name','?')} | {e.get('score','?')}% | {(e.get('created_at') or '')[:10]} [{e['id'][:6]}]": e[
        "id"
    ]
    for e in filtered
}

selected_labels = st.multiselect(
    "Select 2–5 runs to compare",
    options=list(run_options.keys()),
    max_selections=5,
)

if selected_labels and len(selected_labels) >= 2:
    selected_ids = ",".join(run_options[label] for label in selected_labels)
    try:
        cmp_res = requests.get(
            f"{API_URL}/experiments/compare?ids={selected_ids}", timeout=5
        )
        cmp_data = cmp_res.json().get("comparison", [])
    except Exception:
        cmp_data = []

    if cmp_data:
        # Metrics table
        cmp_rows = []
        for r in cmp_data:
            m = r.get("metrics", {})
            hp = r.get("hyperparams", {})
            cmp_rows.append(
                {
                    "Model": r.get("model_name", "—"),
                    "Score": f"{float(r['score']):.1f}%" if r.get("score") else "—",
                    "Metric": r.get("metric_name", "—"),
                    "Mode": r.get("mode", "—"),
                    "CV Folds": hp.get("cv_folds", "—"),
                    "Features": r.get("feature_count", "—"),
                    "Precision": (
                        f"{m.get('precision', '—')}%" if m.get("precision") else "—"
                    ),
                    "Recall": f"{m.get('recall', '—')}%" if m.get("recall") else "—",
                    "F1": f"{m.get('f1', '—')}%" if m.get("f1") else "—",
                }
            )

        render_safe_dataframe(pd.DataFrame(cmp_rows), width="stretch", hide_index=True)

        # Bar chart comparison
        chart_df = pd.DataFrame(
            {
                r.get("model_name", "?"): [float(r["score"]) if r.get("score") else 0]
                for r in cmp_data
            }
        ).T
        chart_df.columns = ["Score %"]
        st.markdown("#### 📊 Score Comparison")
        st.bar_chart(chart_df)

        # Hyperparameter diff
        st.markdown("#### ⚙️ Hyperparameter Details")
        for r in cmp_data:
            with st.expander(f"🔧 {r.get('model_name','?')} — hyperparams"):
                hp = r.get("hyperparams", {})
                for k, v in hp.items():
                    st.write(f"- **{k}**: `{v}`")

        st.markdown("#### ⚔️ Model Battle Arena")
        arena_rows = []
        for r in cmp_data[:4]:
            metrics = r.get("metrics", {}) or {}
            arena_rows.append(
                {
                    "Model": r.get("model_name", "—"),
                    "Score": (
                        float(r["score"]) if r.get("score") not in (None, "") else 0.0
                    ),
                    "Precision": float(metrics.get("precision") or 0),
                    "Recall": float(metrics.get("recall") or 0),
                    "F1": float(metrics.get("f1") or 0),
                    "Features": float(r.get("feature_count") or 0),
                }
            )

        arena_df = pd.DataFrame(arena_rows).set_index("Model")
        if not arena_df.empty:
            st.bar_chart(arena_df[["Score", "Precision", "Recall", "F1"]])
            render_safe_dataframe(arena_df, width="stretch")
            st.caption(
                "Compare up to four runs at once to spot the strongest trade-offs across score and classification quality."
            )

        if len(cmp_data) >= 2:
            st.markdown("#### 🧠 Run-to-Run Diff Engine")
            diff_left, diff_right = st.columns(2)
            diff_a = diff_left.selectbox(
                "Baseline Run",
                options=selected_labels,
                key="diff_run_a",
            )
            diff_b = diff_right.selectbox(
                "Compare Against",
                options=selected_labels,
                index=1 if len(selected_labels) > 1 else 0,
                key="diff_run_b",
            )
            if diff_a != diff_b:
                try:
                    diff_res = requests.get(
                        f"{API_URL}/experiments/diff",
                        params={
                            "run_a": run_options[diff_a],
                            "run_b": run_options[diff_b],
                        },
                        timeout=10,
                    )
                    diff_payload = diff_res.json()
                except Exception as e:
                    diff_payload = {"error": str(e)}

                if diff_payload.get("error"):
                    st.error(diff_payload["error"])
                else:
                    for line in diff_payload.get("explanations", []):
                        st.caption(f"• {line}")
                    diff_tabs = st.columns(2)
                    with diff_tabs[0]:
                        st.markdown("**Config Changes**")
                        config_df = pd.DataFrame(diff_payload.get("config_changes", []))
                        if config_df.empty:
                            st.info("No config changes detected.")
                        else:
                            render_safe_dataframe(
                                config_df, width="stretch", hide_index=True
                            )
                    with diff_tabs[1]:
                        st.markdown("**Output Changes**")
                        output_df = pd.DataFrame(diff_payload.get("output_changes", []))
                        if output_df.empty:
                            st.info("No output changes detected.")
                        else:
                            render_safe_dataframe(
                                output_df, width="stretch", hide_index=True
                            )
else:
    st.info(
        "Select 2 to 5 runs to unlock side-by-side comparison and the battle arena."
    )

st.markdown("</div>", unsafe_allow_html=True)
