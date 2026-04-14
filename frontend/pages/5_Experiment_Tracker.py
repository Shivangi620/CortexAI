import streamlit as st
import requests
import pandas as pd
import os

API_URL = "http://localhost:8000/api"

st.set_page_config(page_title="Experiment Tracker", page_icon="📊", layout="wide")


def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "style.css")
    try:
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass


load_css()
st.markdown('<h2 class="gradient-text">📊 Experiment & Leaderboard Hub</h2>', unsafe_allow_html=True)
st.markdown("Track training runs, compare models, and view the global all-time winners.")

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
                        st.markdown(f"""
                        <div class="glass-panel" style="text-align:center; border-top: 3px solid #00f2fe;">
                            <div style="font-size: 2rem;">{medal}</div>
                            <h5 style="margin:5px 0;">{row['model']}</h5>
                            <div style="font-size: 1.2rem; font-weight: bold; color: #00f2fe;">{row['score']}%</div>
                            <div style="font-size: 0.7rem; opacity:0.7;">{row['task']}</div>
                        </div>
                        """, unsafe_allow_html=True)
    except Exception:
        st.caption("Unable to load global stats.")

st.markdown("---")

# ── Fetch experiments ─────────────────────────────────────────────────────────
try:
    res = requests.get(f"{API_URL}/experiments", timeout=5)
    experiments = res.json() if res.status_code == 200 else []
except Exception:
    st.error("❌ Cannot reach backend.")
    st.stop()

if not experiments:
    st.info("🔬 No experiments yet. Train a model on the Home page to start tracking runs.")
    st.stop()

# ── Filters ───────────────────────────────────────────────────────────────────
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    task_filter = st.selectbox("Filter by Task", ["All", "classification", "regression"])
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

    table_rows.append({
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
    })

df_table = pd.DataFrame(table_rows)
display_df = df_table.drop(columns=["_id"])
st.dataframe(display_df, width="stretch", hide_index=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── Compare selected ──────────────────────────────────────────────────────────
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown("### 🔬 Compare Experiments")

run_options = {f"{e.get('model_name','?')} | {e.get('score','?')}% | {(e.get('created_at') or '')[:10]} [{e['id'][:6]}]": e["id"]
               for e in filtered}

selected_labels = st.multiselect(
    "Select 2–5 runs to compare",
    options=list(run_options.keys()),
    max_selections=5,
)

if selected_labels and len(selected_labels) >= 2:
    selected_ids = ",".join(run_options[label] for label in selected_labels)
    try:
        cmp_res = requests.get(f"{API_URL}/experiments/compare?ids={selected_ids}", timeout=5)
        cmp_data = cmp_res.json().get("comparison", [])
    except Exception:
        cmp_data = []

    if cmp_data:
        # Metrics table
        cmp_rows = []
        for r in cmp_data:
            m = r.get("metrics", {})
            hp = r.get("hyperparams", {})
            cmp_rows.append({
                "Model": r.get("model_name", "—"),
                "Score": f"{float(r['score']):.1f}%" if r.get("score") else "—",
                "Metric": r.get("metric_name", "—"),
                "Mode": r.get("mode", "—"),
                "CV Folds": hp.get("cv_folds", "—"),
                "Features": r.get("feature_count", "—"),
                "Precision": f"{m.get('precision', '—')}%" if m.get("precision") else "—",
                "Recall": f"{m.get('recall', '—')}%" if m.get("recall") else "—",
                "F1": f"{m.get('f1', '—')}%" if m.get("f1") else "—",
            })

        st.dataframe(pd.DataFrame(cmp_rows), width="stretch", hide_index=True)

        # Bar chart comparison
        chart_df = pd.DataFrame({
            r.get("model_name", "?"): [float(r["score"]) if r.get("score") else 0]
            for r in cmp_data
        }).T
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

st.markdown('</div>', unsafe_allow_html=True)
