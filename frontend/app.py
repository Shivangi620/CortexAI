import streamlit as st

from ui_shell import (
    ensure_session_state,
    fetch_backend_overview,
    load_css,
    render_backend_notice,
    render_page_shell,
    render_workspace_banner,
)

st.set_page_config(
    page_title="AutoML Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()
ensure_session_state()

overview = fetch_backend_overview()
total = overview["total"]
completed = overview["completed"]
running = overview["running"]
failed = overview["failed"]
backend_ok = overview["backend_ok"]

render_page_shell(
    title="AutoML Studio",
    eyebrow="Machine Learning Command Center",
    description="Upload tabular data, orchestrate training, inspect model behavior, monitor drift, and work with AI utilities from one cohesive control surface.",
    stats=[
        ("Tracked Runs", total),
        ("Completed", completed),
        ("Training", running),
        ("Failures", failed),
    ],
)
render_backend_notice(backend_ok)
render_workspace_banner()

st.markdown(
    """
    <div class="live-chip">
        <span class="live-chip__dot"></span>
        System interface synced for live dataset, training, and monitoring workflows
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Mission Runs", total)
with c2:
    st.metric("Successful", completed)
with c3:
    st.metric("In Training", running)
with c4:
    st.metric("Faults", failed)

st.markdown(
    """
    <div class="feature-ribbon">
        <div class="feature-ribbon__item"><span>Ingest</span><strong>Universal file upload and dataset restore</strong></div>
        <div class="feature-ribbon__item"><span>Train</span><strong>Fast, balanced, and deep AutoML execution</strong></div>
        <div class="feature-ribbon__item"><span>Inspect</span><strong>Results, drift, explainability, and exports</strong></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="command-grid">
        <div class="command-card">
            <span class="command-card__eyebrow">Telemetry</span>
            <div class="command-card__title">Runtime Pulse</div>
            <div class="command-card__copy">Track backend readiness, active runs, and workspace state from a single control layer before you commit compute.</div>
            <div class="command-card__meta">Live mission status</div>
        </div>
        <div class="command-card">
            <span class="command-card__eyebrow">Decisioning</span>
            <div class="command-card__title">Model Route Map</div>
            <div class="command-card__copy">Move from upload to leakage detection, training, explainability, and drift review with clearer handoff points between pages.</div>
            <div class="command-card__meta">Faster flow between tools</div>
        </div>
        <div class="command-card">
            <span class="command-card__eyebrow">Operator Mode</span>
            <div class="command-card__title">Futuristic HUD</div>
            <div class="command-card__copy">The new shell uses animated scanlines, orbit rings, glow depth, and stronger contrast so the dark mode feels intentional and premium.</div>
            <div class="command-card__meta">New dark theme activated</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Navigation Cards ──────────────────────────────────────────────────────────
st.markdown("### Neural Navigation", unsafe_allow_html=True)

nav_data = [
    ("🏠 Home", "Upload dataset & configure training", "pages/1_Home.py"),
    ("🧬 Dataset DNA", "Deep dive into your dataset's structure", "pages/2_Dataset_DNA.py"),
    ("🔥 Training Lab", "Live training monitor + leaderboard", "pages/3_Training_Lab.py"),
    ("📊 Results Console", "SHAP, pipeline map, scenario playground, augmentation, and model chat", "pages/4_Results_Console.py"),
    ("📜 Experiment Tracker", "Run history, archive, battle arena, and experiment comparisons", "pages/5_Experiment_Tracker.py"),
    ("🌊 Drift Monitor", "Monitor dataset drift against saved baselines", "pages/6_Drift_Monitor.py"),
    ("🤖 Smart AI Hub", "Synthetic data, ensembles, and NL tooling", "pages/7_Smart_AI_Hub.py"),
]

row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
rows = [row1, row2, row3]

for i, (title, desc, _) in enumerate(nav_data):
    row_idx = i // 3
    col_idx = i % 3
    if row_idx >= len(rows):
        rows.append(st.columns(3))
    with rows[row_idx][col_idx]:
        icon = title.split()[0]
        name = " ".join(title.split()[1:])
        st.markdown(f"""
        <div class="glass-panel nav-tile">
            <div class="nav-tile__icon">{icon}</div>
            <div class="nav-tile__title">{name}</div>
            <div class="nav-tile__desc">{desc}</div>
            <div class="nav-tile__cta">Open from sidebar and keep your workspace state</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="mission-footer-note">
    Select any page from the sidebar to enter upload, training, monitoring, or AI utility workflows.
</div>
""", unsafe_allow_html=True)

st.markdown("### Next-Level Features", unsafe_allow_html=True)
st.markdown(
    """
    <div class="ideas-grid">
        <div class="idea-card">
            <span class="idea-card__eyebrow">Feature Idea</span>
            <div class="idea-card__title">Scenario Simulator</div>
            <div class="idea-card__copy">Let users tweak a few feature values with sliders and watch predicted outcomes, SHAP shifts, and confidence move in real time.</div>
            <div class="idea-card__meta">Great for demos and stakeholder buy-in</div>
        </div>
        <div class="idea-card">
            <span class="idea-card__eyebrow">Feature Idea</span>
            <div class="idea-card__title">Data Anomaly Radar</div>
            <div class="idea-card__copy">Add an always-on anomaly watch that highlights strange rows, suspicious spikes, and unstable features before training begins.</div>
            <div class="idea-card__meta">Prevents bad runs early</div>
        </div>
        <div class="idea-card">
            <span class="idea-card__eyebrow">Feature Idea</span>
            <div class="idea-card__title">Experiment Narrator</div>
            <div class="idea-card__copy">Auto-write a concise story of why the current winning model beat the field and what changed compared with recent runs.</div>
            <div class="idea-card__meta">Useful for handoffs and reviews</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
