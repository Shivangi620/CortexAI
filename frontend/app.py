import streamlit as st
import requests
import os

st.set_page_config(
    page_title="AutoML Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000/api"


def load_css():
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
    try:
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass


load_css()

# ── Initialise session state ──────────────────────────────────────────────────
for key, default in [('dataset_id', None), ('profile', None), ('job_id', None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Hero Section ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 3rem 1rem 2rem;">
    <h1 class="gradient-text" style="font-size: clamp(2.8rem, 6vw, 5rem); margin-bottom: 0.5rem;">
        ✨ AutoML Studio
    </h1>
    <p class="hero-tagline" style="max-width:560px; margin: 0 auto 1.5rem;">
        Upload. Train. Analyze. All in one intelligent place.
    </p>
    <div class="gradient-separator"></div>
</div>
""", unsafe_allow_html=True)

# ── Live backend stats ────────────────────────────────────────────────────────
try:
    jobs = requests.get(f"{API_URL}/jobs", timeout=3).json()
    total = len(jobs)
    completed = sum(1 for j in jobs if j.get("status") == "completed")
    running   = sum(1 for j in jobs if j.get("status") == "training")
    failed    = sum(1 for j in jobs if j.get("status") == "failed")
    backend_ok = True
except Exception:
    total = completed = running = failed = 0
    backend_ok = False

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("🧪 Total Runs", total)
with c2:
    st.metric("✅ Completed", completed)
with c3:
    st.metric("⚡ Training", running)
with c4:
    st.metric("❌ Failed", failed)

# Status banner
if backend_ok:
    st.markdown(
        '<div class="status-badge completed dot" style="margin-bottom: 1rem;">Backend Connected</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="status-badge failed dot" style="margin-bottom: 1rem;">Backend Offline — Start FastAPI on port 8000</div>',
        unsafe_allow_html=True
    )

st.markdown('<div class="gradient-separator"></div>', unsafe_allow_html=True)

# ── Navigation Cards ──────────────────────────────────────────────────────────
st.markdown("### 🗺️ Navigation", unsafe_allow_html=True)

nav_data = [
    ("🏠 Home", "Upload dataset & configure training", "pages/1_Home.py"),
    ("🧬 Dataset DNA", "Deep dive into your dataset's structure", "pages/2_Dataset_DNA.py"),
    ("🔥 Training Lab", "Live training monitor + leaderboard", "pages/3_Training_Lab.py"),
    ("📊 Results Console", "History, SHAP, pipeline map, chat, and prediction tools", "pages/4_Results_Console.py"),
    ("📜 Experiment Tracker", "Past experiment runs & comparisons", "pages/5_Experiment_Tracker.py"),
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
        st.markdown(f"""
        <div class="glass-panel" style="text-align:center; padding: 20px 16px; min-height:100px;">
            <div style="font-size:1.6rem; margin-bottom:8px;">{title.split()[0]}</div>
            <div style="font-family:'Manrope',sans-serif; font-weight:700; color:var(--on-surface); font-size:0.95rem;">{' '.join(title.split()[1:])}</div>
            <div style="color:var(--on-muted); font-size:0.8rem; margin-top:4px;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<p style="text-align:center; color:var(--on-muted); font-size:0.9rem; margin-top: 2rem;">
    👈 Select any page from the sidebar to get started
</p>
""", unsafe_allow_html=True)
