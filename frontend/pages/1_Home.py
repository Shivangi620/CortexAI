import streamlit as st
import requests
import os

API_URL = "http://localhost:8000/api"

st.set_page_config(page_title="Home - AutoML Studio", page_icon="🏠", layout="wide")


def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "style.css")
    try:
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass


load_css()

st.markdown('<h2 class="gradient-text">1. Upload & Configure</h2>', unsafe_allow_html=True)

SUPPORTED_FORMATS = (
    "CSV, TSV, TXT, DAT  •  Excel (xlsx, xls, ods)  •  JSON / JSONL  •  "
    "Parquet, Feather, ORC, Arrow  •  Stata (dta)  •  SAS (sas7bdat)  •  "
    "SPSS (sav)  •  XML  •  HTML  •  SQLite (db)  •  Pickle (pkl)  •  AutoML export ZIP"
)

st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.caption(f"**Supported formats:** {SUPPORTED_FORMATS}")

uploaded_file = st.file_uploader(
    "📂 Drag & Drop your Dataset here",
    type=None,   # Accept ALL file types — backend handles format detection
    help="Supports regular datasets plus exported AutoML ZIP bundles that restore the training dataset and past run artifacts."
)

if uploaded_file is not None:
    # ── Auto-Analyze on Upload ────────────────────────────────────────────────
    # Track the last analyzed file to avoid infinite loops
    last_file = st.session_state.get('last_analyzed_file')
    current_file_key = f"{uploaded_file.name}_{uploaded_file.size}"

    if last_file != current_file_key:
        with st.spinner("🧬 Analyzing Neural DNA..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
            try:
                res = requests.post(f"{API_URL}/upload", files=files, timeout=600)
                if res.status_code == 200:
                    data = res.json()
                    if "error" in data:
                        st.error(data["error"])
                    else:
                        st.session_state['dataset_id'] = data['dataset_id']
                        st.session_state['profile'] = data['profile']
                        st.session_state['last_analyzed_file'] = current_file_key
                        imported_job_id = data.get("imported_job_id")
                        if imported_job_id:
                            st.session_state['job_id'] = imported_job_id
                            st.success(f"✅ **{uploaded_file.name}** restored successfully!")
                            st.info("This export bundle brought back the training dataset and its completed run. You can jump straight to Results Console or retrain from here.")
                        else:
                            st.success(f"✅ **{uploaded_file.name}** processed successfully!")

                        # ── Feature 2: Auto Problem Type Detection ─────────────────────
                        try:
                            detect_res = requests.post(
                                f"{API_URL}/detect",
                                json={"dataset_id": data['dataset_id']},
                                timeout=15,
                            )
                            if detect_res.status_code == 200:
                                st.session_state['auto_detect'] = detect_res.json()
                        except Exception:
                            pass
                else:
                    st.error(f"Backend error: {res.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach backend. Is FastAPI running on port 8000?")
st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.get('profile'):
    # ── Feature 2: Auto Problem Detection Banner ──────────────────────────────
    detect = st.session_state.get("auto_detect")
    if detect and not detect.get("error"):
        suggested_t = detect.get("suggested_target")
        task_type   = detect.get("task_type", "classification")
        confidence  = detect.get("confidence", 0)
        task_reason = detect.get("task_reason", "")
        warnings    = detect.get("warnings", [])

        icon = "📈" if task_type == "regression" else "🔵"
        det_color = "#4CAF50" if confidence >= 80 else "#FFA500"
        st.markdown(
            f"""<div style="background:linear-gradient(135deg,#1a1a2e,#16213e); border:1px solid {det_color}44;
            border-left: 4px solid {det_color}; border-radius:12px; padding:16px 20px; margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <div>
                <span style="font-size:0.8rem;color:{det_color};font-weight:600;letter-spacing:1px;">AUTO-DETECTED</span><br>
                <span style="font-size:1.15rem;font-weight:700;">
                  {icon} <b>{task_type.title()}</b> task
                  &nbsp;·&nbsp; Target: <code>{suggested_t}</code>
                </span><br>
                <span style="font-size:0.8rem;opacity:0.7;">{task_reason}</span>
              </div>
              <div style="font-size:1.8rem;font-weight:800;color:{det_color};">{confidence}%</div>
            </div></div>""",
            unsafe_allow_html=True,
        )
        if warnings:
            for w in warnings:
                if "⚠️" in w or "🔴" in w:
                    st.warning(w)
                elif "🟡" in w:
                    st.info(w)

    profile = st.session_state['profile']
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Training Configuration")


    columns = profile.get('columns', [])
    suggested_target = profile.get('suggested_target')
    if not suggested_target or suggested_target not in columns:
        suggested_target = columns[-1] if columns else None

    default_idx = columns.index(suggested_target) if suggested_target in columns else 0
    st.info(f"💡 Auto-detected target column: **{suggested_target}**")
    target_col = st.selectbox("🎯 Confirm or Change Target Column", columns, index=default_idx)

    col1, col2 = st.columns(2)
    with col1:
        goal = st.radio(
            "Select Goal",
            ["🎯 Performance(best results)", "⚡ Speed(fast)", "⚖️ Balanced"],
            help="Performance = widest model search | Speed = smaller fast pool | Balanced = strong middle ground"
        )
    with col2:
        mode = st.radio(
            "Select Execution Mode (V3)",
            ["⚡ Fast (Exploration only)", "⚖️ Balanced (Standard optimization)", "🧠 Full (Deep Bayesian Search)"],
            help="Fast = sweep only | Balanced = moderate optimization | Full = deeper Bayesian optimization"
        )

    goal_descriptions = {
        "🎯 Performance(best results)": "Searches the broadest model pool for the highest score.",
        "⚡ Speed(fast)": "Uses a smaller, faster model pool to finish sooner.",
        "⚖️ Balanced": "Keeps a practical mix of strong models without the slowest full search.",
    }
    mode_descriptions = {
        "⚡ Fast (Exploration only)": "Runs only the exploration sweep and picks the best early winner.",
        "⚖️ Balanced (Standard optimization)": "Optimizes the top few candidates with a moderate search budget.",
        "🧠 Full (Deep Bayesian Search)": "Uses a larger sweep sample and deeper tuning for top candidates.",
    }
    st.caption(f"Goal: {goal_descriptions[goal]}")
    st.caption(f"Mode: {mode_descriptions[mode]}")

    st.markdown("---")
    with st.expander("⚙️ Advanced Options", expanded=False):
        adv_col1, adv_col2 = st.columns(2)

        with adv_col1:
            # ── 5. Evaluation Metric ──────────────────────────────────────
            st.markdown("#### 📊 Evaluation Metric")
            detect = st.session_state.get("auto_detect", {})
            task_type = detect.get("task_type") or profile.get("task_type", "classification")
            if task_type == "regression":
                eval_metric = st.selectbox(
                    "Metric (Regression)",
                    ["RMSE", "R²"],
                    index=0,
                    help="Default: RMSE. RMSE = Root Mean Squared Error | R² = Coefficient of Determination"
                )
            else:
                eval_metric = st.selectbox(
                    "Metric (Classification)",
                    ["Accuracy", "F1-score"],
                    index=0,
                    help="Default: Accuracy. F1 = good for imbalanced data."
                )

            st.markdown("---")

            # ── 7. Handle Imbalance ───────────────────────────────────────
            st.markdown("#### ⚖️ Handle Imbalance")
            handle_imbalance = st.toggle(
                "Handle Class Imbalance",
                value=False,
                help="Applies SMOTE / class-weight balancing for skewed target distributions"
            )

        with adv_col2:
            # ── 6. Feature Selection ──────────────────────────────────────
            st.markdown("#### 🧠 Feature Selection")
            all_features = [c for c in profile.get("columns", []) if c != target_col]
            selected_features = st.multiselect(
                "Select Features (leave blank = All)",
                options=all_features,
                default=[],
                help="Advanced: restrict training to specific columns. Leave empty to use all features."
            )
            if not selected_features:
                st.caption("✔ All features selected")

            st.markdown("---")

            # ── 8. Auto Data Cleaning ─────────────────────────────────────
            st.markdown("#### 🧹 Auto Data Cleaning")
            auto_clean = st.checkbox(
                "Fix Issues Automatically",
                value=True,
                help="Fills missing values, removes duplicates, and encodes categories automatically"
            )

            st.markdown("---")

            # ── 10. Cross-Validation ─────────────────────────────────────
            st.markdown("#### 🧪 Cross-Validation")
            cv_folds = st.slider(
                "Number of CV Folds",
                min_value=0,
                max_value=10,
                value=0,
                help="Set to 0-1 to disable CV. 5-10 is standard but takes more time."
            )

        st.markdown("---")

        # ── 9. Output Options ─────────────────────────────────────────────
        st.markdown("#### 📦 Output Options")
        out_col1, out_col2, out_col3 = st.columns(3)
        with out_col1:
            export_model = st.checkbox("✔ Model", value=True, help="Export trained model file (.pkl / .joblib)")
        with out_col2:
            export_code = st.checkbox("✔ Code", value=True, help="Export reproducible Python training script")
        with out_col3:
            export_report = st.checkbox("✔ Report", value=True, help="Export full HTML/PDF analysis report")

    if st.button("▶ Run AutoML Engine", key="run_automl", width="stretch"):
        # Explicit mapping — no fragile emoji splitting
        goal_map = {
            "🎯 Performance(best results)": "Performance",
            "⚡ Speed(fast)": "Speed",
            "⚖️ Balanced": "Balanced"
        }
        mode_map = {
            "⚡ Fast (Exploration only)": "Fast",
            "⚖️ Balanced (Standard optimization)": "Balanced",
            "🧠 Full (Deep Bayesian Search)": "Full"
        }

        payload = {
            "dataset_id": st.session_state['dataset_id'],
            "target_column": target_col,
            "goal": goal_map.get(goal, "Performance"),
            "mode": mode_map.get(mode, "Fast"),
            # Advanced options
            "eval_metric": eval_metric,
            "selected_features": selected_features if selected_features else [],
            "handle_imbalance": handle_imbalance,
            "auto_clean": auto_clean,
            "cv_folds": cv_folds,
            "export_model": export_model,
            "export_code": export_code,
            "export_report": export_report,
        }
        try:
            res = requests.post(f"{API_URL}/train", json=payload, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if "error" in data:
                    st.error(data["error"])
                else:
                    st.session_state['job_id'] = data['job_id']
                    st.success("🚀 Training started!")
                    st.info("👉 Navigate to **Live Training** in the sidebar to watch progress.")
                    st.switch_page("pages/3_Training_Lab.py")
            else:
                st.error("Failed to start training.")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot reach backend.")

    st.markdown('</div>', unsafe_allow_html=True)
