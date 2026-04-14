import streamlit as st
import pandas as pd
import numpy as np
import os
import requests

API_URL = "http://localhost:8000/api"

st.set_page_config(page_title="Dataset DNA", page_icon="🧬", layout="wide")


def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "style.css")
    try:
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass


load_css()

st.markdown('<h2 class="gradient-text">🧬 Dataset Intelligence Panel</h2>', unsafe_allow_html=True)

if not st.session_state.get('profile'):
    dataset_id = st.session_state.get('dataset_id')
    if dataset_id:
        try:
            res = requests.get(f"{API_URL}/dataset/{dataset_id}", timeout=15)
            if res.status_code == 200:
                profile_data = res.json()
                if not profile_data.get('error'):
                    st.session_state['profile'] = profile_data
                else:
                    st.info("Profile not available. Please upload a dataset on the Home page first.")
                    st.stop()
            else:
                st.info("Profile not available. Please upload a dataset on the Home page first.")
                st.stop()
        except Exception:
            st.info("Profile not available. Please upload a dataset on the Home page first.")
            st.stop()
    else:
        st.info("Please upload a dataset on the Home page first.")
        st.stop()

profile = st.session_state['profile']

# ── Summary Metrics ──────────────────────────────────────────────────────
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown("### 📊 Dataset DNA")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Rows", f"{profile.get('rows', 0):,}")
with c2:
    st.metric("Total Columns", f"{profile.get('cols', 0):,}")
with c3:
    st.metric("Missing Values", f"{profile.get('missing_pct', 0)}%")
with c4:
    imbalance = profile.get('imbalance', 'Low')
    st.metric("Imbalance Level", imbalance)

st.markdown("---")

# -- NEW: Health Score Section --
health = profile.get("health", {})
if health:
    h_score = health.get("score", 0)
    h_color = health.get("color", "#fff")
    h_grade = health.get("grade", "N/A")
    
    st.markdown(f"""
    <div style="background: rgba(0,0,0,0.1); padding: 20px; border-radius: 15px; border: 1px solid {h_color}44; margin-bottom: 25px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="margin:0; color: {h_color};">Dataset Health Score: {h_score}/100</h3>
                <p style="margin:5px 0 0 0; opacity: 0.8;">{health.get('summary', '')}</p>
            </div>
            <div style="font-size: 42px; font-weight: bold; color: {h_color}; background: {h_color}22; padding: 10px 25px; border-radius: 10px;">
                {h_grade}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

c5, c6 = st.columns(2)
with c5:
    st.markdown("#### 🗂 Schema Breakdown")
    st.write(f"- **Numerical Features**: {len(profile.get('num_cols', []))}")
    st.write(f"- **Categorical Features**: {len(profile.get('cat_cols', []))}")
    st.write(f"- **Estimated Size**: {profile.get('size', 'Unknown')}")
    suggested = profile.get("suggested_target")
    if suggested:
        st.info(f"🎯 **Suggested target column**: `{suggested}`")

with c6:
    st.markdown("#### 💡 Data Quality Insights")
    if health and health.get("issues"):
        for issue in health["issues"]:
            st.warning(f"⚠️ {issue}")
    if health and health.get("bonuses"):
        for bonus in health["bonuses"]:
            st.success(f"🌟 {bonus}")
    
    if not (health and health.get("issues")):
        st.success(f"**Recommended Pipeline:** {profile.get('suggested_model', 'Unknown')}")

st.markdown('</div>', unsafe_allow_html=True)


dataset_id = st.session_state.get("dataset_id")

if dataset_id:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🕓 Dataset Version Timeline")
    try:
        timeline_res = requests.get(f"{API_URL}/dataset/{dataset_id}/timeline", timeout=15)
        timeline_payload = timeline_res.json() if timeline_res.status_code == 200 else {"error": f"HTTP {timeline_res.status_code}"}
    except Exception as e:
        timeline_payload = {"error": str(e)}

    if timeline_payload.get("error"):
        st.info("Timeline is not available for this dataset yet.")
    else:
        timeline_df = pd.DataFrame(timeline_payload.get("timeline", []))
        if not timeline_df.empty:
            st.dataframe(timeline_df, width="stretch", hide_index=True)
        diff = timeline_payload.get("profile_diff", {})
        if diff:
            d1, d2, d3 = st.columns(3)
            d1.metric("Row Delta", diff.get("rows", 0))
            d2.metric("Column Delta", diff.get("cols", 0))
            d3.metric("Missing % Delta", diff.get("missing_pct", 0))
    st.markdown('</div>', unsafe_allow_html=True)

# ── Per-Column Statistics ─────────────────────────────────────────────────
column_stats = profile.get("column_stats", {})
if column_stats:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🔬 Per-Column Statistics")

    # Build a display DataFrame
    rows_list = []
    for col, stats in column_stats.items():
        row = {
            "Column": col,
            "Type": stats.get("dtype", ""),
            "Missing": f"{stats.get('missing_pct', 0)}%",
            "Unique": stats.get("unique", ""),
        }
        if "mean" in stats:
            row["Mean"] = stats.get("mean")
            row["Std"] = stats.get("std")
            row["Min"] = stats.get("min")
            row["Max"] = stats.get("max")
            row["Skew"] = stats.get("skew")
            row["Top Values"] = ""
        else:
            row["Mean"] = ""
            row["Std"] = ""
            row["Min"] = ""
            row["Max"] = ""
            row["Skew"] = ""
            row["Top Values"] = ", ".join(str(v) for v in stats.get("top_values", []))
        rows_list.append(row)

    stats_df = pd.DataFrame(rows_list).set_index("Column")

    stats_df = stats_df.replace("", np.nan).infer_objects(copy=False)
    
    for col in stats_df.columns:
        try:
            stats_df[col] = pd.to_numeric(stats_df[col])
        except Exception:
            pass

    st.dataframe(stats_df, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Feature 2: Auto-Detect Details ───────────────────────────────────────
    detect = st.session_state.get("auto_detect")
    if detect and not detect.get("error"):
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("### 🤖 Auto Problem Detection")
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.metric("Suggested Target", detect.get("suggested_target", "—"))
        with col_d2:
            st.metric("Task Type", detect.get("task_type", "—").title())
        with col_d3:
            st.metric("Confidence", f"{detect.get('confidence', 0)}%")

        col_scores = detect.get("column_scores", [])
        if col_scores:
            st.markdown("#### 🏆 Top Target Column Candidates")
            df_cands = pd.DataFrame(col_scores[:8])
            st.dataframe(df_cands, width="stretch", hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Feature 4: Leakage Detection Panel ─────────────────────────────────────
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🔍 Data Leakage & Quality Report")
    st.caption("Detect target leakage, ID-like columns, duplicates, constants, and future data leakage.")

    suggested_target = profile.get("suggested_target", "")

    if dataset_id:
        if st.button("🔍 Run Leakage Scan", key="run_leakage"):
            with st.spinner("Scanning for leakage..."):
                try:
                    url = f"{API_URL}/leakage/{dataset_id}"
                    if suggested_target:
                        url += f"?target_column={suggested_target}"
                    lr = requests.get(url, timeout=30)
                    st.session_state["leakage_report"] = lr.json() if lr.status_code == 200 else {}
                except Exception as e:
                    st.session_state["leakage_report"] = {"error": str(e)}

leakage = st.session_state.get("leakage_report")
if leakage:
    if leakage.get("error"):
        st.error(f"❌ {leakage['error']}")
    else:
        severity = leakage.get("severity", "")
        if "🔴" in severity:
            st.error(f"**{severity}** — take action before training!")
        elif "🟡" in severity:
            st.warning(f"**{severity}** — review warnings below.")
        else:
            st.success(f"**{severity}**")

        lc1, lc2, lc3, lc4 = st.columns(4)
        with lc1:
            st.metric("Duplicates", f"{leakage.get('duplicate_rows', 0)} ({leakage.get('duplicate_pct', 0):.1f}%)")
        with lc2:
            st.metric("Constant Columns", len(leakage.get("constant_columns", [])))
        with lc3:
            st.metric("Target Leakage", len(leakage.get("target_correlated", [])))
        with lc4:
            st.metric("Total Issues", leakage.get("total_issues", 0))

        warnings_list = leakage.get("warnings", [])
        for w in warnings_list:
            if "🔴" in w:
                st.error(w)
            elif "🟡" in w or "⚠️" in w:
                st.warning(w)
            else:
                st.success(w)

        # Detailed tables
        if leakage.get("target_correlated"):
            with st.expander("🔴 Target-Correlated Features (likely leakage)"):
                st.dataframe(pd.DataFrame(leakage["target_correlated"]), width="stretch", hide_index=True)
        if leakage.get("near_constant"):
            with st.expander("🟡 Near-Constant Features"):
                st.dataframe(pd.DataFrame(leakage["near_constant"]), width="stretch", hide_index=True)
        if leakage.get("high_missing"):
            with st.expander("🟡 High Missing Value Features (>50%)"):
                st.dataframe(pd.DataFrame(leakage["high_missing"]), width="stretch", hide_index=True)
else:
    st.info("Click **Run Leakage Scan** above to check your dataset for common ML pitfalls.")

st.markdown('</div>', unsafe_allow_html=True)
