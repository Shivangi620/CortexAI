import streamlit as st
import pandas as pd
import numpy as np
import requests
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

st.set_page_config(page_title="Dataset DNA", page_icon="🧬", layout="wide")

load_css()
ensure_session_state()
st.session_state.setdefault("dna_merge_preview", {})

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
render_page_shell(
    title="Dataset DNA",
    eyebrow="Data Profiling",
    description="Inspect structure, health, timeline, leakage risks, and target suggestions before committing more compute to training.",
    stats=[
        ("Rows", profile.get("rows", 0)),
        ("Columns", profile.get("cols", 0)),
        ("Missing %", f"{profile.get('missing_pct', 0)}%"),
        ("Target Guess", profile.get("suggested_target", "—")),
    ],
    accent="analysis",
)
render_workspace_banner()
render_section_intro(
    "Profile Readout",
    "A single place to evaluate dataset quality and modeling readiness.",
    "The panel below keeps the health score, schema mix, timeline changes, column statistics, and leakage report aligned in one review flow.",
)

st.session_state.setdefault("repair_preview", {})

st.markdown(
    f"""
    <div class="scan-strip">
        <div class="scan-card">
            <span class="scan-card__label">Target Signal</span>
            <strong class="scan-card__value">{profile.get("suggested_target", "—")}</strong>
            <div class="scan-card__meta">Recommended prediction anchor</div>
        </div>
        <div class="scan-card">
            <span class="scan-card__label">Schema Mix</span>
            <strong class="scan-card__value">{len(profile.get("num_cols", []))}N / {len(profile.get("cat_cols", []))}C</strong>
            <div class="scan-card__meta">Numerical and categorical split</div>
        </div>
        <div class="scan-card">
            <span class="scan-card__label">Data Mass</span>
            <strong class="scan-card__value">{profile.get("size", "Unknown")}</strong>
            <div class="scan-card__meta">Estimated storage footprint</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
    h_grade = health.get("grade", "N/A")
    
    st.markdown(f"""
    <div class="health-card">
        <div>
            <span class="health-card__eyebrow">Health Matrix</span>
            <div class="health-card__title">Dataset Health Score: {h_score}/100</div>
            <div class="health-card__copy">{health.get('summary', '')}</div>
            <div class="health-card__meta">Stability, completeness, and modeling readiness</div>
        </div>
        <div class="health-card__grade">
            {h_grade}
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

st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown("### 🧹 Data Repair Assistant")
st.caption("Preview or apply automated cleaning before training. This is the right place to review repair decisions at the dataset level.")

repair_target = profile.get("suggested_target") or (profile.get("columns") or [None])[-1]
repair_left, repair_right = st.columns([1, 1])
with repair_left:
    if st.button("🩺 Preview Data Repair", width="stretch"):
        try:
            repair_res = requests.post(
                f"{API_URL}/repair-preview",
                json={"dataset_id": st.session_state["dataset_id"], "target_column": repair_target},
                timeout=60,
            )
            st.session_state["repair_preview"] = repair_res.json()
        except Exception as e:
            st.session_state["repair_preview"] = {"error": str(e)}
with repair_right:
    if st.button("🛠 Apply Data Repair", width="stretch"):
        try:
            repair_res = requests.post(
                f"{API_URL}/repair-apply",
                json={"dataset_id": st.session_state["dataset_id"], "target_column": repair_target},
                timeout=60,
            )
            repair_data = repair_res.json()
            if repair_res.status_code == 200 and not repair_data.get("error"):
                st.session_state["dataset_id"] = repair_data["dataset_id"]
                st.session_state["profile"] = repair_data["profile"]
                sync_workspace_query_params()
                st.success("Repaired dataset loaded into workspace.")
                st.rerun()
            else:
                st.error(repair_data.get("error", "Repair failed."))
        except Exception as e:
            st.error(f"Repair failed: {e}")

repair_preview = st.session_state.get("repair_preview") or {}
if repair_preview:
    if repair_preview.get("error"):
        st.error(repair_preview["error"])
    else:
        summary_cols = st.columns(4)
        summary_cols[0].metric("Rows Before", repair_preview.get("before_rows", 0))
        summary_cols[1].metric("Rows After", repair_preview.get("after_rows", 0))
        summary_cols[2].metric("Cols Before", repair_preview.get("before_columns", 0))
        summary_cols[3].metric("Cols After", repair_preview.get("after_columns", 0))
        for item in repair_preview.get("logs", []):
            st.caption(f"• {item}")
        render_safe_dataframe(repair_preview.get("preview_records", []), width="stretch", hide_index=True)
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
            render_safe_dataframe(timeline_df, width="stretch", hide_index=True)
        diff = timeline_payload.get("profile_diff", {})
        if diff:
            d1, d2, d3 = st.columns(3)
            d1.metric("Row Delta", diff.get("rows", 0))
            d2.metric("Column Delta", diff.get("cols", 0))
            d3.metric("Missing % Delta", diff.get("missing_pct", 0))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🕸 Dataset Lineage Graph")
    try:
        lineage_graph_res = requests.get(f"{API_URL}/dataset/{dataset_id}/lineage-graph", timeout=15)
        lineage_graph = lineage_graph_res.json() if lineage_graph_res.status_code == 200 else {"error": "Unable to load lineage graph"}
    except Exception as e:
        lineage_graph = {"error": str(e)}

    if lineage_graph.get("error"):
        st.info("Lineage graph is not available for this dataset yet.")
    else:
        nodes = lineage_graph.get("nodes", [])
        edges = lineage_graph.get("edges", [])
        if nodes:
            for idx, node in enumerate(nodes):
                label = f"{idx + 1}. {node.get('label', 'Dataset')}"
                st.caption(
                    f"{label} · {node.get('rows', '—')} rows · {node.get('cols', '—')} cols · "
                    f"missing {node.get('missing_pct', '—')}%"
                )
            if edges:
                st.code(
                    "\n".join(
                        f"{edge.get('source', '')[:8]} --> {edge.get('target', '')[:8]}  ({edge.get('label', 'derived')})"
                        for edge in edges
                    ),
                    language="text",
                )
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

    stats_df = stats_df.mask(stats_df == "", np.nan).infer_objects(copy=False)
    
    for col in stats_df.columns:
        try:
            stats_df[col] = pd.to_numeric(stats_df[col])
        except Exception:
            pass

    render_safe_dataframe(stats_df, width="stretch")
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
            render_safe_dataframe(df_cands, width="stretch", hide_index=True)
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
                render_safe_dataframe(pd.DataFrame(leakage["target_correlated"]), width="stretch", hide_index=True)
        if leakage.get("near_constant"):
            with st.expander("🟡 Near-Constant Features"):
                render_safe_dataframe(pd.DataFrame(leakage["near_constant"]), width="stretch", hide_index=True)
        if leakage.get("high_missing"):
            with st.expander("🟡 High Missing Value Features (>50%)"):
                render_safe_dataframe(pd.DataFrame(leakage["high_missing"]), width="stretch", hide_index=True)
else:
    st.info("Click **Run Leakage Scan** above to check your dataset for common ML pitfalls.")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown("### 🧬 Dataset Merge Studio")
st.caption("Use existing datasets from the workspace, preview key overlap, then create a merged dataset with join stats.")
try:
    dataset_catalog = requests.get(f"{API_URL}/datasets", timeout=10).json().get("datasets", [])
except Exception:
    dataset_catalog = []

catalog_options = {
    f"{(item.get('source_type') or 'dataset').replace('_', ' ').title()} · {item.get('rows', 0)} rows · {(item.get('id') or '')[:8]}": item
    for item in dataset_catalog
}
if len(catalog_options) < 2:
    st.info("At least two datasets are needed to use Merge Studio.")
else:
    ms_left, ms_right = st.columns(2)
    left_label = ms_left.selectbox("Left Dataset", list(catalog_options.keys()), key="dna_merge_left")
    right_label = ms_right.selectbox("Right Dataset", list(catalog_options.keys()), index=1 if len(catalog_options) > 1 else 0, key="dna_merge_right")
    left_item = catalog_options[left_label]
    right_item = catalog_options[right_label]
    key_left_col, key_right_col, join_col = st.columns(3)
    left_key = key_left_col.selectbox("Left Join Key", left_item.get("columns") or [], key="dna_left_key")
    right_key = key_right_col.selectbox("Right Join Key", right_item.get("columns") or [], key="dna_right_key")
    join_type = join_col.selectbox("Join Type", ["inner", "left", "right", "outer"], key="dna_join_type")
    merge_signature = (left_item["id"], right_item["id"], left_key, right_key, join_type)
    if st.session_state.get("dna_merge_signature") != merge_signature:
        st.session_state["dna_merge_signature"] = merge_signature
        st.session_state["dna_merge_preview"] = {}

    if left_item["id"] == right_item["id"]:
        st.warning("Choose two different datasets to merge.")

    action_left, action_right = st.columns(2)
    if action_left.button("Preview Join Stats", width="stretch"):
        if left_item["id"] == right_item["id"]:
            st.error("Pick two different datasets.")
        else:
            try:
                preview_res = requests.post(
                    f"{API_URL}/merge-studio/preview",
                    json={
                        "left_dataset_id": left_item["id"],
                        "right_dataset_id": right_item["id"],
                        "join_key_left": left_key,
                        "join_key_right": right_key,
                        "join_type": join_type,
                    },
                    timeout=60,
                )
                st.session_state["dna_merge_preview"] = preview_res.json()
            except Exception as e:
                st.session_state["dna_merge_preview"] = {"error": str(e)}

    if action_right.button("Create Merged Dataset", width="stretch"):
        if left_item["id"] == right_item["id"]:
            st.error("Pick two different datasets.")
        else:
            try:
                merge_res = requests.post(
                    f"{API_URL}/merge-studio",
                    json={
                        "left_dataset_id": left_item["id"],
                        "right_dataset_id": right_item["id"],
                        "join_key_left": left_key,
                        "join_key_right": right_key,
                        "join_type": join_type,
                    },
                    timeout=120,
                )
                merge_data = merge_res.json()
                if merge_res.status_code == 200 and not merge_data.get("error"):
                    st.session_state["dataset_id"] = merge_data["dataset_id"]
                    st.session_state["profile"] = merge_data["profile"]
                    st.session_state["dna_merge_preview"] = {}
                    sync_workspace_query_params()
                    st.success("Merged dataset loaded into the workspace.")
                    st.rerun()
                else:
                    st.error(merge_data.get("error", "Merge failed."))
            except Exception as e:
                st.error(f"Merge failed: {e}")

    merge_preview = st.session_state.get("dna_merge_preview") or {}
    if merge_preview:
        if merge_preview.get("error"):
            st.error(merge_preview["error"])
        else:
            if merge_preview.get("join_key_coerced_to_string"):
                st.info(
                    "Join keys had different data types, so preview matching was normalized using string values."
                )
            preview_cols = st.columns(5)
            preview_cols[0].metric("Estimated Rows", merge_preview.get("estimated_rows", 0))
            preview_cols[1].metric("Overlapping Keys", merge_preview.get("overlapping_keys", 0))
            preview_cols[2].metric("Left Match %", f"{merge_preview.get('left_match_pct', 0)}%")
            preview_cols[3].metric("Right Match %", f"{merge_preview.get('right_match_pct', 0)}%")
            preview_cols[4].metric("Row Multiplier", merge_preview.get("estimated_row_multiplier", "—"))
            st.caption(
                f"Join-key duplicates: left {merge_preview.get('left_duplicate_keys', 0)} · "
                f"right {merge_preview.get('right_duplicate_keys', 0)}"
            )
            if merge_preview.get("preview_records"):
                render_safe_dataframe(merge_preview["preview_records"], width="stretch", hide_index=True)
st.markdown('</div>', unsafe_allow_html=True)
