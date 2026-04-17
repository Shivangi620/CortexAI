import streamlit as st
import requests
import pandas as pd
import json

from ui_shell import (
    API_URL,
    clear_workspace_state,
    ensure_session_state,
    get_query_param,
    load_css,
    render_page_shell,
    render_safe_dataframe,
    render_section_intro,
    render_workspace_banner,
    sync_workspace_query_params,
)

st.set_page_config(page_title="Home - AutoML Studio", page_icon="🏠", layout="wide")

load_css()
ensure_session_state()
st.session_state.setdefault("upload_preview_records", [])
st.session_state.setdefault("upload_ingest_summary", {})
st.session_state.setdefault("home_merge_preview", {})
st.session_state.setdefault("goal_choice", get_query_param("goal") or "🎯 Performance(best results)")
st.session_state.setdefault("mode_choice", get_query_param("mode") or "⚖️ Balanced (Standard optimization)")
st.session_state.setdefault("eval_metric_choice", get_query_param("metric") or "Accuracy")
st.session_state.setdefault("handle_imbalance_choice", (get_query_param("imbalance") or "false").lower() == "true")
st.session_state.setdefault("auto_clean_choice", (get_query_param("auto_clean") or "true").lower() == "true")
st.session_state.setdefault("workspace_name_choice", get_query_param("workspace") or "Default Workspace")
st.session_state.setdefault("preset_choice", get_query_param("preset") or "Balanced")
st.session_state.setdefault("show_archived_datasets", False)
try:
    default_cv_folds = int(get_query_param("cv_folds") or 0)
except (TypeError, ValueError):
    default_cv_folds = 0
st.session_state.setdefault("cv_folds_choice", default_cv_folds)

profile = st.session_state.get("profile") or {}
render_page_shell(
    title="Upload And Configure",
    eyebrow="Data Intake",
    description="Bring in a fresh dataset or restore an exported run, let the system infer the target, and launch the next AutoML mission from one guided workspace.",
    stats=[
        ("Dataset Loaded", "Yes" if st.session_state.get("dataset_id") else "No"),
        ("Rows", profile.get("rows", "—")),
        ("Columns", len(profile.get("columns", []) or []) or "—"),
        ("Run Ready", "Yes" if profile else "Waiting"),
    ],
    accent="soft",
)
render_workspace_banner()
if st.session_state.get("_workspace_restored") and st.session_state.get("dataset_id"):
    st.markdown(
        """
        <div class="inline-notice inline-notice--success">
            <strong>Workspace Restored</strong>
            <span>Your last persisted dataset and run context were recovered, so a browser refresh will continue from the saved workspace instead of forcing a new upload.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
top_actions = st.columns([1, 1, 2])
with top_actions[0]:
    if st.session_state.get("dataset_id") and st.button("Remove Current Dataset", width="stretch"):
        clear_workspace_state()
        st.rerun()
with top_actions[1]:
    if st.session_state.get("_workspace_cleared"):
        st.caption("Current device workspace cleared.")
render_section_intro(
    "Workspace Flow",
    "Ingest data, confirm the prediction setup, and launch training.",
    "This page keeps the high-friction steps together: file upload, auto-detection, training preferences, advanced controls, and export options.",
)

@st.cache_data(show_spinner=False, ttl=20)
def get_training_forecast_cached(payload_json: str):
    payload = json.loads(payload_json)
    forecast_res = requests.post(
        f"{API_URL}/train/forecast",
        json=payload,
        timeout=20,
    )
    return forecast_res.json() if forecast_res.status_code == 200 else {}

st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown("### 🗂 Dataset Manager")
manager_cols = st.columns([1.3, 0.7, 1])
with manager_cols[0]:
    include_archived = st.toggle("Show archived datasets", key="show_archived_datasets")
with manager_cols[1]:
    if st.button("Refresh Catalog", width="stretch"):
        st.rerun()
try:
    datasets_payload = requests.get(
        f"{API_URL}/datasets",
        params={"limit": 50, "include_archived": include_archived},
        timeout=15,
    ).json()
    dataset_catalog = datasets_payload.get("datasets", [])
except Exception:
    dataset_catalog = []

if dataset_catalog:
    dataset_options = {
        f"{(item.get('display_name') or item.get('source_type') or 'dataset')} · {(item.get('rows') or 0)} rows · {(item.get('id') or '')[:8]}{' · archived' if item.get('archived') else ''}": item
        for item in dataset_catalog
    }
    selected_dataset_label = st.selectbox("Dataset Catalog", options=list(dataset_options.keys()), key="dataset_manager_select")
    selected_dataset = dataset_options.get(selected_dataset_label, {})
    action_cols = st.columns(4)
    with action_cols[0]:
        if st.button("Load", width="stretch", key="dataset_manager_load") and selected_dataset.get("id"):
            st.session_state["dataset_id"] = selected_dataset["id"]
            try:
                profile_resp = requests.get(f"{API_URL}/dataset/{selected_dataset['id']}", timeout=10)
                profile_data = profile_resp.json() if profile_resp.status_code == 200 else {}
                if not profile_data.get("error"):
                    st.session_state["profile"] = profile_data
            except Exception:
                pass
            st.session_state["_workspace_cleared"] = False
            sync_workspace_query_params()
            st.rerun()
    with action_cols[1]:
        archive_label = "Unarchive" if selected_dataset.get("archived") else "Archive"
        if st.button(archive_label, width="stretch", key="dataset_manager_archive") and selected_dataset.get("id"):
            endpoint = "unarchive" if selected_dataset.get("archived") else "archive"
            requests.post(f"{API_URL}/dataset/{selected_dataset['id']}/{endpoint}", timeout=15)
            st.rerun()
    with action_cols[2]:
        if st.button("Remove From Workspace", width="stretch", key="dataset_manager_clear"):
            clear_workspace_state()
            st.rerun()
    with action_cols[3]:
        if st.button("Delete Permanently", width="stretch", key="dataset_manager_delete") and selected_dataset.get("id"):
            requests.delete(f"{API_URL}/dataset/{selected_dataset['id']}", timeout=20)
            if st.session_state.get("dataset_id") == selected_dataset.get("id"):
                clear_workspace_state()
            st.rerun()
    catalog_df = pd.DataFrame(dataset_catalog)
    render_safe_dataframe(catalog_df[[c for c in ["display_name", "source_type", "rows", "cols", "created_at", "archived", "id"] if c in catalog_df.columns]], width="stretch", hide_index=True)
else:
    st.caption("No datasets in the catalog yet.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("### Import Mode", unsafe_allow_html=True)
import_mode = st.segmented_control(
    "Choose how you want to ingest data",
    options=["Upload File", "Connectors", "Merge Studio"],
    default="Upload File",
)

SUPPORTED_FORMATS = (
    "CSV, TSV, TXT, DAT, Markdown, RTF  •  Excel (xlsx, xls, ods)  •  JSON / JSONL  •  "
    "Parquet, Feather, ORC, Arrow  •  Stata (dta)  •  SAS (sas7bdat)  •  "
    "SPSS (sav)  •  XML  •  HTML  •  SQLite (db, sqlite, sqlite3)  •  "
    "PDF  •  Images (png, jpg, jpeg, webp, bmp, tiff, gif)  •  Pickle (pkl)  •  AutoML export ZIP"
)

st.markdown(
    f"""
    <div class="glass-panel upload-shell">
        <div class="section-label">INGEST PORTAL</div>
        <div class="upload-shell__head">
            <div>
                <div class="upload-shell__title">Dataset Dock</div>
                <div class="upload-shell__sub">Push in raw files, document-style sources, database files, or restore a previous AutoML export bundle.</div>
            </div>
            <div class="format-chip">Multi-format ready</div>
        </div>
    """,
    unsafe_allow_html=True,
)
st.caption(f"**Supported formats:** {SUPPORTED_FORMATS}")

if import_mode == "Upload File":
    upload_handling = st.selectbox(
        "Upload Handling",
        [
            "Auto Detect",
            "PDF - Plain Text",
            "PDF - Tables",
            "Delimited Text / CSV",
            "Database File",
            "Image OCR",
        ],
        help="Auto Detect is the default and recommended path for most uploads.",
    )

    uploaded_file = st.file_uploader(
        "📂 Drag & Drop your Dataset here",
        type=None,
        help="Supports tabular files, SQLite databases, PDFs, image uploads with OCR extraction, text-style documents, and exported AutoML ZIP bundles.",
    )

    if uploaded_file is not None:
        last_file = st.session_state.get('last_analyzed_file')
        current_file_key = f"{uploaded_file.name}_{uploaded_file.size}_{upload_handling}"

        if last_file != current_file_key:
            with st.spinner("🧬 Analyzing Neural DNA..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
                pdf_mode = "text"
                if upload_handling == "PDF - Tables":
                    pdf_mode = "tables"
                try:
                    res = requests.post(
                        f"{API_URL}/upload",
                        files=files,
                        data={"pdf_mode": pdf_mode},
                        timeout=600,
                    )
                    if res.status_code == 200:
                        data = res.json()
                        if "error" in data:
                            st.error(data["error"])
                        else:
                            st.session_state['dataset_id'] = data['dataset_id']
                            st.session_state['profile'] = data['profile']
                            st.session_state["upload_preview_records"] = data.get("preview_records", [])
                            st.session_state["upload_ingest_summary"] = data.get("ingest_summary", {})
                            st.session_state['last_analyzed_file'] = current_file_key
                            imported_job_id = data.get("imported_job_id")
                            if imported_job_id:
                                st.session_state['job_id'] = imported_job_id
                                st.success(f"✅ **{uploaded_file.name}** restored successfully!")
                                st.info("This export bundle brought back the training dataset and its completed run. You can jump straight to Results Console or retrain from here.")
                            else:
                                st.success(f"✅ **{uploaded_file.name}** processed successfully!")

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
                            sync_workspace_query_params()
                    else:
                        st.error(f"Backend error: {res.status_code}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot reach backend. The internal API service may still be starting.")
elif import_mode == "Connectors":
    connector_left, connector_right = st.columns([1.1, 0.9])
    with connector_left:
        connector_type = st.selectbox(
            "Connector",
            ["PostgreSQL", "MySQL", "Snowflake", "BigQuery"],
            key="connector_type",
        )
        connector_uri = st.text_input(
            "Connection URI",
            placeholder="e.g. postgresql+psycopg://user:pass@host:5432/dbname",
            key="connector_uri",
        )
    with connector_right:
        connector_query = st.text_area(
            "SQL Query",
            value="SELECT * FROM your_table LIMIT 5000",
            height=110,
            key="connector_query",
        )

    if st.button("🔌 Import From Connector", width="stretch"):
        if not connector_uri.strip() or not connector_query.strip():
            st.error("Add both a connection URI and a SQL query.")
        else:
            with st.spinner("Importing from external source..."):
                try:
                    res = requests.post(
                        f"{API_URL}/import-source",
                        json={
                            "source_type": connector_type.lower(),
                            "connection_uri": connector_uri.strip(),
                            "query": connector_query.strip(),
                        },
                        timeout=120,
                    )
                    data = res.json()
                    if res.status_code == 200 and not data.get("error"):
                        st.session_state["dataset_id"] = data["dataset_id"]
                        st.session_state["profile"] = data["profile"]
                        st.session_state["upload_preview_records"] = data.get("preview_records", [])
                        st.session_state["upload_ingest_summary"] = data.get("ingest_summary", {})
                        sync_workspace_query_params()
                        st.success(f"{connector_type} import completed.")
                    else:
                        st.error(data.get("error", f"Import failed: HTTP {res.status_code}"))
                except Exception as e:
                    st.error(f"Import failed: {e}")
else:
    st.markdown("### Dataset Merge Studio", unsafe_allow_html=True)
    st.caption("Combine two datasets with guided pickers, preview join quality, then load the merged dataset into the workspace.")
    try:
        ds_payload = requests.get(f"{API_URL}/datasets", timeout=10).json()
        dataset_items = ds_payload.get("datasets", [])
    except Exception:
        dataset_items = []

    dataset_items = sorted(
        dataset_items,
        key=lambda item: (
            0 if item.get("id") == st.session_state.get("dataset_id") else 1,
            item.get("created_at") or "",
        ),
        reverse=False,
    )

    dataset_options = {}
    for item in dataset_items:
        label = (
            f"{(item.get('source_type') or 'dataset').replace('_', ' ').title()} · "
            f"{item.get('rows', 0)} rows · {item.get('cols', 0)} cols · "
            f"{(item.get('id') or '')[:8]}"
        )
        dataset_options[label] = item

    left_dataset_id = ""
    right_dataset_id = ""
    left_join_key = ""
    right_join_key = ""
    merge_type = "inner"
    if not dataset_options:
        st.info("Upload or import at least two datasets to use Merge Studio.")
    else:
        merge_left, merge_right = st.columns(2)
        with merge_left:
            left_label = st.selectbox(
                "Left Dataset",
                options=list(dataset_options.keys()),
                index=0,
            )
            left_dataset = dataset_options.get(left_label, {})
            left_dataset_id = left_dataset.get("id") or st.session_state.get("dataset_id") or ""
            left_columns = left_dataset.get("columns") or []
            left_join_key = st.selectbox(
                "Left Join Key",
                options=left_columns,
                index=0 if left_columns else 0,
            ) if left_columns else ""
        with merge_right:
            right_label = st.selectbox(
                "Right Dataset",
                options=list(dataset_options.keys()),
                index=1 if len(dataset_options) > 1 else 0,
            )
            right_dataset = dataset_options.get(right_label, {})
            right_dataset_id = right_dataset.get("id") or ""
            right_columns = right_dataset.get("columns") or []
            right_join_key = st.selectbox(
                "Right Join Key",
                options=right_columns,
                index=0 if right_columns else 0,
            ) if right_columns else ""
        merge_type = st.selectbox("Join Type", ["inner", "left", "right", "outer"])
        merge_signature = (left_dataset_id, right_dataset_id, left_join_key, right_join_key, merge_type)
        if st.session_state.get("home_merge_signature") != merge_signature:
            st.session_state["home_merge_signature"] = merge_signature
            st.session_state["home_merge_preview"] = {}

        st.caption(f"Available datasets in catalog: {len(dataset_options)}")
        if left_dataset_id == right_dataset_id and left_dataset_id:
            st.warning("Choose two different datasets to merge.")

    if st.button("🔍 Preview Merge", width="stretch"):
        if left_dataset_id == right_dataset_id:
            st.error("Pick two different datasets.")
        elif not all([left_dataset_id, right_dataset_id, left_join_key, right_join_key]):
            st.error("Choose both datasets and both join keys.")
        else:
            try:
                preview_res = requests.post(
                    f"{API_URL}/merge-studio/preview",
                    json={
                        "left_dataset_id": left_dataset_id,
                        "right_dataset_id": right_dataset_id,
                        "join_key_left": left_join_key,
                        "join_key_right": right_join_key,
                        "join_type": merge_type,
                    },
                    timeout=60,
                )
                st.session_state["home_merge_preview"] = preview_res.json()
            except Exception as e:
                st.session_state["home_merge_preview"] = {"error": str(e)}

    merge_preview = st.session_state.get("home_merge_preview") or {}
    if merge_preview:
        if merge_preview.get("error"):
            st.error(merge_preview["error"])
        else:
            preview_cols = st.columns(5)
            preview_cols[0].metric("Estimated Rows", merge_preview.get("estimated_rows", 0))
            preview_cols[1].metric("Overlap Keys", merge_preview.get("overlapping_keys", 0))
            preview_cols[2].metric("Left Match %", f"{merge_preview.get('left_match_pct', 0)}%")
            preview_cols[3].metric("Right Match %", f"{merge_preview.get('right_match_pct', 0)}%")
            preview_cols[4].metric("Row Multiplier", merge_preview.get("estimated_row_multiplier", "—"))
            st.caption(
                f"Duplicates on join keys: left {merge_preview.get('left_duplicate_keys', 0)} · "
                f"right {merge_preview.get('right_duplicate_keys', 0)}"
            )
            if merge_preview.get("join_key_coerced_to_string"):
                st.info(
                    "Join keys had different data types, so preview matching was normalized using string values."
                )
            if merge_preview.get("preview_records"):
                render_safe_dataframe(merge_preview["preview_records"], width="stretch", hide_index=True)

    if st.button("🧬 Merge Datasets", width="stretch"):
        if left_dataset_id == right_dataset_id:
            st.error("Pick two different datasets.")
        elif not all([left_dataset_id, right_dataset_id, left_join_key, right_join_key]):
            st.error("Choose both datasets and both join keys.")
        else:
            with st.spinner("Merging datasets..."):
                try:
                    merge_res = requests.post(
                        f"{API_URL}/merge-studio",
                        json={
                            "left_dataset_id": left_dataset_id,
                            "right_dataset_id": right_dataset_id,
                            "join_key_left": left_join_key,
                            "join_key_right": right_join_key,
                            "join_type": merge_type,
                        },
                        timeout=120,
                    )
                    merge_data = merge_res.json()
                    if merge_res.status_code == 200 and not merge_data.get("error"):
                        st.session_state["dataset_id"] = merge_data["dataset_id"]
                        st.session_state["profile"] = merge_data["profile"]
                        st.session_state["upload_preview_records"] = merge_data.get("preview_records", [])
                        st.session_state["upload_ingest_summary"] = merge_data.get("ingest_summary", {})
                        st.session_state["home_merge_preview"] = {}
                        sync_workspace_query_params()
                        st.success("Merged dataset loaded into workspace.")
                        merge_summary = merge_data.get("merge_summary", {})
                        st.caption(
                            f"{merge_summary.get('join_type', 'inner')} join produced "
                            f"{merge_summary.get('merged_rows', 0)} rows."
                        )
                        if merge_summary.get("join_key_coerced_to_string"):
                            st.info(
                                "The join keys had different data types, so the merge was performed after converting both keys to string."
                            )
                    else:
                        st.error(merge_data.get("error", "Merge failed."))
                except Exception as e:
                    st.error(f"Merge failed: {e}")
st.markdown(
    """
        <div class="upload-shell__footer">
            <div class="footer-pill">Auto profiling</div>
            <div class="footer-pill">Problem inference</div>
            <div class="footer-pill">Artifact restoration</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.get('profile'):
    ingest_summary = st.session_state.get("upload_ingest_summary") or {}
    preview_records = st.session_state.get("upload_preview_records") or []

    if preview_records:
        st.markdown("### Ingestion Preview", unsafe_allow_html=True)
        render_safe_dataframe(preview_records, width="stretch", hide_index=True)
        if ingest_summary:
            ingest_cols = st.columns(4)
            ingest_cols[0].metric("Source Type", ingest_summary.get("source_type", "—"))
            ingest_cols[1].metric("Rows", ingest_summary.get("rows", 0))
            ingest_cols[2].metric("Columns", ingest_summary.get("columns", 0))
            ingest_cols[3].metric("Preview Rows", len(preview_records))

        if any("ocr_text" in row for row in preview_records):
            default_ocr = "\n".join(str(row.get("ocr_text", "")).strip() for row in preview_records if row.get("ocr_text"))
            reviewed_ocr = st.text_area(
                "Editable OCR Review",
                value=default_ocr,
                height=180,
                help="Edit extracted OCR text before creating a text dataset from it.",
            )
            if st.button("📝 Create Dataset From Reviewed OCR", width="stretch"):
                try:
                    ocr_res = requests.post(
                        f"{API_URL}/dataset/{st.session_state['dataset_id']}/ocr-review",
                        json={"text": reviewed_ocr},
                        timeout=60,
                    )
                    ocr_data = ocr_res.json()
                    if ocr_res.status_code == 200 and not ocr_data.get("error"):
                        st.session_state["dataset_id"] = ocr_data["dataset_id"]
                        st.session_state["profile"] = ocr_data["profile"]
                        st.session_state["upload_preview_records"] = ocr_data.get("preview_records", [])
                        st.session_state["upload_ingest_summary"] = ocr_data.get("ingest_summary", {})
                        sync_workspace_query_params()
                        st.success("OCR-reviewed dataset created and loaded into the workspace.")
                    else:
                        st.error(ocr_data.get("error", "Failed to create OCR-reviewed dataset."))
                except Exception as e:
                    st.error(f"OCR review failed: {e}")

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
            f"""
            <div class="detect-banner" style="--detect-accent:{det_color};">
                <div>
                    <div class="detect-banner__eyebrow">AUTO-DETECTED</div>
                    <div class="detect-banner__title">{icon} {task_type.title()} mission · Target <code>{suggested_t}</code></div>
                    <div class="detect-banner__copy">{task_reason}</div>
                </div>
                <div class="detect-banner__score">{confidence}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if warnings:
            for w in warnings:
                if "⚠️" in w or "🔴" in w:
                    st.warning(w)
                elif "🟡" in w:
                    st.info(w)

    profile = st.session_state['profile']
    rows = profile.get("rows", "—")
    columns_count = len(profile.get("columns", []) or [])
    missing_total = profile.get("missing_values", 0)
    st.markdown(
        f"""
        <div class="profile-strip">
            <div class="profile-strip__item"><span>Rows</span><strong>{rows}</strong></div>
            <div class="profile-strip__item"><span>Columns</span><strong>{columns_count}</strong></div>
            <div class="profile-strip__item"><span>Missing Cells</span><strong>{missing_total}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Training Configuration")

    preset_defaults = {
        "Fast": {"goal": "⚡ Speed(fast)", "mode": "⚡ Fast (Exploration only)", "cv": 3, "imbalance": False, "auto_clean": True},
        "Balanced": {"goal": "⚖️ Balanced", "mode": "⚖️ Balanced (Standard optimization)", "cv": 5, "imbalance": False, "auto_clean": True},
        "High Accuracy": {"goal": "🎯 Performance(best results)", "mode": "🧠 Full (Deep Bayesian Search)", "cv": 7, "imbalance": True, "auto_clean": True},
        "Explainable": {"goal": "⚖️ Balanced", "mode": "⚖️ Balanced (Standard optimization)", "cv": 5, "imbalance": False, "auto_clean": True},
        "Low Resource": {"goal": "⚡ Speed(fast)", "mode": "⚡ Fast (Exploration only)", "cv": 3, "imbalance": False, "auto_clean": True},
    }

    columns = profile.get('columns', [])
    suggested_target = profile.get('suggested_target')
    if not suggested_target or suggested_target not in columns:
        suggested_target = columns[-1] if columns else None

    default_idx = columns.index(suggested_target) if suggested_target in columns else 0
    st.info(f"💡 Auto-detected target column: **{suggested_target}**")
    target_col = st.selectbox("🎯 Confirm or Change Target Column", columns, index=default_idx)

    sanitizer_report = profile.get("sanitizer", {}) or {}
    sanitizer_logs = profile.get("sanitizer_logs", []) or []
    if sanitizer_report:
        san_cols = st.columns(4)
        san_cols[0].metric("Rows After Cleanup", sanitizer_report.get("rows_after", rows))
        san_cols[1].metric("Duplicates Removed", sanitizer_report.get("duplicate_rows_removed", 0))
        san_cols[2].metric("Numeric Coercions", len(sanitizer_report.get("numeric_coercions", []) or []))
        san_cols[3].metric("Datetime Columns", len(sanitizer_report.get("datetime_columns", []) or []))
        if sanitizer_logs:
            with st.expander("Shared Sanitizer Details", expanded=False):
                for line in sanitizer_logs[:10]:
                    st.caption(f"• {line}")

    workspace_col, preset_col, resume_col = st.columns([1, 1, 0.8])
    with workspace_col:
        workspace_name = st.text_input("Workspace Name", key="workspace_name_choice")
    with preset_col:
        preset_name = st.selectbox(
            "Training Preset",
            options=list(preset_defaults.keys()),
            key="preset_choice",
            help="Reusable presets tune goal, mode, and validation defaults.",
        )
    with resume_col:
        st.markdown("<div style='height: 1.9rem'></div>", unsafe_allow_html=True)
        if st.button("Resume Last Run", width="stretch"):
            try:
                resume_payload = requests.get(f"{API_URL}/workspaces/resume", timeout=10).json()
                if resume_payload.get("job_id"):
                    st.session_state["job_id"] = resume_payload["job_id"]
                    st.session_state["dataset_id"] = resume_payload.get("dataset_id")
                    sync_workspace_query_params(workspace=workspace_name)
                    st.switch_page("pages/4_Results_Console.py")
                else:
                    st.info(resume_payload.get("error", "No completed run available yet."))
            except Exception as e:
                st.error(f"Resume failed: {e}")

    preset = preset_defaults.get(preset_name, preset_defaults["Balanced"])
    goal = preset["goal"]
    mode = preset["mode"]
    if st.session_state.get("cv_folds_choice", 0) <= 1:
        st.session_state["cv_folds_choice"] = preset["cv"]
    if "handle_imbalance_choice" in st.session_state:
        st.session_state["handle_imbalance_choice"] = st.session_state.get("handle_imbalance_choice", preset["imbalance"])
    if "auto_clean_choice" in st.session_state:
        st.session_state["auto_clean_choice"] = st.session_state.get("auto_clean_choice", preset["auto_clean"])

    col1, col2 = st.columns(2)
    with col1:
        goal = st.radio(
            "Select Goal",
            ["🎯 Performance(best results)", "⚡ Speed(fast)", "⚖️ Balanced"],
            key="goal_choice",
            index=["🎯 Performance(best results)", "⚡ Speed(fast)", "⚖️ Balanced"].index(goal),
            help="Performance = widest model search | Speed = smaller fast pool | Balanced = strong middle ground"
        )
    with col2:
        mode = st.radio(
            "Select Execution Mode (V3)",
            ["⚡ Fast (Exploration only)", "⚖️ Balanced (Standard optimization)", "🧠 Full (Deep Bayesian Search)"],
            key="mode_choice",
            index=["⚡ Fast (Exploration only)", "⚖️ Balanced (Standard optimization)", "🧠 Full (Deep Bayesian Search)"].index(mode),
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
                if st.session_state.get("eval_metric_choice") not in {"RMSE", "R²"}:
                    st.session_state["eval_metric_choice"] = "RMSE"
                eval_metric = st.selectbox(
                    "Metric (Regression)",
                    ["RMSE", "R²"],
                    key="eval_metric_choice",
                    help="Default: RMSE. RMSE = Root Mean Squared Error | R² = Coefficient of Determination"
                )
            else:
                if st.session_state.get("eval_metric_choice") not in {"Accuracy", "F1-score"}:
                    st.session_state["eval_metric_choice"] = "Accuracy"
                eval_metric = st.selectbox(
                    "Metric (Classification)",
                    ["Accuracy", "F1-score"],
                    key="eval_metric_choice",
                    help="Default: Accuracy. F1 = good for imbalanced data."
                )

            st.markdown("---")

            # ── 7. Handle Imbalance ───────────────────────────────────────
            st.markdown("#### ⚖️ Handle Imbalance")
            handle_imbalance = st.toggle(
                "Handle Class Imbalance",
                key="handle_imbalance_choice",
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
                key="auto_clean_choice",
                help="Fills missing values, removes duplicates, and encodes categories automatically"
            )

            st.markdown("---")

            # ── 10. Cross-Validation ─────────────────────────────────────
            st.markdown("#### 🧪 Cross-Validation")
            cv_folds = st.slider(
                "Number of CV Folds",
                min_value=0,
                max_value=10,
                key="cv_folds_choice",
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

        if st.button("Reset Advanced Options To Recommended Defaults", width="stretch"):
            st.session_state["eval_metric_choice"] = "RMSE" if task_type == "regression" else "Accuracy"
            st.session_state["handle_imbalance_choice"] = False
            st.session_state["auto_clean_choice"] = True
            st.session_state["cv_folds_choice"] = 0
            st.rerun()

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

    forecast_payload = {}
    try:
        forecast_payload = get_training_forecast_cached(
            json.dumps(
                {
                    "dataset_id": st.session_state["dataset_id"],
                    "target_column": target_col,
                    "goal": goal_map.get(goal, "Performance"),
                    "mode": mode_map.get(mode, "Balanced"),
                    "eval_metric": eval_metric,
                    "selected_features": selected_features if selected_features else [],
                    "handle_imbalance": handle_imbalance,
                    "auto_clean": auto_clean,
                    "cv_folds": cv_folds or preset["cv"],
                    "preset_name": preset_name,
                },
                sort_keys=True,
            )
        )
    except Exception:
        forecast_payload = {}

    if forecast_payload and not forecast_payload.get("error"):
        st.markdown("### ⏱ Training Forecast")
        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Estimated Runtime", forecast_payload.get("estimated_duration_label", "—"))
        fc2.metric("Compute", forecast_payload.get("compute_intensity", "—"))
        fc3.metric("Models", forecast_payload.get("estimated_model_count", "—"))
        fc4.metric("Sweep Rows", forecast_payload.get("estimated_sweep_rows", "—"))
        st.caption(
            f"Memory risk: {forecast_payload.get('memory_risk', '—')} · "
            f"Optuna trials: {forecast_payload.get('optuna_trials', 0)} · "
            f"Features in play: {forecast_payload.get('estimated_feature_count', '—')}"
        )
        for note in forecast_payload.get("notes", [])[:3]:
            st.caption(f"• {note}")

    st.caption(
        f"Advanced summary: metric `{eval_metric}` · imbalance `{handle_imbalance}` · auto-clean `{auto_clean}` · CV folds `{cv_folds}`"
    )
    sync_workspace_query_params(
        workspace=workspace_name,
        preset=preset_name,
        goal=goal,
        mode=mode,
        metric=eval_metric,
        imbalance=str(handle_imbalance).lower(),
        auto_clean=str(auto_clean).lower(),
        cv_folds=cv_folds,
    )

    if st.button("▶ Run AutoML Engine", key="run_automl", width="stretch"):
        workspace_id = ""
        try:
            workspace_resp = requests.post(
                f"{API_URL}/workspaces",
                json={
                    "name": workspace_name,
                    "dataset_id": st.session_state["dataset_id"],
                    "settings": {
                        "preset_name": preset_name,
                        "goal": goal,
                        "mode": mode,
                        "eval_metric": eval_metric,
                        "cv_folds": cv_folds or preset["cv"],
                    },
                },
                timeout=10,
            )
            workspace_data = workspace_resp.json() if workspace_resp.status_code == 200 else {}
            workspace_id = workspace_data.get("id", "")
        except Exception:
            workspace_id = ""
        payload = {
            "dataset_id": st.session_state['dataset_id'],
            "target_column": target_col,
            "goal": goal_map.get(goal, "Performance"),
            "mode": mode_map.get(mode, "Fast"),
            "preset_name": preset_name,
            "workspace_id": workspace_id,
            "workspace_name": workspace_name,
            # Advanced options
            "eval_metric": eval_metric,
            "selected_features": selected_features if selected_features else [],
            "handle_imbalance": handle_imbalance,
            "auto_clean": auto_clean,
            "cv_folds": cv_folds or preset["cv"],
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
                    sync_workspace_query_params(
                        workspace=workspace_name,
                        preset=preset_name,
                        goal=goal,
                        mode=mode,
                        metric=eval_metric,
                        imbalance=str(handle_imbalance).lower(),
                        auto_clean=str(auto_clean).lower(),
                        cv_folds=cv_folds,
                    )
                    st.success("🚀 Training started!")
                    st.info("👉 Navigate to **Live Training** in the sidebar to watch progress.")
                    st.switch_page("pages/3_Training_Lab.py")
            else:
                st.error("Failed to start training.")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot reach backend.")

    st.markdown('</div>', unsafe_allow_html=True)
