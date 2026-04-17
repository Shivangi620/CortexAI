import streamlit as st
import requests

from ui_shell import (
    API_URL,
    ensure_session_state,
    load_css,
    render_page_shell,
    render_section_intro,
    render_workspace_banner,
)

st.set_page_config(page_title="Home - AutoML Studio", page_icon="🏠", layout="wide")

load_css()
ensure_session_state()
st.session_state.setdefault("upload_preview_records", [])
st.session_state.setdefault("upload_ingest_summary", {})
st.session_state.setdefault("home_merge_preview", {})
st.session_state.setdefault("goal_choice", "🎯 Performance(best results)")
st.session_state.setdefault("mode_choice", "⚖️ Balanced (Standard optimization)")
st.session_state.setdefault("eval_metric_choice", "Accuracy")
st.session_state.setdefault("handle_imbalance_choice", False)
st.session_state.setdefault("auto_clean_choice", True)
st.session_state.setdefault("cv_folds_choice", 0)

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
render_section_intro(
    "Workspace Flow",
    "Ingest data, confirm the prediction setup, and launch training.",
    "This page keeps the high-friction steps together: file upload, auto-detection, training preferences, advanced controls, and export options.",
)

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
            if merge_preview.get("preview_records"):
                st.dataframe(merge_preview["preview_records"], width="stretch", hide_index=True)

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
                        st.success("Merged dataset loaded into workspace.")
                        merge_summary = merge_data.get("merge_summary", {})
                        st.caption(
                            f"{merge_summary.get('join_type', 'inner')} join produced "
                            f"{merge_summary.get('merged_rows', 0)} rows."
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
        st.dataframe(preview_records, width="stretch", hide_index=True)
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
            key="goal_choice",
            help="Performance = widest model search | Speed = smaller fast pool | Balanced = strong middle ground"
        )
    with col2:
        mode = st.radio(
            "Select Execution Mode (V3)",
            ["⚡ Fast (Exploration only)", "⚖️ Balanced (Standard optimization)", "🧠 Full (Deep Bayesian Search)"],
            key="mode_choice",
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
