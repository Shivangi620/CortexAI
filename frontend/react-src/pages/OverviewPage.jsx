import React, { useEffect } from "react";
import {
  DataTable,
  PageHero,
  Panel,
  StatCard,
  TimelineList,
  SegmentedControl,
  Checkbox,
  Spinner,
  MiniAreaChart,
  Message,
} from "../components/ui.jsx";
import { formatDate, formatNumber } from "../lib/format.js";

export function OverviewPage({
  jobs,
  selectedDatasetId,
  datasetProfile,
  datasetDetect,
  loading,
  uploadLoading,
  uploadLoadingLabel,
  forecast,
  trainingRegistryPreview,
  forms,
  patchForm,
  uploadRef,
  handleUpload,
  uploadMessage,
  uploadPreview,
  ingestSummary,
  handleImportSource,
  handleTrain,
  trainMessage,
  handleOcrReview,
}) {
  const acceptedFileTypes =
    ".csv,.tsv,.xlsx,.xls,.json,.parquet,.pdf,.txt,.html,.htm,.xml,.zip,.sav,.sas7bdat,.dta,.feather";
  const driftReopenCount = jobs.filter((job) => job.launch_source === "drift_recommendation").length;
  const recentTimeline = jobs.slice(0, 5).map((job) => ({
    title: `${job.best_model || "Run"} • ${job.status}${job.launch_source === "drift_recommendation" ? " • Drift Reopen" : ""}`,
    detail: `Dataset ${job.dataset_id?.slice(0, 8) || "unknown"} • ${job.metric_name || "score"} ${formatNumber(job.score)}`,
    meta: formatDate(job.created_at),
  }));

  const hasReviewableText = uploadPreview.some((row) => row.ocr_text || row.text);

  useEffect(() => {
    if (hasReviewableText && !forms.ocrText) {
      const defaultOcr = uploadPreview
        .map((row) => String(row.ocr_text || row.text || "").trim())
        .filter(Boolean)
        .join("\n");
      patchForm({ ocrText: defaultOcr });
    }
  }, [uploadPreview, hasReviewableText]);

  const effectiveTaskType = forms.trainTaskType || datasetDetect?.task_type || "classification";
  const registryTraits = trainingRegistryPreview?.dataset_traits || {};
  const registryGroups = trainingRegistryPreview?.model_groups || {};
  const columnStats = datasetProfile?.column_stats || {};
  const highCardinalityColumns = Object.entries(columnStats)
    .filter(([, stats]) => Number(stats?.unique_pct) >= 0.9 && String(stats?.semantic_type || "") !== "ID/Index")
    .map(([name]) => name);
  const preflightWarnings = [
    !forms.targetColumn ? "Choose a target column before launching AutoML." : null,
    Number(datasetProfile?.missing_pct) >= 20
      ? `Missingness is ${datasetProfile?.missing_pct}%. Expect heavier cleaning and weaker calibration.`
      : null,
    effectiveTaskType === "classification" &&
    String(datasetProfile?.imbalance || "").includes("High") &&
    !forms.trainHandleImbalance
      ? "High class imbalance detected. Turn on imbalance handling before launch."
      : null,
    Number(datasetProfile?.rows) > 0 && Number(datasetProfile?.rows) < 200
      ? "Very small dataset detected. Favor fast baselines and treat leaderboard movement carefully."
      : null,
    highCardinalityColumns.length > 3
      ? `High-cardinality columns detected: ${highCardinalityColumns.slice(0, 3).join(", ")}${highCardinalityColumns.length > 3 ? "..." : ""}.`
      : null,
    forms.trainPcaMode === "always" && Number(datasetProfile?.num_cols?.length || 0) < 3
      ? "PCA is forced on a low-numeric dataset. Auto mode is usually safer here."
      : null,
  ].filter(Boolean);

  return (
    <>
      <PageHero
        eyebrow="Mission Control"
        title="Orchestrate your data, experiments, and automated deployments"
        description="The heart of AutoML Studio. Start new missions by ingesting data, configuring models, or resuming high-priority training runs."
        stats={[
          { label: "Dataset Loaded", value: selectedDatasetId ? "Yes" : "No" },
          { label: "Rows", value: datasetProfile?.rows || "—" },
          { label: "Columns", value: datasetProfile?.columns?.length || "—" },
          { label: "Drift Reopens", value: driftReopenCount, detail: "reopened from monitoring" },
        ]}
      />

      <div className="grid grid--two">
        <Panel title="Import Mode" subtitle="Choose how you want to ingest data into the studio.">
          <SegmentedControl
            options={["Upload File", "Connectors"]}
            value={forms.importMode}
            onChange={(v) => patchForm({ importMode: v })}
          />

          {forms.importMode === "Upload File" && (
            <form className="stack">
              <div className="field">
                <input type="file" ref={uploadRef} accept={acceptedFileTypes} onChange={() => handleUpload()} />
              </div>
              <div className="training-chip-grid">
                {acceptedFileTypes.split(",").map((type) => (
                  <span key={type} className="training-chip training-chip--soft">
                    {type}
                  </span>
                ))}
              </div>
              {uploadMessage && <p className="tiny">{uploadMessage}</p>}
              {uploadLoading && <Spinner label={uploadLoadingLabel || "Scanning dataset DNA..."} />}
            </form>
          )}

          {forms.importMode === "Connectors" && (
            <form className="stack" onSubmit={handleImportSource}>
              <div className="field">
                <span>Connector Type</span>
                <select value={forms.connectorType} onChange={(e) => patchForm({ connectorType: e.target.value })}>
                  <option>PostgreSQL</option>
                  <option>MySQL</option>
                  <option>Snowflake</option>
                  <option>BigQuery</option>
                </select>
              </div>
              <div className="field">
                <span>Connection URI</span>
                <input
                  type="text"
                  placeholder="postgresql://user:pass@host:5432/db"
                  value={forms.connectorUri}
                  onChange={(e) => patchForm({ connectorUri: e.target.value })}
                />
              </div>
              <div className="field">
                <span>SQL Query</span>
                <textarea
                  value={forms.connectorQuery}
                  onChange={(e) => patchForm({ connectorQuery: e.target.value })}
                />
              </div>
              <button className="button button--primary" type="submit">
                🔌 Import From Source
              </button>
              {loading && <Spinner label="Importing and scanning remote data..." />}
            </form>
          )}
        </Panel>

        <Panel title="Auto-Detect Neural Pulse" subtitle="Real-time problem detection and training forecasting.">
          <div className="stack">
            <div className="split">
              <StatCard label="Inferred Task" value={datasetDetect?.task_type || "N/A"} tone="accent" />
              <StatCard label="Confidence" value={datasetDetect?.confidence ? `${datasetDetect.confidence}%` : "0%"} />
            </div>
            {datasetDetect?.warnings?.length > 0 && (
              <div className="message message--warning">
                <strong>Potential Issues Found:</strong>
                <ul className="tiny">
                  {datasetDetect.warnings.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              </div>
            )}
            <div className="sidebar-note">
              Target Suggestions:{" "}
              {datasetDetect?.column_scores
                ?.slice(0, 3)
                .map((c) => c.column)
                .join(", ") || "None"}
            </div>
          </div>
        </Panel>
      </div>

      <div className="grid grid--two">
        <Panel
          title="Project Score Evolution"
          subtitle="Track how your model performance has improved over historical runs."
        >
          <MiniAreaChart
            points={jobs
              .slice()
              .reverse()
              .map((j) => ({ label: formatDate(j.created_at), value: j.score }))}
            valueKey="value"
            labelKey="label"
          />
        </Panel>

        <Panel title="Recent Activity Hub" subtitle="A live feed of the latest operational events in the studio.">
          <TimelineList items={recentTimeline} empty="No recent activity to show." />
        </Panel>
      </div>

      {uploadPreview.length > 0 && (
        <Panel title="Ingestion Preview" subtitle="A snapshot of the data that was just brought into the workspace.">
          <DataTable
            columns={Object.keys(uploadPreview[0]).map((k) => ({ key: k, label: k }))}
            rows={uploadPreview}
            compact
          />
          <div className="grid grid--stats" style={{ marginTop: "1rem" }}>
            <StatCard label="Rows" value={ingestSummary.rows} />
            <StatCard label="Cols" value={ingestSummary.columns} />
          </div>

          {hasReviewableText && (
            <div className="stack" style={{ marginTop: "20px" }}>
              <div className="field">
                <span>Editable OCR / Text Review</span>
                <p className="tiny">Detected extractable text. Review and edit before finalizing the dataset.</p>
                <textarea
                  value={forms.ocrText}
                  onChange={(e) => patchForm({ ocrText: e.target.value })}
                  style={{ height: "180px" }}
                />
              </div>
              {loading && <Spinner label="Creating dataset from OCR..." />}
              <button className="button button--primary" onClick={handleOcrReview}>
                📝 Create Dataset From Reviewed OCR
              </button>
            </div>
          )}
        </Panel>
      )}

      <Panel title="Training Configuration" subtitle="Define optimization goals and computational presets.">
        <div className="stack">
          {preflightWarnings.length > 0 && (
            <div className="stack compact">
              <span className="tiny-eyebrow">Preflight warnings</span>
              {preflightWarnings.map((warning, index) => (
                <Message key={`${warning}-${index}`} text={warning} tone="warning" />
              ))}
            </div>
          )}

          <div className="field">
            <span>Target Feature (Feature to Predict)</span>
            <select
              value={forms.targetColumn}
              onChange={(e) => patchForm({ targetColumn: e.target.value })}
              style={{ border: "1px solid var(--accent)", boxShadow: "0 0 10px var(--accent-alpha)" }}
            >
              <option value="">Select target...</option>
              {datasetProfile?.columns?.map((col) => (
                <option key={col} value={col}>
                  {col}
                </option>
              ))}
            </select>
            <p className="tiny" style={{ marginTop: "4px", color: "var(--accent)" }}>
              🪄 Suggested by engine: {datasetProfile?.suggested_target || "None"}
            </p>
          </div>

          <SegmentedControl
            label="Optimization Preset"
            options={["Fast", "Balanced", "Full"]}
            value={forms.trainPreset}
            onChange={(v) => patchForm({ trainPreset: v })}
          />
          <div className="grid grid--two">
            <div className="field">
              <span>Optimization Goal</span>
              <select value={forms.trainGoal} onChange={(e) => patchForm({ trainGoal: e.target.value })}>
                <option value="Balanced">Balanced</option>
                <option value="Performance">Performance</option>
                <option value="Speed">Speed</option>
                <option value="Explainability">Explainability</option>
              </select>
            </div>
            <div className="field">
              <span>Optimization Mode</span>
              <select value={forms.trainMode} onChange={(e) => patchForm({ trainMode: e.target.value })}>
                <option value="Fast">Fast</option>
                <option value="Balanced">Balanced</option>
                <option value="Full">Full</option>
              </select>
            </div>
          </div>

          {trainingRegistryPreview && (
            <div className="training-registry-preview">
              <div className="split" style={{ alignItems: "center", gap: "12px" }}>
                <div>
                  <h3 className="section-subtitle">Live Model Registry</h3>
                  <p className="tiny">
                    Exact backend-selected families before launch, including adaptive rules and the active search lane.
                  </p>
                </div>
                <span className="training-chip training-chip--soft">
                  {trainingRegistryPreview.selection_goal} Registry
                </span>
              </div>

              <div className="grid grid--stats" style={{ marginTop: "12px" }}>
                <StatCard label="Task" value={trainingRegistryPreview.task_type || effectiveTaskType} />
                <StatCard label="Models" value={trainingRegistryPreview.selected_models?.length || 0} />
                <StatCard
                  label="Search Depth"
                  value={forecast?.uses_bayesian_optimization ? `${forecast.optuna_trials || 0} trials` : "Fast sweep"}
                />
                <StatCard
                  label="Top Candidates"
                  value={forecast?.estimated_model_count || trainingRegistryPreview.selected_models?.length || "—"}
                />
              </div>

              <div className="training-chip-grid" style={{ marginTop: "12px" }}>
                {(trainingRegistryPreview.selected_models || []).map((modelName) => (
                  <span key={modelName} className="training-chip">
                    {modelName}
                  </span>
                ))}
              </div>

              <div className="grid grid--two" style={{ marginTop: "16px" }}>
                <div className="sidebar-note">
                  <strong>Baseline:</strong> {registryGroups.baseline?.join(", ") || "—"}
                </div>
                <div className="sidebar-note">
                  <strong>Boosting:</strong> {registryGroups.boosting?.join(", ") || "—"}
                </div>
              </div>

              {registryGroups.optional?.length > 0 && (
                <div className="sidebar-note" style={{ marginTop: "12px" }}>
                  <strong>Optional Specialists:</strong> {registryGroups.optional.join(", ")}
                </div>
              )}

              <div className="grid grid--stats" style={{ marginTop: "12px" }}>
                <StatCard label="Small Data" value={registryTraits.small_dataset ? "Yes" : "No"} />
                <StatCard label="High Dim" value={registryTraits.high_dimensional ? "Yes" : "No"} />
                <StatCard label="Large Data" value={registryTraits.large_dataset ? "Yes" : "No"} />
                <StatCard label="Very Large" value={registryTraits.very_large_dataset ? "Yes" : "No"} />
              </div>

              <div className="grid grid--stats" style={{ marginTop: "12px" }}>
                <StatCard
                  label="KNN"
                  value={registryTraits.knn_allowed === false ? "Skipped" : "Eligible"}
                  detail={registryTraits.knn_allowed === false ? "feature count > 50" : "pattern specialist"}
                />
                <StatCard
                  label="MLP"
                  value={registryTraits.mlp_allowed === false ? "Skipped" : "Eligible"}
                  detail={registryTraits.mlp_allowed === false ? "rows < 2k" : "neural-network fallback"}
                />
                <StatCard
                  label="Memory signal"
                  value={trainingRegistryPreview.meta_advisory?.memory_applied ? "Applied" : "Idle"}
                  detail={trainingRegistryPreview.meta_advisory?.source || "selector default"}
                />
                <StatCard
                  label="Booster lane"
                  value={registryGroups.boosting?.length || 0}
                  detail={registryGroups.boosting?.join(", ") || "none selected"}
                />
              </div>

              {trainingRegistryPreview.selection_goal !== trainingRegistryPreview.requested_goal && (
                <div className="message message--warning" style={{ marginTop: "12px" }}>
                  Full mode upgrades the model registry to the Performance lane while keeping the deeper search
                  configuration.
                </div>
              )}

              {trainingRegistryPreview.meta_advisory?.reason && (
                <div className="sidebar-note" style={{ marginTop: "12px" }}>
                  <strong>Meta advisory:</strong> {trainingRegistryPreview.meta_advisory.reason}
                </div>
              )}

              {trainingRegistryPreview.rules?.length > 0 && (
                <details className="detail-json" style={{ marginTop: "12px" }}>
                  <summary>Adaptive Selection Rules</summary>
                  <ul className="tiny training-rule-list">
                    {trainingRegistryPreview.rules.map((rule, index) => (
                      <li key={`${rule}-${index}`}>{rule}</li>
                    ))}
                  </ul>
                </details>
              )}
            </div>
          )}

          <details className="detail-json">
            <summary>⚙️ Advanced Strategy Options</summary>
            <div className="stack" style={{ marginTop: "16px" }}>
              <div className="grid grid--three">
                <div className="field">
                  <span>Task Type</span>
                  <select
                    value={forms.trainTaskType}
                    onChange={(e) => patchForm({ trainTaskType: e.target.value, trainMetric: "" })}
                  >
                    <option value="">Auto ({datasetDetect?.task_type || "detect"})</option>
                    <option value="classification">Classification</option>
                    <option value="regression">Regression</option>
                  </select>
                </div>
                <div className="field">
                  <span>Evaluation Metric</span>
                  <select value={forms.trainMetric} onChange={(e) => patchForm({ trainMetric: e.target.value })}>
                    <option value="">
                      {effectiveTaskType === "regression" ? "Auto (RMSE recommended)" : "Auto (F1-score recommended)"}
                    </option>
                    {effectiveTaskType === "regression" ? (
                      <>
                        <option>RMSE</option>
                        <option>R²</option>
                      </>
                    ) : (
                      <>
                        <option>F1-score</option>
                        <option>Accuracy</option>
                        <option>Precision</option>
                        <option>Recall</option>
                        <option>ROC-AUC</option>
                      </>
                    )}
                  </select>
                </div>
                <div className="field">
                  <span>PCA Components</span>
                  <input
                    type="number"
                    min="0"
                    max="200"
                    value={forms.trainPcaComponents}
                    onChange={(e) => patchForm({ trainPcaComponents: e.target.value })}
                  />
                </div>
              </div>
              <div className="grid grid--three">
                <div className="field">
                  <span>CV Folds</span>
                  <input
                    type="number"
                    min="0"
                    max="10"
                    value={forms.trainCvFolds}
                    onChange={(e) => patchForm({ trainCvFolds: e.target.value })}
                  />
                </div>
                <div className="field">
                  <span>PCA Mode</span>
                  <select value={forms.trainPcaMode} onChange={(e) => patchForm({ trainPcaMode: e.target.value })}>
                    <option>auto</option>
                    <option>always</option>
                    <option>off</option>
                  </select>
                </div>
              </div>
              <div className="grid grid--three">
                <Checkbox
                  label="Handle Imbalance"
                  checked={forms.trainHandleImbalance}
                  onChange={(v) => patchForm({ trainHandleImbalance: v })}
                />
                <Checkbox
                  label="Auto Clean Data"
                  checked={forms.trainAutoClean}
                  onChange={(v) => patchForm({ trainAutoClean: v })}
                />
              </div>
              <div className="grid grid--three">
                <Checkbox
                  label="Export Model"
                  checked={forms.exportModel}
                  onChange={(v) => patchForm({ exportModel: v })}
                />
                <Checkbox
                  label="Export Code"
                  checked={forms.exportCode}
                  onChange={(v) => patchForm({ exportCode: v })}
                />
                <Checkbox
                  label="Export Report"
                  checked={forms.exportReport}
                  onChange={(v) => patchForm({ exportReport: v })}
                />
              </div>
            </div>
          </details>
        </div>

        <div className="grid grid--one" style={{ marginTop: "24px" }}>
          <button
            className="button button--primary"
            onClick={handleTrain}
            style={{ height: "56px", fontSize: "1.1rem" }}
          >
            🚀 Run AutoML Engine
          </button>
        </div>
        {trainMessage && <p className="message message--warning">{trainMessage}</p>}
      </Panel>
    </>
  );
}
