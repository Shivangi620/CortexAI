import React, { useEffect } from "react";
import {
  DataTable,
  Message,
  PageHero,
  Panel,
  StatCard,
  TimelineList,
  NotificationList,
  SegmentedControl,
  Checkbox,
  Spinner,
  MiniAreaChart,
  ScatterPlot,
} from "../components/ui.jsx";
import { formatDate, formatNumber } from "../lib/format.js";

export function OverviewPage({
  datasets,
  jobs,
  notifications,
  selectedDatasetId,
  setSelectedDatasetId,
  selectedJobId,
  setSelectedJobId,
  datasetProfile,
  datasetDetect,
  loading,
  forms,
  patchForm,
  uploadRef,
  handleUpload,
  uploadMessage,
  uploadPreview,
  ingestSummary,
  handleImportSource,
  handleArchive,
  handleDeleteDataset,
  handleResumeLastRun,
  handleTrain,
  trainMessage,
  handleMergePreview,
  handleMergeApply,
  mergeState,
  handleOcrReview,
}) {
  const recentTimeline = jobs.slice(0, 5).map((job) => ({
    title: `${job.best_model || "Run"} • ${job.status}`,
    detail: `Dataset ${job.dataset_id?.slice(0, 8) || "unknown"} • ${job.metric_name || "score"} ${formatNumber(job.score)}`,
    meta: formatDate(job.created_at),
  }));

  const hasReviewableText = uploadPreview.some(row => row.ocr_text || row.text);

  useEffect(() => {
    if (hasReviewableText && !forms.ocrText) {
      const defaultOcr = uploadPreview
        .map(row => String(row.ocr_text || row.text || "").trim())
        .filter(Boolean)
        .join("\n");
      patchForm({ ocrText: defaultOcr });
    }
  }, [uploadPreview, hasReviewableText]);

  const leftColumns = datasets.find(d => d.id === forms.mergeLeftId)?.columns || [];
  const rightColumns = datasets.find(d => d.id === forms.mergeRightId)?.columns || [];

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
          { label: "Run Ready", value: datasetProfile ? "Yes" : "Waiting" },
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
                <input
                  type="file"
                  ref={uploadRef}
                  onChange={() => handleUpload()}
                />
              </div>
              {uploadMessage && <p className="tiny">{uploadMessage}</p>}
              {loading && <Spinner label="Scanning dataset DNA..." />}
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
              <button className="button button--primary" type="submit">🔌 Import From Source</button>
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
                  {datasetDetect.warnings.map((w, i) => <li key={i}>{w}</li>)}
                </ul>
              </div>
            )}
            <div className="sidebar-note">
              Target Suggestions: {datasetDetect?.column_scores?.slice(0, 3).map(c => c.column).join(", ") || "None"}
            </div>
          </div>
        </Panel>
      </div>

      <div className="grid grid--two">
        <Panel title="Project Score Evolution" subtitle="Track how your model performance has improved over historical runs.">
          <MiniAreaChart
            points={jobs.slice().reverse().map(j => ({ label: formatDate(j.created_at), value: j.score }))}
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
            columns={Object.keys(uploadPreview[0]).map(k => ({ key: k, label: k }))}
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
          <div className="field">
            <span>Target Feature (Feature to Predict)</span>
            <select 
              value={forms.targetColumn} 
              onChange={(e) => patchForm({ targetColumn: e.target.value })}
              style={{ border: "1px solid var(--accent)", boxShadow: "0 0 10px var(--accent-alpha)" }}
            >
              <option value="">Select target...</option>
              {datasetProfile?.columns?.map((col) => (
                <option key={col} value={col}>{col}</option>
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

          <details className="detail-json">
            <summary>⚙️ Advanced Strategy Options</summary>
            <div className="stack" style={{ marginTop: "16px" }}>
              <div className="grid grid--three">
                <div className="field">
                  <span>Evaluation Metric</span>
                  <select value={forms.trainMetric} onChange={(e) => patchForm({ trainMetric: e.target.value })}>
                    <option value="">Auto (recommended)</option>
                    {datasetDetect?.task_type === "regression" ? (
                      <>
                        <option>RMSE</option>
                        <option>R²</option>
                      </>
                    ) : (
                      <>
                        <option>Accuracy</option>
                        <option>F1-score</option>
                        <option>Precision</option>
                        <option>Recall</option>
                      </>
                    )}
                  </select>
                </div>
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
                <Checkbox label="Handle Imbalance" checked={forms.trainHandleImbalance} onChange={(v) => patchForm({ trainHandleImbalance: v })} />
                <Checkbox label="Auto Clean Data" checked={forms.trainAutoClean} onChange={(v) => patchForm({ trainAutoClean: v })} />
              </div>
              <div className="grid grid--three">
                <Checkbox label="Export Model" checked={forms.exportModel} onChange={(v) => patchForm({ exportModel: v })} />
                <Checkbox label="Export Code" checked={forms.exportCode} onChange={(v) => patchForm({ exportCode: v })} />
                <Checkbox label="Export Report" checked={forms.exportReport} onChange={(v) => patchForm({ exportReport: v })} />
              </div>
            </div>
          </details>
        </div>

        <div className="grid grid--one" style={{ marginTop: "24px" }}>
          <button className="button button--primary" onClick={handleTrain} style={{ height: "56px", fontSize: "1.1rem" }}>
            🚀 Run AutoML Engine
          </button>
        </div>
        {trainMessage && <p className="message message--warning">{trainMessage}</p>}
      </Panel>
    </>
  );
}
