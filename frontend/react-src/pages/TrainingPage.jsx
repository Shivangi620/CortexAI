import React from "react";
import {
  DataTable,
  KeyValueList,
  MiniAreaChart,
  NeuralPulse,
  PageHero,
  Panel,
  StatCard,
  TimelineList,
  Badge,
} from "../components/ui.jsx";
import { formatNumber } from "../lib/format.js";
import { navigateTo } from "../config/routes.js";

function toFiniteNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function toConfidencePercent(value, fallback = 61) {
  const numeric = toFiniteNumber(value);
  if (numeric === null) return fallback;
  const normalized = numeric >= 0 && numeric <= 1 ? numeric * 100 : numeric;
  return Math.max(0, Math.min(99, normalized));
}

function formatValidationStatus(value) {
  const status = String(value || "")
    .replaceAll("_", " ")
    .trim();
  if (!status) return "—";
  return status.replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatDisplayPercent(value) {
  const numeric = toFiniteNumber(value);
  if (numeric === null) return "—";
  const normalized = Math.abs(numeric) <= 1 ? numeric * 100 : numeric;
  return `${formatNumber(normalized, 1)}%`;
}

export function TrainingPage({ forms, forecast, trainingRegistryPreview, jobs, jobStatus }) {
  const status = jobStatus?.status || "idle";
  const activeJob = jobs.find((job) => job.id === jobStatus?.id) || null;
  const completedRuns = jobs.filter((job) => job.status === "completed").length;
  const activeRuns = jobs.filter((job) => job.status === "training").length;
  const history = jobStatus?.history || [];
  const reasoning = jobStatus?.reasoning || [];
  const results = jobStatus?.results || {};
  const runConfig = jobStatus?.config || {};
  const launchContext = runConfig.launch_context || null;
  const execProfile = results?.execution_profile || {};
  const performanceMetrics = results?.performance_metrics || {};
  const allMetrics = performanceMetrics?.all_metrics || {};
  const validationSummary = results?.validation_summary || {};
  const liveRegistry = jobStatus?.model_registry_preview || trainingRegistryPreview || null;
  const registryTraits = liveRegistry?.dataset_traits || {};
  const registryGroups = liveRegistry?.model_groups || {};
  const estimatedStages = Math.max(
    forecast?.estimated_model_count || 0,
    history.length || 0,
    status === "completed" ? 1 : 0,
    5,
  );
  const rawProgress = estimatedStages ? Math.round((history.length / estimatedStages) * 100) : 0;
  const progress = status === "completed" ? 100 : status === "training" ? Math.max(8, Math.min(rawProgress, 96)) : 0;
  const confidence = Math.max(
    22,
    toConfidencePercent(results.score ?? history[history.length - 1]?.metric ?? forecast?.confidence_estimate, 61),
  );
  const telemetryActivity = status === "training" ? 0.96 : status === "completed" ? 0.72 : 0.34;
  const modelCount = history.length || 0;
  const liveMetric = history[history.length - 1]?.metric ?? results.score ?? "—";
  const expectedDuration = forecast?.estimated_duration_label || "Pending calibration";
  const currentStageLabel =
    history[history.length - 1]?.time || (status === "training" ? `Stage ${Math.max(history.length, 1)}` : "Queued");

  const reasoningItems = reasoning.slice(-8).map((item, index) => ({
    title: `Reasoning step ${index + 1}`,
    detail: item,
    meta: status,
  }));

  const historyPoints = history.map((item, index) => ({
    label: item.time || `Step ${index + 1}`,
    value: Number(item.metric) || index + 1,
  }));

  const historyItems = history.map((item, index) => ({
    title: item.time || `Step ${index + 1}`,
    detail: `Metric: ${formatNumber(item.metric)}`,
    meta: item.time === "Final" ? "Final Outcome" : "Checkpoint",
  }));

  const orchestrationSignals = [
    { label: "Realtime Polling", value: status === "training" ? "2.5s cadence" : "Standby" },
    { label: "Connectors", value: forms.importMode === "Connectors" ? "Live source selected" : "File intake ready" },
    {
      label: "Export Bundle",
      value: [forms.exportModel, forms.exportCode, forms.exportReport].filter(Boolean).length + "/3 enabled",
    },
    { label: "Target Field", value: forms.targetColumn || "Awaiting dataset target" },
  ];
  const completedSnapshotItems = [
    { label: "Winner", value: results.best_model || "Unknown" },
    { label: "Metric", value: results.metric_name || "—" },
    { label: "Score", value: formatNumber(results.score, 4) },
    {
      label: "Holdout check",
      value:
        results.holdout_score === null || results.holdout_score === undefined
          ? "—"
          : `${results.holdout_score_label || "Holdout"} ${formatNumber(results.holdout_score, 4)}`,
    },
    {
      label: "CV vs holdout gap",
      value:
        validationSummary?.absolute_gap_display === null || validationSummary?.absolute_gap_display === undefined
          ? "—"
          : formatNumber(validationSummary.absolute_gap_display, 4),
    },
    { label: "Target", value: results.target || forms.targetColumn || "—" },
  ];
  const completedMetricItems = results?.is_classification
    ? [
        { label: "Accuracy", value: formatDisplayPercent(allMetrics?.accuracy) },
        { label: "Precision", value: formatDisplayPercent(allMetrics?.precision) },
        { label: "Recall", value: formatDisplayPercent(allMetrics?.recall) },
        { label: "F1", value: formatDisplayPercent(allMetrics?.f1) },
        { label: "ROC-AUC", value: formatDisplayPercent(allMetrics?.roc_auc) },
      ]
    : [
        { label: "R²", value: formatNumber(allMetrics?.r2, 4) },
        { label: "MAE", value: formatNumber(allMetrics?.mae, 4) },
        { label: "MSE", value: formatNumber(allMetrics?.mse, 4) },
        { label: "RMSE", value: formatNumber(allMetrics?.rmse, 4) },
      ];

  return (
    <>
      <PageHero
        eyebrow="Training Lab"
        title="Realtime orchestration for live training, simulation, and delivery"
        description="Run the model, watch stochastic pulse telemetry, and keep prediction, scenario, ensemble, and connector surfaces aligned from intake through inference contract."
        stats={[
          { label: "Run ID", value: jobStatus?.id?.slice(0, 8) || "None" },
          { label: "Status", value: status.toUpperCase(), detail: "current execution state" },
          { label: "Active Jobs", value: activeRuns, detail: "studio-wide concurrent runs" },
          { label: "Completed Runs", value: completedRuns, detail: "historical wins available" },
        ]}
      />

      {status === "training" && (
        <div className="message message--info" style={{ marginBottom: "2rem" }}>
          Live training is active. The studio is polling backend metrics continuously and refreshing telemetry in near
          realtime.
        </div>
      )}

      {status === "failed" && (
        <div className="message message--warning" style={{ marginBottom: "2rem" }}>
          Training failed: {jobStatus?.error || "Unknown backend error"}
        </div>
      )}

      {launchContext?.source === "drift_recommendation" && (
        <Panel
          title="Launch Context"
          subtitle="This run was reopened from a drift recommendation, so the training handoff stays visible while the new challenger set is running."
        >
          <div className="grid grid--stats">
            <StatCard label="Launch Source" value="Drift Recommendation" tone="warning" />
            <StatCard
              label="Parent Run"
              value={launchContext.parent_job_id?.slice(0, 8) || launchContext.source_job_id?.slice(0, 8) || "—"}
            />
            <StatCard label="Recommended Goal" value={launchContext.recommended_goal || runConfig.goal || "—"} />
            <StatCard label="Recommended Mode" value={launchContext.recommended_mode || runConfig.mode || "—"} />
          </div>
          {launchContext.message && (
            <div className="sidebar-note" style={{ marginTop: "12px" }}>
              <strong>Why reopened:</strong> {launchContext.message}
            </div>
          )}
          <div className="grid grid--two" style={{ marginTop: "16px" }}>
            <div className="sidebar-note">
              <strong>Current winner:</strong> {launchContext.current_model || "—"}
            </div>
            <div className="sidebar-note">
              <strong>Historical winner:</strong> {launchContext.historical_winner || "—"}
            </div>
          </div>
          {Array.isArray(launchContext.candidate_models) && launchContext.candidate_models.length > 0 && (
            <div className="training-chip-grid" style={{ marginTop: "12px" }}>
              {launchContext.candidate_models.map((modelName) => (
                <span key={modelName} className="training-chip training-chip--soft">
                  {modelName}
                </span>
              ))}
            </div>
          )}
        </Panel>
      )}

      {(activeJob || launchContext?.source === "drift_recommendation") && (
        <Panel
          title="Run lineage"
          subtitle="A compact trail keeps the active job connected to its dataset and reopen source while training is live."
        >
          <div className="training-chip-grid">
            <span className="training-chip training-chip--soft">
              Dataset {String(activeJob?.dataset_id || forms.dataset_id || "—").slice(0, 8)}
            </span>
            <span className="tiny">→</span>
            <span className="training-chip training-chip--soft">
              {launchContext?.source === "drift_recommendation" ? "Drift Reopen" : "Manual Launch"}
            </span>
            <span className="tiny">→</span>
            <span className="training-chip training-chip--soft">Run {jobStatus?.id?.slice(0, 8) || "pending"}</span>
            {results?.best_model ? (
              <>
                <span className="tiny">→</span>
                <span className="training-chip training-chip--soft">{results.best_model}</span>
              </>
            ) : null}
          </div>
        </Panel>
      )}

      <div className="training-command-grid">
        <Panel
          title="Mission Progress"
          subtitle="Realtime execution rail with stage checkpoints, ETA, and confidence-linked pulse intensity."
          actions={
            <Badge tone={status === "completed" ? "success" : status === "failed" ? "warning" : "default"}>
              {status}
            </Badge>
          }
        >
          <div className="training-progress-shell">
            <div className="training-progress-topline">
              <div>
                <span className="tiny-eyebrow">Realtime Progress</span>
                <div className="training-progress-value">{progress}%</div>
              </div>
              <div className="training-progress-meta">
                <span>Stage</span>
                <strong>{currentStageLabel}</strong>
                <span>ETA</span>
                <strong>{expectedDuration}</strong>
              </div>
            </div>
            <div className="training-progress-track" aria-hidden="true">
              <div className="training-progress-fill" style={{ width: `${progress}%` }} />
            </div>
            <div className="training-progress-caption">
              <span>{modelCount} checkpoints observed</span>
              <span>{estimatedStages} estimated stages</span>
            </div>
            <div className="training-signal-grid">
              <StatCard
                label="Live Metric"
                value={formatNumber(liveMetric)}
                meta="Latest checkpoint score"
                tone="accent"
              />
              <StatCard label="Forecasted Runtime" value={expectedDuration} meta="Pre-run estimate" />
              <StatCard label="CV Folds" value={forms.trainCvFolds} meta="Validation depth" />
              <StatCard
                label="Execution Profile"
                value={execProfile.sweep_size || forecast?.estimated_model_count || "Auto"}
                meta="Candidate sweep width"
              />
            </div>
          </div>
        </Panel>

        <Panel
          title="Neural Pulse Telemetry"
          subtitle="Live visualization of the model's stochastic vital signs. Ripple intensity maps to confidence."
        >
          <div className="training-telemetry-shell">
            <NeuralPulse confidence={confidence} activity={telemetryActivity} />
            <div className="stack compact">
              {orchestrationSignals.map((signal) => (
                <div key={signal.label} className="training-inline-stat">
                  <span>{signal.label}</span>
                  <strong>{signal.value}</strong>
                </div>
              ))}
            </div>
          </div>
          <MiniAreaChart points={historyPoints} valueKey="value" labelKey="label" />
        </Panel>
      </div>

      <div className="grid grid--stats">
        <StatCard label="Single Prediction" value="Ready" meta="Instant payload scoring lane" tone="success" />
        <StatCard label="Future Sweep" value="Ready" meta="Forward scenario trajectories" />
        <StatCard label="Scenario Simulator" value="Ready" meta="Multi-case policy analysis" />
        <StatCard label="Inference Contract" value="Guarded" meta="Schema and export checks active" tone="warning" />
      </div>

      <div className="grid grid--two">
        <Panel title="Training Log" subtitle="Live score trajectory and checkpoint snapshots from the active run.">
          <DataTable
            columns={[
              { key: "time", label: "Checkpoint" },
              { key: "metric", label: "Metric Value", render: (row) => formatNumber(row.metric) },
            ]}
            rows={history}
            compact
          />
          <TimelineList items={historyItems} empty="Warming up the engine..." />
        </Panel>

        <Panel title="Live Reasoning" subtitle="What the AutoML agent is thinking and doing right now.">
          <TimelineList items={reasoningItems} empty="Reasoning steps appear once a run begins." />
          <KeyValueList
            items={[
              { label: "Best model", value: results.best_model || "Pending selection" },
              { label: "Metric", value: results.metric_name || "Awaiting metric" },
              { label: "Score", value: formatNumber(results.score) },
              { label: "Top K", value: execProfile.top_k || "Auto" },
            ]}
          />
        </Panel>
      </div>

      {liveRegistry && (
        <Panel
          title="Selected Registry Snapshot"
          subtitle="The same model registry chosen at launch, carried forward into the live run so the execution story stays connected."
        >
          <div className="training-registry-preview training-registry-preview--embedded">
            <div className="training-progress-topline">
              <div>
                <span className="tiny-eyebrow">Execution Lane</span>
                <div className="training-progress-value" style={{ fontSize: "clamp(1.3rem, 2.4vw, 2rem)" }}>
                  {liveRegistry.selection_goal} Registry
                </div>
              </div>
              <div className="training-progress-meta">
                <span>Mode</span>
                <strong>{liveRegistry.mode || forms.trainMode || "—"}</strong>
                <span>Task</span>
                <strong>{formatValidationStatus(liveRegistry.task_type)}</strong>
              </div>
            </div>

            <div className="grid grid--stats" style={{ marginTop: "12px" }}>
              <StatCard label="Models" value={liveRegistry.selected_models?.length || 0} />
              <StatCard label="Baseline" value={registryGroups.baseline?.length || 0} />
              <StatCard label="Boosting" value={registryGroups.boosting?.length || 0} />
              <StatCard label="Optional" value={registryGroups.optional?.length || 0} />
            </div>

            <div className="training-chip-grid" style={{ marginTop: "12px" }}>
              {(liveRegistry.selected_models || []).map((modelName) => (
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
                <strong>Conditional:</strong> {registryGroups.optional.join(", ")}
              </div>
            )}

            <div className="grid grid--stats" style={{ marginTop: "12px" }}>
              <StatCard label="Small Data" value={registryTraits.small_dataset ? "Yes" : "No"} />
              <StatCard label="High Dim" value={registryTraits.high_dimensional ? "Yes" : "No"} />
              <StatCard label="KNN Allowed" value={registryTraits.knn_allowed ? "Yes" : "No"} />
              <StatCard label="MLP Allowed" value={registryTraits.mlp_allowed ? "Yes" : "No"} />
            </div>

            {liveRegistry.meta_advisory?.reason && (
              <div className="sidebar-note" style={{ marginTop: "12px" }}>
                <strong>Meta advisory:</strong> {liveRegistry.meta_advisory.reason}
              </div>
            )}

            {liveRegistry.rules?.length > 0 && (
              <details className="detail-json" style={{ marginTop: "12px" }}>
                <summary>Registry Rules In Effect</summary>
                <ul className="tiny training-rule-list">
                  {liveRegistry.rules.map((rule, index) => (
                    <li key={`${rule}-${index}`}>{rule}</li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        </Panel>
      )}

      {status === "completed" && (
        <Panel
          title="Training Completed"
          subtitle="The AutoML engine has finished and the downstream analysis surface is unlocked."
        >
          <div className="grid grid--stats">
            <StatCard label="Best Model" value={results.best_model || "Unknown"} tone="success" />
            <StatCard label={results.metric_name || "Score"} value={formatNumber(results.score)} tone="accent" />
            <StatCard label="Sweep Size" value={execProfile.sweep_size || "Auto"} />
            <StatCard label="Optuna Trials" value={execProfile.n_trials || "0"} />
          </div>
          <div className="grid grid--two" style={{ marginTop: "1rem" }}>
            <div className="feature-card">
              <span className="tiny-eyebrow">Winner Snapshot</span>
              <div className="results-summary-list">
                {completedSnapshotItems.map((item) => (
                  <div key={item.label} className="results-summary-row">
                    <span>{item.label}</span>
                    <strong>{item.value}</strong>
                  </div>
                ))}
              </div>
            </div>
            <div className="feature-card">
              <div className="training-progress-topline">
                <div>
                  <span className="tiny-eyebrow">Ready For Review</span>
                  <div className="training-progress-value" style={{ fontSize: "clamp(1.6rem, 3vw, 2.4rem)" }}>
                    {formatValidationStatus(validationSummary?.status)}
                  </div>
                </div>
                <Badge
                  tone={
                    validationSummary?.status === "possible_overfit" || validationSummary?.status === "watch"
                      ? "warning"
                      : "success"
                  }
                >
                  {results?.is_classification ? "Classification" : "Regression"}
                </Badge>
              </div>
              <div className="results-summary-list">
                {completedMetricItems.map((item) => (
                  <div key={item.label} className="results-summary-row">
                    <span>{item.label}</span>
                    <strong>{item.value}</strong>
                  </div>
                ))}
                <div className="results-summary-row">
                  <span>Holdout drift</span>
                  <strong>{validationSummary?.message || "Results console has the full review."}</strong>
                </div>
              </div>
            </div>
          </div>
          <div className="inline-actions">
            <button className="button button--primary" type="button" onClick={() => navigateTo("/results")}>
              View Results Console
            </button>
          </div>
        </Panel>
      )}
    </>
  );
}
