import React from "react";
import {
  BarList,
  DataTable,
  EmptyState,
  Message,
  MiniAreaChart,
  PageHero,
  Panel,
  ScatterPlot,
  Heatmap,
  StatCard,
  Badge,
} from "../components/ui.jsx";
import { formatNumber, formatPercent } from "../lib/format.js";

function toReadableText(value, fallback = "Unavailable") {
  if (value === null || value === undefined || value === "") return fallback;
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value))
    return (
      value
        .map((item) => toReadableText(item, ""))
        .filter(Boolean)
        .join(" • ") || fallback
    );
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return fallback;
  }
}

function asFeatureRows(payload, fallbackKey) {
  if (!payload) return [];
  const direct = payload.feature_importance || payload.features || payload[fallbackKey] || [];
  if (Array.isArray(direct)) {
    return direct
      .map((item) => ({
        label: item.feature || item.name || item.label,
        value: item.importance ?? item.value ?? item.score ?? item.psi,
      }))
      .filter((item) => item.label);
  }
  if (direct && typeof direct === "object") {
    return Object.entries(direct).map(([label, value]) => ({ label, value }));
  }
  return [];
}

function asThresholdRows(payload) {
  const rows = payload?.thresholds || payload?.points || payload?.curve || [];
  return Array.isArray(rows)
    ? rows.map((row, index) => ({
        id: row.threshold ?? index,
        threshold: row.threshold ?? row.cutoff ?? "—",
        precision: row.precision ?? row.ppv ?? "—",
        recall: row.recall ?? row.tpr ?? "—",
        score: row.score ?? row.f1 ?? "—",
      }))
    : [];
}

function formatValidationStatus(value) {
  const status = String(value || "")
    .replaceAll("_", " ")
    .trim();
  if (!status) return "—";
  return status.replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatDisplayPercent(value) {
  if (value === null || value === undefined || value === "") return "—";
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return String(value);
  return formatPercent(Math.abs(numeric) <= 1 ? numeric * 100 : numeric);
}

function formatGapRatio(value) {
  if (value === null || value === undefined || value === "") return "—";
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return String(value);
  return `${formatNumber(numeric * 100, 2)}%`;
}

function normalizeRecommendation(item, index) {
  if (item && typeof item === "object" && !Array.isArray(item)) {
    return {
      title: item.title || `Recommendation ${index + 1}`,
      priority: item.priority || "",
      detail: item.detail || item.description || item.message || "No recommendation detail available.",
      action: item.action || "",
    };
  }

  return {
    title: `Recommendation ${index + 1}`,
    priority: "",
    detail: toReadableText(item, "No recommendation detail available."),
    action: "",
  };
}

function asHeatmapPayload(payload, shapRows) {
  const matrix = payload?.heatmap || payload?.matrix || payload?.impact_matrix || payload?.values;
  if (!Array.isArray(matrix) || !matrix.length || !Array.isArray(matrix[0])) {
    return { data: [], xLabels: [], yLabels: [] };
  }

  const width = matrix[0].length;
  const xLabels = payload?.x_labels || payload?.feature_names || shapRows.slice(0, width).map((row) => row.label) || [];
  const yLabels = payload?.y_labels || payload?.row_labels || matrix.map((_, index) => `Row ${index + 1}`);

  return {
    data: matrix,
    xLabels: xLabels.slice(0, width),
    yLabels: yLabels.slice(0, matrix.length),
  };
}

function reviewItems(items) {
  return items.map((item) => ({
    ...item,
    value: item.value === null || item.value === undefined || item.value === "" ? "—" : item.value,
  }));
}

export function ResultsPage({
  results,
  launchContext,
  jobs,
  shapSummary,
  permutationSummary,
  pipelineGraph,
  featureLineage,
  calibration,
  thresholds,
  trustHeatmap,
  recommendations,
  narration,
  selectedJobId,
  scenarioResult,
}) {
  if (!selectedJobId) {
    return (
      <>
        <PageHero
          eyebrow="Decision Room"
          title="Select a completed run to open the results room"
          description="Results become available after a training run finishes successfully. Pick a run from the sidebar to load metrics, explanations, and exports."
        />
        <EmptyState text="No run is currently selected." />
      </>
    );
  }

  if (!results) {
    return (
      <>
        <PageHero
          eyebrow="Decision Room"
          title="Results are not available for this run yet"
          description="If training is still running, the results room will populate automatically. If the run failed, review the Training page reasoning stream for the error."
        />
        <EmptyState text="This run does not have a results payload yet." />
      </>
    );
  }

  const activeJob = Array.isArray(jobs) ? jobs.find((job) => job.id === selectedJobId) : null;
  const leaderboard = Array.isArray(results?.leaderboard) ? results.leaderboard : [];
  const winner = leaderboard[0] || {};
  const scoreLabel = winner?.score_label || results?.metric_name || "Score";
  const performanceMetrics = results?.performance_metrics || {};
  const optimizedMetric = performanceMetrics?.optimized_metric || {};
  const allMetrics = performanceMetrics?.all_metrics || {};
  const shapRows = asFeatureRows(shapSummary, "shap_values").slice(0, 8);
  const permutationRows = asFeatureRows(permutationSummary, "feature_importance").slice(0, 8);
  const trustRows = asFeatureRows(trustHeatmap, "matrix").slice(0, 8);
  const recommendationRows = Array.isArray(recommendations?.recommendations)
    ? recommendations.recommendations.map(normalizeRecommendation)
    : [];
  const thresholdRows = asThresholdRows(thresholds);
  const validationSummary = results?.validation_summary || {};
  const warnings = Array.isArray(results?.warnings) ? results.warnings : [];
  const validationTone =
    validationSummary?.status === "possible_overfit" || validationSummary?.status === "watch" ? "warning" : "success";
  const calibrationBins = calibration?.bins || calibration?.curve || [];
  const calibrationPoints = calibrationBins.map((item, index) => ({
    label: item.bin || item.bucket || `Bin ${index + 1}`,
    value: item.fraction_positive ?? item.observed ?? item.accuracy ?? item.value ?? item.confidence ?? index + 1,
  }));
  const heatmapPayload = asHeatmapPayload(shapSummary, shapRows);
  const leaderboardColumns = results?.is_classification
    ? [
        { key: "model", label: "Model" },
        { key: "score", label: scoreLabel, render: (row) => formatNumber(row.score, 2) },
        { key: "phase", label: "Phase", render: (row) => toReadableText(row.phase, "—") },
        { key: "holdout_score", label: "Holdout", render: (row) => formatNumber(row.holdout_score, 4) },
        { key: "absolute_gap_display", label: "Gap", render: (row) => formatNumber(row.absolute_gap_display, 4) },
        { key: "validation_status", label: "Drift", render: (row) => formatValidationStatus(row.validation_status) },
        { key: "accuracy", label: "Accuracy", render: (row) => formatDisplayPercent(row.accuracy) },
        { key: "precision", label: "Precision", render: (row) => formatDisplayPercent(row.precision) },
        { key: "recall", label: "Recall", render: (row) => formatDisplayPercent(row.recall) },
        { key: "f1", label: "F1", render: (row) => formatDisplayPercent(row.f1) },
        { key: "roc_auc", label: "ROC-AUC", render: (row) => formatDisplayPercent(row.roc_auc) },
      ]
    : [
        { key: "model", label: "Model" },
        { key: "score", label: scoreLabel, render: (row) => formatNumber(row.score, 4) },
        { key: "phase", label: "Phase", render: (row) => toReadableText(row.phase, "—") },
        { key: "holdout_score", label: "Holdout", render: (row) => formatNumber(row.holdout_score, 4) },
        { key: "absolute_gap_display", label: "Gap", render: (row) => formatNumber(row.absolute_gap_display, 4) },
        { key: "validation_status", label: "Drift", render: (row) => formatValidationStatus(row.validation_status) },
        { key: "r2", label: "R²", render: (row) => formatNumber(row.r2, 4) },
        { key: "mae", label: "MAE", render: (row) => formatNumber(row.mae, 4) },
        { key: "mse", label: "MSE", render: (row) => formatNumber(row.mse, 4) },
        { key: "rmse", label: "RMSE", render: (row) => formatNumber(row.rmse, 4) },
      ];
  const metricItems = results?.is_classification
    ? reviewItems([
        { label: "Optimized for", value: optimizedMetric?.requested || results?.metric_name || "—" },
        { label: "Goal", value: optimizedMetric?.goal || results?.goal || "—" },
        { label: "Mode", value: optimizedMetric?.mode || results?.mode || "—" },
        { label: "Accuracy", value: formatDisplayPercent(allMetrics?.accuracy) },
        { label: "Precision", value: formatDisplayPercent(allMetrics?.precision) },
        { label: "Recall", value: formatDisplayPercent(allMetrics?.recall) },
        { label: "F1", value: formatDisplayPercent(allMetrics?.f1) },
        { label: "ROC-AUC", value: formatDisplayPercent(allMetrics?.roc_auc) },
      ])
    : reviewItems([
        { label: "Optimized for", value: optimizedMetric?.requested || results?.metric_name || "—" },
        { label: "Goal", value: optimizedMetric?.goal || results?.goal || "—" },
        { label: "Mode", value: optimizedMetric?.mode || results?.mode || "—" },
        { label: "R²", value: formatNumber(allMetrics?.r2, 4) },
        { label: "MAE", value: formatNumber(allMetrics?.mae, 4) },
        { label: "MSE", value: formatNumber(allMetrics?.mse, 4) },
        { label: "RMSE", value: formatNumber(allMetrics?.rmse, 4) },
        {
          label: "MAPE",
          value:
            allMetrics?.mape === null || allMetrics?.mape === undefined ? "—" : formatDisplayPercent(allMetrics?.mape),
        },
      ]);
  const outcomeItems = reviewItems([
    { label: "Winner", value: results?.best_model || winner?.model || "—" },
    { label: "Metric", value: results?.metric_name || scoreLabel },
    { label: "Score", value: formatNumber(results?.score ?? winner?.score, 4) },
    {
      label: "Holdout check",
      value:
        results?.holdout_score === null || results?.holdout_score === undefined
          ? "—"
          : `${results?.holdout_score_label || "Holdout"} ${formatNumber(results?.holdout_score, 4)}`,
    },
    {
      label: "CV vs holdout gap",
      value:
        validationSummary?.absolute_gap_display === null || validationSummary?.absolute_gap_display === undefined
          ? "—"
          : formatNumber(validationSummary?.absolute_gap_display, 4),
    },
    { label: "Target", value: results?.target || "—" },
  ]);
  const validationItems = reviewItems([
    { label: "Status", value: formatValidationStatus(validationSummary?.status) },
    { label: validationSummary?.score_label || "CV score", value: formatNumber(validationSummary?.cv_score, 4) },
    {
      label: validationSummary?.holdout_score_label || "Holdout score",
      value: formatNumber(validationSummary?.holdout_score, 4),
    },
    { label: "Absolute gap", value: formatNumber(validationSummary?.absolute_gap_display, 4) },
    { label: "Relative gap", value: formatGapRatio(validationSummary?.absolute_gap_ratio) },
  ]);
  const scenarioRows = Array.isArray(scenarioResult?.scenarios) ? scenarioResult.scenarios : [];
  const isDriftReopen = launchContext?.source === "drift_recommendation";
  const parentScore = Number(launchContext?.parent_score);
  const currentScore = Number(results?.score ?? winner?.score);
  const parentGap = Number(launchContext?.parent_validation_gap);
  const currentGap = Number(validationSummary?.absolute_gap_display);
  const hasRetrainDelta = Number.isFinite(parentScore) || Number.isFinite(parentGap);
  const scoreDelta = Number.isFinite(parentScore) && Number.isFinite(currentScore) ? currentScore - parentScore : null;
  const gapDelta = Number.isFinite(parentGap) && Number.isFinite(currentGap) ? currentGap - parentGap : null;
  const runOriginItems = reviewItems([
    { label: "Launch source", value: "Drift recommendation" },
    {
      label: "Parent run",
      value: launchContext?.parent_job_id?.slice(0, 8) || launchContext?.source_job_id?.slice(0, 8) || "—",
    },
    {
      label: "Recommended lane",
      value:
        launchContext?.recommended_goal || launchContext?.recommended_mode
          ? `${launchContext?.recommended_goal || "Balanced"} / ${launchContext?.recommended_mode || "Balanced"}`
          : "—",
    },
    { label: "Current winner", value: launchContext?.current_model || "—" },
    { label: "Historical winner", value: launchContext?.historical_winner || "—" },
    {
      label: "Parent score",
      value: Number.isFinite(parentScore) ? formatNumber(parentScore, 4) : "—",
    },
    {
      label: "Score delta",
      value: Number.isFinite(scoreDelta) ? `${scoreDelta >= 0 ? "+" : ""}${formatNumber(scoreDelta, 4)}` : "—",
    },
    {
      label: "Parent validation gap",
      value: Number.isFinite(parentGap) ? formatNumber(parentGap, 4) : "—",
    },
    {
      label: "Gap delta",
      value: Number.isFinite(gapDelta) ? `${gapDelta >= 0 ? "+" : ""}${formatNumber(gapDelta, 4)}` : "—",
    },
  ]);
  const artifactRows = [
    {
      category: "SHAP Drivers",
      status: shapSummary ? "Generated" : "Missing",
      detail: `${shapRows.length} features analyzed`,
    },
    {
      category: "Permutation",
      status: permutationSummary ? "Generated" : "Missing",
      detail: `${permutationRows.length} rounds complete`,
    },
    {
      category: "Trust Matrix",
      status: trustHeatmap ? "Verified" : "Pending",
      detail: `${trustRows.length} stable signals`,
    },
    {
      category: "Thresholds",
      status: thresholds ? "Optimized" : "Default",
      detail: `${thresholdRows.length} points calculated`,
    },
    {
      category: "Pipeline Graph",
      status: pipelineGraph ? "Captured" : "Missing",
      detail: pipelineGraph ? "Execution path returned" : "No graph payload returned",
    },
    {
      category: "Feature Lineage",
      status: featureLineage ? "Mapped" : "Pending",
      detail: featureLineage ? "Lineage context available" : "No lineage payload returned",
    },
  ];

  return (
    <>
      <PageHero
        eyebrow="Decision Room"
        title="Outcome review for the winning run"
        description="Read the winner in seconds, verify CV-versus-holdout stability, inspect thresholds and explainability, then open the deployment artifacts."
        stats={[
          { label: "Winner", value: results?.best_model || winner?.model || "—" },
          { label: "Metric", value: results?.metric_name || scoreLabel },
          { label: "Score", value: formatNumber(results?.score ?? winner?.score, 4) },
          { label: "Task", value: results?.is_classification ? "Classification" : "Regression" },
        ]}
      />

      <div className="grid grid--stats">
        <StatCard
          label="Outcome"
          value={results?.best_model || winner?.model || "—"}
          meta={results?.metric_name || scoreLabel}
          tone="warning"
        />
        <StatCard
          label="Holdout Check"
          value={
            results?.holdout_score === null || results?.holdout_score === undefined
              ? "—"
              : formatNumber(results?.holdout_score, 4)
          }
          meta={results?.holdout_score_label || "Holdout validation"}
        />
        <StatCard
          label="Validation Drift"
          value={formatValidationStatus(validationSummary?.status)}
          meta={validationSummary?.message || "CV vs holdout stability"}
          tone={validationTone}
        />
        <StatCard
          label="Runner-Up"
          value={leaderboard[1]?.model || "—"}
          meta={leaderboard[1] ? `${formatNumber(leaderboard[1].score, 4)} ${scoreLabel}` : "No second run"}
          tone="success"
        />
      </div>

      <Panel
        title="Run lineage"
        subtitle="The review room keeps the dataset-to-run trail visible so drift reopen runs and manual launches read cleanly."
      >
        <div className="training-chip-grid">
          <span className="training-chip training-chip--soft">
            Dataset {String(activeJob?.dataset_id || results?.dataset_id || "—").slice(0, 8)}
          </span>
          <span className="tiny">→</span>
          <span className="training-chip training-chip--soft">{isDriftReopen ? "Drift Reopen" : "Manual Launch"}</span>
          <span className="tiny">→</span>
          <span className="training-chip training-chip--soft">Run {selectedJobId?.slice(0, 8) || "—"}</span>
          <span className="tiny">→</span>
          <span className="training-chip training-chip--soft">
            {results?.best_model || winner?.model || "Outcome pending"}
          </span>
        </div>
      </Panel>

      {isDriftReopen && (
        <Panel
          title="Run Origin"
          subtitle="This completed run was reopened from a drift recommendation, so the review room keeps that operational context visible."
        >
          <div className="results-summary-list">
            {runOriginItems.map((item) => (
              <div key={item.label} className="results-summary-row">
                <span>{item.label}</span>
                <strong>{item.value}</strong>
              </div>
            ))}
          </div>
          {launchContext?.message && (
            <div className="results-note">
              <Message text={launchContext.message} tone="info" />
            </div>
          )}
          {hasRetrainDelta && (
            <div className="results-note">
              <Message
                tone={Number.isFinite(scoreDelta) && scoreDelta >= 0 ? "info" : "warning"}
                text={
                  Number.isFinite(scoreDelta) && Number.isFinite(gapDelta)
                    ? `Compared with the parent run, score moved ${scoreDelta >= 0 ? "up" : "down"} by ${formatNumber(Math.abs(scoreDelta), 4)} and validation gap ${gapDelta <= 0 ? "tightened" : "widened"} by ${formatNumber(Math.abs(gapDelta), 4)}.`
                    : "Parent-run metrics were carried into this retrain so the recovery delta stays visible."
                }
              />
            </div>
          )}
          {Array.isArray(launchContext?.candidate_models) && launchContext.candidate_models.length > 0 && (
            <div className="results-note">
              <span className="tiny-eyebrow">Challenger Set</span>
              <div className="training-chip-grid" style={{ marginTop: "12px" }}>
                {launchContext.candidate_models.map((modelName) => (
                  <span key={modelName} className="training-chip training-chip--soft">
                    {modelName}
                  </span>
                ))}
              </div>
            </div>
          )}
        </Panel>
      )}

      <div className="grid grid--two">
        <Panel title="Outcome brief" subtitle="A compact summary of the winning run that can be read in seconds.">
          <div className="results-summary-list">
            {outcomeItems.map((item) => (
              <div key={item.label} className="results-summary-row">
                <span>{item.label}</span>
                <strong>{item.value}</strong>
              </div>
            ))}
          </div>
        </Panel>

        <Panel
          title="Performance metrics"
          subtitle="All metrics are computed for the winning run while optimization still follows the advanced metric and mode you selected."
        >
          <div className="results-summary-list">
            {metricItems.map((item) => (
              <div key={item.label} className="results-summary-row">
                <span>{item.label}</span>
                <strong>{item.value}</strong>
              </div>
            ))}
          </div>
        </Panel>

        <Panel
          title="Validation drift"
          subtitle="The winning run now exposes CV-vs-holdout drift directly so leaderboard movement is easier to trust."
          actions={<Badge tone={validationTone}>{formatValidationStatus(validationSummary?.status)}</Badge>}
        >
          <div className="results-summary-list">
            {validationItems.map((item) => (
              <div key={item.label} className="results-summary-row">
                <span>{item.label}</span>
                <strong>{item.value}</strong>
              </div>
            ))}
          </div>
          <div className="results-note">
            <Message
              text={validationSummary?.message || "Validation drift details were not returned for this run."}
              tone={validationTone === "success" ? "info" : "warning"}
            />
          </div>
        </Panel>

        <Panel
          title="Scenario simulator snapshot"
          subtitle="Live what-if comparisons stay visible beside the final metrics once you run them from the Advanced Lab."
        >
          <div className="results-summary-list">
            <div className="results-summary-row">
              <span>Baseline</span>
              <strong>{scenarioResult?.baseline?.prediction ?? "—"}</strong>
            </div>
            <div className="results-summary-row">
              <span>Scenarios</span>
              <strong>{scenarioRows.length || 0}</strong>
            </div>
            <div className="results-summary-row">
              <span>Sweep feature</span>
              <strong>{scenarioResult?.sweep_feature || "—"}</strong>
            </div>
            <div className="results-summary-row">
              <span>Cohort rows</span>
              <strong>{scenarioResult?.cohort?.rows || "—"}</strong>
            </div>
          </div>
          <div className="results-note">
            {scenarioRows.length > 0 ? (
              <DataTable
                compact
                columns={[
                  { key: "name", label: "Scenario" },
                  { key: "prediction", label: "Prediction", render: (row) => formatNumber(row.prediction, 4) },
                  { key: "delta", label: "Delta", render: (row) => formatNumber(row.delta, 4) },
                  {
                    key: "delta_pct",
                    label: "Delta %",
                    render: (row) =>
                      row.delta_pct === null || row.delta_pct === undefined
                        ? "—"
                        : `${formatNumber(row.delta_pct, 2)}%`,
                  },
                ]}
                rows={scenarioRows.slice(0, 4)}
              />
            ) : (
              <div className="empty-state">
                Run the Scenario Studio from Advanced Lab to compare what-if cases side by side.
              </div>
            )}
          </div>
        </Panel>

        <Panel
          title="Calibration trace"
          subtitle="A simple shape chart keeps the confidence story visible without extra dependencies."
        >
          {calibrationPoints.length ? (
            <>
              <MiniAreaChart points={calibrationPoints} valueKey="value" labelKey="label" />
              <div className="results-note">
                <span className="tiny-eyebrow">Calibration Bins</span>
                <DataTable
                  compact
                  columns={[
                    {
                      key: "mean_predicted",
                      label: "Predicted",
                      render: (row) => formatNumber(row.mean_predicted ?? row.expected),
                    },
                    {
                      key: "fraction_positive",
                      label: "Observed",
                      render: (row) => formatNumber(row.fraction_positive ?? row.observed),
                    },
                  ]}
                  rows={calibrationBins.slice(0, 6)}
                />
              </div>
            </>
          ) : (
            <EmptyState text="Calibration trace is not available for this run." />
          )}
        </Panel>
      </div>

      <div className="grid grid--two">
        <Panel
          title="Leaderboard"
          subtitle="Model ranking stays front and center, now with a stronger table treatment."
          actions={
            <Badge tone={isDriftReopen ? "warning" : "default"}>{isDriftReopen ? "Drift Reopen" : "Manual Run"}</Badge>
          }
        >
          <div className="results-leaderboard-head">
            <div className="results-leaderboard-spotlight">
              <span className="tiny-eyebrow">Winner</span>
              <strong>{results?.best_model || winner?.model || "—"}</strong>
              <small>
                {formatNumber(results?.score ?? winner?.score, 4)} {scoreLabel}
              </small>
            </div>
            <div className="results-leaderboard-spotlight">
              <span className="tiny-eyebrow">Runner-up</span>
              <strong>{leaderboard[1]?.model || "—"}</strong>
              <small>
                {leaderboard[1] ? `${formatNumber(leaderboard[1].score, 4)} ${scoreLabel}` : "No comparison run"}
              </small>
            </div>
          </div>
          <div className="results-note">
            <MiniAreaChart
              points={leaderboard.slice(0, 10).map((row) => ({ label: row.model, value: row.score }))}
              valueKey="value"
              labelKey="label"
            />
          </div>
          <DataTable
            columns={leaderboardColumns}
            rows={leaderboard}
            empty="No leaderboard details were returned for this run."
          />
        </Panel>

        <Panel
          title="Quality warnings"
          subtitle="Guardrails highlight drift, variance, imbalance, and leakage risks before you ship."
        >
          {warnings.length ? (
            <div className="stack">
              {warnings.map((warning, index) => (
                <Message
                  key={`${warning.type || "warning"}-${index}`}
                  tone={warning.severity === "high" ? "warning" : "info"}
                  text={`${formatValidationStatus(warning.type)}: ${warning.message}`}
                />
              ))}
            </div>
          ) : (
            <EmptyState text="No quality warnings were triggered for this run." />
          )}
        </Panel>
      </div>

      {results?.is_classification ? (
        <>
          <div className="grid grid--two">
            <Panel title="Threshold review" subtitle="Useful for classification deployment and stakeholder review.">
              <DataTable
                columns={[
                  { key: "threshold", label: "Threshold" },
                  { key: "precision", label: "Precision", render: (row) => formatNumber(row.precision) },
                  { key: "recall", label: "Recall", render: (row) => formatNumber(row.recall) },
                  { key: "score", label: "Score", render: (row) => formatNumber(row.score) },
                ]}
                rows={thresholdRows}
                empty="No threshold tuning payload was returned."
                compact
              />
            </Panel>

            <Panel
              title="Precision-Recall Curve"
              subtitle="Visualizing the trade-off between precision and recall across thresholds."
            >
              <ScatterPlot
                points={thresholdRows.map((row) => ({ x: row.recall, y: row.precision }))}
                xKey="x"
                yKey="y"
              />
              <div className="chart-card__labels">
                <span>Recall</span>
                <span>Precision</span>
              </div>
            </Panel>
          </div>

          <div className="grid grid--two">
            <Panel
              title="Trust surface"
              subtitle="Trust or sensitivity signals returned by the backend are shown as ranked bars."
            >
              <BarList items={trustRows} empty="Trust heatmap summary is not available for this run." />
            </Panel>
          </div>
        </>
      ) : (
        <div className="grid grid--two">
          <Panel
            title="Regression diagnostics"
            subtitle="Numeric prediction quality is summarized with error and fit metrics for this run."
          >
            <div className="results-summary-list">
              {metricItems.slice(3).map((item) => (
                <div key={item.label} className="results-summary-row">
                  <span>{item.label}</span>
                  <strong>{item.value}</strong>
                </div>
              ))}
            </div>
          </Panel>

          <Panel
            title="Trust surface"
            subtitle="Trust or sensitivity signals returned by the backend are shown as ranked bars."
          >
            <BarList items={trustRows} empty="Trust heatmap summary is not available for this run." />
          </Panel>
        </div>
      )}

      <div className="grid grid--two">
        <Panel
          title="Explainability Heatmap"
          subtitle="Multi-row impact analysis. Blue represents positive impact, Red represents negative impact."
        >
          {heatmapPayload.data.length ? (
            <Heatmap data={heatmapPayload.data} xLabels={heatmapPayload.xLabels} yLabels={heatmapPayload.yLabels} />
          ) : (
            <EmptyState text="No heatmap data available." />
          )}
        </Panel>

        <Panel title="SHAP drivers" subtitle="Global importance from the backend explainability service.">
          <BarList items={shapRows} empty="SHAP summary is not available for this run." />
        </Panel>
      </div>

      <div className="grid grid--two">
        <Panel title="Permutation impact" subtitle="A second feature ranking to compare against SHAP.">
          <BarList items={permutationRows} empty="Permutation importance is not available for this run." />
        </Panel>

        <Panel
          title="Narrative and recommendations"
          subtitle="The results page now reads more like a studio review room than a payload dump."
        >
          <div className="results-panel-stack">
            <div className="feature-card">
              <span className="eyebrow">Narration</span>
              <p>
                {toReadableText(
                  narration?.story || narration?.narration || results?.story,
                  "No run narrative was returned.",
                )}
              </p>
            </div>
            <div className="results-recommendation-list">
              {recommendationRows.length ? (
                recommendationRows.map((item) => (
                  <div key={item.title} className="results-recommendation">
                    <div className="results-recommendation__head">
                      <strong>{item.title}</strong>
                      {item.priority ? <span>{item.priority}</span> : null}
                    </div>
                    <p>{item.detail}</p>
                    {item.action ? <pre className="detail-json code-editor">{item.action}</pre> : null}
                  </div>
                ))
              ) : (
                <div className="empty-state">No recommendations were generated for this run.</div>
              )}
            </div>
          </div>
        </Panel>
      </div>

      <div className="grid grid--two">
        <Panel
          title="Artifacts and lineage"
          subtitle="Deployment and reporting actions stay visible, with supporting detail underneath."
        >
          <div className="inline-actions">
            <a
              className="button button--secondary"
              href={selectedJobId ? `/api/export/${selectedJobId}` : "#"}
              target="_blank"
              rel="noreferrer"
            >
              Export Bundle
            </a>
            <a
              className="button button--secondary"
              href={selectedJobId ? `/api/report/${selectedJobId}/pdf` : "#"}
              target="_blank"
              rel="noreferrer"
            >
              Open PDF Report
            </a>
            <a
              className="button button--primary"
              href={selectedJobId ? `/api/report/${selectedJobId}/model-card` : "#"}
              target="_blank"
              rel="noreferrer"
            >
              Open Model Card
            </a>
          </div>
          <div className="results-note">
            <span className="tiny-eyebrow">Technical Audit Log</span>
            <DataTable
              compact
              columns={[
                { key: "category", label: "Artifact" },
                { key: "status", label: "Status" },
                { key: "detail", label: "Key Insight" },
              ]}
              rows={artifactRows}
            />
          </div>
        </Panel>
      </div>
    </>
  );
}
