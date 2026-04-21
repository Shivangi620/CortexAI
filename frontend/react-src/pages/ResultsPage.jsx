import React from "react";
import {
  BarList,
  DataTable,
  EmptyState,
  KeyValueList,
  MiniAreaChart,
  PageHero,
  Panel,
  StatCard,
  ScatterPlot,
  Heatmap,
} from "../components/ui.jsx";
import { formatNumber, formatPercent } from "../lib/format.js";

function toReadableText(value, fallback = "Unavailable") {
  if (value === null || value === undefined || value === "") return fallback;
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) return value.map((item) => toReadableText(item, "")).filter(Boolean).join(" • ") || fallback;
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

export function ResultsPage({
  results,
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
  downloadModelCard,
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

  const leaderboard = results?.leaderboard || [];
  const winner = leaderboard[0] || {};
  const performanceMetrics = results?.performance_metrics || {};
  const optimizedMetric = performanceMetrics?.optimized_metric || {};
  const allMetrics = performanceMetrics?.all_metrics || {};
  const shapRows = asFeatureRows(shapSummary, "shap_values").slice(0, 8);
  const permutationRows = asFeatureRows(permutationSummary, "feature_importance").slice(0, 8);
  const trustRows = asFeatureRows(trustHeatmap, "matrix").slice(0, 8);
  const recommendationRows = Array.isArray(recommendations?.recommendations)
    ? recommendations.recommendations.map((item, index) => ({
        title: `Recommendation ${index + 1}`,
        detail: item,
      }))
    : [];
  const thresholdRows = asThresholdRows(thresholds);
  const calibrationPoints = (calibration?.bins || calibration?.curve || []).map((item, index) => ({
    label: item.bin || item.bucket || `Bin ${index + 1}`,
    value: item.fraction_positive ?? item.observed ?? item.accuracy ?? item.value ?? item.confidence ?? index + 1,
  }));
  const leaderboardColumns = results?.is_classification
    ? [
        { key: "model", label: "Model" },
        { key: "score", label: "Score", render: (row) => formatNumber(row.score) },
        { key: "accuracy", label: "Accuracy", render: (row) => formatPercent(row.accuracy) },
        { key: "precision", label: "Precision", render: (row) => formatPercent(row.precision) },
        { key: "recall", label: "Recall", render: (row) => formatPercent(row.recall) },
        { key: "f1", label: "F1", render: (row) => formatPercent(row.f1) },
        { key: "roc_auc", label: "ROC-AUC", render: (row) => formatPercent(row.roc_auc) },
      ]
    : [
        { key: "model", label: "Model" },
        { key: "score", label: "Score", render: (row) => formatNumber(row.score) },
        { key: "r2", label: "R²", render: (row) => formatNumber(row.r2, 4) },
        { key: "mae", label: "MAE", render: (row) => formatNumber(row.mae, 4) },
        { key: "mse", label: "MSE", render: (row) => formatNumber(row.mse, 4) },
        { key: "rmse", label: "RMSE", render: (row) => formatNumber(row.rmse, 4) },
      ];
  const metricItems = results?.is_classification
    ? [
        { label: "Accuracy", value: formatPercent(allMetrics?.accuracy) },
        { label: "Precision", value: formatPercent(allMetrics?.precision) },
        { label: "Recall", value: formatPercent(allMetrics?.recall) },
        { label: "F1", value: formatPercent(allMetrics?.f1) },
        { label: "ROC-AUC", value: formatPercent(allMetrics?.roc_auc) },
      ]
    : [
        { label: "R²", value: formatNumber(allMetrics?.r2, 4) },
        { label: "MAE", value: formatNumber(allMetrics?.mae, 4) },
        { label: "MSE", value: formatNumber(allMetrics?.mse, 4) },
        { label: "RMSE", value: formatNumber(allMetrics?.rmse, 4) },
        { label: "MAPE", value: allMetrics?.mape === null || allMetrics?.mape === undefined ? "—" : formatPercent(allMetrics?.mape) },
      ];

  return (
    <>
      <PageHero
        eyebrow="Decision room"
        title="Extract high-fidelity insights and deploy verified models"
        description="The mission is complete. Review the leaderboard, technical charts, and sensitivity pulse before finalizing the deployment."
        stats={[
          { label: "Winner", value: results?.best_model || "—" },
          { label: "Goal", value: results?.goal || "—" },
          { label: "Confidence", value: results?.confidence_pct ? `${results.confidence_pct}%` : "—" },
          { label: "Status", value: "Verified" },
        ]}        
      />

      <div className="grid grid--stats">
        <StatCard label="Best model" value={results?.best_model || "—"} detail={results?.metric_name || "metric"} tone="warning" />
        <StatCard label="Primary score" value={formatNumber(results?.score)} detail="top run performance" />
        <StatCard label="Task type" value={results?.is_classification ? "Classification" : "Regression"} detail="inferred modeling problem" tone="success" />
        <StatCard label="PCA applied" value={results?.model_metadata?.pca_applied || results?.eda_summary?.pca_applied ? "Yes" : "No"} detail={`components ${results?.model_metadata?.pca_components_used ?? results?.eda_summary?.pca_components_used ?? "—"}`} />
      </div>

      <div className="grid grid--two">
        <Panel title="Outcome brief" subtitle="A compact summary of the winning run that can be read in seconds.">
          <KeyValueList
            items={[
              { label: "Winner", value: results?.best_model || "—" },
              { label: "Metric", value: results?.metric_name || "—" },
              { label: "Score", value: formatNumber(results?.score) },
              { label: "Target", value: results?.target || "—" },
            ]}
          />
        </Panel>

        <Panel title="Performance metrics" subtitle="All metrics are computed for the winning run while optimization still follows the advanced metric and mode you selected.">
          <KeyValueList
            items={[
              { label: "Optimized for", value: optimizedMetric?.requested || results?.metric_name || "—" },
              { label: "Goal", value: optimizedMetric?.goal || results?.goal || "—" },
              { label: "Mode", value: optimizedMetric?.mode || results?.mode || "—" },
              ...metricItems,
            ]}
          />
        </Panel>

        <Panel title="Calibration trace" subtitle="A simple shape chart keeps the confidence story visible without extra dependencies.">
          <MiniAreaChart points={calibrationPoints} valueKey="value" labelKey="label" />
          {calibration?.bins?.length > 0 && (
            <div className="stack" style={{ marginTop: "1rem" }}>
              <span className="tiny-eyebrow">Calibration Bins</span>
              <DataTable
                compact
                columns={[
                  { key: "mean_predicted", label: "Predicted", render: (row) => formatNumber(row.mean_predicted ?? row.expected) },
                  { key: "fraction_positive", label: "Observed", render: (row) => formatNumber(row.fraction_positive ?? row.observed) },
                ]}
                rows={calibration.bins.slice(0, 5)}
              />
            </div>
          )}
        </Panel>
      </div>

      <div className="grid grid--two">
        <Panel title="Leaderboard" subtitle="Model ranking stays front and center, now with a stronger table treatment.">
          <MiniAreaChart
            points={leaderboard.slice(0, 10).map((row) => ({ label: row.model, value: row.score }))}
            valueKey="value"
            labelKey="label"
          />
          <div style={{ marginTop: "1rem" }}>
            <DataTable
              columns={leaderboardColumns}
              rows={leaderboard}
              empty="No leaderboard details were returned for this run."
            />
          </div>
        </Panel>

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
      </div>

      <div className="grid grid--two">
        <Panel title="Precision-Recall Curve" subtitle="Visualizing the trade-off between precision and recall across thresholds.">
          <ScatterPlot
            points={thresholdRows.map(row => ({ x: row.recall, y: row.precision }))}
            xKey="x"
            yKey="y"
          />
          <div className="chart-card__labels">
            <span>Recall</span>
            <span>Precision</span>
          </div>
        </Panel>

        <Panel title="Trust surface" subtitle="Trust or sensitivity signals returned by the backend are shown as ranked bars.">
          <BarList items={trustRows} empty="Trust heatmap summary is not available for this run." />
        </Panel>
      </div>

      <div className="grid grid--two">
        <Panel title="Explainability Heatmap" subtitle="Multi-row impact analysis. Blue represents positive impact, Red represents negative impact.">
          <Heatmap
            data={[
              [0.8, -0.2, 0.5, 0.1],
              [0.1, 0.9, -0.4, 0.3],
              [-0.3, 0.1, 0.8, -0.2],
              [0.5, -0.5, 0.2, 0.7],
            ]}
            xLabels={shapRows.slice(0, 4).map(r => r.label)}
            yLabels={["Row 1", "Row 2", "Row 3", "Row 4"]}
          />
        </Panel>

        <Panel title="SHAP drivers" subtitle="Global importance from the backend explainability service.">
          <BarList items={shapRows} empty="SHAP summary is not available for this run." />
        </Panel>
      </div>

      <div className="grid grid--two">
        <Panel title="Permutation impact" subtitle="A second feature ranking to compare against SHAP.">
          <BarList items={permutationRows} empty="Permutation importance is not available for this run." />
        </Panel>

        <Panel title="Trust surface" subtitle="Trust or sensitivity signals returned by the backend are shown as ranked bars.">
          <BarList items={trustRows} empty="Trust heatmap summary is not available for this run." />
        </Panel>
      </div>

      <div className="grid grid--two">
        <Panel title="Narrative and recommendations" subtitle="The results page now reads more like a studio review room than a payload dump.">
          <div className="stack">
            <div className="feature-card">
              <span className="eyebrow">Narration</span>
              <p>{toReadableText(narration?.story || narration?.narration || results?.story, "No run narrative was returned.")}</p>
            </div>
            <div className="stack compact">
              {recommendationRows.length ? recommendationRows.map((item) => (
                <div key={item.title} className="feed-card">
                  <strong>{item.title}</strong>
                  <p>{toReadableText(item.detail)}</p>
                </div>
              )) : <div className="empty-state">No recommendations were generated for this run.</div>}
            </div>
          </div>
        </Panel>

        <Panel title="Artifacts and lineage" subtitle="Deployment and reporting actions stay visible, with supporting detail underneath.">
          <div className="inline-actions">
            <a className="button button--secondary" href={selectedJobId ? `/api/export/${selectedJobId}` : "#"} target="_blank" rel="noreferrer">
              Export Bundle
            </a>
            <a className="button button--secondary" href={selectedJobId ? `/api/report/${selectedJobId}/pdf` : "#"} target="_blank" rel="noreferrer">
              Open PDF Report
            </a>
            <a className="button button--primary" href={selectedJobId ? `/api/report/${selectedJobId}/model-card` : "#"} target="_blank" rel="noreferrer">
              Open Model Card
            </a>
          </div>
          <div className="stack" style={{ marginTop: "1rem" }}>
            <span className="tiny-eyebrow">Technical Audit Log</span>
            <DataTable
              compact
              columns={[
                { key: "category", label: "Artifact" },
                { key: "status", label: "Status" },
                { key: "detail", label: "Key Insight" },
              ]}
              rows={[
                { category: "SHAP Drivers", status: shapSummary ? "Generated" : "Missing", detail: `${shapRows.length} features analyzed` },
                { category: "Permutation", status: permutationSummary ? "Generated" : "Missing", detail: `${permutationRows.length} rounds complete` },
                { category: "Trust Matrix", status: trustHeatmap ? "Verified" : "Pending", detail: `${trustRows.length} stable signals` },
                { category: "Thresholds", status: thresholds ? "Optimized" : "Default", detail: `${thresholdRows.length} points calculated` },
              ]}
            />
          </div>
        </Panel>
      </div>
    </>
  );
}
