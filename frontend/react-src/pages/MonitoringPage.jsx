import React from "react";
import { DataTable, Message, MiniAreaChart, PageHero, Panel, StatCard, Spinner } from "../components/ui.jsx";
import { formatDate, formatNumber } from "../lib/format.js";

export function MonitoringPage({
  driftUploadRef,
  retrainUploadRef,
  driftDetect,
  driftScheduleSave,
  driftRetrain,
  detectMessage,
  scheduleMessage,
  retrainMessage,
  driftResult,
  driftHistory,
  driftTimeline,
  driftSchedule,
  forms,
  patchForm,
  loading,
  predictResult,
  futureResult,
  handlePredict,
  handleFutureSweep,
  getGoalSeeker,
  goalSeeker,
  datasetProfile,
}) {
  const historyRows = driftHistory?.history || [];
  const timelineRows = driftTimeline?.timeline || [];
  const alertSummary = driftResult?.alert_summary || driftSchedule?.last_alert_summary || null;
  const retrainPlan = driftResult?.retrain_recommendation || null;
  const timelinePoints = timelineRows.slice(0, 16).map((item, index) => ({
    label: item.feature || item.uploaded_name || `Check ${index + 1}`,
    value: Number(item.psi) || 0,
  }));

  return (
    <>
      <PageHero
        eyebrow="Ops Watch"
        title="Production drift, threshold schedules, and recovery actions"
        description="Monitoring now feels like a real operations surface: upload fresh samples, inspect the drift curve, tune alerting thresholds, and trigger retraining when the model falls behind reality."
        stats={[
          { label: "Drift checks", value: historyRows.length, detail: "recorded comparisons" },
          { label: "Feature events", value: timelineRows.length, detail: "timeline points" },
          {
            label: "Schedule",
            value: driftSchedule?.enabled ? "Enabled" : "Disabled",
            detail: `${driftSchedule?.frequency_days || "—"} day cadence`,
          },
          { label: "Last alert", value: driftSchedule?.last_alert_status || "—", detail: "latest alert state" },
        ]}
      />

      <div className="grid grid--stats">
        <StatCard
          label="Due now"
          value={driftSchedule?.due_now ? "Yes" : "No"}
          detail={driftSchedule?.next_due_at || "no next due date"}
          tone="warning"
        />
        <StatCard
          label="Warning PSI"
          value={formatNumber(driftSchedule?.warning_threshold)}
          detail="warning boundary"
        />
        <StatCard
          label="Critical PSI"
          value={formatNumber(driftSchedule?.critical_threshold)}
          detail="critical boundary"
          tone="success"
        />
        <StatCard
          label="Overall status"
          value={driftResult?.overall_status || "—"}
          detail={driftResult?.alert_level || "no fresh report"}
        />
      </div>

      <div className="grid grid--two">
        <Panel
          title="Fresh drift check"
          subtitle="Upload the latest production slice and compare it to the run baseline."
        >
          <form className="stack" onSubmit={driftDetect}>
            <label className="field">
              <span>Production sample</span>
              <input ref={driftUploadRef} type="file" />
            </label>
            <button className="button button--primary" type="submit">
              Run Drift Detection
            </button>
          </form>
          <Message text={detectMessage} />
          {alertSummary && (
            <div className="message message--warning tiny" style={{ marginTop: "1rem" }}>
              <strong>{alertSummary.headline || "Latest alert"}:</strong>{" "}
              {alertSummary.recommended_action || alertSummary.message || "Review the latest drift report."}
            </div>
          )}
          {retrainPlan && (
            <div className="monitoring-retrain-plan" style={{ marginTop: "1rem" }}>
              <div className="monitoring-retrain-plan__header">
                <span className="tiny-eyebrow">Recommended Retrain Lane</span>
                <strong>
                  {retrainPlan.recommended_goal || "Balanced"} / {retrainPlan.recommended_mode || "Balanced"}
                </strong>
              </div>
              <p>{retrainPlan.message || "A fresh retrain recommendation is available for this drift report."}</p>
              <div className="monitoring-retrain-plan__grid">
                <div className="monitoring-retrain-plan__item">
                  <span>Current model</span>
                  <strong>{retrainPlan.current_model || "—"}</strong>
                </div>
                <div className="monitoring-retrain-plan__item">
                  <span>Historical winner</span>
                  <strong>{retrainPlan.historical_winner || "—"}</strong>
                </div>
                <div className="monitoring-retrain-plan__item">
                  <span>Meta confidence</span>
                  <strong>{formatNumber(retrainPlan.memory_confidence)}%</strong>
                </div>
                <div className="monitoring-retrain-plan__item">
                  <span>Historical runs</span>
                  <strong>{formatNumber(retrainPlan.historical_runs)}</strong>
                </div>
              </div>
              {Array.isArray(retrainPlan.candidate_models) && retrainPlan.candidate_models.length > 0 && (
                <div className="monitoring-retrain-plan__chips">
                  {retrainPlan.candidate_models.map((model) => (
                    <span key={model} className="training-chip training-chip--soft">
                      {model}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
          {driftResult?.feature_drift?.length > 0 && (
            <div className="stack" style={{ marginTop: "1rem" }}>
              <span className="tiny-eyebrow">Drift Breakdown</span>
              <DataTable
                compact
                columns={[
                  { key: "feature", label: "Feature" },
                  { key: "psi", label: "PSI", render: (row) => formatNumber(row.psi) },
                  { key: "severity", label: "Status" },
                ]}
                rows={driftResult.feature_drift.slice(0, 8)}
              />
            </div>
          )}
        </Panel>

        <Panel title="Alert policy" subtitle="Tune monitoring cadence and thresholds without leaving the page.">
          <form className="stack" onSubmit={driftScheduleSave}>
            <label className="checkbox">
              <input
                type="checkbox"
                checked={forms.driftEnabled}
                onChange={(event) => patchForm("driftEnabled", event.target.checked)}
              />
              <span>Enable scheduled monitoring</span>
            </label>
            <div className="split">
              <label className="field">
                <span>Frequency (days)</span>
                <input
                  type="number"
                  min="1"
                  value={forms.driftFrequencyDays}
                  onChange={(event) => patchForm("driftFrequencyDays", event.target.value)}
                />
              </label>
              <label className="field">
                <span>Feature filter</span>
                <input
                  value={forms.driftFeature}
                  onChange={(event) => patchForm("driftFeature", event.target.value)}
                  placeholder="optional feature name"
                />
              </label>
            </div>
            <div className="split">
              <label className="field">
                <span>Warning threshold</span>
                <input
                  type="number"
                  step="0.01"
                  value={forms.driftWarningThreshold}
                  onChange={(event) => patchForm("driftWarningThreshold", event.target.value)}
                />
              </label>
              <label className="field">
                <span>Critical threshold</span>
                <input
                  type="number"
                  step="0.01"
                  value={forms.driftCriticalThreshold}
                  onChange={(event) => patchForm("driftCriticalThreshold", event.target.value)}
                />
              </label>
            </div>
            <button className="button button--secondary" type="submit">
              Save Policy
            </button>
          </form>
          <Message text={scheduleMessage} />
        </Panel>
      </div>

      <div className="grid grid--two">
        <Panel
          title="Feature drift curve"
          subtitle="A lightweight chart shows PSI movement over the recorded feature timeline."
        >
          <MiniAreaChart
            points={timelinePoints}
            valueKey="value"
            labelKey="label"
            empty="Run at least one drift check to generate a curve."
          />
          {driftResult?.critical_features?.length > 0 && (
            <div className="message message--danger tiny" style={{ marginTop: "1rem" }}>
              Critical features: {driftResult.critical_features.join(", ")}
            </div>
          )}
        </Panel>

        <Panel
          title="Retraining response"
          subtitle="If the drift check becomes actionable, launch retraining directly from this room."
        >
          <form className="stack" onSubmit={driftRetrain}>
            <label className="field">
              <span>Retraining dataset</span>
              <input ref={retrainUploadRef} type="file" />
            </label>
            {retrainPlan && (
              <div className="message message--accent tiny">
                This launch will use {retrainPlan.recommended_goal || "Balanced"} /{" "}
                {retrainPlan.recommended_mode || "Balanced"} from the drift recommendation.
              </div>
            )}
            <button className="button button--primary" type="submit">
              {retrainPlan
                ? `Retrain With ${retrainPlan.recommended_goal || "Balanced"} / ${retrainPlan.recommended_mode || "Balanced"}`
                : "Retrain From Drift Sample"}
            </button>
          </form>
          <Message text={retrainMessage} />
          {retrainPlan && (
            <div className="monitoring-retrain-plan monitoring-retrain-plan--compact" style={{ marginTop: "1rem" }}>
              <div className="monitoring-retrain-plan__header">
                <span className="tiny-eyebrow">Challenger Set</span>
                <strong>
                  {retrainPlan.recommended_goal || "Balanced"} / {retrainPlan.recommended_mode || "Balanced"}
                </strong>
              </div>
              <p>{retrainPlan.message || "Use the suggested challenger set for the next retrain."}</p>
              {retrainPlan.memory_applied &&
                Array.isArray(retrainPlan.memory_reordered_models) &&
                retrainPlan.memory_reordered_models.length > 0 && (
                  <div className="message message--accent tiny">
                    Historical winner memory nudged the shortlist order for{" "}
                    {retrainPlan.memory_reordered_models.slice(0, 3).join(", ")}.
                  </div>
                )}
              {Array.isArray(retrainPlan.candidate_models) && retrainPlan.candidate_models.length > 0 && (
                <DataTable
                  compact
                  columns={[
                    { key: "name", label: "Candidate" },
                    { key: "role", label: "Role" },
                  ]}
                  rows={retrainPlan.candidate_models.map((name, index) => ({
                    name,
                    role: index === 0 ? "Lead challenger" : "Compare",
                  }))}
                />
              )}
            </div>
          )}
          {historyRows[0] && (
            <div className="message message--accent tiny" style={{ marginTop: "1rem" }}>
              Latest recorded check: {historyRows[0].status || "unknown"} on {formatDate(historyRows[0].created_at)}
            </div>
          )}
        </Panel>
      </div>

      <div className="grid grid--two">
        <Panel
          title="Recent drift checks"
          subtitle="Operational history now has a proper table instead of only raw JSON."
        >
          <DataTable
            columns={[
              { key: "uploaded_name", label: "Dataset" },
              { key: "status", label: "Status" },
              { key: "drift_score_pct", label: "Drift score", render: (row) => formatNumber(row.drift_score_pct) },
              { key: "created_at", label: "Created", render: (row) => formatDate(row.created_at) },
            ]}
            rows={historyRows}
            empty="No drift checks have been recorded yet."
          />
        </Panel>

        <Panel
          title="Feature event table"
          subtitle="The feature-level timeline remains inspectable when you need the detail behind the chart."
        >
          <DataTable
            columns={[
              { key: "feature", label: "Feature" },
              { key: "psi", label: "PSI", render: (row) => formatNumber(row.psi) },
              { key: "severity", label: "Severity" },
              { key: "drift_detected", label: "Detected", render: (row) => (row.drift_detected ? "Yes" : "No") },
              { key: "created_at", label: "Observed", render: (row) => formatDate(row.created_at) },
            ]}
            rows={timelineRows}
            empty="No feature-level drift events are available yet."
            compact
          />
        </Panel>
      </div>

      <div className="grid grid--two">
        <Panel title="Single Prediction" subtitle="Send a feature object through the selected trained model.">
          <div className="stack">
            <div className="split">
              <span className="tiny-eyebrow">Input JSON</span>
              <button
                className="button button--ghost tiny"
                onClick={() => {
                  const target = datasetProfile?.suggested_target;
                  const template = {};
                  datasetProfile?.columns?.forEach((c) => {
                    if (c !== target) template[c] = 0;
                  });
                  patchForm("predictPayload", template);
                }}
              >
                🪄 Load Template
              </button>
            </div>
            <textarea
              className="code-editor"
              rows={6}
              value={
                typeof forms.predictPayload === "string"
                  ? forms.predictPayload
                  : JSON.stringify(forms.predictPayload, null, 2)
              }
              onChange={(e) => patchForm("predictPayload", e.target.value)}
            />
            <button className="button" onClick={handlePredict}>
              Run Prediction
            </button>
            {predictResult && (
              <div className="message message--success">
                <strong>Prediction Result:</strong> {predictResult.prediction}
                {predictResult.probabilities && (
                  <p className="tiny">
                    {Object.entries(predictResult.probabilities)
                      .map(([label, value]) => `${label}: ${formatNumber(value)}%`)
                      .join(" • ")}
                  </p>
                )}
              </div>
            )}
          </div>
        </Panel>

        <Panel
          title="Future Sweep"
          subtitle="Sweep one feature across candidate values to explore directional behavior."
        >
          <div className="stack">
            <label className="field">
              <span>Feature to Sweep</span>
              <select value={forms.sweepFeature || ""} onChange={(e) => patchForm("sweepFeature", e.target.value)}>
                <option value="">Select a feature...</option>
                {datasetProfile?.columns?.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </label>
            <span className="tiny-eyebrow">Base Feature JSON</span>
            <textarea
              className="code-editor"
              rows={4}
              value={
                typeof forms.baseFeatures === "string"
                  ? forms.baseFeatures
                  : JSON.stringify(forms.baseFeatures, null, 2)
              }
              onChange={(e) => patchForm("baseFeatures", e.target.value)}
            />
            <label className="field">
              <span>Sweep values</span>
              <input
                value={forms.futureValues}
                onChange={(event) => patchForm("futureValues", event.target.value)}
                placeholder="e.g. 1,2,3,4 or 0.1,0.5,0.9"
              />
            </label>
            <button className="button" onClick={handleFutureSweep}>
              Run Sweep
            </button>
            {futureResult?.predictions?.length > 0 && (
              <div className="stack" style={{ marginTop: "1rem" }}>
                <span className="tiny-eyebrow">Sweep Analysis</span>
                <MiniAreaChart
                  points={futureResult.predictions.map((row, index) => ({
                    label: String(row.x ?? index + 1),
                    value: Number(row.prediction) || 0,
                  }))}
                  valueKey="value"
                  labelKey="label"
                />
                <DataTable
                  compact
                  columns={[
                    { key: "x", label: "Sweep Value" },
                    { key: "prediction", label: "Prediction", render: (row) => formatNumber(row.prediction) },
                    {
                      key: "confidence",
                      label: "Confidence",
                      render: (row) =>
                        row.confidence === null || row.confidence === undefined
                          ? "—"
                          : `${formatNumber(row.confidence)}%`,
                    },
                    { key: "error", label: "Error" },
                  ]}
                  rows={futureResult.predictions}
                />
              </div>
            )}
          </div>
        </Panel>
      </div>
    </>
  );
}
