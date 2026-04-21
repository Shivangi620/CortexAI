import React from "react";
import {
  DataTable,
  Message,
  MiniAreaChart,
  PageHero,
  Panel,
  StatCard,
  Spinner,
  LineageTree,
  NeuralPulse,
} from "../components/ui.jsx";
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
  const timelinePoints = timelineRows.slice(0, 16).map((item, index) => ({
    label: item.feature || item.uploaded_name || `Check ${index + 1}`,
    value: Number(item.psi) || 0,
  }));
  const baselineMessage =
    driftResult?.alert_message ||
    alertSummary?.recommended_action ||
    (predictResult?.confidence_pct
      ? `Latest prediction confidence is ${predictResult.confidence_pct}%. Run drift checks when production batches change materially.`
      : "Run a drift check to compare the latest production slice against the saved baseline.");

  return (
    <>
      <PageHero
        eyebrow="Ops Watch"
        title="Production drift, threshold schedules, and recovery actions"
        description="Monitoring now feels like a real operations surface: upload fresh samples, inspect the drift curve, tune alerting thresholds, and trigger retraining when the model falls behind reality."
        stats={[
          { label: "Drift checks", value: historyRows.length, detail: "recorded comparisons" },
          { label: "Feature events", value: timelineRows.length, detail: "timeline points" },
          { label: "Schedule", value: driftSchedule?.enabled ? "Enabled" : "Disabled", detail: `${driftSchedule?.frequency_days || "—"} day cadence` },
          { label: "Last alert", value: driftSchedule?.last_alert_status || "—", detail: "latest alert state" },
        ]}
      />

      <div className="grid grid--stats">
        <StatCard label="Due now" value={driftSchedule?.due_now ? "Yes" : "No"} detail={driftSchedule?.next_due_at || "no next due date"} tone="warning" />
        <StatCard label="Warning PSI" value={formatNumber(driftSchedule?.warning_threshold)} detail="warning boundary" />
        <StatCard label="Critical PSI" value={formatNumber(driftSchedule?.critical_threshold)} detail="critical boundary" tone="success" />
        <StatCard label="Overall status" value={driftResult?.overall_status || "—"} detail={driftResult?.alert_level || "no fresh report"} />
      </div>

      <div className="grid grid--two">
        <Panel title="Neural Pulse Telemetry" subtitle="Live visualization of the model's stochastic vital signs. Ripple intensity maps to confidence." tone="accent">
          <NeuralPulse confidence={predictResult?.confidence_pct || 88} activity={loading ? 1 : 0.4} />
          <div className="message message--accent tiny" style={{ marginTop: "1rem" }}>
            <strong>Ops Summary:</strong> {baselineMessage}
          </div>
        </Panel>

        <Panel title="Fresh drift check" subtitle="Upload the latest production slice and compare it to the run baseline.">
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
              <strong>{alertSummary.headline || "Latest alert"}:</strong> {alertSummary.recommended_action || alertSummary.message || "Review the latest drift report."}
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
              <input type="checkbox" checked={forms.driftEnabled} onChange={(event) => patchForm("driftEnabled", event.target.checked)} />
              <span>Enable scheduled monitoring</span>
            </label>
            <div className="split">
              <label className="field">
                <span>Frequency (days)</span>
                <input type="number" min="1" value={forms.driftFrequencyDays} onChange={(event) => patchForm("driftFrequencyDays", event.target.value)} />
              </label>
              <label className="field">
                <span>Feature filter</span>
                <input value={forms.driftFeature} onChange={(event) => patchForm("driftFeature", event.target.value)} placeholder="optional feature name" />
              </label>
            </div>
            <div className="split">
              <label className="field">
                <span>Warning threshold</span>
                <input type="number" step="0.01" value={forms.driftWarningThreshold} onChange={(event) => patchForm("driftWarningThreshold", event.target.value)} />
              </label>
              <label className="field">
                <span>Critical threshold</span>
                <input type="number" step="0.01" value={forms.driftCriticalThreshold} onChange={(event) => patchForm("driftCriticalThreshold", event.target.value)} />
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
        <Panel title="Feature drift curve" subtitle="A lightweight chart shows PSI movement over the recorded feature timeline.">
          <MiniAreaChart points={timelinePoints} valueKey="value" labelKey="label" empty="Run at least one drift check to generate a curve." />
          {driftResult?.critical_features?.length > 0 && (
            <div className="message message--danger tiny" style={{ marginTop: "1rem" }}>
              Critical features: {driftResult.critical_features.join(", ")}
            </div>
          )}
        </Panel>

        <Panel title="Retraining response" subtitle="If the drift check becomes actionable, launch retraining directly from this room.">
          <form className="stack" onSubmit={driftRetrain}>
            <label className="field">
              <span>Retraining dataset</span>
              <input ref={retrainUploadRef} type="file" />
            </label>
            <button className="button button--primary" type="submit">
              Retrain From Drift Sample
            </button>
          </form>
          <Message text={retrainMessage} />
          {historyRows[0] && (
            <div className="message message--accent tiny" style={{ marginTop: "1rem" }}>
              Latest recorded check: {historyRows[0].status || "unknown"} on {formatDate(historyRows[0].created_at)}
            </div>
          )}
        </Panel>
      </div>

      <div className="grid grid--two">
        <Panel title="Recent drift checks" subtitle="Operational history now has a proper table instead of only raw JSON.">
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

        <Panel title="Feature event table" subtitle="The feature-level timeline remains inspectable when you need the detail behind the chart.">
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
        <Panel title="Goal Seeker (Counterfactuals)" subtitle="Set a target prediction and find the minimal changes needed to reach it.">
          <div className="stack">
            <label className="field">
              <span>Target Prediction</span>
              <input
                type="number"
                step="0.1"
                value={forms.goalSeekTarget || ""}
                onChange={(e) => patchForm("goalSeekTarget", e.target.value)}
                placeholder="e.g. 0.95"
              />
            </label>
            <button className="button button--accent" onClick={getGoalSeeker}>
              Find Optimal Path
            </button>
            {goalSeeker?.current_prediction !== undefined && (
              <div className="message message--accent tiny">
                Current prediction: {formatNumber(goalSeeker.current_prediction)} • Target: {formatNumber(goalSeeker.target_prediction)}
              </div>
            )}
            {goalSeeker?.suggestions?.length > 0 && (
              <div className="stack box" style={{ marginTop: "1rem" }}>
                <strong>Suggestions:</strong>
                {goalSeeker.suggestions.map((s, i) => (
                  <div key={i} className="message message--success tiny">
                    {(s.change || `Change ${s.feature}`)} → <strong>{s.feature}</strong> → New Pred: {s.new_prediction}
                  </div>
                ))}
              </div>
            )}
            {goalSeeker?.error && <div className="message message--danger">{goalSeeker.error}</div>}
          </div>
        </Panel>

        <Panel
          title="Neural Simulation Sandbox"
          subtitle="Tweak feature values in real-time to see how the model reacts. Impactful changes will trigger a sensitivity glow."
        >
          <div className="stack">
            <div className="grid grid--two">
              {predictResult?.feature_names?.slice(0, 10).map((name) => {
                const sensitivity = predictResult?.sensitivity?.[name] || 0;
                const glowIntensity = Math.min(sensitivity * 5, 1);
                return (
                  <div
                    key={name}
                    className="field sensitivity-glow"
                    style={{ "--glow-intensity": glowIntensity, "--glow-color": sensitivity > 0.1 ? "var(--accent)" : "var(--line)" }}
                  >
                    <span>{name} {sensitivity > 0 ? `(Impact: ${Math.round(sensitivity * 100)}%)` : ""}</span>
                    <input
                      type="text"
                      placeholder="Enter value..."
                      value={forms.predictPayload?.[name] || ""}
                      onChange={(e) => {
                        const newPayload = { ...forms.predictPayload, [name]: e.target.value };
                        patchForm("predictPayload", newPayload);
                        // Pass null or a dummy object if handlePredict expects an event
                        handlePredict(null, newPayload);
                      }}
                      className="neural-pulse"
                    />
                  </div>
                );
              })}
            </div>
            {loading && <Spinner label="Simulating neural response..." />}
            {predictResult && (
              <div className="message message--accent" style={{ marginTop: "1rem" }}>
                <strong>Simulated Prediction: {predictResult.prediction}</strong>
                {predictResult.confidence_pct && <p className="tiny">Confidence: {predictResult.confidence_pct}%</p>}
              </div>
            )}
          </div>
        </Panel>
      </div>

      <div className="grid grid--two">
        <Panel
          title="Single Prediction"
          subtitle="Send a feature object through the selected trained model."
        >
          <div className="stack">
            <div className="split">
              <span className="tiny-eyebrow">Input JSON</span>
              <button 
                className="button button--ghost tiny" 
                onClick={() => {
                  const target = datasetProfile?.suggested_target;
                  const template = {};
                  datasetProfile?.columns?.forEach(c => {
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
              value={typeof forms.predictPayload === "string" ? forms.predictPayload : JSON.stringify(forms.predictPayload, null, 2)}
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
                    {Object.entries(predictResult.probabilities).map(([label, value]) => `${label}: ${formatNumber(value)}%`).join(" • ")}
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
              <select 
                value={forms.sweepFeature || ""} 
                onChange={(e) => patchForm("sweepFeature", e.target.value)}
              >
                <option value="">Select a feature...</option>
                {datasetProfile?.columns?.map(c => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </label>
            <span className="tiny-eyebrow">Base Feature JSON</span>
            <textarea
              className="code-editor"
              rows={4}
              value={typeof forms.baseFeatures === "string" ? forms.baseFeatures : JSON.stringify(forms.baseFeatures, null, 2)}
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
                      render: (row) => row.confidence === null || row.confidence === undefined ? "—" : `${formatNumber(row.confidence)}%`,
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
