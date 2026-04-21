import React from "react";
import {
  DataTable,
  DetailJson,
  KeyValueList,
  Message,
  MiniAreaChart,
  PageHero,
  Panel,
  StatCard,
  TimelineList,
  Badge,
} from "../components/ui.jsx";
import { formatDate, formatNumber } from "../lib/format.js";
import { navigateTo } from "../config/routes.js";

export function TrainingPage({
  forms,
  patchForm,
  forecast,
  trainMessage,
  handleTrain,
  jobs,
  jobStatus,
  datasetColumns = [],
}) {
  const status = jobStatus?.status || "idle";

  const reasoningItems = (jobStatus?.reasoning || []).slice(-8).map((item, index) => ({
    title: `Reasoning step ${index + 1}`,
    detail: item,
    meta: status,
  }));

  const historyPoints = (jobStatus?.history || []).map((item, index) => ({
    label: item.time || `Step ${index + 1}`,
    value: Number(item.metric) || index + 1,
  }));

  const historyItems = (jobStatus?.history || []).map((item) => ({
    title: `${item.time || "Event"}`,
    detail: `Metric: ${formatNumber(item.metric)}`,
    meta: item.time === "Final" ? "⭐ Final Outcome" : "👉 Step",
  }));

  const results = jobStatus?.results || {};
  const execProfile = results?.execution_profile || {};

  return (
    <>
      <PageHero
        eyebrow="Training Lab"
        title="Live Execution & Neural Monitoring"
        description="Monitor the active run, watch score checkpoints, and move into the results workspace as soon as the winning model lands."
        stats={[
          { label: "Run ID", value: jobStatus?.id?.slice(0, 8) || "None" },
          { label: "Status", value: status.toUpperCase(), detail: "current execution state" },
          { label: "History Events", value: jobStatus?.history?.length || 0, detail: "checkpoints reached" },
          { label: "Reasoning Notes", value: jobStatus?.reasoning?.length || 0, detail: "backend logic steps" },
        ]}
      />

      {status === "training" && (
        <div className="message message--info" style={{ marginBottom: "2rem" }}>
          ⏳ <strong>Training in progress...</strong> The studio is polling the backend every 2s for fresh metrics.
        </div>
      )}

      {status === "completed" && (
        <Panel title="🎉 Training Completed!" subtitle="The AutoML engine has finished. Here is the winning model summary.">
          <div className="grid grid--one" style={{ marginTop: "1rem" }}>
            <StatCard label="Best Model" value={results.best_model || "Unknown"} tone="success" />
            <StatCard label={results.metric_name || "Score"} value={formatNumber(results.score)} tone="accent" />
          </div>
          {execProfile.sweep_size && (
            <p className="tiny" style={{ marginTop: "1rem" }}>
              Mode profile: sweep_size={execProfile.sweep_size}, top_k={execProfile.top_k},
              optuna={execProfile.run_optuna ? "Yes" : "No"}, trials={execProfile.n_trials}
            </p>
          )}
          <div className="stack" style={{ marginTop: "2rem" }}>
            <p>Training is finished. Deep insights, SHAP values, and the prediction playground are now available in the Results Console.</p>
            <div className="split">
              <button
                className="button button--primary"
                style={{ flex: 2, height: "56px", fontSize: "1.1rem" }}
                onClick={() => navigateTo("/results")}
              >
                🔍 View Deep Analysis in Decision Room
              </button>
            </div>
          </div>
        </Panel>
      )}

      {status === "failed" && (
        <div className="message message--warning" style={{ marginBottom: "2rem" }}>
          ❌ <strong>Training failed:</strong> {jobStatus?.error || "Unknown backend error"}
        </div>
      )}

      <div className="grid grid--two">

        <Panel title="Neural Forecast facts" subtitle="Pre-run estimations and training strategy.">
          <div className="stack">
            <KeyValueList
              items={[
                { label: "Expected runtime", value: forecast?.estimated_duration_label || "Calculating..." },
                { label: "Memory Risk", value: forecast?.memory_risk || "Low" },
                { label: "Compute Intensity", value: forecast?.compute_intensity || "Low" },
                { label: "CV Folds", value: forms.trainCvFolds },
              ]}
            />
            
            {status === "training" && (
              <div className="stack" style={{ marginTop: "1rem", padding: "16px", background: "var(--card-bg)", borderRadius: "12px", border: "1px solid var(--border)" }}>
                <div className="split">
                  <span className="tiny">Mission Progress</span>
                  <span className="tiny accent-text">{Math.min(Math.round(((jobStatus?.history?.length || 0) / (forecast?.estimated_model_count || 5)) * 100), 95)}%</span>
                </div>
                <div className="progress-bar-container" style={{ height: "6px", background: "rgba(255,255,255,0.05)", borderRadius: "3px", overflow: "hidden", marginTop: "8px" }}>
                  <div 
                    className="progress-bar-fill" 
                    style={{ 
                      height: "100%", 
                      background: "var(--accent)", 
                      boxShadow: "0 0 15px var(--accent)",
                      width: `${Math.min(Math.round(((jobStatus?.history?.length || 0) / (forecast?.estimated_model_count || 5)) * 100), 95)}%`,
                      transition: "width 1s ease-in-out"
                    }} 
                  />
                </div>
                <p className="tiny" style={{ marginTop: "8px", color: "var(--text-dim)" }}>
                  Executing stage {jobStatus?.history?.length || 0} of {forecast?.estimated_model_count || 5} (Estimated)
                </p>
              </div>
            )}
          </div>
          <div style={{ marginTop: "1rem" }}>
            <MiniAreaChart points={historyPoints} valueKey="value" labelKey="label" />
          </div>
        </Panel>
      </div>

      <div className="grid grid--two">
        <Panel title="📋 Training Log" subtitle="Live score trajectory and model checkpoints.">
          <DataTable
            columns={[
              { key: "time", label: "Checkpoint" },
              { key: "metric", label: "Metric Value", render: (row) => formatNumber(row.metric) },
            ]}
            rows={jobStatus?.history || []}
            compact
          />
          <TimelineList items={historyItems} empty="Warming up the engine..." />
        </Panel>

        <Panel title="🧠 Live Reasoning" subtitle="What the AutoML agent is thinking and doing right now.">
          <TimelineList items={reasoningItems} empty="Reasoning steps appear once a run begins." />
        </Panel>
      </div>  
    </>
  );
}

// Internal helper for Checkbox within TrainingPage
function Checkbox({ label, checked, onChange }) {
  return (
    <label className="checkbox">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span>{label}</span>
    </label>
  );
}
