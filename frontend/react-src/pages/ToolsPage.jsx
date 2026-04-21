import React, { useMemo, useState } from "react";
import {
  DataTable,
  KeyValueList,
  Message,
  MiniAreaChart,
  NeuralPulse,
  PageHero,
  Panel,
  StatCard,
} from "../components/ui.jsx";
import { formatNumber } from "../lib/format.js";

const LAB_TABS = [
  { id: "telemetry", label: "Neural Pulse Telemetry" },
  { id: "goal-seeker", label: "Goal Seeker" },
  { id: "single-prediction", label: "Single Prediction" },
  { id: "future-sweep", label: "Future Sweep" },
];

export function ToolsPage({
  forms,
  patchForm,
  contractCheck,
  quicktrain,
  syntheticExpand,
  syntheticJudge,
  zeroShot,
  metaInsights,
  parseIntent,
  chat,
  batchPredict,
  uploadRefs,
  messages,
  outputs,
  metaStatus,
  datasetColumns = [],
  trainedFeatureNames = [],
  selectedJobId,
  loading,
  predictResult,
  predictMessage,
  handlePredict,
  futureResult,
  futureMessage,
  handleFutureSweep,
  goalSeeker,
  goalSeekerMessage,
  getGoalSeeker,
  isClassification,
}) {
  const [activeLabTab, setActiveLabTab] = useState("telemetry");
  const availablePredictionFeatures = useMemo(
    () => (trainedFeatureNames.length ? trainedFeatureNames : datasetColumns),
    [trainedFeatureNames, datasetColumns],
  );

  const templatePayload = useMemo(() => {
    const template = {};
    availablePredictionFeatures.forEach((feature) => {
      template[feature] = 0;
    });
    return template;
  }, [availablePredictionFeatures]);

  const telemetryPoints = (futureResult?.predictions || [])
    .filter((row) => row && row.prediction !== undefined && row.prediction !== null)
    .map((row, index) => ({
      label: String(row.x ?? index + 1),
      value: Number(row.prediction) || 0,
    }));

  function loadPredictionTemplate(targetKey = "predictPayload") {
    patchForm(targetKey, templatePayload);
  }

  function loadFutureTemplate() {
    patchForm({
      baseFeatures: templatePayload,
      sweepFeature: availablePredictionFeatures[0] || "",
    });
  }

  function renderLabTab() {
    if (!selectedJobId) {
      return (
        <Panel
          title="Advanced inference tools"
          subtitle="Select a completed run to activate telemetry, counterfactuals, live prediction, and sweep analysis."
        >
          <Message text="No completed run is selected yet." />
        </Panel>
      );
    }

    if (activeLabTab === "telemetry") {
      const confidence = predictResult?.confidence_pct || 0;
      const opsSummary = predictResult
        ? `Latest prediction confidence is ${predictResult.confidence_pct || "—"}%. Use this panel after each live inference to keep a pulse on certainty.`
        : "Run a single prediction to light up the telemetry pulse and confidence summary.";
      return (
        <div className="grid grid--two">
          <Panel
            title="Neural Pulse Telemetry"
            subtitle="A live visualization of confidence and activity for the selected model run."
            tone="accent"
          >
            <NeuralPulse confidence={confidence || 88} activity={loading ? 1 : predictResult ? 0.72 : 0.28} />
            <div className="message message--accent tiny" style={{ marginTop: "1rem" }}>
              <strong>Ops Summary:</strong> {opsSummary}
            </div>
            <Message text={predictMessage} />
          </Panel>

          <Panel
            title="Live signal readout"
            subtitle="The latest prediction payload is summarized here so the telemetry view has concrete context."
          >
            <KeyValueList
              items={[
                { label: "Prediction", value: predictResult?.prediction ?? "—" },
                { label: "Confidence", value: predictResult?.confidence_pct ? `${predictResult.confidence_pct}%` : "—" },
                { label: "Tracked features", value: predictResult?.feature_names?.length || availablePredictionFeatures.length || "—" },
              ]}
            />
            {predictResult?.probabilities && (
              <div style={{ marginTop: "1rem" }}>
                <DataTable
                  compact
                  columns={[
                    { key: "label", label: "Label" },
                    { key: "probability", label: "Probability", render: (row) => `${formatNumber(row.probability)}%` },
                  ]}
                  rows={Object.entries(predictResult.probabilities).map(([label, probability]) => ({ label, probability }))}
                />
              </div>
            )}
          </Panel>
        </div>
      );
    }

    if (activeLabTab === "goal-seeker") {
      return (
        <div className="grid grid--two">
          <Panel
            title="Goal Seeker (Counterfactuals)"
            subtitle="Use the current payload and ask the model for the smallest changes that move toward a different outcome."
          >
            <div className="stack">
              <div className="split">
                <span className="tiny-eyebrow">Working payload</span>
                <button className="button button--ghost tiny" type="button" onClick={() => loadPredictionTemplate("predictPayload")}>
                  Load Template
                </button>
              </div>
              <textarea
                className="code-editor"
                rows={7}
                value={typeof forms.predictPayload === "string" ? forms.predictPayload : JSON.stringify(forms.predictPayload, null, 2)}
                onChange={(event) => patchForm("predictPayload", event.target.value)}
              />
              <label className="field">
                <span>{isClassification ? "Target confidence / score" : "Target prediction"}</span>
                <input
                  type="number"
                  step="0.1"
                  value={forms.goalSeekTarget || ""}
                  onChange={(event) => patchForm("goalSeekTarget", event.target.value)}
                  placeholder={isClassification ? "e.g. 95" : "e.g. 250.5"}
                />
              </label>
              <button className="button button--accent" type="button" onClick={getGoalSeeker}>
                Find Optimal Path
              </button>
              <Message text={goalSeekerMessage} />
            </div>
          </Panel>

          <Panel
            title="Counterfactual results"
            subtitle="Suggestions are ranked by the smallest one-step change that improves the outcome."
          >
            <KeyValueList
              items={[
                { label: "Current output", value: goalSeeker?.current_prediction ?? goalSeeker?.prediction ?? "—" },
                { label: "Target", value: (goalSeeker?.target_prediction ?? forms.goalSeekTarget) || "—" },
                { label: "Confidence", value: goalSeeker?.confidence_pct ? `${goalSeeker.confidence_pct}%` : "—" },
              ]}
            />
            {goalSeeker?.suggestions?.length > 0 ? (
              <div style={{ marginTop: "1rem" }}>
                <DataTable
                  compact
                  columns={[
                    { key: "feature", label: "Feature" },
                    { key: "from", label: "From" },
                    { key: "to", label: "To" },
                    { key: "new_prediction", label: "New output" },
                  ]}
                  rows={goalSeeker.suggestions}
                />
              </div>
            ) : (
              <Message text={goalSeeker?.error || goalSeeker?.message || "No counterfactual suggestions yet."} />
            )}
          </Panel>
        </div>
      );
    }

    if (activeLabTab === "single-prediction") {
      return (
        <div className="grid grid--two">
          <Panel
            title="Single Prediction"
            subtitle="Send one feature object through the selected trained model."
          >
            <div className="stack">
              <div className="split">
                <span className="tiny-eyebrow">Input JSON</span>
                <button className="button button--ghost tiny" type="button" onClick={() => loadPredictionTemplate("predictPayload")}>
                  Load Template
                </button>
              </div>
              <textarea
                className="code-editor"
                rows={8}
                value={typeof forms.predictPayload === "string" ? forms.predictPayload : JSON.stringify(forms.predictPayload, null, 2)}
                onChange={(event) => patchForm("predictPayload", event.target.value)}
              />
              <button className="button button--primary" type="button" onClick={handlePredict}>
                Run Prediction
              </button>
              <Message text={predictMessage} />
            </div>
          </Panel>

          <Panel
            title="Prediction result"
            subtitle="This panel shows the direct model output plus class probabilities when available."
          >
            <KeyValueList
              items={[
                { label: "Prediction", value: predictResult?.prediction ?? "—" },
                { label: "Confidence", value: predictResult?.confidence_pct ? `${predictResult.confidence_pct}%` : "—" },
                { label: "Features in payload", value: Object.keys(typeof forms.predictPayload === "object" && forms.predictPayload ? forms.predictPayload : {}).length || "—" },
              ]}
            />
            {predictResult?.probabilities && (
              <div style={{ marginTop: "1rem" }}>
                <DataTable
                  compact
                  columns={[
                    { key: "label", label: "Label" },
                    { key: "probability", label: "Probability", render: (row) => `${formatNumber(row.probability)}%` },
                  ]}
                  rows={Object.entries(predictResult.probabilities).map(([label, probability]) => ({ label, probability }))}
                />
              </div>
            )}
          </Panel>
        </div>
      );
    }

    return (
      <div className="grid grid--two">
        <Panel
          title="Future Sweep"
          subtitle="Sweep one feature across candidate values and watch the model response curve."
        >
          <div className="stack">
            <div className="split">
              <span className="tiny-eyebrow">Base payload</span>
              <button className="button button--ghost tiny" type="button" onClick={loadFutureTemplate}>
                Load Template
              </button>
            </div>
            <label className="field">
              <span>Feature to sweep</span>
              <select value={forms.sweepFeature || ""} onChange={(event) => patchForm("sweepFeature", event.target.value)}>
                <option value="">Select a feature...</option>
                {availablePredictionFeatures.map((feature) => (
                  <option key={feature} value={feature}>
                    {feature}
                  </option>
                ))}
              </select>
            </label>
            <textarea
              className="code-editor"
              rows={6}
              value={typeof forms.baseFeatures === "string" ? forms.baseFeatures : JSON.stringify(forms.baseFeatures, null, 2)}
              onChange={(event) => patchForm("baseFeatures", event.target.value)}
            />
            <label className="field">
              <span>Sweep values</span>
              <input
                value={forms.futureValues}
                onChange={(event) => patchForm("futureValues", event.target.value)}
                placeholder="e.g. 1,2,3,4 or 0.1,0.5,0.9"
              />
            </label>
            <button className="button button--primary" type="button" onClick={handleFutureSweep}>
              Run Sweep
            </button>
            <Message text={futureMessage} />
          </div>
        </Panel>

        <Panel
          title="Sweep analysis"
          subtitle="Visualize the response curve and inspect each swept value."
        >
          <MiniAreaChart
            points={telemetryPoints}
            valueKey="value"
            labelKey="label"
            empty="Run a future sweep to generate the response curve."
          />
          {futureResult?.predictions?.length > 0 && (
            <div style={{ marginTop: "1rem" }}>
              <DataTable
                compact
                columns={[
                  { key: "x", label: "Sweep value" },
                  { key: "prediction", label: "Prediction", render: (row) => formatNumber(row.prediction) },
                  { key: "confidence", label: "Confidence", render: (row) => (row.confidence === null || row.confidence === undefined ? "—" : `${formatNumber(row.confidence)}%`) },
                  { key: "error", label: "Error" },
                ]}
                rows={futureResult.predictions}
              />
            </div>
          )}
        </Panel>
      </div>
    );
  }

  return (
    <>
      <PageHero
        eyebrow="Advanced Lab"
        title="Inference, synthetic data, quick experimentation, and AI assistance"
        description="Advanced features now live inside a single lab. The inference workbench is split into focused tabs so each job has a clear place and the power-user flow stays organized."
        stats={[
          { label: "Meta backend", value: metaStatus?.backend || "heuristics", detail: metaStatus?.is_trained ? "trained" : "fallback" },
          { label: "Inference tabs", value: LAB_TABS.length, detail: "telemetry, counterfactuals, prediction, sweep" },
          { label: "Selected run", value: selectedJobId ? "Ready" : "Missing", detail: selectedJobId ? "tools are active" : "pick a completed run" },
        ]}
      />

      <div className="grid grid--stats">
        <StatCard label="Meta engine" value={metaStatus?.backend || "heuristics"} detail={metaStatus?.is_trained ? "trained engine" : "fallback engine"} tone="warning" />
        <StatCard label="Zero-shot" value={outputs.zeroshot ? "Ready" : "Pending"} detail="model family recommendation" />
        <StatCard label="Synthetic ops" value={outputs.synthetic ? "Ready" : "Pending"} detail="augmentation and judging" tone="success" />
        <StatCard label="Contract checks" value={outputs.contractCheck ? "Ready" : "Pending"} detail="schema verification" />
      </div>

      <Panel
        title="Advanced inference lab"
        subtitle="Each inference workflow now has its own tab under Advanced Lab."
      >
        <div className="tab-strip" role="tablist" aria-label="Advanced inference tabs">
          {LAB_TABS.map((tab) => (
            <button
              key={tab.id}
              className={`tab-strip__button ${activeLabTab === tab.id ? "tab-strip__button--active" : ""}`}
              type="button"
              role="tab"
              aria-selected={activeLabTab === tab.id}
              onClick={() => setActiveLabTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>
        <div style={{ marginTop: "1.25rem" }}>{renderLabTab()}</div>
      </Panel>

      <div className="grid grid--two">
        <Panel title="Inference Contract" subtitle="Schema validation for production readiness.">
          <form className="stack" onSubmit={contractCheck}>
            <label className="field">
              <span>Inference contract file</span>
              <input ref={uploadRefs.contractUploadRef} type="file" />
            </label>
            <button className="button button--secondary" type="submit">
              Validate Contract
            </button>
          </form>
          <Message text={messages.contractCheck} />
          {outputs.contractCheck && (
            <div className="stack" style={{ marginTop: "1rem" }}>
              <span className="tiny-eyebrow">Contract Status: {outputs.contractCheck.status}</span>
              <DataTable
                compact
                columns={[
                  { key: "type", label: "Issue Type" },
                  { key: "count", label: "Count" },
                  { key: "details", label: "Details" },
                ]}
                rows={[
                  { type: "Missing Features", count: outputs.contractCheck.missing_features?.length || 0, details: outputs.contractCheck.missing_features?.join(", ") || "None" },
                  { type: "Extra Columns", count: outputs.contractCheck.extra_columns?.length || 0, details: outputs.contractCheck.extra_columns?.join(", ") || "None" },
                  { type: "Type Mismatches", count: outputs.contractCheck.dtype_mismatches?.length || 0, details: outputs.contractCheck.dtype_mismatches?.map((m) => m.column).join(", ") || "None" },
                ]}
              />
            </div>
          )}
        </Panel>

        <Panel title="Batch Prediction" subtitle="Upload a CSV and get predictions for every row.">
          <form className="stack" onSubmit={batchPredict}>
            <label className="field">
              <span>Batch input (CSV)</span>
              <input ref={uploadRefs.batchUploadRef} type="file" />
            </label>
            <button className="button button--primary" type="submit">
              Run Batch Inference
            </button>
          </form>
          <Message text={messages.batchPredict} />
          {outputs.batchPredict?.preview && (
            <div style={{ marginTop: "1rem" }}>
              <DataTable
                columns={Object.keys(outputs.batchPredict.preview[0] || {}).map((key) => ({ key, label: key }))}
                rows={outputs.batchPredict.preview}
                compact
              />
            </div>
          )}
        </Panel>

        <Panel title="Synthetic and meta-learning tools" subtitle="Augmentation, judging, and model-prior discovery tools stay in the same lab.">
          <div className="inline-actions">
            <button className="button button--secondary" type="button" onClick={syntheticExpand}>
              Expand Dataset
            </button>
            <button className="button button--secondary" type="button" onClick={syntheticJudge}>
              Judge Synthetic Quality
            </button>
          </div>
          <Message text={messages.synthetic} />
          {outputs.synthetic && (
            <div className="stack" style={{ marginTop: "1rem" }}>
              <span className="tiny-eyebrow">Synthetic Audit</span>
              <DataTable
                compact
                columns={[
                  { key: "metric", label: "Quality Metric" },
                  { key: "value", label: "Score" },
                ]}
                rows={[
                  { metric: "Realism Score", value: formatNumber(outputs.synthetic.realism_score) },
                  { metric: "Verdict", value: outputs.synthetic.verdict },
                  { metric: "Rows Evaluated", value: outputs.synthetic.rows_evaluated },
                ]}
              />
            </div>
          )}

          <div className="inline-actions">
            <button className="button button--secondary" type="button" onClick={zeroShot}>
              Zero-Shot Recommendation
            </button>
            <button className="button button--secondary" type="button" onClick={metaInsights}>
              Cross-Dataset Insights
            </button>
          </div>
          <Message text={messages.zeroshot || messages.metaInsights} />
        </Panel>
      </div>

      <div className="grid grid--two">
        <Panel title="Natural-language intent" subtitle="Describe what you want to do and let the backend parse it into ML intent.">
          <form className="stack" onSubmit={parseIntent}>
            <label className="field">
              <span>Prompt</span>
              <textarea rows="6" value={forms.nlPrompt} onChange={(event) => patchForm("nlPrompt", event.target.value)} placeholder="Train a balanced churn classifier and prioritize recall." />
            </label>
            <button className="button button--primary" type="submit">
              Parse Intent
            </button>
          </form>
          <Message text={messages.intent} />
        </Panel>

        <Panel title="Chat with run context" subtitle="Ask questions about the currently selected run without leaving the studio.">
          <form className="stack" onSubmit={chat}>
            <label className="field">
              <span>Question</span>
              <textarea rows="8" value={forms.chatPrompt} onChange={(event) => patchForm("chatPrompt", event.target.value)} placeholder="Why did this run choose this model?" />
            </label>
            <button className="button button--primary" type="submit">
              Ask Studio Assistant
            </button>
          </form>
          <Message text={messages.chat} />
          {outputs.chat && (
            <div className="message message--accent" style={{ marginTop: "1rem" }}>
              <strong>Assistant Response:</strong>
              <p>{outputs.chat.response || outputs.chat.answer || "No response text was returned."}</p>
            </div>
          )}
        </Panel>

        <Panel title="Quicktrain sandbox" subtitle="Run a compact experimental fit without leaving Advanced Lab.">
          <form className="stack" onSubmit={quicktrain}>
            <label className="field">
              <span>Target column</span>
              <input value={forms.quicktrainTarget} onChange={(event) => patchForm("quicktrainTarget", event.target.value)} placeholder="Target column" />
            </label>
            <label className="field">
              <span>Models</span>
              <input value={forms.quicktrainModels} onChange={(event) => patchForm("quicktrainModels", event.target.value)} />
            </label>
            <button className="button button--secondary" type="submit">
              Run Quicktrain
            </button>
          </form>
          <Message text={messages.quicktrain} />
        </Panel>
      </div>
    </>
  );
}
