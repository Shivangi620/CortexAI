import React, { useEffect, useMemo, useState } from "react";
import {
  BeforeAfterChart,
  Checkbox,
  DataTable,
  DetailJson,
  InsightSummary,
  KeyValueList,
  Message,
  MiniAreaChart,
  PageHero,
  Panel,
  StatCard,
} from "../components/ui.jsx";
import { formatNumber } from "../lib/format.js";

const LAB_TABS = [
  { id: "scenario-simulator", label: "Scenario Simulator" },
  { id: "goal-seeker", label: "Goal Seeker" },
];

const ENSEMBLE_STRATEGY_COPY = {
  bagging:
    "Bagging reweights the completed runs across bootstrap slices of the reference dataset and then averages them.",
  boosting:
    "Boosting stages the completed runs in score order and emphasizes the rows that earlier runs handled poorly.",
  voting: "Weighted voting blends the completed runs using their observed training scores as the base weights.",
  stacking: "Stacking trains a lightweight meta-model on top of the base-run outputs from the reference dataset.",
};

function toFiniteMetric(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function recommendEnsembleStrategy(runs) {
  if (!runs?.length) {
    return {
      strategy: "voting",
      label: "Weighted Voting",
      reason: "Pick at least two completed runs to generate a recommendation.",
      confidence: "low",
    };
  }

  const scores = runs.map((run) => toFiniteMetric(run.score)).filter((value) => value !== null);

  const sortedScores = [...scores].sort((left, right) => right - left);
  const count = runs.length;
  const scoreSpread = sortedScores.length > 1 ? sortedScores[0] - sortedScores[sortedScores.length - 1] : 0;
  const topLead = sortedScores.length > 1 ? sortedScores[0] - sortedScores[1] : 0;

  if (count >= 4 && sortedScores.length >= 3) {
    return {
      strategy: "stacking",
      label: "Stacking",
      reason: "You have a broad enough run set to let a meta-model learn when each base run is strongest.",
      confidence: "high",
    };
  }

  if (topLead >= 5 && count >= 3) {
    return {
      strategy: "boosting",
      label: "Boosting",
      reason:
        "One run is clearly leading, so staged weighting is a good way to lean on the strongest model while still borrowing signal from the others.",
      confidence: "medium",
    };
  }

  if (scoreSpread <= 3 && count >= 3) {
    return {
      strategy: "bagging",
      label: "Bagging",
      reason:
        "The selected runs are clustered closely, which usually makes a bootstrap-style average the steadiest first choice.",
      confidence: "medium",
    };
  }

  return {
    strategy: "voting",
    label: "Weighted Voting",
    reason:
      "The selected runs are reasonably mixed, so a score-weighted blend is the best conservative starting point.",
    confidence: "medium",
  };
}

function buildScenarioDefinitions(focusFeatures, scenarioValues, baselineValues) {
  return [
    {
      name: "Upside Case",
      description: "Optimistic what-if profile",
      adjustments: {
        delta: Object.fromEntries(
          focusFeatures.filter(Boolean).map((feature) => {
            const value = Number(scenarioValues.upside?.[feature] ?? 0);
            const baseline = Number(baselineValues?.[feature] ?? 0);
            return [feature, roundNumeric(value - baseline, 4)];
          }),
        ),
      },
    },
    {
      name: "Downside Case",
      description: "Stress-test scenario",
      adjustments: {
        delta: Object.fromEntries(
          focusFeatures.filter(Boolean).map((feature) => {
            const value = Number(scenarioValues.downside?.[feature] ?? 0);
            const baseline = Number(baselineValues?.[feature] ?? 0);
            return [feature, roundNumeric(value - baseline, 4)];
          }),
        ),
      },
    },
  ];
}

function roundNumeric(value, digits = 2) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? Number(numeric.toFixed(digits)) : 0;
}

function formatPercent(value, digits = 1) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${numeric.toFixed(digits).replace(/\.0$/, "")}%` : "—";
}

function parseScenarioFilters(value) {
  if (Array.isArray(value)) return value;
  if (!value) return [];
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function ToolsPage({
  forms,
  patchForm,
  contractCheck,
  quicktrain,
  syntheticExpand,
  syntheticJudge,
  trainSyntheticDataset,
  zeroShot,
  metaInsights,
  parseIntent,
  chat,
  batchPredict,
  uploadRefs,
  messages,
  outputs,
  metaStatus,
  selectedDatasetId,
  datasetProfile,
  datasetColumns = [],
  trainedFeatureNames = [],
  selectedJobId,
  futureResult,
  futureMessage,
  handleFutureSweep,
  scenarioContext,
  loadScenarioContext,
  runScenarioSimulator,
  scenarioResult,
  scenarioMessage,
  scenarioPacks = [],
  scenarioPacksMessage,
  saveScenarioPack,
  applyScenarioPack,
  applySuggestedScenarios,
  approveScenarioReviews,
  goalSeeker,
  goalSeekerMessage,
  getGoalSeeker,
  buildEnsemble,
  ensembleResult,
  ensembleMessage,
  jobs = [],
  isClassification,
}) {
  const [activeLabTab, setActiveLabTab] = useState("scenario-simulator");
  const [referenceId, setReferenceId] = useState("median-profile");
  const [focusFeatures, setFocusFeatures] = useState([]);
  const [scenarioValues, setScenarioValues] = useState({ upside: {}, downside: {} });

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

  const scenarioChartPoints = (scenarioResult?.scenarios || [])
    .filter((row) => row && row.prediction !== undefined && row.prediction !== null)
    .map((row) => ({
      label: row.name,
      value: Number(row.confidence_pct ?? row.prediction) || 0,
    }));

  const completedJobs = useMemo(() => jobs.filter((job) => job.status === "completed"), [jobs]);
  const featureRanges = scenarioContext?.feature_ranges || [];
  const numericRanges = featureRanges.filter((item) => item.kind === "numeric");
  const scenarioFilters = useMemo(() => parseScenarioFilters(forms.scenarioFilters), [forms.scenarioFilters]);
  const selectedReferenceProfile =
    (scenarioContext?.sample_rows || []).find((item) => item.id === referenceId) ||
    scenarioContext?.sample_rows?.[0] ||
    null;
  const baselineValues = selectedReferenceProfile?.values || scenarioContext?.default_payload || {};
  const selectedRun = useMemo(() => jobs.find((job) => job.id === selectedJobId) || null, [jobs, selectedJobId]);
  const syntheticRecommendedRows = useMemo(() => {
    const rows = Number(datasetProfile?.rows || 0);
    if (!rows) return null;
    if (rows < 100) return Math.max(100, rows * 10);
    if (rows < 500) return rows * 4;
    if (rows < 1000) return rows * 2;
    return Math.floor(rows / 2);
  }, [datasetProfile]);
  const syntheticRequestedRows = String(forms.syntheticRowCount || "").trim();
  const syntheticPlannedRows = syntheticRequestedRows || (syntheticRecommendedRows ?? "Auto");
  const selectedEnsembleIds = useMemo(
    () =>
      Array.from(
        new Set(
          String(forms.ensembleJobIds || "")
            .split(",")
            .map((item) => item.trim())
            .filter(Boolean)
            .concat(selectedJobId ? [selectedJobId] : []),
        ),
      ),
    [forms.ensembleJobIds, selectedJobId],
  );
  const selectedEnsembleRuns = useMemo(
    () => selectedEnsembleIds.map((jobId) => jobs.find((job) => job.id === jobId)).filter(Boolean),
    [jobs, selectedEnsembleIds],
  );
  const compatibleCompletedJobs = useMemo(
    () =>
      completedJobs.filter((job) => {
        if (job.id === selectedJobId) return false;
        if (
          selectedRun &&
          typeof selectedRun.is_classification === "boolean" &&
          typeof job.is_classification === "boolean"
        ) {
          if (job.is_classification !== selectedRun.is_classification) return false;
        }
        if (selectedRun?.target && job.target && job.target !== selectedRun.target) return false;
        return true;
      }),
    [completedJobs, selectedJobId, selectedRun],
  );
  const ensembleRecommendation = useMemo(() => recommendEnsembleStrategy(selectedEnsembleRuns), [selectedEnsembleRuns]);
  const ensembleScoreSpread = useMemo(() => {
    const numericScores = selectedEnsembleRuns
      .map((job) => toFiniteMetric(job.score))
      .filter((value) => value !== null);
    if (numericScores.length < 2) return null;
    return Math.max(...numericScores) - Math.min(...numericScores);
  }, [selectedEnsembleRuns]);
  const sweepChartPoints = (scenarioResult?.sweep_predictions || [])
    .filter((row) => row && row.prediction !== undefined && row.prediction !== null)
    .map((row) => ({
      label: String(row.x ?? "—"),
      value: Number(row.prediction) || 0,
    }));
  const scenarioComparisonSummary = useMemo(() => {
    const rows = scenarioResult?.scenarios || [];
    const executed = rows.filter((item) => item.executed);
    const positive = executed.filter((item) => Number(item.delta) > 0);
    const strongest = executed
      .filter((item) => item.delta !== null && item.delta !== undefined)
      .sort((left, right) => Number(right.delta || 0) - Number(left.delta || 0))[0];
    return {
      executed: executed.length,
      positive: positive.length,
      strongest,
    };
  }, [scenarioResult]);
  const syntheticComparisonChartItems = useMemo(() => {
    const syntheticOutput = outputs.synthetic || {};
    const originalProfile = syntheticOutput.original_profile || {};
    const generatedRows = Number(syntheticOutput.synthetic_rows_added);
    const originalRows = Number(syntheticOutput.original_rows ?? originalProfile.rows);
    const totalRows = Number(syntheticOutput.total_rows ?? syntheticOutput.profile?.rows);
    const originalMissing = Number(originalProfile.missing_pct);
    const currentMissing = Number(syntheticOutput.profile?.missing_pct);
    const realismScore = Number(syntheticOutput.realism_score);

    return [
      {
        label: "Row count",
        before: Number.isFinite(originalRows) ? originalRows : null,
        after: Number.isFinite(totalRows) ? totalRows : null,
        beforeLabel: Number.isFinite(originalRows) ? formatNumber(originalRows, 0) : "—",
        afterLabel: Number.isFinite(totalRows) ? formatNumber(totalRows, 0) : "—",
        detail: Number.isFinite(generatedRows) ? `+${formatNumber(generatedRows, 0)} synthetic rows` : "Expanded dataset",
      },
      {
        label: "Missingness",
        before: Number.isFinite(originalMissing) ? originalMissing : null,
        after: Number.isFinite(currentMissing) ? currentMissing : null,
        beforeLabel: formatPercent(originalMissing),
        afterLabel: formatPercent(currentMissing),
        detail: "Overall missing-value share",
        max: Math.max(
          Number.isFinite(originalMissing) ? originalMissing : 0,
          Number.isFinite(currentMissing) ? currentMissing : 0,
          1,
        ),
      },
      {
        label: "Realism score",
        before: 100,
        after: Number.isFinite(realismScore) ? realismScore : null,
        beforeLabel: "100 baseline",
        afterLabel: Number.isFinite(realismScore) ? formatNumber(realismScore, 1) : "—",
        detail: syntheticOutput.verdict || "Synthetic quality judge",
        max: 100,
      },
    ];
  }, [outputs.synthetic]);

  useEffect(() => {
    if (!scenarioContext || focusFeatures.length) return;
    const nextFeatures = numericRanges.slice(0, 2).map((item) => item.feature);
    setReferenceId(scenarioContext.sample_rows?.[0]?.id || "median-profile");
    setFocusFeatures(nextFeatures);
    const seeded = { upside: {}, downside: {} };
    numericRanges.slice(0, 2).forEach((item) => {
      seeded.upside[item.feature] = item.median;
      seeded.downside[item.feature] = item.median;
    });
    setScenarioValues(seeded);
  }, [scenarioContext, numericRanges, focusFeatures.length]);

  useEffect(() => {
    setReferenceId("median-profile");
    setFocusFeatures([]);
    setScenarioValues({ upside: {}, downside: {} });
  }, [selectedJobId]);

  useEffect(() => {
    if (activeLabTab !== "scenario-simulator" || !selectedJobId || !scenarioContext || !focusFeatures.length) return;
    const timer = window.setTimeout(() => {
      const scenarioMode = forms.scenarioMode || "payload";
      const isRowMode = scenarioMode === "row" && referenceId.startsWith("row-");
      const rowIndex = isRowMode ? Number(referenceId.replace("row-", "")) : 0;
      const simulatorPayload = {
        base_mode: scenarioMode,
        row_index: rowIndex,
        base_payload: baselineValues,
        filters: scenarioMode === "cohort" ? scenarioFilters : [],
        scenarios: buildScenarioDefinitions(focusFeatures, scenarioValues, baselineValues),
        sweep_feature: focusFeatures[0] || "",
        sweep_values: (() => {
          const selected = numericRanges.find((item) => item.feature === focusFeatures[0]);
          if (!selected) return [];
          return [selected.min, selected.median, selected.max].map((value) => roundNumeric(value, 4));
        })(),
      };
      patchForm({
        scenarioMode,
        scenarioRowIndex: rowIndex,
        scenarioSweepFeature: focusFeatures[0] || "",
        scenarioSweepValues: (() => {
          const selected = numericRanges.find((item) => item.feature === focusFeatures[0]);
          if (!selected) return forms.scenarioSweepValues;
          const values = [selected.min, selected.median, selected.max].map((value) => roundNumeric(value, 4));
          return values.join(",");
        })(),
        scenarioDefinitions: JSON.stringify(
          buildScenarioDefinitions(focusFeatures, scenarioValues, baselineValues),
          null,
          2,
        ),
        predictPayload: JSON.stringify(baselineValues, null, 2),
      });
      runScenarioSimulator(simulatorPayload);
    }, 240);
    return () => window.clearTimeout(timer);
  }, [
    activeLabTab,
    selectedJobId,
    scenarioContext,
    focusFeatures,
    scenarioValues,
    referenceId,
    baselineValues,
    numericRanges,
    forms.scenarioMode,
    forms.scenarioSweepValues,
    scenarioFilters,
  ]);

  function loadPredictionTemplate(targetKey = "predictPayload") {
    patchForm(targetKey, templatePayload);
  }

  function loadFutureTemplate() {
    patchForm({
      baseFeatures: templatePayload,
      sweepFeature: availablePredictionFeatures[0] || "",
    });
  }

  function toggleEnsembleSelection(jobId, checked) {
    const selected = new Set(
      String(forms.ensembleJobIds || "")
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean),
    );
    if (checked) selected.add(jobId);
    else selected.delete(jobId);
    patchForm("ensembleJobIds", Array.from(selected).join(","));
  }

  function runAutoSuggestedScenarios() {
    const suggestions = scenarioResult?.auto_suggestions || [];
    if (!suggestions.length) return;
    const scenarioMode = forms.scenarioMode || "payload";
    const isRowMode = scenarioMode === "row" && referenceId.startsWith("row-");
    const rowIndex = isRowMode ? Number(referenceId.replace("row-", "")) : 0;
    const scenarios = suggestions.map((item) => item.scenario);
    patchForm("scenarioDefinitions", JSON.stringify(scenarios, null, 2));
    if (applySuggestedScenarios) applySuggestedScenarios();
    runScenarioSimulator({
      base_mode: scenarioMode,
      row_index: rowIndex,
      base_payload: baselineValues,
      filters: scenarioMode === "cohort" ? scenarioFilters : [],
      scenarios,
      sweep_feature: forms.scenarioSweepFeature,
      sweep_values: forms.scenarioSweepValues
        ? String(forms.scenarioSweepValues)
            .split(",")
            .map((item) => item.trim())
            .filter(Boolean)
        : [],
    });
  }

  function renderScenarioLab() {
    const scenarioMode = forms.scenarioMode || "payload";
    return (
      <div className="grid grid--two">
        <Panel
          title="Scenario-Based Model Simulator"
          subtitle="Move the sliders, compare upside/downside cases, and keep the selected row or median profile as the live baseline."
        >
          <div className="stack">
            <div className="split">
              <span className="tiny-eyebrow">Simulator context</span>
              <div className="inline-actions">
                <button className="button button--ghost tiny" type="button" onClick={loadScenarioContext}>
                  Refresh Context
                </button>
                <button
                  className="button button--secondary tiny"
                  type="button"
                  onClick={runAutoSuggestedScenarios}
                  disabled={!scenarioResult?.auto_suggestions?.length}
                >
                  Use Best Uplift / Least Risk
                </button>
              </div>
            </div>
            <label className="field">
              <span>Baseline mode</span>
              <select value={scenarioMode} onChange={(event) => patchForm("scenarioMode", event.target.value)}>
                <option value="payload">Payload baseline</option>
                <option value="row">Reference row</option>
                <option value="cohort">Cohort median / mode</option>
              </select>
            </label>
            <label className="field">
              <span>{scenarioMode === "cohort" ? "Reference seed preview" : "Reference row or cohort seed"}</span>
              <select value={referenceId} onChange={(event) => setReferenceId(event.target.value)}>
                {(scenarioContext?.sample_rows || []).map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
            {selectedReferenceProfile ? (
              <div className="message">
                <strong>{selectedReferenceProfile.label}</strong>
                <p>{selectedReferenceProfile.preview}</p>
              </div>
            ) : null}
            {scenarioMode === "cohort" ? (
              <div className="scenario-cohort-shell">
                <div className="split">
                  <span className="tiny-eyebrow">Cohort filters</span>
                  <button
                    className="button button--ghost tiny"
                    type="button"
                    onClick={() =>
                      patchForm(
                        "scenarioFilters",
                        JSON.stringify(
                          focusFeatures.filter(Boolean).map((feature) => ({
                            column: feature,
                            operator: "gte",
                            value: baselineValues?.[feature] ?? 0,
                          })),
                          null,
                          2,
                        ),
                      )
                    }
                  >
                    Seed From Drivers
                  </button>
                </div>
                <textarea
                  className="code-editor"
                  rows={5}
                  value={
                    typeof forms.scenarioFilters === "string"
                      ? forms.scenarioFilters
                      : JSON.stringify(forms.scenarioFilters || [], null, 2)
                  }
                  onChange={(event) => patchForm("scenarioFilters", event.target.value)}
                />
                <div className="message message--accent tiny">
                  Use JSON filters like <code>{'[{"column":"age","operator":"gte","value":45}]'}</code> to simulate a
                  cohort median or mode baseline.
                </div>
              </div>
            ) : null}
            <fieldset className="form-fieldset">
              <legend>Scenario drivers</legend>
              <p className="form-fieldset__hint">Choose the features that power the upside and downside simulator lanes.</p>
              <div className="grid grid--two">
                {[0, 1].map((slot) => (
                  <label key={slot} className="field">
                    <span>Driver {slot + 1}</span>
                    <select
                      value={focusFeatures[slot] || ""}
                      onChange={(event) => {
                        const next = [...focusFeatures];
                        next[slot] = event.target.value;
                        setFocusFeatures(next.filter(Boolean));
                      }}
                    >
                      <option value="">Select feature...</option>
                      {numericRanges.map((item) => (
                        <option key={item.feature} value={item.feature}>
                          {item.feature}
                        </option>
                      ))}
                    </select>
                  </label>
                ))}
              </div>
            </fieldset>

            <fieldset className="form-fieldset">
              <legend>Scenario adjustments</legend>
              <p className="form-fieldset__hint">Tune the optimistic and stress cases with the selected driver sliders.</p>
              <div className="scenario-grid">
                {["upside", "downside"].map((kind) => (
                  <article key={kind} className="scenario-card">
                    <div className="scenario-card__header">
                      <strong>{kind === "upside" ? "Upside Case" : "Downside Case"}</strong>
                      <span className="tiny-eyebrow">{kind === "upside" ? "optimistic" : "stress"}</span>
                    </div>
                    <div className="stack compact">
                      {focusFeatures.filter(Boolean).map((featureName) => {
                        const range = numericRanges.find((item) => item.feature === featureName);
                        if (!range) return null;
                        const currentValue = scenarioValues[kind]?.[featureName] ?? range.median;
                        return (
                          <label key={`${kind}-${featureName}`} className="field">
                            <span>{featureName}</span>
                            <input
                              type="range"
                              min={range.min}
                              max={range.max}
                              step={range.step || 0.1}
                              value={currentValue}
                              onChange={(event) =>
                                setScenarioValues((current) => ({
                                  ...current,
                                  [kind]: {
                                    ...current[kind],
                                    [featureName]: Number(event.target.value),
                                  },
                                }))
                              }
                            />
                            <div className="slider-meta">
                              <span>{range.min}</span>
                              <strong>{formatNumber(currentValue, 2)}</strong>
                              <span>{range.max}</span>
                            </div>
                          </label>
                        );
                      })}
                    </div>
                  </article>
                ))}
              </div>
            </fieldset>

            <fieldset className="form-fieldset">
              <legend>Guardrails and sweep</legend>
              <p className="form-fieldset__hint">Control sweep behavior, approval thresholds, and blocked features in one place.</p>
              <div className="scenario-policy-grid">
                <label className="field">
                  <span>Sweep feature</span>
                  <select
                    value={forms.scenarioSweepFeature || ""}
                    onChange={(event) => patchForm("scenarioSweepFeature", event.target.value)}
                  >
                    <option value="">Select a sweep feature...</option>
                    {numericRanges.map((item) => (
                      <option key={item.feature} value={item.feature}>
                        {item.feature}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="field">
                  <span>Sweep values</span>
                  <input
                    value={forms.scenarioSweepValues}
                    onChange={(event) => patchForm("scenarioSweepValues", event.target.value)}
                    placeholder="10,20,30"
                  />
                </label>
                <label className="field">
                  <span>Max change before approval (%)</span>
                  <input
                    type="number"
                    min="0"
                    value={forms.scenarioMaxChangePct}
                    onChange={(event) => patchForm("scenarioMaxChangePct", event.target.value)}
                  />
                </label>
                <label className="field">
                  <span>Blocked features</span>
                  <input
                    value={forms.scenarioBlockedFeatures}
                    onChange={(event) => patchForm("scenarioBlockedFeatures", event.target.value)}
                    placeholder="comma,separated,features"
                  />
                </label>
              </div>
              <Checkbox
                checked={Boolean(forms.scenarioHardBounds)}
                onChange={(checked) => patchForm("scenarioHardBounds", checked)}
                label="Block scenarios that move outside the trained feature range"
              />
              <label className="field">
                <span>Approved scenario IDs</span>
                <input
                  value={forms.scenarioApprovedIds}
                  onChange={(event) => patchForm("scenarioApprovedIds", event.target.value)}
                  placeholder="scenario-1,scenario-2"
                />
              </label>
            </fieldset>
            <div className="inline-actions">
              <button className="button button--ghost tiny" type="button" onClick={approveScenarioReviews}>
                Approve Review Scenarios
              </button>
              <button className="button button--primary tiny" type="button" onClick={saveScenarioPack}>
                Save Scenario Pack
              </button>
            </div>
            <fieldset className="form-fieldset">
              <legend>Scenario pack metadata</legend>
              <p className="form-fieldset__hint">Name and describe reusable packs before saving them for the active run.</p>
              <div className="grid grid--two">
                <label className="field">
                  <span>Pack name</span>
                  <input
                    value={forms.scenarioPackName}
                    onChange={(event) => patchForm("scenarioPackName", event.target.value)}
                    placeholder="Q2 pricing guardrail pack"
                  />
                </label>
                <label className="field">
                  <span>Pack description</span>
                  <input
                    value={forms.scenarioPackDescription}
                    onChange={(event) => patchForm("scenarioPackDescription", event.target.value)}
                    placeholder="Reusable scenarios for this run"
                  />
                </label>
              </div>
            </fieldset>
            <Message text={scenarioMessage} />
            <Message text={scenarioPacksMessage} tone="warning" />
          </div>
        </Panel>

        <Panel
          title="Realtime comparison"
          subtitle="Prediction movement, confidence, and sweep response update from the backend scenario engine."
        >
          <div className="grid grid--stats">
            <StatCard label="Approved" value={scenarioResult?.guardrail_summary?.approved ?? 0} tone="success" />
            <StatCard
              label="Needs Review"
              value={scenarioResult?.guardrail_summary?.review_required ?? 0}
              tone="warning"
            />
            <StatCard label="Blocked" value={scenarioResult?.guardrail_summary?.blocked ?? 0} tone="default" />
            <StatCard label="Saved Packs" value={scenarioPacks.length} tone="accent" />
          </div>
          <div className="grid grid--stats" style={{ marginTop: "12px" }}>
            <StatCard label="Executed" value={scenarioComparisonSummary.executed} />
            <StatCard label="Positive Delta" value={scenarioComparisonSummary.positive} tone="success" />
            <StatCard
              label="Best Case"
              value={scenarioComparisonSummary.strongest?.name || "—"}
              meta={
                scenarioComparisonSummary.strongest?.delta === null ||
                scenarioComparisonSummary.strongest?.delta === undefined
                  ? "no executed scenario"
                  : `Delta ${formatNumber(scenarioComparisonSummary.strongest.delta, 4)}`
              }
              tone="warning"
            />
            <StatCard
              label="Baseline Mode"
              value={scenarioResult?.cohort?.mode || scenarioMode}
              meta={scenarioResult?.cohort?.rows ? `${scenarioResult.cohort.rows} rows` : "single baseline"}
            />
          </div>
          {scenarioResult ? (
            <InsightSummary
              title="Scenario result summary"
              tone="success"
              items={[
                `${scenarioComparisonSummary.executed} scenarios executed with ${scenarioComparisonSummary.positive} positive deltas.`,
                scenarioComparisonSummary.strongest?.name
                  ? `${scenarioComparisonSummary.strongest.name} is the current best case.`
                  : null,
                scenarioResult?.guardrail_summary
                  ? `${scenarioResult.guardrail_summary.review_required ?? 0} scenarios need review and ${scenarioResult.guardrail_summary.blocked ?? 0} are blocked.`
                  : null,
              ]}
            />
          ) : null}
          <MiniAreaChart
            points={scenarioChartPoints}
            valueKey="value"
            labelKey="label"
            empty="Move a scenario slider to generate live comparison output."
          />
          <div style={{ marginTop: "1rem" }}>
            <span className="tiny-eyebrow">Sweep response</span>
            <MiniAreaChart
              points={sweepChartPoints}
              valueKey="value"
              labelKey="label"
              empty="Pick a sweep feature and values to generate the response curve."
            />
          </div>
          {scenarioResult?.baseline ? (
            <div className="stack" style={{ marginTop: "1rem" }}>
              <KeyValueList
                items={[
                  { label: "Baseline prediction", value: scenarioResult.baseline.prediction ?? "—" },
                  {
                    label: "Baseline confidence",
                    value: scenarioResult.baseline.confidence_pct ? `${scenarioResult.baseline.confidence_pct}%` : "—",
                  },
                  { label: "Sweep feature", value: scenarioResult.sweep_feature || "—" },
                  { label: "Cohort mode", value: scenarioResult?.cohort?.mode || "—" },
                  { label: "Cohort rows", value: scenarioResult?.cohort?.rows || "—" },
                ]}
              />
              <DetailJson title="Baseline payload" value={scenarioResult.baseline.payload} />
              {scenarioResult?.cohort?.mode === "cohort" && scenarioFilters.length > 0 ? (
                <DetailJson title="Applied cohort filters" value={scenarioFilters} />
              ) : null}
              <DataTable
                compact
                columns={[
                  { key: "name", label: "Scenario" },
                  { key: "approval_status", label: "Approval" },
                  { key: "executed", label: "Executed", render: (row) => (row.executed ? "Yes" : "No") },
                  { key: "prediction", label: "Prediction", render: (row) => formatNumber(row.prediction) },
                  {
                    key: "confidence_pct",
                    label: "Confidence",
                    render: (row) =>
                      row.confidence_pct === null || row.confidence_pct === undefined
                        ? "—"
                        : `${formatNumber(row.confidence_pct)}%`,
                  },
                  {
                    key: "delta",
                    label: "Delta",
                    render: (row) => (row.delta === null || row.delta === undefined ? "—" : formatNumber(row.delta, 4)),
                  },
                  {
                    key: "delta_pct",
                    label: "Delta %",
                    render: (row) =>
                      row.delta_pct === null || row.delta_pct === undefined
                        ? "—"
                        : `${formatNumber(row.delta_pct, 2)}%`,
                  },
                ]}
                rows={scenarioResult.scenarios || []}
              />
              {(scenarioResult?.sweep_predictions || []).length > 0 ? (
                <div style={{ marginTop: "1rem" }}>
                  <span className="tiny-eyebrow">Sweep table</span>
                  <DataTable
                    compact
                    columns={[
                      { key: "x", label: "Sweep value" },
                      { key: "prediction", label: "Prediction", render: (row) => formatNumber(row.prediction) },
                      {
                        key: "confidence",
                        label: "Confidence",
                        render: (row) =>
                          row.confidence === null || row.confidence === undefined
                            ? "—"
                            : `${formatNumber(row.confidence)}%`,
                      },
                    ]}
                    rows={scenarioResult.sweep_predictions}
                  />
                </div>
              ) : null}
              {scenarioResult?.recommended_features?.length ? (
                <div className="scenario-suggestion-list" style={{ marginTop: "1rem" }}>
                  {scenarioResult.recommended_features.map((item, index) => (
                    <article key={`${item}-${index}`} className="scenario-suggestion">
                      <span className="tiny-eyebrow">Simulator note</span>
                      <strong>{item}</strong>
                    </article>
                  ))}
                </div>
              ) : null}
              {scenarioResult?.auto_suggestions?.length ? (
                <div className="scenario-suggestion-list" style={{ marginTop: "1rem" }}>
                  {scenarioResult.auto_suggestions.map((item) => (
                    <article key={item.type} className="scenario-suggestion">
                      <span className="tiny-eyebrow">{item.title}</span>
                      <strong>{item.summary}</strong>
                      <p>
                        {item.feature}: {formatNumber(item.baseline_value, 3)} to{" "}
                        {formatNumber(item.candidate_value, 3)}
                        {" • "}
                        Delta {item.delta === null || item.delta === undefined ? "—" : formatNumber(item.delta, 4)}
                      </p>
                    </article>
                  ))}
                </div>
              ) : null}
              {scenarioPacks.length ? (
                <div style={{ marginTop: "1rem" }}>
                  <span className="tiny-eyebrow">Saved scenario packs for this run</span>
                  <DataTable
                    compact
                    columns={[
                      { key: "name", label: "Pack" },
                      { key: "description", label: "Description" },
                      { key: "base_mode", label: "Baseline" },
                      {
                        key: "approved_scenarios",
                        label: "Approved",
                        render: (row) => (Array.isArray(row.approved_scenarios) ? row.approved_scenarios.length : 0),
                      },
                      {
                        key: "apply",
                        label: "Apply",
                        render: (row) => (
                          <button
                            className="button button--ghost tiny"
                            type="button"
                            onClick={() => applyScenarioPack(row)}
                          >
                            Load
                          </button>
                        ),
                      },
                    ]}
                    rows={scenarioPacks}
                  />
                </div>
              ) : null}
            </div>
          ) : (
            <Message text="Scenario comparison results will appear here." />
          )}
        </Panel>
      </div>
    );
  }

  function renderLabTab() {
    if (!selectedJobId) {
      return (
        <Panel
          title="Advanced inference tools"
          subtitle="Select a completed run to activate telemetry, counterfactuals, live prediction, sweep analysis, and scenario simulation."
        >
          <Message text="No completed run is selected yet." />
        </Panel>
      );
    }

    if (activeLabTab === "scenario-simulator") return renderScenarioLab();

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
                <button
                  className="button button--ghost tiny"
                  type="button"
                  onClick={() => loadPredictionTemplate("predictPayload")}
                >
                  Load Template
                </button>
              </div>
              <textarea
                className="code-editor"
                rows={7}
                value={
                  typeof forms.predictPayload === "string"
                    ? forms.predictPayload
                    : JSON.stringify(forms.predictPayload, null, 2)
                }
                onChange={(event) => patchForm("predictPayload", event.target.value)}
              />
              <label className="field">
                <span>{isClassification ? "Target confidence / score" : "Target prediction"}</span>
                <input
                  type="number"
                  step="0.1"
                  value={forms.goalSeekTarget || ""}
                  onChange={(event) => patchForm("goalSeekTarget", event.target.value)}
                />
              </label>
              <button className="button button--accent" type="button" onClick={getGoalSeeker}>
                Find Optimal Path
              </button>
              <Message text={goalSeekerMessage} />
              {goalSeeker ? (
                <InsightSummary
                  title="Goal seeker summary"
                  items={[
                    `Current output is ${goalSeeker?.current_prediction ?? goalSeeker?.prediction ?? "unknown"} and the target is ${goalSeeker?.target_prediction ?? forms.goalSeekTarget ?? "unknown"}.`,
                    Array.isArray(goalSeeker?.suggestions) && goalSeeker.suggestions.length
                      ? `${goalSeeker.suggestions.length} counterfactual adjustments are available.`
                      : goalSeeker?.message || goalSeeker?.error || "No counterfactual adjustments are available yet.",
                  ]}
                />
              ) : null}
            </div>
          </Panel>

          <Panel
            title="Counterfactual results"
            subtitle="Suggestions are ranked by the smallest one-step change that improves the outcome."
          >
            <KeyValueList
              items={[
                { label: "Current output", value: goalSeeker?.current_prediction ?? goalSeeker?.prediction ?? "—" },
                {
                  label: isClassification ? "Current confidence / score" : "Current score",
                  value: goalSeeker?.current_score ?? "—",
                },
                { label: "Target", value: goalSeeker?.target_prediction ?? forms.goalSeekTarget ?? "—" },
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
                    { key: "new_score", label: isClassification ? "New score" : "New prediction" },
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
              <select
                value={forms.sweepFeature || ""}
                onChange={(event) => patchForm("sweepFeature", event.target.value)}
              >
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
              value={
                typeof forms.baseFeatures === "string"
                  ? forms.baseFeatures
                  : JSON.stringify(forms.baseFeatures, null, 2)
              }
              onChange={(event) => patchForm("baseFeatures", event.target.value)}
            />
            <label className="field">
              <span>Sweep values</span>
              <input value={forms.futureValues} onChange={(event) => patchForm("futureValues", event.target.value)} />
            </label>
            <button className="button button--primary" type="button" onClick={handleFutureSweep}>
              Run Sweep
            </button>
            <Message text={futureMessage} />
            {futureResult?.predictions?.length > 0 ? (
              <InsightSummary
                title="Future sweep summary"
                items={[
                  `${futureResult.predictions.length} sweep points were generated for ${forms.sweepFeature || "the selected feature"}.`,
                  `Predictions range from ${formatNumber(
                    Math.min(...futureResult.predictions.map((row) => Number(row.prediction) || 0)),
                  )} to ${formatNumber(Math.max(...futureResult.predictions.map((row) => Number(row.prediction) || 0)))}.`,
                ]}
              />
            ) : null}
          </div>
        </Panel>

        <Panel title="Sweep analysis" subtitle="Visualize the response curve and inspect each swept value.">
          <MiniAreaChart
            points={telemetryPoints}
            valueKey="value"
            labelKey="label"
            empty="Run a future sweep to generate the response curve."
          />
          {futureResult?.predictions?.length > 0 ? (
            <div style={{ marginTop: "1rem" }}>
              <DataTable
                compact
                columns={[
                  { key: "x", label: "Sweep value" },
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
          ) : null}
        </Panel>
      </div>
    );
  }

  return (
    <>
      <PageHero
        eyebrow="Advanced Lab"
        title="Inference, simulation, ensembles, synthetic data, and AI assistance"
        description="This lab now includes a scenario-based simulator, live prediction tools, a proper ensemble builder, and the existing advanced ML utilities in one place."
        stats={[
          {
            label: "Meta backend",
            value: metaStatus?.backend || "heuristics",
            detail: metaStatus?.is_trained ? "trained" : "fallback",
          },
          {
            label: "Inference tabs",
            value: LAB_TABS.length,
            detail: "simulator, telemetry, counterfactuals, prediction, sweep",
          },
          {
            label: "Selected run",
            value: selectedJobId ? "Ready" : "Missing",
            detail: selectedJobId ? "tools are active" : "pick a completed run",
          },
          { label: "Simulator", value: scenarioContext ? "Ready" : "Booting", detail: "realtime what-if analysis" },
        ]}
      />

      <div className="grid grid--stats">
        <StatCard
          label="Meta engine"
          value={metaStatus?.backend || "heuristics"}
          detail={metaStatus?.is_trained ? "trained engine" : "fallback engine"}
          tone="warning"
        />
        <StatCard
          label="Zero-shot"
          value={outputs.zeroshot ? "Ready" : "Pending"}
          detail="model family recommendation"
        />
        <StatCard
          label="Synthetic ops"
          value={outputs.synthetic ? "Ready" : "Pending"}
          detail="augmentation and judging"
          tone="success"
        />
        <StatCard label="Ensemble builder" value={ensembleResult ? "Ready" : "Pending"} detail="blend completed runs" />
      </div>

      <Panel title="Advanced inference lab" subtitle="Each inference workflow now has its own tab under Advanced Lab.">
        <div className="tab-strip" role="tablist" aria-label="Advanced inference tabs">
          {LAB_TABS.map((tab) => (
            <button
              key={tab.id}
              id={`${tab.id}-tab`}
              className={`tab-strip__button ${activeLabTab === tab.id ? "tab-strip__button--active" : ""}`}
              type="button"
              role="tab"
              aria-selected={activeLabTab === tab.id}
              aria-controls={`${tab.id}-panel`}
              tabIndex={activeLabTab === tab.id ? 0 : -1}
              onClick={() => setActiveLabTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>
        <div
          id={`${activeLabTab}-panel`}
          role="tabpanel"
          aria-labelledby={`${activeLabTab}-tab`}
          style={{ marginTop: "1.25rem" }}
        >
          {renderLabTab()}
        </div>
      </Panel>

      <div className="grid grid--two">
        <Panel
          title="Ensemble Builder"
          subtitle="Blend multiple completed runs into reusable bagging, boosting, voting, or stacking ensembles."
        >
          <div className="stack">
            <label className="field">
              <span>Ensemble strategy</span>
              <select
                value={forms.ensembleStrategy}
                onChange={(event) => patchForm("ensembleStrategy", event.target.value)}
              >
                <option value="bagging">Bagging</option>
                <option value="boosting">Boosting</option>
                <option value="voting">Weighted Voting</option>
                <option value="stacking">Stacking</option>
              </select>
            </label>
            <div className="grid grid--stats">
              <StatCard label="Selected runs" value={selectedEnsembleRuns.length} detail="including the base run" />
              <StatCard
                label="Score spread"
                value={ensembleScoreSpread === null ? "—" : formatNumber(ensembleScoreSpread, 2)}
                detail="max minus min"
              />
              <StatCard
                label="Recommended"
                value={ensembleRecommendation.label}
                detail={`${ensembleRecommendation.confidence} confidence`}
                tone="success"
              />
            </div>
            <div className="message">
              <strong>Strategy guide:</strong> {ensembleRecommendation.reason}
            </div>
            <div className="inline-actions">
              <button
                className="button button--secondary tiny"
                type="button"
                onClick={() => patchForm("ensembleStrategy", ensembleRecommendation.strategy)}
              >
                Use {ensembleRecommendation.label}
              </button>
              {["bagging", "boosting", "voting", "stacking"]
                .filter((strategy) => strategy !== ensembleRecommendation.strategy)
                .map((strategy) => (
                  <button
                    key={strategy}
                    className="button button--ghost tiny"
                    type="button"
                    onClick={() => patchForm("ensembleStrategy", strategy)}
                  >
                    {strategy === "voting" ? "Weighted Voting" : strategy.charAt(0).toUpperCase() + strategy.slice(1)}
                  </button>
                ))}
            </div>
            <Message text={ENSEMBLE_STRATEGY_COPY[forms.ensembleStrategy] || ENSEMBLE_STRATEGY_COPY.voting} />
            <div className="message">
              <strong>Base run included:</strong>{" "}
              {selectedJobId ? `${selectedJobId.slice(0, 8)} is always included.` : "Select a run first."}
            </div>
            <div className="scenario-scroll">
              <div className="stack compact">
                {compatibleCompletedJobs.map((job) => (
                  <Checkbox
                    key={job.id}
                    label={`${job.id.slice(0, 8)} • ${job.best_model || job.status}`}
                    checked={String(forms.ensembleJobIds || "")
                      .split(",")
                      .includes(job.id)}
                    onChange={(checked) => toggleEnsembleSelection(job.id, checked)}
                  />
                ))}
              </div>
            </div>
            <button className="button button--primary" type="button" onClick={buildEnsemble}>
              Build Ensemble
            </button>
            <Message text={ensembleMessage} />
            {ensembleResult ? (
              <>
                <InsightSummary
                  title="Ensemble result summary"
                  tone="success"
                  items={[
                    `${ensembleResult.models_combined?.length || 0} models were combined with ${ensembleResult.strategy || forms.ensembleStrategy}.`,
                    `${ensembleResult.metric_name || "Ensemble score"} is ${
                      ensembleResult.ensemble_score === undefined ? "not available yet" : formatNumber(ensembleResult.ensemble_score, 4)
                    }.`,
                  ]}
                />
                <KeyValueList
                  items={[
                    { label: "New run", value: ensembleResult.job_id || "—" },
                    { label: "Strategy", value: ensembleResult.strategy || forms.ensembleStrategy },
                    { label: "Models combined", value: ensembleResult.models_combined?.length || 0 },
                    { label: ensembleResult.metric_name || "Score", value: ensembleResult.ensemble_score ?? "—" },
                  ]}
                />
                {ensembleResult.component_weights?.length ? (
                  <DataTable
                    compact
                    columns={[
                      { key: "job_id", label: "Run", render: (row) => String(row.job_id || "—").slice(0, 8) },
                      { key: "model", label: "Model" },
                      { key: "score", label: "Score", render: (row) => formatNumber(row.score) },
                      {
                        key: "weight",
                        label: "Weight",
                        render: (row) =>
                          row.weight === null || row.weight === undefined ? "—" : formatNumber(row.weight, 4),
                      },
                    ]}
                    rows={ensembleResult.component_weights}
                  />
                ) : null}
                <DetailJson title="Ensemble details" value={ensembleResult} />
              </>
            ) : null}
            {!compatibleCompletedJobs.length ? (
              <Message text="No additional completed runs are compatible with the selected run yet." />
            ) : null}
          </div>
        </Panel>

        <Panel title="Inference Contract" subtitle="Schema validation for production readiness.">
          <form className="stack" onSubmit={contractCheck}>
            <label className="field">
              <span>Inference contract file</span>
              <input ref={uploadRefs.contractUploadRef} type="file" />
              <p className="field__helper">Upload the contract or schema file you want to validate for production readiness.</p>
            </label>
            <button className="button button--secondary" type="submit">
              Validate Contract
            </button>
          </form>
          <Message text={messages.contractCheck} />
        </Panel>

        <Panel title="Batch Prediction" subtitle="Upload a CSV and get predictions for every row.">
          <form className="stack" onSubmit={batchPredict}>
            <label className="field">
              <span>Batch input (CSV)</span>
              <input ref={uploadRefs.batchUploadRef} type="file" />
              <p className="field__helper">Choose a CSV payload to score every row in one batch inference run.</p>
            </label>
            <button className="button button--primary" type="submit">
              Run Batch Inference
            </button>
          </form>
          <Message text={messages.batchPredict} />
        </Panel>

        <Panel
          title="Synthetic and meta-learning tools"
          subtitle="Augmentation, judging, and model-prior discovery tools stay in the same lab."
        >
          <div className="monitoring-retrain-plan monitoring-retrain-plan--compact">
            <div className="monitoring-retrain-plan__header">
              <strong>Expansion plan</strong>
              <span className="tiny-eyebrow">{selectedDatasetId ? "dataset ready" : "select dataset"}</span>
            </div>
            <div className="monitoring-retrain-plan__grid">
              <div className="monitoring-retrain-plan__item">
                <span>Current dataset</span>
                <strong>{selectedDatasetId || "—"}</strong>
              </div>
              <div className="monitoring-retrain-plan__item">
                <span>Current rows</span>
                <strong>{datasetProfile?.rows ?? "—"}</strong>
              </div>
              <div className="monitoring-retrain-plan__item">
                <span>Recommended rows</span>
                <strong>{syntheticRecommendedRows ?? "—"}</strong>
              </div>
              <div className="monitoring-retrain-plan__item">
                <span>Planned addition</span>
                <strong>{syntheticPlannedRows}</strong>
              </div>
            </div>
          </div>
          <label className="field">
            <span>Requested synthetic rows</span>
            <input
              inputMode="numeric"
              placeholder="Auto"
              value={forms.syntheticRowCount || ""}
              onChange={(event) => patchForm("syntheticRowCount", event.target.value)}
            />
          </label>
          <div className="inline-actions">
            <button className="button button--secondary" type="button" onClick={syntheticExpand}>
              Expand Dataset
            </button>
            <button className="button button--secondary" type="button" onClick={syntheticJudge}>
              Judge Synthetic Quality
            </button>
            {syntheticRecommendedRows ? (
              <button
                className="button button--ghost"
                type="button"
                onClick={() => patchForm("syntheticRowCount", String(syntheticRecommendedRows))}
              >
                Use Recommended
              </button>
            ) : null}
          </div>
          <Message text={messages.synthetic} />

          <div className="inline-actions">
            <button className="button button--secondary" type="button" onClick={zeroShot}>
              Zero-Shot Recommendation
            </button>
            <button className="button button--secondary" type="button" onClick={metaInsights}>
              Cross-Dataset Insights
            </button>
          </div>
          <Message text={messages.zeroshot || messages.metaInsights} />
          {outputs.synthetic ? (
            <>
              <div className="stack compact">
                <span className="tiny-eyebrow">Synthetic comparison</span>
                <BeforeAfterChart items={syntheticComparisonChartItems} />
              </div>
              <KeyValueList
                items={[
                  { label: "Mode", value: outputs.synthetic.generation_mode || "—" },
                  { label: "Dataset", value: outputs.synthetic.dataset_id || "—" },
                  { label: "New dataset", value: outputs.synthetic.new_dataset_id || "—" },
                  { label: "Requested rows", value: outputs.synthetic.requested_rows ?? "Auto" },
                  { label: "Recommended rows", value: outputs.synthetic.recommended_rows ?? "—" },
                  {
                    label: "Rows added",
                    value: outputs.synthetic.synthetic_rows_added ?? outputs.synthetic.rows_evaluated ?? "—",
                  },
                  { label: "Augmentation ratio", value: outputs.synthetic.augmentation_ratio ?? "—" },
                  { label: "Realism score", value: outputs.synthetic.realism_score ?? "—" },
                  { label: "Verdict", value: outputs.synthetic.verdict || "—" },
                  { label: "Duplicate ratio", value: outputs.synthetic.duplicate_ratio ?? "—" },
                  { label: "Missingness drift", value: outputs.synthetic.avg_missing_delta ?? "—" },
                ]}
              />
              {outputs.synthetic.adjustment_note ? (
                <div className="message message--info">{outputs.synthetic.adjustment_note}</div>
              ) : null}
              {outputs.synthetic.new_dataset_id ? (
                <div className="inline-actions">
                  <button className="button button--primary" type="button" onClick={trainSyntheticDataset}>
                    Train On Synthetic Dataset
                  </button>
                </div>
              ) : null}
              {Array.isArray(outputs.synthetic.notes) && outputs.synthetic.notes.length ? (
                <div className="stack compact">
                  <span className="tiny-eyebrow">Judge notes</span>
                  <DataTable
                    compact
                    columns={[{ key: "note", label: "Note" }]}
                    rows={outputs.synthetic.notes.map((note, index) => ({ id: index, note }))}
                  />
                </div>
              ) : null}
              {Array.isArray(outputs.synthetic.preview) && outputs.synthetic.preview.length ? (
                <div className="stack compact">
                  <span className="tiny-eyebrow">Generated row preview</span>
                  <DataTable
                    compact
                    columns={Object.keys(outputs.synthetic.preview[0] || {}).map((key) => ({ key, label: key }))}
                    rows={outputs.synthetic.preview}
                  />
                </div>
              ) : null}
              <DetailJson title="Synthetic output" value={outputs.synthetic} />
            </>
          ) : null}
          {outputs.zeroshot ? <DetailJson title="Zero-shot recommendation" value={outputs.zeroshot} /> : null}
          {outputs.metaInsights ? <DetailJson title="Cross-dataset insights" value={outputs.metaInsights} /> : null}
        </Panel>
      </div>

      <div className="grid grid--two">
        <Panel
          title="Natural-language intent"
          subtitle="Describe what you want to do and let the backend parse it into ML intent."
        >
          <form className="stack" onSubmit={parseIntent}>
            <label className="field">
              <span>Prompt</span>
              <textarea
                rows="6"
                value={forms.nlPrompt}
                onChange={(event) => patchForm("nlPrompt", event.target.value)}
              />
            </label>
            <button className="button button--primary" type="submit">
              Parse Intent
            </button>
          </form>
          <Message text={messages.intent} />
          {outputs.intent ? (
            <>
              <KeyValueList
                items={[
                  { label: "Task", value: outputs.intent.task || "—" },
                  { label: "Target", value: outputs.intent.target || "—" },
                  { label: "Goal", value: outputs.intent.goal || "—" },
                  { label: "Mode", value: outputs.intent.mode || "—" },
                  { label: "Confidence", value: outputs.intent.confidence ?? "—" },
                  { label: "Ready to train", value: outputs.intent.ready_to_train ? "Yes" : "No" },
                ]}
              />
              <Message text={outputs.intent.explanation} />
              <DetailJson title="Parsed intent payload" value={outputs.intent} />
            </>
          ) : null}
        </Panel>

        <Panel
          title="Chat with run context"
          subtitle="Ask questions about the currently selected run without leaving the studio."
        >
          <form className="stack" onSubmit={chat}>
            <label className="field">
              <span>Question</span>
              <textarea
                rows="8"
                value={forms.chatPrompt}
                onChange={(event) => patchForm("chatPrompt", event.target.value)}
              />
            </label>
            <button className="button button--primary" type="submit">
              Ask Studio Assistant
            </button>
          </form>
          <Message text={messages.chat} />
          {outputs.chat?.response ? <Message text={outputs.chat.response} /> : null}
          {outputs.chat ? <DetailJson title="Chat payload" value={outputs.chat} /> : null}
        </Panel>

        <Panel title="Quicktrain sandbox" subtitle="Run a compact experimental fit without leaving Advanced Lab.">
          <form className="stack" onSubmit={quicktrain}>
            <label className="field">
              <span>Target column</span>
              <input
                value={forms.quicktrainTarget}
                onChange={(event) => patchForm("quicktrainTarget", event.target.value)}
              />
            </label>
            <label className="field">
              <span>Models</span>
              <input
                value={forms.quicktrainModels}
                onChange={(event) => patchForm("quicktrainModels", event.target.value)}
              />
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
