import React from "react";
import {
  DataTable,
  DetailJson,
  KeyValueList,
  Message,
  PageHero,
  Panel,
  StatCard,
  Checkbox,
  TimelineList,
  MiniAreaChart,
  LineageTree,
} from "../components/ui.jsx";
import { formatDate, formatNumber } from "../lib/format.js";

export function TrackingPage({
  experiments,
  workspaces,
  forms,
  patchForm,
  compareExperiments,
  diffExperiments,
  notesMessage,
  compareMessage,
  diffMessage,
  compareResult,
  diffResult,
  notesResult,
  fetchNotes,
  saveNote,
  datasets,
  selectedDatasetId,
  setSelectedDatasetId,
  handleArchive,
  handleDeleteDataset,
  datasetLineage,
  datasetTimeline,
  datasetVersions,
}) {
  const filteredDatasets = datasets.filter((ds) => forms.showArchived || !ds.archived);
  const selectedDataset = datasets.find((ds) => ds.id === selectedDatasetId);
  const experimentOptions = experiments.slice(0, 50);
  const selectedDiffRunA = experimentOptions.find((run) => run.id === forms.diffRunA || run.job_id === forms.diffRunA);
  const selectedDiffRunB = experimentOptions.find((run) => run.id === forms.diffRunB || run.job_id === forms.diffRunB);
  const selectedCompareIds = String(forms.compareExperimentIds || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  const timelineItems = datasetTimeline?.timeline || [];
  const schemaStability =
    timelineItems.length <= 1
      ? 100
      : Math.max(
          0,
          100 -
            timelineItems.reduce((acc, item) => acc + Math.abs(Number(item.cols || 0) - Number(timelineItems[0]?.cols || 0)), 0) /
              Math.max(timelineItems.length - 1, 1),
        );
  const avgHealthProxy =
    selectedDataset?.missing_pct !== undefined && selectedDataset?.missing_pct !== null
      ? Math.max(0, Math.round(100 - Number(selectedDataset.missing_pct || 0)))
      : "—";
  const topBattlegroundRuns = [...experiments]
    .filter((run) => run.score !== null && run.score !== undefined)
    .sort((left, right) => Number(right.score || 0) - Number(left.score || 0))
    .slice(0, 2);
  const battlegroundDelta =
    topBattlegroundRuns.length === 2
      ? (Number(topBattlegroundRuns[0].score || 0) - Number(topBattlegroundRuns[1].score || 0)).toFixed(2)
      : null;

  return (
    <>
      <PageHero
        eyebrow="History"
        title="Experiment governance with comparisons, workspace memory, and team notes"
        description="This section is redesigned to feel like a proper run ledger for an ML studio, with stronger tables for historical review and cleaner collaboration surfaces."
        stats={[
          { label: "Experiments", value: experiments.length, detail: "stored comparisons" },
          { label: "Workspaces", value: workspaces.length, detail: "persistent contexts" },
          { label: "Compared runs", value: compareResult?.count || 0, detail: "current compare set" },
          { label: "Notes loaded", value: notesResult?.notes?.length || 0, detail: "collaboration context" },
        ]}
      />

      <div className="grid grid--stats">
        <StatCard label="Top experiment" value={experiments[0]?.model_name || "—"} detail={experiments[0]?.metric_name || "no metric yet"} tone="warning" />
        <StatCard label="Top score" value={formatNumber(experiments[0]?.score)} detail="most recent experiment" />
        <StatCard label="Workspace resume" value={workspaces[0]?.name || "—"} detail="latest saved studio space" tone="success" />
        <StatCard label="Compare set" value={compareResult?.comparison?.length || 0} detail="side-by-side runs" />
      </div>

      <Panel title="🗂 Dataset Manager" subtitle="Refresh catalog, load datasets, or manage archives.">
        <div className="stack compact">
          <Checkbox
            label="Show archived datasets"
            checked={forms.showArchived}
            onChange={(v) => patchForm("showArchived", v)}
          />
          <div className="field">
            <span>Dataset Catalog</span>
            <select
              value={selectedDatasetId}
              onChange={(e) => setSelectedDatasetId(e.target.value)}
            >
              <option value="">Select a dataset...</option>
              {filteredDatasets.map((ds) => (
                <option key={ds.id} value={ds.id}>
                  {ds.display_name || ds.source_type} • {ds.rows} rows • {ds.id.slice(0, 8)} {ds.archived ? "(archived)" : ""}
                </option>
              ))}
            </select>
          </div>
          <div className="grid grid--two">
            <button className="button button--secondary" onClick={() => handleArchive(selectedDatasetId, !selectedDataset?.archived)} disabled={!selectedDatasetId}>
              {selectedDataset?.archived ? "Unarchive" : "Archive"}
            </button>
            <button className="button button--ghost" onClick={() => setSelectedDatasetId("")} disabled={!selectedDatasetId}>
              Remove From Workspace
            </button>
          </div>
          <button className="button" style={{ color: "var(--danger)", borderColor: "var(--danger)" }} onClick={() => handleDeleteDataset(selectedDatasetId)} disabled={!selectedDatasetId}>
            Delete Permanently
          </button>
        </div>
      </Panel>
      <div className="grid grid--two">
        <Panel title="Model Lineage Tree" subtitle="Trace the full audit trail from raw data to production mission.">
          <LineageTree
            nodes={datasetLineage?.nodes || [
              { id: "1", type: "Dataset", label: "Raw Ingestion", detail: "raw_data_v1.csv" },
              { id: "2", type: "Processing", label: "Feature Engineering", detail: "8 columns added" },
              { id: "3", type: "Training", label: "Mission: Alpha", detail: "XGBoost • Score: 0.92" },
            ]}
          />
        </Panel>
        <Panel title="Dataset Time-Machine" subtitle="Historical snapshots of data DNA. Monitor schema evolution, null rates, and quality drift.">
          <div className="stack">
            <div className="split">
              <StatCard label="Schema Stability" value={typeof schemaStability === "number" ? `${Math.round(schemaStability)}%` : "—"} detail="lineage consistency" tone="success" />
              <StatCard label="Avg. Health" value={typeof avgHealthProxy === "number" ? `${avgHealthProxy}/100` : "—"} detail="missingness-adjusted proxy" />
            </div>
            <MiniAreaChart
              points={(timelineItems.length ? timelineItems.slice().reverse() : datasets.slice().reverse()).map((d) => ({
                label: String(d.dataset_id || d.id).slice(0, 8),
                value: d.rows || 0,
              }))}
              valueKey="value"
              labelKey="label"
            />
            <TimelineList
              items={(timelineItems.length ? timelineItems : datasets).slice(0, 5).map((d) => ({
                title: `Snapshot: ${String(d.dataset_id || d.id).slice(0, 8)}`,
                detail: `${d.cols || d.columns?.length || 0} columns • ${d.rows || 0} rows • Nulls: ${formatNumber(d.missing_pct ?? selectedDataset?.missing_pct)}%`,
                meta: d.source_type || (d.archived ? "Archived" : "Live Source"),
              }))}
            />
            {datasetVersions?.profile_diff && (
              <div className="message message--accent tiny">
                Row delta: {formatNumber(datasetVersions.profile_diff.rows)} • Column delta: {formatNumber(datasetVersions.profile_diff.cols)} • Missingness delta: {formatNumber(datasetVersions.profile_diff.missing_pct)}%
              </div>
            )}
          </div>
        </Panel>

        <Panel title="Mission Diffs" subtitle="Inspect exact technical changes between two specific mission runs.">
          <div className="stack">
            <div className="split">
              <label className="field">
                <span>Run A</span>
                <select value={forms.diffRunA} onChange={(event) => patchForm("diffRunA", event.target.value)}>
                  <option value="">Select run A...</option>
                  {experimentOptions.map((run) => (
                    <option key={`diff-a-${run.id}`} value={run.id}>
                      {run.model_name || "Run"} • {String(run.id).slice(0, 8)} • {formatNumber(run.score)}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Run B</span>
                <select value={forms.diffRunB} onChange={(event) => patchForm("diffRunB", event.target.value)}>
                  <option value="">Select run B...</option>
                  {experimentOptions.map((run) => (
                    <option key={`diff-b-${run.id}`} value={run.id}>
                      {run.model_name || "Run"} • {String(run.id).slice(0, 8)} • {formatNumber(run.score)}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {(selectedDiffRunA || selectedDiffRunB) && (
              <div className="split">
                <StatCard
                  label="Run A"
                  value={selectedDiffRunA?.model_name || "—"}
                  detail={selectedDiffRunA ? `${selectedDiffRunA.metric_name || "metric"} • ${formatNumber(selectedDiffRunA.score)}` : "select a run"}
                />
                <StatCard
                  label="Run B"
                  value={selectedDiffRunB?.model_name || "—"}
                  detail={selectedDiffRunB ? `${selectedDiffRunB.metric_name || "metric"} • ${formatNumber(selectedDiffRunB.score)}` : "select a run"}
                  tone="accent"
                />
              </div>
            )}
            <button className="button button--secondary" type="button" onClick={diffExperiments}>
              Calculate Differential
            </button>
            <Message text={diffMessage} />
            {diffResult && (
              <div className="stack" style={{ marginTop: "1rem" }}>
                <span className="tiny-eyebrow">Differential Summary</span>
                <DataTable
                  compact
                  columns={[
                    { key: "field", label: "Feature" },
                    { key: "before", label: "Run A" },
                    { key: "after", label: "Run B" },
                  ]}
                  rows={[...(diffResult.config_changes || []), ...(diffResult.output_changes || [])]}
                />
                <div className="message message--accent tiny">
                  {diffResult.explanations?.join(" • ") || "No specific variance detected."}
                </div>
              </div>
            )}
          </div>
        </Panel>
      </div>

      <div className="grid grid--two">
        <Panel title="Multi-Model Battleground" subtitle="Pit two models against each other on the latest workspace slice.">
          <div className="stack">
            {topBattlegroundRuns.length >= 2 ? (
              <>
                <div className="split">
                  <StatCard
                    label="Challenger A"
                    value={formatNumber(topBattlegroundRuns[0].score)}
                    detail={topBattlegroundRuns[0].model_name || "Unknown model"}
                    tone="accent"
                  />
                  <StatCard
                    label="Challenger B"
                    value={formatNumber(topBattlegroundRuns[1].score)}
                    detail={topBattlegroundRuns[1].model_name || "Unknown model"}
                  />
                </div>
                <div className="message message--warning tiny">
                  <strong>Winner: {topBattlegroundRuns[0].model_name}</strong> ({battlegroundDelta} point lift over {topBattlegroundRuns[1].model_name})
                </div>
              </>
            ) : (
              <Message text="At least two scored experiments are needed for a battleground view." />
            )}
          </div>
        </Panel>

        <Panel title="Comparison Arena" subtitle="Select multiple experiments to see side-by-side hyperparameters and metrics.">
          <div className="stack">
            <label className="field">
              <span>Experiment IDs (comma separated)</span>
              <input value={forms.compareExperimentIds} onChange={(event) => patchForm("compareExperimentIds", event.target.value)} placeholder="Paste experiment IDs or job IDs..." />
            </label>
            <div className="stack compact">
              <span className="tiny-eyebrow">Quick pick recent runs</span>
              <div className="grid grid--two">
                {experimentOptions.slice(0, 6).map((run) => {
                  const selected = selectedCompareIds.includes(run.id) || selectedCompareIds.includes(run.job_id);
                  return (
                    <label key={`compare-${run.id}`} className="checkbox">
                      <input
                        type="checkbox"
                        checked={selected}
                        onChange={(event) => {
                          const next = new Set(selectedCompareIds);
                          if (event.target.checked) next.add(run.id);
                          else {
                            next.delete(run.id);
                            if (run.job_id) next.delete(run.job_id);
                          }
                          patchForm("compareExperimentIds", Array.from(next).join(", "));
                        }}
                      />
                      <span>{run.model_name || "Run"} • {String(run.id).slice(0, 8)}</span>
                    </label>
                  );
                })}
              </div>
            </div>
            <button className="button button--primary" type="button" onClick={compareExperiments}>
              Launch Comparison
            </button>
            <Message text={compareMessage} />
            {compareResult?.unresolved?.length > 0 && (
              <div className="message message--warning tiny">
                Unresolved identifiers: {compareResult.unresolved.join(", ")}
              </div>
            )}
            {compareResult?.comparison?.length > 0 && (
              <div className="grid grid--two" style={{ marginTop: "1rem" }}>
                {compareResult.comparison.map((run) => (
                  <div key={run.id} className="stack box">
                    <strong>{run.model_name}</strong>
                    <p className="tiny">{run.metric_name}: {formatNumber(run.score)}</p>
                    <KeyValueList
                      items={Object.entries(run.hyperparams || {}).slice(0, 4).map(([label, value]) => ({ label, value: String(value) }))}
                    />
                  </div>
                ))}
              </div>
            )}
          </div>
        </Panel>
      </div>

      <Panel title="Operational registry" subtitle="A complete history of every automated mission performed in this workspace.">
        <DataTable
          columns={[
            { key: "id", label: "Run ID", render: (row) => row.id.slice(0, 10) },
            { key: "model_name", label: "Model" },
            { key: "task_type", label: "Task" },
            { key: "metric_name", label: "Metric" },
            { key: "score", label: "Score", render: (row) => formatNumber(row.score) },
            { key: "created_at", label: "Created", render: (row) => formatDate(row.created_at) },
          ]}
          rows={experiments}
          empty="No experiment history is available yet."
        />
      </Panel>

      <Panel title="Run annotations" subtitle="Annotate missions with operational context for future audits.">
        <form className="stack" onSubmit={saveNote}>
          <div className="split">
            <label className="field">
              <span>Entity type</span>
              <input value={forms.noteEntityType} onChange={(event) => patchForm("noteEntityType", event.target.value)} placeholder="job, experiment, dataset" />
            </label>
            <label className="field">
              <span>Entity ID</span>
              <input value={forms.noteEntityId} onChange={(event) => patchForm("noteEntityId", event.target.value)} placeholder="target entity id" />
            </label>
          </div>
          <label className="field">
            <span>Annotation text</span>
            <textarea rows="4" value={forms.noteText} onChange={(event) => patchForm("noteText", event.target.value)} placeholder="Why does this run matter?" />
          </label>
          <div className="inline-actions">
            <button className="button button--secondary" type="button" onClick={fetchNotes}>Load Notes</button>
            <button className="button button--primary" type="submit">Save Annotation</button>
          </div>
          <Message text={notesMessage} />
          {notesResult?.notes?.length > 0 && (
            <div className="stack" style={{ marginTop: "1rem" }}>
              <span className="tiny-eyebrow">Audit Annotations</span>
              <DataTable
                compact
                columns={[
                  { key: "id", label: "Note ID", render: (row) => String(row.id || "").slice(0, 8) },
                  { key: "note", label: "Annotation" },
                  { key: "created_at", label: "Logged", render: (row) => formatDate(row.created_at) },
                ]}
                rows={notesResult.notes}
              />
            </div>
          )}
        </form>
      </Panel>
    </>
  );
}
