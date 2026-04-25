import React from "react";
import {
  DataTable,
  KeyValueList,
  Message,
  PageHero,
  Panel,
  StatCard,
  HealthCard,
  Badge,
  EmptyState,
  Spinner,
} from "../components/ui.jsx";
import { formatDate, formatPercent } from "../lib/format.js";

export function DataPage({
  datasets,
  datasetProfile,
  datasetHealth,
  datasetDetect,
  datasetLeakage,
  datasetLineage,
  datasetVersions,
  loading,
  forms,
  patchForm,
  repairPreview,
  repairApply,
  repairMessage,
  runLeakageScan,
  handleMergePreview,
  handleMergeApply,
  mergeState,
}) {
  const datasetColumns = datasetProfile?.columns || [];

  // Health Data
  const health = datasetProfile?.health || datasetHealth || {};
  const sanitizer = datasetProfile?.sanitizer || {};

  // Column Stats Rows
  const columnStatsRows = Object.entries(datasetProfile?.column_stats || {}).map(([col, stats]) => ({
    column: col,
    dtype: stats.dtype,
    missing_pct: formatPercent(stats.missing_pct),
    unique: stats.unique,
    mean: stats.mean ?? "—",
    std: stats.std ?? "—",
    min: stats.min ?? "—",
    max: stats.max ?? "—",
    top: stats.top_values?.join(", ") || "—",
  }));

  const leftColumns = datasets.find((d) => d.id === forms.mergeLeftId)?.columns || [];
  const rightColumns = datasets.find((d) => d.id === forms.mergeRightId)?.columns || [];

  return (
    <>
      <PageHero
        eyebrow="Data DNA"
        title="Dataset intake, health, and lineage in one working surface"
        description="Inspect structure, health, timeline, leakage risks, and target suggestions before committing more compute to training."
        stats={[
          { label: "Rows", value: datasetProfile?.rows ?? "0" },
          { label: "Columns", value: datasetProfile?.cols ?? "0" },
          { label: "Missing %", value: formatPercent(datasetProfile?.missing_pct) },
          { label: "Target Guess", value: datasetProfile?.suggested_target || "—" },
        ]}
      />

      <div className="grid grid--stats">
        <StatCard
          label="Target Signal"
          value={datasetProfile?.suggested_target || "—"}
          meta="Recommended prediction anchor"
          tone="accent"
        />
        <StatCard
          label="Schema Mix"
          value={`${datasetProfile?.num_cols?.length || 0}N / ${datasetProfile?.cat_cols?.length || 0}C`}
          meta="Numerical and categorical split"
        />
        <StatCard label="Data Mass" value={datasetProfile?.size || "Unknown"} meta="Estimated storage footprint" />
        <StatCard
          label="Imbalance"
          value={datasetProfile?.imbalance || "Low"}
          meta="Label distribution skew"
          tone={datasetProfile?.imbalance?.includes("⚠️") ? "warning" : "default"}
        />
      </div>

      <div className="grid grid--two">
        <Panel title="Central Sanitizer" subtitle="Automated cleaning performed during normalization.">
          <KeyValueList
            items={[
              { label: "Rows After", value: sanitizer.rows_after || "—" },
              { label: "Duplicates Removed", value: sanitizer.duplicate_rows_removed ?? "—" },
              { label: "Numeric Coercions", value: sanitizer.numeric_coercions?.length ?? "—" },
              { label: "Datetime Columns", value: sanitizer.datetime_columns?.length ?? "—" },
            ]}
          />
        </Panel>
      </div>

      {health.score !== undefined && (
        <HealthCard
          score={health.score}
          grade={health.grade || "N/A"}
          summary={health.summary || "Stability, completeness, and modeling readiness"}
          issues={health.issues || []}
          bonuses={health.bonuses || []}
        />
      )}

      <Panel title="🔬 Per-Column Statistics" subtitle="Deep dive into each feature's distribution and quality.">
        <DataTable
          columns={[
            { key: "column", label: "Column" },
            { key: "dtype", label: "Type" },
            { key: "missing_pct", label: "Missing" },
            { key: "unique", label: "Unique" },
            { key: "mean", label: "Mean" },
            { key: "std", label: "Std" },
            { key: "min", label: "Min" },
            { key: "max", label: "Max" },
            { key: "top", label: "Top Values" },
          ]}
          rows={columnStatsRows}
          compact
        />
      </Panel>

      <div className="grid grid--two">
        <Panel title="🧹 Data Repair Assistant" subtitle="Preview or apply automated cleaning before training.">
          <div className="stack">
            <label className="field">
              <span>Target column for repair</span>
              <select value={forms.repairTarget} onChange={(event) => patchForm("repairTarget", event.target.value)}>
                <option value="">Select target column</option>
                {datasetColumns.map((column) => (
                  <option key={column} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </label>
            <div className="inline-actions">
              <button className="button button--secondary" type="button" onClick={repairPreview}>
                Preview Cleaned Shape
              </button>
              <button className="button button--primary" type="button" onClick={repairApply}>
                Create Repaired Dataset
              </button>
            </div>
            {loading && <Spinner label="Repairing and profiling data..." />}
          </div>
          <Message text={repairMessage} />
          {forms.repairPreviewData && (
            <div className="stack" style={{ marginTop: "1rem" }}>
              <KeyValueList
                items={[
                  { label: "Rows Before", value: forms.repairPreviewData.before_rows },
                  { label: "Rows After", value: forms.repairPreviewData.after_rows },
                ]}
              />
              <DataTable
                columns={Object.keys(forms.repairPreviewData.preview_records[0] || {}).map((k) => ({
                  key: k,
                  label: k,
                }))}
                rows={forms.repairPreviewData.preview_records}
                compact
              />
            </div>
          )}
        </Panel>

        <Panel title="🤖 Auto Problem Detection" subtitle="Suggested task and target candidates.">
          <KeyValueList
            items={[
              { label: "Suggested Target", value: datasetDetect?.suggested_target || "—" },
              { label: "Task Type", value: datasetDetect?.task_type || "—" },
              { label: "Confidence", value: datasetDetect?.confidence ? `${datasetDetect.confidence}%` : "—" },
            ]}
          />
          {datasetDetect?.column_scores && (
            <div style={{ marginTop: "1rem" }}>
              <DataTable
                columns={[
                  { key: "column", label: "Candidate" },
                  { key: "score", label: "Score" },
                  { key: "task", label: "Likely Task" },
                ]}
                rows={datasetDetect.column_scores.slice(0, 5)}
                compact
              />
            </div>
          )}
        </Panel>
      </div>

      <div className="grid grid--one">
        <Panel
          title="🔍 Data Leakage & Quality Report"
          subtitle="Detect target leakage, constants, and future leakage."
        >
          <div className="stack">
            <div className="inline-actions">
              <button className="button button--secondary" type="button" onClick={runLeakageScan}>
                Run Leakage Scan
              </button>
            </div>
            {loading && <Spinner label="Performing leakage scan..." />}
            {datasetLeakage && (
              <>
                <div className="health-card__signals" style={{ marginTop: "1rem" }}>
                  <Badge tone={datasetLeakage.severity?.includes("Critical") ? "warning" : "success"}>
                    {datasetLeakage.severity}
                  </Badge>
                </div>
                <KeyValueList
                  items={[
                    {
                      label: "Duplicates",
                      value: `${datasetLeakage.duplicate_rows} (${formatPercent(datasetLeakage.duplicate_pct)})`,
                    },
                    { label: "Constant Cols", value: datasetLeakage.constant_columns?.length },
                    { label: "Target Leakage", value: datasetLeakage.target_correlated?.length },
                    { label: "Total Issues", value: datasetLeakage.total_issues },
                  ]}
                />
                {datasetLeakage.warnings?.map((w, i) => (
                  <Message key={i} text={w} tone={w.includes("🔴") ? "warning" : "info"} />
                ))}
              </>
            )}
          </div>
        </Panel>
      </div>

      <div className="grid grid--two">
        <Panel title="🧾 Dataset Version Comparison" subtitle="Detailed diff between current and previous versions.">
          {datasetVersions && !datasetVersions.error ? (
            <div className="stack">
              <KeyValueList
                items={[
                  { label: "Current Rows", value: datasetVersions.current_rows },
                  { label: "Previous Rows", value: datasetVersions.previous_rows },
                  { label: "Row Delta", value: datasetVersions.row_delta },
                  { label: "Added Columns", value: datasetVersions.added_columns?.length },
                ]}
              />
            </div>
          ) : (
            <EmptyState text="Version comparison will appear after creation of derived datasets." />
          )}
        </Panel>

        <Panel title="🕸 Dataset Lineage Graph" subtitle="Visualizing the provenance of this dataset.">
          {datasetLineage?.nodes ? (
            <div className="stack">
              {datasetLineage.nodes.map((node, idx) => (
                <div key={node.id} className="feed-card">
                  <strong>
                    {idx + 1}. {node.label}
                  </strong>
                  <p className="tiny">
                    {node.type || "Lineage event"} • {node.detail || "No detail available."}
                  </p>
                  {node.created_at && <p className="tiny">{formatDate(node.created_at)}</p>}
                </div>
              ))}
              {datasetLineage.edges?.length > 0 && (
                <div style={{ marginTop: "1rem" }}>
                  <span className="tiny-eyebrow">Provenance Flow</span>
                  <DataTable
                    compact
                    columns={[
                      { key: "source", label: "Source" },
                      { key: "target", label: "Result" },
                      { key: "label", label: "Operation" },
                    ]}
                    rows={datasetLineage.edges.map((e) => ({
                      source: e.source.slice(0, 8),
                      target: e.target.slice(0, 8),
                      label: e.label,
                    }))}
                  />
                </div>
              )}
            </div>
          ) : (
            <EmptyState text="Lineage graph not available yet." />
          )}
        </Panel>
      </div>

      <Panel title="🧬 Dataset Merge Studio" subtitle="Select datasets and join keys to create derived data.">
        <div className="stack">
          <div className="grid grid--two">
            <div className="field">
              <span>Left Dataset</span>
              <select value={forms.mergeLeftId} onChange={(e) => patchForm("mergeLeftId", e.target.value)}>
                <option value="">Select dataset</option>
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.display_name} ({d.id.slice(0, 8)})
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <span>Right Dataset</span>
              <select value={forms.mergeRightId} onChange={(e) => patchForm("mergeRightId", e.target.value)}>
                <option value="">Select dataset</option>
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.display_name} ({d.id.slice(0, 8)})
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="grid grid--three">
            <div className="field">
              <span>Left Key</span>
              <select value={forms.mergeLeftKey} onChange={(e) => patchForm("mergeLeftKey", e.target.value)}>
                <option value="">Select key</option>
                {leftColumns.map((k) => (
                  <option key={k} value={k}>
                    {k}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <span>Right Key</span>
              <select value={forms.mergeRightKey} onChange={(e) => patchForm("mergeRightKey", e.target.value)}>
                <option value="">Select key</option>
                {rightColumns.map((k) => (
                  <option key={k} value={k}>
                    {k}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <span>Join Type</span>
              <select value={forms.mergeJoinType} onChange={(e) => patchForm("mergeJoinType", e.target.value)}>
                <option value="inner">Inner</option>
                <option value="left">Left</option>
                <option value="right">Right</option>
                <option value="outer">Outer</option>
              </select>
            </div>
          </div>
          <div className="inline-actions">
            <button className="button button--secondary" type="button" onClick={handleMergePreview}>
              Preview Join Stats
            </button>
            <button className="button button--primary" type="button" onClick={handleMergeApply}>
              Create Merged Dataset
            </button>
          </div>
          {loading && <Spinner label="Analyzing merge strategy..." />}
          <Message text={mergeState.message} />
          {mergeState.preview && (
            <div className="grid grid--stats" style={{ marginTop: "1rem" }}>
              <StatCard label="Estimated Rows" value={mergeState.preview.estimated_rows} />
              <StatCard label="Overlapping Keys" value={mergeState.preview.overlapping_keys} />
              <StatCard label="Left Match %" value={`${mergeState.preview.left_match_pct}%`} />
              <StatCard label="Right Match %" value={`${mergeState.preview.right_match_pct}%`} />
              <StatCard label="Row Multiplier" value={mergeState.preview.estimated_row_multiplier} />
            </div>
          )}
        </div>
      </Panel>
    </>
  );
}
