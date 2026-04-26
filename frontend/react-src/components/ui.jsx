import React from "react";
import { formatDate } from "../lib/format.js";

function renderDisplayValue(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) {
    return value.map((item) => renderDisplayValue(item)).join(", ");
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function PageHero({ eyebrow, title, description, actions, stats = [] }) {
  return (
    <section className="page-hero">
      <div className="page-hero__copy">
        <span className="eyebrow">{eyebrow}</span>
        <h2>{title}</h2>
        <p>{description}</p>
        {actions ? <div className="hero-actions">{actions}</div> : null}
      </div>
      {stats.length ? (
        <div className="hero-stats">
          {stats.map((stat) => (
            <article key={stat.label} className="hero-stat">
              <span>{stat.label}</span>
              <strong>{stat.value}</strong>
              <small>{stat.detail}</small>
            </article>
          ))}
        </div>
      ) : null}
    </section>
  );
}

export function Panel({ title, subtitle, actions, tone = "default", children }) {
  return (
    <section className={`panel panel--${tone}`}>
      <div className="panel__header">
        <div>
          <h3>{title}</h3>
          {subtitle ? <p className="panel__subtitle">{subtitle}</p> : null}
        </div>
        {actions ? <div className="panel__actions">{actions}</div> : null}
      </div>
      {children}
    </section>
  );
}

export function StatCard({ label, value, detail, tone = "default", meta }) {
  return (
    <article className={`stat-card stat-card--${tone}`}>
      <span className="stat-card__label">{label}</span>
      <strong className="stat-card__value">{value}</strong>
      {detail ? <span className="stat-card__detail">{detail}</span> : null}
      {meta ? <div className="stat-card__meta">{meta}</div> : null}
    </article>
  );
}

export function HealthCard({ score, grade, summary, issues = [], bonuses = [] }) {
  return (
    <article className="health-card">
      <div className="health-card__content">
        <span className="eyebrow">Health Matrix</span>
        <div className="health-card__title">Dataset Health Score: {score}/100</div>
        <div className="health-card__copy">{summary}</div>
        <div className="health-card__meta">Stability, completeness, and modeling readiness</div>
        {issues.length || bonuses.length ? (
          <div className="health-card__signals">
            {issues.map((issue, idx) => (
              <span key={idx} className="health-card__signal health-card__signal--warning">
                ⚠️ {issue}
              </span>
            ))}
            {bonuses.map((bonus, idx) => (
              <span key={idx} className="health-card__signal health-card__signal--success">
                🌟 {bonus}
              </span>
            ))}
          </div>
        ) : null}
      </div>
      <div className="health-card__grade">{grade}</div>
    </article>
  );
}

export function Badge({ children, tone = "default", compact = false }) {
  return <span className={`badge badge--${tone} ${compact ? "badge--compact" : ""}`.trim()}>{children}</span>;
}

export function EmptyState({ text, title = "Nothing to show yet", action = null }) {
  return (
    <div className="empty-state">
      <strong>{title}</strong>
      <p>{text}</p>
      {action ? <div className="empty-state__action">{action}</div> : null}
    </div>
  );
}

export function InsightSummary({ title = "Update summary", items = [], tone = "default" }) {
  const rows = items.filter(Boolean);
  if (!rows.length) return null;
  return (
    <div className={`insight-summary insight-summary--${tone}`} role="status" aria-live="polite">
      <strong>{title}</strong>
      <ul>
        {rows.map((item, index) => (
          <li key={`${title}-${index}`}>{item}</li>
        ))}
      </ul>
    </div>
  );
}

export function Message({ text, tone = "info", onDismiss = null, dismissLabel = "Dismiss message" }) {
  if (!text) return null;
  const isUrgent = tone === "danger" || tone === "warning";
  return (
    <div className={`message message--${tone}`} role={isUrgent ? "alert" : "status"} aria-live={isUrgent ? "assertive" : "polite"}>
      <span>{text}</span>
      {typeof onDismiss === "function" ? (
        <button type="button" className="message__dismiss" onClick={onDismiss} aria-label={dismissLabel}>
          Dismiss
        </button>
      ) : null}
    </div>
  );
}

export function DataTable({ columns, rows, empty = "No data available.", compact = false }) {
  if (!rows?.length) return <EmptyState text={empty} />;
  return (
    <div className="table-stack">
      <div
        className={`table-shell ${compact ? "table-shell--compact" : ""}`}
        tabIndex={0}
        role="region"
        aria-label="Scrollable data table"
      >
        <table>
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column.key}>{column.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={row.id || rowIndex}>
                {columns.map((column) => (
                  <td key={column.key}>{column.render ? column.render(row) : renderDisplayValue(row[column.key])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="table-shell__hint" aria-hidden="true">
        Scroll sideways to view all columns.
      </div>
    </div>
  );
}

export function DetailJson({ title = "Inspect payload", value }) {
  if (!value) return null;
  return (
    <details className="detail-json">
      <summary>{title}</summary>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </details>
  );
}

export function NotificationList({ items }) {
  if (!items?.length) return <EmptyState text="No notifications yet." />;
  return (
    <div className="stack">
      {items.slice(0, 6).map((item) => (
        <article key={item.id} className="feed-card">
          <div className="feed-card__row">
            <strong>{item.title}</strong>
            <Badge tone={item.level === "success" ? "success" : item.level === "warning" ? "warning" : "default"}>
              {item.level}
            </Badge>
          </div>
          <p>{item.message}</p>
          <span className="tiny">{formatDate(item.created_at)}</span>
        </article>
      ))}
    </div>
  );
}

export function MeterCard({ label, value = 0, detail, tone = "accent" }) {
  const safeValue = Number.isFinite(Number(value)) ? Math.max(0, Math.min(100, Number(value))) : 0;
  return (
    <article className={`meter-card meter-card--${tone}`}>
      <div className="meter-card__ring" style={{ "--progress": `${safeValue}%` }}>
        <span>{Math.round(safeValue)}</span>
      </div>
      <div>
        <strong>{label}</strong>
        <p>{detail}</p>
      </div>
    </article>
  );
}

export function BarList({ items, valueKey = "value", labelKey = "label", empty = "No ranked signals available." }) {
  if (!items?.length) return <EmptyState text={empty} />;
  const numbers = items
    .map((item) => Number(item?.[valueKey]))
    .filter((value) => Number.isFinite(value))
    .map((value) => Math.abs(value));
  const maxValue = Math.max(...numbers, 1);

  return (
    <div className="bar-list">
      {items.map((item, index) => {
        const numeric = Number(item?.[valueKey]);
        const width = Number.isFinite(numeric) ? `${(Math.abs(numeric) / maxValue) * 100}%` : "0%";
        return (
          <div key={`${item?.[labelKey] || "bar"}-${index}`} className="bar-row">
            <div className="bar-row__meta">
              <strong>{item?.[labelKey] || "Untitled"}</strong>
              <span>{Number.isFinite(numeric) ? numeric.toFixed(3).replace(/\.000$/, "") : "—"}</span>
            </div>
            <div className="bar-row__track">
              <div className="bar-row__fill" style={{ width }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function MiniAreaChart({ points, valueKey = "value", labelKey = "label", empty = "Not enough timeline data." }) {
  if (!points?.length) return <EmptyState text={empty} />;
  const values = points.map((point) => Number(point?.[valueKey])).filter((value) => Number.isFinite(value));
  if (!values.length) return <EmptyState text={empty} />;

  const max = Math.max(...values);
  const min = Math.min(...values);
  const spread = max - min || 1;
  const startLabel = points[0]?.[labelKey] || "Start";
  const endLabel = points[points.length - 1]?.[labelKey] || "Now";
  const startValue = Number(points[0]?.[valueKey]);
  const endValue = Number(points[points.length - 1]?.[valueKey]);
  const trend =
    Number.isFinite(startValue) && Number.isFinite(endValue)
      ? endValue > startValue
        ? "rises"
        : endValue < startValue
          ? "falls"
          : "holds steady"
      : "changes";
  const ariaLabel = `Trend chart from ${startLabel} to ${endLabel}. Values range from ${min.toFixed(2)} to ${max.toFixed(2)} and the series ${trend}.`;
  const coords = points
    .map((point, index) => {
      const raw = Number(point?.[valueKey]);
      const normalized = Number.isFinite(raw) ? (raw - min) / spread : 0;
      const x = points.length === 1 ? 50 : (index / (points.length - 1)) * 100;
      const y = 88 - normalized * 72;
      return `${x},${y}`;
    })
    .join(" ");
  const area = `0,88 ${coords} 100,88`;

  return (
    <div className="chart-card">
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="area-chart" role="img" aria-label={ariaLabel}>
        <polygon points={area} className="area-chart__fill" />
        <polyline points={coords} className="area-chart__line" />
      </svg>
      <div className="chart-card__labels">
        <span>{startLabel}</span>
        <span>{endLabel}</span>
      </div>
    </div>
  );
}

export function BeforeAfterChart({ items = [], empty = "Comparison metrics will appear here." }) {
  const rows = items
    .map((item) => {
      const before = Number(item?.before);
      const after = Number(item?.after);
      const max = Math.max(
        Number.isFinite(before) ? before : 0,
        Number.isFinite(after) ? after : 0,
        Number(item?.max) || 0,
        1,
      );
      return {
        label: item?.label || "Metric",
        before: Number.isFinite(before) ? before : null,
        after: Number.isFinite(after) ? after : null,
        beforeLabel: item?.beforeLabel,
        afterLabel: item?.afterLabel,
        detail: item?.detail,
        max,
      };
    })
    .filter((item) => item.before !== null || item.after !== null);

  if (!rows.length) return <EmptyState text={empty} />;

  return (
    <div className="before-after-chart">
      {rows.map((item) => {
        const beforeWidth = item.before === null ? "0%" : `${(item.before / item.max) * 100}%`;
        const afterWidth = item.after === null ? "0%" : `${(item.after / item.max) * 100}%`;
        return (
          <article key={item.label} className="before-after-chart__row">
            <div className="before-after-chart__meta">
              <strong>{item.label}</strong>
              {item.detail ? <span>{item.detail}</span> : null}
            </div>
            <div className="before-after-chart__tracks">
              <div className="before-after-chart__lane">
                <span>Before</span>
                <div className="before-after-chart__track">
                  <div className="before-after-chart__fill before-after-chart__fill--before" style={{ width: beforeWidth }} />
                </div>
                <strong>{item.beforeLabel ?? renderDisplayValue(item.before)}</strong>
              </div>
              <div className="before-after-chart__lane">
                <span>After</span>
                <div className="before-after-chart__track">
                  <div className="before-after-chart__fill before-after-chart__fill--after" style={{ width: afterWidth }} />
                </div>
                <strong>{item.afterLabel ?? renderDisplayValue(item.after)}</strong>
              </div>
            </div>
          </article>
        );
      })}
    </div>
  );
}

export function ScatterPlot({ points = [], xKey = "x", yKey = "y", empty = "No data points available." }) {
  if (!points?.length) return <EmptyState text={empty} />;
  const xValues = points.map((p) => Number(p[xKey])).filter(Number.isFinite);
  const yValues = points.map((p) => Number(p[yKey])).filter(Number.isFinite);
  if (!xValues.length || !yValues.length) return <EmptyState text={empty} />;

  const xMax = Math.max(...xValues);
  const xMin = Math.min(...xValues);
  const yMax = Math.max(...yValues);
  const yMin = Math.min(...yValues);

  const xSpread = xMax - xMin || 1;
  const ySpread = yMax - yMin || 1;

  return (
    <div className="chart-card">
      <svg viewBox="0 0 100 100" className="scatter-chart">
        {points.map((p, i) => {
          const cx = ((Number(p[xKey]) - xMin) / xSpread) * 100;
          const cy = 100 - ((Number(p[yKey]) - yMin) / ySpread) * 100;
          return <circle key={i} cx={cx} cy={cy} r="1.5" className="scatter-chart__dot" />;
        })}
      </svg>
    </div>
  );
}

export function Heatmap({ data = [], xLabels = [], yLabels = [], empty = "No heatmap data available." }) {
  if (!data?.length || !xLabels.length || !yLabels.length) return <EmptyState text={empty} />;

  const allValues = data.flat().filter(Number.isFinite);
  const max = Math.max(...allValues, 1);
  const min = Math.min(...allValues, -1);
  const absMax = Math.max(Math.abs(max), Math.abs(min));

  return (
    <div className="heatmap">
      <div className="heatmap__grid" style={{ gridTemplateColumns: `auto repeat(${xLabels.length}, 1fr)` }}>
        <div />
        {xLabels.map((label) => (
          <div key={label} className="heatmap__label heatmap__label--x">
            {label}
          </div>
        ))}
        {yLabels.map((yLabel, yIndex) => (
          <React.Fragment key={yLabel}>
            <div className="heatmap__label heatmap__label--y">{yLabel}</div>
            {data[yIndex].map((val, xIndex) => {
              const intensity = Math.abs(val) / absMax;
              const color = val >= 0 ? `rgba(63, 181, 255, ${intensity})` : `rgba(255, 63, 63, ${intensity})`;
              return (
                <div
                  key={`${yIndex}-${xIndex}`}
                  className="heatmap__cell"
                  style={{ backgroundColor: color }}
                  title={`${yLabel} x ${xLabels[xIndex]}: ${val}`}
                />
              );
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

export function TimelineList({
  items,
  titleKey = "title",
  detailKey = "detail",
  metaKey = "meta",
  empty = "No timeline yet.",
}) {
  if (!items?.length) return <EmptyState text={empty} />;
  return (
    <div className="timeline">
      {items.map((item, index) => (
        <article key={`${item?.[titleKey] || "time"}-${index}`} className="timeline__item">
          <span className="timeline__dot" />
          <div>
            <strong>{item?.[titleKey] || "Untitled"}</strong>
            <p>{item?.[detailKey] || "—"}</p>
            {item?.[metaKey] ? <small>{item[metaKey]}</small> : null}
          </div>
        </article>
      ))}
    </div>
  );
}

export function KeyValueList({ items }) {
  if (!items?.length) return <EmptyState text="No key metrics available." />;
  return (
    <div className="kv-list">
      {items.map((item, index) => (
        <div key={`${item.label}-${index}`} className="kv-list__row">
          <span>{item.label}</span>
          <strong>{renderDisplayValue(item.value)}</strong>
        </div>
      ))}
    </div>
  );
}

export function SegmentedControl({ options, value, onChange, label }) {
  const labelId = React.useId();
  return (
    <div className="field">
      {label && <span id={labelId}>{label}</span>}
      <div className="segmented-control" role="group" aria-labelledby={label ? labelId : undefined}>
        {options.map((opt) => (
          <button
            key={opt}
            type="button"
            className={`button ${value === opt ? "button--secondary" : "button--ghost"}`}
            onClick={() => onChange(opt)}
            aria-pressed={value === opt}
            style={{ borderRadius: "12px", minHeight: "38px" }}
          >
            {opt}
          </button>
        ))}
      </div>
    </div>
  );
}

export function Checkbox({ label, checked, onChange, help }) {
  const helpId = React.useId();
  return (
    <label className="checkbox">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        aria-describedby={help ? helpId : undefined}
      />
      <span className="checkbox__copy">
        <span>{label}</span>
        {help ? (
          <span id={helpId} className="checkbox__helper">
            {help}
          </span>
        ) : null}
      </span>
    </label>
  );
}

export function Spinner({ size = "md", label }) {
  return (
    <div className={`spinner-shell spinner-shell--${size}`}>
      <div className="spinner" />
      {label && <span className="spinner-label">{label}</span>}
    </div>
  );
}

export function NeuralPulse({ confidence = 85, activity = 0.5 }) {
  const safeConfidence = Number.isFinite(Number(confidence)) ? Math.max(0, Math.min(99, Number(confidence))) : 0;
  const safeActivity = Number.isFinite(Number(activity)) ? Math.max(0, Math.min(1, Number(activity))) : 0.5;

  return (
    <div className="neural-pulse-container">
      <div className="pulse-ring" style={{ animationDelay: "0s" }} />
      <div className="pulse-ring" style={{ animationDelay: "1s" }} />
      <div className="pulse-ring" style={{ animationDelay: "2s" }} />
      <div className="pulse-circle" style={{ opacity: safeActivity }} />
      <svg className="absolute-full" viewBox="0 0 200 100" preserveAspectRatio="none">
        <path className="fluid-wave" d={`M0 50 Q 50 ${50 - safeConfidence / 4} 100 50 T 200 50 V 100 H 0 Z`} />
      </svg>
      <div className="absolute-center stack text-center">
        <span className="tiny-eyebrow">Neural Vitality</span>
        <strong className="massive">{safeConfidence}%</strong>
      </div>
    </div>
  );
}

export function LineageTree({ nodes = [] }) {
  if (!nodes.length) return <EmptyState text="No lineage data available." />;

  return (
    <div className="lineage-flow">
      {nodes.map((node, i) => (
        <div key={node.id} className="lineage-step">
          <div className="lineage-marker">
            <div className={`lineage-dot ${node.is_job ? "lineage-dot--job" : "lineage-dot--data"}`} />
            {i < nodes.length - 1 && <div className="lineage-line" />}
          </div>
          <div className="lineage-content">
            <div className="lineage-card">
              <div className="split">
                <span className="tiny-eyebrow accent-text">{node.type}</span>
                <span className="tiny" style={{ color: "var(--muted)" }}>
                  {node.created_at ? new Date(node.created_at).toLocaleDateString() : ""}
                </span>
              </div>
              <strong className="lineage-card__title">{node.label}</strong>
              <p className="tiny lineage-card__detail">{node.detail}</p>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
