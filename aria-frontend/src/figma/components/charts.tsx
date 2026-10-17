import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  ResponsiveContainer, ComposedChart, LineChart, Line, AreaChart, Area,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Brush,
  ScatterChart, Scatter, PieChart, Pie, Cell,
} from "recharts";
import { fmtBiz } from "../lib/format";
import { colorForName, CATEGORY_COLORS } from "../lib/charts";

/* =====================================================================
   Shared chart library (s.txt §§7–10, §17): ONE reusable pattern.
   - selectChart: intent + data shape -> chart type (never hardcoded per page)
   - ChartFrame: business title + explanation + empty/loading/error states
   - BizTooltip: Period / Metric / Value labels, human-scale numbers
   - BusinessMetric, ConfidenceIndicator: compact consistent facts
   - ToggleLegend: click series on/off (recharts `hide` wiring)
   ===================================================================== */

export type ChartIntent =
  | "trend"
  | "forecast"
  | "comparison"
  | "ranking"
  | "composition"
  | "relationship"
  | "distribution";

export interface DataShape {
  points: number;
  series: number;
  hasTime: boolean;
  categories: number;
}

export type ChartType =
  | "line"
  | "area"
  | "bar"
  | "hbar"
  | "stacked"
  | "pie"
  | "scatter"
  | "histogram"
  | "table";

export function selectChart(
  intent: ChartIntent,
  shape: DataShape,
): { type: ChartType; reason: string } {
  if (shape.points < 2) {
    return { type: "table", reason: "too few points to draw — table instead" };
  }
  switch (intent) {
    case "forecast":
      return { type: "line", reason: "history then forecast read left to right" };
    case "trend":
      return shape.series > 1
        ? { type: "line", reason: "several series over time" }
        : { type: "line", reason: "movement over time" };
    case "relationship":
      return shape.series >= 1 && !shape.hasTime
        ? { type: "scatter", reason: "two numbers against each other" }
        : { type: "line", reason: "relationship along time" };
    case "distribution":
      return { type: "histogram", reason: "spread of one number" };
    case "composition":
      return shape.categories <= 6
        ? { type: "pie", reason: "few parts of a whole" }
        : { type: "stacked", reason: "too many parts for a pie" };
    case "ranking":
      return shape.categories > 8
        ? { type: "hbar", reason: "many names read best sideways" }
        : { type: "bar", reason: "ranked comparison" };
    case "comparison":
    default:
      return shape.categories > 8
        ? { type: "hbar", reason: "many names read best sideways" }
        : { type: "bar", reason: "categories side by side" };
  }
}

/* ---------- human-scale numbers live in lib/format (single source) ---------- */

/** Short period labels: 2026-08-14 -> Aug 14, 2026-08 -> Aug 2026. */
export function fmtPeriod(label: string): string {
  const m = /^(\d{4})-(\d{2})(?:-(\d{2}))?/.exec(String(label || ""));
  if (!m) return String(label || "");
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const mon = months[Number(m[2]) - 1] || m[2];
  return m[3] ? `${mon} ${Number(m[3])}, ${m[1]}` : `${mon} ${m[1]}`;
}

/* ---------- frame: title + explanation + states (§10) ---------- */

export function ChartFrame({
  title,
  explanation,
  empty,
  emptyText = "Not enough data to draw this chart.",
  loading,
  loadingText = "Preparing chart…",
  error,
  height = 260,
  children,
}: {
  title: string;
  explanation?: string;
  empty?: boolean;
  emptyText?: string;
  loading?: boolean;
  loadingText?: string;
  error?: string;
  height?: number;
  children: ReactNode;
}) {
  return (
    <div>
      <p className="text-[13px] font-semibold text-foreground">{title}</p>
      {explanation && (
        <p className="text-[11px] text-muted-foreground mt-0.5">{explanation}</p>
      )}
      <div className="mt-2" style={{ height }}>
        {loading ? (
          <div className="h-full flex items-center justify-center text-xs text-muted-foreground">
            {loadingText}
          </div>
        ) : error ? (
          <div className="h-full flex items-center justify-center text-xs text-muted-foreground px-4 text-center">
            Chart unavailable ({error}). The written explanation above still stands.
          </div>
        ) : empty ? (
          <div className="h-full flex items-center justify-center text-xs text-muted-foreground px-4 text-center">
            {emptyText}
          </div>
        ) : (
          children
        )}
      </div>
    </div>
  );
}

/* ---------- business tooltip: Period / Metric / Value (§7) ---------- */

export function BizTooltip({
  active,
  payload,
  label,
  periodLabel,
  metricNames,
}: {
  active?: boolean;
  payload?: any[];
  label?: string;
  periodLabel?: string;
  metricNames?: Record<string, string>;
}) {
  if (!active || !payload?.length) return null;
  const humanizeKey = (key: string): string => {
    const raw = metricNames?.[key];
    const s = String(raw ?? key ?? "").trim();
    if (!s) return "";
    const spaced = s
      .replace(/[_-]+/g, " ")
      .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
      .replace(/\s+/g, " ")
      .trim();
    return spaced.charAt(0).toUpperCase() + spaced.slice(1);
  };
  const fmtTooltipNumber = (n: number): string => {
    if (!isFinite(n)) return "—";
    const a = Math.abs(n);
    if (a >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
    if (a >= 100) return Math.round(n).toLocaleString();
    if (a >= 1) return String(Math.round(n * 100) / 100);
    if (a > 0) return String(Math.round(n * 1000) / 1000);
    return "0";
  };
  return (
    <div
      className="rounded-lg border border-border px-3 py-2 shadow-md"
      style={{ background: "var(--card)" }}
    >
      {(periodLabel || label || (payload[0] as any)?.name) && (
        <p className="text-[11px] font-bold text-foreground mb-1">
          {fmtPeriod(periodLabel ?? String(label ?? (payload[0] as any)?.name ?? ""))}
        </p>
      )}
      {payload.filter((p: any) => {
        const k = String(p.dataKey ?? p.name ?? "").toLowerCase();
        return k !== "bandlower" && k !== "bandupper" && k !== "expected";
      }).map((p: any, i: number) => {
        const key = String(p.dataKey ?? p.name ?? "");
        const raw = p.value;
        const val =
          typeof raw === "number"
            ? fmtTooltipNumber(raw)
            : typeof raw === "boolean"
              ? (raw ? "Yes" : "No")
              : String(raw ?? "—").slice(0, 60);
        return (
          <p key={i} className="text-[11px] text-muted-foreground">
            <span
              className="inline-block w-2 h-2 rounded-full mr-1.5"
              style={{ background: p.color || p.payload?.fill || "var(--primary)" }}
            />
            {humanizeKey(key)}:{" "}
            <span className="text-foreground font-semibold">{val}</span>
          </p>
        );
      })}
    </div>
  );
}

export const tooltipStyle = {
  background: "var(--card)",
  border: "1px solid var(--border)",
  borderRadius: 10,
  fontSize: 12,
};

/* ---------- compact facts ---------- */

export function BusinessMetric({
  label,
  value,
  sub,
}: {
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <div className="rounded-lg border border-border px-2.5 py-2">
      <p className="text-[10px] text-muted-foreground uppercase tracking-wide">{label}</p>
      <p className="text-sm font-bold text-foreground mt-0.5">{value}</p>
      {sub && <p className="text-[10px] text-muted-foreground mt-0.5">{sub}</p>}
    </div>
  );
}

export function ConfidenceIndicator({
  level,
}: {
  level: string;
}) {
  const l = (level || "").toLowerCase();
  const dots = l === "high" ? 3 : l === "moderate" ? 2 : l === "low" ? 1 : 0;
  const tone =
    l === "high" ? "text-positive" : l === "moderate" ? "text-primary" : "text-warning";
  const label = l ? l[0].toUpperCase() + l.slice(1) : "Unknown";
  return (
    <span className="inline-flex items-center gap-1.5" title={`Confidence: ${label}`}>
      <span className="flex items-center gap-0.5" aria-hidden>
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className={`w-1.5 h-1.5 rounded-full ${i < dots ? "bg-positive" : "bg-border"}`}
          />
        ))}
      </span>
      <span className={`text-[10px] font-medium ${tone}`}>{label}</span>
    </span>
  );
}

/* ---------- series toggle (§7: toggle relevant series) ---------- */

export function useSeriesToggle(series: string[]) {
  const [hidden, setHidden] = useState<string[]>([]);
  return {
    hidden,
    toggle: (s: string) =>
      setHidden((h) => (h.includes(s) ? h.filter((x) => x !== s) : [...h, s])),
    isHidden: (s: string) => hidden.includes(s),
    reset: () => setHidden([]),
    count: series.length,
  };
}

export function ToggleLegend({
  items,
  isHidden,
  onToggle,
  colors,
}: {
  items: string[];
  isHidden: (s: string) => boolean;
  onToggle: (s: string) => void;
  colors: Record<string, string>;
}) {
  if (items.length < 2) return null;
  return (
    <div className="flex flex-wrap gap-3 mt-1">
      {items.map((s) => {
        const off = isHidden(s);
        return (
          <button
            key={s}
            onClick={() => onToggle(s)}
            className="inline-flex items-center gap-1.5 text-[11px] text-muted-foreground hover:text-foreground transition-colors"
            aria-pressed={!off}
            title={off ? `Show ${s}` : `Hide ${s}`}
          >
            <span
              className="inline-block w-4 border-t-2"
              style={{
                borderColor: off ? "var(--border)" : colors[s] || "var(--primary)",
              }}
            />
            <span className={off ? "line-through opacity-60" : ""}>{s}</span>
          </button>
        );
      })}
    </div>
  );
}

/* =====================================================================
   n.txt-8/9 + P2: controlled insight chart spec (frontend side).
   Backend is the only authority: {type,x,y,series,title,interactive,data}.
   This module VALIDATES the spec before render and renders spec-only —
   arbitrary LLM frontend code is never accepted or executed.
   Interactivity (recharts): hover tooltips, Brush zoom/pan/time-range,
   clickable legend toggle, click drill-down, resize via ResponsiveContainer,
   data-point inspection, fullscreen expand.
   ===================================================================== */

export const INSIGHT_CHART_VOCAB = [
  "line", "bar", "area", "scatter", "pie", "donut", "histogram", "heatmap",
] as const;

export type InsightChartType = (typeof INSIGHT_CHART_VOCAB)[number];

export interface InsightChartSpec {
  type: InsightChartType;
  x: string | null;
  y: string | null;
  series: string[];
  title: string;
  interactive: boolean;
  data: Record<string, any>[];
  insufficient?: boolean;
  unsupported?: boolean;
}

const _ALIASES: Record<string, InsightChartType> = {
  leaderboard: "bar",
  share: "pie",
  pie_share: "pie",
  doughnut: "donut",
  step_line: "line",
};

/** Validate a backend viz payload. Returns null when it must not render
 *  (missing/unsupported/insufficient) so callers show a graceful state. */
export function normalizeInsightViz(viz: any, fallbackTitle = ""): InsightChartSpec | null {
  if (!viz || typeof viz !== "object") return null;
  if (viz.unsupported) return null;
  let t = String(viz.type || "").toLowerCase();
  if (_ALIASES[t]) t = _ALIASES[t];
  if (!(INSIGHT_CHART_VOCAB as readonly string[]).includes(t)) return null;
  const data = Array.isArray(viz.data)
    ? viz.data.filter((r: any) => r && typeof r === "object").slice(0, 500)
    : [];
  if (!data.length) return null;
  if (viz.insufficient && data.length < 2 && (t === "line" || t === "area" || t === "scatter")) return null;
  const x = typeof viz.x === "string" && viz.x ? viz.x : null;
  const y = typeof viz.y === "string" && viz.y ? viz.y : null;
  if (t !== "heatmap" && (!x || !y)) {
    // Infer from common detector shapes so older payloads still validate.
    const keys = Object.keys(data[0] || {});
    let ix: string | null = null;
    let iy: string | null = null;
    if (keys.includes("name") && keys.includes("value")) { ix = "name"; iy = "value"; }
    else if (keys.includes("period") && keys.includes("value")) { ix = "period"; iy = "value"; }
    else if (keys.includes("table1") && keys.includes("table2")) { ix = "table1"; iy = "table2"; }
    else if (keys.includes("x") && keys.includes("y")) { ix = "x"; iy = "y"; }
    else if (keys.length >= 2) { ix = keys[0]; iy = keys[1]; }
    if (!ix || !iy) return null;
    return {
      type: t as InsightChartType, x: ix, y: iy,
      series: Array.isArray(viz.series) ? viz.series.filter((s: any) => typeof s === "string").slice(0, 8) : [],
      title: String(viz.title || fallbackTitle || ""),
      interactive: viz.interactive !== false, data,
    };
  }
  return {
    type: t as InsightChartType, x, y,
    series: Array.isArray(viz.series) ? viz.series.filter((s: any) => typeof s === "string").slice(0, 8) : [],
    title: String(viz.title || fallbackTitle || ""),
    interactive: viz.interactive !== false, data,
  };
}

export interface DrillSelection {
  label: string;
  value: number | string | null;
  row: Record<string, any>;
}

/** Spec-only interactive chart. All interaction is data-aware and driven by
 *  the validated spec — click a mark to drill down via onDrillDown. */
export function InsightChart({
  spec,
  onDrillDown,
  height = 260,
}: {
  spec: InsightChartSpec;
  onDrillDown?: (sel: DrillSelection) => void;
  height?: number;
}) {
  const [hidden, setHidden] = useState<string[]>([]);
  const [selected, setSelected] = useState<DrillSelection | null>(null);
  const [expanded, setExpanded] = useState(false);
  const toggle = (s: string) =>
    setHidden((h) => (h.includes(s) ? h.filter((x) => x !== s) : [...h, s]));

  const seriesKeys = useMemo(() => {
    if (spec.series?.length) return spec.series.filter((s) => !hidden.includes(s));
    return spec.y ? [spec.y] : [];
  }, [spec.series, spec.y, hidden]);

  const pick = (row: any): DrillSelection => {
    const label = String(row?.[spec.x as string] ?? row?.name ?? row?.period ?? row?.label ?? "");
    const raw = spec.y ? row?.[spec.y] : null;
    return { label, value: typeof raw === "number" ? raw : (raw ?? null), row };
  };
  const handlePick = (row: any) => {
    if (!row) return;
    const sel = pick(row);
    setSelected(sel);
    onDrillDown?.(sel);
  };

  const legendItems = spec.series?.length ? spec.series : [];
  const colors: Record<string, string> = {};
  [...legendItems, spec.y || ""].forEach((s) => { if (s) colors[s] = colorForName(s); });

  const body = (h: number) => {
    if (spec.type === "pie" || spec.type === "donut") {
      return (
        <ResponsiveContainer width="100%" height={h}>
          <PieChart>
            <Tooltip content={<BizTooltip />} />
            <Pie
              data={spec.data}
              dataKey={spec.y as string}
              nameKey={spec.x as string}
              cx="50%" cy="50%"
              outerRadius={Math.max(50, h / 2 - 30)}
              innerRadius={spec.type === "donut" ? Math.max(30, h / 2 - 60) : 0}
              label={false}
              labelLine={false}
              onClick={(d: any) => handlePick(d?.payload || d)}
            >
              {spec.data.map((row: any, i: number) => (
                <Cell key={i} fill={colorForName(String(row?.[spec.x as string] ?? i))} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
      );
    }
    if (spec.type === "scatter") {
      const pts = spec.data
        .map((r) => ({ ...r, __x: Number(r[spec.x as string]), __y: Number(r[spec.y as string]) }))
        .filter((r) => !isNaN(r.__x) && !isNaN(r.__y));
      if (pts.length < 2) return null;
      return (
        <ResponsiveContainer width="100%" height={h}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="__x" name={spec.x || ""} tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} />
            <YAxis dataKey="__y" name={spec.y || ""} tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} width={56} />
            <Tooltip content={<BizTooltip />} cursor={{ strokeDasharray: "3 3" }} />
            <Scatter data={pts} fill="#2F6FAD" fillOpacity={0.65} onClick={(d: any) => handlePick(d?.payload || d)}>
              {pts.map((pt: any, i: number) => (
                <Cell
                  key={i}
                  fill={pt?.anomaly === true || pt?.outlier === true ? "#E11D48" : "#2F6FAD"}
                />
              ))}
            </Scatter>
            <Brush dataKey="__x" height={22} travellerWidth={10} />
          </ScatterChart>
        </ResponsiveContainer>
      );
    }
    if (spec.type === "heatmap") {
      const xs = Array.from(new Set(spec.data.map((r) => String(r.x ?? r[spec.x as string] ?? "")))).slice(0, 24);
      const ys = Array.from(new Set(spec.data.map((r) => String(r.y ?? r[spec.y as string] ?? "")))).slice(0, 12);
      const lut = new Map(spec.data.map((r) => [`${r.x ?? r[spec.x as string]}|${r.y ?? r[spec.y as string]}`, Number(r.value ?? 0)]));
      const vals = spec.data.map((r) => Number(r.value ?? 0)).filter((v) => !isNaN(v));
      const mx = Math.max(1, ...vals);
      return (
        <div style={{ height: h, overflow: "auto" }}>
          <div style={{ display: "grid", gridTemplateColumns: `80px repeat(${xs.length}, minmax(34px, 1fr))`, gap: 2, minWidth: 320 }}>
            <span />
            {xs.map((x) => (<span key={x} style={{ fontSize: 9, opacity: 0.6, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{x}</span>))}
            {ys.map((y) => (
              <>
                <span key={"r" + y} style={{ fontSize: 9, opacity: 0.6, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{y}</span>
                {xs.map((x) => {
                  const v = lut.get(`${x}|${y}`) ?? 0;
                  const a = 0.08 + (0.85 * v) / mx;
                  return (
                    <button key={`${x}|${y}`} title={`${x} · ${y}: ${v}`}
                      onClick={() => handlePick({ [spec.x as string]: x, [spec.y as string]: y, value: v })}
                      style={{ background: `rgba(47,111,173,${a.toFixed(2)})`, borderRadius: 3, minHeight: 22, fontSize: 9, color: a > 0.5 ? "#fff" : "inherit" }}>
                      {v || ""}
                    </button>
                  );
                })}
              </>
            ))}
          </div>
        </div>
      );
    }
    // line / area / bar / histogram — Brush gives zoom + pan + time-range.
    // Backend anomaly flags (conventional keys only: anomaly, expected,
    // bandLower, bandUpper, outlier) are detected at runtime — no hardcoded
    // field assumptions beyond these keys, no invented values.
    const rows: Record<string, any>[] = Array.isArray(spec.data) ? spec.data : [];
    const hasExpected = rows.some((r) => typeof r?.expected === "number" && !isNaN(r.expected));
    const hasBandLower = rows.some((r) => typeof r?.bandLower === "number" && !isNaN(r.bandLower));
    const hasBandUpper = rows.some((r) => typeof r?.bandUpper === "number" && !isNaN(r.bandUpper));
    const isFlaggedRow = (r: any): boolean => r?.anomaly === true || r?.outlier === true;
    // Per-point anomaly coloring for line/area: flagged rows draw rose,
    // everything else keeps the series color. White/navy/indigo theme kept.
    const anomalyDot =
      (baseColor: string) =>
      (p: any): ReactNode => {
        const flagged = p?.payload?.anomaly === true;
        if (p?.cx == null || p?.cy == null) return <g />;
        if (flagged) {
          return <circle cx={p.cx} cy={p.cy} r={5} fill="#E11D48" stroke="#ffffff" strokeWidth={1.5} />;
        }
        return <circle cx={p.cx} cy={p.cy} r={3} fill={baseColor} stroke="none" />;
      };
    const Chart: any = spec.type === "line" ? LineChart : spec.type === "area" ? AreaChart : BarChart;
    return (
      <ResponsiveContainer width="100%" height={h}>
        <Chart data={spec.data} onClick={(st: any) => { if (st?.activePayload?.[0]?.payload) handlePick(st.activePayload[0].payload); }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
          <XAxis dataKey={spec.x as string} tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} minTickGap={24} />
          <YAxis tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} width={56} />
          <Tooltip content={<BizTooltip />} />
          <Legend wrapperStyle={{ fontSize: 11 }} onClick={(e: any) => { const v = e?.value; if (typeof v === "string" && legendItems.includes(v)) toggle(v); }} />
          {seriesKeys.map((s, i) =>
            spec.type === "line" ? (
              <Line key={s} type="monotone" dataKey={s} stroke={colors[s] || CATEGORY_COLORS[i % CATEGORY_COLORS.length]} strokeWidth={2} dot={anomalyDot(colors[s] || CATEGORY_COLORS[i % CATEGORY_COLORS.length])} activeDot={{ r: 5, onClick: (_e: any, p: any) => handlePick(p?.payload) }} />
            ) : spec.type === "area" ? (
              <Area key={s} type="monotone" dataKey={s} stroke={colors[s] || "#2F6FAD"} fill={colors[s] || "#2F6FAD"} fillOpacity={0.16} strokeWidth={2} dot={anomalyDot(colors[s] || "#2F6FAD")} activeDot={{ r: 4, onClick: (_e: any, p: any) => handlePick(p?.payload) }} />
            ) : (
              <Bar key={s} dataKey={s} fill={spec.series?.length ? colors[s] || CATEGORY_COLORS[i % CATEGORY_COLORS.length] : undefined}
                radius={[4, 4, 0, 0]}
                onClick={(d: any) => handlePick(d?.payload || d)}
              >
                {spec.data.map((row: any, j: number) => {
                  if (isFlaggedRow(row)) {
                    return <Cell key={j} fill="#E11D48" />;
                  }
                  if (!spec.series?.length) {
                    return <Cell key={j} fill={colorForName(String(row?.[spec.x as string] ?? j))} />;
                  }
                  return <Cell key={j} fill={colors[s] || CATEGORY_COLORS[i % CATEGORY_COLORS.length]} />;
                })}
              </Bar>
            )
          )}
          {hasExpected && (
            <Line type="monotone" dataKey="expected" stroke="#1E3A8A" strokeDasharray="6 4" strokeWidth={1.5} dot={false} activeDot={{ r: 3 }} name="Expected" />
          )}
          {hasBandLower && (
            <Line type="monotone" dataKey="bandLower" stroke="#6366F1" strokeDasharray="4 4" strokeWidth={1} dot={false} activeDot={{ r: 3 }} name="Lower bound" />
          )}
          {hasBandUpper && (
            <Line type="monotone" dataKey="bandUpper" stroke="#6366F1" strokeDasharray="4 4" strokeWidth={1} dot={false} activeDot={{ r: 3 }} name="Upper bound" />
          )}
          <Brush dataKey={spec.x as string} height={22} travellerWidth={10} />
        </Chart>
      </ResponsiveContainer>
    );
  };

  return (
    <div>
      <div className="flex items-center gap-2 mb-1">
        <div className="flex-1 min-w-0">
          {legendItems.length > 0 && (
            <ToggleLegend items={legendItems} isHidden={(s) => hidden.includes(s)} onToggle={toggle} colors={colors} />
          )}
        </div>
        <button onClick={() => setExpanded(true)}
          className="text-[11px] px-2 py-1 rounded-md border border-border text-muted-foreground hover:text-foreground hover:border-primary/50 transition-colors"
          title="Expand chart" aria-label="Expand chart">⛶ Expand</button>
      </div>
      {/* Keyboard access for drill-down: the chart region is focusable and
         Enter/Space inspects the first point (same pick as click). Legend
         toggles + Expand above are native buttons; a data-table alternative
         lives beside each evidence chart on its host page. */}
      <div
        tabIndex={0}
        role="group"
        aria-label={`${spec.title || "Evidence chart"}: ${spec.type} chart with ${spec.data.length} points. Press Enter to inspect the first point.`}
        onKeyDown={(e) => {
          if ((e.key === "Enter" || e.key === " ") && spec.data.length > 0) {
            e.preventDefault();
            handlePick(spec.data[0]);
          }
        }}
        className="rounded-lg focus:outline-none focus-visible:ring-2 focus-visible:ring-primary/50"
      >
        {body(height)}
      </div>
      {selected && (
        <div className="mt-2 rounded-lg border border-border bg-background/60 px-3 py-2 text-[11px]">
          <span className="font-semibold text-foreground">{selected.label || "Selected point"}</span>
          <span className="text-muted-foreground"> · </span>
          <span className="text-foreground font-semibold">
            {typeof selected.value === "number" ? fmtBiz(selected.value) : String(selected.value ?? "—")}
          </span>
          <span className="text-muted-foreground"> — click another mark to inspect, or drill down below.</span>
        </div>
      )}
      {expanded && (
        <div className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-4" role="dialog" aria-modal="true" aria-label="Expanded chart">
          <div className="bg-card border border-border rounded-xl p-4 w-full max-w-4xl">
            <div className="flex items-center justify-between mb-2">
              <p className="text-sm font-semibold text-foreground truncate">{spec.title || "Evidence chart"}</p>
              <button onClick={() => setExpanded(false)}
                className="text-xs px-2.5 py-1.5 rounded-lg border border-border text-foreground hover:border-primary/50"
                aria-label="Close expanded chart">Close ✕</button>
            </div>
            {body(440)}
          </div>
        </div>
      )}
    </div>
  );
}

/* =====================================================================
   Dashboard visualizations: controlled spec validation (backend-driven).
   The backend is the only authority: {type,title,data,dataKey,xKey,format}.
   These guards VALIDATE the spec before render — malformed payloads never
   reach recharts; callers show a graceful empty state instead.
   ===================================================================== */

const DASHBOARD_CHART_VOCAB = [
  "line", "area", "bar", "hbar", "scatter", "pie", "donut", "table",
] as const;

/** Validate one backend dashboard visualization. False => must not render. */
export function isValidDashboardChart(c: any): boolean {
  if (!c || typeof c !== "object") return false;
  const type = String(c.type || "").toLowerCase();
  if (!(DASHBOARD_CHART_VOCAB as readonly string[]).includes(type)) return false;
  if (!Array.isArray(c.data) || c.data.length === 0) return false;
  const xk = typeof c.xKey === "string" && c.xKey ? c.xKey : null;
  const yk = typeof c.dataKey === "string" && c.dataKey ? c.dataKey : null;
  if (!xk || !yk) return false;
  // At least one row must carry both keys with a real numeric value.
  // Numeric strings ("123.4") count — CSV-backed payloads often encode
  // numbers as text. Matches api.js isNumericColumn semantics.
  const asNum = (v: any): number | null => {
    if (typeof v === "number" && !isNaN(v)) return v;
    if (typeof v === "string" && v.trim() !== "" && !isNaN(Number(v))) return Number(v);
    return null;
  };
  return c.data.some(
    (r: any) =>
      r &&
      typeof r === "object" &&
      r[xk] !== undefined &&
      r[xk] !== null &&
      asNum(r[yk]) !== null,
  );
}

/** Filter + cap backend visualizations to renderable specs only. */
export function validateDashboardCharts(charts: any, cap = 4): any[] {
  if (!Array.isArray(charts)) return [];
  return charts.filter(isValidDashboardChart).slice(0, cap);
}
