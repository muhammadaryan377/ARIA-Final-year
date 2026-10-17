import CsvStatisticalEvidence from "../components/CsvStatisticalEvidence";
import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { ArrowRight, ArrowUpRight, ArrowDownRight, Zap, Search, AlertTriangle, ChevronRight, RefreshCw, Database, DollarSign, Store, CalendarDays, Percent, TrendingUp, Users, ShoppingCart, RotateCcw, X, Sparkles, MessageSquare, Send, Lightbulb, Shield, BarChart3, GitBranch, Sun, FileText } from "lucide-react";
import { AreaChart, Area, Line, BarChart, Bar, Cell, PieChart, Pie, ResponsiveContainer, Tooltip, CartesianGrid, XAxis, YAxis, Brush, ReferenceDot, ScatterChart, Scatter, ZAxis } from "recharts";
import type { Page } from "../App";
import { useTheme } from "../App";
import { TimeRangeSelector } from "../components/TimeRangeSelector";
import { colorForName } from "../lib/charts";
import { fmtBiz } from "../lib/format";
import { fmtPeriod, validateDashboardCharts } from "../components/charts";
import { apiFetch } from "../../services/api";

interface DashboardProps {
  setPage: (p: Page) => void;
  onAsk?: (question: string) => void;
  user?: { email?: string } | null;
  connectionStatus?: string;
  dbName?: string;
  insightResult?: any;
  analysisResult?: any;
  connectionError?: string;
}

// Fallback suggestions when the backend has none yet (domain-neutral).
const fallbackExamples = [
  "What are the totals of the main measures?",
  "Which segments appear most frequently?",
  "Show the trend over time",
  "Find unusual changes in the data",
];

/* ---------- filter semantics (mirrors Insights.matchesFilter, generic) ---------- */
function matchesFilter(type: string, filter: string): boolean {
  if (filter === "all") return true;
  const t = (type || "").toLowerCase();
  switch (filter) {
    case "trends": return t.includes("trend") || t.includes("growth") || t.includes("decline");
    case "anomalies": return t.includes("anomal") || t.includes("outlier");
    case "opportunities": return t.includes("opportunit") || t.includes("growth");
    case "risks": return t.includes("risk") || t.includes("decline");
    default: return true;
  }
}

type Category = "trends" | "anomalies" | "performance" | "relationships" | "opportunities" | "risks";

function primaryCategory(type: string): Category {
  const t = (type || "").toLowerCase();
  if (t.includes("anomal") || t.includes("outlier")) return "anomalies";
  if (t.includes("opportunit")) return "opportunities";
  if (t.includes("risk")) return "risks";
  if (t.includes("correl") || t.includes("relation")) return "relationships";
  if (t.includes("growth") || t.includes("decline") || t.includes("perform")) return "performance";
  return "trends";
}

const CAT_META: Record<Category, { label: string; Icon: typeof TrendingUp; pill: string; iconWrap: string }> = {
  trends: { label: "Trend", Icon: TrendingUp, pill: "bg-violet-50 dark:bg-violet-500/10 text-violet-700 dark:text-violet-400 border-violet-200 dark:border-violet-500/25", iconWrap: "bg-violet-50 dark:bg-violet-500/10 text-violet-600 dark:text-violet-400" },
  anomalies: { label: "Anomaly", Icon: AlertTriangle, pill: "bg-blue-50 dark:bg-blue-500/10 text-blue-700 dark:text-blue-400 border-blue-200 dark:border-blue-500/25", iconWrap: "bg-blue-50 dark:bg-blue-500/10 text-blue-600 dark:text-blue-400" },
  performance: { label: "Performance", Icon: BarChart3, pill: "bg-sky-50 text-sky-700 border-sky-200", iconWrap: "bg-sky-50 text-sky-600" },
  relationships: { label: "Relationship", Icon: GitBranch, pill: "bg-indigo-50 dark:bg-indigo-500/10 text-indigo-700 dark:text-indigo-400 border-indigo-200 dark:border-indigo-500/25", iconWrap: "bg-indigo-50 dark:bg-indigo-500/10 text-indigo-600 dark:text-indigo-400" },
  opportunities: { label: "Opportunity", Icon: Lightbulb, pill: "bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-500/25", iconWrap: "bg-emerald-50 dark:bg-emerald-500/10 text-emerald-600 dark:text-emerald-400" },
  risks: { label: "Risk", Icon: Shield, pill: "bg-rose-50 dark:bg-rose-500/10 text-rose-700 dark:text-rose-400 border-rose-200 dark:border-rose-500/25", iconWrap: "bg-rose-50 dark:bg-rose-500/10 text-rose-600 dark:text-rose-400" },
};

/* ---------- timestamps: real values only, never synthesized ---------- */
function fmtWhen(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return String(iso).slice(0, 40);
  const now = new Date();
  const day = (x: Date) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const diffDays = Math.round((day(now) - day(d)) / 86400000);
  const time = d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  if (diffDays <= 0) return `Today, ${time}`;
  if (diffDays === 1) return `Yesterday, ${time}`;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function relTime(iso: string): string {
  if (!iso) return "";
  const t = new Date(iso).getTime();
  if (isNaN(t)) return "";
  const secs = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (secs < 45) return "just now";
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 30) return `${days}d ago`;
  return fmtWhen(iso);
}

const fmt = (n: number | null | undefined, prefix = "") => {
  if (n === null || n === undefined) return "—";
  const v = Number(n);
  if (!Number.isFinite(v)) return "—";
  return v >= 1_000_000
    ? `${prefix}${(v / 1_000_000).toFixed(2)}M`
    : v >= 1_000
    ? `${prefix}${(v / 1_000).toFixed(1)}K`
    : `${prefix}${v.toLocaleString()}`;
};

/** Exact value for tooltips/drill panels (aggregates, never raw rows). */
const fmtExact = (n: number | null | undefined, prefix = "") => {
  if (n === null || n === undefined || isNaN(Number(n))) return "—";
  return `${prefix}${Number(n).toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
};

const humanize = (s: string) =>
  String(s || "")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .trim();

/** Business-worded error copy: never leak raw fetch/stack text to the UI. */
function businessError(raw: string): string {
  const s = String(raw || "");
  if (!s) return "";
  if (/failed to fetch|network|load failed|request failed|fetch/i.test(s)) {
    return "ARIA couldn't load this analysis. Check your connection and try again.";
  }
  if (/session expired|sign in|unauthorized|401/i.test(s)) return s.slice(0, 200);
  if (/select\s.+\sfrom|traceback|exception|stack/i.test(s)) {
    return "ARIA couldn't load this analysis. Try again in a moment.";
  }
  return s.slice(0, 220);
}

/* =====================================================================
   Semantic KPI tints — presentation only, chosen dynamically by keyword
   match on the backend-provided metric name. Subtle tinted wash +
   colored icon container + top accent; text stays neutral. Fallback is
   a neutral slate when nothing matches. No numbers are invented here.
   ===================================================================== */

interface KpiTint {
  wash: string;
  iconBg: string;
  accent: string;
  chipBg: string;
}

const NEUTRAL_TINT: KpiTint = {
  wash: "rgba(100,116,139,0.07)",
  iconBg: "linear-gradient(135deg, #64748B, #475569)",
  accent: "#64748B",
  chipBg: "rgba(100,116,139,0.12)",
};

const TINT_RULES: { re: RegExp; tint: KpiTint }[] = [
  { re: /risk|anomal|fraud|loss|churn|return|null|missing|critical|warn|alert/i, tint: { wash: "rgba(225,29,72,0.07)", iconBg: "linear-gradient(135deg, #F43F5E, #BE123C)", accent: "#E11D48", chipBg: "rgba(225,29,72,0.12)" } },
  { re: /profit|margin|earn/i, tint: { wash: "rgba(5,150,105,0.08)", iconBg: "linear-gradient(135deg, #10B981, #047857)", accent: "#059669", chipBg: "rgba(5,150,105,0.12)" } },
  { re: /customer|client|user|member|subscriber|visitor/i, tint: { wash: "rgba(124,58,237,0.07)", iconBg: "linear-gradient(135deg, #8B5CF6, #5B21B6)", accent: "#7C3AED", chipBg: "rgba(124,58,237,0.12)" } },
  { re: /order|transaction|purchase|count|record|store|entit|product|item|unit/i, tint: { wash: "rgba(8,145,178,0.08)", iconBg: "linear-gradient(135deg, #06B6D4, #0E7490)", accent: "#0891B2", chipBg: "rgba(8,145,178,0.12)" } },
  { re: /growth|lift|change|increas|uplift|trend|rate|pct|percent/i, tint: { wash: "rgba(13,148,136,0.08)", iconBg: "linear-gradient(135deg, #14B8A6, #0F766E)", accent: "#0D9488", chipBg: "rgba(13,148,136,0.12)" } },
  { re: /revenue|sales|income|turnover|amount|total|value|\$|payment|spend/i, tint: { wash: "rgba(47,111,173,0.08)", iconBg: "linear-gradient(135deg, #3B82C4, #1E4E79)", accent: "#2F6FAD", chipBg: "rgba(47,111,173,0.12)" } },
];

function tintFor(kpi: any): KpiTint {
  const label = String(kpi?.label || "");
  const custom = typeof kpi?.color === "string" && /^#[0-9a-fA-F]{3,8}$/.test(kpi.color.trim())
    ? kpi.color.trim()
    : null;
  if (custom) {
    return { wash: `${custom}12`, iconBg: `linear-gradient(135deg, ${custom}, ${custom})`, accent: custom, chipBg: `${custom}1F` };
  }
  for (const rule of TINT_RULES) {
    if (rule.re.test(label)) return rule.tint;
  }
  return NEUTRAL_TINT;
}

function iconFor(kpi: any) {
  const icon = String(kpi?.icon || "").toLowerCase();
  const label = String(kpi?.label || "").toLowerCase();
  if (icon.includes("database") || label.includes("record")) return Database;
  if (label.includes("customer") || label.includes("client") || label.includes("user") || label.includes("member")) return Users;
  if (label.includes("order") || label.includes("transaction") || label.includes("purchase") || label.includes("product")) return ShoppingCart;
  if (icon.includes("bar") || label.includes("count") || label.includes("store") || label.includes("entit")) return Store;
  if (label.includes("date") || label.includes("range") || label.includes("period") || label.includes("calendar")) return CalendarDays;
  if (icon.includes("percent") || kpi?.format === "percent" || label.includes("lift") || label.includes("holiday") || label.includes("rate")) return Percent;
  if (kpi?.format === "currency" || label.includes("revenue") || label.includes("sale") || label.includes("$") || label.includes("profit")) return DollarSign;
  return TrendingUp;
}

/** Mount-only subtle motion: true when the OS asks for reduced motion. */
function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(
    () => typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches,
  );
  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    if (!mq) return;
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches);
    mq.addEventListener?.("change", onChange);
    return () => mq.removeEventListener?.("change", onChange);
  }, []);
  return reduced;
}

/** Number transition on update only: exact value on mount, eased glide on change. */
function useAnimatedNumber(target: number | null, duration = 550) {
  const reduced = usePrefersReducedMotion();
  const [display, setDisplay] = useState<number | null>(target);
  const prevRef = useRef<number | null>(target);
  const firstRef = useRef(true);
  useEffect(() => {
    if (target === null) { setDisplay(null); prevRef.current = null; return; }
    if (firstRef.current) {
      firstRef.current = false;
      prevRef.current = target;
      setDisplay(target);
      return;
    }
    const from = prevRef.current;
    prevRef.current = target;
    if (from === null || from === target || reduced) {
      setDisplay(target);
      return;
    }
    let raf = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      setDisplay(from + (target - from) * eased);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, duration, reduced]);
  return display;
}

/** Shared drill context: stored for handoff, then the caller navigates. */
function storeDrillContext(ctx: Record<string, any>) {
  try {
    localStorage.setItem("aria_dashboard_drill", JSON.stringify({ at: Date.now(), ...ctx }));
  } catch { /* private mode: navigation still works */ }
}

function MetricCard({ kpi, index = 0, onDrill, onAsk, onWhy }: { kpi: any; index?: number; onDrill: (kpi: any) => void; onAsk?: (q: string) => void; onWhy?: (metric: string) => void }) {
  const [draftMsg, setDraftMsg] = useState("");
  const tint = tintFor(kpi);
  const Icon = iconFor(kpi);
  const isText = kpi?.format === "text";
  const isPercent = kpi?.format === "percent";
  const prefix = kpi?.format === "currency" ? "$" : (typeof kpi?.prefix === "string" ? kpi.prefix : "");

  const numericTarget =
    !isText && !isPercent && kpi?.rawValue !== null && kpi?.rawValue !== undefined && !isNaN(Number(kpi.rawValue))
      ? Number(kpi.rawValue)
      : null;
  const animated = useAnimatedNumber(numericTarget);

  let displayValue: string;
  if (isText || isPercent) {
    displayValue = kpi?.value === null || kpi?.value === undefined || kpi?.value === "" ? "—" : String(kpi.value);
  } else if (kpi?.value === "N/A" || kpi?.rawValue === null || kpi?.rawValue === undefined) {
    displayValue = typeof kpi?.value === "string" && kpi.value ? kpi.value : "—";
    if (displayValue !== "N/A" && displayValue !== "—" && animated !== null) displayValue = fmt(animated, prefix);
    if (numericTarget === null && displayValue !== "N/A") displayValue = typeof kpi?.value === "string" ? kpi.value : "—";
  } else {
    displayValue = fmt(animated ?? numericTarget, prefix);
  }

  const changeNum = kpi?.change === null || kpi?.change === undefined || kpi?.change === "" ? null : Number(kpi.change);
  const showChange = changeNum !== null && Number.isFinite(changeNum) && changeNum !== 0 && !isText;
  const positive = (changeNum ?? 0) >= 0;
  const spark = Array.isArray(kpi.spark) && kpi.spark.length > 1
    ? kpi.spark.map((v: any) => ({ v: Number(v) || 0 }))
    : Array.isArray(kpi.trend) && kpi.trend.length > 1
      ? kpi.trend.map((v: any) => ({ v: Number(v?.value ?? v) || 0 }))
      : [];
  const gradId = `kpi-spark-${index}`;
  const periodText = /prev/i.test(String(kpi?.period || "")) ? "Compared with previous period" : (kpi?.period || "All time");

  return (
    <div
      onClick={() => onDrill(kpi)}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onDrill(kpi); } }}
      role="button"
      tabIndex={0}
      title={kpi?.reason ? `${kpi.label} — ${kpi.reason} Select for deeper analysis.` : `${kpi.label} — select for deeper analysis.`}
      className="relative overflow-hidden bg-card border border-border/80 rounded-2xl p-5 flex flex-col gap-3 text-left cursor-pointer shadow-[0_1px_2px_rgba(15,36,64,0.05)] hover:shadow-[0_10px_28px_rgba(47,111,173,0.14)] hover:border-blue-200 dark:hover:border-blue-500/25 hover:-translate-y-0.5 transition-all duration-200 min-w-0"
      style={{ animationDelay: `${Math.min(index, 5) * 60}ms` }}
    >
      <span
        aria-hidden
        className="absolute inset-x-0 top-0 h-1"
        style={{ background: `linear-gradient(90deg, ${tint.accent}, ${tint.accent}55)` }}
      />
      <div className="flex items-center gap-2.5 min-w-0">
        <span
          className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
          style={{ background: tint.chipBg, color: tint.accent }}
        >
          <Icon size={18} />
        </span>
        <p className="text-[13px] font-medium text-muted-foreground truncate">{kpi.label || "Metric"}</p>
      </div>
      <div className="flex items-end justify-between gap-3">
        <div className="min-w-0">
          <p className="text-[28px] leading-tight font-semibold text-foreground tracking-tight tabular-nums truncate">
            {displayValue}
          </p>
          {showChange ? (
            <p className={`flex items-center gap-1 text-[12px] font-semibold mt-1 ${positive ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
              {positive ? <ArrowUpRight size={13} /> : <ArrowDownRight size={13} />}
              {Math.abs(changeNum as number)}%
            </p>
          ) : (
            <p className="text-[11px] text-muted-foreground mt-1 truncate">{periodText}</p>
          )}
        </div>
        {spark.length > 0 && (
          <div className="w-24 h-11 flex-shrink-0" aria-hidden>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={spark} margin={{ top: 2, right: 0, bottom: 0, left: 0 }}>
                <defs>
                  <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={tint.accent} stopOpacity={0.28} />
                    <stop offset="95%" stopColor={tint.accent} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <Area
                  type="monotone" dataKey="v"
                  stroke={tint.accent}
                  strokeWidth={1.75} fill={`url(#${gradId})`} dot={false}
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
      {showChange && (
        <p className="text-[11px] text-muted-foreground truncate">{periodText}</p>
      )}
      {onAsk && (
        <button
          onClick={(e) => { e.stopPropagation(); onAsk(`What should I know about ${kpi?.label || "this metric"}?`); }}
          className="self-start text-[11px] font-semibold text-blue-600 dark:text-blue-400 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
          aria-label={`Ask ARIA about ${kpi?.label || "this metric"}`}
        >
          Ask about this →
        </button>
      )}
      {onWhy && (
        <button
          onClick={(e) => { e.stopPropagation(); onWhy(String(kpi?.label || "")); }}
          className="self-start text-[11px] font-medium text-muted-foreground hover:text-blue-700 dark:hover:text-blue-400 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded -mt-2"
          aria-label={`Why did ${kpi?.label || "this metric"} move?`}
        >
          Why did this move?
        </button>
      )}
      <button
        onClick={(e) => {
          e.stopPropagation();
          try {
            const key = "aria_report_draft";
            const title = String(kpi?.label || "Dashboard metric");
            const raw = localStorage.getItem(key);
            const cur = Array.isArray(JSON.parse(raw || "[]")) ? JSON.parse(raw || "[]") : [];
            const exists = cur.some((x: any) => String(x?.title || "") === title);
            if (!exists) {
              cur.push({ id: String(kpi?.id || ""), title });
              localStorage.setItem(key, JSON.stringify(cur));
              setDraftMsg("Added to the report draft.");
            } else {
              setDraftMsg("Already in the report draft.");
            }
          } catch { setDraftMsg("Couldn't update the report draft."); }
        }}
        className="self-start inline-flex items-center gap-1 text-[11px] font-medium text-muted-foreground hover:text-blue-700 dark:hover:text-blue-400 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded -mt-1"
        aria-label={`Add ${kpi?.label || "this metric"} to report`}
      >
        <FileText size={11} /> Add to report
      </button>
      {draftMsg && <p className="text-[11px] text-emerald-700 dark:text-emerald-400" role="status">{draftMsg}</p>}
    </div>
  );
}

/* =====================================================================
   Dynamic visualization selection (schema + data driven, never hardcoded).
   chart1: best time-series candidate; chart2: best candidate of a
   DIFFERENT kind, preferring a different measure. Fallback derives a
   second perspective from the first chart's own aggregates, or an
   intentional empty state naming the analytical limitation (never
   implying the connection failed). Titles generated from data names.
   ===================================================================== */

type VizKind = "time" | "category" | "ranking" | "composition" | "scatter" | "empty";

interface SelectedViz {
  key: string;
  kind: VizKind;
  chartType: string;
  title: string;
  subtitle: string;
  data: any[];
  xKey: string;
  yKey: string;
  format: string;
  measure: string;
  dim: string;
  grain: string;
  periodLabel: string;
  current: number | null;
  trendPct: number | null;
  peakLabel: string;
  peakValue: number | null;
  annotation: string;
  annotationFromBackend: boolean;
  emptyReason: string;
  sourceId: string;
}

const normMeasure = (s: string) => String(s || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();

function kindOfChart(c: any): "time" | "category" | "scatter" {
  const t = String(c?.type || "").toLowerCase();
  if (t === "scatter") return "scatter";
  const xk = String(c?.xKey || "");
  if (xk === "period" || /date|time|period|month|year|day|week|quarter/i.test(xk)) return "time";
  return "category";
}

function measureOfChart(c: any, kind: string): string {
  const t = String(c?.title || "");
  if (kind === "time") return humanize(t.replace(/\s+over\s+time.*$/i, "")) || "Metric";
  const m = t.split(/\s+by\s+/i);
  return humanize(m[0] || t) || "Metric";
}

function dimOfChart(c: any): string {
  const t = String(c?.title || "");
  const m = t.split(/\s+by\s+/i);
  if (m.length > 1) return humanize(m[1].replace(/\s*\(.*\)\s*$/, "")) || "Category";
  return "Category";
}

function grainOfChart(c: any, data: any[]): string {
  const sub = String(c?.subtitle || "");
  const m = /(\w+)\s+periods?/i.exec(sub);
  if (m) {
    const g = m[1].toLowerCase();
    return g[0].toUpperCase() + g.slice(1);
  }
  const first = String(data[0]?.[c?.xKey] ?? "");
  if (/^\d{4}-\d{2}-\d{2}/.test(first)) return "Day";
  if (/^\d{4}-\d{2}/.test(first)) return "Month";
  if (/^\d{4}$/.test(first)) return "Year";
  return "Period";
}

function trendOf(data: any[], yKey: string): number | null {
  const vals = data.map((r) => Number(r?.[yKey])).filter((v) => Number.isFinite(v));
  if (vals.length < 2 || vals[0] === 0) return null;
  const pct = ((vals[vals.length - 1] - vals[0]) / Math.abs(vals[0])) * 100;
  return Number.isFinite(pct) ? pct : null;
}

function peakOf(data: any[], xKey: string, yKey: string): { label: string; value: number | null } {
  let best: any = null;
  for (const r of data) {
    const v = Number(r?.[yKey]);
    if (!Number.isFinite(v)) continue;
    if (!best || v > Number(best[yKey])) best = r;
  }
  if (!best) return { label: "", value: null };
  const bv = Number(best[yKey]);
  return { label: String(best[xKey] ?? ""), value: Number.isFinite(bv) ? bv : null };
}

/** Share text "12.3%" or null when uncomputable (zero/empty/missing totals).
    Central guard so no annotation, tooltip, or badge can print NaN/Infinity. */
function sharePct(part: any, total: any): string | null {
  const p = Number(part);
  const t = Number(total);
  if (!Number.isFinite(p) || !Number.isFinite(t) || t === 0) return null;
  const v = (p / Math.abs(t)) * 100;
  return Number.isFinite(v) ? v.toFixed(1) : null;
}

/** Backend insight/anomaly text mentioning this measure, else "". */
function backendAnnotationFor(measure: string, insights: any[], anomalies: any[]): string {
  const token = normMeasure(measure).split(" ").filter((w) => w.length >= 3)[0];
  if (!token) return "";
  const pool = [...(insights || []), ...(anomalies || [])];
  for (const ins of pool) {
    const hay = normMeasure(
      `${ins?.title || ""} ${ins?.description || ""} ${ins?.explanation || ""} ${ins?.metric || ""} ${ins?.column || ""}`,
    );
    if (hay.includes(token)) {
      const text = String(ins?.description || ins?.explanation || ins?.title || "").trim();
      if (text) return text.length > 160 ? `${text.slice(0, 157)}…` : text;
    }
  }
  return "";
}

function toSelectedViz(c: any, kind: VizKind, insights: any[], anomalies: any[]): SelectedViz {
  const data = Array.isArray(c?.data) ? c.data : [];
  const xKey = kind === "scatter" ? String(c?.xKey || "x") : String(c?.xKey || (kind === "time" ? "period" : "name"));
  const yKey = kind === "scatter" ? String(c?.yKey || "y") : String(c?.dataKey || "value");
  const format = String(c?.format || "number");
  const prefix = format === "currency" ? "$" : "";
  const measure = kind === "time" || kind === "category"
    ? measureOfChart(c, kind)
    : String(c?.measure || "Metric");
  const dim = kind === "category" ? dimOfChart(c) : String(c?.dim || "Category");
  const grain = kind === "time" ? grainOfChart(c, data) : "";
  const title = kind === "time"
    ? `${measure} by ${grain}`
    : kind === "category"
      ? `Top ${data.length} ${dim} by ${measure}`
      : String(c?.title || measure);
  const subtitle = kind === "time" && data.length >= 2
    ? `${fmtPeriod(String(data[0]?.[xKey] ?? ""))} → ${fmtPeriod(String(data[data.length - 1]?.[xKey] ?? ""))} · ${data.length} ${grain.toLowerCase()}s`
    : kind === "category"
      ? `${data.length} groups · ranked`
      : String(c?.subtitle || "");
  const current = data.length ? Number(data[data.length - 1]?.[yKey]) : NaN;
  const trendPct = trendOf(data, yKey);
  const peak = peakOf(data, xKey, yKey);
  const fromBackend = backendAnnotationFor(measure, insights, anomalies);
  let annotation = fromBackend;
  let annotationFromBackend = true;
  if (!annotation && peak.value !== null) {
    annotationFromBackend = false;
    if (kind === "time") {
      const first = fmtPeriod(String(data[0]?.[xKey] ?? ""));
      annotation = `Peak ${fmtExact(peak.value, prefix)} in ${fmtPeriod(peak.label)}${trendPct !== null ? ` (${trendPct >= 0 ? "+" : ""}${trendPct.toFixed(1)}% since ${first})` : ""}.`;
    } else if (data.length) {
      const total = data.reduce((s, r) => s + (Number(r?.[yKey]) || 0), 0);
      const share = sharePct(peak.value, total);
      annotation = `${peak.label} leads with ${fmtExact(peak.value, prefix)}${share !== null ? ` (${share}% of ${measure.toLowerCase()})` : ""}.`;
    }
  }
  return {
    key: String(c?.id || c?.title || measure),
    kind,
    chartType: String(c?.type || (kind === "time" ? "area" : "bar")).toLowerCase(),
    title,
    subtitle,
    data,
    xKey,
    yKey,
    format,
    measure,
    dim,
    grain,
    periodLabel: subtitle,
    current: isNaN(current) ? null : current,
    trendPct,
    peakLabel: peak.label,
    peakValue: peak.value,
    annotation,
    annotationFromBackend,
    emptyReason: "",
    sourceId: String(c?.id || c?.title || ""),
  };
}

function emptyViz(kind: VizKind, key: string): SelectedViz {
  const isTime = kind === "time";
  return {
    key, kind: "empty", chartType: "table", title: isTime ? "Trend over time" : "Breakdown",
    subtitle: "", data: [], xKey: "x", yKey: "y", format: "number",
    measure: "", dim: "", grain: "", periodLabel: "", current: null, trendPct: null,
    peakLabel: "", peakValue: null, annotation: "", annotationFromBackend: false,
    emptyReason: isTime
      ? "No time-based trend available — ARIA couldn't identify a suitable date field for a reliable time trend."
      : "No meaningful categorical breakdown available — ARIA couldn't identify a suitable category and measure for a reliable breakdown.",
    sourceId: "",
  };
}

/** Derive a second perspective from one chart's own aggregates (real data, new view). */
function derivedViz(from: SelectedViz): SelectedViz {
  if (from.kind === "time" && from.data.length >= 2) {
    const ranked = [...from.data]
      .map((r) => ({ name: fmtPeriod(String(r?.[from.xKey] ?? "")), value: Number(r?.[from.yKey]) || 0 }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 8);
    const total = ranked.reduce((s, r) => s + r.value, 0) || 1;
    const top = ranked[0];
    return {
      ...from,
      key: `${from.key}::derived-ranking`,
      kind: "ranking",
      chartType: "hbar",
      title: `Top ${ranked.length} ${from.grain}s by ${from.measure}`,
      subtitle: `Ranked ${from.grain.toLowerCase()}s · same ${from.measure.toLowerCase()} data, new view`,
      data: ranked,
      xKey: "name",
      yKey: "value",
      annotation: `${top.name} ranked first at ${fmtExact(top.value, from.format === "currency" ? "$" : "")}${(() => { const s = sharePct(top.value, total); return s !== null ? ` (${s}% of shown ${from.measure.toLowerCase()})` : ""; })()}.`,
      annotationFromBackend: false,
    };
  }
  if (from.kind === "category" && from.data.length >= 2) {
    const top = [...from.data].sort((a, b) => (Number(b?.[from.yKey]) || 0) - (Number(a?.[from.yKey]) || 0)).slice(0, 6);
    const total = top.reduce((s, r) => s + (Number(r?.[from.yKey]) || 0), 0) || 1;
    const lead = top[0];
    return {
      ...from,
      key: `${from.key}::derived-composition`,
      kind: "composition",
      chartType: "donut",
      title: `${from.measure} share by ${from.dim}`,
      subtitle: `Composition of top ${top.length} · same data, new view`,
      data: top,
      annotation: `${String(lead?.[from.xKey] ?? "")} holds ${(() => { const s = sharePct(Number(lead?.[from.yKey]) || 0, total); return s !== null ? `${s}%` : "the largest share"; })()} of ${from.measure.toLowerCase()}.`,
      annotationFromBackend: false,
    };
  }
  return emptyViz("category", "derived-empty");
}

function selectDashboardViz(charts: any[], insights: any[], anomalies: any[]): [SelectedViz, SelectedViz] {
  const valid = validateDashboardCharts(charts, 4);
  if (valid.length === 0) return [emptyViz("time", "empty-time"), emptyViz("category", "empty-cat")];
  const withKind = valid.map((c) => ({ c, kind: kindOfChart(c) as VizKind }));
  const times = withKind.filter((w) => w.kind === "time");
  const first = times.length
    ? [...times].sort((a, b) => (Number(b.c?.score) || 0) - (Number(a.c?.score) || 0) || (b.c?.data?.length || 0) - (a.c?.data?.length || 0))[0]
    : [...withKind].sort((a, b) => (Number(b.c?.score) || 0) - (Number(a.c?.score) || 0))[0];
  const viz1 = toSelectedViz(first.c, first.kind, insights, anomalies);
  const rest = withKind.filter((w) => w.c !== first.c);
  const firstNorm = normMeasure(viz1.measure);
  const ranked = [...rest].sort((a, b) => {
    const kindScore = (w: typeof a) => {
      let s = 0;
      if (w.kind !== first.kind) s += 2;
      if (w.kind === "scatter" && first.kind !== "scatter") s += 3;
      if (normMeasure(measureOfChart(w.c, w.kind)) !== firstNorm) s += 1;
      return s;
    };
    return kindScore(b) - kindScore(a) || (Number(b.c?.score) || 0) - (Number(a.c?.score) || 0);
  });
  if (ranked.length === 0) return [viz1, derivedViz(viz1)];
  const second = ranked[0];
  return [viz1, toSelectedViz(second.c, second.kind, insights, anomalies)];
}

/* Map a live /intelligence/explore breakdown into a dashboard card so a
   clicked insight/anomaly swaps in a real re-computed visualization
   (same shape VizCard renders). Defensive key picking: first known
   label/value keys, else first string/numeric keys. */
function exploreToFocusViz(rows: any[], metric: string, dim: string, insightTitle: string): SelectedViz | null {
  if (!rows.length) return null;
  const r0 = rows[0] || {};
  const keys = Object.keys(r0);
  const xKey = ["category", "label", "name", "segment", "group"].find((k) => k in r0)
    || keys.find((k) => typeof (r0 as any)[k] === "string") || keys[0];
  const yKey = ["current", "value", "total", "current_value"].find((k) => typeof (r0 as any)[k] === "number")
    || keys.find((k) => typeof (r0 as any)[k] === "number");
  if (!xKey || !yKey) return null;
  const data = rows.slice(0, 12).map((r: any) => ({ [xKey]: String(r?.[xKey] ?? ""), [yKey]: Number(r?.[yKey]) || 0 }));
  if (!data.length) return null;
  const measure = humanize(metric);
  const dimLabel = humanize(dim);
  const peak = peakOf(data, xKey, yKey);
  const format = /revenue|sales|profit|cost|price|amount|payment|budget|spend|expense|income|money/i.test(metric) ? "currency" : "number";
  return {
    key: `focus:${metric}:${dim}`, kind: "category", chartType: "bar",
    title: `${measure} by ${dimLabel}`,
    subtitle: `Focused view · ${insightTitle.slice(0, 90)}`,
    data, xKey, yKey, format, measure, dim: dimLabel, grain: "", periodLabel: "",
    current: Number(data[0]?.[yKey]) || null, trendPct: trendOf(data, yKey),
    peakLabel: peak.label, peakValue: peak.value,
    annotation: peak.label ? `Focused from this insight — ${peak.label} leads.` : "Focused from this insight.",
    annotationFromBackend: false, emptyReason: "", sourceId: "focus",
  };
}

/* ---------------- business tooltip: value + % change + supporting context ---------------- */

function DashTooltip({ active, payload, label, viz, total }: {
  active?: boolean; payload?: any[]; label?: string; viz: SelectedViz; total: number;
}) {
  if (!active || !payload?.length) return null;
  const row = payload[0]?.payload || {};
  const raw = Number(row?.[viz.yKey]);
  const prefix = viz.format === "currency" ? "$" : "";
  // Header name: Recharts passes no `label` for Pie/Scatter hovers, so fall
  // back to the row's category field, then the payload series name — the
  // header never renders blank (generic: works for every chart kind).
  const headName = String(
    label ?? row?.[viz.xKey] ?? (payload[0] as any)?.name ?? "",
  );
  const idx = viz.data.findIndex((r) => String(r?.[viz.xKey]) === String(label ?? row?.[viz.xKey]));
  const prev = idx > 0 ? Number(viz.data[idx - 1]?.[viz.yKey]) : null;
  const change = prev !== null && Number.isFinite(prev) && prev !== 0 && Number.isFinite(raw)
    ? ((raw - prev) / Math.abs(prev)) * 100
    : null;
  const share = total > 0 && Number.isFinite(raw) ? (raw / total) * 100 : null;
  const changeTxt = change !== null && Number.isFinite(change)
    ? `${change >= 0 ? "+" : ""}${change.toFixed(1)}%` : null;
  const shareTxt = share !== null && Number.isFinite(share) ? `${share.toFixed(1)}%` : null;
  return (
    <div className="rounded-lg border border-border bg-card px-3 py-2 shadow-md">
      <p className="text-[11px] font-bold text-foreground mb-1">
        {viz.kind === "time" || viz.kind === "ranking" ? fmtPeriod(headName) : (headName || humanize(String(viz.measure || viz.yKey || "Value")))}
      </p>
      <p className="text-[11px] text-muted-foreground">
        {humanize(String(viz.measure || viz.yKey || "Value"))}: <span className="text-foreground font-semibold">{fmtExact(raw, prefix)}</span>
      </p>
      {changeTxt !== null && (
        <p className="text-[11px] text-muted-foreground">
          vs prior: <span className={`font-semibold ${change !== null && (change as number) >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
            {changeTxt}
          </span>
        </p>
      )}
      {shareTxt !== null && (viz.kind === "category" || viz.kind === "composition" || viz.kind === "ranking") && (
        <p className="text-[11px] text-muted-foreground">
          share: <span className="text-foreground font-semibold">{shareTxt}</span>
          <span className="opacity-70"> · {viz.data.length} groups</span>
        </p>
      )}
      {viz.kind === "time" && (
        <p className="text-[10px] text-muted-foreground opacity-70 mt-0.5">
          {idx + 1} of {viz.data.length} {viz.grain.toLowerCase()}s
        </p>
      )}
    </div>
  );
}

/* ---------------- one large visualization card ---------------- */

function ScatterTooltip({ active, payload, viz }: {
  active?: boolean; payload?: any[]; viz: SelectedViz;
}) {
  if (!active || !payload?.length) return null;
  const row = payload[0]?.payload || {};
  // Point name first (generic): the category field when present, else the
  // first non-numeric descriptive field on the row — never numbers alone.
  const rawName = row?.[viz.xKey];
  const pointName = (typeof rawName === "string" && rawName.trim())
    ? rawName
    : (Object.entries(row).map(([k, v]) => ({ k, v }))
        .find(({ k, v }) => k !== viz.xKey && k !== viz.yKey && k !== "x" && k !== "y"
          && typeof v === "string" && (v as string).trim()
          && !/^\d+(\.\d+)?$/.test((v as string).trim()))?.v as string) || "";
  const xv = row?.x ?? row?.[viz.xKey];
  const yv = row?.y ?? row?.[viz.yKey];
  const prefix = viz.format === "currency" ? "$" : "";
  return (
    <div className="rounded-lg border border-border bg-card px-3 py-2 shadow-md">
      <p className="text-[11px] font-bold text-foreground mb-1">
        {pointName ? humanize(pointName) : humanize(String(viz.measure || viz.yKey || "Point"))}
      </p>
      <p className="text-[11px] text-muted-foreground">
        {humanize(String(viz.xKey || "x"))}:{" "}
        <span className="text-foreground font-semibold">
          {typeof xv === "number" ? fmtExact(xv, "") : String(xv ?? "—")}
        </span>
      </p>
      <p className="text-[11px] text-muted-foreground">
        {humanize(String(viz.measure || viz.yKey || "y"))}:{" "}
        <span className="text-foreground font-semibold">
          {typeof yv === "number" ? fmtExact(yv, prefix) : String(yv ?? "—")}
        </span>
      </p>
    </div>
  );
}

export interface DrillSelection {
  chart: string;
  measure: string;
  label: string;
  value: number | string | null;
}

function VizCard({ viz, grid, axis, onSelect, selection, contextChip, onClearContext, resetKey, onDrill }: {
  viz: SelectedViz;
  grid: string;
  axis: string;
  onSelect: (sel: DrillSelection | null, row: any) => void;
  selection: DrillSelection | null;
  contextChip: string;
  onClearContext: () => void;
  resetKey: number;
  onDrill: (target: Page) => void;
}) {
  const isTime = viz.kind === "time";
  const [range, setRange] = useState<"all" | "12" | "6">("all");
  const [topN, setTopN] = useState<number>(8);
  const [hidden, setHidden] = useState<string[]>([]);
  const [showTrend, setShowTrend] = useState(true);

  useEffect(() => {
    setRange("all");
    setTopN(8);
    setHidden([]);
    setShowTrend(true);
  }, [resetKey, viz.key]);

  const total = useMemo(
    () => viz.data.reduce((s, r) => s + (Number(r?.[viz.yKey]) || 0), 0),
    [viz.data, viz.yKey],
  );

  const viewData = useMemo(() => {
    if (isTime) {
      const sorted = [...viz.data].sort((a, b) => String(a?.[viz.xKey]).localeCompare(String(b?.[viz.xKey])));
      const sliced = range === "6" ? sorted.slice(-6) : range === "12" ? sorted.slice(-12) : sorted;
      return sliced.map((r, i, arr) => {
        const w = arr.slice(Math.max(0, i - 2), i + 1).map((x) => Number(x?.[viz.yKey]) || 0);
        return { ...r, __trend: w.reduce((s, v) => s + v, 0) / w.length };
      });
    }
    const sorted = [...viz.data].sort((a, b) => (Number(b?.[viz.yKey]) || 0) - (Number(a?.[viz.yKey]) || 0));
    const sliced = topN >= sorted.length ? sorted : sorted.slice(0, topN);
    return sliced.filter((r) => !hidden.includes(String(r?.[viz.xKey])));
  }, [viz, isTime, range, topN, hidden]);

  const modified = (isTime && range !== "all") || (!isTime && viz.kind === "category" && topN !== 8) || hidden.length > 0;

  const pick = (row: any): DrillSelection => {
    const label = String(row?.[viz.xKey] ?? "");
    const raw = row?.[viz.yKey];
    return { chart: viz.title, measure: viz.measure, label, value: typeof raw === "number" ? raw : (raw ?? null) };
  };
  const handlePick = (row: any) => {
    if (!row) return;
    const sel = pick(row);
    onSelect(selection?.label === sel.label && selection?.chart === sel.chart ? null : sel, row);
  };

  const mine = selection && selection.chart === viz.title ? selection : null;
  const prefix = viz.format === "currency" ? "$" : "";
  /* Per-measure color so time/scatter charts differ by metric instead of
     all rendering the same blue. */
  const measureColor = colorForName(String(viz.measure || viz.title || "metric"));

  if (viz.kind === "empty" || viz.data.length === 0) {
    return (
      <div className="bg-card border border-border rounded-2xl p-5 flex flex-col gap-2 min-h-[320px] shadow-sm">
        <p className="text-[13px] font-semibold text-foreground">{viz.title}</p>
        <div className="flex-1 flex flex-col items-center justify-center text-center gap-2 py-10">
          <span className="inline-flex items-center justify-center w-10 h-10 rounded-full bg-blue-50 dark:bg-blue-500/10 text-blue-500 dark:text-blue-400">
            <Database size={18} />
          </span>
          <p className="text-[13px] font-medium text-foreground">{viz.title === "Trend over time" ? "No time-based trend available" : "No meaningful categorical breakdown available"}</p>
          <p className="text-[12px] text-muted-foreground leading-relaxed max-w-sm">{viz.emptyReason || "Not enough suitable data to draw this chart yet."}</p>
        </div>
      </div>
    );
  }

  const isHBar = viz.chartType === "hbar";
  const isDonut = viz.chartType === "donut" || viz.chartType === "pie";
  const isScatter = viz.chartType === "scatter";
  const peakIdx = viewData.findIndex((r) => String(r?.[viz.xKey]) === viz.peakLabel);

  return (
    <div className="bg-card border border-border rounded-2xl p-5 flex flex-col gap-3 min-h-[380px] shadow-sm hover:shadow-md transition-shadow min-w-0">
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div className="min-w-0">
          <p className="text-[14px] font-semibold text-foreground tracking-tight truncate">{viz.title}</p>
          <p className="text-[11px] text-muted-foreground mt-0.5">{viz.subtitle || viz.periodLabel}</p>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          {viz.current !== null && (
            <span className="text-sm font-bold text-foreground tabular-nums">{fmt(viz.current, prefix)}</span>
          )}
          {viz.trendPct !== null && (
            <span className={`flex items-center gap-0.5 text-[11px] font-semibold px-1.5 py-0.5 rounded-full ${viz.trendPct >= 0 ? "text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-500/10" : "text-rose-600 dark:text-rose-400 bg-rose-50 dark:bg-rose-500/10"}`}>
              {viz.trendPct >= 0 ? <ArrowUpRight size={11} /> : <ArrowDownRight size={11} />}
              {Math.abs(viz.trendPct).toFixed(1)}%
            </span>
          )}
        </div>
      </div>

      {viz.annotation && (
        <p className="flex items-start gap-1.5 text-[11px] leading-relaxed text-muted-foreground bg-secondary border border-border rounded-xl px-2.5 py-1.5">
          <Sparkles size={11} className="text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
          <span><span className="font-semibold text-foreground">ARIA found: </span>{viz.annotation}</span>
        </p>
      )}

      <div className="flex items-center gap-2 flex-wrap">
        {isTime ? (
          <>
            <button
              onClick={() => setShowTrend((s) => !s)}
              aria-pressed={showTrend}
              className={`text-[11px] font-medium px-2 py-1 rounded-md border transition-colors ${showTrend ? "border-blue-300 dark:border-blue-500/25 text-foreground bg-blue-50 dark:bg-blue-500/10" : "border-border text-muted-foreground line-through opacity-70"}`}
              title="Toggle 3-period moving average"
            >
              <span className="inline-block w-3 border-t-2 border-blue-600 mr-1.5 align-middle" />Trend
            </button>
            <TimeRangeSelector
              options={[{ key: "all", label: "All" }, { key: "12", label: "Last 12" }, { key: "6", label: "Last 6" }]}
              value={range}
              onChange={(r) => setRange(r as any)}
            />
          </>
        ) : !isDonut ? (
          <>
            {[5, 8].map((n) => (
              <button
                key={n}
                onClick={() => setTopN(n)}
                aria-pressed={topN === n}
                className={`text-[11px] font-medium px-2 py-1 rounded-md border transition-colors ${topN === n ? "border-blue-300 dark:border-blue-500/25 text-foreground bg-blue-50 dark:bg-blue-500/10" : "border-border text-muted-foreground hover:text-foreground"}`}
              >
                Top {n}
              </button>
            ))}
            <button
              onClick={() => setTopN(Number.MAX_SAFE_INTEGER)}
              aria-pressed={topN === Number.MAX_SAFE_INTEGER}
              className={`text-[11px] font-medium px-2 py-1 rounded-md border transition-colors ${topN === Number.MAX_SAFE_INTEGER ? "border-blue-300 dark:border-blue-500/25 text-foreground bg-blue-50 dark:bg-blue-500/10" : "border-border text-muted-foreground hover:text-foreground"}`}
            >
              All
            </button>
            {viz.data.length > 1 && (
              <span className="text-[10px] text-muted-foreground">click legend names to hide/show</span>
            )}
          </>
        ) : null}
        {modified && (
          <button
            onClick={() => { setRange("all"); setTopN(8); setHidden([]); }}
            className="ml-auto flex items-center gap-1 text-[11px] font-medium px-2 py-1 rounded-md border border-border text-muted-foreground hover:text-foreground transition-colors"
            title="Reset this chart"
          >
            <RotateCcw size={11} /> Reset
          </button>
        )}
      </div>

      {contextChip && (
        <div className="flex items-center gap-2 text-[11px] text-muted-foreground bg-blue-50/60 dark:bg-blue-500/10 border border-blue-100 dark:border-blue-500/20 rounded-xl px-2.5 py-1.5">
          <span className="truncate">Context from other chart: <span className="font-semibold text-foreground">{contextChip}</span></span>
          <button onClick={onClearContext} className="ml-auto flex items-center gap-0.5 hover:text-foreground flex-shrink-0" title="Clear selection">
            <X size={11} /> Clear
          </button>
        </div>
      )}

      <div className="h-[260px] min-w-0">
        <ResponsiveContainer width="100%" height="100%">
          {isDonut ? (
            <PieChart>
              <Tooltip content={<DashTooltip viz={viz} total={total} />} />
              <Pie
                data={viewData}
                dataKey={viz.yKey}
                nameKey={viz.xKey}
                cx="50%" cy="50%"
                outerRadius={95}
                innerRadius={55}
                label={({ name, percent }: any) => `${String(name).slice(0, 12)} ${((percent || 0) * 100).toFixed(0)}%`}
                labelLine={false}
                onClick={(d: any) => handlePick(d?.payload || d)}
              >
                {viewData.map((row: any, i: number) => {
                  const active = mine && String(row?.[viz.xKey]) === mine.label;
                  return <Cell key={i} fill={colorForName(String(row?.[viz.xKey] ?? i))} opacity={mine && !active ? 0.35 : 1} stroke={active ? "var(--foreground)" : "none"} strokeWidth={active ? 2 : 0} />;
                })}
              </Pie>
            </PieChart>
          ) : isHBar ? (
            <BarChart data={viewData} layout="vertical" margin={{ left: 8, right: 12 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={grid} horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 11, fill: axis }} axisLine={false} tickLine={false} tickFormatter={(v: number) => fmtBiz(v)} />
              <YAxis type="category" dataKey={viz.xKey} tick={{ fontSize: 11, fill: axis }} axisLine={false} tickLine={false} width={110} />
              <Tooltip content={<DashTooltip viz={viz} total={total} />} cursor={{ fill: "var(--secondary)", opacity: 0.7 }} />
              <Bar dataKey={viz.yKey} radius={[0, 4, 4, 0]} onClick={(d: any) => handlePick(d?.payload || d)}>
                {viewData.map((row: any, i: number) => {
                  const active = mine && String(row?.[viz.xKey]) === mine.label;
                  return <Cell key={i} fill={colorForName(String(row?.[viz.xKey] ?? i))} opacity={mine && !active ? 0.35 : 1} />;
                })}
              </Bar>
              <Brush dataKey={viz.xKey} height={20} travellerWidth={8} />
            </BarChart>
          ) : isScatter ? (
            <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={grid} />
              <XAxis dataKey="x" type="number" name={viz.xKey || "x"} tick={{ fontSize: 11, fill: axis }} axisLine={false} tickLine={false} />
              <YAxis dataKey="y" type="number" name={viz.yKey || "y"} tick={{ fontSize: 11, fill: axis }} axisLine={false} tickLine={false} width={56} />
              <ZAxis range={[40, 40]} />
              <Tooltip content={<ScatterTooltip viz={viz} />} cursor={{ strokeDasharray: "3 3" }} />
              <Scatter data={viewData} fill={measureColor} fillOpacity={0.6} onClick={(d: any) => handlePick(d?.payload || d)} />
            </ScatterChart>
          ) : isTime ? (
            <AreaChart
              data={viewData}
              onClick={(st: any) => { if (st?.activePayload?.[0]?.payload) handlePick(st.activePayload[0].payload); }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke={grid} vertical={false} />
              <XAxis dataKey={viz.xKey} tick={{ fontSize: 11, fill: axis }} axisLine={false} tickLine={false} minTickGap={28} tickFormatter={(v: string) => fmtPeriod(v)} />
              <YAxis tick={{ fontSize: 11, fill: axis }} axisLine={false} tickLine={false} width={56} tickFormatter={(v: number) => fmtBiz(v)} />
              <Tooltip content={<DashTooltip viz={viz} total={total} />} />
              <Area type="monotone" dataKey={viz.yKey} stroke={measureColor} strokeWidth={2.5} fill={measureColor} fillOpacity={0.14} dot={false} activeDot={{ r: 5 }} isAnimationActive={false} />
              {showTrend && <Line type="monotone" dataKey="__trend" stroke="#0D9488" strokeWidth={1.5} strokeDasharray="5 4" dot={false} activeDot={false} isAnimationActive={false} />}
              {peakIdx >= 0 && peakIdx < viewData.length && (
                <ReferenceDot
                  x={viewData[peakIdx]?.[viz.xKey]} y={Number(viewData[peakIdx]?.[viz.yKey]) || 0}
                  r={5} fill={measureColor} stroke="#ffffff" strokeWidth={2}
                />
              )}
              <Brush dataKey={viz.xKey} height={20} travellerWidth={8} tickFormatter={(v: string) => fmtPeriod(String(v))} />
            </AreaChart>
          ) : (
            <BarChart
              data={viewData}
              onClick={(st: any) => { if (st?.activePayload?.[0]?.payload) handlePick(st.activePayload[0].payload); }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke={grid} vertical={false} />
              <XAxis dataKey={viz.xKey} tick={{ fontSize: 11, fill: axis }} axisLine={false} tickLine={false} minTickGap={8} interval="preserveStartEnd" />
              <YAxis tick={{ fontSize: 11, fill: axis }} axisLine={false} tickLine={false} width={56} tickFormatter={(v: number) => fmtBiz(v)} />
              <Tooltip content={<DashTooltip viz={viz} total={total} />} cursor={{ fill: "var(--secondary)", opacity: 0.7 }} />
              <Bar dataKey={viz.yKey} radius={[4, 4, 0, 0]} onClick={(d: any) => handlePick(d?.payload || d)}>
                {viewData.map((row: any, i: number) => {
                  const active = mine && String(row?.[viz.xKey]) === mine.label;
                  return <Cell key={i} fill={colorForName(String(row?.[viz.xKey] ?? i))} opacity={mine && !active ? 0.35 : 1} />;
                })}
              </Bar>
              {viewData.length > 4 && <Brush dataKey={viz.xKey} height={20} travellerWidth={8} />}
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>

      {!isTime && !isDonut && viz.data.length > 1 && (
        <div className="flex flex-wrap gap-2">
          {[...viz.data]
            .sort((a, b) => (Number(b?.[viz.yKey]) || 0) - (Number(a?.[viz.yKey]) || 0))
            .slice(0, Math.max(topN, 8))
            .map((r: any) => {
              const name = String(r?.[viz.xKey] ?? "");
              const off = hidden.includes(name);
              return (
                <button
                  key={name}
                  onClick={() => setHidden((h) => (h.includes(name) ? h.filter((x) => x !== name) : [...h, name]))}
                  aria-pressed={!off}
                  className="inline-flex items-center gap-1.5 text-[11px] text-muted-foreground hover:text-foreground transition-colors"
                  title={off ? `Show ${name}` : `Hide ${name}`}
                >
                  <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: off ? "var(--border)" : colorForName(name) }} />
                  <span className={off ? "line-through opacity-60" : ""}>{name.length > 18 ? `${name.slice(0, 17)}…` : name}</span>
                </button>
              );
            })}
        </div>
      )}

      {mine && (
        <div className="rounded-xl border border-blue-200 dark:border-blue-500/25 bg-blue-50/60 dark:bg-blue-500/10 px-3 py-2.5 space-y-2">
          <p className="text-[11px]">
            <span className="font-semibold text-foreground">{mine.label || "Selected point"}</span>
            <span className="text-muted-foreground"> · </span>
            <span className="text-foreground font-semibold">
              {typeof mine.value === "number" ? fmtExact(mine.value, prefix) : String(mine.value ?? "—")}
            </span>
            <span className="text-muted-foreground"> — analyze this further:</span>
          </p>
          <div className="flex flex-wrap gap-1.5">
            {([["analytics", "AI Analytics"], ["insights", "Insights"], ["predictive", "Forecast"]] as [Page, string][]).map(([target, label]) => (
              <button
                key={target}
                onClick={() => onDrill(target)}
                className="text-[11px] font-medium px-2 py-1 rounded-md bg-card border border-border text-foreground hover:border-blue-400 transition-colors"
              >
                {label}
              </button>
            ))}
            <button
              onClick={() => onSelect(null, null)}
              className="text-[11px] font-medium px-2 py-1 rounded-md text-muted-foreground hover:text-foreground transition-colors"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default function Dashboard({ setPage, onAsk, user, connectionStatus = "not-connected", dbName = "", insightResult = null, analysisResult = null, connectionError = "" }: DashboardProps) {
  const { isDark } = useTheme();
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [config, setConfig] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState<number | null>(null);
  const [lastUpdatedIso, setLastUpdatedIso] = useState<string>("");
  const [selection, setSelection] = useState<DrillSelection | null>(null);
  const [resetKey, setResetKey] = useState(0);
  const [activeFilter, setActiveFilter] = useState<string>("all");
  const [, setAnomDismiss] = useState(0);
  const [sessionInfo, setSessionInfo] = useState<{ dbName: string; sourceType: string }>({ dbName: "", sourceType: "" });
  const inputRef = useRef<HTMLInputElement>(null);
  const debounceRef = useRef<any>(null);
  const configBusy = useRef(false);
  const prevDashDb = useRef<string | null>(null);

  /* Insight-focus state: clicking an insight/anomaly swaps in a live
     re-computed focus card (POST /intelligence/explore scoped to it). */
  const [focusViz, setFocusViz] = useState<SelectedViz | null>(null);
  const [focusTitle, setFocusTitle] = useState("");
  const [focusLoading, setFocusLoading] = useState(false);
  const [focusError, setFocusError] = useState("");
  const focusAbort = useRef<AbortController | null>(null);
  const clearFocus = () => {
    focusAbort.current?.abort();
    setFocusViz(null); setFocusTitle(""); setFocusLoading(false); setFocusError("");
  };
  const scrollToCharts = () => {
    try { document.getElementById("dash-charts")?.scrollIntoView({ behavior: "smooth", block: "start" }); } catch { /* ignore */ }
  };

  const grid = isDark ? "#2A2A35" : "#E5E7EB";
  const axis = isDark ? "#8889A0" : "#9CA3AF";

  const trackSuggestionClick = async (text: string) => {
    try {
      await apiFetch("/feedback/suggestion", {
        method: "POST",
        body: JSON.stringify({ suggestion_text: text, source: "dashboard" }),
      });
    } catch (err) { console.debug("[ARIA] background request failed:", err); }
  };

  const fetchConfig = useCallback(async () => {
    if (configBusy.current) return;
    configBusy.current = true;
    setLoading(true);
    setError("");
    try {
      const data: any = await apiFetch("/dashboard/config");
      if (data.ok) {
        setConfig(data.config);
        setLastUpdated(Date.now());
        setLastUpdatedIso(new Date().toISOString());
      } else {
        setError(data?.error || "Failed to load dashboard.");
      }
    } catch (e: any) {
      setError(e?.status === 401
        ? "Session expired. Please sign in again."
        : (e?.message || "Network error."));
    } finally {
      setLoading(false);
      configBusy.current = false;
    }
  }, []);

  const fetchAutocomplete = useCallback(async (partial: string) => {
    try {
      const data: any = await apiFetch(`/autocomplete?partial=${encodeURIComponent(partial)}&limit=6`);
      const list = Array.isArray(data?.suggestions)
        ? data.suggestions.map((s: any) => typeof s === "string" ? s : s.text)
        : [];
      setSuggestions(list);
    } catch {
      setSuggestions([]);
    }
  }, []);

  useEffect(() => {
    if (connectionStatus === "connected") {
      if (prevDashDb.current !== null && prevDashDb.current !== (dbName || "")) {
        // Dataset switched: drop the stale overview + focus state so the
        // previous database never lingers on screen.
        setConfig(null); setSelection(null); clearFocus();
        setActiveFilter("all"); setResetKey((k) => k + 1);
      }
      fetchConfig();
      fetchAutocomplete("");
    }
    prevDashDb.current = dbName || "";
  }, [connectionStatus, dbName, fetchConfig, fetchAutocomplete]);

  /* Real session source for the connected-data card (props first). */
  useEffect(() => {
    if (connectionStatus !== "connected") return;
    if (dbName) return;
    (async () => {
      try {
        const body: any = await apiFetch("/session/status");
        setSessionInfo({
          dbName: String(body?.db_name || ""),
          sourceType: String(body?.source_type || body?.db_type || ""),
        });
      } catch { /* card falls back to config summary labels */ }
    })();
  }, [connectionStatus, dbName]);

  useEffect(() => {
    if (connectionStatus !== "connected") return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => fetchAutocomplete(query || ""), 250);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [query, connectionStatus, fetchAutocomplete]);

  const handleSubmit = () => {
    const q = query.trim();
    if (q) {
      trackSuggestionClick(q);
      onAsk?.(q);
    }
  };

  const askExample = (q: string) => {
    trackSuggestionClick(q);
    setQuery(q);
    onAsk?.(q);
  };

  if (connectionStatus !== "connected") {
    // App.tsx connection states: not-connected | connecting | connected |
    // schema-loading | error. Transient states show loading (never a stale
    // "Connect" prompt); "error" surfaces the real failure + retry.
    const isTransitional = connectionStatus === "connecting" || connectionStatus === "schema-loading";
    const hasFailed = connectionStatus === "error" || connectionStatus === "failed";
    return (
      <div className="min-h-full bg-secondary text-foreground">
        <div className="p-6 max-w-screen-xl mx-auto">
          <div className="text-center py-20">
            <div className="w-16 h-16 rounded-2xl mx-auto mb-5 flex items-center justify-center bg-gradient-to-br from-blue-600 to-indigo-600">
              <Search size={24} className="text-white" />
            </div>
            <h2 className="text-2xl font-semibold tracking-tight mb-2">
              {isTransitional ? "Connecting…" : hasFailed ? "Connection Failed" : "Connect a Data Source"}
            </h2>
            <p className="text-sm text-muted-foreground mb-4 max-w-sm mx-auto">
              {isTransitional
                ? "ARIA is reaching your data source — this takes a moment."
                : hasFailed && connectionError ? businessError(connectionError) : "Get started by connecting your database to unlock AI-powered insights."}
            </p>
            {!isTransitional && (
            <button onClick={() => setPage("data")} className="px-6 py-2.5 rounded-xl text-sm font-semibold text-white transition-all bg-gradient-to-br from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700">
              {hasFailed ? "Retry Connection" : "Connect Now"}
            </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  const metrics: any[] = config?.metrics || [];
  const charts: any[] = config?.visualizations || [];
  const recommendations: any[] = config?.recommendations || [];
  const summary = config?.summary || {};
  const rawInsights: any[] = config?.insights?.length > 0
    ? config.insights
    : (insightResult?.keyInsights || insightResult?.insights || []);
  const anomalies: any[] = config?.anomalies?.length > 0
    ? config.anomalies
    : (insightResult?.anomalies || analysisResult?.anomalies || []);
  const exampleQuestions = recommendations.length > 0
    ? recommendations.slice(0, 4).map((r: any) => typeof r === "string" ? r : r.question)
    : fallbackExamples;

  const rawPrefix = user?.email?.split("@")[0] || "";
  const firstToken = rawPrefix.split(/[._-]+/).filter(Boolean)[0] || "";
  const firstName = firstToken ? firstToken[0].toUpperCase() + firstToken.slice(1) : "there";
  const hour = new Date().getHours();
  const daypart = hour < 12 ? "morning" : hour < 18 ? "afternoon" : "evening";
  const greeting = `Good ${daypart}, ${firstName}`;

  // Filtered signal pool (insights + anomalies share one taxonomy).
  const pool = [
    ...rawInsights.map((ins: any) => ({
      id: ins.id || ins.title,
      title: ins.title || "Insight",
      description: ins.description || ins.explanation || "",
      type: ins.type || ins.category || "Insight",
      detectedAt: ins.detectedAt || ins.created_at || ins.timestamp || "",
      raw: ins,
    })),
    ...anomalies.map((a: any, i: number) => ({
      id: a.id || `anomaly_${i}`,
      title: a.title || `${a.metric || a.column || "Metric"} anomaly`,
      description: a.explanation || a.description || "",
      type: a.type || "Anomaly",
      detectedAt: a.detectedAt || a.created_at || a.timestamp || "",
      raw: a,
    })),
  ];
  const filteredPool = pool.filter((p) => matchesFilter(p.type, activeFilter));
  const importantInsights = filteredPool.slice(0, 4);
  const recentInsights = filteredPool.slice(0, 6);

  const [viz1, viz2] = selectDashboardViz(charts, rawInsights, anomalies);
  /* Chart range toggle (honest subsetting): Last 6 / Last 12 / All points
     of time-kind charts. Category charts are unaffected. */
  const [chartRange, setChartRange] = useState<"all" | "12" | "6">("all");
  const rangeViz = (v: any) => {
    if (!v || v.kind !== "time" || chartRange === "all" || !Array.isArray(v.data)) return v;
    const n = chartRange === "6" ? 6 : 12;
    if (v.data.length <= n) return v;
    return { ...v, data: v.data.slice(-n) };
  };
  const rviz1 = rangeViz(viz1);
  const rviz2 = rangeViz(viz2);

  const drillKpi = (kpi: any) => {
    storeDrillContext({ kind: "kpi", metric: kpi?.label || "Metric", value: kpi?.rawValue ?? kpi?.value ?? null, period: kpi?.period || "" });
    setPage("analytics");
  };
  const askAbout = (q: string) => {
    const question = (q || "").trim();
    if (!question) return;
    if (onAsk) {
      onAsk(question);
      return;
    }
    try {
      sessionStorage.setItem("aria_copilot_context", JSON.stringify({ origin: "dashboard", question }));
    } catch { /* best-effort */ }
    setPage("copilot");
  };
  const askWhy = (metricLabel: string) => {
    const m = (metricLabel || "").trim();
    if (!m) return;
    try {
      sessionStorage.setItem("aria_hypothesis_prefill", JSON.stringify({ metric: m }));
    } catch { /* best-effort */ }
    try {
      window.location.hash = "#/intelligence?tab=hypotheses";
    } catch {
      setPage("intelligence");
    }
  };
  /* On-demand prescription line per insight row (real engine call). */
  const [rxFor, setRxFor] = useState<string | null>(null);
  const [rxAns, setRxAns] = useState<Record<string, string>>({});
  const [rxBusy, setRxBusy] = useState<string | null>(null);
  const askPrescription = async (ins: any) => {
    const id = String(ins.id || ins.title);
    if (rxFor === id) {
      setRxFor(null);
      return;
    }
    setRxFor(id);
    if (rxAns[id]) return;
    setRxBusy(id);
    try {
      const { prescriptionApi } = await import("../../services/api");
      const q = `What should I consider for ${ins.title || "this finding"}?`;
      const r = await (prescriptionApi as any).analyze({ question: q });
      const d: any = (r as any)?.data ?? null;
      const top = (d?.prescriptions || [])[0];
      setRxAns((s) => ({
        ...s,
        [id]: top
          ? String(top?.display?.action_text || top?.title || "")
          : String(d?.refusal?.headline || d?.reason || "Nothing qualified — the evidence bar held."),
      }));
    } catch {
      setRxAns((s) => ({ ...s, [id]: "ARIA couldn't answer that just now." }));
    } finally {
      setRxBusy(null);
    }
  };

  const drillFromChart = (target: Page) => {
    if (selection) {
      storeDrillContext({ kind: "chart-point", metric: selection.measure, chart: selection.chart, label: selection.label, value: selection.value });
    }
    setPage(target);
  };

  /* Highlight the chart point matching a metric/segment (used when an
     insight has no structured breakdown to re-run). */
  const highlightVizFor = (metric: string, label: string): boolean => {
    const target = normMeasure(metric);
    const all = [viz1, viz2].filter((v: any) => v && v.data && v.data.length);
    const match = all.find((v: any) => normMeasure(v.measure) === target) || all[0];
    if (!match) return false;
    const rows = match.data as any[];
    const row = (label && rows.find((r) => normMeasure(String(r?.[match.xKey] ?? "")) === normMeasure(label)))
      || rows.reduce((best: any, r: any) => (!best || Number(r?.[match.yKey]) > Number(best?.[match.yKey])) ? r : best, null);
    if (!row) return false;
    setSelection({ chart: match.title, measure: match.measure, label: String(row?.[match.xKey] ?? label), value: row?.[match.yKey] ?? null });
    return true;
  };

  /* Click an insight/anomaly: re-run analysis scoped to it and swap in a
     live focus card. Falls back to highlighting the related chart point
     when the insight carries no structured metric/dimension. */
  const focusInsight = async (p: any) => {
    const raw = (p && p.raw) || {};
    // Backend metric labels carry table suffixes ("Sales (sample_superstore)")
    // and stray spaces — strip to the bare column name for re-querying.
    const cleanMetric = String(raw.metric || raw.column || raw.measure || "")
      .replace(/\s*\(.*?\)\s*$/, "").trim();
    const recDims = (raw.recommendedNextAnalysis && (raw.recommendedNextAnalysis as any).dimensions) || [];
    const dim = String(
      (Array.isArray(recDims) ? recDims[0] : recDims)
      || raw.segment || raw.dimension || raw.breakdown || "",
    ).trim();
    const title = String(p?.title || "Insight");
    const metric = cleanMetric;
    if (!metric || !dim) {
      // Unstructured insight: highlight by measure token found in its text.
      const hay = normMeasure(`${p?.title || ""} ${p?.description || ""}`);
      const all = [viz1, viz2].filter((v: any) => v && v.measure);
      const byText = all.find((v: any) => {
        const tok = normMeasure(v.measure).split(" ").filter((w) => w.length >= 3)[0];
        return tok && hay.includes(tok);
      });
      if (byText) highlightVizFor(byText.measure, "");
      else if (metric) highlightVizFor(metric, dim);
      scrollToCharts();
      return;
    }
    focusAbort.current?.abort();
    const ctrl = new AbortController(); focusAbort.current = ctrl;
    setFocusLoading(true); setFocusError(""); setFocusTitle(title);
    try {
      const { intelligenceApi } = await import("../../services/api");
      const r: any = await intelligenceApi.runExplore({ metric, breakdown: dim });
      if (ctrl.signal.aborted) return;
      const rd: any = r?.data ?? r;
      if (rd && rd.ok === false) throw new Error(rd.detail || rd.message || "Explore failed");
      const rows: any[] = Array.isArray(rd?.rows) ? rd.rows
        : Array.isArray(rd?.breakdown) ? rd.breakdown
        : Array.isArray(rd?.table) ? rd.table : [];
      const fv = exploreToFocusViz(rows, metric, dim, title);
      if (!fv) throw new Error("No plottable rows returned.");
      setFocusViz(fv);
      const top = fv.data[0];
      if (top) setSelection({ chart: fv.title, measure: fv.measure, label: String(top?.[fv.xKey] ?? ""), value: top?.[fv.yKey] ?? null });
    } catch (e: any) {
      if (ctrl.signal.aborted) return;
      setFocusError(e?.message || "Couldn't focus this insight.");
      if (metric) highlightVizFor(metric, dim);
    } finally {
      if (focusAbort.current === ctrl) setFocusLoading(false);
    }
    scrollToCharts();
  };

  const globalReset = () => {
    setSelection(null);
    clearFocus();
    setResetKey((k) => k + 1);
  };

  const handleRefresh = () => {
    fetchConfig();
  };

  const focusAsk = () => {
    if (query.trim()) { handleSubmit(); return; }
    inputRef.current?.focus();
    setShowSuggestions(true);
  };

  const displayDbName = dbName || summary.database || sessionInfo.dbName || "";
  const displaySourceType = sessionInfo.sourceType || config?.sourceType || summary.sourceType || summary.db_type || "";
  const qualityScore = typeof config?.quality?.score === "number" ? config.quality.score : null;
  const analyzing = loading && metrics.length === 0 && charts.length === 0 && pool.length === 0;

  return (
    <div className="min-h-full bg-secondary text-foreground overflow-x-hidden">
      <style>{`@keyframes dashCardIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}.dash-card-in{animation:dashCardIn .45s ease both}`}</style>
      <div className="p-4 md:p-6 max-w-[1400px] mx-auto space-y-5 min-w-0 max-w-full">
        {/* ---- Hero: greeting + actions + connected data ---- */}
        <section className="flex flex-col lg:flex-row lg:items-start gap-4">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2.5">
              <span className="inline-flex items-center justify-center w-9 h-9 rounded-full bg-amber-50 dark:bg-amber-500/10 text-amber-500 dark:text-amber-400 flex-shrink-0">
                <Sun size={18} />
              </span>
              <h1 className="text-[24px] md:text-[28px] font-semibold tracking-tight leading-tight truncate">{greeting}</h1>
            </div>
            <p className="mt-1.5 text-[13px] text-muted-foreground">
              {summary.title || (displayDbName ? `Here's what ARIA found in ${displayDbName}.` : "Here's what ARIA found in your data.")}
            </p>
            {summary.totalRows > 0 && (
              <p className="text-[11px] text-muted-foreground mt-0.5">
                {summary.tableCount} tables · {summary.totalRows.toLocaleString()} records
              </p>
            )}
            <div className="mt-3 flex items-center gap-2 flex-wrap">
              <button
                onClick={handleRefresh}
                disabled={loading}
                className="inline-flex items-center gap-1.5 text-[12px] font-semibold px-3.5 py-2 rounded-xl bg-card border border-border shadow-sm hover:border-blue-300 dark:hover:border-blue-500/25 hover:shadow transition-all disabled:opacity-50"
              >
                <RefreshCw size={13} className={loading ? "animate-spin" : ""} /> Refresh
              </button>
              <button
                onClick={focusAsk}
                className="inline-flex items-center gap-1.5 text-[12px] font-semibold px-3.5 py-2 rounded-xl bg-blue-700 text-white shadow-md hover:bg-blue-800 transition-all"
              >
                <MessageSquare size={13} /> Ask ARIA
              </button>
              {lastUpdated && (
                <span className="text-[11px] text-muted-foreground">Updated {new Date(lastUpdated).toLocaleTimeString()}</span>
              )}
            </div>
          {/* Upper insight-category filter pills removed — Recent/Important Insights below show the full unfiltered pool. */}
          </div>

          {/* Connected data card — real source only */}
          <div className="w-full lg:w-[300px] flex-shrink-0 rounded-2xl border border-border bg-card p-4 shadow-sm">
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center justify-center w-8 h-8 rounded-xl bg-blue-50 dark:bg-blue-500/10 text-blue-600 dark:text-blue-400">
                <Database size={15} />
              </span>
              <p className="text-[12px] font-semibold flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 inline-block" />
                Connected Data
              </p>
              <span className="ml-auto text-[10px] text-muted-foreground">Last analyzed</span>
            </div>
            <div className="mt-3 flex items-start justify-between gap-2">
              <div className="min-w-0">
                <p className="text-[13px] font-semibold truncate">{displayDbName || "Connected source"}</p>
                {displaySourceType && <p className="text-[11px] text-muted-foreground">{humanize(String(displaySourceType))}</p>}
              </div>
              <p className="text-[11px] font-medium text-secondary-foreground flex-shrink-0" title={lastUpdatedIso ? fmtWhen(lastUpdatedIso) : undefined}>
                {lastUpdatedIso ? relTime(lastUpdatedIso) : "—"}
              </p>
            </div>
            <button onClick={() => setPage("data")} className="mt-2 text-[11px] font-semibold text-blue-600 dark:text-blue-400 hover:underline inline-flex items-center gap-1">
              View details <ChevronRight size={12} />
            </button>
          </div>
        </section>

        {/* ---- Tracked numbers: KPIs pinned from Analytics ---- */}
        {(() => {
          let tracked: any[] = [];
          try {
            const v = JSON.parse(localStorage.getItem("aria_tracked_numbers") || "[]");
            if (Array.isArray(v)) tracked = v.slice(0, 6);
          } catch { /* best-effort */ }
          if (!tracked.length) return null;
          return (
            <section className="rounded-2xl border border-border bg-card p-4 md:p-5 shadow-sm dash-card-in">
              <div className="flex items-center justify-between">
                <h2 className="text-[14px] font-semibold tracking-tight">Tracked numbers</h2>
                <button
                  onClick={() => { try { localStorage.removeItem("aria_tracked_numbers"); } catch {} window.location.reload(); }}
                  className="text-[11px] font-medium text-muted-foreground hover:text-rose-600 dark:hover:text-rose-400"
                  aria-label="Clear tracked numbers"
                >
                  Clear
                </button>
              </div>
              <div className="mt-2 grid grid-cols-2 md:grid-cols-3 gap-2">
                {tracked.map((t: any, i: number) => (
                  <div key={i} className="rounded-xl bg-secondary border border-border px-3 py-2">
                    <p className="text-[11px] text-muted-foreground truncate">{String(t?.label || "Metric")}</p>
                    <p className="text-[18px] font-bold text-foreground tabular-nums">
                      {typeof t?.value === "number" ? t.value.toLocaleString(undefined, { maximumFractionDigits: 1 }) : "—"}
                    </p>
                  </div>
                ))}
              </div>
            </section>
          );
        })()}

        {/* ---- Anomaly banner: severe flags stay visible until dismissed ---- */}
        {(() => {
          const sev = pool.filter((p) => {
            const t = String(p.type || "").toLowerCase();
            if (!(t.includes("anomal") || t.includes("outlier"))) return false;
            const s = String((p.raw as any)?.severity || "").toLowerCase();
            return s.includes("high") || s.includes("severe") || s.includes("critical");
          });
          if (!sev.length) return null;
          const top = sev[0];
          const dismissed: string[] = (() => {
            try {
              return JSON.parse(sessionStorage.getItem("aria_anom_dismissed") || "[]");
            } catch {
              return [];
            }
          })();
          if (dismissed.includes(String(top.id))) return null;
          return (
            <section className="rounded-2xl border border-rose-200 dark:border-rose-500/25 bg-gradient-to-r from-rose-50/80 dark:from-rose-500/10 to-card p-4 flex items-center gap-3 shadow-sm" role="alert">
              <span className="inline-flex items-center justify-center w-9 h-9 rounded-full bg-rose-100 dark:bg-rose-500/15 text-rose-700 dark:text-rose-400 flex-shrink-0">
                <AlertTriangle size={16} />
              </span>
              <div className="min-w-0 flex-1">
                <p className="text-[13px] font-semibold text-foreground truncate">
                  {sev.length} anomal{sev.length === 1 ? "y" : "ies"} flagged{sev.length > 1 ? ` — latest: ${top.title}` : `: ${top.title}`}
                </p>
                <p className="text-[11px] text-muted-foreground truncate">{String(top.description || "Unusual movement detected in your data.")}</p>
              </div>
              <button
                onClick={() => focusInsight(top)}
                className="flex-shrink-0 text-[12px] font-semibold px-3.5 py-2 rounded-xl bg-rose-600 text-white hover:bg-rose-700 transition-colors"
                title="Show this anomaly in the charts below"
              >
                Investigate
              </button>
              <button
                onClick={() => {
                  try {
                    const cur = JSON.parse(sessionStorage.getItem("aria_anom_dismissed") || "[]");
                    sessionStorage.setItem("aria_anom_dismissed", JSON.stringify([...cur, String(top.id)]));
                  } catch { /* best-effort */ }
                  setAnomDismiss((n) => n + 1);
                }}
                className="flex-shrink-0 text-[11px] font-medium text-muted-foreground hover:text-secondary-foreground"
                aria-label="Dismiss anomaly banner"
              >
                Dismiss
              </button>
            </section>
          );
        })()}

        {/* ---- Business-worded error (never raw fetch text) ---- */}
        {error && (
          <div className="rounded-2xl border border-rose-200 dark:border-rose-500/25 bg-card p-4 flex gap-3 shadow-sm">
            <AlertTriangle size={18} className="text-rose-500 dark:text-rose-400 flex-shrink-0 mt-0.5" />
            <div className="min-w-0">
              <p className="text-[13px] font-semibold">ARIA couldn&apos;t load this analysis</p>
              <p className="text-[12px] text-muted-foreground mt-0.5">{businessError(error)}</p>
              <button onClick={handleRefresh} className="mt-1.5 text-[12px] font-semibold text-blue-600 dark:text-blue-400 hover:underline">Try again</button>
            </div>
          </div>
        )}

        {/* ---- Business Overview: first 4 real KPIs ---- */}
        <section className="space-y-3 min-w-0">
          <div className="flex items-center gap-2">
            <h2 className="text-[14px] font-semibold tracking-tight">Business Overview</h2>
            {qualityScore !== null && (
              <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-50 dark:bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
                Data quality {qualityScore}%
              </span>
            )}
            {(() => {
              const v: any = (config as any)?.verdict;
              if (!v) return null;
              const ok = v.verified === true;
              const fails = Array.isArray(v.checks) ? v.checks.filter((c: any) => !c?.pass).length : 0;
              return (
                <span
                  title={ok
                    ? "ARIA double-checked these numbers against your data. Everything lines up."
                    : "ARIA couldn't fully double-check these numbers yet. Treat them as a guide, not the final answer."}
                  className={`inline-flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-full border ${ok
                    ? "bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-500/25"
                    : "bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-500/25"}`}
                >
                  <Shield size={11} />
                  {ok ? "Checked" : "Not fully checked"}
                  {!ok && fails > 0 ? ` · ${fails} to review` : ""}
                </span>
              );
            })()}
          </div>
          {analyzing ? (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4" aria-busy="true">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="bg-card border border-border rounded-2xl p-4 shadow-sm animate-pulse" aria-hidden>
                  <div className="h-3 bg-secondary rounded w-16 mb-3" />
                  <div className="h-7 bg-border rounded w-24 mb-2" />
                  <div className="h-10 bg-secondary rounded w-full" />
                </div>
              ))}
            </div>
          ) : (() => {
            // Date/range/period labels are metadata, not KPIs — drop them
            // and backfill from further real metrics so all 4 cards stay
            // useful on every dataset.
            const isDateCard = (k: any) =>
              /date|range|period|calendar|time\b/i.test(String(k?.label || "")) &&
              (k?.rawValue === null || k?.rawValue === undefined);
            const useful = metrics.filter((k: any) => !isDateCard(k));
            const cards = (useful.length ? useful : metrics).slice(0, 4);
            return cards.length > 0 ? (
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                {cards.map((kpi: any, i: number) => (
                  <MetricCard key={kpi.id || kpi.label || i} kpi={kpi} index={i} onDrill={drillKpi} onAsk={askAbout} onWhy={askWhy} />
                ))}
              </div>
            ) : !error ? (
              <div className="bg-card border border-dashed border-border rounded-2xl p-6 text-center shadow-sm">
                <p className="text-[13px] font-medium">No KPIs available for this dataset yet.</p>
                <p className="text-[11px] text-muted-foreground mt-1">ARIA shows a KPI only when the data supports a meaningful measure — nothing is fabricated.</p>
              </div>
            ) : null;
          })()}
        </section>

        {/* ---- Executive summary strip ---- */}
        {summary.description && (
          <section aria-label="Executive summary">
            <div className="flex items-center gap-3 bg-card border border-border rounded-2xl px-4 py-3 text-[13px] text-muted-foreground leading-relaxed shadow-sm min-w-0">
              <Sparkles size={14} className="text-blue-600 dark:text-blue-400 flex-shrink-0" />
              <p className="flex-1 min-w-0">{summary.description}</p>
            </div>
          </section>
        )}

        {config?.dataset_profile && <details className="rounded-2xl border border-border bg-card p-4 text-sm">
          <summary className="cursor-pointer font-semibold">Dataset understanding and data quality</summary>
          <p className="mt-3">{config.dataset_understanding?.description}</p>
          {(config.dataset_understanding?.assumptions || []).map((text: string, i: number) => <p key={i} className="mt-1">Assumption: {text}</p>)}
          <p className="text-xs text-muted-foreground mt-2">Profile scope: {config.dataset_profile.profile?.scope}. SQL analysis uses the connected dataset.</p>
          {(config.quality?.issues || []).map((text: string, i: number) => <p key={i} className="mt-1 text-muted-foreground">{text}</p>)}
          <p className="mt-2 text-xs text-muted-foreground">Metric meaning is AI-interpreted and has not been independently verified.</p>
          <p className="mt-2 text-xs">{config.llm_trace?.cached ? "Cached analysis" : "Fresh analysis"} · {config.llm_trace?.provider}</p>
        </details>}

        {config?.validation && <details className="rounded-2xl border border-border bg-card p-4 text-xs">
          <summary className="cursor-pointer font-semibold">Analysis checks and review</summary>
          {(config.validation.checks || []).map((check: any, i: number) => <p className="mt-2" key={i}>{check.detail}</p>)}
          <p className="mt-2">AI review: {config.validation.semantic_review?.status}. {config.validation.semantic_review?.reason}</p>
          <p className="mt-2 text-muted-foreground">Model review is not independent semantic verification.</p>
        </details>}
        {config?.ranked_insights?.length > 0 && <details className="rounded-2xl border border-border bg-card p-4 text-xs"><summary className="cursor-pointer font-semibold">Patterns to investigate</summary>{config.ranked_insights.map((item: any, i: number) => <p className="mt-2" key={i}>{item.title} — {item.effect_measure.replace(/_/g, " ")}: {item.largest_absolute_effect ?? "Unavailable"}. {item.note}</p>)}</details>}

        {(config?.statistical_insights || []).map((result: any, index: number) => <section key={index} className="rounded-2xl border border-border bg-card p-4">
          <h3 className="font-semibold">{result.title}</h3>
          <p className="text-sm text-muted-foreground mt-1">{result.reason}</p>
          <CsvStatisticalEvidence result={result} showTable />
        </section>)}

        {/* ---- Main + rail grid ---- */}
        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_300px] gap-5 items-start min-w-0">
          <div className="min-w-0 space-y-5">
            {/* Key Visualizations */}
            <section id="dash-charts" className="space-y-3 min-w-0 scroll-mt-4">
              <div className="flex items-center gap-2">
                <Sparkles size={14} className="text-blue-600 dark:text-blue-400" />
                <h2 className="text-[14px] font-semibold tracking-tight">Key Visualizations</h2>
                <div className="ml-auto flex items-center gap-2">
                  <TimeRangeSelector
                    options={[{ key: "all", label: "All" }, { key: "12", label: "Last 12" }, { key: "6", label: "Last 6" }]}
                    value={chartRange}
                    onChange={(r) => setChartRange(r as any)}
                  />
                  {selection && (
                    <button onClick={() => setSelection(null)} className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground transition-colors" title="Clear chart selection">
                      <X size={11} /> Clear selection
                    </button>
                  )}
                  <button onClick={globalReset} className="hidden sm:flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground transition-colors" title="Reset all chart controls">
                    <RotateCcw size={11} /> Reset
                  </button>
                  <button onClick={() => setPage("analytics")} className="text-[11px] font-semibold text-blue-600 dark:text-blue-400 hover:underline">
                    View all →
                  </button>
                </div>
              </div>
              {loading && charts.length === 0 ? (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4" aria-busy="true">
                  {[1, 2].map((i) => (
                    <div key={i} className="bg-card border border-border rounded-2xl p-5 shadow-sm animate-pulse" aria-hidden>
                      <div className="h-4 bg-border rounded w-40 mb-2" />
                      <div className="h-3 bg-secondary rounded w-56 mb-4" />
                      <div className="h-[260px] bg-secondary rounded-xl w-full" />
                    </div>
                  ))}
                </div>
              ) : (
                <>
                  {/* Insight focus: live re-computed card for the clicked insight/anomaly */}
                  {(focusViz || focusLoading || focusError) && (
                    <div className="rounded-2xl border border-blue-200 dark:border-blue-500/25 bg-blue-50/60 dark:bg-blue-500/10 p-3 shadow-sm">
                      <div className="flex items-center gap-2 mb-2 px-1">
                        <Sparkles size={13} className="text-blue-600 dark:text-blue-400 flex-shrink-0" />
                        <p className="text-[12px] font-semibold text-foreground truncate">
                          Insight focus{focusTitle ? `: ${focusTitle}` : ""}
                        </p>
                        <button
                          onClick={() => { clearFocus(); setSelection(null); }}
                          className="ml-auto inline-flex items-center gap-1 text-[11px] font-medium text-muted-foreground hover:text-foreground transition-colors flex-shrink-0"
                          aria-label="Clear insight focus"
                        >
                          <X size={12} /> Back to overview
                        </button>
                      </div>
                      {focusLoading && (
                        <div className="bg-card border border-blue-100 dark:border-blue-500/20 rounded-xl p-5" aria-busy="true">
                          <div className="h-4 bg-border rounded w-40 mb-2 animate-pulse" />
                          <div className="h-[220px] bg-secondary rounded-xl w-full animate-pulse" />
                        </div>
                      )}
                      {!focusLoading && focusError && (
                        <p className="text-[12px] text-rose-600 dark:text-rose-400 bg-card border border-rose-100 dark:border-rose-500/25 rounded-xl px-4 py-3">{focusError}</p>
                      )}
                      {!focusLoading && focusViz && (
                        <VizCard
                          viz={focusViz} grid={grid} axis={axis} resetKey={resetKey} onDrill={drillFromChart}
                          selection={selection}
                          onSelect={(sel) => setSelection(sel)}
                          contextChip={selection && selection.chart !== focusViz.title ? `${selection.label} = ${typeof selection.value === "number" ? fmtExact(selection.value) : selection.value ?? "—"}` : ""}
                          onClearContext={() => setSelection(null)}
                        />
                      )}
                    </div>
                  )}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 min-w-0">
                  <VizCard
                    viz={rviz1} grid={grid} axis={axis} resetKey={resetKey} onDrill={drillFromChart}
                    selection={selection}
                    onSelect={(sel) => setSelection(sel)}
                    contextChip={selection && selection.chart !== viz1.title ? `${selection.label} = ${typeof selection.value === "number" ? fmtExact(selection.value) : selection.value ?? "—"}` : ""}
                    onClearContext={() => setSelection(null)}
                  />
                  <VizCard
                    viz={rviz2} grid={grid} axis={axis} resetKey={resetKey} onDrill={drillFromChart}
                    selection={selection}
                    onSelect={(sel) => setSelection(sel)}
                    contextChip={selection && selection.chart !== viz2.title ? `${selection.label} = ${typeof selection.value === "number" ? fmtExact(selection.value) : selection.value ?? "—"}` : ""}
                    onClearContext={() => setSelection(null)}
                  />
                </div>
                </>
              )}
            </section>

            {/* Important Insights */}
            <section className="rounded-2xl border border-border bg-card shadow-sm p-4 space-y-3 min-w-0">
              <div className="flex items-center gap-2">
                <Sparkles size={14} className="text-blue-600 dark:text-blue-400" />
                <h2 className="text-[14px] font-semibold tracking-tight">Important Insights</h2>
                <button onClick={() => setPage("insights")} className="ml-auto text-[11px] font-semibold text-blue-600 dark:text-blue-400 hover:underline">
                  View all →
                </button>
              </div>
              {analyzing ? (
                <div className="space-y-2.5 animate-pulse" aria-hidden>
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="rounded-xl border border-border p-3">
                      <div className="h-3.5 bg-border rounded w-3/4 mb-2" />
                      <div className="h-3 bg-secondary rounded w-full" />
                    </div>
                  ))}
                </div>
              ) : importantInsights.length > 0 ? (
                <div className="divide-y divide-border">
                  {importantInsights.map((ins: any) => {
                    const meta = CAT_META[primaryCategory(ins.type)];
                    const Icon = meta.Icon;
                    return (
                      <button key={ins.id} onClick={() => focusInsight(ins)} className="w-full text-left py-2.5 flex items-center gap-2.5 group min-w-0" title="Show this insight in the charts above">
                        <span className={`inline-flex items-center justify-center w-8 h-8 rounded-full flex-shrink-0 ${meta.iconWrap}`}>
                          <Icon size={14} />
                        </span>
                        <span className="min-w-0 flex-1">
                          <span className="block text-[12px] font-semibold leading-snug group-hover:text-blue-700 dark:group-hover:text-blue-400 line-clamp-2">{ins.title}</span>
                          <span className="block text-[11px] text-muted-foreground leading-snug line-clamp-2 mt-0.5">{ins.description}</span>
                          <span className="mt-1 flex items-center gap-2">
                            <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${meta.pill}`}>{meta.label}</span>
                            {ins.detectedAt && <span className="text-[10px] text-muted-foreground">{fmtWhen(ins.detectedAt)}</span>}
                          </span>
                        </span>
                        <ChevronRight size={14} className="text-slate-300 dark:text-muted-foreground group-hover:text-blue-500 dark:group-hover:text-blue-400 flex-shrink-0" />
                      </button>
                    );
                  })}
                </div>
              ) : (
                <div className="text-center py-8">
                  <p className="text-[13px] font-medium">No signals under this filter.</p>
                  <p className="text-[11px] text-muted-foreground mt-1">The analysis ran — this view just has nothing notable in this category.</p>
                  {activeFilter !== "all" && (
                    <button onClick={() => setActiveFilter("all")} className="mt-2 text-[11px] font-semibold text-blue-600 dark:text-blue-400 hover:underline">Clear filter</button>
                  )}
                </div>
              )}
            </section>

            {/* Ask ARIA Anything — existing Copilot flow with autocomplete */}
            <section className="rounded-2xl border border-blue-100 dark:border-blue-500/20 bg-gradient-to-r from-blue-50/80 dark:from-blue-500/10 to-indigo-50/80 dark:to-indigo-500/10 p-4 flex flex-col sm:flex-row sm:items-center gap-3 shadow-sm">
              <div className="flex items-center gap-3 min-w-0 sm:w-64 flex-shrink-0">
                <span className="inline-flex items-center justify-center w-10 h-10 rounded-full bg-blue-100 dark:bg-blue-500/15 text-blue-700 dark:text-blue-400 flex-shrink-0">
                  <Sparkles size={16} />
                </span>
                <span className="min-w-0">
                  <span className="block text-[13px] font-semibold">Ask ARIA Anything</span>
                  <span className="block text-[11px] text-muted-foreground leading-snug">Get deeper insights, ask follow-up questions, or explore your data in natural language.</span>
                </span>
              </div>
              <div className="relative flex-1 min-w-0">
                <div className="flex items-center gap-2 bg-card border border-border rounded-xl pl-3 pr-1.5 py-1.5 shadow-sm focus-within:border-blue-400">
                  <MessageSquare size={14} className="text-muted-foreground flex-shrink-0" />
                  <input
                    ref={inputRef}
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onFocus={() => setShowSuggestions(true)}
                    onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
                    onKeyDown={(e) => { if (e.key === "Enter") handleSubmit(); }}
                    placeholder="Ask a question about your data…"
                    className="flex-1 min-w-0 bg-transparent text-[12px] text-foreground placeholder:text-muted-foreground outline-none"
                    aria-label="Ask ARIA anything"
                  />
                  <button onClick={handleSubmit} disabled={!query.trim()} className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-40 transition-colors flex-shrink-0" title="Ask Copilot" aria-label="Send question">
                    <Send size={14} />
                  </button>
                </div>
                {showSuggestions && (query.length > 0 ? suggestions.length > 0 : exampleQuestions.length > 0) && (
                  <div className="absolute bottom-full left-0 right-0 mb-2 bg-card border border-border rounded-xl shadow-lg py-1.5 z-20 max-h-[240px] overflow-y-auto">
                    {(query.length > 0 ? suggestions : exampleQuestions).map((s: string, i: number) => (
                      <button key={s + i} onMouseDown={() => askExample(s)}
                        className="w-full flex items-center gap-3 px-4 py-2.5 text-[12px] text-foreground hover:bg-secondary transition-colors text-left">
                        <Zap size={13} className="text-blue-600 dark:text-blue-400 flex-shrink-0" />
                        <span className="truncate">{s}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </section>
          </div>

          {/* Right rail: Recent Insights + Quick Actions */}
          <aside id="dash-insights" className="min-w-0 space-y-4 xl:sticky xl:top-4 scroll-mt-4">
            <div className="rounded-2xl border border-border bg-card shadow-sm overflow-hidden">
              <div className="flex items-center gap-2 px-4 pt-3.5 pb-2.5 border-b border-border">
                <Sparkles size={14} className="text-blue-600 dark:text-blue-400" />
                <p className="text-[13px] font-semibold">Recent Insights</p>
                <button onClick={() => setPage("insights")} className="ml-auto text-[11px] font-semibold text-blue-600 dark:text-blue-400 hover:underline">
                  View all →
                </button>
              </div>
              <div className="divide-y divide-border">
                {analyzing && (
                  <div className="px-4 py-3 space-y-2 animate-pulse" aria-hidden>
                    {[1, 2, 3].map((i) => (
                      <div key={i}>
                        <div className="h-3 bg-border rounded w-3/4 mb-1.5" />
                        <div className="h-2.5 bg-secondary rounded w-1/2" />
                      </div>
                    ))}
                  </div>
                )}
                {!analyzing && recentInsights.length === 0 && (
                  <p className="px-4 py-5 text-[11px] text-muted-foreground">
                    {activeFilter !== "all"
                      ? `No ${activeFilter} insights right now — try All or another category.`
                      : "No insights loaded yet — connect data and they will appear here."}
                  </p>
                )}
                {!analyzing && recentInsights.map((ins: any) => {
                  const meta = CAT_META[primaryCategory(ins.type)];
                  const Icon = meta.Icon;
                  return (
                    <div
                      key={ins.id}
                      onClick={() => focusInsight(ins)}
                      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); focusInsight(ins); } }}
                      role="button"
                      tabIndex={0}
                      title="Show this insight in the charts above"
                      className="w-full text-left px-4 py-2.5 flex items-center gap-2.5 hover:bg-blue-50/50 dark:hover:bg-blue-500/10 transition-colors group min-w-0 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                    >
                      <span className={`inline-flex items-center justify-center w-7 h-7 rounded-full flex-shrink-0 ${meta.iconWrap}`}>
                        <Icon size={13} />
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="block text-[12px] font-medium leading-snug line-clamp-2 group-hover:text-blue-700 dark:group-hover:text-blue-400">{ins.title}</span>
                        <span className="mt-0.5 flex items-center gap-1.5">
                          <span className={`text-[10px] font-semibold px-1.5 py-px rounded-full border ${meta.pill}`}>{meta.label}</span>
                          {ins.detectedAt && <span className="text-[10px] text-muted-foreground">{fmtWhen(ins.detectedAt)}</span>}
                          <button
                            onClick={(e) => { e.stopPropagation(); askAbout(`Explain this finding: ${ins.title}`); }}
                            className="text-[10px] font-semibold text-blue-600 dark:text-blue-400 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
                            aria-label={`Ask ARIA about ${ins.title}`}
                          >
                            Ask
                          </button>
                          <button
                            onClick={(e) => { e.stopPropagation(); void askPrescription(ins); }}
                            className="text-[10px] font-medium text-muted-foreground hover:text-blue-700 dark:hover:text-blue-400 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
                            aria-label={`What should I do about ${ins.title}`}
                          >
                            {rxBusy === String(ins.id || ins.title) ? "Checking…" : "What should I do?"}
                          </button>
                        </span>
                        {rxFor === String(ins.id || ins.title) && rxAns[String(ins.id || ins.title)] && (
                          <span className="mt-1 block text-[11px] text-secondary-foreground leading-snug bg-blue-50/60 dark:bg-blue-500/10 border border-blue-100 dark:border-blue-500/20 rounded-lg px-2 py-1.5">
                            {rxAns[String(ins.id || ins.title)]}
                          </span>
                        )}
                      </span>
                      <ChevronRight size={13} className="text-slate-300 dark:text-muted-foreground group-hover:text-blue-500 dark:group-hover:text-blue-400 flex-shrink-0" />
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="rounded-2xl border border-border bg-card shadow-sm p-4">
              <p className="text-[13px] font-semibold">Quick Actions</p>
              <div className="mt-2.5 space-y-2">
                <button onClick={focusAsk} className="w-full text-left rounded-xl border border-border px-3 py-2.5 hover:border-blue-300 dark:hover:border-blue-500/25 hover:shadow-sm transition-all group">
                  <span className="block text-[12px] font-semibold group-hover:text-blue-700 dark:group-hover:text-blue-400">Ask a question</span>
                  <span className="block text-[11px] text-muted-foreground">Ask ARIA about your data.</span>
                </button>
                <button
                  onClick={() => { storeDrillContext({ kind: "quick-action", action: "create-chart" }); setPage("analytics"); }}
                  className="w-full text-left rounded-xl border border-border px-3 py-2.5 hover:border-blue-300 dark:hover:border-blue-500/25 hover:shadow-sm transition-all group"
                >
                  <span className="block text-[12px] font-semibold group-hover:text-blue-700 dark:group-hover:text-blue-400">Create a chart</span>
                  <span className="block text-[11px] text-muted-foreground">Visualize using your connected data.</span>
                </button>
                <button onClick={() => setPage("insights")} className="w-full text-left rounded-xl border border-border px-3 py-2.5 hover:border-blue-300 dark:hover:border-blue-500/25 hover:shadow-sm transition-all group">
                  <span className="block text-[12px] font-semibold group-hover:text-blue-700 dark:group-hover:text-blue-400">Generate insights</span>
                  <span className="block text-[11px] text-muted-foreground">Open the insight workflow.</span>
                </button>
              </div>
            </div>
          </aside>
        </div>

        {anomalies.length > 0 && (
          <p className="flex items-center gap-1.5 text-[11px] text-muted-foreground pb-4">
            <AlertTriangle size={11} />
            {anomalies.length} anomal{anomalies.length === 1 ? "y" : "ies"} flagged — see Insights for the full list.
            <button onClick={() => setPage("insights")} className="ml-1 inline-flex items-center gap-0.5 font-semibold text-blue-600 dark:text-blue-400 hover:underline">
              Explore <ArrowRight size={11} />
            </button>
          </p>
        )}
      </div>
    </div>
  );
}
