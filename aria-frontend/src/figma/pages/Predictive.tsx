import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { RefreshCw, Sparkles, Target, TrendingUp, TrendingDown, Crosshair, Lightbulb, Check, Shield } from "lucide-react";
import { ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Brush } from "recharts";
import { useTheme } from "../App";
import Evidence from "../components/Evidence";
import { ChartFrame, useSeriesToggle, ToggleLegend } from "../components/charts";
import { CATEGORY_COLORS } from "../lib/charts";
import AIProcessingIndicator from "../components/AIProcessingIndicator";
import { loadAnalysisContext, readHashParams, saveAnalysisContext } from "../lib/analysisContext";
import { apiFetch, normalizeApiError, prescriptionApi } from "../../services/api";

interface PredictiveProps {
  insightResult?: any;
  goalResult?: any;
  connectionStatus?: string;
  dbName?: string;
}

type PeriodTab = "Monthly" | "Quarterly" | "Yearly";

/* Theme-palette series colors (src/figma/lib/charts.ts — never ad-hoc hex):
   Actual = green, Forecast = steel blue, Baseline = slate, CI = steel blue wash. */
const ACTUAL_C = CATEGORY_COLORS[1];
const FORECAST_C = CATEGORY_COLORS[0];
const BASELINE_C = CATEGORY_COLORS[4];

const plainName = (c: string) => (c || "").split(" (")[0].trim();

const fmtNum = (v: any) => {
  if (v === null || v === undefined || isNaN(Number(v))) return "—";
  const n = Number(v);
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (Math.abs(n) >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toLocaleString(undefined, { maximumFractionDigits: 1 });
};

const shortLabel = (label: string) => {
  const m = /^(\d{4})-(\d{2})(?:-(\d{2}))?$/.exec(String(label || ""));
  if (m) {
    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    return months[Number(m[2]) - 1] || label;
  }
  return String(label || "");
};

const fullLabel = (label: string) => {
  const m = /^(\d{4})-(\d{2})(?:-(\d{2}))?$/.exec(String(label || ""));
  if (m) {
    const months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    const mon = months[Number(m[2]) - 1] || m[2];
    return m[3] ? `${mon} ${Number(m[3])}, ${m[1]}` : `${mon} ${m[1]}`;
  }
  return String(label || "");
};

const monthKey = (label: string) => String(label || "").slice(0, 7);

/** Pure forecast-story builder (exported for verification): dynamic
 *  2–4 sentence business story + key signal from scenario values. */
export function buildForecastStory(args: {
  hasPredictions: boolean;
  scenarios: { label: string; total: number; tone: string; pct: number }[] | null;
  targetName: string;
  horizonPeriods: number;
  accuracy: number | null;
  momDelta: number | null;
}): { story: string; signal: string } {
  const { hasPredictions, scenarios, targetName, horizonPeriods: n, accuracy, momDelta } = args;
  const empty = { story: "", signal: "" };
  if (!hasPredictions || !scenarios) return empty;
  const best = scenarios.find((s) => s.tone === "good");
  const commit = scenarios.find((s) => s.tone === "base");
  const worst = scenarios.find((s) => s.tone === "bad");
  if (!commit) return empty;
  const horizonTxt = n === 1 ? "next period" : `next ${n} periods`;
  const name = targetName || "the metric";

  let s1: string;
  if (commit.pct > 3) {
    s1 = `ARIA expects ${name} to keep growing, reaching about ${fmtNum(commit.total)} over the ${horizonTxt} (+${commit.pct.toFixed(0)}% vs the recent pace).`;
  } else if (commit.pct < -3) {
    s1 = `ARIA expects ${name} to soften to about ${fmtNum(commit.total)} over the ${horizonTxt} (${commit.pct.toFixed(0)}% vs the recent pace).`;
  } else {
    s1 = `ARIA expects ${name} to hold around ${fmtNum(commit.total)}, close to the current pace.`;
  }

  let s2 = "";
  const spread = best && worst ? Math.abs(best.total - worst.total) : 0;
  const base = Math.abs(commit.total) || 1;
  if (best && worst && spread / base > 0.05) {
    s2 = ` Stronger momentum could lift it toward ${fmtNum(best.total)}, while a slowdown could bring it closer to ${fmtNum(worst.total)}.`;
  }

  let s3: string;
  if (best && worst && spread / base > 0.5) {
    s3 = " The range between scenarios is wide, so the outcome will depend heavily on how the recent trend develops.";
  } else if (best && worst && spread / base < 0.15) {
    s3 = " The scenarios sit close together, suggesting a steadier outlook.";
  } else if (accuracy !== null && accuracy < 70) {
    s3 = " Backtesting scored below 70% accuracy, so treat this as directional rather than precise.";
  } else if (accuracy !== null) {
    s3 = ` Backtesting scored ${accuracy.toFixed(0)}% accuracy, so this is a working estimate, not a promise.`;
  } else {
    s3 = " Treat distant points as directional — near-term points are the most trustworthy.";
  }

  let signal: string;
  if (momDelta !== null && momDelta > 5) {
    signal = "Recent momentum is strengthening.";
  } else if (momDelta !== null && momDelta < -5) {
    signal = "Recent momentum is weakening.";
  } else if (best && worst && spread / base > 0.5) {
    signal = "The upside depends on continued recent momentum.";
  } else if (accuracy !== null && accuracy < 70) {
    signal = "Uncertainty is higher than usual — watch incoming actuals.";
  } else {
    signal = "The forecast remains relatively stable.";
  }

  return { story: `${s1}${s2}${s3}`, signal };
}

function labelsParseAsDates(rows: any[]): boolean {
  if (!rows || rows.length === 0) return false;
  // Every plotted label must look like a backend date (YYYY-MM or YYYY-MM-DD).
  // Anything else (period names, buckets, free text) locks regrouping.
  return rows.every((r: any) => /^\d{4}-\d{2}(?:-\d{2})?$/.test(String(r?.label || "").slice(0, 10)));
}

function rebin(rows: any[], tab: PeriodTab) {
  if (tab === "Monthly") return rows;
  // Regroup only when labels parse as dates; otherwise return rows untouched
  // so Quarterly/Yearly can never invent buckets from free-text labels.
  if (!labelsParseAsDates(rows)) return rows;
  const groups = new Map<string, any>();
  for (const r of rows) {
    const mk = monthKey(r.label);
    if (!/^\d{4}-\d{2}$/.test(mk)) continue;
    const [y, m] = mk.split("-").map(Number);
    const key = tab === "Quarterly" ? `${y}-Q${Math.floor((m - 1) / 3) + 1}` : `${y}`;
    if (!groups.has(key)) groups.set(key, { label: key, short: key, actual: 0, forecast: 0, lower95: 0, upper95: 0, _a: false, _f: false, _lo: false, _hi: false });
    const g = groups.get(key);
    if (r.actual !== null && r.actual !== undefined) { g.actual += Number(r.actual) || 0; g._a = true; }
    if (r.forecast !== null && r.forecast !== undefined) { g.forecast += Number(r.forecast) || 0; g._f = true; }
    // Uncertainty bands are sums of the backend bounds in the bucket —
    // same regrouping arithmetic as actual/forecast, never invented.
    if (r.lower95 !== null && r.lower95 !== undefined) { g.lower95 += Number(r.lower95) || 0; g._lo = true; }
    if (r.upper95 !== null && r.upper95 !== undefined) { g.upper95 += Number(r.upper95) || 0; g._hi = true; }
  }
  return [...groups.entries()]
    .sort((a, b) => (a[0] < b[0] ? -1 : 1))
    .map(([, g]) => ({
      label: g.label,
      short: g.label,
      actual: g._a ? Math.round(g.actual * 100) / 100 : null,
      forecast: g._f ? Math.round(g.forecast * 100) / 100 : null,
      lower95: g._lo ? Math.round(g.lower95 * 100) / 100 : null,
      upper95: g._hi ? Math.round(g.upper95 * 100) / 100 : null,
    }));
}

/* Cross-link context: "Forecast this" from AI Analytics / Insights writes
 * localStorage `aria_forecast_context`
 *   {metric, dataset, date, filters, horizon, granularity, question}
 * (legacy sessionStorage `aria_forecast_prefill` is still honored). The
 * Forecast page consumes it once so Metric/Dataset/Date/Filters arrive
 * preselected without retyping. */
export interface ForecastLinkContext {
  metric?: string;
  dataset?: string;
  table?: string;
  date?: string;
  time_dim?: string;
  timeDim?: string;
  period?: string;
  filters?: any;
  horizon?: number;
  granularity?: string;
  question?: string;
}

function normalizeLinkFilters(f: any): { column: string; value: string }[] {
  if (!Array.isArray(f)) return [];
  const out: { column: string; value: string }[] = [];
  for (const item of f) {
    if (typeof item === "string") {
      const m = item.split("=").map((s) => s.trim());
      if (m[0]) out.push({ column: m[0], value: m[1] || "" });
    } else if (item && typeof item === "object" && item.column) {
      out.push({ column: String(item.column), value: String(item.value ?? "") });
    }
  }
  return out.slice(0, 5);
}

function readForecastLinkContext(): ForecastLinkContext | null {
  try {
    const ls = localStorage.getItem("aria_forecast_context");
    if (ls) {
      localStorage.removeItem("aria_forecast_context");
      const c = JSON.parse(ls);
      if (c && typeof c === "object") return c;
    }
  } catch { /* best-effort */ }
  try {
    const ss = sessionStorage.getItem("aria_forecast_prefill");
    if (ss) {
      sessionStorage.removeItem("aria_forecast_prefill");
      const h = JSON.parse(ss);
      if (h && typeof h === "object") {
        return {
          metric: typeof h.metric === "string" ? h.metric : (typeof h.question === "string" ? h.question : undefined),
          dataset: h.dataset || h.table || h.source?.tables?.[0],
          date: h.date || h.time_dim || h.timeDim || h.period,
          time_dim: h.time_dim || h.timeDim,
          filters: h.filters,
          horizon: h.horizon,
          granularity: h.granularity,
          question: typeof h.question === "string" ? h.question : undefined,
        };
      }
    }
  } catch { /* best-effort */ }
  return null;
}

/* Forecast loading copy lives in AIProcessingIndicator (lifecycle-driven
   stages); no timer-based stage list here. */

/* Rich hover tooltip: date, actual/predicted, CI, period change vs the
 * previous plotted point, and region context. All values come from the
 * plotted backend rows passed in via `rowsByLabel` — nothing invented. */
function ForecastTooltip({ active, payload, rowsByLabel, modelName }: any) {
  if (!active || !payload?.length) return null;
  const row = payload[0]?.payload;
  if (!row) return null;
  const isFwd = row.forecast !== null && row.forecast !== undefined;
  const info = (rowsByLabel || {})[row.label];
  const pct = info?.pct ?? null;
  return (
    <div className="rounded-lg border border-border px-3 py-2 shadow-md" style={{ background: "var(--card)" }}>
      <p className="text-[11px] font-bold text-foreground mb-1">{fullLabel(row.label)}</p>
      {row.actual !== null && row.actual !== undefined && (
        <p className="text-[11px] text-muted-foreground">
          <span className="inline-block w-2 h-2 rounded-full mr-1.5" style={{ background: ACTUAL_C }} />
          Actual: <span className="text-foreground font-semibold">{fmtNum(row.actual)}</span>
        </p>
      )}
      {isFwd && (
        <p className="text-[11px] text-muted-foreground">
          <span className="inline-block w-2 h-2 rounded-full mr-1.5" style={{ background: FORECAST_C }} />
          Predicted: <span className="text-foreground font-semibold">{fmtNum(row.forecast)}</span>
        </p>
      )}
      {isFwd && row.lower95 !== null && row.lower95 !== undefined && row.upper95 !== null && row.upper95 !== undefined && (
        <p className="text-[11px] text-muted-foreground">
          Likely range: <span className="text-foreground font-semibold">{fmtNum(row.lower95)} – {fmtNum(row.upper95)}</span>
        </p>
      )}
      {pct !== null && (
        <p className="text-[11px] text-muted-foreground">
          vs prior {isFwd ? "point" : "period"}:{" "}
          <span className={`font-semibold ${pct >= 0 ? "text-positive" : "text-negative"}`}>
            {pct >= 0 ? "+" : ""}{pct.toFixed(1)}%
          </span>
        </p>
      )}
      <p className="text-[10px] text-muted-foreground mt-1">
        {isFwd ? `Forecast${modelName ? ` · ${modelName}` : ""}` : "Historical actual"}
      </p>
    </div>
  );
}

export default function Predictive({ insightResult = null, goalResult = null, connectionStatus = "not-connected", dbName = "" }: PredictiveProps) {
  const { isDark } = useTheme();
  const [forecast, setForecast] = useState<any>(null);
  const [forecastInfo, setForecastInfo] = useState<any>(null);
  const [selection, setSelection] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [loadStage, setLoadStage] = useState(0);
  const [error, setError] = useState("");
  const [elig, setElig] = useState<any>(null);
  const [eligLoading, setEligLoading] = useState(false);
  const [eligError, setEligError] = useState("");
  const [selectedTarget, setSelectedTarget] = useState<string>("");
  const [selectedTable, setSelectedTable] = useState<string>("");
  const [selectedTime, setSelectedTime] = useState<string>("");
  const [horizon, setHorizon] = useState(6);
  const [tab, setTab] = useState<PeriodTab>("Monthly");
  const [filters, setFilters] = useState<{ column: string; value: string }[]>([]);
  const [filterCol, setFilterCol] = useState("");
  const [filterVal, setFilterVal] = useState("");
  const [schemaCols, setSchemaCols] = useState<any[]>([]);
  const [pendingCtx] = useState<ForecastLinkContext | null>(() => readForecastLinkContext());
  const [ctxApplied, setCtxApplied] = useState(false);
  const [pendingDateHint, setPendingDateHint] = useState<string | null>(null);
  const [win, setWin] = useState<{ s: number; e: number } | null>(null);
  const [pointSel, setPointSel] = useState<any | null>(null);
  const [showWhy, setShowWhy] = useState(false);
  const [confidenceOpen, setConfidenceOpen] = useState(false);
  const [revealKey, setRevealKey] = useState(0);
  const [scenarioPct, setScenarioPct] = useState(10);
  const [scenario, setScenario] = useState<any>(null);
  const [scenarioLoading, setScenarioLoading] = useState(false);
  const [scenarioError, setScenarioError] = useState("");
  // Request outcome: idle = no result yet, ok = plottable forecast,
  // unforecastable = backend answered ok:false (tailored empty card).
  const [forecastStatus, setForecastStatus] = useState<"idle" | "ok" | "unforecastable">("idle");
  // Verification verdict attached best-effort by the backend (never blocks data).
  const [forecastVerdict, setForecastVerdict] = useState<any>(null);
  // Single auto-fetch guard: one ref gates both auto-run effects so StrictMode
  // double-invoke and competing effects never fire two initial requests.
  const autoFetchRef = useRef<string>("");
  /* So-consider actions: top prescriptions for the forecast target, fetched
     once per forecast from the existing engine. Honest omission on refusal. */
  const [rxItems, setRxItems] = useState<any[]>([]);
  const rxFor = useRef<string>("");
  useEffect(() => {
    const m = String((forecast as any)?.column || (forecast as any)?.target || selectedTarget || "").trim();
    if (!forecast || !m) {
      rxFor.current = "";
      setRxItems([]);
      return;
    }
    if (rxFor.current === m) return;
    rxFor.current = m;
    let cancelled = false;
    (async () => {
      try {
        const r = await (prescriptionApi as any).analyze({ metric: m });
        const d: any = (r as any)?.data ?? null;
        const list = Array.isArray(d?.prescriptions) ? d.prescriptions : [];
        if (!cancelled) setRxItems(list.slice(0, 2));
      } catch {
        if (!cancelled) setRxItems([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [forecast, selectedTarget]);
  const fcToggle = useSeriesToggle(["Actual", "Forecast", "Baseline", "Likely range"]);
  // Forecast race guard: abort the obsolete in-flight request before a new
  // one starts (double-click Refresh / fast target switches).
  const forecastAbort = useRef<AbortController | null>(null);

  // (Single client) apiFetch sends credentials + X-Request-ID and
  // handles 401 centrally; no per-call auth header is needed.

  // ---- Eligibility: what can actually be forecast ----
  const loadEligibility = useCallback(async () => {
    setEligLoading(true);
    setEligError("");
    try {
      const data: any = await apiFetch("/predictive/eligibility");
      if (data.ok) {
        setElig(data);
        const rec = data.recommended;
        if (rec) {
          setSelectedTarget((prev) => prev !== "" ? prev : (rec.measure || ""));
          setSelectedTable((prev) => prev !== "" ? prev : (rec.table || ""));
          setSelectedTime((prev) => prev !== "" ? prev : (rec.time_dim || ""));
        }
      } else {
        setEligError(data.detail || "Couldn't inspect the data for forecasting.");
      }
    } catch (e: any) {
      if (e?.status === 401) { setEligError("Session expired. Please sign in again."); return; }
      const msg = String(e?.message || "");
      setEligError(msg === "Failed to fetch"
        ? "ARIA couldn't reach the server. Check that the backend is running, then retry."
        : (normalizeApiError({ message: msg }, 0, null).message || "Couldn't inspect the data for forecasting."));
    } finally {
      setEligLoading(false);
    }
  }, []);

  useEffect(() => {
    if (connectionStatus === "connected") loadEligibility();
    else { setElig(null); setForecast(null); setError(""); setForecastStatus("idle"); setForecastVerdict(null); autoFetchRef.current = ""; }
  }, [connectionStatus, loadEligibility]);
  // Also re-fetch when DB name changes (user switched databases without disconnecting).
  const prevDbRef = useRef(dbName);
  useEffect(() => {
    if (prevDbRef.current !== dbName) {
      prevDbRef.current = dbName;
      forecastAbort.current?.abort();
      setElig(null); setForecast(null); setError(""); setScenario(null);
      setEligError(""); setLoading(false); setForecastStatus("idle"); setForecastVerdict(null); autoFetchRef.current = "";
      setSelectedTarget(""); setSelectedTable(""); setSelectedTime("");
      setFilters([]); setSchemaCols([]); setRxItems([]); rxFor.current = "";
      if (connectionStatus === "connected") loadEligibility();
    }
  }, [dbName, connectionStatus, loadEligibility]);

  const tables: any[] = elig?.tables || [];
  const activeTable = tables.find((t) => t.table === selectedTable) || tables.find((t) => (t.measures || []).includes(selectedTarget)) || null;
  const activeMeasures: string[] = activeTable?.measures || [];
  const activeTimes: any[] = activeTable?.time_dims || [];
  const historyPoints = activeTable && selectedTime
    ? (activeTimes.find((d) => d.column === selectedTime)?.distinct ?? null)
    : (elig?.recommended?.history_points ?? null);

  // ---- Filter dimensions: non-numeric columns of the selected dataset,
  // from the real backend schema (/api/data/columns). Values are typed by
  // the user; the filter travels inside the forecast question. ----
  useEffect(() => {
    if (connectionStatus !== "connected" || !selectedTable) { setSchemaCols([]); return; }
    let cancelled = false;
    (async () => {
      try {
        const data: any = await apiFetch(`/data/columns?table=${encodeURIComponent(selectedTable)}`);
        const cols: any[] = data?.columns || data?.column_details || data?.data?.columns || [];
        const dims = cols
          .map((c: any) => ({
            name: String(c?.column ?? c?.name ?? ""),
            dtype: String(c?.data_type ?? c?.dtype ?? c?.type ?? "").toLowerCase(),
          }))
          .filter((c: any) => c.name && !/(int|float|double|real|numeric|decimal|number)/.test(c.dtype));
        if (!cancelled) {
          setSchemaCols(dims);
          setFilterCol((prev) => (dims.some((d: any) => d.name === prev) ? prev : (dims[0]?.name || "")));
        }
      } catch {
        if (!cancelled) setSchemaCols([]);
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connectionStatus, selectedTable]);

  const runScenario = useCallback(async () => {
    setScenarioLoading(true);
    setScenarioError("");
    try {
      const data: any = await apiFetch("/predictive/scenario", {
        method: "POST",
        body: JSON.stringify({
          column: plainName(selectedTarget) || undefined,
          pct_change: Number(scenarioPct) || 0,
          periods: 3,
        }),
      });
      if (data.ok) {
        setScenario(data);
      } else {
        setScenario(null);
        setScenarioError(data.detail || "Couldn't run that scenario. Try a different number.");
      }
    } catch (e: any) {
      setScenario(null);
      setScenarioError(e?.status === 401
        ? "Session expired. Please sign in again."
        : (e?.message || "Couldn't run that scenario."));
    } finally {
      setScenarioLoading(false);
    }
  }, [selectedTarget, scenarioPct]);

  const grid = isDark ? "#2A2A35" : "#E5E7EB";
  const axis = isDark ? "#8889A0" : "#9CA3AF";

  const insightPredictions = insightResult?.predictions || {};
  const hasInsightPredictions = insightPredictions && (insightPredictions.valid || insightPredictions.points?.length > 0);

  const buildQuestion = (t: string, td: string, flt: { column: string; value: string }[]) => {
    let q = td ? `Forecast ${t} by ${td} over time` : `Forecast ${t} over time`;
    for (const f of flt) {
      if (f.column && f.value) q += ` with ${f.column} = ${f.value}`;
    }
    return q;
  };

  const fetchForecast = useCallback(async (target?: string, timeDim?: string, h?: number, flt?: { column: string; value: string }[]) => {
    const t = plainName(target || selectedTarget || "");
    const td = plainName(timeDim || selectedTime || "");
    const hr = Math.max(1, Math.min(24, h !== undefined ? h : horizon));
    const useFlt = flt !== undefined ? flt : filters;
    if (!t) return;

    // Cancel any obsolete in-flight forecast, then start the new one.
    forecastAbort.current?.abort();
    const ctrl = new AbortController();
    forecastAbort.current = ctrl;
    setLoading(true);
    // Lifecycle-driven stage (real milestones, never timers):
    // 1 = request in flight (Analyzing patterns); 2 = response received,
    // building the reveal (Finding evidence); the forecast-reveal + CI
    // fade keyed by revealKey is stage 3 (Generating visualization).
    setLoadStage(1);
    setError("");
    setPointSel(null);
    try {
      const data: any = await apiFetch("/predictive/forecast", {
        method: "POST",
        signal: ctrl.signal,
        body: JSON.stringify({
          question: buildQuestion(t, td, useFlt),
          column: t,
          time_dim: td || undefined,
          horizon: hr,
        }),
      });
      // Response received — evidence stage before the reveal builds.
      setLoadStage(2);
      if (data.ok && data.forecast) {
        setForecast(data.forecast);
        setForecastStatus("ok");
        setForecastInfo(data.info || null);
        setSelection(data.selection || null);
        setForecastVerdict((data as any)?.verdict || null);
        setRevealKey((k) => k + 1);
        // Share context back (metric/dataset/date/filters preserved) so
        // cross-module consumers can preselect without re-analysis.
        try {
          saveAnalysisContext({
            origin: "forecast",
            question: buildQuestion(t, td, useFlt),
            metric: t || null,
            dataset: selectedTable || null,
            table: selectedTable || null,
            dateRange: td || null,
            dimension: null,
            filters: (useFlt || []).length
              ? Object.fromEntries(useFlt.filter((f) => f.column && f.value).map((f) => [f.column, f.value]))
              : null,
          });
        } catch { /* best-effort */ }
        const f = data.forecast;
        if (!f.valid && !(f.points?.length > 0) && !((f.historical || []).length > 0) && !((f.forecast || []).length > 0)) {
          if ((data.info?.rows_analyzed ?? 1) === 0) {
            setError("There's no data to forecast yet. Connect a source with numbers tracked over time, then try again.");
          } else {
            setError(f.reason || "Couldn't build a reliable forecast from this data. Try a different number above.");
          }
        }
      } else if (data && data.ok === false) {
        // Genuinely unforecastable: backend refused with reason + suggestion.
        // Track status so the tailored empty card renders (forecast stays null).
        setForecast(null);
        setForecastStatus("unforecastable");
        setForecastInfo(data.info || null);
        setSelection(data.selection || null);
        setForecastVerdict((data as any)?.verdict || null);
        setError(data.reason || data.detail || "This selection can't be forecast from the available history.");
      } else {
        setError(data.detail || "Couldn't build a forecast. Try a different number above.");
        setForecast(null);
        setForecastStatus("idle");
        setForecastVerdict((data as any)?.verdict || null);
      }
    } catch (e: any) {
      if (e?.name === "AbortError" || ctrl.signal.aborted) return; // superseded — not an error state
      const msg = String(e?.message || "");
      setError(msg === "Failed to fetch"
        ? "ARIA couldn't reach the server. Check that the backend is running, then retry."
        : (e.message || "Couldn't build a forecast. Check your connection and try again."));
      setForecast(null);
      setForecastStatus("idle");
      setForecastVerdict(null);
    } finally {
      // Only the latest request clears loading (an aborted older one must
      // not flicker the spinner off while its replacement is in flight).
      if (forecastAbort.current === ctrl) {
        forecastAbort.current = null;
        setLoading(false);
      }
    }
  }, [selectedTarget, selectedTime, horizon, filters]);

  // Cross-module prefill: localStorage `aria_forecast_context` (preferred) or
  // legacy sessionStorage prefill. Applies Metric/Dataset/Date/Filters/
  // Horizon/Granularity once eligibility is known, then forecasts directly.
  useEffect(() => {
    if (connectionStatus !== "connected" || !elig || ctxApplied) return;
    if (!elig.eligible) { setCtxApplied(true); return; }
    if (!pendingCtx) return;
    const ctx: any = { ...pendingCtx };
    // Shared cross-module context (localStorage aria_analysis_context +
    // hash params) fills gaps the link context doesn't carry — metric,
    // dataset, date hint, filters, dimension. Never invents values.
    try {
      const shared = loadAnalysisContext();
      const hash = readHashParams();
      if (shared || Object.keys(hash).length) {
        const s: any = shared || {};
        if (!ctx.metric) ctx.metric = s.metric || hash.metric || ctx.metric;
        if (!ctx.dataset && !ctx.table) ctx.dataset = s.dataset || s.table || hash.dataset || hash.table;
        if (!ctx.date && !ctx.period) ctx.date = s.period || s.dateRange || hash.period || hash.date || undefined;
        if ((!ctx.filters || ctx.filters.length === 0) && s.filters && typeof s.filters === "object") {
          ctx.filters = Object.entries(s.filters).map(([column, value]) => ({ column, value: String(value) }));
        }
        if (!ctx.question && s.question) ctx.question = s.question;
      }
    } catch { /* best-effort */ }
    const tbls: any[] = elig.tables || [];
    const like = (s: any) => plainName(String(s || ""));
    let table = tbls.find((x) => x.table === (ctx.dataset || ctx.table)) || null;
    if (!table && ctx.metric) table = tbls.find((x) => (x.measures || []).includes(like(ctx.metric))) || null;
    table = table || tbls.find((x) => x.table === elig.recommended?.table) || tbls[0];
    if (!table) { setCtxApplied(true); return; }
    const metric = (ctx.metric && (table.measures || []).includes(like(ctx.metric)))
      ? like(ctx.metric)
      : ((table.table === selectedTable && selectedTarget && (table.measures || []).includes(selectedTarget))
        ? selectedTarget
        : (table.table === elig.recommended?.table ? elig.recommended.measure : (table.measures || [])[0]));
    const tds: any[] = table.time_dims || [];
    const wantTd = like(ctx.time_dim || ctx.timeDim || "");
    const timeDim = (wantTd && tds.some((d) => d.column === wantTd))
      ? wantTd
      : (tds[0]?.column || elig.recommended?.time_dim || "");
    const hz = ctx.horizon ? Math.max(1, Math.min(24, Number(ctx.horizon) || 6)) : horizon;
    const flt = normalizeLinkFilters(ctx.filters);
    const gran = String(ctx.granularity || "").toLowerCase();
    const gtab: PeriodTab | null = gran.startsWith("quarter") ? "Quarterly" : gran.startsWith("year") ? "Yearly" : gran.startsWith("month") ? "Monthly" : null;
    setSelectedTable(table.table);
    setSelectedTarget(metric || "");
    setSelectedTime(timeDim);
    setHorizon(hz);
    if (gtab) setTab(gtab);
    if (flt.length) setFilters(flt);
    const hint = like(ctx.date || ctx.period || "");
    if (hint) setPendingDateHint(hint);
    setCtxApplied(true);
    // Never force re-analysis: preselect only. Auto-fetch only when there
    // is no usable forecast yet; an existing result stays untouched.
    // Single auto-fetch ref gates this path so StrictMode + the default
    // auto-run below never fire two initial requests for the same target.
    if (metric) {
      const empty = !forecast || (!Array.isArray((forecast as any)?.forecast) && !Array.isArray((forecast as any)?.points)
        && !Array.isArray((forecast as any)?.historical) && !Array.isArray((forecast as any)?.history))
        || (((forecast as any)?.forecast || (forecast as any)?.points || []).length === 0
          && ((forecast as any)?.historical || (forecast as any)?.history || []).length === 0);
      if (empty && !hasInsightPredictions) {
        const key = `ctx:${table.table}|${metric}|${timeDim}|${hz}`;
        if (autoFetchRef.current !== key) {
          autoFetchRef.current = key;
          const timer = setTimeout(() => fetchForecast(metric, timeDim, hz, flt), 300);
          return () => clearTimeout(timer);
        }
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connectionStatus, elig, ctxApplied]);

  // Auto-run once eligibility + a valid default exist (skipped when a
  // cross-link context is driving the first run). Shares the same single
  // auto-fetch ref so only one initial request ever fires.
  useEffect(() => {
    if (connectionStatus !== "connected" || !elig?.eligible || !selectedTarget) return;
    if (pendingCtx && !ctxApplied) return;
    const empty = !forecast || (!Array.isArray(forecast?.forecast) && !Array.isArray(forecast?.points)
      && !Array.isArray(forecast?.historical) && !Array.isArray(forecast?.history))
      || ((forecast?.forecast || forecast?.points || []).length === 0
        && (forecast?.historical || forecast?.history || []).length === 0);
    if (!hasInsightPredictions && empty) {
      const key = `auto:${selectedTable}|${selectedTarget}|${selectedTime}`;
      if (autoFetchRef.current === key) return;
      autoFetchRef.current = key;
      const timer = setTimeout(() => fetchForecast(), 500);
      return () => clearTimeout(timer);
    }
  }, [connectionStatus, elig, selectedTarget, hasInsightPredictions, fetchForecast, pendingCtx, ctxApplied]);

  const useAutomatic = () => {
    const rec = elig?.recommended;
    if (!rec) return;
    setSelectedTable(rec.table);
    setSelectedTarget(rec.measure);
    setSelectedTime(rec.time_dim);
    setFilters([]);
    setWin(null);
    setForecast(null);
    setForecastStatus("idle");
    setForecastVerdict(null);
    autoFetchRef.current = "";
    setError("");
    setTimeout(() => fetchForecast(rec.measure, rec.time_dim, horizon, []), 50);
  };

  const predictions = forecast || insightPredictions;
  // Accept the data contract (historical/forecast) or the legacy shape
  // (history/points). Either way every plotted timestamp comes from the
  // backend — entries without a backend date are dropped, never indexed.
  // Memoized so chartRows below has stable inputs (single memoized source).
  const rawHistorical: any[] = useMemo(() => {
    if (Array.isArray((predictions as any)?.historical)) return (predictions as any).historical;
    if (Array.isArray(predictions?.history)) return predictions.history;
    return [];
  }, [predictions]);
  const rawFuture: any[] = useMemo(() => {
    if (Array.isArray((predictions as any)?.forecast)) return (predictions as any).forecast;
    if (Array.isArray((predictions as any)?.forecast_series)) return (predictions as any).forecast_series;
    if (Array.isArray(predictions?.points)) return predictions.points;
    return [];
  }, [predictions]);
  const hasPredictions = predictions && (
    predictions.valid || rawFuture.length > 0 || rawHistorical.length > 0);
  const targetName = plainName(
    selection?.measure || (predictions as any)?.target || predictions?.column || selectedTarget);
  const timeName = plainName(selection?.time_dim || selectedTime);
  // Backend accuracy lives in two shapes: accuracy_pct (number) or
  // accuracy: { pct, summary }. Support both verbatim.
  const accuracy: number | null = (() => {
    const a = (predictions as any)?.accuracy_pct;
    if (a !== null && a !== undefined && !isNaN(Number(a))) return Number(a);
    const b = (predictions as any)?.accuracy?.pct;
    if (b !== null && b !== undefined && !isNaN(Number(b))) return Number(b);
    return null;
  })();
  const backendReliability: string | null = (predictions as any)?.reliability ?? null;
  const backendValid: boolean | null = typeof (predictions as any)?.valid === "boolean" ? (predictions as any).valid : null;
  const forecastStage: string | null = (predictions as any)?.forecast_stage ?? null;
  // Counts come from the backend total, never from a request-side row count.
  // The chart shows at most the last 24 backend periods; the label below the
  // chart states the full total.
  const nObs: number = (() => {
    const n = Number((predictions as any)?.n_obs);
    if (Number.isFinite(n) && n > 0) return Math.round(n);
    return 0;
  })();
  const contractMetrics: any = (predictions as any)?.metrics || null;
  const validation: any = (predictions as any)?.validation || null;
  const backendDrivers: any[] = Array.isArray((predictions as any)?.drivers) ? (predictions as any).drivers : [];
  const modelName: string | null = (predictions as any)?.model || predictions?.method || null;
  const frequency: string | null = (predictions as any)?.frequency || null;
  const selectionStatement: string | null = (predictions as any)?.selection_statement || null;
  const uncertaintyNote: string | null = (predictions as any)?.uncertainty_note || null;
  // Regrouped buckets sum the backend bounds, so the band is an
  // approximation of the regrouped range — never pointwise certainty.
  const intervalMethod: string = tab === "Monthly"
    ? ((predictions as any)?.interval_method || "pointwise")
    : "summed_bounds_approx";

  const histDate = (h: any) => h?.date ?? h?.label ?? null;
  const histVal = (h: any) => h?.value ?? h?.actual ?? null;
  const futDate = (p: any) => p?.date ?? p?.label ?? null;
  const futVal = (p: any) => p?.value ?? p?.forecast ?? null;
  const futLow = (p: any) => p?.lower ?? p?.lower95 ?? null;
  const futHigh = (p: any) => p?.upper ?? p?.upper95 ?? null;

  const chartRows = useMemo(() => {
    const scenRx = rxItems.find(
      (p: any) =>
        String(p?.action_type || "").toLowerCase() === "scenario" &&
        p?.impact?.rel_pct !== null &&
        p?.impact?.rel_pct !== undefined,
    );
    const relPct = scenRx ? Number(scenRx.impact.rel_pct) : null;
    const rows: any[] = [
      ...rawHistorical
        .filter((h: any) => histDate(h) !== null && histDate(h) !== "" && histVal(h) !== null)
        .map((h: any) => ({
          // x-axis = backend timestamp, verbatim. `short` is display-only.
          label: String(histDate(h)),
          short: shortLabel(String(histDate(h))),
          actual: Number(histVal(h)),
          forecast: null,
          lower95: null,
          upper95: null,
        })),
      ...rawFuture
        .filter((p: any) => futDate(p) !== null && futDate(p) !== "" && futVal(p) !== null)
        .map((p: any) => ({
          label: String(futDate(p)),
          short: shortLabel(String(futDate(p))),
          actual: null,
          forecast: Number(futVal(p)),
          // If-we-act estimate: forecast shifted by the modeled scenario
          // percent from a real scenario prescription for this metric.
          // Omitted entirely when no scenario action exists.
          act:
            relPct !== null && Number.isFinite(Number(futVal(p)))
              ? Math.round(Number(futVal(p)) * (1 + relPct / 100) * 100) / 100
              : null,
          lower95: futLow(p) !== null ? Number(futLow(p)) : null,
          upper95: futHigh(p) !== null ? Number(futHigh(p)) : null,
        })),
    ];
    if (rows.length === 0) return rows;
    const rebinned = rebin(rows, tab);
    // Rebinning groups backend months (Quarterly/Yearly); it must never
    // invent or drop the series — fall back to raw rows if it empties them.
    // Forecast points are never plotted as history: actual stays null on
    // future rows and forecast stays null on history rows, before and after.
    // Zero-width intervals are never widened here — backend values stay
    // untouched; the uncertain marker below carries the meaning.
    const base = rebinned.length > 0 ? rebinned : rows;
    return base.map((r: any) => ({ ...r, short: r.short || shortLabel(r.label) }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rawHistorical, rawFuture, tab, rxItems]);

  // Stable forecast identity for the zoom window: a new forecast resets the
  // window, tab switches and plain re-renders never do.
  const forecastId = useMemo(() => {
    const parts = [
      selection?.measure || targetName || "",
      selection?.table || selectedTable || "",
      selection?.time_dim || timeName || "",
      String(nObs || rawHistorical.length || 0),
      String(rawFuture.length || 0),
    ];
    return parts.join("|");
  }, [selection, targetName, selectedTable, timeName, nObs, rawHistorical.length, rawFuture.length]);
  // Any new forecast resets the zoom/range window — stale indices must never
  // window the wrong data.
  useEffect(() => { setWin(null); }, [forecastId]);

  // Cross-link date hint: jump the range window to the first backend
  // timestamp containing the hinted date, once rows exist.
  useEffect(() => {
    if (!pendingDateHint || chartRows.length === 0) return;
    const idx = chartRows.findIndex((r) => String(r.label).toLowerCase().includes(pendingDateHint.toLowerCase()));
    if (idx >= 0) setWin({ s: Math.max(0, idx - 2), e: chartRows.length - 1 });
    setPendingDateHint(null);
  }, [pendingDateHint, chartRows]);

  const viewRows = useMemo(
    () => (win ? chartRows.slice(Math.max(0, win.s), Math.min(chartRows.length, win.e + 1)) : chartRows),
    [chartRows, win],
  );

  // Per-point lookup for tooltip + popup: value + change vs previous row,
  // plus index/total so boundary points label themselves.
  const rowsByLabel = useMemo(() => {
    const map: Record<string, { pct: number | null; prevLabel: string | null; idx: number }> = {};
    chartRows.forEach((r: any, i: number) => {
      const v = r.actual !== null && r.actual !== undefined ? Number(r.actual)
        : (r.forecast !== null && r.forecast !== undefined ? Number(r.forecast) : null);
      let pct: number | null = null;
      let prevLabel: string | null = null;
      for (let j = i - 1; j >= 0; j--) {
        const pv = chartRows[j].actual !== null && chartRows[j].actual !== undefined ? Number(chartRows[j].actual)
          : (chartRows[j].forecast !== null && chartRows[j].forecast !== undefined ? Number(chartRows[j].forecast) : null);
        if (pv !== null && !isNaN(pv)) {
          prevLabel = chartRows[j].label;
          if (v !== null && !isNaN(v) && pv !== 0) pct = ((v - pv) / Math.abs(pv)) * 100;
          break;
        }
      }
      map[r.label] = { pct, prevLabel, idx: i };
    });
    return map;
  }, [chartRows]);

  const historyVals: number[] = useMemo(() => {
    return rawHistorical
      .map((h: any) => Number(histVal(h)))
      .filter((v: number) => !isNaN(v));
  }, [rawHistorical]);

  const lastActual = predictions?.last_actual?.value ?? (historyVals.length ? historyVals[historyVals.length - 1] : null);
  const fwdPts: any[] = rawFuture;
  const fwdVals = fwdPts.map((p: any) => Number(futVal(p))).filter((v: number) => !isNaN(v));
  const projectedTotal = fwdVals.reduce((a, b) => a + b, 0);
  const avgActual = historyVals.length ? historyVals.reduce((a, b) => a + b, 0) / historyVals.length : null;
  const pctChange = predictions?.projected_pct_change ?? null;
  // Counts come from the backend total, never from a request-side row count.
  // The plotted history is capped at the last 24 backend periods; the full
  // total below drives the preliminary flag and the "Showing last …" label.
  const historyCount = nObs > 0 ? nObs : (historyVals.length || forecastInfo?.rows_analyzed || 0);
  const plottedHistory = rawHistorical.length;
  const historyLabel = nObs > plottedHistory && plottedHistory > 0
    ? `Showing last ${plottedHistory} of ${nObs} periods`
    : (historyCount > 0 ? `${historyCount} past ${historyCount === 1 ? "period" : "periods"}` : "");

  const sumKey = (k: string) => fwdPts.reduce((a, p) => {
    const v = k === "upper95" ? futHigh(p) : k === "lower95" ? futLow(p) : (p as any)?.[k];
    return a + (Number(v) || 0);
  }, 0);
  // Likely-range cards derived from the backend range (upper/central/lower).
  // Monthly is authoritative; regrouped Quarterly/Yearly buckets sum the
  // backend bounds so they are marked approximate in the UI.
  const scenarios = useMemo(() => {
    if (!fwdPts.length || lastActual === null) return null;
    const base = (avgActual !== null ? avgActual * Math.max(1, fwdVals.length) : Math.abs(Number(lastActual)) * Math.max(1, fwdVals.length)) || 1;
    const mk = (label: string, total: number, tone: string, note: string, sub: string) => {
      const pct = ((total - base) / Math.abs(base)) * 100;
      return { label, total, tone, note, sub, pct, ofTarget: base ? (total / base) * 100 : 100 };
    };
    return [
      mk("Likely range — upper", sumKey("upper95") || projectedTotal, "good", "Upper edge of the expected range", "If momentum holds"),
      mk("Central projection", projectedTotal, "base", "Central projection", "Based on recent trend"),
      mk("Likely range — lower", sumKey("lower95") || projectedTotal, "bad", "Lower edge of the expected range", "If momentum slows"),
    ];
  }, [fwdPts, lastActual, avgActual, projectedTotal, fwdVals.length]);
  const scenMax = scenarios ? Math.max(...scenarios.map((s) => s.total), 1) : 1;

  const recentTop = useMemo(() => {
    const total = historyVals.reduce((a, b) => a + b, 0);
    return [...rawHistorical]
      .sort((a: any, b: any) => Number(histVal(b)) - Number(histVal(a)))
      .slice(0, 3)
      .map((h: any) => ({ label: shortLabel(String(histDate(h))), value: Number(histVal(h)), share: total ? (Number(histVal(h)) / total) * 100 : 0 }));
  }, [rawHistorical, historyVals]);

  const momDelta = historyVals.length >= 2 && historyVals[historyVals.length - 2]
    ? ((historyVals[historyVals.length - 1] - historyVals[historyVals.length - 2]) / Math.abs(historyVals[historyVals.length - 2])) * 100
    : null;

  // Dynamic forecast story — 2–4 sentences generated from the actual
  // scenario values, never static copy (see buildForecastStory).
  const forecastStory: { story: string; signal: string } = useMemo(() => {
    return buildForecastStory({
      hasPredictions, scenarios, targetName,
      horizonPeriods: fwdVals.length, accuracy, momDelta,
    });
  }, [hasPredictions, scenarios, targetName, fwdVals.length, accuracy, momDelta]);

  // Preliminary comes from the backend total: the backend marks fewer than
  // 50 observations preliminary, mirrored here via forecast_stage + n_obs.
  const isPreliminary = !!hasPredictions && (forecastStage === "preliminary" || (historyCount > 0 && historyCount < 50));
  // Unforecastable: backend answered ok:false (tracked status), so the
  // tailored empty card renders instead of the generic error.
  const isUnforecastable = forecastStatus === "unforecastable";

  // ---- Confidence shows backend reliability/valid verbatim. High only when
  // the backend says so AND valid is true AND not preliminary AND backtested
  // accuracy is at least 85. ----
  const confidenceLevel: "High" | "Moderate" | "Limited" = (() => {
    const rel = String(backendReliability || "").toLowerCase();
    let base: "High" | "Moderate" | "Limited";
    if (rel === "high" || rel === "moderate" || rel === "limited") {
      base = rel === "high" ? "High" : rel === "moderate" ? "Moderate" : "Limited";
    } else {
      base = accuracy !== null && accuracy >= 85 ? "High" : accuracy !== null && accuracy >= 70 ? "Moderate" : "Limited";
    }
    if (base === "High" && !(backendValid === true && !isPreliminary && accuracy !== null && accuracy >= 85)) {
      base = "Moderate";
      if (backendValid === false || accuracy === null || accuracy < 70) base = "Limited";
    }
    return base;
  })();
  const coverage = validation ? {
    origins: validation.walk_forward_origins ?? null,
    evaluated: validation.walk_forward_evaluated ?? null,
    zeroSkipped: validation.zero_actuals_skipped ?? null,
  } : null;

  // ---- "Why ARIA expects this": backend explanation fields + backend
  // drivers[] rendered verbatim (name + evidence). Nothing invented. ----
  const whyItems: { title: string; body: string; source: string }[] = useMemo(() => {
    if (!hasPredictions) return [];
    const items: { title: string; body: string; source: string }[] = [];
    if (selectionStatement) items.push({ title: "Model selection", body: selectionStatement, source: "selection_statement" });
    if (modelName && modelName !== "none") items.push({ title: "Approach", body: `Projected with ${modelName}, the approach that held up best against past actuals.`, source: "model" });
    for (const d of backendDrivers) {
      const name = String(d?.name || "").trim();
      if (!name) continue;
      const ev = String(d?.evidence || "").trim();
      const mv = d?.metric_value ?? d?.value ?? null;
      const body = ev
        ? (mv !== null && mv !== undefined && String(mv).trim() !== "" ? `${ev} (value: ${mv})` : ev)
        : (mv !== null && mv !== undefined ? `Observed value: ${mv}.` : "Observed in your history.");
      items.push({ title: name, body, source: "drivers" });
    }
    if (pctChange !== null && pctChange !== undefined) items.push({
      title: "Expected move",
      body: `The final projected point sits ${Math.abs(pctChange).toFixed(1)}% ${pctChange >= 0 ? "above" : "below"} the last actual value.`,
      source: "projected_pct_change",
    });
    if (momDelta !== null) items.push({
      title: "Recent momentum",
      body: momDelta > 1 ? `The last two actuals rose ${momDelta.toFixed(1)}%, and the projection carries that direction forward.`
        : momDelta < -1 ? `The last two actuals fell ${Math.abs(momDelta).toFixed(1)}%, and the projection carries that direction forward.`
        : "The last two actuals sit near their average, so the projection holds close to the current pace.",
      source: "observed history",
    });
    if (validation?.reason) items.push({ title: "Validation", body: validation.reason, source: "validation.reason" });
    if (frequency) items.push({ title: "Time base", body: `History was aggregated per ${frequency === "MS" ? "month" : frequency === "W" ? "week" : frequency === "D" ? "day" : frequency}.`, source: "frequency" });
    return items;
  }, [hasPredictions, selectionStatement, modelName, backendDrivers, pctChange, momDelta, validation, frequency]);
  const topSignal = forecastStory.signal || (whyItems[0] ? `${whyItems[0].title}: ${whyItems[0].body}` : "");

  const horizonWarning = historyPoints !== null && horizon > historyPoints
    ? `Heads up: ${horizon} periods ahead from only ${historyPoints} past points — treat the far end as directional.`
    : null;

  // ---- Zoom / pan / range: one index window over the backend rows. ----
  const zoomDepth = win ? (win.e - win.s + 1) / Math.max(1, chartRows.length) : 1;
  const zoomIn = () => {
    if (chartRows.length < 3) return;
    setWin((w) => {
      const s = w ? w.s : 0, e = w ? w.e : chartRows.length - 1;
      const span = e - s + 1;
      if (span <= 3) return w;
      const cut = Math.max(1, Math.floor(span * 0.2));
      return { s: s + cut, e: e - cut };
    });
  };
  const zoomOut = () => {
    if (chartRows.length < 3) return;
    setWin((w) => {
      if (!w) return null;
      const span = w.e - w.s + 1;
      const grow = Math.max(1, Math.floor(span * 0.25));
      const s = Math.max(0, w.s - grow), e = Math.min(chartRows.length - 1, w.e + grow);
      return (s === 0 && e === chartRows.length - 1) ? null : { s, e };
    });
  };
  const pan = (dir: -1 | 1) => {
    if (chartRows.length < 3) return;
    setWin((w) => {
      if (!w) return null;
      const span = w.e - w.s + 1;
      const step = Math.max(1, Math.floor(span * 0.25)) * dir;
      let s = w.s + step, e = w.e + step;
      if (s < 0) { e -= s; s = 0; }
      if (e > chartRows.length - 1) { s -= (e - (chartRows.length - 1)); e = chartRows.length - 1; }
      s = Math.max(0, s);
      return { s, e };
    });
  };
  const resetView = () => setWin(null);

  const applyRange = (from: string, to: string) => {
    if (!from && !to) { setWin(null); return; }
    const labels = chartRows.map((r) => r.label);
    let s = from ? labels.indexOf(from) : 0;
    let e = to ? labels.indexOf(to) : labels.length - 1;
    if (s < 0) s = 0;
    if (e < 0) e = labels.length - 1;
    if (s > e) [s, e] = [e, s];
    if (s === 0 && e === labels.length - 1) setWin(null);
    else setWin({ s, e });
  };
  const rangeFrom = win ? chartRows[Math.max(0, win.s)]?.label || "" : "";
  const rangeTo = win ? chartRows[Math.min(chartRows.length - 1, win.e)]?.label || "" : "";
  // Regrouping is only offered when every plotted label parses as a date.
  // Monthly stays authoritative; Quarterly/Yearly are approximate sums.
  const canRebin = useMemo(() => {
    if (chartRows.length === 0) return true;
    return labelsParseAsDates(chartRows);
  }, [chartRows]);
  // Zero-width intervals are never widened — flag for the uncertain marker.
  const hasZeroWidth = useMemo(() => {
    return chartRows.some((r: any) =>
      r.forecast !== null && r.forecast !== undefined
      && r.lower95 !== null && r.lower95 !== undefined
      && r.upper95 !== null && r.upper95 !== undefined
      && Number(r.lower95) === Number(r.upper95));
  }, [chartRows]);
  // Lock regroup tabs when labels don't parse as dates.
  useEffect(() => {
    if (!canRebin && tab !== "Monthly") setTab("Monthly");
  }, [canRebin, tab]);

  const exportCsv = () => {
    if (chartRows.length === 0) return;
    // Full series export (not just the zoomed view) with provenance header.
    const prov: string[] = [
      `# metric,${String(targetName || "").replace(/,/g, ";")}`,
      `# dataset,${String(selectedTable || "").replace(/,/g, ";")}`,
      `# time field,${String(timeName || "").replace(/,/g, ";")}`,
      `# generated,${new Date().toISOString()}`,
      `# model,${String(modelName || "").replace(/,/g, ";")}`,
      `# observations,${historyCount}`,
      `# horizon,${fwdVals.length}`,
      `# reliability,${String(backendReliability || "").replace(/,/g, ";")}`,
      `# valid,${backendValid === null ? "" : String(backendValid)}`,
    ];
    const header = "date,actual,forecast,lower95,upper95";
    const lines = chartRows.map((r: any) =>
      [r.label, r.actual ?? "", r.forecast ?? "", r.lower95 ?? "", r.upper95 ?? ""].join(","));
    const blob = new Blob([[...prov, header, ...lines].join("\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `forecast-${(targetName || "metric").replace(/[^a-z0-9-_]+/gi, "_")}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const openPoint = (row: any) => {
    if (!row) return;
    const info = rowsByLabel[row.label] || { pct: null, prevLabel: null, idx: -1 };
    const idx = typeof (info as any).idx === "number" ? (info as any).idx : chartRows.findIndex((r: any) => r.label === row.label);
    const total = chartRows.length;
    const isFirst = idx === 0;
    const isLast = idx === total - 1;
    const kind = row.forecast !== null && row.forecast !== undefined
      ? (row.actual !== null && row.actual !== undefined ? "history + forecast" : "forecast")
      : "history";
    setPointSel({ ...row, pct: info.pct, prevLabel: info.prevLabel, idx, total, isFirst, isLast, kind });
  };

  if (connectionStatus !== "connected") {
    return (
      <div className="p-6 max-w-screen-xl mx-auto">
        <div className="text-center py-20">
          <div className="w-16 h-16 rounded-2xl mx-auto mb-5 flex items-center justify-center bg-primary">
            <TrendingUp size={24} className="text-primary-foreground" />
          </div>
          <h2 className="text-2xl font-semibold text-foreground tracking-tight mb-2">Forecast</h2>
          <p className="text-sm text-muted-foreground mb-4 max-w-sm mx-auto">
            See where your business is heading. Connect your data and ARIA will project your key numbers into the future.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-screen-xl mx-auto space-y-4">
      <style>{`
        @keyframes fc-fade { from { opacity: 0; } to { opacity: 1; } }
        .fc-skeleton { animation: fc-fade 0.4s ease-out 1 both; }
      `}</style>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-bold text-foreground tracking-tight flex items-center gap-2">
            Forecast
            {(() => {
              const v: any = forecastVerdict;
              if (!v) return null;
              const ok = v.verified === true;
              const fails = Array.isArray(v.checks) ? v.checks.filter((c: any) => !c?.pass).length : 0;
              return (
                <span
                  title={ok
                    ? "ARIA double-checked this forecast against your data. Everything lines up."
                    : "ARIA couldn't fully double-check this forecast yet. Treat it as a guide, not the final answer."}
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
          </h2>
          <p className="text-sm text-muted-foreground mt-0.5">See where your business is heading.</p>
        </div>
        <div className="flex flex-col items-end gap-1">
          <div className="flex bg-secondary border border-border rounded-xl p-0.5 text-xs font-medium" role="group" aria-label="Granularity">
            {(["Monthly", "Quarterly", "Yearly"] as PeriodTab[]).map((p) => {
              const locked = p !== "Monthly" && !canRebin;
              const approx = p !== "Monthly";
              return (
                <button
                  key={p}
                  onClick={() => { if (!locked) setTab(p); }}
                  aria-pressed={tab === p}
                  disabled={locked}
                  title={p === "Monthly" ? "Monthly is authoritative" : locked ? "Regrouping needs date labels — showing monthly" : "Approximate regrouping of monthly history"}
                  aria-label={p === "Monthly" ? "Monthly, authoritative" : `${p}, approximate regrouping`}
                  className={`px-3 py-1.5 rounded-lg transition-all duration-200 ${tab === p ? "bg-card text-foreground shadow-sm border border-border/60" : "text-muted-foreground hover:text-foreground border border-transparent"} disabled:opacity-40`}
                >
                  {p}{approx ? " ~approx" : ""}
                </button>
              );
            })}
          </div>
          <p className="text-[10px] text-muted-foreground">Monthly is authoritative; regrouped views are approximate.</p>
        </div>
      </div>

      {/* Eligibility / setup */}
      {eligLoading && (
        <div className="bg-card border border-border rounded-xl p-5 space-y-3 fc-skeleton">
          <AIProcessingIndicator compact stage={0} label="Understanding data" sublabel="Inspecting your data for forecastable numbers and time" />
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className="chart-skeleton h-10" />
            ))}
          </div>
          <div className="chart-skeleton h-40" />
        </div>
      )}
      {eligError && !eligLoading && (
        <div className="bg-negative/10 border border-negative/20 rounded-xl p-4">
          <p className="text-sm text-negative font-medium">{eligError}</p>
          <button onClick={loadEligibility} className="text-xs text-primary hover:underline mt-1">Retry inspection</button>
        </div>
      )}

      {elig && !elig.eligible && !eligLoading && (
        <div className="bg-card border border-border rounded-xl p-5 space-y-2">
          <p className="text-sm font-semibold text-foreground">ARIA cannot produce a meaningful time-based forecast for this dataset.</p>
          {(elig.reasons || []).map((r: string, i: number) => (
            <p key={i} className="text-sm text-muted-foreground leading-relaxed">· {r}</p>
          ))}
          {(!elig.reasons || elig.reasons.length === 0) && (
            <p className="text-sm text-muted-foreground leading-relaxed">· No usable time dimension or measurable number with enough history was found.</p>
          )}
          <p className="text-xs text-muted-foreground pt-1">ARIA needs a time dimension and a measurable number with enough history. Try connecting a dataset with dates and amounts.</p>
          <div className="pt-2">
            <button
              onClick={() => { window.location.hash = "#/copilot"; }}
              className="text-xs font-semibold px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:opacity-90 transition-opacity"
            >
              Ask ARIA to analyze this dataset instead
            </button>
          </div>
        </div>
      )}

      {elig?.eligible && (
        <div className="bg-card border border-border rounded-xl p-4 space-y-3">
          <p className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">Forecast setup</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 items-end">
            <div className="space-y-1.5">
              <p className="text-xs font-medium text-muted-foreground">Dataset</p>
              <select
                value={selectedTable}
                onChange={(e) => {
                  const v = e.target.value;
                  setSelectedTable(v);
                  const owner = tables.find((t) => t.table === v);
                  if (owner) {
                    setSelectedTarget((owner.measures || [])[0] || "");
                    setSelectedTime((owner.time_dims || [])[0]?.column || "");
                  }
                  setFilters([]); setWin(null); setForecast(null); setForecastStatus("idle"); setForecastVerdict(null); setError("");
                }}
                className="w-full bg-secondary border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:border-primary outline-none cursor-pointer"
              >
                {tables.map((t) => (
                  <option key={t.table} value={t.table}>{t.table} ({t.row_count ?? "?"} rows)</option>
                ))}
              </select>
            </div>
            <div className="space-y-1.5">
              <p className="text-xs font-medium text-muted-foreground">Metric</p>
              <select
                value={selectedTarget}
                onChange={(e) => {
                  setSelectedTarget(e.target.value);
                  setForecast(null); setForecastStatus("idle"); setForecastVerdict(null); setError("");
                }}
                className="w-full bg-secondary border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:border-primary outline-none cursor-pointer"
              >
                {activeMeasures.length === 0 && <option value="">No measures in this dataset</option>}
                {activeMeasures.map((m: string) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </div>
            <div className="space-y-1.5">
              <p className="text-xs font-medium text-muted-foreground">Date field</p>
              <select
                value={selectedTime}
                onChange={(e) => { setSelectedTime(e.target.value); setForecast(null); setForecastStatus("idle"); setForecastVerdict(null); setError(""); }}
                className="w-full bg-secondary border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:border-primary outline-none cursor-pointer"
              >
                {activeTimes.length === 0 && <option value="">Auto-detect…</option>}
                {activeTimes.map((d: any) => (
                  <option key={d.column} value={d.column}>{d.column}{d.kind === "text" ? " (text)" : ""} · {d.distinct ?? "?"} points</option>
                ))}
              </select>
              {selectedTime && (
                <p className="text-[10px] text-muted-foreground">
                  {(() => {
                    const td = activeTimes.find((d: any) => d.column === selectedTime);
                    const pts = td?.distinct ?? historyPoints ?? null;
                    const grain = td?.kind === "text" ? "text dates grouped as listed" : frequency === "MS" ? "monthly history" : frequency === "W" ? "weekly history" : frequency === "D" ? "daily history" : "history in listed order";
                    return `${grain}${pts !== null ? ` · ${pts} time points` : ""}. Needs at least 5 periods.`;
                  })()}
                </p>
              )}
              <p className="text-[10px] leading-relaxed">
                {(historyPoints ?? 0) >= 5
                  ? <span className="text-positive font-semibold">✓ Ready: {(historyPoints ?? 0)} periods (needs at least 5).</span>
                  : <span className="text-warning font-semibold">Needs at least 5 periods — currently {historyPoints ?? 0}. Pick a date field with more history.</span>}
              </p>
            </div>
            <div className="space-y-1.5">
              <p className="text-xs font-medium text-muted-foreground">Forecast horizon (periods)</p>
              <div className="flex items-center gap-2">
                <input
                  type="number" min={1} max={24} step={1}
                  value={horizon}
                  onChange={(e) => {
                    const n = Math.max(1, Math.min(24, Number(e.target.value) || 1));
                    setHorizon(n);
                  }}
                  aria-label="Forecast horizon in periods"
                  className="w-20 bg-secondary border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:border-primary outline-none"
                />
                <div className="flex bg-secondary border border-border rounded-xl p-0.5" role="group" aria-label="Quick horizon">
                  {[3, 6, 12].map((h) => (
                    <button
                      key={h}
                      onClick={() => setHorizon(h)}
                      aria-pressed={horizon === h}
                      className={`text-[11px] font-medium px-2.5 py-1.5 rounded-lg transition-all duration-200 ${horizon === h ? "bg-card text-primary shadow-sm border border-border/60" : "text-muted-foreground hover:text-foreground border border-transparent"}`}
                    >
                      {h}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 items-end">
            <div className="space-y-1.5">
              <p className="text-xs font-medium text-muted-foreground">Date range · from</p>
              <select
                value={rangeFrom}
                onChange={(e) => applyRange(e.target.value, rangeTo)}
                disabled={chartRows.length === 0}
                className="w-full bg-secondary border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:border-primary outline-none cursor-pointer disabled:opacity-50"
              >
                <option value="">Earliest</option>
                {chartRows.map((r: any) => (
                  <option key={`f-${r.label}`} value={r.label}>{fullLabel(r.label)}</option>
                ))}
              </select>
            </div>
            <div className="space-y-1.5">
              <p className="text-xs font-medium text-muted-foreground">Date range · to</p>
              <select
                value={rangeTo}
                onChange={(e) => applyRange(rangeFrom, e.target.value)}
                disabled={chartRows.length === 0}
                className="w-full bg-secondary border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:border-primary outline-none cursor-pointer disabled:opacity-50"
              >
                <option value="">Latest</option>
                {chartRows.map((r: any) => (
                  <option key={`t-${r.label}`} value={r.label}>{fullLabel(r.label)}</option>
                ))}
              </select>
            </div>
            <div className="space-y-1.5 sm:col-span-2">
              <p className="text-xs font-medium text-muted-foreground">Filters {schemaCols.length > 0 && <span className="font-normal">· travel inside the forecast question</span>}</p>
              {schemaCols.length > 0 ? (
                <div className="flex items-center gap-2">
                  <select
                    value={filterCol}
                    onChange={(e) => setFilterCol(e.target.value)}
                    className="bg-secondary border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:border-primary outline-none cursor-pointer max-w-[160px]"
                    aria-label="Filter column"
                  >
                    {schemaCols.map((c: any) => (
                      <option key={c.name} value={c.name}>{c.name}</option>
                    ))}
                  </select>
                  <input
                    value={filterVal}
                    onChange={(e) => setFilterVal(e.target.value)}
                    placeholder="value, e.g. Berlin"
                    aria-label="Filter value"
                    list="predictive-filter-values"
                    className="flex-1 min-w-0 bg-secondary border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:border-primary outline-none"
                  />
                  <datalist id="predictive-filter-values">
                    {filters.map((f) => (
                      <option key={`sug-${f.column}-${f.value}`} value={f.value}>{f.column}</option>
                    ))}
                  </datalist>
                  <button
                    onClick={() => {
                      if (!filterCol || !filterVal.trim()) return;
                      setFilters((f) => [...f.filter((x) => x.column !== filterCol), { column: filterCol, value: filterVal.trim() }].slice(0, 5));
                      setFilterVal("");
                    }}
                    disabled={!filterCol || !filterVal.trim()}
                    className="text-xs font-semibold px-3 py-2 rounded-lg border border-border text-foreground hover:border-primary/50 transition-colors disabled:opacity-50"
                  >
                    Add
                  </button>
                </div>
              ) : (
                <p className="text-[11px] text-muted-foreground py-2">No text dimensions found in this dataset — pick a dataset with categories to filter.</p>
              )}
            </div>
          </div>
          {filters.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {filters.map((f) => (
                <span key={f.column} className="inline-flex items-center gap-1.5 text-[11px] font-medium px-2.5 py-1 rounded-full bg-secondary text-foreground">
                  {f.column} = {f.value}
                  <button
                    onClick={() => setFilters((prev) => prev.filter((x) => x.column !== f.column))}
                    className="text-muted-foreground hover:text-foreground"
                    aria-label={`Remove filter ${f.column}`}
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          )}
          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={() => fetchForecast()}
              disabled={loading || !selectedTarget}
              className="bg-primary text-primary-foreground px-5 py-2 rounded-lg text-sm font-semibold hover:opacity-90 transition-opacity disabled:opacity-50"
            >
              {loading ? "Working…" : hasPredictions ? "Update" : "Build forecast"}
            </button>
            {timeName && <p className="text-[11px] text-muted-foreground">Time base: {timeName} · detected from your data, changeable above.</p>}
          </div>
          {horizonWarning && <p className="text-[11px] text-warning mt-1">{horizonWarning}</p>}
        </div>
      )}

      {loading && !hasPredictions && (
        <div className="bg-card border border-border rounded-xl p-6 space-y-3 fc-skeleton" aria-live="polite">
          <AIProcessingIndicator compact stage={loadStage} />
          <div className="chart-skeleton h-48" />
          <div className="grid grid-cols-3 gap-3">
            {[0, 1, 2].map((i) => (
              <div key={i} className="chart-skeleton h-16" />
            ))}
          </div>
        </div>
      )}

      {error && !hasPredictions && !loading && (
        isUnforecastable ? (
          <div className="bg-card border border-border rounded-xl p-5 space-y-2">
            <p className="text-sm font-semibold text-foreground">ARIA cannot produce a meaningful time-based forecast for this selection.</p>
            <p className="text-sm text-muted-foreground leading-relaxed">· {error}</p>
            <div className="flex flex-wrap gap-3 pt-1">
              <button onClick={() => fetchForecast()} className="text-xs font-semibold px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:opacity-90">Retry</button>
              <button onClick={useAutomatic} className="text-xs font-semibold text-primary hover:underline self-center">Try automatic selection</button>
              <button onClick={() => { window.location.hash = "#/copilot"; }} className="text-xs font-medium text-muted-foreground hover:text-foreground self-center">Ask ARIA to analyze this dataset instead</button>
            </div>
          </div>
        ) : (
          <div className="bg-card border border-warning/40 rounded-xl p-4 space-y-2">
            <p className="text-sm font-semibold text-foreground">Couldn't forecast that selection</p>
            <p className="text-sm text-muted-foreground leading-relaxed">{error}</p>
            <div className="flex gap-3 pt-1">
              <button onClick={() => fetchForecast()} className="text-xs font-semibold px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:opacity-90">Retry</button>
              <button onClick={useAutomatic} className="text-xs font-semibold text-primary hover:underline self-center">Try automatic selection</button>
            </div>
          </div>
        )
      )}

      {!hasPredictions && !loading && !error && elig?.eligible && (
        <div className="text-center py-10 bg-card border border-border rounded-xl">
          <TrendingUp size={28} className="text-muted-foreground mx-auto mb-2" />
          <p className="text-sm text-muted-foreground">Your forecast will appear here.</p>
        </div>
      )}

      {hasPredictions && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2 space-y-4">
            <div className={`bg-card border border-border rounded-xl p-5 forecast-reveal ${loading ? "opacity-70" : ""}`} key={`fc-${revealKey}`}>
              <ChartFrame
                title={`${plainName(selectedTarget) || "Metric"} forecast`}
                explanation={`Solid line: what actually happened${frequency ? ` (${frequency})` : ""}. Line ahead: what ARIA expects next. Shaded band: likely range. Hover any point for detail, click a point to inspect it, drag the range bar to zoom. Click legend items to toggle series.`}
                empty={chartRows.length === 0}
                emptyText="Chart couldn't be drawn from this result — the numbers above still apply."
                height={300}
              >
              <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
                <ToggleLegend
                  items={["Actual", "Forecast", ...(avgActual !== null ? ["Baseline"] : []), "Likely range"]}
                  isHidden={fcToggle.isHidden}
                  onToggle={fcToggle.toggle}
                  colors={{ Actual: ACTUAL_C, Forecast: FORECAST_C, Baseline: BASELINE_C, "Likely range": FORECAST_C }}
                />
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-[11px] font-bold px-2.5 py-1 rounded-full bg-secondary text-foreground">
                    {fmtNum(projectedTotal)} total
                  </span>
                  <div className="flex items-center gap-1" role="group" aria-label="Zoom and pan">
                    <button onClick={() => pan(-1)} disabled={chartRows.length < 3 || !win} title="Pan back" aria-label="Pan back" className="text-[11px] font-bold px-2 py-1 rounded-lg bg-secondary text-foreground border border-border disabled:opacity-40">‹</button>
                    <button onClick={zoomOut} disabled={chartRows.length < 3 || !win} title="Zoom out" aria-label="Zoom out" className="text-[11px] font-bold px-2 py-1 rounded-lg bg-secondary text-foreground border border-border disabled:opacity-40">−</button>
                    <button onClick={zoomIn} disabled={chartRows.length < 3} title="Zoom in" aria-label="Zoom in" className="text-[11px] font-bold px-2 py-1 rounded-lg bg-secondary text-foreground border border-border disabled:opacity-40">+</button>
                    <button onClick={() => pan(1)} disabled={chartRows.length < 3 || !win} title="Pan forward" aria-label="Pan forward" className="text-[11px] font-bold px-2 py-1 rounded-lg bg-secondary text-foreground border border-border disabled:opacity-40">›</button>
                    <button onClick={resetView} disabled={!win} title="Reset view" aria-label="Reset zoom" className="text-[11px] font-medium px-2 py-1 rounded-lg bg-secondary text-foreground border border-border disabled:opacity-40">Reset</button>
                  </div>
                  <button
                    onClick={exportCsv}
                    disabled={chartRows.length === 0}
                    title="Download the full series as CSV with source details"
                    aria-label="Export full forecast series as CSV"
                    className="text-[11px] font-medium px-2.5 py-1 rounded-lg bg-secondary hover:bg-secondary/70 text-foreground transition-colors border border-border disabled:opacity-50"
                  >
                    Export CSV
                  </button>
                  <button
                    onClick={() => fetchForecast()}
                    disabled={loading}
                    className="flex items-center gap-1.5 text-[11px] font-medium px-2.5 py-1 rounded-lg bg-secondary hover:bg-secondary/70 text-foreground transition-colors border border-border disabled:opacity-50"
                  >
                    <RefreshCw size={11} className={loading ? "animate-spin" : ""} /> Refresh
                  </button>
                </div>
              </div>
              {historyLabel && (
                <p className="text-[11px] text-muted-foreground mb-1">{historyLabel}{tab !== "Monthly" ? " · regrouped view is approximate, Monthly is authoritative" : ""}</p>
              )}
              {win && viewRows.length > 0 && (
                <p className="text-[11px] text-muted-foreground mb-1">
                  Showing {fullLabel(viewRows[0]?.label)} – {fullLabel(viewRows[viewRows.length - 1]?.label)} ({Math.round(zoomDepth * 100)}% of {chartRows.length} points). <button onClick={resetView} className="text-primary hover:underline font-medium">Show all</button>
                </p>
              )}
              {chartRows.length > 0 ? (
                <ResponsiveContainer width="100%" height={248}>
                  <ComposedChart
                    key={revealKey}
                    data={viewRows}
                    margin={{ top: 8, right: 8, bottom: 0, left: 0 }}
                    onClick={(st: any) => { if (st?.activePayload?.[0]?.payload) openPoint(st.activePayload[0].payload); }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke={grid} vertical={false} />
                    <XAxis dataKey="short" tick={{ fontSize: 11, fill: axis }} axisLine={false} tickLine={false} minTickGap={20} />
                    <YAxis tick={{ fontSize: 11, fill: axis }} axisLine={false} tickLine={false} width={56} tickFormatter={(v: number) => fmtNum(v)} />
                    <Tooltip
                      content={<ForecastTooltip rowsByLabel={rowsByLabel} modelName={modelName} />}
                      cursor={{ stroke: FORECAST_C, strokeDasharray: "4 4", strokeWidth: 1 }}
                    />
                    {/* Interval = shaded band between backend lower/upper.
                        History = solid line (never animated, stays stable);
                        forecast + CI reveal once per forecast via recharts'
                        one-shot enter animation (no continuous motion after).
                        x positions are backend timestamps only (see
                        chartRows). Rebin sums the backend bounds per bucket,
                        so the band survives Quarterly/Yearly grouping. */}
                    {!fcToggle.isHidden("Likely range") && (
                      <Area type="monotone" dataKey="upper95" stroke="none" fill={FORECAST_C} fillOpacity={0.14} connectNulls name="Likely range" isAnimationActive={true} animationBegin={350} animationDuration={800} />
                    )}
                    {!fcToggle.isHidden("Likely range") && (
                      <Area type="monotone" dataKey="lower95" stroke="none" fill="var(--card)" fillOpacity={1} connectNulls name="Likely range" isAnimationActive={true} animationBegin={350} animationDuration={800} />
                    )}
                    {avgActual !== null && !fcToggle.isHidden("Baseline") && (
                      <ReferenceLine y={avgActual} stroke={BASELINE_C} strokeWidth={1.5} strokeDasharray="4 4" label={{ value: "Baseline (avg)", fontSize: 10, fill: axis, position: "insideTopRight" }} />
                    )}
                    {!fcToggle.isHidden("Actual") && (
                      <Line type="monotone" dataKey="actual" stroke={ACTUAL_C} strokeWidth={2} dot={{ r: 2 }} connectNulls name="Actual" isAnimationActive={false} />
                    )}
                    {!fcToggle.isHidden("Forecast") && (
                      <Line type="monotone" dataKey="forecast" stroke={FORECAST_C} strokeWidth={2} strokeDasharray="6 4" dot={{ r: 3 }} connectNulls name="Forecast" isAnimationActive={true} animationBegin={0} animationDuration={900} />
                    )}
                    {!fcToggle.isHidden("Forecast") && chartRows.some((r: any) => r.act !== null && r.act !== undefined) && (
                      <Line type="monotone" dataKey="act" stroke="#8B5CF6" strokeWidth={2} strokeDasharray="2 3" dot={false} connectNulls name="If we act (estimate)" isAnimationActive={true} animationBegin={200} animationDuration={900} />
                    )}
                    {chartRows.length > 5 && (
                      <Brush
                        dataKey="short" height={24} stroke="var(--primary)" travellerWidth={10}
                        startIndex={win ? Math.max(0, win.s) : undefined}
                        endIndex={win ? Math.min(chartRows.length - 1, win.e) : undefined}
                        onChange={(r: any) => {
                          if (r && typeof r.startIndex === "number" && typeof r.endIndex === "number") {
                            if (r.startIndex === 0 && r.endIndex === chartRows.length - 1) setWin(null);
                            else setWin({ s: r.startIndex, e: r.endIndex });
                          }
                        }}
                      />
                    )}
                  </ComposedChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-sm text-muted-foreground py-8 text-center">Chart couldn't be drawn from this result — the numbers above still apply.</p>
              )}
              </ChartFrame>
              {/* Uncertainty + regrouping honesty: backend values are never
                  widened — a zero-width band shows an uncertain marker.
                  Regrouped Quarterly/Yearly bounds are marked approximate. */}
              {(uncertaintyNote || hasZeroWidth) && (
                <p className="mt-2 text-[11px] text-warning leading-relaxed">
                  {uncertaintyNote || "Very low observed spread — treat as uncertain, not guaranteed."}
                  {hasZeroWidth ? " (marked uncertain: range has no width)" : ""}
                </p>
              )}
              {tab !== "Monthly" && (
                <p className="mt-1 text-[11px] text-muted-foreground leading-relaxed">Approximate range from regrouping — Monthly is authoritative. See Technical details.</p>
              )}
            </div>

            {/* Why: small button — technical details only on click. */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowWhy((v) => !v)}
                aria-expanded={showWhy}
                className="inline-flex items-center gap-1.5 text-[11.5px] font-semibold text-blue-700 dark:text-blue-400 border border-blue-200 dark:border-blue-500/25 rounded-full px-3 py-1.5 hover:bg-blue-50 dark:hover:bg-blue-500/10 transition-colors"
                aria-label="Show why ARIA expects this"
              >
                Why?
              </button>
              {whyItems.length > 0 && (
                <span className="text-[11px] text-muted-foreground">{whyItems.length} reason{whyItems.length === 1 ? "" : "s"} from your data</span>
              )}
            </div>
            {showWhy && (
              <div className="bg-card border border-border rounded-xl p-4 space-y-2.5">
                {whyItems.length > 0 ? (
                  <ul className="space-y-2">
                    {whyItems.map((w, i) => (
                      <li key={i} className="flex items-start gap-2.5">
                        <span className="w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center flex-shrink-0 mt-0.5">
                          <Check size={12} />
                        </span>
                        <div className="min-w-0">
                          <p className="text-[13px] font-semibold text-foreground">{w.title}</p>
                          <p className="text-xs text-muted-foreground leading-relaxed">{w.body}</p>
                          <p className="text-[10px] text-muted-foreground/70">source: backend · {w.source}</p>
                        </div>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-xs text-muted-foreground leading-relaxed">The backend did not return an explanation for this forecast.</p>
                )}
                <p className="text-[10px] text-muted-foreground/70 leading-relaxed pt-1">
                  Driver notes above come straight from the backend explanation; anything missing was not returned, never filled in.
                </p>
              </div>
            )}

            {scenarios && (
              <>
                {tab !== "Monthly" && (
                  <p className="text-[11px] text-muted-foreground">Regrouped {tab.toLowerCase()} totals are approximate — Monthly is authoritative.</p>
                )}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {scenarios.map((s) => (
                  <div key={s.label} className="bg-card border border-border rounded-xl p-4 space-y-2">
                    <div className="flex items-center gap-2">
                      <span
                        className="w-7 h-7 rounded-full flex items-center justify-center"
                        style={{
                          background: s.tone === "good" ? "rgba(5,150,105,0.12)" : s.tone === "bad" ? "rgba(239,68,68,0.12)" : "rgba(47,111,173,0.12)",
                          color: s.tone === "good" ? "#059669" : s.tone === "bad" ? "#EF4444" : "#2F6FAD",
                        }}
                      >
                        {s.tone === "good" ? <TrendingUp size={14} /> : s.tone === "bad" ? <TrendingDown size={14} /> : <Crosshair size={14} />}
                      </span>
                      <p className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">{s.label}</p>
                      <span className={`ml-auto text-[11px] font-bold ${s.pct >= 0 ? "text-positive" : "text-negative"}`}>
                        {s.pct >= 0 ? "+" : ""}{s.pct.toFixed(0)}%
                      </span>
                    </div>
                    <p className="text-xl font-bold text-foreground leading-none">{fmtNum(s.total)}</p>
                    <p className="text-[11px] text-muted-foreground">{s.ofTarget.toFixed(0)}% of average pace</p>
                    <div className="h-1.5 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${scenMax ? Math.max(4, (s.total / scenMax) * 100) : 4}%`,
                          background: s.tone === "good" ? "#059669" : s.tone === "bad" ? "#EF4444" : "#2F6FAD",
                        }}
                      />
                    </div>
                    <div className="space-y-0.5 pt-0.5">
                      {[s.note, s.tone === "base" ? "Based on recent trend" : s.tone === "good" ? "If momentum holds" : "If momentum slows"].map((b) => (
                        <p key={b} className="text-[11px] text-muted-foreground flex items-center gap-1.5">
                          <Check size={11} className="text-positive flex-shrink-0" /> {b}
                        </p>
                      ))}
                    </div>
                    {tab !== "Monthly" && (
                      <p className="text-[10px] text-muted-foreground">Approximate — regrouped view.</p>
                    )}
                  </div>
                ))}
              </div>
              </>
            )}

            {/* What-if estimate: wired to the real scenario endpoint. */}
            <div className="bg-card border border-border rounded-xl p-4 space-y-2">
              <p className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">What-if estimate</p>
              <p className="text-xs text-muted-foreground leading-relaxed">See what a change to {targetName || "this number"} could look like. Arithmetic estimate from your history, not a forecast.</p>
              <div className="flex items-center gap-2 flex-wrap">
                <label className="text-xs text-muted-foreground" htmlFor="predictive-scenario-pct">Change (%)</label>
                <input
                  id="predictive-scenario-pct"
                  type="number" min={-95} max={95} step={1}
                  value={scenarioPct}
                  onChange={(e) => setScenarioPct(Number(e.target.value) || 0)}
                  aria-label="What-if change percent"
                  className="w-20 bg-secondary border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:border-primary outline-none"
                />
                <button
                  onClick={runScenario}
                  disabled={scenarioLoading || !selectedTarget}
                  aria-label="Run what-if estimate"
                  className="text-xs font-semibold px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:opacity-90 disabled:opacity-50"
                >
                  {scenarioLoading ? "Working…" : "Run estimate"}
                </button>
              </div>
              {scenarioError && <p className="text-xs text-negative">{scenarioError}</p>}
              {scenario && scenario.ok && (
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between"><span className="text-muted-foreground">Starting level</span><span className="font-bold text-foreground">{fmtNum(scenario.baseline)}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">With change</span><span className="font-bold text-foreground">{fmtNum(scenario.scenario_value)} ({Number(scenario.pct_change) >= 0 ? "+" : ""}{scenario.pct_change}%)</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Difference</span><span className="font-bold text-foreground">{fmtNum(scenario.delta)}</span></div>
                  <p className="text-[11px] text-muted-foreground">Based on the last {scenario.periods_used ?? 3} periods of {scenario.column || targetName}.</p>
                  {Array.isArray(scenario.assumptions) && scenario.assumptions.length > 0 && (
                    <ul className="space-y-0.5 pt-1">
                      {scenario.assumptions.map((a: string, i: number) => (
                        <li key={i} className="text-[11px] text-muted-foreground">· {a}</li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
            </div>

            {forecastStory.story && (
              <div className="bg-card border border-border rounded-xl p-4 space-y-1.5">
                <p className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">What this means</p>
                <p className="text-sm text-foreground leading-relaxed">{forecastStory.story}</p>
                {forecastStory.signal && (
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    <span className="font-semibold text-foreground">Key signal:</span> {forecastStory.signal}
                  </p>
                )}
                {(() => {
                  const fwd = (fwdVals || []).map((v: any) => Number(v)).filter((v: number) => Number.isFinite(v));
                  if (fwd.length >= 2 && fwd.every((v: number) => v === fwd[0])) {
                    return (
                      <p className="text-[11px] text-amber-700 dark:text-amber-400 bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/25 rounded-lg px-2.5 py-2 leading-relaxed">
                        Flat projection — recent history shows no stable trend, so ARIA holds the last level
                        {modelName && modelName !== "none" ? ` (${modelName})` : ""} rather than inventing movement.
                      </p>
                    );
                  }
                  return null;
                })()}
              </div>
            )}
                {(() => {
                  const acc = accuracy;
                  const ok = (predictions as any)?.valid !== false;
                  const verdict = !ok || (acc !== null && acc < 70)
                    ? { t: "Direction only — don't plan on precise numbers.", c: "text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-500/25 bg-amber-50 dark:bg-amber-500/10" }
                    : acc !== null && acc >= 80
                      ? { t: "Plan on this — backtesting supports the outlook.", c: "text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-500/25 bg-emerald-50 dark:bg-emerald-500/10" }
                      : { t: "Working estimate — plan with margin.", c: "text-blue-700 dark:text-blue-400 border-blue-200 dark:border-blue-500/25 bg-blue-50 dark:bg-blue-500/10" };
                  return (
                    <p className={`text-xs font-semibold rounded-lg border px-2.5 py-2 ${verdict.c}`}>
                      {verdict.t}{acc !== null ? ` Backtested accuracy ${acc.toFixed(0)}%.` : ""}
                    </p>
                  );
                })()}

            {hasPredictions && rxItems.length > 0 && (
              <div className="bg-card border border-border rounded-xl p-4 space-y-2">
                <p className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">So consider — from your data</p>
                {rxItems.map((p: any, i: number) => (
                  <div key={String(p?.id || i)} className="rounded-lg bg-muted/50 px-3 py-2">
                    <p className="text-sm font-semibold text-foreground leading-snug">{String(p?.display?.action_text || p?.title || "Recommended action")}</p>
                    {p?.display?.outcome_text && <p className="text-sm font-bold text-primary tabular-nums">{String(p.display.outcome_text)}</p>}
                    {p?.display?.why_summary && <p className="text-xs text-muted-foreground leading-relaxed mt-0.5">{String(p.display.why_summary)}</p>}
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="space-y-4">
            {/* Confidence indicator — click to expand backend evidence. */}
            <div className="bg-card border border-border rounded-xl p-4 space-y-2">
              <button
                onClick={() => setConfidenceOpen((o) => !o)}
                className="w-full flex items-center gap-2 text-left"
                aria-expanded={confidenceOpen}
              >
                <Sparkles size={13} className="text-primary flex-shrink-0" />
                <p className="text-[11px] font-semibold text-primary uppercase tracking-wider">Forecast confidence</p>
                <span
                  className="ml-auto text-[11px] font-bold px-2.5 py-1 rounded-full"
                  style={confidenceLevel === "High"
                    ? (isDark ? { background: "rgba(52,211,153,0.14)", color: "#34D399" } : { background: "rgba(5,150,105,0.12)", color: "#059669" })
                    : confidenceLevel === "Moderate"
                      ? (isDark ? { background: "rgba(123,167,217,0.14)", color: "#7BA7D9" } : { background: "rgba(47,111,173,0.12)", color: "#2F6FAD" })
                      : (isDark ? { background: "rgba(251,191,36,0.14)", color: "#FBBF24" } : { background: "rgba(201,138,27,0.14)", color: "#C98A1B" })}
                >
                  {confidenceLevel === "Limited" ? "Limited historical evidence" : confidenceLevel}
                </span>
                <span className="text-muted-foreground text-xs">{confidenceOpen ? "▾" : "▸"}</span>
              </button>
              <p className="text-sm text-foreground leading-relaxed">
                {targetName ? <><strong>{targetName}</strong> is </> : "Your number is "}
                {pctChange !== null && pctChange > 1 && <>heading up about <strong>+{pctChange.toFixed(1)}%</strong></>}
                {pctChange !== null && pctChange < -1 && <>heading down about <strong>{pctChange.toFixed(1)}%</strong></>}
                {(pctChange === null || Math.abs(pctChange) <= 1) && <>holding roughly steady</>}
                {historyCount > 0 && <> across {historyCount} past {historyCount === 1 ? "point" : "points"}</>}.
              </p>
              {confidenceOpen && (
                <div className="pt-1 space-y-1.5 text-xs border-t border-border mt-2">
                  <div className="flex justify-between pt-2"><span className="text-muted-foreground">Historical observations</span><span className="font-bold text-foreground">{historyCount}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Backend reliability</span><span className="font-bold text-foreground">{backendReliability ?? "—"}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Backend valid</span><span className="font-bold text-foreground">{backendValid === null ? "—" : String(backendValid)}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Stage</span><span className="font-bold text-foreground">{forecastStage ?? "—"}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Backtest accuracy</span><span className="font-bold text-foreground">{accuracy !== null ? `${accuracy.toFixed(1)}%` : "—"}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">MAE</span><span className="font-bold text-foreground">{contractMetrics?.mae ?? "—"}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">RMSE</span><span className="font-bold text-foreground">{contractMetrics?.rmse ?? "—"}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">MAPE</span><span className="font-bold text-foreground">{contractMetrics?.mape !== null && contractMetrics?.mape !== undefined ? `${contractMetrics.mape}%` : "—"}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Walk-forward coverage</span><span className="font-bold text-foreground">{coverage && coverage.evaluated !== null ? `${coverage.evaluated}/${coverage.origins ?? "?"} evaluated${coverage.zeroSkipped ? ` (${coverage.zeroSkipped} zero-demand skipped)` : ""}` : "—"}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Model</span><span className="font-bold text-foreground">{modelName || "—"}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Horizon</span><span className="font-bold text-foreground">{fwdVals.length} {fwdVals.length === 1 ? "period" : "periods"}</span></div>
                </div>
              )}
            </div>

            {isPreliminary && (
              <div className="bg-secondary/60 border border-warning/40 rounded-xl p-4 space-y-1.5">
                <p className="text-[11px] font-semibold text-warning uppercase tracking-wider">Preliminary forecast</p>
                <p className="text-xs text-foreground leading-relaxed">ARIA generated a preliminary forecast using the available historical data.</p>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  Limited historical data — only {historyCount} past {historyCount === 1 ? "point" : "points"} underpin this projection, so uncertainty is higher than usual{uncertaintyNote ? ` (${uncertaintyNote.toLowerCase()})` : ""}.
                  Connect more historical data to sharpen this forecast.
                </p>
              </div>
            )}

            <div className="bg-card border border-border rounded-xl p-4 space-y-3">
              <p className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">Key factors</p>
              {[
                { icon: Target, label: "Period average", value: avgActual !== null ? fmtNum(avgActual) : "—", pill: momDelta !== null ? `${momDelta > 0 ? "+" : ""}${momDelta.toFixed(1)}%` : null, good: (momDelta ?? 0) >= 0 },
                { icon: TrendingUp, label: "Last actual", value: lastActual !== null ? fmtNum(lastActual) : "—", pill: null as string | null, good: true },
                { icon: Crosshair, label: "Past accuracy", value: accuracy !== null ? `${accuracy.toFixed(0)}%` : "—", pill: accuracy !== null ? (accuracy >= 70 ? "Reliable" : "Weak") : null, good: (accuracy ?? 0) >= 70 },
                { icon: Sparkles, label: "History", value: historyCount > 0 ? `${historyCount} periods` : "—", pill: null as string | null, good: true },
              ].map((f) => (
                <div key={f.label} className="flex items-center gap-2.5">
                  <span className="w-7 h-7 rounded-full bg-primary/10 text-primary flex items-center justify-center flex-shrink-0">
                    <f.icon size={13} />
                  </span>
                  <span className="text-[13px] text-muted-foreground">{f.label}</span>
                  <span className="ml-auto text-[13px] font-bold text-foreground">{f.value}</span>
                  {f.pill && (
                    <span
                      className="text-[10px] font-bold px-1.5 py-0.5 rounded"
                      style={f.good ? { background: "rgba(5,150,105,0.12)", color: "var(--positive)" } : { background: "rgba(239,68,68,0.12)", color: "var(--negative)" }}
                    >
                      {f.pill}
                    </span>
                  )}
                </div>
              ))}
            </div>

            {recentTop.length > 0 && (
              <div className="bg-card border border-border rounded-xl p-4 space-y-2.5">
                <p className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">Largest periods</p>
                {recentTop.map((r) => (
                  <div key={r.label} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-foreground font-medium">{r.label}</span>
                      <span className="text-muted-foreground">{fmtNum(r.value)} · {r.share.toFixed(0)}%</span>
                    </div>
                    <div className="h-1.5 bg-secondary rounded-full overflow-hidden">
                      <div className="h-full rounded-full bg-primary" style={{ width: `${Math.min(100, r.share)}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            )}

            <div key={`tip-${revealKey}`} className="rounded-xl p-3.5 flex items-start gap-2.5 bg-secondary/60 border border-border ci-fade" style={{ ["--ci-opacity" as string]: 1 } as any}>
              <Lightbulb size={14} className="text-primary flex-shrink-0 mt-0.5" />
              <p className="text-xs text-foreground leading-relaxed">
                <strong>Tip:</strong>{" "}
                {uncertaintyNote
                  ? uncertaintyNote
                  : accuracy !== null && accuracy < 70
                  ? "Accuracy is below 70% — treat this as directional and watch incoming actuals."
                  : pctChange !== null && pctChange > 10
                    ? "Strong upward momentum — make sure capacity can cover the projected peak."
                    : pctChange !== null && pctChange < -10
                      ? "A downward trend is ahead — consider actions to recover volume."
                      : "The projection tracks your recent average — small moves compound over time."}
              </p>
            </div>

            {/* Evidence: plain-language method up front; algorithm + error
                metrics live only under Technical details — never in the main view. */}
            <Evidence
              sections={[
                { title: "Data analyzed", body: `${targetName || "The selected number"} over the last ${historyCount > 0 ? `${historyCount} periods` : "available history"}${frequency ? ` (${frequency}).` : "."}` },
                { title: "Key observation", body: pctChange !== null && Math.abs(pctChange) > 1 ? `Recent momentum points ${pctChange > 0 ? "up" : "down"} about ${Math.abs(Math.round(pctChange))}%.` : "Recent values sit close to their average — no strong direction." },
                { title: "Analysis method", body: "ARIA compared several projection approaches and kept the one that held up best against past actuals before showing any projection." },
                { title: "Confidence", body: accuracy !== null ? `${accuracy >= 70 ? "Reliable" : "Weak"} in backtesting (${Math.round(accuracy)}% accuracy). Uncertainty grows further out — treat distant points as directional.` : "Backtesting could not score this series — treat the projection as directional." },
                { title: "Assumptions", body: "Past patterns continue; no shocks, price changes, or seasonality shifts are modeled. A forecast is an estimate, not a promise." },
              ]}
              technical={`How ARIA calculated this: ${modelName ? `selected approach ${modelName}` : "best of several candidate approaches"}${selectionStatement ? ` — ${selectionStatement}` : ""}${contractMetrics && (contractMetrics.mae !== null || contractMetrics.mape !== null) ? ` Holdout error: ${contractMetrics.mape !== null ? `MAPE ${contractMetrics.mape}%` : ""}${contractMetrics.mape !== null && contractMetrics.rmse !== null ? ", " : ""}${contractMetrics.rmse !== null ? `RMSE ${contractMetrics.rmse}` : ""}${contractMetrics.mae !== null ? `, MAE ${contractMetrics.mae}` : ""}.` : ""}${intervalMethod === "summed_bounds_approx" ? " Approximate range from regrouping." : ""}${uncertaintyNote ? ` ${uncertaintyNote}.` : ""} For analyst review.`}
            />
          </div>
        </div>
      )}


      {/* Click-point detail popup: backend row + prior-period change + boundary labels. */}
      {pointSel && (
        <div className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-4" role="dialog" aria-modal="true" aria-label="Forecast point detail">
          <div className="bg-card border border-border rounded-xl p-5 w-full max-w-sm space-y-2">
            <div className="flex items-center justify-between">
              <p className="text-sm font-bold text-foreground">{fullLabel(pointSel.label)}</p>
              <button onClick={() => setPointSel(null)} className="text-xs px-2.5 py-1.5 rounded-lg border border-border text-foreground hover:border-primary/50" aria-label="Close point detail">Close ×</button>
            </div>
            {typeof pointSel.idx === "number" && typeof pointSel.total === "number" && pointSel.total > 0 && (
              <p className="text-[11px] text-muted-foreground">
                Point {(pointSel.idx ?? 0) + 1} of {pointSel.total}
                {pointSel.isFirst ? " · first in view" : ""}{pointSel.isLast ? " · last in view" : ""}
                {pointSel.kind ? ` · ${pointSel.kind}` : ""}
              </p>
            )}
            {pointSel.actual !== null && pointSel.actual !== undefined && (
              <p className="text-xs text-muted-foreground">Actual: <span className="text-foreground font-semibold">{fmtNum(pointSel.actual)}</span></p>
            )}
            {pointSel.forecast !== null && pointSel.forecast !== undefined && (
              <p className="text-xs text-muted-foreground">Predicted: <span className="text-foreground font-semibold">{fmtNum(pointSel.forecast)}</span></p>
            )}
            {pointSel.lower95 !== null && pointSel.lower95 !== undefined && (
              <p className="text-xs text-muted-foreground">Likely range: <span className="text-foreground font-semibold">{fmtNum(pointSel.lower95)} – {fmtNum(pointSel.upper95)}</span></p>
            )}
            {pointSel.pct !== null && pointSel.pct !== undefined && (
              <p className="text-xs text-muted-foreground">
                Change vs {pointSel.prevLabel ? fullLabel(pointSel.prevLabel) : "prior point"}:{" "}
                <span className={`font-semibold ${pointSel.pct >= 0 ? "text-positive" : "text-negative"}`}>
                  {pointSel.pct >= 0 ? "+" : ""}{pointSel.pct.toFixed(1)}%
                </span>
              </p>
            )}
            <p className="text-[11px] text-muted-foreground pt-1">
              {pointSel.forecast !== null && pointSel.forecast !== undefined
                ? `Forecast${modelName ? ` · ${modelName}` : ""} · ${targetName}`
                : `Historical actual · ${targetName}`}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
