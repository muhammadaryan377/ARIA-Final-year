/* Explore builder state + analysis lifecycle for the Intelligence page.
   All values flow from GET /intelligence/meta (options) and
   POST /intelligence/explore (results) — the hook formats and sends, the
   backend computes. No mocks, no invented numbers. */

import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { intelligenceApi, normalizeApiError } from "../../../services/api";
import { asList, strOf, humanize, businessError } from "../../lib/intelHelpers";

export type RangeKey = "3m" | "6m" | "12m" | "all";

export const RANGE_OPTIONS: { key: RangeKey; label: string; months: number | null }[] = [
  { key: "3m", label: "Last 3 months", months: 3 },
  { key: "6m", label: "Last 6 months", months: 6 },
  { key: "12m", label: "Last 12 months", months: 12 },
  { key: "all", label: "All data", months: null },
];

export interface ExploreFilter {
  column: string;
  value: string;
}

type DateRangePayload = string | { start: string; end: string; label: string };

/* ---------- explicit window derivation (real filtering) ----------
   The backend honors date_range as {start,end,label} by filtering to that
   window vs the prior equal-length window; a bare string is display-only.
   Derive the window from the live result's trend labels (latest point back
   N calendar months). When labels are not dates (e.g. snapshot categories),
   return null so the caller falls back to the display-only string. */

function parsePointDate(label: string): Date | null {
  const s = String(label || "").trim();
  if (!s) return null;
  let m = /^(\d{4})-(\d{1,2})(?:-(\d{1,2}))?/.exec(s);
  if (m) return new Date(Number(m[1]), Number(m[2]) - 1, 1);
  m = /^(\d{1,2})\/(\d{4})/.exec(s);
  if (m) return new Date(Number(m[2]), Number(m[1]) - 1, 1);
  m = /^[Qq]([1-4])\s+(\d{4})/.exec(s);
  if (m) return new Date(Number(m[2]), (Number(m[1]) - 1) * 3, 1);
  try {
    const d = new Date(s);
    if (!isNaN(d.getTime())) return new Date(d.getFullYear(), d.getMonth(), 1);
  } catch {
    /* unparseable — fall through */
  }
  return null;
}

const pad2 = (n: number): string => String(n).padStart(2, "0");
const isoDay = (d: Date): string =>
  `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;

export function explicitWindowFor(
  result: any,
  months: number,
): { start: string; end: string; label: string } | null {
  const raw = asList(
    result?.trend?.points || result?.trend || result?.series || result?.points || result?.chart?.data,
  );
  const dates = raw
    .map((p: any, i: number) =>
      parsePointDate(strOf(p?.label ?? p?.period ?? p?.x ?? p?.month ?? "")),
    )
    .filter((d): d is Date => d instanceof Date && !isNaN(d.getTime()))
    .sort((a, b) => a.getTime() - b.getTime());
  if (dates.length === 0 || !(months > 0)) return null;
  const latest = dates[dates.length - 1];
  const end = new Date(latest.getFullYear(), latest.getMonth() + 1, 0);
  const start = new Date(latest.getFullYear(), latest.getMonth() - (months - 1), 1);
  return { start: isoDay(start), end: isoDay(end), label: `Last ${months} months` };
}

export function dateRangeFor(rangeKey: RangeKey, result: any): DateRangePayload {
  if (rangeKey === "all") return "All data";
  const months = rangeKey === "3m" ? 3 : rangeKey === "6m" ? 6 : 12;
  return explicitWindowFor(result, months) || `Last ${months} months`;
}

/* Map a saved/display-only range string back onto the selector. */
export function rangeKeyForLabel(v: unknown): RangeKey | null {
  const s = String(v || "").toLowerCase();
  if (/all\s*data/.test(s)) return "all";
  if (/(^|\D)3\s*(m|month)/.test(s)) return "3m";
  if (/(^|\D)6\s*(m|month)/.test(s)) return "6m";
  if (/(^|\D)12\s*(m|month)/.test(s) || /last\s*year/.test(s)) return "12m";
  return null;
}

const optOf = (x: any): { value: string; label: string } => {
  if (typeof x === "string") return { value: x, label: humanize(x) };
  const value = strOf(x?.column || x?.name || x?.label);
  const label = strOf(x?.label || x?.name ? humanize(strOf(x?.label || x?.name)) : humanize(value));
  return { value, label };
};

export function useExplore(meta: any) {
  const metrics: { value: string; label: string }[] = useMemo(() => {
    const m = meta?.metrics ?? meta?.measures ?? [];
    return asList(m).map(optOf).filter((o) => o.value);
  }, [meta]);
  const breakdowns: { value: string; label: string }[] = useMemo(() => {
    const b = meta?.breakdowns ?? meta?.dimensions ?? [];
    return asList(b).map(optOf).filter((o) => o.value);
  }, [meta]);
  const grains: string[] = useMemo(
    () =>
      asList(meta?.time_grains ?? meta?.timeGrains ?? ["Monthly"])
        .map((x) => strOf(x))
        .filter(Boolean),
    [meta],
  );
  const comparisons: string[] = useMemo(
    () =>
      asList(meta?.comparisons ?? ["Previous Period"])
        .map((x) => strOf(typeof x === "string" ? x : x?.name || x?.label))
        .filter(Boolean),
    [meta],
  );
  const filterCols: { value: string; label: string }[] = useMemo(() => {
    const c = meta?.filterable_columns ?? meta?.filterColumns ?? breakdowns;
    return asList(c).map(optOf).filter((o) => o.value);
  }, [meta, breakdowns]);

  const [metric, setMetric] = useState("");
  const [breakdown, setBreakdown] = useState("");
  const [grain, setGrain] = useState("");
  const [comparison, setComparison] = useState("");
  const [rangeKey, setRangeKey] = useState<RangeKey>("12m");
  const [secondary, setSecondary] = useState("");
  const [showSecondary, setShowSecondary] = useState(false);
  const [filters, setFilters] = useState<ExploreFilter[]>([]);
  const [filterCol, setFilterCol] = useState("");
  const [filterVal, setFilterVal] = useState("");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState("");
  const [errorId, setErrorId] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState("");
  const [recent, setRecent] = useState<any[]>([]);
  const [savedCols, setSavedCols] = useState<any[]>([]);
  const [recentError, setRecentError] = useState("");
  /* The exact date_range value sent on the last run (dict or string) —
     drives the honest "filtered vs all-data" range chip. */
  const [lastRange, setLastRange] = useState<DateRangePayload | null>(null);

  const runAbort = useRef<AbortController | null>(null);
  const resultRef = useRef<any>(null);
  useEffect(() => {
    resultRef.current = result;
  }, [result]);

  /* Preselect first backend options once meta arrives — never invented. */
  useEffect(() => {
    if (!metric && metrics.length) setMetric(metrics[0].value);
    if (!breakdown && breakdowns.length) setBreakdown(breakdowns[0].value);
    if (!grain && grains.length) setGrain(grains[0]);
    if (!comparison && comparisons.length) setComparison(comparisons[0]);
    if (!filterCol && filterCols.length) setFilterCol(filterCols[0].value);
  }, [metrics, breakdowns, grains, comparisons, filterCols, metric, breakdown, grain, comparison, filterCol]);

  /* Hypothesis Test-this handoff: consume `aria_explore_prefill`
     ({metric, breakdown}) plus the drill-context pattern
     (`aria_dashboard_drill`) once backend options arrive. Values are
     matched against live options — never invented. One-shot. */
  const prefillApplied = useRef(false);
  useEffect(() => {
    if (prefillApplied.current) return;
    if (!metrics.length && !breakdowns.length) return;
    let prefill: { metric?: string; breakdown?: string } | null = null;
    try {
      const raw = sessionStorage.getItem("aria_explore_prefill");
      if (raw) {
        const parsed = JSON.parse(raw);
        prefill = {
          ...(parsed?.metric ? { metric: String(parsed.metric) } : {}),
          ...(parsed?.breakdown ? { breakdown: String(parsed.breakdown) } : {}),
        };
        sessionStorage.removeItem("aria_explore_prefill");
      }
    } catch { prefill = null; }
    if (!prefill?.metric && !prefill?.breakdown) {
      try {
        const raw = localStorage.getItem("aria_dashboard_drill");
        if (raw) {
          const parsed = JSON.parse(raw);
          if (parsed?.kind === "hypothesis" || parsed?.kind === "hypothesis-factor") {
            prefill = {
              ...(parsed?.metric ? { metric: String(parsed.metric) } : {}),
              ...((parsed?.breakdown || parsed?.dimension) ? { breakdown: String(parsed.breakdown || parsed.dimension) } : {}),
            };
          }
        }
      } catch { /* best-effort */ }
    }
    if (!prefill || (!prefill.metric && !prefill.breakdown)) return;
    const matchOpt = (opts: { value: string }[], v: string): string => {
      const s = (v || "").trim();
      if (!s) return "";
      return opts.find((o) => o.value === s)?.value
        || opts.find((o) => o.value.toLowerCase() === s.toLowerCase())?.value
        || "";
    };
    let applied = false;
    if (prefill.metric) {
      const m = matchOpt(metrics, prefill.metric);
      if (m) { setMetric(m); applied = true; }
    }
    if (prefill.breakdown) {
      const b = matchOpt(breakdowns, prefill.breakdown);
      if (b) { setBreakdown(b); applied = true; }
    }
    if (applied) prefillApplied.current = true;
  }, [metrics, breakdowns]);

  const loadRecent = useCallback(async () => {
    try {
      const r = await intelligenceApi.listAnalyses({ page: 1, page_size: 6, sort: "recent" });
      const items = asList((r as any).data);
      let reportItems: any[] = [];
      try {
        const rr = await intelligenceApi.listReports();
        reportItems = asList((rr as any).data)
          .slice(0, 6)
          .map((x: any) => ({
            ...x,
            collection: strOf(x?.collection || x?.template || "Reports") || "Reports",
          }));
      } catch {
        /* reports optional for Saved Collections */
      }
      setRecent(items.filter((a) => !a?.saved && !a?.collection));
      setSavedCols([...items.filter((a) => a?.saved || a?.collection), ...reportItems].slice(0, 12));
      setRecentError("");
    } catch (e: any) {
      console.warn("Failed to load recent analyses:", e?.message || e);
      setRecentError(businessError(e?.message) || "ARIA couldn't load recent analyses.");
    }
  }, []);
  useEffect(() => {
    loadRecent();
  }, [loadRecent]);

  const buildPayload = useCallback(
    (rangeOverride?: RangeKey) => ({
      metric: metric || undefined,
      breakdown: breakdown || undefined,
      secondary_breakdown: showSecondary && secondary ? secondary : undefined,
      time_grain: grain || undefined,
      comparison: comparison || undefined,
      date_range: dateRangeFor(rangeOverride ?? rangeKey, resultRef.current),
      filters: filters.filter((f) => f.column && f.value),
    }),
    [metric, breakdown, secondary, showSecondary, grain, comparison, rangeKey, filters],
  );

  const runAnalysis = useCallback(
    async (override?: Record<string, unknown>) => {
      runAbort.current?.abort();
      const ctrl = new AbortController();
      runAbort.current = ctrl;
      setRunning(true);
      setError("");
      setErrorId("");
      setSaveMsg("");
      try {
        const payload = { ...buildPayload(), ...(override || {}) };
        setLastRange((payload.date_range as DateRangePayload) ?? null);
        const r = await intelligenceApi.runExplore(payload);
        if (ctrl.signal.aborted) return;
        const rd: any = (r as any)?.data ?? null;
        if (rd && rd.ok === false) {
          const msg = rd.detail || rd.message || "This analysis could not be completed from your data.";
          const withHint = rd.code ? `${msg} (${rd.code})` : msg;
          setError(withHint);
          setErrorId("");
          setResult(null);
          return;
        }
        setResult(rd);
      } catch (e: any) {
        if (e?.name === "AbortError" || ctrl.signal.aborted) return;
        const norm = normalizeApiError({ message: e?.message }, e?.status || 0, null);
        setError(norm.message);
        setErrorId(norm.requestId);
        setResult(null);
      } finally {
        if (runAbort.current === ctrl) setRunning(false);
      }
    },
    [buildPayload],
  );

  useEffect(() => () => runAbort.current?.abort(), []);

  const autoRanFor = useRef<string>("");

  /* Auto-run once per dataset: when meta first yields metrics and a metric
     is preselected, run with the first metric + default breakdown. Also
     honors ?run=1 in the URL hash for deep links. Guarded by ref to avoid
     loops; manual Run always still works. */
  useEffect(() => {
    if (!metrics.length || !metric || result || running || error) return;
    let wantAuto = false;
    try {
      if (/[?&]run=1/.test(window.location.hash || "")) wantAuto = true;
    } catch {
      /* ignore */
    }
    if (!autoRanFor.current) wantAuto = true;
    if (!wantAuto) return;
    const key = `${metric}|${breakdown}|${grain}|${comparison}|${rangeKey}`;
    if (autoRanFor.current === key) return;
    autoRanFor.current = key;
    runAnalysis();
  }, [metrics, metric, breakdown, grain, comparison, rangeKey, result, running, error, runAnalysis]);

  /* Switching the range selector re-runs immediately with an explicit
     {start,end,label} window derived from the live trend (or the
     display-only label while no window is known yet). */
  const applyRange = useCallback(
    (key: RangeKey) => {
      setRangeKey(key);
      runAnalysis({ date_range: dateRangeFor(key, resultRef.current) });
    },
    [runAnalysis],
  );

  const reset = useCallback(() => {
    setSecondary("");
    setShowSecondary(false);
    setFilters([]);
    setFilterVal("");
    setResult(null);
    setError("");
    setErrorId("");
    setSaveMsg("");
    setLastRange(null);
    setRangeKey("12m");
    if (metrics.length) setMetric(metrics[0].value);
    if (breakdowns.length) setBreakdown(breakdowns[0].value);
    if (grains.length) setGrain(grains[0]);
    if (comparisons.length) setComparison(comparisons[0]);
  }, [metrics, breakdowns, grains, comparisons]);

  const loadSavedAnalysis = useCallback(
    async (a: any) => {
      const resolveOpt = (opts: { value: string; label: string }[], v: string): string => {
        const s = strOf(v);
        if (!s) return "";
        const hit = opts.find(
          (o) =>
            o.value === s ||
            o.label === s ||
            o.value.toLowerCase() === s.toLowerCase() ||
            o.label.toLowerCase() === s.toLowerCase(),
        );
        return hit ? hit.value : s;
      };
      const m = resolveOpt(metrics, strOf(a?.metric));
      const b = resolveOpt(breakdowns, strOf(a?.breakdown));
      const g = strOf(a?.time_grain);
      const c = strOf(a?.comparison);
      const savedRangeRaw = strOf(a?.date_range);
      const rk = rangeKeyForLabel(savedRangeRaw);
      if (m) setMetric(m);
      if (b) setBreakdown(b);
      if (g) setGrain(g);
      if (c) setComparison(c);
      const savedFilters = asList(a?.filters)
        .map((f: any) => ({ column: strOf(f?.column), value: strOf(f?.value) }))
        .filter((f) => f.column && f.value);
      if (asList(a?.filters).length) setFilters(savedFilters);
      /* Run with explicit overrides (no stale-closure wait): the date_range
         reuses the selector mapping when recognized, else the saved string
         as-is (display-only, backend handles it). */
      const nextRange = rk ?? rangeKey;
      if (rk) setRangeKey(rk);
      await runAnalysis({
        ...(m ? { metric: m } : {}),
        ...(b ? { breakdown: b } : {}),
        ...(g ? { time_grain: g } : {}),
        ...(c ? { comparison: c } : {}),
        date_range: rk ? dateRangeFor(rk, resultRef.current) : savedRangeRaw || dateRangeFor(nextRange, resultRef.current),
        ...(asList(a?.filters).length ? { filters: savedFilters } : {}),
      });
    },
    [metrics, breakdowns, rangeKey, runAnalysis],
  );

  const deleteSavedAnalysis = useCallback(
    async (id: string) => {
      if (!id) return;
      try {
        await intelligenceApi.deleteAnalysis(id);
        loadRecent();
      } catch (e: any) {
        setSaveMsg(businessError(e?.message) || "ARIA couldn't delete this analysis.");
      }
    },
    [loadRecent],
  );

  const saveAnalysis = useCallback(async () => {
    if (!result) return;
    setSaving(true);
    setSaveMsg("");
    try {
      await intelligenceApi.saveAnalysis({
        ...buildPayload(),
        result_summary: result?.observation?.summary || result?.summary || "",
      });
      setSaveMsg("Analysis saved. Find it under Recent Analyses.");
      loadRecent();
    } catch (e: any) {
      setSaveMsg(businessError(e?.message) || "ARIA couldn't save this analysis.");
    } finally {
      setSaving(false);
    }
  }, [result, buildPayload, loadRecent]);

  /* Honest range chip: what the last run actually sent. */
  const activeRangeLabel =
    typeof lastRange === "string"
      ? lastRange
      : lastRange
        ? lastRange.label
        : RANGE_OPTIONS.find((o) => o.key === rangeKey)?.label || "Last 12 months";
  const rangeIsExplicit = lastRange !== null && typeof lastRange !== "string";
  const windowDates =
    lastRange !== null && typeof lastRange !== "string"
      ? { start: lastRange.start, end: lastRange.end }
      : null;

  return {
    metrics,
    breakdowns,
    grains,
    comparisons,
    filterCols,
    metric,
    setMetric,
    breakdown,
    setBreakdown,
    grain,
    setGrain,
    comparison,
    setComparison,
    rangeKey,
    applyRange,
    activeRangeLabel,
    rangeIsExplicit,
    windowDates,
    secondary,
    setSecondary,
    showSecondary,
    setShowSecondary,
    filters,
    setFilters,
    filterCol,
    setFilterCol,
    filterVal,
    setFilterVal,
    running,
    result,
    error,
    errorId,
    saving,
    saveMsg,
    recent,
    savedCols,
    recentError,
    loadRecent,
    runAnalysis,
    buildPayload,
    reset,
    saveAnalysis,
    loadSavedAnalysis,
    deleteSavedAnalysis,
  };
}

export type UseExploreReturn = ReturnType<typeof useExplore>;

