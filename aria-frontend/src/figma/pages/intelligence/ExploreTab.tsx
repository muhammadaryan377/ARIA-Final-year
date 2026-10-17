/* Explore tab: builder card + trend + observation + breakdown + next steps
   + evidence drawer trigger. All numbers come from POST
   /api/intelligence/explore via useExplore — this file only renders. */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence, MotionConfig } from "framer-motion";
import {
  BarChart3, Compass, FlaskConical, AlertTriangle, Target, SlidersHorizontal,
  FileText, Play, RotateCcw, Bookmark, ChevronRight, RefreshCw, Sparkles,
  Send, X, ArrowUpRight, ArrowDownRight, Info, Search, CalendarDays,
  Download, Maximize2, TrendingUp, TrendingDown, ShieldCheck,
} from "lucide-react";
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, Cell,
} from "recharts";
import { colorForName } from "../../lib/charts";
import { fmtPctBiz } from "../../lib/format";
import {
  asList, strOf, numOf, humanize, titleCase, fmtMetric, getNoSchemaInfo,
} from "../../lib/intelHelpers";
import {
  Card, CardHead, NoDatasetPanel, ErrorPanel, LoadingPanel, Field, StatPill,
  CountUp, StaggerGroup, StaggerItem, fadeUp, Sparkline,
  selectCls, inputCls, btnPrimary, btnGhost,
} from "./cards";
import { useExplore, RANGE_OPTIONS, type RangeKey } from "./useExplore";
import { prescriptionApi } from "../../../services/api";
import EvidenceDrawer, { type EvidenceContext } from "./EvidenceDrawer";
import CompareView from "./CompareView";
import { TimeRangeSelector } from "../../components/TimeRangeSelector";
import type { IntelTab } from "../Intelligence";

export interface ExploreTabProps {
  meta: any;
  onRunQuick: (actionId: string) => void;
  onGoto: (t: IntelTab) => void;
  onAsk: (q: string) => void;
}

const QUICK_ACTIONS = [
  { id: "metric", label: "Explore a Metric", sub: "Analyze any metric", Icon: BarChart3 },
  { id: "segments", label: "Compare Segments", sub: "Compare performance", Icon: Compass },
  { id: "change", label: "Investigate a Change", sub: "Understand what changed", Icon: FlaskConical },
  { id: "anomaly", label: "Find Anomalies", sub: "Discover unusual patterns", Icon: AlertTriangle },
  { id: "trend", label: "Analyze a Trend", sub: "View trends over time", Icon: BarChart3 },
  { id: "compare", label: "Build a Comparison", sub: "Side-by-side analysis", Icon: Compass },
  { id: "goal", label: "Create a Goal", sub: "Track a business target", Icon: Target },
  { id: "scenario", label: "Run a Scenario", sub: "Test a what-if scenario", Icon: SlidersHorizontal },
] as const;

const HOME_SUGGESTIONS = [
  "Why did revenue change?",
  "Compare sales across regions",
  "Which products are declining?",
  "Show me customer growth",
  "Find unusual changes in orders",
];

interface TrendPoint {
  label: string;
  current: number;
  previous: number | null;
}

/* Animated chart (mount animation via recharts, ~600ms). Module-level so it
   never remounts on parent renders; reused by the fullscreen modal. */
function TrendChart({
  points,
  mode,
  axisMetric,
  tipMetric,
}: {
  points: TrendPoint[];
  mode: "bar" | "line";
  axisMetric: string;
  tipMetric: string;
}) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      {mode === "bar" ? (
        <BarChart data={points} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
          <XAxis dataKey="label" tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} minTickGap={24} />
          <YAxis tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} width={52} tickFormatter={(v: number) => fmtMetric(v, axisMetric)} />
          <Tooltip formatter={(v: any, name: any) => [fmtMetric(numOf(v), tipMetric), humanize(String(name))]} contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="current" name="Current Period" fill="#2F6FAD" radius={[4, 4, 0, 0]} isAnimationActive animationDuration={600} />
          <Bar dataKey="previous" name="Previous Period" fill="#B9D0E8" radius={[4, 4, 0, 0]} isAnimationActive animationDuration={600} />
        </BarChart>
      ) : (
        <LineChart data={points} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
          <XAxis dataKey="label" tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} minTickGap={24} />
          <YAxis tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} width={52} tickFormatter={(v: number) => fmtMetric(v, axisMetric)} />
          <Tooltip formatter={(v: any, name: any) => [fmtMetric(numOf(v), tipMetric), humanize(String(name))]} contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line type="monotone" dataKey="current" name="Current Period" stroke="#2F6FAD" strokeWidth={2.5} dot={false} isAnimationActive animationDuration={600} />
          <Line type="monotone" dataKey="previous" name="Previous Period" stroke="#B9D0E8" strokeWidth={2} strokeDasharray="5 4" dot={false} isAnimationActive animationDuration={600} />
        </LineChart>
      )}
    </ResponsiveContainer>
  );
}

export default function ExploreTab({ meta, onRunQuick, onGoto, onAsk }: ExploreTabProps) {
  const ex = useExplore(meta);
  const {
    metrics, breakdowns, grains, comparisons, filterCols,
    metric, setMetric, breakdown, setBreakdown, grain, setGrain,
    comparison, setComparison, rangeKey, applyRange,
    activeRangeLabel, rangeIsExplicit, windowDates,
    secondary, setSecondary, showSecondary, setShowSecondary,
    filters, setFilters, filterCol, setFilterCol, filterVal, setFilterVal,
    running, result, error, errorId, saving, saveMsg,
    recent, savedCols, recentError, loadRecent,
    runAnalysis, reset, saveAnalysis, loadSavedAnalysis, deleteSavedAnalysis,
  } = ex;

  const [chartMode, setChartMode] = useState<"bar" | "line">("bar");
  const [homeQuery, setHomeQuery] = useState("");
  const [showFilterRow, setShowFilterRow] = useState(false);
  const [showAllRecent, setShowAllRecent] = useState(false);
  const [showAllSaved, setShowAllSaved] = useState(false);
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [showCompare, setShowCompare] = useState(false);
  /* Alerts that matter + one-question answers (home). */
  const [alerts, setAlerts] = useState<any[]>([]);
  const [alertsState, setAlertsState] = useState<"idle" | "loading" | "ready">("idle");
  const [askText, setAskText] = useState("");
  const [askState, setAskState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [askAnswer, setAskAnswer] = useState<any>(null);
  const [askError, setAskError] = useState("");

  useEffect(() => {
    if (result || alertsState !== "idle") return;
    setAlertsState("loading");
    (async () => {
      try {
        const r = await (prescriptionApi as any).alerts();
        setAlerts(asList((r as any)?.data?.alerts));
      } catch {
        setAlerts([]);
      } finally {
        setAlertsState("ready");
      }
    })();
  }, [result, alertsState]);

  const askOne = useCallback(async () => {
    const q = askText.trim();
    if (!q) return;
    setAskState("loading");
    setAskError("");
    setAskAnswer(null);
    try {
      /* The engine needs a metric to test — default to the first available
         metric so a bare question still gets a real answer. */
      const fallbackMetric = strOf(metrics[0]?.value || metrics[0]?.label);
      const [a, b] = await Promise.all([
        (prescriptionApi as any).analyze({ question: q, ...(fallbackMetric ? { metric: fallbackMetric } : {}) }),
        (prescriptionApi as any).inaction({ ...(fallbackMetric ? { metric: fallbackMetric } : {}) }),
      ]);
      const ad: any = (a as any)?.data ?? null;
      const bd: any = (b as any)?.data ?? null;
      const list = asList(ad?.prescriptions);
      setAskAnswer({ top: list[0] || null, refusal: ad?.status === "no_prescription" ? ad : null, inaction: bd });
      setAskState("ready");
    } catch (e: any) {
      setAskError(e?.message || "ARIA couldn't answer that just now.");
      setAskState("error");
    }
  }, [askText]);
  const csvUrlRef = useRef<string | null>(null);

  /* Recommended Actions — powered ONLY by the existing prescription engine
     (POST /api/prescriptions/analyze). Auto-fetched once per explore result;
     honest refusal (ok:False / no_prescription / empty) omits the card. */
  const [rxStatus, setRxStatus] = useState<"idle" | "loading" | "ready" | "refused" | "error">("idle");
  const [rxItems, setRxItems] = useState<any[]>([]);
  const [rxError, setRxError] = useState("");
  const [rxRefusal, setRxRefusal] = useState<any>(null);
  const [rxQuestion, setRxQuestion] = useState("");
  const [rxSavingId, setRxSavingId] = useState<string | null>(null);
  const [rxSavedIds, setRxSavedIds] = useState<string[]>([]);
  const [rxSaveMsg, setRxSaveMsg] = useState("");
  const rxFetchedFor = useRef<string>("");

  /* Cost of inaction — fetched once per explore result from the existing
     engine endpoint; honest empty when history is short. */
  const [inaction, setInaction] = useState<any>(null);
  const inactionFor = useRef<string>("");

  useEffect(
    () => () => {
      if (csvUrlRef.current) {
        try {
          URL.revokeObjectURL(csvUrlRef.current);
        } catch {
          /* ignore */
        }
      }
    },
    [],
  );

  /* Replay the stagger-in on every fresh result load. */
  const [runId, setRunId] = useState(0);
  const prevResult = useRef<any>(null);
  useEffect(() => {
    if (result && result !== prevResult.current) {
      prevResult.current = result;
      setRunId((id) => id + 1);
    }
  }, [result]);

  const metricRef = useRef<HTMLSelectElement>(null);
  const focusedMetricOnce = useRef(false);

  /* Guide the home state: focus the metric selector once so keyboard users
     land on the first step. */
  useEffect(() => {
    if (!result && !running && metrics.length && metricRef.current && !focusedMetricOnce.current) {
      focusedMetricOnce.current = true;
      try {
        metricRef.current.focus({ preventScroll: true });
      } catch {
        try {
          metricRef.current.focus();
        } catch {
          /* ignore */
        }
      }
    }
  }, [result, running, metrics.length]);

  /* Fullscreen chart modal: Escape to close + scroll lock. */
  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setExpanded(false);
    };
    document.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [expanded]);

  const relativeOpened = (iso: string): string => {
    if (!iso) return "";
    try {
      const d = new Date(iso);
      const now = new Date();
      const sameDay = d.toDateString() === now.toDateString();
      const yest = new Date(now);
      yest.setDate(now.getDate() - 1);
      const time = d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
      if (sameDay) return `Opened today, ${time}`;
      if (d.toDateString() === yest.toDateString()) return `Opened yesterday, ${time}`;
      return `Opened ${d.toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" })}`;
    } catch {
      return strOf(iso);
    }
  };

  const savedBuckets = (() => {
    const map = new Map<string, any[]>();
    for (const a of savedCols) {
      const key = strOf(
        a?.metric
          ? humanize(strOf(a.metric))
          : a?.collection
            ? strOf(a.collection)
            : a?.title
              ? strOf(a.title).split(" by ")[0]
              : "General",
      );
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(a);
    }
    return Array.from(map.entries()).map(([name, items]) => ({ name, count: items.length, items }));
  })();

  const submitHomeQuery = () => {
    const q = homeQuery.trim();
    if (!q) {
      runAnalysis();
      return;
    }
    onAsk(q);
    runAnalysis();
  };

  const quickRun = (actionId: string) => {
    onRunQuick(actionId);
    if (actionId === "goal") {
      onGoto("goals");
      return;
    }
    if (actionId === "scenario") {
      onGoto("whatif");
      return;
    }
    const override: Record<string, unknown> = {};
    if (actionId === "compare" && comparisons.length > 1) {
      setComparison(comparisons[1]);
      override.comparison = comparisons[1];
    }
    if (actionId === "trend" && grains.length) {
      setGrain(grains[0]);
      override.time_grain = grains[0];
    }
    runAnalysis(override);
  };

  /* ---- result normalization (backend contract: trend.points[{label,
     current,previous}]; rows[{category,current,previous,change,
     change_pct,trend:[nums]}]; observation{summary,bullets,basis_note}.
     Defensive fallbacks for legacy keys; previous null renders as "—". ---- */
  const trendPoints: TrendPoint[] = useMemo(() => {
    const raw = asList(result?.trend?.points || result?.trend || result?.series || result?.points || result?.chart?.data);
    return raw.slice(0, 60).map((p: any, i: number) => ({
      label: strOf(p?.label || p?.period || p?.x || p?.month || `P${i + 1}`),
      current: numOf(p?.current ?? p?.current_value ?? p?.value ?? p?.y) ?? 0,
      previous: numOf(p?.previous ?? p?.previous_value ?? p?.prev),
    }));
  }, [result]);
  const trendTitle = strOf(
    result?.trend?.title || result?.title || (metric && breakdown ? `${titleCase(metric)} Trend by ${titleCase(breakdown)}` : "Trend"),
    "Trend",
  );
  const trendSub = strOf(result?.trend?.subtitle || result?.subtitle || "");
  const observation = result?.observation || result?.aria_observation || null;
  const obsSummary = strOf(
    typeof observation === "string" ? observation : observation?.summary || observation?.text || result?.summary || "",
  );
  const obsBullets: string[] = asList(
    typeof observation === "object" ? observation?.bullets || observation?.points || observation?.details : [],
  )
    .map((b) => strOf(typeof b === "string" ? b : b?.text || b?.label))
    .filter(Boolean)
    .slice(0, 8);
  const obsBasis = strOf(observation?.basis_note || observation?.basis || result?.basis_note || "");
  const rows = useMemo(
    () => asList(result?.rows || result?.breakdown || result?.breakdown_rows || result?.table),
    [result],
  );
  const breakdownTitle = strOf(
    result?.breakdown_title || (breakdown ? `${titleCase(breakdown)} Breakdown` : "Breakdown"),
    "Breakdown",
  );
  /* Backend-computed totals for the KPI strip (never summed client-side). */
  const total = numOf(result?.total);
  const prevTotal = numOf(result?.previous_total);
  const totalChange = numOf(result?.change);
  const totalChangePct = numOf(result?.change_pct ?? result?.changePct);

  /* ---- segment drill-down: the active filter on the breakdown column
     (if any) is the drilled segment. Clicking a row replaces any existing
     filter on that column and re-runs; the backend joins cross-table. ---- */
  const activeDrill = breakdown ? filters.find((f) => f.column === breakdown) : undefined;
  const drillSegment = (category: string) => {
    if (!breakdown || !category || running) return;
    const next = [...filters.filter((f) => f.column !== breakdown), { column: breakdown, value: category }];
    setFilters(next);
    runAnalysis({ filters: next });
  };
  const clearDrill = () => {
    if (!breakdown) return;
    const next = filters.filter((f) => f.column !== breakdown);
    setFilters(next);
    runAnalysis({ filters: next });
  };
  const drillable = !!breakdown && !running;

  /* ---- Recommended Actions: fetch from the EXISTING prescription engine.
     Payload per contract: { metric, driver, dimensions[], segment, question }.
     Segment sends both `field` (route contract) and `column` (explore filter
     shape); the backend ignores extras. No client-side recommendation logic. */
  const fetchRecommendations = useCallback(
    async (force = false, ov: { metric?: string; driver?: string } | null = null) => {
      const m = strOf(ov?.metric || metric);
      if (!result || !m) return;
      const drv = strOf(ov?.driver !== undefined ? ov.driver : breakdown);
      const seg = drv && !ov ? filters.find((f) => f.column === drv) : undefined;
      const key = `${runId}|${m}|${drv}|${seg ? `${seg.column}=${seg.value}` : ""}`;
      if (!force && rxFetchedFor.current === key) return;
      rxFetchedFor.current = key;
      const question = drv
        ? `What should I consider for ${m} by ${drv}?`
        : `What should I consider for ${m}?`;
      setRxQuestion(question);
      setRxStatus("loading");
      setRxError("");
      setRxRefusal(null);
      setRxSaveMsg("");
      try {
        const payload: Record<string, unknown> = { metric: m, question };
        if (drv) {
          payload.driver = drv;
          payload.dimensions = [drv];
        }
        if (seg) {
          payload.segment = { field: seg.column, value: seg.value, column: seg.column };
        }
        const r = await prescriptionApi.analyze(payload);
        const d: any = (r as any)?.data ?? null;
        const list = asList(d?.prescriptions);
        /* Honest refusal (ok:False, no_prescription, or empty) -> guidance panel. */
        if (!d || d.ok === false || d.status === "no_prescription" || list.length === 0) {
          setRxStatus("refused");
          setRxItems([]);
          setRxRefusal(d?.refusal && typeof d.refusal === "object" ? d.refusal : null);
          return;
        }
        setRxItems(list.slice(0, 5));
        setRxRefusal(null);
        setRxStatus("ready");
      } catch (e: any) {
        setRxStatus("error");
        setRxError(e?.message || "ARIA couldn't load recommendations.");
      }
    },
    [result, runId, metric, breakdown, filters],
  );

  /* Auto-fetch ONCE per successful explore result (guard like auto-run);
     manual "Get recommendations" button forces a retry. */
  useEffect(() => {
    if (!result) {
      rxFetchedFor.current = "";
      setRxStatus((s) => (s === "idle" ? s : "idle"));
      setRxItems([]);
      setRxError("");
      setRxRefusal(null);
      return;
    }
    void fetchRecommendations(false);
  }, [result, fetchRecommendations]);

  /* Cost of inaction: one fetch per result via the engine endpoint. */
  useEffect(() => {
    if (!result || !metric) {
      inactionFor.current = "";
      setInaction(null);
      return;
    }
    const key = `${runId}|${metric}`;
    if (inactionFor.current === key) return;
    inactionFor.current = key;
    let cancelled = false;
    (async () => {
      try {
        const r = await (prescriptionApi as any).inaction({ metric });
        if (!cancelled) setInaction((r as any)?.data ?? null);
      } catch {
        if (!cancelled) setInaction(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [result, runId, metric]);

  const saveRecommendation = useCallback(
    async (p: any, idx: number) => {
      const id = strOf(p?.id || `rx-${idx}`);
      if (rxSavingId || rxSavedIds.includes(id)) return;
      setRxSavingId(id);
      setRxSaveMsg("");
      try {
        await (prescriptionApi as any).save(p, rxQuestion || null);
        setRxSavedIds((s) => (s.includes(id) ? s : [...s, id]));
        setRxSaveMsg("Recommendation saved.");
      } catch (e: any) {
        setRxSaveMsg(e?.message || "ARIA couldn't save this recommendation.");
      } finally {
        setRxSavingId(null);
      }
    },
    [rxSavingId, rxSavedIds, rxQuestion],
  );

  /* Display-safe helpers over the route's prescription shape (display +
     details first, legacy fallbacks). Primary views NEVER render raw
     r/p/n/delta: cards show display fields; raw numbers live only in the
     "Why is this suggested?" expander. Never invent numbers. */
  const rxDispOf = (p: any): any =>
    p?.display && typeof p.display === "object" ? p.display : {};
  const rxDetailsOf = (p: any): any =>
    p?.details && typeof p.details === "object" ? p.details : {};
  const rxNumbersOf = (p: any): Record<string, unknown> => {
    const n = rxDetailsOf(p)?.numbers;
    return n && typeof n === "object" ? n : {};
  };
  const rxLimsOf = (p: any): string[] => {
    const det = rxDetailsOf(p);
    const src = asList(det.limitations).length ? det.limitations : p?.limitations;
    return asList(src).map((x) => strOf(typeof x === "string" ? x : x?.text || x?.label)).filter(Boolean);
  };
  const rxFmtNumberRow = (key: string, v: unknown): string => {
    if (typeof v === "string") return v;
    const n = numOf(v);
    if (n === null) return strOf(v);
    const k = key.toLowerCase();
    if (k === "n" || k === "n_points") return Math.round(n).toLocaleString();
    if (k.includes("pct") || k.includes("gap") || k.includes("share") || k.includes("momentum") || k === "rel_pct") {
      return `${n >= 0 ? "+" : ""}${Math.round(n * 10) / 10}%`;
    }
    if (k === "p_value") return n < 0.001 ? "<0.001" : n.toFixed(3);
    if (k === "effect_r" || k === "r" || k === "spearman_r" || k === "slope") {
      return `${n >= 0 ? "+" : ""}${Math.round(n * 100) / 100}`;
    }
    if (["baseline", "scenario", "delta", "first", "last", "mean", "driver_value"].includes(k)) {
      return fmtMetric(n, metric);
    }
    return strOf(v);
  };

  const rxConfidenceTone = (c: unknown): "up" | "info" | "neutral" | "warn" => {
    const s = String(c ?? "").toLowerCase();
    if (s === "high") return "up";
    if (s === "moderate" || s === "medium") return "info";
    if (s === "low" || s === "limited") return "warn";
    return "neutral";
  };

  const nextSteps = [
    { label: "Investigate this pattern", sub: "Explore possible factors", tab: "hypotheses" as IntelTab, Icon: FlaskConical },
    { label: "Test a scenario", sub: "See what might happen", tab: "whatif" as IntelTab, Icon: SlidersHorizontal },
    { label: "Compare with a goal", sub: "Check against your targets", tab: "goals" as IntelTab, Icon: Target },
    { label: "Create a report", sub: "Turn this analysis into a report", tab: "reports" as IntelTab, Icon: FileText },
  ];

  /* Export the live breakdown rows as CSV (Blob download, real data only). */
  const exportCsv = () => {
    if (!rows.length) return;
    const esc = (v: unknown): string => {
      const s = v === null || v === undefined ? "" : String(v);
      return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const lines = ["category,current,previous,change,change_pct"];
    for (const r of rows) {
      const cat = strOf(r?.category ?? r?.label ?? r?.name);
      const cur = numOf(r?.current ?? r?.current_value ?? r?.value);
      const prev = numOf(r?.previous ?? r?.previous_value ?? r?.prev);
      const chg = numOf(r?.change ?? r?.delta);
      const pct = numOf(r?.change_pct ?? r?.pct_change ?? r?.pct ?? r?.percent_change);
      lines.push(
        [esc(cat), cur ?? "", prev ?? "", chg ?? "", pct ?? ""].map(esc).join(","),
      );
    }
    try {
      if (csvUrlRef.current) {
        try {
          URL.revokeObjectURL(csvUrlRef.current);
        } catch {
          /* ignore */
        }
      }
      const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      csvUrlRef.current = url;
      const a = document.createElement("a");
      a.href = url;
      const base =
        [metric, breakdown]
          .map((s) => strOf(s).replace(/[^a-z0-9-_]+/gi, "-").toLowerCase())
          .filter(Boolean)
          .join("-by-") || "analysis";
      a.download = `${base}-breakdown.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch {
      /* ignore */
    }
  };

  const evidenceCtx: EvidenceContext = useMemo(
    () => ({
      metric,
      breakdown,
      comparison,
      trendTitle,
      trendSub,
      rows,
      trendPoints,
      obsSummary,
      obsBullets,
      obsBasis,
    }),
    [metric, breakdown, comparison, trendTitle, trendSub, rows, trendPoints, obsSummary, obsBullets, obsBasis],
  );

  const hasDataset = metrics.length > 0;
  if (!hasDataset) {
    const noSchema = getNoSchemaInfo(meta);
    return (
      <div className="space-y-4">
        <NoDatasetPanel hint={noSchema.isNoSchema ? noSchema.hint : undefined} />
      </div>
    );
  }
  return (
    <MotionConfig reducedMotion="user">
      <div className="space-y-4">
        <motion.div variants={fadeUp} initial="hidden" animate="show">
          <Card className="p-4 md:p-5">
            <div className="flex items-start justify-between gap-3 flex-wrap">
              <div>
                <h3 className="text-[14px] font-semibold text-foreground tracking-tight flex items-center gap-1.5">
                  <Sparkles size={14} className="text-blue-700 dark:text-blue-400" /> Build Your Analysis
                </h3>
                <p className="text-[11px] text-muted-foreground mt-0.5">Select what you want to analyze. ARIA will calculate and visualize the results from your data.</p>
              </div>
              <div className="flex items-center gap-2">
                <button onClick={saveAnalysis} disabled={!result || saving} className={btnGhost} aria-label="Save analysis">
                  <Bookmark size={13} /> {saving ? "Saving…" : "Save Analysis"}
                </button>
                <button onClick={reset} className={btnGhost} aria-label="Reset analysis builder">
                  <RotateCcw size={13} /> Reset
                </button>
              </div>
            </div>
            {saveMsg && <p className="mt-2 text-[11px] text-secondary-foreground" role="status">{saveMsg}</p>}

            <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 items-end">
              <Field label="Metric" htmlFor="intel-metric">
                <select id="intel-metric" ref={metricRef} value={metric} onChange={(e) => setMetric(e.target.value)} className={selectCls} aria-label="Metric">
                  {metrics.length === 0 && <option value="">No metrics available</option>}
                  {metrics.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}
                </select>
              </Field>
              <Field label="Break down by" htmlFor="intel-breakdown">
                <select id="intel-breakdown" value={breakdown} onChange={(e) => setBreakdown(e.target.value)} className={selectCls} aria-label="Break down by">
                  <option value="">Overall (no breakdown)</option>
                  {breakdowns.map((b) => <option key={b.value} value={b.value}>{b.label}</option>)}
                </select>
              </Field>
              <Field label="Time" htmlFor="intel-time">
                <select id="intel-time" value={grain} onChange={(e) => setGrain(e.target.value)} className={selectCls} aria-label="Time granularity">
                  {grains.map((g) => <option key={g} value={g}>{humanize(g)}</option>)}
                </select>
              </Field>
              <Field label="Compare" htmlFor="intel-compare">
                <select id="intel-compare" value={comparison} onChange={(e) => setComparison(e.target.value)} className={selectCls} aria-label="Comparison">
                  {comparisons.map((c) => <option key={c} value={c}>{humanize(c)}</option>)}
                </select>
              </Field>
              <div className="min-w-0">
                <span className="hidden lg:block text-[11px] font-medium text-transparent mb-1 select-none">Run</span>
                <button onClick={() => runAnalysis()} disabled={running || !metric} className={`${btnPrimary} w-full !h-[40px]`} aria-label="Run analysis">
                  {running ? "Running..." : "Run Analysis"} <Play size={13} />
                </button>
              </div>
            </div>

            <div className="mt-3 flex items-center gap-3 flex-wrap">
              <div className="flex items-center gap-3">
                {!showSecondary ? (
                  <button onClick={() => setShowSecondary(true)} className="text-[12px] font-medium text-blue-700 dark:text-blue-400 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded" aria-label="Add secondary breakdown">
                    + Add secondary breakdown
                  </button>
                ) : (
                  <div className="flex items-center gap-2 min-w-0">
                    <select value={secondary} onChange={(e) => setSecondary(e.target.value)} className={`${selectCls} !w-48`} aria-label="Secondary breakdown">
                      <option value="">Secondary breakdown…</option>
                      {breakdowns.filter((b) => b.value !== breakdown).map((b) => <option key={b.value} value={b.value}>{b.label}</option>)}
                    </select>
                    <button onClick={() => { setShowSecondary(false); setSecondary(""); }} className="text-muted-foreground hover:text-foreground" aria-label="Remove secondary breakdown"><X size={14} /></button>
                  </div>
                )}
              </div>
              <span className="flex-1" />
              {!showFilterRow ? (
                <button onClick={() => setShowFilterRow(true)} className="text-[12px] font-medium text-blue-700 dark:text-blue-400 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded" aria-label="Add filter">
                  + Add filter
                </button>
              ) : (
                <div className="flex items-center gap-2 min-w-0">
                  <select value={filterCol} onChange={(e) => setFilterCol(e.target.value)} className={`${selectCls} !w-40`} aria-label="Filter column">
                    {filterCols.map((c) => <option key={c.value} value={c.value}>{c.label}</option>)}
                  </select>
                  <input value={filterVal} onChange={(e) => setFilterVal(e.target.value)} placeholder="value…" className={`${inputCls} !w-32`} aria-label="Filter value" />
                  <button
                    onClick={() => { if (filterCol && filterVal.trim()) { setFilters((f) => [...f, { column: filterCol, value: filterVal.trim() }]); setFilterVal(""); setShowFilterRow(false); } }}
                    disabled={!filterCol || !filterVal.trim()}
                    className="text-[12px] font-medium text-blue-700 dark:text-blue-400 hover:underline disabled:opacity-40"
                  >
                    + Add filter
                  </button>
                  <button onClick={() => setShowFilterRow(false)} className="text-muted-foreground hover:text-foreground" aria-label="Collapse filter"><X size={14} /></button>
                </div>
              )}
              <TimeRangeSelector
                options={RANGE_OPTIONS.map((o) => ({ key: o.key, label: o.key === "all" ? "All data" : o.label.replace("Last ", "") }))}
                value={rangeKey}
                onChange={(k) => applyRange(k as RangeKey)}
              />
            </div>

            <div className="mt-2 flex items-center gap-1.5 flex-wrap" aria-label="Active filters and range">
              <StatPill tone={rangeIsExplicit ? "info" : "neutral"} title={rangeIsExplicit && windowDates ? `Filtered ${windowDates.start} → ${windowDates.end}` : "All data (no date filter)"}>
                <CalendarDays size={11} />
                {activeRangeLabel}
                {rangeIsExplicit && windowDates ? ` · ${windowDates.start} → ${windowDates.end}` : ""}
              </StatPill>
              {filters.map((f, i) => (
                <span key={i} className="inline-flex items-center gap-1 text-[11px] bg-blue-50 dark:bg-blue-500/10 text-blue-800 dark:text-blue-300 border border-blue-100 dark:border-blue-500/20 rounded-full px-2.5 py-1">
                  {humanize(f.column)} = {f.value}
                  <button onClick={() => setFilters((list) => list.filter((_, j) => j !== i))} aria-label={`Remove filter ${f.column}`} className="hover:text-blue-900 dark:hover:text-blue-300"><X size={11} /></button>
                </span>
              ))}
            </div>
          </Card>
        </motion.div>

        {/* Running */}
        {running && <LoadingPanel label="Running analysis — calculating from your data…" />}

        {/* Error */}
        {!running && error && <ErrorPanel message={error} requestId={errorId} onRetry={() => runAnalysis()} retryLabel="Run again" />}

        {/* Results: workspace header */}
        {!running && !error && result && (
          <div className="text-[11px] text-muted-foreground flex items-center gap-1">
            <span>Intelligence</span><ChevronRight size={11} /><span className="text-secondary-foreground font-medium">Explore</span>
          </div>
        )}
        {!running && !error && result && (
          <div className="flex items-start justify-between gap-3 flex-wrap">
            <div>
              <h3 className="text-[16px] font-semibold text-foreground tracking-tight">{trendTitle}</h3>
              <p className="text-[12px] text-muted-foreground">Explore and analyze your business data.</p>
            </div>
            <div className="flex items-center gap-1.5">
              <button onClick={saveAnalysis} disabled={saving} className={btnGhost} aria-label="Save result">Save</button>
              <button onClick={() => onAsk(`Share this analysis: ${humanize(metric)} by ${humanize(breakdown)}`)} className={btnGhost} aria-label="Share result">Share</button>
              <button onClick={exportCsv} disabled={!rows.length} className={btnGhost} aria-label="Export breakdown as CSV" title={rows.length ? "Download the live breakdown rows as CSV" : "No rows to export"}>
                Export
              </button>
            </div>
          </div>
        )}
        {/* Results */}
        {!running && !error && result && (
          <StaggerGroup key={runId} className="space-y-4 chart-reveal">
            {total !== null && (
              <StaggerItem>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3" aria-label="Totals">
                  <Card className="p-4">
                    <p className="text-[11px] font-medium text-muted-foreground">Total · current period</p>
                    <p className="text-[24px] font-bold text-foreground tabular-nums tracking-tight mt-0.5">
                      <CountUp value={total} format={(n) => fmtMetric(n, result?.metric || metric)} />
                    </p>
                    <p className="text-[11px] text-muted-foreground mt-0.5">{trendSub || activeRangeLabel}</p>
                  </Card>
                  <Card className="p-4">
                    <p className="text-[11px] font-medium text-muted-foreground">Total · previous period</p>
                    <p className="text-[24px] font-bold text-foreground tabular-nums tracking-tight mt-0.5">
                      <CountUp value={prevTotal} format={(n) => fmtMetric(n, result?.metric || metric)} />
                    </p>
                    <p className="text-[11px] text-muted-foreground mt-0.5">{strOf(result?.comparison || comparison || "Previous period")}</p>
                  </Card>
                  <Card className="p-4">
                    <p className="text-[11px] font-medium text-muted-foreground">Change</p>
                    <p className={`text-[24px] font-bold tabular-nums tracking-tight mt-0.5 flex items-center gap-1.5 ${(totalChange ?? 0) >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                      {(totalChange ?? 0) >= 0 ? <ArrowUpRight size={20} /> : <ArrowDownRight size={20} />}
                      <CountUp value={totalChange === null ? null : Math.abs(totalChange)} format={(n) => fmtMetric(n, result?.metric || metric)} />
                      {totalChangePct !== null && (
                        <StatPill tone={totalChangePct >= 0 ? "up" : "down"} className="ml-1">
                          {fmtPctBiz(totalChangePct)}
                        </StatPill>
                      )}
                    </p>
                    <p className="text-[11px] text-muted-foreground mt-0.5">Current vs previous</p>
                  </Card>
                </div>
              </StaggerItem>
            )}
            <StaggerItem>
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <Card className="lg:col-span-2 p-4 md:p-5">
                  <CardHead
                    title={trendTitle}
                    sub={trendSub}
                    right={
                      <div className="flex items-center gap-1.5">
                        <StatPill tone={rangeIsExplicit ? "info" : "neutral"} title={rangeIsExplicit && windowDates ? `Filtered ${windowDates.start} → ${windowDates.end}` : activeRangeLabel}>
                          {activeRangeLabel}
                        </StatPill>
                        <div className="flex items-center gap-1.5 text-[11px] font-medium border border-border rounded-lg px-2 py-1" aria-label="Chart type">
                          <BarChart3 size={12} className="text-blue-700 dark:text-blue-400" />
                          <select value={chartMode} onChange={(e) => setChartMode(e.target.value as "bar" | "line")} className="bg-transparent outline-none cursor-pointer text-foreground" aria-label="Chart type">
                            <option value="bar">Bar Chart</option>
                            <option value="line">Line Chart</option>
                          </select>
                        </div>
                        <button
                          onClick={() => setExpanded(true)}
                          className="inline-flex items-center justify-center w-7 h-7 rounded-lg border border-border text-muted-foreground hover:text-foreground hover:border-border focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                          aria-label="Expand chart"
                          title="Expand chart"
                        >
                          <Maximize2 size={13} />
                        </button>
                      </div>
                    }
                  />
                  {trendPoints.length > 0 ? (
                    <div className="h-[260px] mt-2">
                      <TrendChart points={trendPoints} mode={chartMode} axisMetric={metric || trendTitle} tipMetric={trendTitle} />
                    </div>
                  ) : (
                    <p className="text-xs text-muted-foreground mt-3">The backend returned no trend points for this selection — try a different metric or range.</p>
                  )}
                </Card>

                <Card className="p-4 md:p-5">
                  <h3 className="text-[14px] font-semibold text-foreground tracking-tight flex items-center gap-1.5">
                    <Sparkles size={14} className="text-blue-700 dark:text-blue-400" /> ARIA Observation
                  </h3>
                  {obsSummary ? (
                    <div className="mt-2">
                      <p className="text-[12.5px] text-secondary-foreground leading-relaxed">{obsSummary}</p>
                      {obsBullets.length > 0 && (
                        <ul className="mt-2 space-y-1.5">
                          {obsBullets.map((b, i) => (
                            <li key={i} className="flex items-start gap-1.5 text-[12px] text-secondary-foreground leading-relaxed">
                              <span className="w-1.5 h-1.5 rounded-full bg-blue-600 flex-shrink-0 mt-1.5" />{b}
                            </li>
                          ))}
                        </ul>
                      )}
                      {(obsBasis || `This analysis is based on your connected data${trendSub ? ` (${trendSub})` : ""}.`) && (
                        <div className="mt-3 flex items-start gap-1.5 text-[11px] text-muted-foreground bg-blue-50/60 dark:bg-blue-500/10 border border-blue-100 dark:border-blue-500/20 rounded-xl px-2.5 py-2">
                          <Info size={13} className="text-blue-600 dark:text-blue-400 flex-shrink-0 mt-px" />
                          <span>{obsBasis || `This analysis is based on your connected data${trendSub ? ` (${trendSub})` : ""}.`}</span>
                        </div>
                      )}
                      <button onClick={() => setEvidenceOpen(true)} className="mt-2 inline-flex items-center gap-1 text-[11.5px] font-semibold text-blue-700 dark:text-blue-400 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded" aria-label="View evidence">View Evidence <ChevronRight size={12} /></button>
                    </div>
                  ) : (
                    <p className="text-xs text-muted-foreground mt-2">No observation was returned for this analysis.</p>
                  )}
                </Card>
              </div>
            </StaggerItem>

            {/* Recommended Actions — engine prescriptions only (no templates).
                Refusal renders as guidance with one-click alternatives. */}
            {(rxStatus === "loading" || rxStatus === "ready" || rxStatus === "refused" || rxStatus === "error") && (
              <StaggerItem>
                <Card className="p-4 md:p-5">
                  <div className="flex items-start justify-between gap-3 flex-wrap">
                    <div>
                      <h3 className="text-[14px] font-semibold text-foreground tracking-tight flex items-center gap-1.5">
                        <Sparkles size={14} className="text-blue-700 dark:text-blue-400" /> Recommended Actions
                      </h3>
                      <p className="text-[11px] text-muted-foreground mt-0.5">
                        Evidence-backed suggestions for {humanize(metric || "this metric")}
                        {breakdown ? ` by ${humanize(breakdown)}` : ""} — from the prescription engine, never templates.
                      </p>
                    </div>
                    <button
                      onClick={() => void fetchRecommendations(true)}
                      disabled={rxStatus === "loading" || running}
                      className={btnGhost}
                      aria-label="Get recommendations"
                    >
                      <RefreshCw size={13} /> {rxStatus === "loading" ? "Loading…" : "Get recommendations"}
                    </button>
                  </div>

                  {rxStatus === "loading" && (
                    <div className="mt-3 space-y-2" aria-busy="true" aria-label="Loading recommendations">
                      {[0, 1].map((i) => (
                        <div key={i} className="rounded-xl border border-border p-3 space-y-2">
                          <div className="skeleton h-4 rounded" style={{ width: "55%" }} />
                          <div className="skeleton h-3 rounded" style={{ width: "90%" }} />
                          <div className="skeleton h-3 rounded" style={{ width: "70%" }} />
                        </div>
                      ))}
                      <p className="sr-only">Loading recommendations…</p>
                    </div>
                  )}

                  {rxStatus === "error" && (
                    <div className="mt-3 rounded-xl border border-rose-100 dark:border-rose-500/25 bg-rose-50/50 dark:bg-rose-500/10 p-3" role="alert">
                      <p className="text-[12px] text-rose-700 dark:text-rose-400">
                        ARIA couldn&apos;t load recommendations: {rxError || "request failed."}
                      </p>
                      <button
                        onClick={() => void fetchRecommendations(true)}
                        className="mt-1.5 inline-flex items-center gap-1.5 text-[12px] font-semibold text-blue-700 dark:text-blue-400 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
                      >
                        <RefreshCw size={12} /> Retry
                      </button>
                    </div>
                  )}

                  {rxStatus === "refused" && (
                    <div className="mt-3 rounded-xl border border-border/80 p-3">
                      <p className="text-[12.5px] text-secondary-foreground leading-relaxed">
                        {strOf(rxRefusal?.headline || "Nothing in your data cleared the evidence bar for an intervention.")}
                      </p>
                      {asList(rxRefusal?.alternatives).length > 0 && (
                        <div className="mt-2 flex items-center gap-1.5 flex-wrap">
                          {asList(rxRefusal.alternatives).map((a: any, i: number) => (
                            <button
                              key={i}
                              onClick={() => {
                                const am = strOf(a?.metric);
                                if (am) void fetchRecommendations(true, { metric: am, driver: strOf(a?.driver) });
                              }}
                              className="text-[11.5px] font-medium text-blue-700 dark:text-blue-400 border border-blue-200 dark:border-blue-500/25 rounded-full px-2.5 py-1 hover:bg-blue-50 dark:hover:bg-blue-500/10"
                              aria-label={`Re-run with ${strOf(a?.label || `${a?.metric} ${a?.driver}`)}`}
                            >
                              {strOf(a?.label || `${humanize(a?.driver)} for ${humanize(a?.metric)}`)}
                            </button>
                          ))}
                        </div>
                      )}
                      <p className="mt-2 text-[11px] text-muted-foreground">
                        ARIA would rather refuse than invent an action.
                      </p>
                    </div>
                  )}

                  {rxStatus === "ready" && (
                    <div className="mt-3 space-y-2.5">
                      {rxItems.map((p: any, i: number) => {
                        const id = strOf(p?.id || `rx-${i}`);
                        const disp = rxDispOf(p);
                        const det = rxDetailsOf(p);
                        const title = strOf(disp.title || p?.title || p?.name, `Recommendation ${i + 1}`);
                        const problem = strOf(disp.problem_text || p?.summary_plain || p?.story || p?.summary);
                        const actionText = strOf(disp.action_text);
                        const outcomeText = strOf(disp.outcome_text);
                        const confTxt = strOf(disp.confidence_label || p?.confidence_label || p?.confidence) || "—";
                        const risk = strOf(p?.risk);
                        const whyBullets = asList(disp.why_text?.length ? disp.why_text : null)
                          .concat(strOf(disp.why_summary || p?.why_summary) ? [strOf(disp.why_summary || p?.why_summary)] : [])
                          .filter(Boolean).slice(0, 3);
                        const numbers = rxNumbersOf(p);
                        const lims = rxLimsOf(p);
                        const honesty = strOf(p?.impact_note || p?.impactNote);
                        const evPass = det.evidence_pass;
                        const impact = p?.impact && typeof p.impact === "object" ? p.impact : {};
                        const bl = numOf(impact.baseline);
                        const sc = numOf(impact.scenario);
                        const dl = numOf(impact.delta);
                        const relPct = numOf(impact.rel_pct);
                        const n = numOf(det?.numbers?.n || det?.numbers?.n_points);
                        const saved = rxSavedIds.includes(id);
                        const savingThis = rxSavingId === id;
                        const showChart = bl !== null && sc !== null;
                        const s = (confTxt || "").toLowerCase();
                        const confFilled = s === "high" ? 3 : s === "moderate" || s === "medium" ? 2 : 1;
                        const fmt = (v: number) => {
                          if (Math.abs(v) >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
                          if (Math.abs(v) >= 1_000) return `${(v / 1_000).toFixed(0)}K`;
                          return v.toFixed(0);
                        };
                        return (
                          <div key={id} className="rounded-xl border border-border/80 p-4 hover:border-blue-200 dark:hover:border-blue-500/25 transition-colors">
                            {/* Header: Title + Confidence dots + Risk badge */}
                            <div className="flex items-start justify-between gap-2 flex-wrap">
                              <p className="text-[13px] font-semibold text-foreground leading-snug flex-1 min-w-[200px]">
                                {title}
                              </p>
                              <span className="flex items-center gap-2 flex-shrink-0">
                                <span className="inline-flex items-center gap-0.5" title={`Confidence: ${confTxt}`}>
                                  {[0, 1, 2].map((dot) => (
                                    <span key={dot} className={`inline-block w-1.5 h-1.5 rounded-full ${dot < confFilled ? "bg-blue-600" : "bg-border"}`} />
                                  ))}
                                </span>
                                {risk && (() => {
                                  const rl = risk.toLowerCase();
                                  if (rl === "high") return <span className="inline-flex items-center gap-1 text-[10px] text-red-700 dark:text-red-400 bg-red-50 dark:bg-red-500/10 border border-red-200 dark:border-red-500/25 rounded-full px-1.5 py-0.5"><AlertTriangle size={10} /> High risk</span>;
                                  if (rl === "moderate" || rl === "medium") return <span className="inline-flex items-center gap-1 text-[10px] text-amber-700 dark:text-amber-400 bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/25 rounded-full px-1.5 py-0.5"><ShieldCheck size={10} /> Moderate risk</span>;
                                  return <span className="inline-flex items-center gap-1 text-[10px] text-green-700 dark:text-green-400 bg-green-50 dark:bg-green-500/10 border border-green-200 dark:border-green-500/25 rounded-full px-1.5 py-0.5"><ShieldCheck size={10} /> Low risk</span>;
                                })()}
                              </span>
                            </div>

                            {/* Problem */}
                            {problem && (
                              <p className="mt-2 text-[12.5px] text-secondary-foreground leading-relaxed">{problem}</p>
                            )}

                            {/* Why — always visible, no drill-down needed */}
                            {whyBullets.length > 0 && (
                              <div className="mt-2">
                                {whyBullets.map((b, j) => (
                                  <p key={j} className="text-[12px] text-secondary-foreground leading-relaxed">• {b}</p>
                                ))}
                              </div>
                            )}

                            {/* Mini chart: baseline vs projected */}
                            {showChart && bl !== null && sc !== null && (
                              <div className="h-16 w-full mt-2">
                                <ResponsiveContainer width="100%" height="100%">
                                  <BarChart data={[{ name: "Current", value: bl, fill: "#94a3b8" }, { name: "Projected", value: sc, fill: (dl ?? 0) >= 0 ? "#22c55e" : "#ef4444" }]} layout="vertical" margin={{ left: 0, right: 0, top: 0, bottom: 0 }} barSize={14}>
                                    <XAxis type="number" hide domain={[0, Math.max(bl, sc) * 1.15 || 1]} />
                                    <YAxis type="category" dataKey="name" width={65} tick={{ fontSize: 10, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} />
                                    <Tooltip formatter={(v: number) => [fmt(v), "Value"]} contentStyle={{ fontSize: 11, borderRadius: 8, border: "1px solid var(--border)", background: "var(--card)", color: "var(--foreground)" }} cursor={false} />
                                    <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                                      {[{ fill: "#94a3b8" }, { fill: (dl ?? 0) >= 0 ? "#22c55e" : "#ef4444" }].map((entry, idx) => (
                                        <Cell key={idx} fill={entry.fill} />
                                      ))}
                                    </Bar>
                                  </BarChart>
                                </ResponsiveContainer>
                              </div>
                            )}

                            {/* Action + Outcome — prominent */}
                            {actionText && (
                              <div className="mt-3 rounded-lg bg-blue-50/60 dark:bg-blue-500/10 border border-blue-100 dark:border-blue-500/20 px-3 py-2.5">
                                <p className="text-[12.5px] font-semibold text-foreground">{actionText}</p>
                                {outcomeText && (
                                  <p className="mt-1 text-[12px] text-secondary-foreground">
                                    {dl !== null && relPct !== null ? (
                                      <span className="font-medium">
                                        {dl >= 0 ? <TrendingUp size={12} className="inline mr-1 text-green-600 dark:text-green-400" /> : <TrendingDown size={12} className="inline mr-1 text-red-600 dark:text-red-400" />}
                                        {dl >= 0 ? "+" : ""}{relPct.toFixed(1)}% {dl >= 0 ? "lift" : "drop"} expected
                                      </span>
                                    ) : outcomeText}
                                  </p>
                                )}
                              </div>
                            )}

                            {/* Inline evidence summary */}
                            <div className="mt-2 flex items-center gap-3 flex-wrap text-[11px] text-muted-foreground">
                              {n !== null && <span className="flex items-center gap-1"><BarChart3 size={11} /> {Math.round(n).toLocaleString()} data points</span>}
                              {evPass === true && <span className="flex items-center gap-1 text-green-600 dark:text-green-400"><ShieldCheck size={11} /> Evidence verified</span>}
                              {evPass === false && <span className="flex items-center gap-1 text-amber-600 dark:text-amber-400"><AlertTriangle size={11} /> Limited evidence</span>}
                              {lims.length > 0 && (
                                <span className="text-muted-foreground" title={lims.join("; ")}>
                                  {lims.length} limitation{lims.length === 1 ? "" : "s"}
                                </span>
                              )}
                            </div>

                            {/* Honest caveat */}
                            {honesty && (
                              <div className="mt-2 flex items-start gap-1.5 text-[11px] text-amber-700 dark:text-amber-400 bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/25 rounded-lg px-2.5 py-1.5">
                                <Info size={11} className="text-amber-500 dark:text-amber-400 flex-shrink-0 mt-px" />
                                <span>{honesty}</span>
                              </div>
                            )}

                            {/* Actions */}
                            <div className="mt-2.5 flex items-center gap-2">
                              <button
                                onClick={() => void saveRecommendation(p, i)}
                                disabled={savingThis || saved}
                                className={btnGhost}
                                aria-label={saved ? `Saved: ${title}` : `Save recommendation: ${title}`}
                              >
                                <Bookmark size={13} /> {saved ? "Saved ✓" : savingThis ? "Saving…" : "Save"}
                              </button>
                            </div>
                          </div>
                        );
                      })}
                      {rxSaveMsg && (
                        <p className="text-[11px] text-secondary-foreground" role="status">{rxSaveMsg}</p>
                      )}
                    </div>
                  )}
                </Card>
              </StaggerItem>
            )}

            <StaggerItem>
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <div id="intel-breakdown-table" className="lg:col-span-2">
                  <Card className="p-4 md:p-5 overflow-hidden">
                    <CardHead
                      title={breakdownTitle}
                      sub={breakdown ? `Current vs previous by ${titleCase(breakdown)}` : "Current vs previous overall"}
                      right={
                        <button
                          onClick={exportCsv}
                          disabled={!rows.length}
                          className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11.5px] font-semibold text-blue-700 dark:text-blue-400 border border-blue-200 dark:border-blue-500/25 hover:bg-blue-50 dark:hover:bg-blue-500/10 disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                          aria-label="Export breakdown as CSV"
                          title={rows.length ? "Download the live breakdown rows as CSV" : "No rows to export"}
                        >
                          <Download size={13} /> Export CSV
                        </button>
                      }
                    />
                    {breakdown && (
                      <nav aria-label="Segment drill-down" className="mt-2 flex items-center gap-1 text-[11.5px] text-muted-foreground flex-wrap">
                        <button
                          onClick={clearDrill}
                          className="hover:text-blue-700 dark:hover:text-blue-400 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
                          aria-label={activeDrill ? `Clear segment filter, show all ${titleCase(breakdown)} segments` : `Showing all ${titleCase(breakdown)} segments`}
                        >
                          All {titleCase(breakdown)} segments
                        </button>
                        {activeDrill && (
                          <>
                            <ChevronRight size={12} className="text-slate-300 dark:text-muted-foreground" aria-hidden />
                            <span className="font-semibold text-foreground">{activeDrill.value}</span>
                            <button
                              onClick={clearDrill}
                              className="inline-flex items-center text-muted-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
                              aria-label={`Clear segment filter ${activeDrill.value}`}
                            >
                              <X size={12} />
                            </button>
                          </>
                        )}
                      </nav>
                    )}
                    {rows.length > 0 ? (
                      <div className="mt-2 overflow-x-auto -mx-4 px-4 md:-mx-5 md:px-5">
                        <table className="w-full text-[12px] min-w-[560px]">
                          <thead>
                            <tr className="text-left text-muted-foreground border-b border-border">
                              <th scope="col" className="py-2 pr-3 font-medium">{titleCase(breakdown) || "Category"}</th>
                              <th scope="col" className="py-2 pr-3 font-medium text-right">Current Period</th>
                              <th scope="col" className="py-2 pr-3 font-medium text-right">Previous Period</th>
                              <th scope="col" className="py-2 pr-3 font-medium text-right">Change</th>
                              <th scope="col" className="py-2 pr-3 font-medium text-right">% Change</th>
                              <th scope="col" className="py-2 font-medium text-right">Trend</th>
                            </tr>
                          </thead>
                          <tbody>
                            {rows.map((r: any, i: number) => {
                              const cur = numOf(r?.current ?? r?.current_value ?? r?.value);
                              const prev = numOf(r?.previous ?? r?.previous_value ?? r?.prev);
                              const chg = numOf(r?.change ?? r?.delta);
                              const pct = numOf(r?.change_pct ?? r?.pct_change ?? r?.pct ?? r?.percent_change);
                              const spark = asList(r?.trend || r?.spark || r?.history).map((v) => numOf(typeof v === "number" ? v : v?.value ?? v?.v)).filter((v) => v !== null) as number[];
                              const up = (chg ?? 0) >= 0;
                              const cat = strOf(r?.category || r?.label || r?.name || `Row ${i + 1}`);
                              const isDrilled = !!activeDrill && activeDrill.value === cat;
                              return (
                                <tr key={i} className={`border-b border-border last:border-0 ${isDrilled ? "bg-blue-50/50 dark:bg-blue-500/10" : ""}`}>
                                  <td className="py-2 pr-3 font-medium text-foreground">
                                    {drillable ? (
                                      <button
                                        onClick={() => drillSegment(cat)}
                                        className="inline-flex items-center gap-2 rounded hover:text-blue-700 dark:hover:text-blue-400 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-1 text-left"
                                        aria-label={`Drill into ${cat}: filter ${titleCase(breakdown)} to this segment`}
                                        title={`Drill into ${cat}`}
                                      >
                                        <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: colorForName(cat) }} />
                                        {cat}
                                      </button>
                                    ) : (
                                      <span className="inline-flex items-center gap-2">
                                        <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: colorForName(cat) }} />
                                        {cat}
                                      </span>
                                    )}
                                  </td>
                                  <td className="py-2 pr-3 text-right tabular-nums">{cur === null ? "—" : fmtMetric(cur, metric)}</td>
                                  <td className="py-2 pr-3 text-right tabular-nums text-muted-foreground">{prev === null ? "—" : fmtMetric(prev, metric)}</td>
                                  <td className={`py-2 pr-3 text-right tabular-nums font-medium ${chg === null ? "text-muted-foreground" : up ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                                    {chg === null ? "—" : <span className="inline-flex items-center gap-0.5 justify-end">{up ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}{fmtMetric(Math.abs(chg), metric)}</span>}
                                  </td>
                                  <td className={`py-2 pr-3 text-right tabular-nums font-medium ${pct === null ? "text-muted-foreground" : pct >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                                    {pct === null ? "—" : fmtPctBiz(pct)}
                                  </td>
                                  <td className="py-1 text-right">
                                    <Sparkline values={spark} color={colorForName(cat)} />
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <div className="mt-3">
                        <p className="text-xs text-muted-foreground">
                          {activeDrill
                            ? `No rows matched ${titleCase(breakdown)} = ${activeDrill.value} for this selection — the filter returned an empty result, not an error.`
                            : "No breakdown rows were returned for this selection."}
                        </p>
                        {activeDrill && (
                          <button
                            onClick={clearDrill}
                            className="mt-2 text-[11.5px] font-semibold text-blue-700 dark:text-blue-400 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
                          >
                            Clear segment filter
                          </button>
                        )}
                      </div>
                    )}
                  </Card>
                </div>

                {/* Cost of doing nothing — engine run-rate. */}
                {(() => {
                  if (!inaction?.headline) return null;
                  const burn = numOf(inaction?.monthly_burn);
                  return (
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                      <Card className="p-4 md:p-5 lg:col-span-2">
                        <h3 className="text-[14px] font-semibold text-foreground tracking-tight">Cost of doing nothing</h3>
                        <p className="mt-1.5 text-[12.5px] text-secondary-foreground leading-relaxed">{strOf(inaction.headline)}</p>
                        {burn !== null && burn > 0 && (
                          <p className="mt-1 text-[18px] font-bold text-rose-700 dark:text-rose-400 tabular-nums">~{fmtMetric(burn, metric)}<span className="text-[12px] font-medium text-muted-foreground">/month at stake</span></p>
                        )}
                        {strOf(inaction?.note) && <p className="mt-1 text-[11px] text-muted-foreground">{strOf(inaction.note)}</p>}
                      </Card>
                    </div>
                  );
                })()}

                <Card className="p-4 md:p-5">
                  <h3 className="text-[14px] font-semibold text-foreground tracking-tight flex items-center gap-1.5">
                    <Send size={13} className="text-blue-700 dark:text-blue-400" /> Next Steps
                  </h3>
                  <p className="text-[11px] text-muted-foreground">Take the next step with this analysis.</p>
                  <div className="mt-2 divide-y divide-border">
                    {nextSteps.map(({ label, sub, tab, Icon }) => (
                      <button key={label} onClick={() => onGoto(tab)} className="w-full text-left py-2.5 flex items-center gap-2.5 hover:bg-blue-50/50 dark:hover:bg-blue-500/10 hover:shadow-sm hover:-translate-y-px rounded-lg px-1 -mx-1 transition-all group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500">
                        <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-blue-50 dark:bg-blue-500/10 text-blue-700 dark:text-blue-400 flex-shrink-0"><Icon size={13} /></span>
                        <span className="min-w-0 flex-1">
                          <span className="block text-[12.5px] font-medium text-foreground group-hover:text-blue-700 dark:group-hover:text-blue-400">{label}</span>
                          <span className="block text-[11px] text-muted-foreground">{sub}</span>
                        </span>
                        <ChevronRight size={13} className="text-slate-300 dark:text-muted-foreground group-hover:text-blue-500 dark:group-hover:text-blue-400 flex-shrink-0" />
                      </button>
                    ))}
                    <button onClick={() => onAsk(`Explain this analysis: ${humanize(metric)} by ${humanize(breakdown)}`)} className="w-full text-left py-2.5 flex items-center gap-2.5 hover:bg-blue-50/50 dark:hover:bg-blue-500/10 hover:shadow-sm hover:-translate-y-px rounded-lg px-1 -mx-1 transition-all group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500">
                      <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-violet-50 dark:bg-violet-500/10 text-violet-700 dark:text-violet-400 flex-shrink-0"><Sparkles size={13} /></span>
                      <span className="min-w-0 flex-1">
                        <span className="block text-[12.5px] font-medium text-foreground group-hover:text-blue-700 dark:group-hover:text-blue-400">Ask ARIA</span>
                        <span className="block text-[11px] text-muted-foreground">Get a deeper explanation</span>
                      </span>
                      <ChevronRight size={13} className="text-slate-300 dark:text-muted-foreground group-hover:text-blue-500 dark:group-hover:text-blue-400 flex-shrink-0" />
                    </button>
                    <button onClick={() => setShowCompare(true)} className="w-full text-left py-2.5 flex items-center gap-2.5 hover:bg-blue-50/50 dark:hover:bg-blue-500/10 hover:shadow-sm hover:-translate-y-px rounded-lg px-1 -mx-1 transition-all group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500" aria-label="Compare saved analyses side-by-side">
                      <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-blue-50 dark:bg-blue-500/10 text-blue-700 dark:text-blue-400 flex-shrink-0"><Compass size={13} /></span>
                      <span className="min-w-0 flex-1">
                        <span className="block text-[12.5px] font-medium text-foreground group-hover:text-blue-700 dark:group-hover:text-blue-400">Compare saved analyses</span>
                        <span className="block text-[11px] text-muted-foreground">Side-by-side of two saved results</span>
                      </span>
                      <ChevronRight size={13} className="text-slate-300 dark:text-muted-foreground group-hover:text-blue-500 dark:group-hover:text-blue-400 flex-shrink-0" />
                    </button>
                  </div>
                </Card>
              </div>
            </StaggerItem>
          </StaggerGroup>
        )}

        {/* Home state: Start an analysis + Quick Actions + Recent/Saved */}
        {!running && !error && !result && (
          <div className="space-y-4">
            <Card className="p-4 md:p-5">
              <CardHead title="Start an analysis" sub="Ask anything — ARIA will run it against your connected data." />
              <div className="mt-3 flex items-center gap-2 rounded-xl border border-border bg-secondary/60 px-3 py-2 focus-within:border-blue-400 transition-colors">
                <Search size={14} className="text-muted-foreground flex-shrink-0" />
                <input value={homeQuery} onChange={(e) => setHomeQuery(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") submitHomeQuery(); }} placeholder="What would you like to investigate?" className="flex-1 bg-transparent outline-none text-[13px] text-foreground placeholder:text-muted-foreground" aria-label="What would you like to investigate" />
                <button onClick={submitHomeQuery} className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-blue-700 text-white hover:bg-blue-800 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500" aria-label="Start analysis"><Play size={14} /></button>
              </div>
              <div className="mt-2.5 flex items-center gap-1.5 flex-wrap" aria-label="Suggested investigations">
                {HOME_SUGGESTIONS.map((s) => (
                  <button key={s} onClick={() => { setHomeQuery(s); onAsk(s); runAnalysis(); }} className="text-[11px] text-muted-foreground border border-border rounded-full px-2.5 py-1 hover:border-blue-300 dark:hover:border-blue-500/25 hover:text-blue-700 dark:hover:text-blue-400 hover:shadow-sm hover:-translate-y-px bg-card transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500">{s}</button>
                ))}
              </div>
            </Card>
            <Card className="p-4 md:p-5">
              <CardHead title="Quick Analysis Actions" sub="Start from a common investigation — ARIA fills the builder and runs it." />
              <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-2">
                {QUICK_ACTIONS.map(({ id, label, sub, Icon }) => (
                  <button key={id} onClick={() => quickRun(id)} className="text-left rounded-xl border border-border hover:border-blue-300 dark:hover:border-blue-500/25 hover:shadow-md hover:-translate-y-0.5 p-3 transition-all group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500" aria-label={label}>
                    <span className="inline-flex items-center justify-center w-7 h-7 rounded-lg bg-blue-50 dark:bg-blue-500/10 text-blue-700 dark:text-blue-400 mb-1.5"><Icon size={14} /></span>
                    <span className="block text-[12.5px] font-medium text-foreground group-hover:text-blue-700 dark:group-hover:text-blue-400">{label}</span>
                    <span className="block text-[11px] text-muted-foreground">{sub}</span>
                  </button>
                ))}
              </div>
            </Card>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card className="p-4 md:p-5">
                <CardHead title="Ask one question" sub="One recommendation, what backs it, what could go wrong." />
                <div className="mt-2.5 flex items-center gap-2 rounded-xl border border-border bg-secondary/60 px-3 py-2 focus-within:border-blue-400 transition-colors">
                  <Search size={14} className="text-muted-foreground flex-shrink-0" />
                  <input value={askText} onChange={(e) => setAskText(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") void askOne(); }} placeholder="e.g. Should I raise prices?" className="flex-1 bg-transparent outline-none text-[13px] text-foreground placeholder:text-muted-foreground" aria-label="Ask one business question" />
                  <button onClick={() => void askOne()} disabled={askState === "loading" || !askText.trim()} className="inline-flex items-center justify-center h-8 px-3 rounded-lg bg-blue-700 text-white text-[12px] font-semibold hover:bg-blue-800 disabled:opacity-40 transition-colors" aria-label="Get answer"><Play size={13} /></button>
                </div>
                {askState === "loading" && <p className="mt-2 text-[12px] text-muted-foreground">Checking your data…</p>}
                {askState === "error" && <p className="mt-2 text-[12px] text-rose-600 dark:text-rose-400" role="alert">{askError}</p>}
                {askState === "ready" && askAnswer && (
                  <div className="mt-2.5 rounded-xl border border-border bg-secondary/60 px-3 py-2.5">
                    {askAnswer.top ? (
                      <>
                        <p className="text-[13px] font-semibold text-foreground">{strOf(askAnswer.top?.display?.action_text || askAnswer.top?.title)}</p>
                        {strOf(askAnswer.top?.display?.outcome_text) && <p className="text-[13px] font-bold text-blue-800 dark:text-blue-300 tabular-nums mt-0.5">{strOf(askAnswer.top.display.outcome_text)}</p>}
                        {strOf(askAnswer.top?.display?.why_summary) && <p className="mt-1 text-[12px] text-secondary-foreground leading-relaxed">{strOf(askAnswer.top.display.why_summary)}</p>}
                        {strOf(askAnswer.inaction?.headline) && <p className="mt-1 text-[11.5px] text-muted-foreground">{strOf(askAnswer.inaction.headline)}</p>}
                      </>
                    ) : (
                      <p className="text-[12.5px] text-secondary-foreground leading-relaxed">{strOf(askAnswer.refusal?.reason || askAnswer.refusal?.headline || "Nothing in your data clears the bar for that yet — try a metric with more history.")}</p>
                    )}
                  </div>
                )}
              </Card>
              <Card className="p-4 md:p-5">
                <CardHead title="Alerts that matter" sub="Only when a metric really moves." />
                <div className="mt-2 divide-y divide-border">
                  {alertsState === "loading" && <p className="text-[12px] text-muted-foreground py-2">Watching your metrics…</p>}
                  {alertsState === "ready" && alerts.length === 0 && <p className="text-[12px] text-muted-foreground py-2">All quiet — no unusual moves lately.</p>}
                  {alerts.map((a: any, i: number) => (
                    <div key={i} className="py-2 flex items-start gap-2">
                      <span className={`mt-1 w-2 h-2 rounded-full flex-shrink-0 ${a?.severity === "high" ? "bg-rose-500" : "bg-amber-400"}`} />
                      <div className="min-w-0">
                        <p className="text-[12.5px] font-medium text-foreground leading-snug">{strOf(a?.headline)}</p>
                        <p className="text-[11px] text-muted-foreground">Unusual vs its usual wobble (±{strOf(a?.threshold_pct)}%)</p>
                      </div>
                    </div>
                  ))}
                </div>
              </Card>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card className="p-4 md:p-5">
                <CardHead title="Recent Analyses" sub="Your latest explorations." right={<span className="inline-flex items-center gap-2"><button onClick={() => setShowCompare(true)} className="text-[11px] font-semibold text-blue-700 dark:text-blue-400 border border-blue-200 dark:border-blue-500/25 rounded-full px-2.5 py-1 hover:bg-blue-50 dark:hover:bg-blue-500/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500" aria-label="Compare saved analyses">Compare saved</button><button onClick={() => setShowAllRecent((v) => !v)} className="text-[11px] font-semibold text-blue-700 dark:text-blue-400 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded">{showAllRecent ? "Show less" : "View all"}</button></span>} />
                <div className="mt-2 divide-y divide-border">
                  {recentError && <p className="text-[12px] text-rose-600 dark:text-rose-400 py-2" role="alert">{recentError} <button onClick={loadRecent} className="font-semibold text-blue-700 dark:text-blue-400 hover:underline ml-1">Retry</button></p>}
                  {!recentError && recent.length === 0 && <p className="text-[12px] text-muted-foreground py-3">No recent analyses yet — run your first analysis above.</p>}
                  {(showAllRecent ? recent : recent.slice(0, 4)).map((a: any, i: number) => (
                    <div key={strOf(a?.id || i)} className="py-2 flex items-center gap-2">
                      <BarChart3 size={13} className="text-blue-600 dark:text-blue-400 flex-shrink-0" />
                      <div className="min-w-0 flex-1">
                        <p className="text-[12.5px] font-medium text-foreground truncate">{strOf(a?.title || a?.name || `${humanize(a?.metric)} by ${humanize(a?.breakdown)}`, "Analysis")}</p>
                        <p className="text-[11px] text-muted-foreground">{relativeOpened(strOf(a?.updated_at || a?.created_at || ""))}</p>
                      </div>
                      <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-blue-50 dark:bg-blue-500/10 text-blue-700 dark:text-blue-400 border border-blue-100 dark:border-blue-500/20">{strOf(a?.type || "Explore")}</span>
                      <button onClick={() => loadSavedAnalysis(a)} className="text-[11px] font-semibold text-blue-700 dark:text-blue-400 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded" aria-label={`Reload analysis ${strOf(a?.title || "")}`}>View</button>
                      {strOf(a?.id) && <button onClick={() => deleteSavedAnalysis(strOf(a.id))} className="text-[11px] font-medium text-muted-foreground hover:text-rose-600 dark:hover:text-rose-400" aria-label={`Delete analysis ${strOf(a?.title || "")}`}>Delete</button>}
                    </div>
                  ))}
                </div>
              </Card>
              <Card className="p-4 md:p-5">
                <CardHead title="Saved Collections" sub="Analyses you kept for later." right={<button onClick={() => setShowAllSaved((v) => !v)} className="text-[11px] font-semibold text-blue-700 dark:text-blue-400 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded">{showAllSaved ? "Show less" : "View all"}</button>} />
                <div className="mt-2 divide-y divide-border">
                  {savedCols.length === 0 && <p className="text-[12px] text-muted-foreground py-3">Nothing saved yet — use Save Analysis on any result.</p>}
                  {(showAllSaved ? savedCols : savedCols.slice(0, 4)).map((a: any, i: number) => (
                    <div key={strOf(a?.id || i)} className="py-2 flex items-center gap-2">
                      <Bookmark size={13} className="text-violet-600 dark:text-violet-400 flex-shrink-0" />
                      <div className="min-w-0 flex-1">
                        <p className="text-[12.5px] font-medium text-foreground truncate">{strOf(a?.title || a?.name || a?.collection || "Saved analysis")}</p>
                        <p className="text-[11px] text-muted-foreground">{relativeOpened(strOf(a?.updated_at || a?.created_at || ""))}</p>
                      </div>
                      <button onClick={() => loadSavedAnalysis(a)} className="text-[11px] font-semibold text-blue-700 dark:text-blue-400 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded" aria-label={`Reload saved analysis ${strOf(a?.title || "")}`}>View</button>
                      {strOf(a?.id) && <button onClick={() => deleteSavedAnalysis(strOf(a.id))} className="text-[11px] font-medium text-muted-foreground hover:text-rose-600 dark:hover:text-rose-400" aria-label={`Delete saved analysis ${strOf(a?.title || "")}`}>Delete</button>}
                    </div>
                  ))}
                </div>
                <div className="mt-3 grid grid-cols-2 gap-2">
                  {(showAllSaved ? savedBuckets : savedBuckets.slice(0, 4)).map((b) => (
                    <button key={b.name} onClick={() => { const first = b.items[0]; if (first) loadSavedAnalysis(first); }} className="text-left rounded-xl border border-border p-3 hover:border-blue-300 dark:hover:border-blue-500/25 hover:shadow-md hover:-translate-y-0.5 transition-all flex items-center gap-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500" aria-label={`Open ${b.name} collection`}>
                      <Bookmark size={13} className="text-violet-600 dark:text-violet-400 flex-shrink-0" />
                      <span className="min-w-0 flex-1">
                        <span className="block text-[12.5px] font-semibold text-foreground truncate">{b.name}</span>
                        <span className="block text-[11px] text-muted-foreground">{b.count} analyses</span>
                      </span>
                      <ChevronRight size={13} className="text-slate-300 dark:text-muted-foreground" />
                    </button>
                  ))}
                </div>
              </Card>
            </div>
          </div>
        )}

        {/* Chart fullscreen: larger view reusing the same live data. */}
        <AnimatePresence>
          {expanded && trendPoints.length > 0 && (
            <div className="fixed inset-0 z-50" role="dialog" aria-modal="true" aria-label="Expanded trend chart">
              <motion.div
                className="absolute inset-0 bg-slate-900/50"
                onClick={() => setExpanded(false)}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
              />
              <div className="absolute inset-0 overflow-y-auto pointer-events-none">
                <motion.div
                  className="pointer-events-auto mx-auto mt-[5vh] mb-[5vh] w-[calc(100%-2rem)] max-w-5xl bg-card rounded-2xl shadow-2xl p-5"
                  initial={{ opacity: 0, y: 24, scale: 0.98 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 24, scale: 0.98 }}
                  transition={{ duration: 0.28, ease: "easeOut" }}
                >
                  <div className="flex items-start justify-between gap-3 flex-wrap">
                    <div>
                      <h3 className="text-[15px] font-semibold text-foreground tracking-tight">{trendTitle}</h3>
                      {trendSub && <p className="text-[11.5px] text-muted-foreground mt-0.5">{trendSub}</p>}
                    </div>
                    <div className="flex items-center gap-1.5">
                      <div className="flex items-center gap-1.5 text-[11px] font-medium border border-border rounded-lg px-2 py-1" aria-label="Chart type">
                        <BarChart3 size={12} className="text-blue-700 dark:text-blue-400" />
                        <select value={chartMode} onChange={(e) => setChartMode(e.target.value as "bar" | "line")} className="bg-transparent outline-none cursor-pointer text-foreground" aria-label="Chart type">
                          <option value="bar">Bar Chart</option>
                          <option value="line">Line Chart</option>
                        </select>
                      </div>
                      <button
                        onClick={() => setExpanded(false)}
                        className="inline-flex items-center justify-center w-8 h-8 rounded-lg hover:bg-secondary text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                        aria-label="Close expanded chart"
                      >
                        <X size={16} />
                      </button>
                    </div>
                  </div>
                  <div className="h-[60vh] min-h-[320px] mt-3">
                    <TrendChart points={trendPoints} mode={chartMode} axisMetric={metric || trendTitle} tipMetric={trendTitle} />
                  </div>
                </motion.div>
              </div>
            </div>
          )}
        </AnimatePresence>

        <EvidenceDrawer
          open={evidenceOpen}
          onClose={() => setEvidenceOpen(false)}
          result={result}
          context={evidenceCtx}
        />

        {/* Compare saved analyses: modal hosting CompareView (own data load). */}
        <AnimatePresence>
          {showCompare && (
            <div className="fixed inset-0 z-50" role="dialog" aria-modal="true" aria-label="Compare saved analyses">
              <motion.div
                className="absolute inset-0 bg-slate-900/50"
                onClick={() => setShowCompare(false)}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
              />
              <div className="absolute inset-0 overflow-y-auto pointer-events-none">
                <motion.div
                  className="pointer-events-auto mx-auto mt-[5vh] mb-[5vh] w-[calc(100%-2rem)] max-w-6xl bg-card rounded-2xl shadow-2xl p-5"
                  initial={{ opacity: 0, y: 24, scale: 0.98 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 24, scale: 0.98 }}
                  transition={{ duration: 0.28, ease: "easeOut" }}
                >
                  <CompareView onClose={() => setShowCompare(false)} />
                </motion.div>
              </div>
            </div>
          )}
        </AnimatePresence>
      </div>
    </MotionConfig>
  );
}
