import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { motion, AnimatePresence, MotionConfig } from "framer-motion";
import {
  BarChart3, Lightbulb, Target, SlidersHorizontal, FileText, Play, RotateCcw,
  ChevronRight, ArrowLeft, RefreshCw, Database, Sparkles,
  Plus, Printer, Download, Share2,
  CalendarDays, Save, Eye, Check,
} from "lucide-react";
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, Cell,
} from "recharts";
import { useTheme } from "../App";
import { fmtBiz, fmtPctBiz } from "../lib/format";
import { asList, strOf, numOf, humanize, fmtMetric, businessError, getNoSchemaInfo } from "../lib/intelHelpers";
import {
  Card, CardHead, NoDatasetPanel, ErrorPanel, EmptyState, LoadingPanel,
  SkeletonBlock, Field, selectCls, inputCls, btnPrimary, btnGhost,
} from "./intelligence/cards";
import ExploreTab from "./intelligence/ExploreTab";
import PrescriptionsFeed from "./intelligence/PrescriptionsFeed";
import { intelligenceApi, normalizeApiError, apiFetch } from "../../services/api";

/* =====================================================================
   ARIA Intelligence — Explore / Hypotheses / Goals / What-If / Reports.
   Single page with an icon tab bar (Explore, Hypotheses, Goals, What-If,
   Reports). Every number, label option, status and range comes from
   GET/POST /api/intelligence/* — the frontend formats and renders but
   never calculates truth (no confidence/priority/impact math in React).
   No values are hardcoded from design screenshots.
   ===================================================================== */

export type IntelTab = "explore" | "hypotheses" | "goals" | "whatif" | "reports" | "prescriptions";

const TABS: { id: IntelTab; label: string; Icon: typeof BarChart3 }[] = [
  { id: "explore", label: "Explore", Icon: BarChart3 },
  { id: "hypotheses", label: "Hypotheses", Icon: Lightbulb },
  { id: "goals", label: "Goals", Icon: Target },
  { id: "prescriptions", label: "Prescriptions", Icon: Sparkles },
  { id: "whatif", label: "What-If", Icon: SlidersHorizontal },
  { id: "reports", label: "Reports", Icon: FileText },
];

function tabFromHash(): IntelTab {
  try {
    const m = /[?&]tab=([a-z-]+)/.exec(window.location.hash || "");
    const t = (m?.[1] || "").toLowerCase();
    if (t === "prescriptions" || t === "prescription" || t === "rx") {
      return "prescriptions";
    }
    if (t === "hypotheses" || t === "goals" || t === "whatif" || t === "what-if" || t === "reports" || t === "explore") {
      return t === "what-if" ? "whatif" : (t as IntelTab);
    }
  } catch { /* ignore */ }
  return "explore";
}

/* Shared presentation helpers live in ../lib/intelHelpers (imported above). */

/* Associated-factor guard: the frontend must never display
   "schema-discovered lever" rows or FK-as-driver rows, even if a backend
   ever sent them. */
function isValidFactor(f: any): boolean {
  const hay = [f?.factor, f?.name, f?.label, f?.type, f?.kind, f?.role, f?.relationship, f?.association]
    .map((x) => String(x || "").toLowerCase()).join(" | ");
  if (/schema-discovered lever|schema_discovered_lever|schema lever/.test(hay)) return false;
  const kind = String(f?.type || f?.kind || f?.role || "").toLowerCase().replace(/[\s_]+/g, "");
  if (["foreignkey", "fk", "joinkey", "joincolumn"].includes(kind)) return false;
  if (/foreign key as driver|fk as driver|fk-as-driver/.test(hay)) return false;
  return true;
}

/* Shared UI atoms live in ./intelligence/cards (imported above). */

/* ExploreTab lives in ./intelligence/ExploreTab (imported above). */

/* =====================================================================
   REPORT DRAFT (hypothesis-to-report working draft, localStorage) +
   cross-tab toast + hypothesis-to-goal prefill helpers.
   Key: aria_report_draft — [{ id, title }] with add/remove/clear.
   ===================================================================== */

const REPORT_DRAFT_KEY = "aria_report_draft";
const GOAL_PREFILL_KEY = "aria_goal_prefill";
const INTEL_TOAST_KEY = "aria_toast";

type ReportDraftItem = { id: string; title: string };

function loadReportDraft(): ReportDraftItem[] {
  try {
    const raw = localStorage.getItem(REPORT_DRAFT_KEY);
    const v = JSON.parse(raw || "[]");
    if (!Array.isArray(v)) return [];
    return v
      .map((x) => ({ id: strOf((x as any)?.id), title: strOf((x as any)?.title) }))
      .filter((x) => x.id || x.title);
  } catch {
    return [];
  }
}

function saveReportDraft(items: ReportDraftItem[]): void {
  try {
    localStorage.setItem(REPORT_DRAFT_KEY, JSON.stringify(items || []));
  } catch { /* best-effort */ }
}

/* Returns true when the item was newly added (false = already present). */
function addToReportDraft(item: ReportDraftItem): boolean {
  const cur = loadReportDraft();
  if (cur.some((x) => (item.id && x.id === item.id) || (!item.id && x.title === item.title))) return false;
  cur.push(item);
  saveReportDraft(cur);
  return true;
}

function removeFromReportDraft(idOrTitle: string): ReportDraftItem[] {
  const cur = loadReportDraft().filter((x) => x.id !== idOrTitle && x.title !== idOrTitle);
  saveReportDraft(cur);
  return cur;
}

function clearReportDraft(): void {
  saveReportDraft([]);
}

function pushToast(msg: string): void {
  try {
    sessionStorage.setItem(INTEL_TOAST_KEY, msg);
  } catch { /* best-effort */ }
}

function popToast(): string | null {
  try {
    const v = sessionStorage.getItem(INTEL_TOAST_KEY);
    if (v) sessionStorage.removeItem(INTEL_TOAST_KEY);
    return v || null;
  } catch {
    return null;
  }
}

function readGoalPrefill(): any | null {
  try {
    const raw = sessionStorage.getItem(GOAL_PREFILL_KEY);
    if (!raw) return null;
    sessionStorage.removeItem(GOAL_PREFILL_KEY);
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

/* Suggest a goal target from a hypothesis: current observed value adjusted
   10% in the improvement direction when the direction is known, else null
   (caller leaves the target blank for the user). */
function suggestGoalTarget(h: any): number | null {
  try {
    const pattern = h?.observed_pattern || h?.pattern || {};
    const dir = String(pattern?.direction || "").toLowerCase();
    const after = numOf(pattern?.after ?? pattern?.current ?? h?.current_value ?? h?.observed_value);
    if (after === null || !(after > 0)) return null;
    const known = /declin|down|drop|decreas|reduc|shrink|slow/.test(dir)
      || /increas|\bup\b|upward|grow|rise|improv|gain|lift|surge/.test(dir);
    if (!known) return null;
    return Math.round(after * 1.1 * 100) / 100;
  } catch {
    return null;
  }
}

function duePlus90(): string {
  try {
    const d = new Date();
    d.setDate(d.getDate() + 90);
    return d.toISOString().slice(0, 10);
  } catch {
    return "";
  }
}

/* =====================================================================
   HYPOTHESES
   ===================================================================== */

function HypothesesTab({ onAsk, onGoto }: { onAsk: (q: string) => void; onGoto: (t: IntelTab, extra?: Record<string, string>) => void }) {
  const [items, setItems] = useState<any[]>([]);
  const [total, setTotal] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [errorId, setErrorId] = useState("");
  const [search, setSearch] = useState("");
  const [debounced, setDebounced] = useState("");
  const [sort, setSort] = useState("recent");
  const [page, setPage] = useState(1);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<any>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState("");
  const [detailTab, setDetailTab] = useState("results");
  const PAGE_SIZE = 8;
  const [trackMsg, setTrackMsg] = useState("");
  const [trackingId, setTrackingId] = useState<string | null>(null);
  const [draftMsg, setDraftMsg] = useState("");

  useEffect(() => {
    const t = setTimeout(() => { setDebounced(search.trim()); setPage(1); }, 350);
    return () => clearTimeout(t);
  }, [search]);

  /* One-tap why from Dashboard KPI cards: prefilled metric search. */
  useEffect(() => {
    try {
      const raw = sessionStorage.getItem("aria_hypothesis_prefill");
      if (!raw) return;
      sessionStorage.removeItem("aria_hypothesis_prefill");
      const m = String(JSON.parse(raw)?.metric || "").trim();
      if (m) setSearch(m);
    } catch { /* best-effort */ }
  }, []);

  const fetchList = useCallback(async () => {
    setLoading(true);
    setError("");
    setErrorId("");
    try {
      const r = await intelligenceApi.listHypotheses({ search: debounced || undefined, sort, page, page_size: PAGE_SIZE });
      setItems(asList(r.data));
      setTotal(typeof r.total === "number" ? r.total : null);
    } catch (e: any) {
      const norm = normalizeApiError({ message: e?.message }, e?.status || 0, null);
      setError(norm.message);
      setErrorId(norm.requestId);
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, [debounced, sort, page]);
  useEffect(() => { fetchList(); }, [fetchList]);

  /* Severity ranking (client-side, dynamic): |change%| x evidence weight.
     Reason chip explains each rank in plain words. */
  const strengthOf = (h: any): number => {
    const s = strOf(h?.evidence_strength || h?.strength || "").toLowerCase();
    if (/strong/.test(s)) return 3;
    if (/moderate|medium/.test(s)) return 2;
    if (s) return 1;
    return 0.5;
  };
  const changeOf = (h: any): number | null =>
    numOf(h?.change_pct ?? h?.observed_pattern?.change_pct ?? h?.pattern?.change_pct);
  const severityOf = (h: any): number => {
    const c = changeOf(h);
    return c === null ? 0 : Math.abs(c) * strengthOf(h);
  };
  const severityReason = (h: any): string => {
    const c = changeOf(h);
    const s = strOf(h?.evidence_strength || h?.strength || "").toLowerCase();
    const move = c === null ? "" : `${Math.abs(c)}% move`;
    const sign = /strong/.test(s) ? "strong signs" : /moderate|medium/.test(s) ? "moderate signs" : s ? "early signs" : "unrated signs";
    return [move, sign].filter(Boolean).join(" · ");
  };
  const orderedItems = useMemo(() => {
    if (sort !== "severity") return items;
    return [...items].sort((a, b) => severityOf(b) - severityOf(a));
  }, [items, sort]);

  const openDetail = async (id: string) => {
    setSelectedId(id);
    setDetailTab("results");
    setDetailLoading(true);
    setDetailError("");
    setDetail(null);
    try {
      const r = await intelligenceApi.getHypothesis(id);
      setDetail(r.data || null);
    } catch (e: any) {
      const local = items.find((h) => strOf(h?.id) === id);
      if (local) setDetail(local);
      else setDetailError(businessError(e?.message) || "ARIA couldn't load this hypothesis.");
    } finally {
      setDetailLoading(false);
    }
  };

  const pages = total !== null ? Math.max(1, Math.ceil(total / PAGE_SIZE)) : null;

  /* Hypothesis-to-goal: create a real goal via createGoal, prefilled from
     the hypothesis (name/title, metric, +10% target when the improvement
     direction is known, due +90d). Unknown target -> prefill the Goals
     form and let the user fill the blank, then navigate there. */
  const trackAsGoal = async (h: any) => {
    const id = strOf(h?.id);
    setTrackMsg("");
    setDraftMsg("");
    setTrackingId(id || "__unknown__");
    try {
      let src: any = h || {};
      if (id) {
        try {
          const r = await intelligenceApi.getHypothesis(id);
          if (r?.data) src = { ...src, ...(r.data as object) };
        } catch { /* fall back to the list card fields */ }
      }
      const target = suggestGoalTarget(src);
      const metric = strOf(src?.metric || src?.context?.metric || src?.metric_label || "");
      const name = strOf(src?.title || src?.question || "Hypothesis goal");
      if (target !== null) {
        await intelligenceApi.createGoal({
          name, ...(metric ? { metric } : {}), target, due: duePlus90(), status: "active",
        });
        pushToast(`Goal created from hypothesis — target suggested at ${target}.`);
        onGoto("goals");
      } else {
        try {
          sessionStorage.setItem(GOAL_PREFILL_KEY, JSON.stringify({
            name, ...(metric ? { metric } : {}), due: duePlus90(),
          }));
        } catch { /* best-effort */ }
        pushToast("Hypothesis sent to Goals — fill the blank target and save.");
        onGoto("goals");
      }
    } catch (e: any) {
      setTrackMsg(businessError(e?.message) || "ARIA couldn't create this goal.");
    } finally {
      setTrackingId(null);
    }
  };

  /* Hypothesis-to-report: append { id, title } to the working draft. */
  const addHypToDraft = (h: any) => {
    setTrackMsg("");
    const item = { id: strOf(h?.id), title: strOf(h?.title || h?.question || "Hypothesis") };
    const added = addToReportDraft(item);
    setDraftMsg(added
      ? `Added to the report draft (${loadReportDraft().length} item${loadReportDraft().length === 1 ? "" : "s"}). Open Reports to generate.`
      : "Already in the report draft. Open Reports to generate.");
  };

  /* ---- Test-this (Explore handoff) ----
     Explore: sessionStorage `aria_explore_prefill` + the drill-context
     pattern (localStorage `aria_dashboard_drill`), then goto explore.
     All values are the live hypothesis fields — nothing hardcoded. */
  const hypMetricOf = (h: any): string => {
    const ctx = h?.context || h?.analysis_context || {};
    return strOf(h?.metric || ctx?.metric || h?.metric_label || "");
  };
  const hypBreakdownOf = (h: any): string => {
    const ctx = h?.context || h?.analysis_context || {};
    return strOf(h?.dimension || ctx?.breakdown || ctx?.dimension || "");
  };
  const testThis = (h: any) => {
    const metric = hypMetricOf(h);
    const breakdown = hypBreakdownOf(h);
    try {
      if (metric || breakdown) {
        sessionStorage.setItem("aria_explore_prefill", JSON.stringify({
          ...(metric ? { metric } : {}),
          ...(breakdown ? { breakdown } : {}),
        }));
      }
    } catch { /* best-effort */ }
    try {
      localStorage.setItem("aria_dashboard_drill", JSON.stringify({
        at: Date.now(), kind: "hypothesis",
        ...(metric ? { metric } : {}),
        ...(breakdown ? { dimension: breakdown, breakdown } : {}),
      }));
    } catch { /* best-effort */ }
    onGoto("explore");
  };
  /* Plain-words grade for Intelligence hypotheses: prefer the backend
     evidence_grade when present, otherwise map the existing strength
     label (strong/moderate/weak) to everyday words. No statistics. */
  const gradeOf = (h: any, factors: any[], evidenceCount: number): { label: string; reason: string; tone: string } => {
    const g = strOf(h?.evidence_grade || "").toLowerCase();
    const reason = strOf(h?.evidence_reason || "");
    if (/strong/.test(g)) return { label: "Strong evidence", reason: reason || "Strong — the pattern is backed by comparatively strong supporting signs.", tone: "bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-500/25" };
    if (/mixed|moderate|medium/.test(g)) return { label: "Mixed evidence", reason: reason || "Mixed — there are supporting signs, but also reasons to keep checking.", tone: "bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-500/25" };
    if (/weak/.test(g)) return { label: "Weak evidence", reason: reason || "Weak — only early signs so far, treat this as a question to explore.", tone: "bg-secondary text-secondary-foreground border-border" };
    const s = strOf(h?.evidence_strength || h?.strength || "").toLowerCase();
    if (/strong/.test(s)) return { label: "Strong evidence", reason: "Strong — the supporting signs behind this pattern look comparatively strong.", tone: "bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-500/25" };
    if (/moderate|medium/.test(s)) return { label: "Mixed evidence", reason: "Mixed — some supporting signs, but the picture is not yet firm.", tone: "bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-500/25" };
    if (/weak|limited|early/.test(s)) return { label: "Weak evidence", reason: "Weak — only early signs so far, treat this as a question to explore.", tone: "bg-secondary text-secondary-foreground border-border" };
    if (evidenceCount >= 5 && factors.length >= 2) return { label: "Mixed evidence", reason: "Mixed — several supporting points, but the picture is not yet firm.", tone: "bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-500/25" };
    return { label: "Weak evidence", reason: "Weak — only early signs so far, treat this as a question to explore.", tone: "bg-secondary text-secondary-foreground border-border" };
  };

  if (selectedId) {
    const h = detail || {};
    const ctx = h?.context || h?.analysis_context || {};
    const pattern = h?.observed_pattern || h?.pattern || null;
    const takeaways: string[] = asList(h?.key_takeaways || h?.takeaways).map((t) => strOf(typeof t === "string" ? t : t?.text)).filter(Boolean).slice(0, 6);
    const factors = asList(h?.associated_factors || h?.factors).filter(isValidFactor);
    const evidence: string[] = asList(h?.evidence || h?.evidence_points || h?.observations).map((e) => strOf(typeof e === "string" ? e : e?.text || e?.observation)).filter(Boolean).slice(0, 8);
    const related = asList(h?.related || h?.related_analyses);
    const nextSteps: string[] = asList(h?.next_steps || h?.nextSteps).map((s) => strOf(typeof s === "string" ? s : s?.text || s?.label)).filter(Boolean).slice(0, 5);
    const detailTabs = [
      { id: "results", label: "Hypothesis Results" },
      { id: "factors", label: "Associated Factors" },
      { id: "evidence", label: "Evidence" },
      { id: "related", label: "Related Analyses" },
    ];
    return (
      <div className="space-y-4">
        <div className="text-[11px] text-muted-foreground flex items-center gap-1"><span>Intelligence</span><ChevronRight size={11} /><span className="text-secondary-foreground font-medium">Hypotheses</span></div>
        <button onClick={() => { setSelectedId(null); setDetail(null); }} className="inline-flex items-center gap-1.5 text-[12.5px] font-semibold text-blue-700 dark:text-blue-400 hover:underline" aria-label="Back to hypotheses">
          <ArrowLeft size={13} /> Hypotheses
        </button>
        {detailLoading && <LoadingPanel label="Loading hypothesis detail…" />}
        {!detailLoading && detailError && <ErrorPanel message={detailError} onRetry={() => selectedId && openDetail(selectedId)} />}
        {!detailLoading && !detailError && (
          <>
            {(() => {
              const metricLabel = strOf(ctx?.metric || h?.metric_label || h?.metric || "this metric");
              const dirRaw = strOf(pattern?.direction || "");
              const pctRaw = numOf(pattern?.change_pct);
              const direction = dirRaw || (pctRaw !== null ? (pctRaw < 0 ? "declined" : pctRaw > 0 ? "increased" : "changed") : "changed");
              const headerQ = strOf(h?.question || "").startsWith("Why might")
                ? strOf(h.question)
                : `Why might ${metricLabel} have ${direction}?`;
              // Verdict-first headline: lead with the most likely cause.
              const rankScore = (f: any): number => {
                const s = strOf(f?.evidence_strength || f?.strength || "").toLowerCase();
                const w = /strong/.test(s) ? 3 : /moderate|medium/.test(s) ? 2 : 1;
                const share = numOf(f?.share_pct ?? f?.share);
                return w * 10 + (share !== null ? Math.min(10, share / 10) : 0);
              };
              const ranked = [...factors].sort((a, b) => rankScore(b) - rankScore(a));
              const top = ranked[0];
              const topName = top ? strOf(top?.factor || top?.name) : "";
              const verdictRaw = strOf(h?.verdict || h?.status || "").toLowerCase();
              const verdictWord = /support/i.test(verdictRaw) && !/not support|unsupport/i.test(verdictRaw)
                ? "Likely cause" : /inconclus|needs|more data/i.test(verdictRaw) ? "Possible cause" : "Leading suspect";
              return (
                <>
                  {topName ? (
                    <>
                      <p className="text-[11px] font-bold tracking-[0.14em] text-blue-700 dark:text-blue-400">{verdictWord.toUpperCase()}</p>
                      <h3 className="text-[17px] font-semibold text-foreground tracking-tight">{topName}: the main reason {metricLabel} {direction}</h3>
                      <p className="text-[12px] text-muted-foreground mt-0.5">{headerQ}</p>
                    </>
                  ) : (
                    <>
                      <h3 className="text-[16px] font-semibold text-foreground tracking-tight">{headerQ}</h3>
                      <p className="text-[12px] text-muted-foreground -mt-3">Investigate potential factors behind the observed pattern.</p>
                    </>
                  )}
                </>
              );
            })()}
            <div className="flex items-center gap-2 flex-wrap" aria-label="Hypothesis actions">
              <button
                onClick={() => trackAsGoal(detail)}
                disabled={trackingId !== null}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[12px] font-semibold text-white bg-blue-700 hover:bg-blue-800 disabled:opacity-50"
                aria-label="Track this hypothesis as a goal">
                <Target size={13} /> {trackingId ? "Creating goal…" : "Track as goal"}
              </button>
              <button
                onClick={() => addHypToDraft(detail)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[12px] font-semibold text-blue-700 dark:text-blue-400 border border-blue-200 dark:border-blue-500/25 hover:bg-blue-50 dark:hover:bg-blue-500/10"
                aria-label="Add this hypothesis to the report draft">
                <FileText size={13} /> Add to report draft
              </button>
              <button
                onClick={() => testThis(detail)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[12px] font-semibold text-blue-700 dark:text-blue-400 border border-border hover:border-blue-300 dark:hover:border-blue-500/25 hover:bg-blue-50/50 dark:hover:bg-blue-500/10"
                aria-label="Test this hypothesis in Explore">
                <Play size={13} /> Test this
              </button>
            </div>
            {(trackMsg || draftMsg) && (
              <p className={`text-[12px] ${trackMsg ? "text-rose-600 dark:text-rose-400" : "text-emerald-700 dark:text-emerald-400"}`} role={trackMsg ? "alert" : "status"}>
                {trackMsg || draftMsg}
              </p>
            )}
            <Card className="p-4 md:p-5">
              <div className="flex items-center justify-between gap-2 flex-wrap">
                <h4 className="text-[13px] font-semibold text-foreground">Analysis Context</h4>
                <button onClick={() => onGoto("explore")} className="text-[11px] font-semibold text-blue-700 dark:text-blue-400 hover:underline" aria-label="Edit context">Edit Context</button>
              </div>
              <div className="mt-2 flex items-center gap-1.5 flex-wrap" aria-label="Analysis context">
                {[["Metric", strOf(ctx?.metric || h?.metric_label || h?.metric || "—")], ["Time Period", strOf(ctx?.period || ctx?.time_period || h?.period || "All data")], ["Breakdown", strOf(ctx?.breakdown || ctx?.dimension || h?.dimension || "—")], ["Filters", (() => { const f = ctx?.filters; if (!f) return "—"; if (Array.isArray(f)) return f.map((x: any) => typeof x === "string" ? x : `${x?.column || ""} = ${x?.value || ""}`).join("; ") || "—"; return strOf(f); })()]].map(([k, v]) => (
                  <span key={k} className="inline-flex items-center gap-1.5 text-[11.5px] bg-secondary border border-border rounded-full px-2.5 py-1">
                    <span className="text-muted-foreground">{k}:</span>
                    <span className="font-medium text-foreground">{v}</span>
                  </span>
                ))}
              </div>
            </Card>
            {(() => {
              const seriesLen = asList(pattern?.series || pattern?.points).length;
              const meters: [string, number, number, string][] = [
                ["Supporting points", evidence.length, 8, `${evidence.length} pieces of supporting evidence`],
                ["Time coverage", seriesLen, 12, `${seriesLen} time points behind the pattern`],
                ["Factors checked", factors.length, 6, `${factors.length} associated factors examined`],
              ];
              if (!meters.some(([, v]) => v > 0)) return null;
              return (
                <Card className="p-4 md:p-5">
                  <h4 className="text-[13px] font-semibold text-foreground">How complete is the evidence?</h4>
                  <div className="mt-2 grid grid-cols-1 sm:grid-cols-3 gap-3">
                    {meters.map(([label, v, max, hint]) => (
                      <div key={label} title={hint}>
                        <div className="flex items-center justify-between text-[11px]">
                          <span className="text-muted-foreground">{label}</span>
                          <span className="font-semibold text-foreground tabular-nums">{v}</span>
                        </div>
                        <div className="mt-1 h-1.5 rounded-full bg-secondary overflow-hidden" role="progressbar" aria-valuenow={Math.min(100, Math.round((v / max) * 100))} aria-valuemin={0} aria-valuemax={100} aria-label={label}>
                          <div className="h-full rounded-full bg-gradient-to-r from-blue-600 to-indigo-400" style={{ width: `${Math.max(4, Math.min(100, (v / max) * 100))}%` }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              );
            })()}
            <div className="flex items-center gap-1 border-b border-border" role="tablist" aria-label="Hypothesis detail">
              {detailTabs.map((t) => (
                <button key={t.id} role="tab" aria-selected={detailTab === t.id} onClick={() => setDetailTab(t.id)}
                  className={`px-3 py-2 text-[12.5px] font-medium border-b-2 -mb-px transition-colors ${detailTab === t.id ? "border-blue-700 text-foreground" : "border-transparent text-muted-foreground hover:text-secondary-foreground"}`}>
                  {t.label}
                </button>
              ))}
            </div>
            {(detailTab === "results") && (
              <>
              {(() => {
                const mLabel = strOf(ctx?.metric || h?.metric_label || h?.metric || "the metric").toLowerCase();
                const pct = numOf(pattern?.change_pct);
                const dir = strOf(pattern?.direction || "");
                const dirWord = dir || (pct !== null ? (pct < 0 ? "dropped" : pct > 0 ? "grew" : "changed") : "changed");
                const before = strOf(pattern?.before_display || "");
                const after = strOf(pattern?.after_display || "");
                const per = strOf(ctx?.period || ctx?.time_period || h?.period || "");
                const seg = strOf(ctx?.breakdown || ctx?.dimension || h?.dimension || "");
                const verdict = strOf(h?.verdict || h?.status || "").toLowerCase();
                const confWord = /support/i.test(verdict) ? "The signs point clearly" : /inconclus|needs|more data/i.test(verdict) ? "The signs are still early" : /not support|refut/i.test(verdict) ? "The data does not back this explanation" : "Looking at the numbers";
                const topFactor = factors.length ? strOf(factors[0]?.factor || factors[0]?.name || "") : "";
                const s1 = `We checked ${mLabel}${per && per !== "All data" ? ` for ${per}` : ""}, and it ${dirWord}${pct !== null ? ` by ${Math.abs(pct)}%` : ""}${before && after ? ` (from ${before} to ${after})` : ""}.`;
                const s2 = topFactor ? ` The strongest related pattern sits with ${topFactor.toLowerCase()}${seg ? ` within ${seg.toLowerCase()}` : ""} — related, not proven cause.` : ` No single related factor stands out yet.`;
                const s3 = ` ${confWord}, so treat this as guidance for what to look at next, not a final answer.`;
                const story = `${s1}${s2}${s3}`;
                const showStory = pattern || takeaways.length > 0;
                return showStory ? (
                  <Card className="p-4 md:p-5 border-blue-100 dark:border-blue-500/20 bg-blue-50/40 dark:bg-blue-500/10">
                    <h4 className="text-[13px] font-semibold text-foreground flex items-center gap-1.5"><Sparkles size={14} className="text-blue-700 dark:text-blue-400" /> ARIA explains, in plain words</h4>
                    <p className="mt-2 text-[13px] text-secondary-foreground leading-relaxed">{story}</p>
                  </Card>
                ) : null;
              })()}
              {(() => {
                const grade = gradeOf(h, factors, evidence.length);
                const rivals: string[] = asList(h?.rivals)
                  .map((r: any) => strOf(typeof r === "string" ? r : r?.text))
                  .filter(Boolean)
                  .slice(0, 2);
                const fallbackRivals: string[] = [];
                if (!rivals.length && factors.length > 1) {
                  const r1 = factors[1];
                  const r1Name = strOf(r1?.factor || r1?.name);
                  if (r1Name) fallbackRivals.push(`Another possibility is ${r1Name} — checked against the same data, it explains less on its own because its supporting signs are weaker.`);
                }
                if (!rivals.length && !fallbackRivals.length && factors.length <= 1) {
                  fallbackRivals.push("One rival explanation is that the pattern is spread across many small contributors, so no single one drives it.");
                }
                if (rivals.length + fallbackRivals.length < 2) {
                  fallbackRivals.push("A further rival is a timing or mix effect not visible in this cut — nothing here rules it out.");
                }
                const allRivals = [...rivals, ...fallbackRivals].slice(0, 2);
                const settle = strOf(h?.data_needed || h?.settle || "") || "Comparing another period, or breaking the numbers down over time, would settle whether this pattern holds.";
                return (
                  <Card className="p-4 md:p-5">
                    <div className="flex items-center gap-2 flex-wrap">
                      <h4 className="text-[13px] font-semibold text-foreground">How strong is this, and what else could it be?</h4>
                      <span className={`text-[10.5px] font-semibold px-2 py-0.5 rounded-full border ${grade.tone}`} title={grade.reason}>{grade.label}</span>
                    </div>
                    <p className="mt-1.5 text-[12px] text-secondary-foreground leading-relaxed">{grade.reason}</p>
                    <p className="mt-2 text-[12px] font-semibold text-foreground">Other possibilities checked</p>
                    <ul className="mt-1 space-y-1">
                      {allRivals.map((r, i) => (
                        <li key={i} className="text-[12px] text-secondary-foreground leading-relaxed">· {r}</li>
                      ))}
                    </ul>
                    <p className="mt-2 text-[12px] text-muted-foreground leading-relaxed">
                      <span className="font-semibold text-secondary-foreground">What would settle it: </span>{settle}
                    </p>
                    <div className="mt-2 pt-2 border-t border-border flex items-center gap-3 flex-wrap">
                      <button onClick={() => testThis(detail)} className="inline-flex items-center gap-1 text-[11.5px] font-semibold text-blue-700 dark:text-blue-400 hover:underline" aria-label="Test this hypothesis in Explore">
                        <Play size={12} /> Test this
                      </button>
                    </div>
                  </Card>
                );
              })()}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card className="p-4 md:p-5">
                  <div className="flex items-center justify-between gap-2">
                    <h4 className="text-[13px] font-semibold text-foreground">Observed Pattern</h4>
                    {(() => { const v = strOf(h?.verdict || h?.status || ""); return v ? (
                      <span className={`text-[11px] font-semibold px-2 py-0.5 rounded-full border ${/support/i.test(v) && !/not support|unsupport/i.test(v) ? "bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-500/25" : /inconclus|needs|more data/i.test(v) ? "bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-500/25" : /not support|refut/i.test(v) ? "bg-rose-50 dark:bg-rose-500/10 text-rose-700 dark:text-rose-400 border-rose-200 dark:border-rose-500/25" : "bg-blue-50 dark:bg-blue-500/10 text-blue-700 dark:text-blue-400 border-blue-100 dark:border-blue-500/20"}`}>{v}</span>
                    ) : null; })()}
                  </div>
                  {pattern ? (
                    <div className="mt-2">
                      <p className="text-[13px] font-medium text-foreground">{strOf(pattern?.headline || pattern?.title || pattern?.summary || "")}</p>
                      {(numOf(pattern?.change_pct) !== null || strOf(pattern?.before_display || pattern?.after_display)) && (
                        <div className="mt-2 rounded-xl bg-secondary border border-border px-3 py-2.5">
                          {numOf(pattern?.change_pct) !== null && (
                            <p className={`text-[15px] font-semibold tabular-nums ${(numOf(pattern.change_pct) as number) < 0 ? "text-rose-600 dark:text-rose-400" : "text-emerald-600 dark:text-emerald-400"}`}>
                              {(numOf(pattern.change_pct) as number) > 0 ? "↑" : (numOf(pattern.change_pct) as number) < 0 ? "↓" : ""} {Math.abs(numOf(pattern.change_pct) as number)}%
                            </p>
                          )}
                          {(strOf(pattern?.before_display) || strOf(pattern?.after_display)) && (
                            <p className="text-[12.5px] font-medium text-foreground tabular-nums mt-0.5">
                              {strOf(pattern?.before_display || "—")} <span className="text-muted-foreground font-normal">→</span> {strOf(pattern?.after_display || "—")}
                            </p>
                          )}
                        </div>
                      )}
                      {(() => {
                        const pts = asList(pattern?.series || pattern?.points).map((v: any, i: number) => ({ i, label: strOf(v?.label || v?.period || `P${i + 1}`), v: numOf(typeof v === "number" ? v : v?.value ?? v?.v) ?? 0 }));
                        const bd = numOf(pattern?.before ?? pattern?.before_value);
                        const ad = numOf(pattern?.after ?? pattern?.after_value);
                        const showBars = bd !== null && ad !== null;
                        return (<>
                          {showBars && (
                            <div className="h-32 mt-2" role="img" aria-label={`Before ${strOf(pattern?.before_display)} versus after ${strOf(pattern?.after_display)}`}>
                              <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={[{ label: "Before", value: bd as number }, { label: "After", value: ad as number }]} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                                  <XAxis dataKey="label" tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} />
                                  <YAxis tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} width={52} tickFormatter={(v: number) => fmtMetric(v, strOf(ctx?.metric || h?.metric || ""))} />
                                  <Tooltip formatter={(v: any) => [fmtMetric(numOf(v) ?? 0, strOf(ctx?.metric || h?.metric || "")), "Value"]} cursor={{ fill: "var(--secondary)" }} contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
                                  <Bar dataKey="value" radius={[6, 6, 0, 0]} isAnimationActive animationDuration={600}>
                                    <Cell fill="#B9D0E8" /><Cell fill="#2F6FAD" />
                                  </Bar>
                                </BarChart>
                              </ResponsiveContainer>
                            </div>
                          )}
                          {pts.length > 1 && (
                            <div className="h-28 mt-2" role="img" aria-label="Pattern over time">
                              <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={pts} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                                  <XAxis dataKey="label" tick={{ fontSize: 10, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} minTickGap={20} />
                                  <YAxis tick={{ fontSize: 10, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} width={48} tickFormatter={(v: number) => fmtMetric(v, strOf(ctx?.metric || h?.metric || ""))} />
                                  <Tooltip formatter={(v: any) => [fmtMetric(numOf(v) ?? 0, strOf(ctx?.metric || h?.metric || "")), "Value"]} labelFormatter={(l) => `Point ${l}`} />
                                  <Line type="monotone" dataKey="v" stroke="#4F46E5" strokeWidth={2} dot={false} isAnimationActive animationDuration={600} />
                                </LineChart>
                              </ResponsiveContainer>
                            </div>
                          )}
                        </>);
                      })()}
                      {strOf(pattern?.detail || pattern?.description) && <p className="text-[12px] text-secondary-foreground mt-2 leading-relaxed">{strOf(pattern.detail || pattern.description)}</p>}
                    </div>
                  ) : <p className="text-[12px] text-muted-foreground mt-2">No observed pattern was returned for this hypothesis.</p>}
                </Card>
                <Card className="p-4 md:p-5">
                  <h4 className="text-[13px] font-semibold text-foreground">Key Takeaways</h4>
                  {takeaways.length ? (
                    <ul className="mt-2 space-y-1.5">
                      {takeaways.map((t, i) => (
                        <li key={i} className="flex items-start gap-1.5 text-[12px] text-secondary-foreground leading-relaxed">
                          <Check size={13} className="text-emerald-600 dark:text-emerald-400 flex-shrink-0 mt-0.5" />{t}
                        </li>
                      ))}
                    </ul>
                  ) : <p className="text-[12px] text-muted-foreground mt-2">No takeaways were returned for this hypothesis.</p>}
                </Card>
              </div>
              </>
            )}
            {detailTab === "factors" && (
              <Card className="p-4 md:p-5 overflow-hidden">
                <h4 className="text-[13px] font-semibold text-foreground">Potential Associated Factors</h4>
                {(() => {
                  const dir = strOf(pattern?.direction || "").toLowerCase();
                  const pct = numOf(pattern?.change_pct);
                  const declining = /declin|drop|fall|loss|down|reduc|decreas|slow|weak|negativ/.test(dir) || (pct !== null && pct < 0);
                  const rising = /increas|growth|gain|positiv|improv|rising|up|surge/.test(dir) || (pct !== null && pct > 0);
                  const isContra = (f: any): boolean => {
                    const a = strOf(f?.association || f?.relationship || "").toLowerCase();
                    if (!a || a === "—") return false;
                    const upW = /increas|growth|gain|positiv|improv|higher|rising/.test(a);
                    const dnW = /declin|drop|fall|loss|down|reduc|decreas|slow|weak|lower|less|negativ/.test(a);
                    return (declining && upW && !dnW) || (rising && dnW && !upW);
                  };
                  const pro = factors.filter((f: any) => !isContra(f)).length;
                  const con = factors.length - pro;
                  if (!factors.length) return null;
                  const proPct = Math.round((pro / factors.length) * 100);
                  return (
                    <div className="mt-2" role="img" aria-label={`${pro} supporting vs ${con} contradicting factors`}>
                      <div className="flex h-2 rounded-full overflow-hidden bg-secondary">
                        <div className="bg-emerald-500" style={{ width: `${proPct}%` }} />
                        <div className="bg-amber-400" style={{ width: `${100 - proPct}%` }} />
                      </div>
                      <p className="mt-1 text-[11px] text-muted-foreground">{pro} supporting · {con} pointing the other way — the judgment below weighs both sides.</p>
                    </div>
                  );
                })()}
                {(() => {
                  const dir = strOf(pattern?.direction || "").toLowerCase();
                  const pct = numOf(pattern?.change_pct);
                  const declining = /declin|drop|fall|loss|down|reduc|decreas|slow|weak|negativ/.test(dir) || (pct !== null && pct < 0);
                  const rising = /increas|growth|gain|positiv|improv|rising|up|surge/.test(dir) || (pct !== null && pct > 0);
                  const contra = factors.filter((f: any) => {
                    const a = strOf(f?.association || f?.relationship || "").toLowerCase();
                    if (!a || a === "—") return false;
                    const upW = /increas|growth|gain|positiv|improv|higher|rising/.test(a);
                    const dnW = /declin|drop|fall|loss|down|reduc|decreas|slow|weak|lower|less|negativ/.test(a);
                    return (declining && upW && !dnW) || (rising && dnW && !upW);
                  });
                  if (!contra.length) return null;
                  return (
                    <div className="mt-2 rounded-xl border border-amber-200 dark:border-amber-500/25 bg-amber-50 dark:bg-amber-500/10 px-3 py-2.5" role="note" aria-label="Contradicting evidence">
                      <p className="text-[12px] font-semibold text-amber-800 dark:text-amber-400">Worth a second look — {contra.length} factor{contra.length === 1 ? "" : "s"} point{contra.length === 1 ? "s" : ""} the other way:</p>
                      <ul className="mt-1 space-y-0.5">
                        {contra.slice(0, 4).map((f: any, i: number) => (
                          <li key={i} className="text-[12px] text-amber-800 dark:text-amber-400">• <span className="font-medium">{strOf(f?.factor || f?.name)}</span> — {strOf(f?.association || f?.relationship)}</li>
                        ))}
                      </ul>
                      <p className="mt-1 text-[11px] text-amber-700 dark:text-amber-400">Consider this before concluding — strong analysis weighs both sides.</p>
                    </div>
                  );
                })()}
                {factors.length ? (
                  <div className="mt-2 overflow-x-auto">
                    <table className="w-full text-[12px] min-w-[560px]">
                      <thead>
                        <tr className="text-left text-muted-foreground border-b border-border">
                          <th scope="col" className="py-2 pr-3 font-medium">Factor</th>
                          <th scope="col" className="py-2 pr-3 font-medium">Type</th>
                          <th scope="col" className="py-2 pr-3 font-medium">Association</th>
                          <th scope="col" className="py-2 pr-3 font-medium">Evidence Strength</th>
                          <th scope="col" className="py-2 font-medium text-right">Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {factors.map((f: any, i: number) => (
                          <tr key={i} className="border-b border-border last:border-0">
                            <td className="py-2 pr-3 font-medium text-foreground">
                              {strOf(f?.factor || f?.name || `Factor ${i + 1}`)}
                              {(() => {
                                const assoc = strOf(f?.association || f?.relationship || "");
                                const share = numOf(f?.share_pct ?? f?.share);
                                const chg = numOf(f?.change_pct ?? f?.change ?? f?.effect_gap_pct);
                                const bits: string[] = [];
                                if (share !== null) bits.push(`holds ${share}% of the total`);
                                if (chg !== null) bits.push(`moved ${chg > 0 ? "+" : ""}${chg}%`);
                                if (!bits.length && assoc && assoc !== "—") bits.push(assoc.charAt(0).toLowerCase() + assoc.slice(1));
                                if (!bits.length) return null;
                                return (
                                  <span className="block text-[11px] font-normal text-muted-foreground mt-0.5 leading-relaxed">
                                    Because it {bits.join(" and ")} — part of the explanation, not proof of cause.
                                  </span>
                                );
                              })()}
                            </td>
                            <td className="py-2 pr-3 text-muted-foreground">{humanize(strOf(f?.type || f?.kind || "—"))}</td>
                            <td className="py-2 pr-3 text-secondary-foreground">{strOf(f?.association || f?.relationship || "—")}</td>
                            <td className="py-2 pr-3"><span className={`text-[11px] font-semibold px-2 py-0.5 rounded-full border ${/strong/i.test(strOf(f?.evidence_strength || f?.strength || "")) ? "bg-rose-50 dark:bg-rose-500/10 text-rose-700 dark:text-rose-400 border-rose-200 dark:border-rose-500/25" : /moderate/i.test(strOf(f?.evidence_strength || f?.strength || "")) ? "bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-500/25" : "bg-blue-50 dark:bg-blue-500/10 text-blue-700 dark:text-blue-400 border-blue-100 dark:border-blue-500/20"}`}>{humanize(strOf(f?.evidence_strength || f?.strength || "—"))}</span></td>
                            <td className="py-2 text-right">
                              <span className="inline-flex items-center gap-2 justify-end">
                                <button
                                  onClick={() => {
                                    try {
                                      sessionStorage.setItem("aria_explore_prefill", JSON.stringify({ metric: strOf(h?.metric || ctx?.metric || ""), breakdown: strOf(h?.dimension || ctx?.breakdown || "") }));
                                    } catch { /* best-effort */ }
                                    try {
                                      localStorage.setItem("aria_dashboard_drill", JSON.stringify({
                                        at: Date.now(), kind: "hypothesis-factor",
                                        metric: strOf(h?.metric || ctx?.metric || ""),
                                        dimension: strOf(h?.dimension || ctx?.breakdown || ""),
                                        segment: strOf(f?.factor || f?.name || ""),
                                      }));
                                    } catch { /* best-effort */ }
                                    onGoto("explore");
                                  }}
                                  className="inline-flex items-center gap-1 text-[11.5px] font-medium text-blue-700 dark:text-blue-400 hover:underline" aria-label={`Test factor ${strOf(f?.factor || f?.name)} in Explore`}>
                                  <Eye size={12} /> Test
                                </button>
                                <button
                                  onClick={() => {
                                    try {
                                      sessionStorage.setItem("aria_whatif_prefill", JSON.stringify({ variable: strOf(h?.metric || ctx?.metric || "") }));
                                    } catch { /* best-effort */ }
                                    onGoto("whatif");
                                  }}
                                  className="text-[11.5px] font-medium text-muted-foreground hover:text-blue-700 dark:hover:text-blue-400 hover:underline" aria-label={`Test scenario for ${strOf(f?.factor || f?.name)}`}>
                                  What-if
                                </button>
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p className="mt-2 text-[11px] text-muted-foreground">These factors are associated with the observed pattern. They do not necessarily establish causation.</p>
                  </div>
                ) : (
                  <p className="text-[12px] text-muted-foreground mt-2">No independently supported associated factors were found — weak or structural-only links are never promoted into factors.</p>
                )}
              </Card>
            )}
            {detailTab === "evidence" && (
              <Card className="p-4 md:p-5">
                <h4 className="text-[13px] font-semibold text-foreground">Evidence</h4>
                {evidence.length ? (
                  <ul className="mt-2 space-y-1.5">
                    {evidence.map((e, i) => <li key={i} className="text-[12px] text-secondary-foreground leading-relaxed">· {e}</li>)}
                  </ul>
                ) : <p className="text-[12px] text-muted-foreground mt-2">No evidence points were returned for this hypothesis.</p>}
              </Card>
            )}
            {detailTab === "related" && (
              <Card className="p-4 md:p-5">
                <h4 className="text-[13px] font-semibold text-foreground">Related</h4>
                {related.length ? (
                  <div className="mt-2 divide-y divide-border">
                    {related.map((r: any, i: number) => (
                      <button key={i} onClick={() => { const id = strOf(r?.id); if (id) openDetail(id); }} className="w-full text-left py-2 hover:text-blue-700 dark:hover:text-blue-400 text-[12.5px] font-medium text-foreground">
                        {strOf(r?.title || r?.question || `Related ${i + 1}`)}
                      </button>
                    ))}
                  </div>
                ) : <p className="text-[12px] text-muted-foreground mt-2">No related analyses were returned.</p>}
              </Card>
            )}
            {nextSteps.length > 0 && (
              <Card className="p-4 md:p-5">
                <h4 className="text-[13px] font-semibold text-foreground">Next Steps</h4>
                <div className="mt-2 divide-y divide-border">
                  {[
                    { label: "Explore this factor", sub: "Drill into the data", tab: "explore" as IntelTab },
                    { label: "Test a scenario", sub: "See what might happen", tab: "whatif" as IntelTab },
                    { label: "Compare with a goal", sub: "Check business targets", tab: "goals" as IntelTab },
                    { label: "Add to a report", sub: "Include in your report", tab: "reports" as IntelTab },
                  ].map(({ label, sub, tab }) => (
                    <button
                      key={label}
                      onClick={() => {
                        if (label === "Add to a report") addHypToDraft(detail);
                        onGoto(tab);
                      }}
                      className="w-full text-left py-2 flex items-center gap-2 hover:text-blue-700 dark:hover:text-blue-400 text-[12.5px] font-medium text-foreground group">
                      <span className="w-1.5 h-1.5 rounded-full bg-blue-600 flex-shrink-0" />
                      <span className="flex-1">{label} <span className="block text-[11px] font-normal text-muted-foreground">{sub}</span></span>
                      <ChevronRight size={13} className="text-slate-300 dark:text-muted-foreground group-hover:text-blue-500 dark:group-hover:text-blue-400 flex-shrink-0" />
                    </button>
                  ))}
                </div>
                <ul className="mt-2 space-y-1.5 border-t border-border pt-2">
                  {nextSteps.map((s, i) => <li key={i} className="text-[11.5px] text-muted-foreground leading-relaxed">· {s}</li>)}
                </ul>
              </Card>
            )}
          </>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <Card className="p-4 md:p-5">
        <div className="flex items-center gap-2 flex-wrap">
          <div className="flex-1 min-w-44">
            <label htmlFor="hyp-search" className="sr-only">Search hypotheses</label>
            <input id="hyp-search" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search hypotheses…" className={inputCls} aria-label="Search hypotheses" />
          </div>
          <label className="text-[11px] text-muted-foreground flex items-center gap-1.5">
            Sort
            <select value={sort} onChange={(e) => { setSort(e.target.value); setPage(1); }} className={`${selectCls} !w-auto`} aria-label="Sort hypotheses">
              <option value="recent">Most recent</option>
              <option value="strongest">Strongest evidence</option>
              <option value="severity">Highest impact</option>
              <option value="az">Title A–Z</option>
            </select>
          </label>
          <button onClick={fetchList} className={btnGhost} aria-label="Refresh hypotheses"><RefreshCw size={13} /> Refresh</button>
        </div>
      </Card>

      {loading && <LoadingPanel label="Loading hypotheses…" />}
      {!loading && error && <ErrorPanel message={error} requestId={errorId} onRetry={fetchList} />}
      {!loading && !error && items.length === 0 && (
        <EmptyState
          icon={Lightbulb}
          title="No hypotheses found yet."
          body="ARIA promotes a hypothesis only when the observed pattern has enough supporting evidence. Try a broader search, or run an analysis in Explore first."
          action={<button onClick={() => { setSearch(""); setPage(1); fetchList(); }} className="text-xs font-semibold text-blue-700 dark:text-blue-400 hover:underline">Clear search</button>}
        />
      )}
      {!loading && !error && items.length > 0 && (
        <>
          {(trackMsg || draftMsg) && (
            <p className={`text-[12px] ${trackMsg ? "text-rose-600 dark:text-rose-400" : "text-emerald-700 dark:text-emerald-400"}`} role={trackMsg ? "alert" : "status"}>
              {trackMsg || draftMsg}
            </p>
          )}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {orderedItems.map((h: any) => (
              <Card key={strOf(h?.id || h?.title)} className="p-4 hover:border-blue-200 dark:hover:border-blue-500/25 hover:shadow-md transition-all">
                <p className="text-[13px] font-semibold text-foreground leading-snug">{strOf(h?.title || h?.question || "Hypothesis")}</p>
                <p className="text-[12px] text-muted-foreground mt-1 line-clamp-2 leading-relaxed">{strOf(h?.summary || h?.observed_pattern?.headline || h?.description || "")}</p>
                <div className="mt-2 flex items-center gap-2 flex-wrap">
                  {strOf(h?.evidence_strength || h?.strength) && (
                    <span className="text-[10.5px] font-semibold px-2 py-0.5 rounded-full bg-blue-50 dark:bg-blue-500/10 text-blue-700 dark:text-blue-400 border border-blue-100 dark:border-blue-500/20">{humanize(strOf(h.evidence_strength || h.strength))} evidence</span>
                  )}
                  {strOf(h?.status) && <span className="text-[10.5px] text-muted-foreground">{humanize(strOf(h.status))}</span>}
                  {sort === "severity" && severityOf(h) > 0 && (
                    <span className="text-[10.5px] font-medium px-2 py-0.5 rounded-full bg-violet-50 dark:bg-violet-500/10 text-violet-700 dark:text-violet-400 border border-violet-200 dark:border-violet-500/25" title="Ranked by size of move times evidence strength">{severityReason(h)}</span>
                  )}
                  <button onClick={() => openDetail(strOf(h?.id))} disabled={!h?.id} className="ml-auto inline-flex items-center gap-1 text-[12px] font-semibold text-blue-700 dark:text-blue-400 hover:underline disabled:opacity-40" aria-label={`View hypothesis ${strOf(h?.title || "")}`}>
                    <Eye size={12} /> View
                  </button>
                </div>
                <div className="mt-2 pt-2 border-t border-border flex items-center gap-3 flex-wrap">
                  <button
                    onClick={() => trackAsGoal(h)}
                    disabled={trackingId !== null}
                    className="inline-flex items-center gap-1 text-[11.5px] font-semibold text-violet-700 dark:text-violet-400 hover:underline disabled:opacity-40"
                    aria-label={`Track hypothesis as goal: ${strOf(h?.title || "")}`}>
                    <Target size={12} /> {trackingId === strOf(h?.id) ? "Creating…" : "Track as goal"}
                  </button>
                  <button
                    onClick={() => addHypToDraft(h)}
                    className="inline-flex items-center gap-1 text-[11.5px] font-medium text-muted-foreground hover:text-blue-700 dark:hover:text-blue-400 hover:underline"
                    aria-label={`Add hypothesis to report draft: ${strOf(h?.title || "")}`}>
                    <FileText size={12} /> Add to report
                  </button>
                  <button
                    onClick={() => testThis(h)}
                    className="inline-flex items-center gap-1 text-[11.5px] font-medium text-muted-foreground hover:text-blue-700 dark:hover:text-blue-400 hover:underline"
                    aria-label={`Test hypothesis in Explore: ${strOf(h?.title || "")}`}>
                    <Play size={12} /> Test this
                  </button>
                </div>
              </Card>
            ))}
          </div>
          {pages !== null && pages > 1 && (
            <div className="flex items-center justify-center gap-2" role="navigation" aria-label="Hypotheses pages">
              <button onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={page <= 1} className={btnGhost} aria-label="Previous page">← Prev</button>
              <span className="text-[12px] text-muted-foreground">Page {page} of {pages}{total !== null ? ` · ${total} total` : ""}</span>
              <button onClick={() => setPage((p) => Math.min(pages, p + 1))} disabled={page >= pages} className={btnGhost} aria-label="Next page">Next →</button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

/* =====================================================================
   GOALS
   ===================================================================== */

type GoalFilter = "active" | "completed" | "paused" | "all";

function GoalsTab({ meta }: { meta: any }) {
  const [filter, setFilter] = useState<GoalFilter>("active");
  const [goals, setGoals] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [errorId, setErrorId] = useState("");
  const [showForm, setShowForm] = useState(false);
  const [fTitle, setFTitle] = useState("");
  const [fMetric, setFMetric] = useState("");
  const [fTarget, setFTarget] = useState("");
  const [fDue, setFDue] = useState("");
  const [formError, setFormError] = useState("");
  const [formBusy, setFormBusy] = useState(false);
  const [actingId, setActingId] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [refreshMsg, setRefreshMsg] = useState("");
  const [refreshingId, setRefreshingId] = useState<string | null>(null);
  const [refreshingAll, setRefreshingAll] = useState(false);

  const metrics: string[] = useMemo(() => {
    const m = meta?.metrics ?? meta?.measures ?? [];
    return asList(m).map((x) => strOf(typeof x === "string" ? x : x?.name || x?.label)).filter(Boolean);
  }, [meta]);

  const fetchGoals = useCallback(async () => {
    setLoading(true);
    setError("");
    setErrorId("");
    try {
      const r = await intelligenceApi.listGoals({ status: filter === "all" ? undefined : filter, sort: "due" });
      setGoals(asList(r.data));
    } catch (e: any) {
      const norm = normalizeApiError({ message: e?.message }, e?.status || 0, null);
      setError(norm.message);
      setErrorId(norm.requestId);
      setGoals([]);
    } finally {
      setLoading(false);
    }
  }, [filter]);
  useEffect(() => { fetchGoals(); }, [fetchGoals]);

  /* Cross-tab arrivals: toast from hypothesis-to-goal + prefill when the
     target was left blank for the user. */
  useEffect(() => {
    const t = popToast();
    if (t) setToast(t);
    const p = readGoalPrefill();
    if (p && typeof p === "object") {
      if (strOf(p?.name)) setFTitle(strOf(p.name));
      if (strOf(p?.metric)) setFMetric(strOf(p.metric));
      if (p?.target !== undefined && p?.target !== null && String(p.target) !== "") setFTarget(String(p.target));
      if (strOf(p?.due)) setFDue(strOf(p.due));
      setShowForm(true);
    }
  }, []);

  const createGoal = async () => {
    setFormError("");
    if (!fTitle.trim() || !fTarget.trim()) { setFormError("Give the goal a title and a target value."); return; }
    setFormBusy(true);
    try {
      /* Seed progress from the Explore API: current = live total for the
         chosen metric, so progress = current/target is backend truth. */
      let current: number | undefined;
      if (fMetric) {
        try {
          const r = await intelligenceApi.runExplore({ metric: fMetric });
          const t = numOf(r.data?.total ?? r.data?.total_display);
          if (t !== null) current = t;
        } catch { /* fall through without a seeded current */ }
      }
      await intelligenceApi.createGoal({
        name: fTitle.trim(), metric: fMetric || undefined, target: fTarget.trim(),
        ...(current !== undefined ? { current } : {}),
        due: fDue || undefined, status: "active",
      });
      setShowForm(false);
      setFTitle(""); setFMetric(""); setFTarget(""); setFDue("");
      setFilter("active");
      fetchGoals();
    } catch (e: any) {
      setFormError(businessError(e?.message) || "ARIA couldn't save this goal.");
    } finally {
      setFormBusy(false);
    }
  };

  const deleteGoal = async (id: string) => {
    if (!id) return;
    setActingId(id);
    try {
      await intelligenceApi.deleteGoal(id);
      fetchGoals();
    } catch (e: any) {
      setError(businessError(e?.message) || "ARIA couldn't delete this goal.");
    } finally {
      setActingId(null);
    }
  };

  const setStatus = async (id: string, status: string) => {
    setActingId(id);
    try {
      await intelligenceApi.updateGoal(id, { status });
      fetchGoals();
    } catch (e: any) {
      setError(businessError(e?.message) || "ARIA couldn't update this goal.");
    } finally {
      setActingId(null);
    }
  };

  /* Goal auto-progress: recompute current from live data server-side.
     Honest failure surfaces the backend message; the goal stays unchanged. */
  const refreshOne = async (id: string) => {
    if (!id) return;
    setRefreshingId(id);
    setRefreshMsg("");
    try {
      await intelligenceApi.refreshGoal(id);
      await fetchGoals();
    } catch (e: any) {
      setRefreshMsg(businessError(e?.message) || "ARIA couldn't refresh this goal — it was left unchanged.");
    } finally {
      setRefreshingId(null);
    }
  };

  const refreshAll = async () => {
    if (!goals.length || refreshingAll) return;
    setRefreshingAll(true);
    setRefreshMsg("");
    let failed = 0;
    let lastErr = "";
    for (const g of goals) {
      const id = strOf(g?.id || g?.goal_id);
      if (!id) { failed += 1; continue; }
      try {
        await intelligenceApi.refreshGoal(id);
      } catch (e: any) {
        failed += 1;
        lastErr = businessError(e?.message) || "";
      }
    }
    await fetchGoals();
    setRefreshingAll(false);
    if (failed > 0) {
      setRefreshMsg(lastErr
        ? `${failed} goal${failed === 1 ? "" : "s"} couldn't refresh and ${failed === 1 ? "was" : "were"} left unchanged: ${lastErr}`
        : `${failed} goal${failed === 1 ? "" : "s"} couldn't refresh and ${failed === 1 ? "was" : "were"} left unchanged.`);
    }
  };

  const statusTone = (s: string) => {
    const t = s.toLowerCase();
    if (/behind|at risk|off track/.test(t)) return "bg-rose-50 dark:bg-rose-500/10 text-rose-700 dark:text-rose-400 border-rose-200 dark:border-rose-500/25";
    if (/near|close|watch/.test(t)) return "bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-500/25";
    if (/on track|ahead|complete|achieved/.test(t)) return "bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-500/25";
    return "bg-secondary text-secondary-foreground border-border";
  };

  const filters: { id: GoalFilter; label: string }[] = [
    { id: "active", label: "Active Goals" },
    { id: "completed", label: "Completed" },
    { id: "paused", label: "Paused" },
    { id: "all", label: "All Goals" },
  ];

  if (!loading && !error && metrics.length === 0 && goals.length === 0) {
    const noSchema = getNoSchemaInfo(meta);
    return (
      <div className="space-y-4">
        <NoDatasetPanel onRetry={fetchGoals} hint={noSchema.isNoSchema ? noSchema.hint : undefined} />
      </div>
    );
  }
  return (
    <div className="space-y-4">
      <div className="text-[11px] text-muted-foreground flex items-center gap-1">
        <span>Intelligence</span><ChevronRight size={11} /><span className="text-secondary-foreground font-medium">Goals</span>
      </div>
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div>
          <h3 className="text-[16px] font-semibold text-foreground tracking-tight">Business Goals</h3>
          <p className="text-[12px] text-muted-foreground">Track your targets and monitor progress.</p>
        </div>
        <div className="flex items-center gap-1.5">
          <button
            onClick={refreshAll}
            disabled={refreshingAll || goals.length === 0}
            className={btnGhost}
            aria-label="Refresh all goals from live data">
            <RefreshCw size={13} className={refreshingAll ? "animate-spin" : ""} /> {refreshingAll ? "Refreshing…" : "Refresh all"}
          </button>
          <button onClick={() => setShowForm((v) => !v)} className={btnPrimary} aria-label="Create goal">
            <Plus size={13} /> Create Goal
          </button>
        </div>
      </div>
      {toast && (
        <div className="rounded-xl border border-emerald-200 dark:border-emerald-500/25 bg-emerald-50 dark:bg-emerald-500/10 px-3 py-2 flex items-start gap-2" role="status">
          <Check size={14} className="text-emerald-600 dark:text-emerald-400 flex-shrink-0 mt-0.5" />
          <p className="text-[12px] text-emerald-800 dark:text-emerald-400 flex-1">{toast}</p>
          <button onClick={() => setToast(null)} className="text-[11px] font-semibold text-emerald-700 dark:text-emerald-400 hover:underline" aria-label="Dismiss notification">Dismiss</button>
        </div>
      )}
      {refreshMsg && (
        <p className="text-[12px] text-amber-700 dark:text-amber-400" role="alert">{refreshMsg}</p>
      )}

      {showForm && (
        <Card className="p-4 md:p-5">
          <h4 className="text-[13px] font-semibold text-foreground">Set a new goal</h4>
          <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-3">
            <Field label="Goal title" htmlFor="goal-title">
              <input id="goal-title" value={fTitle} onChange={(e) => setFTitle(e.target.value)} placeholder="e.g. Grow monthly revenue" className={inputCls} aria-label="Goal title" />
            </Field>
            <Field label="Metric" htmlFor="goal-metric">
              <select id="goal-metric" value={fMetric} onChange={(e) => setFMetric(e.target.value)} className={selectCls} aria-label="Goal metric">
                <option value="">Select a metric…</option>
                {metrics.map((m) => <option key={m} value={m}>{humanize(m)}</option>)}
              </select>
            </Field>
            <Field label="Target value" htmlFor="goal-target">
              <input id="goal-target" value={fTarget} onChange={(e) => setFTarget(e.target.value)} placeholder="e.g. 350000" inputMode="decimal" className={inputCls} aria-label="Target value" />
            </Field>
            <Field label="Due date" htmlFor="goal-due">
              <input id="goal-due" type="date" value={fDue} onChange={(e) => setFDue(e.target.value)} className={inputCls} aria-label="Due date" />
            </Field>
          </div>
          {formError && <p className="mt-2 text-[12px] text-rose-600 dark:text-rose-400" role="alert">{formError}</p>}
          <div className="mt-3 flex items-center gap-2">
            <button onClick={createGoal} disabled={formBusy} className={btnPrimary} aria-label="Save goal">
              {formBusy ? "Saving…" : "Save Goal"}
            </button>
            <button onClick={() => { setShowForm(false); setFormError(""); }} className={btnGhost}>Cancel</button>
          </div>
        </Card>
      )}

      <div className="flex items-center gap-1 border-b border-border" role="tablist" aria-label="Goal status filter">
        {filters.map((f) => (
          <button key={f.id} role="tab" aria-selected={filter === f.id} onClick={() => setFilter(f.id)}
            className={`px-3 py-2 text-[12.5px] font-medium border-b-2 -mb-px transition-colors ${filter === f.id ? "border-blue-700 text-foreground" : "border-transparent text-muted-foreground hover:text-secondary-foreground"}`}>
            {f.label}
          </button>
        ))}
        <button onClick={fetchGoals} className="ml-auto p-1.5 rounded-lg hover:bg-secondary text-muted-foreground hover:text-foreground" title="Refresh goals" aria-label="Refresh goals">
          <RefreshCw size={13} />
        </button>
      </div>

      {loading && <LoadingPanel label="Loading goals…" />}
      {!loading && error && <ErrorPanel message={error} requestId={errorId} onRetry={fetchGoals} />}
      {!loading && !error && goals.length === 0 && (
        <EmptyState
          icon={Target}
          title={filter === "all" ? "No goals yet." : `No ${filter} goals.`}
          body="Define a target, track progress, and let ARIA help you stay on track. Goals persist, so they survive a reload."
          action={<button onClick={() => setShowForm(true)} className="inline-flex items-center gap-1.5 text-xs font-semibold text-blue-700 dark:text-blue-400 hover:underline"><Plus size={12} /> Set your first goal</button>}
        />
      )}
      {!loading && !error && goals.length > 0 && (
        <div className="space-y-3">
          {goals.map((g: any) => {
            const id = strOf(g?.id || g?.goal_id);
            const cur = numOf(g?.current ?? g?.current_value);
            const tgt = numOf(g?.target ?? g?.target_value);
            const delta = numOf(g?.delta ?? g?.difference);
            const deltaPct = numOf(g?.delta_pct ?? g?.delta_percent ?? g?.pct_off);
            const prog = numOf(g?.progress ?? g?.progress_pct ?? g?.percent_complete);
            const status = strOf(g?.status_label || g?.status || "");
            const due = strOf(g?.due_date || g?.due || "");
            return (
              <Card key={id || strOf(g?.title)} className="p-4">
                <div className="flex items-start gap-3 flex-wrap">
                  <span className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-violet-50 dark:bg-violet-500/10 text-violet-700 dark:text-violet-400 flex-shrink-0"><Target size={15} /></span>
                  <div className="min-w-0 flex-1">
                    <p className="text-[13.5px] font-semibold text-foreground">{strOf(g?.title || g?.name || "Goal")}</p>
                    <p className="text-[11px] text-muted-foreground mt-0.5 flex items-center gap-1">
                      <CalendarDays size={11} /> {due ? `Due ${due}` : "No due date"}{strOf(g?.metric) ? ` · ${humanize(strOf(g.metric))}` : ""}
                    </p>
                    <div className="mt-2 flex items-center gap-4 flex-wrap text-[12px]">
                      <span className="text-muted-foreground">Current <span className="font-semibold text-foreground tabular-nums">{cur === null ? "—" : fmtBiz(cur)}</span></span>
                      <span className="text-muted-foreground">Target <span className="font-semibold text-foreground tabular-nums">{tgt === null ? "—" : fmtBiz(tgt)}</span></span>
                      {(delta !== null || deltaPct !== null) && (
                        <span className={`font-semibold tabular-nums ${(delta ?? 0) >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                          {delta !== null ? fmtBiz(delta) : ""}{deltaPct !== null ? ` (${fmtPctBiz(deltaPct)})` : ""}
                        </span>
                      )}
                    </div>
                    {prog !== null && (
                      <div className="mt-2 max-w-md">
                        <div className="h-1.5 rounded-full bg-secondary overflow-hidden" role="progressbar" aria-valuenow={Math.round(prog)} aria-valuemin={0} aria-valuemax={100} aria-label={`Progress for ${strOf(g?.title)}`}>
                          <div className={`h-full rounded-full ${/behind|at risk|off track/i.test(status) ? "bg-gradient-to-r from-rose-500 to-rose-400" : /near|watch/i.test(status) ? "bg-gradient-to-r from-amber-500 to-amber-400" : "bg-gradient-to-r from-emerald-500 to-emerald-400"}`} style={{ width: `${Math.max(0, Math.min(100, prog))}%` }} />
                        </div>
                      </div>
                    )}
                  </div>
                  <div className="flex flex-col items-end gap-2">
                    {status && <span className={`text-[10.5px] font-semibold px-2 py-0.5 rounded-full border ${statusTone(status)}`}>{status}</span>}
                    {filter !== "completed" && (
                      <div className="flex items-center gap-1.5 flex-wrap justify-end">
                        <button onClick={() => id && refreshOne(id)} disabled={!id || refreshingId === id} className="inline-flex items-center gap-1 text-[11px] font-semibold text-blue-700 dark:text-blue-400 hover:underline disabled:opacity-40" aria-label={`Refresh goal from live data: ${strOf(g?.title)}`}>
                          <RefreshCw size={11} className={refreshingId === id ? "animate-spin" : ""} /> {refreshingId === id ? "…" : "Refresh"}
                        </button>
                        {filter !== "paused" && (
                          <button onClick={() => id && setStatus(id, "paused")} disabled={!id || actingId === id} className="text-[11px] font-medium text-muted-foreground hover:text-foreground disabled:opacity-40" aria-label={`Pause goal ${strOf(g?.title)}`}>Pause</button>
                        )}
                        {(filter === "paused") && (
                          <button onClick={() => id && setStatus(id, "active")} disabled={!id || actingId === id} className="text-[11px] font-medium text-blue-700 dark:text-blue-400 hover:underline disabled:opacity-40" aria-label={`Resume goal ${strOf(g?.title)}`}>Resume</button>
                        )}
                        <button onClick={() => id && setStatus(id, "completed")} disabled={!id || actingId === id} className="text-[11px] font-medium text-emerald-700 dark:text-emerald-400 hover:underline disabled:opacity-40" aria-label={`Complete goal ${strOf(g?.title)}`}>
                          {actingId === id ? "…" : "Complete"}
                        </button>
                        <button onClick={() => id && deleteGoal(id)} disabled={!id || actingId === id} className="text-[11px] font-medium text-rose-600 dark:text-rose-400 hover:underline disabled:opacity-40" aria-label={`Delete goal ${strOf(g?.title)}`}>Delete</button>
                      </div>
                    )}
                  </div>
                </div>
              </Card>
            );
          })}
          <Card className="p-4 flex items-center gap-3 flex-wrap border-dashed !border-2">
            <span className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-blue-50 dark:bg-blue-500/10 text-blue-700 dark:text-blue-400"><Target size={15} /></span>
            <div className="flex-1 min-w-40">
              <p className="text-[13px] font-semibold text-foreground">Set a new goal</p>
              <p className="text-[11px] text-muted-foreground">Define a target, track progress, and let ARIA help you stay on track.</p>
            </div>
            <button onClick={() => setShowForm(true)} className={btnPrimary} aria-label="Create another goal"><Plus size={13} /> Create Goal</button>
          </Card>
        </div>
      )}
    </div>
  );
}

/* =====================================================================
   WHAT-IF
   ===================================================================== */

function WhatIfTab() {
  const [meta, setMeta] = useState<any>(null);
  const [metaLoading, setMetaLoading] = useState(true);
  const [metaError, setMetaError] = useState("");
  const [metaId, setMetaId] = useState("");
  const [variable, setVariable] = useState("");
  const [changeType, setChangeType] = useState("");
  const [changeValue, setChangeValue] = useState<number>(10);
  const [applyTo, setApplyTo] = useState("");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState("");
  const [errorId, setErrorId] = useState("");
  const [saved, setSaved] = useState<any[]>([]);
  const [savedError, setSavedError] = useState("");

  const variables: string[] = useMemo(() => asList(meta?.variables).map((v) => strOf(typeof v === "string" ? v : v?.name || v?.label)).filter(Boolean), [meta]);
  const changeTypes: string[] = useMemo(() => asList(meta?.change_types ?? meta?.changeTypes).map((v) => strOf(typeof v === "string" ? v : v?.name || v?.label)).filter(Boolean), [meta]);
  const applyTargets: string[] = useMemo(() => asList(meta?.apply_targets ?? meta?.applyTargets ?? meta?.scopes).map((v) => strOf(typeof v === "string" ? v : v?.name || v?.label)).filter(Boolean), [meta]);
  const range = useMemo(() => {
    const r = meta?.ranges?.[variable] ?? meta?.ranges?.[humanize(variable)] ?? meta?.range ?? null;
    if (!r || typeof r !== "object") return null;
    const min = numOf(r?.min ?? r?.minimum);
    const max = numOf(r?.max ?? r?.maximum);
    const step = numOf(r?.step) ?? 1;
    if (min === null || max === null) return null;
    return { min, max, step };
  }, [meta, variable]);

  const loadMeta = useCallback(async () => {
    setMetaLoading(true);
    setMetaError("");
    setMetaId("");
    try {
      const r = await intelligenceApi.getWhatIfMeta();
      setMeta(r.data || null);
    } catch (e: any) {
      const norm = normalizeApiError({ message: e?.message }, e?.status || 0, null);
      setMetaError(norm.message);
      setMetaId(norm.requestId);
      setMeta(null);
    } finally {
      setMetaLoading(false);
    }
  }, []);
  useEffect(() => { loadMeta(); }, [loadMeta]);

  useEffect(() => {
    if (!variable && variables.length) {
      /* Match the board default: prefer an Average-Order-Value-like measure. */
      const aov = variables.find((v) => /average.*order|aov|avg.*order/i.test(v));
      setVariable(aov || variables[0]);
    }
    if (!changeType && changeTypes.length) setChangeType(changeTypes[0]);
    if (!applyTo && applyTargets.length) setApplyTo(applyTargets[0]);
  }, [variables, changeTypes, applyTargets, variable, changeType, applyTo]);

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem("aria_whatif_prefill");
      if (raw && variables.length) {
        const p = JSON.parse(raw);
        const v = strOf(p?.variable);
        if (v && variables.some((x) => x.toLowerCase() === v.toLowerCase())) setVariable(v);
        sessionStorage.removeItem("aria_whatif_prefill");
      }
    } catch { /* ignore */ }
  }, [variables]);

  useEffect(() => {
    if (range) setChangeValue((v) => Math.max(range.min, Math.min(range.max, v)));
  }, [range]);

  const loadSaved = useCallback(async () => {
    try {
      const r = await intelligenceApi.listScenarios({ page_size: 6, sort: "recent" });
      setSaved(asList(r.data));
      setSavedError("");
    } catch (e: any) {
      console.warn("Failed to load saved scenarios:", e?.message || e);
      setSavedError(businessError(e?.message) || "ARIA couldn't load saved scenarios.");
    }
  }, []);
  useEffect(() => { loadSaved(); }, [loadSaved]);

  const [saveMsg, setSaveMsg] = useState("");
  /* Real segment values for "Apply to": live breakdown rows for the chosen
     variable (never hardcoded segment names). */
  const [segValues, setSegValues] = useState<string[]>([]);
  const [segLoading, setSegLoading] = useState(false);
  const [segDim, setSegDim] = useState("");

  useEffect(() => {
    let cancelled = false;
    const loadSegments = async () => {
      if (!variable) {
        setSegValues([]);
        setSegDim("");
        return;
      }
      setSegLoading(true);
      try {
        const m = await intelligenceApi.getMeta();
        const dims = asList((m.data as any)?.dimensions ?? (m.data as any)?.breakdowns)
          .map((d) => strOf(typeof d === "string" ? d : d?.column || d?.name || d?.label))
          .filter(Boolean);
        for (const dim of dims.slice(0, 3)) {
          try {
            const r = await intelligenceApi.runExplore({ metric: variable, breakdown: dim });
            const rows = asList((r.data as any)?.breakdown_rows
              ?? (r.data as any)?.breakdown_data ?? (r.data as any)?.rows);
            const cats = rows
              .map((x) => strOf(x?.category ?? x?.label ?? x?.name))
              .filter(Boolean);
            if (cats.length && !cancelled) {
              setSegValues(cats.slice(0, 8));
              setSegDim(dim);
              return;
            }
          } catch { /* try the next dimension */ }
        }
        if (!cancelled) {
          setSegValues([]);
          setSegDim("");
        }
      } catch {
        if (!cancelled) {
          setSegValues([]);
          setSegDim("");
        }
      } finally {
        if (!cancelled) setSegLoading(false);
      }
    };
    loadSegments();
    return () => { cancelled = true; };
  }, [variable]);

  /* Apply-to options: "All data" plus real segment values when available;
     generic backend placeholders are only a fallback. */
  const applyOptions: string[] = useMemo(() => {
    if (segValues.length) return ["All data", ...segValues];
    return applyTargets.length ? applyTargets : ["All data"];
  }, [segValues, applyTargets]);

  useEffect(() => {
    if (applyTo && !applyOptions.includes(applyTo)) setApplyTo("All data");
  }, [applyOptions, applyTo]);
  const saveCurrentScenario = async () => {
    if (!result || !variable) return;
    setSaveMsg("");
    try {
      await intelligenceApi.saveScenario({ variable, change_type: changeType, change_value: changeValue, apply_to: applyTo, title: `${humanize(variable)} ${changeValue}%`, result });
      setSaveMsg("Scenario saved below.");
      loadSaved();
    } catch (e: any) {
      setSaveMsg(businessError(e?.message) || "ARIA couldn't save this scenario.");
    }
  };

  const loadSavedScenario = (s: any) => {
    if (strOf(s?.variable)) setVariable(strOf(s.variable));
    if (numOf(s?.change_value) !== null) setChangeValue(numOf(s.change_value) as number);
    if (strOf(s?.change_type)) setChangeType(strOf(s.change_type));
    if (strOf(s?.apply_to)) setApplyTo(strOf(s.apply_to));
    setTimeout(() => runScenario(), 120);
  };

  const deleteSavedScenario = async (id: string) => {
    if (!id) return;
    try {
      await intelligenceApi.deleteScenario(id);
      loadSaved();
    } catch (e: any) {
      console.warn("Failed to delete scenario:", e?.message || e);
      setSavedError(businessError(e?.message) || "ARIA couldn't delete this scenario.");
    }
  };

  const runScenario = async () => {
    setRunning(true);
    setError("");
    setErrorId("");
    setResult(null);
    try {
      /* Results come ONLY from POST /api/intelligence/what-if/run. */
      const r = await intelligenceApi.runWhatIf({ variable, change_type: changeType, change_value: changeValue, apply_to: applyTo });
      const rd: any = (r as any)?.data ?? null;
      /* Honest backend refusal (ok:False, 200) must surface as an error —
         never as an empty "no projected impact" card. */
      if (rd && rd.ok === false) {
        setError(rd.detail || rd.message || "ARIA couldn't model this scenario from your data.");
        setErrorId(rd.code ? String(rd.code) : "");
        setResult(null);
        return;
      }
      setResult(rd);
    } catch (e: any) {
      const norm = normalizeApiError({ message: e?.message }, e?.status || 0, null);
      setError(norm.message);
      setErrorId(norm.requestId);
    } finally {
      setRunning(false);
    }
  };

  const impacts: any[] = asList(result?.projected_impact || result?.impacts || result?.kpis);
  const compSeries = useMemo(() => {
    const raw = asList(result?.revenue_comparison || result?.comparison_series || result?.comparison || result?.monthly || result?.chart?.data);
    // Drop incomplete rows instead of plotting invented zeros / P-labels.
    return raw
      .map((p: any) => ({
        label: strOf(p?.label || p?.period),
        baseline: numOf(p?.baseline ?? p?.base ?? p?.current),
        scenario: numOf(p?.scenario ?? p?.projected ?? p?.value),
      }))
      .filter((p: any) => p.label && (p.baseline !== null || p.scenario !== null))
      .slice(0, 24);
  }, [result]);
  const compTitle = strOf(
    result?.comparison_title || (variable ? `${humanize(variable)} Comparison` : "Scenario Comparison"),
    "Scenario Comparison",
  );
  const summary = strOf(result?.scenario_summary || result?.summary || "");
  const assumptions: string[] = asList(result?.assumptions).map((a) => strOf(typeof a === "string" ? a : a?.text)).filter(Boolean);

  if (metaLoading) return <LoadingPanel label="Loading scenario options…" />;
  if (metaError) return <ErrorPanel message={metaError} requestId={metaId} onRetry={loadMeta} />;
  if (!meta || (variables.length === 0 && changeTypes.length === 0)) {
    const noSchema = getNoSchemaInfo(meta);
    if (noSchema.isNoSchema) {
      return <NoDatasetPanel onRetry={loadMeta} hint={noSchema.hint} />;
    }
    return <EmptyState icon={SlidersHorizontal} title="No scenario variables available." body="ARIA couldn't identify supported scenario variables for this dataset, so there is nothing meaningful to model yet." action={<button onClick={loadMeta} className="text-xs font-semibold text-blue-700 dark:text-blue-400 hover:underline">Retry</button>} />;
  }

  const resetScenario = () => { setResult(null); setError(""); setErrorId(""); setChangeValue(10); };
  return (
    <div className="space-y-4">
      <div className="text-[11px] text-muted-foreground flex items-center gap-1">
        <span>Intelligence</span><ChevronRight size={11} /><span className="text-secondary-foreground font-medium">What-If</span>
      </div>
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div>
          <h3 className="text-[16px] font-semibold text-foreground tracking-tight">What-If Scenario Analysis</h3>
          <p className="text-[12px] text-muted-foreground">Explore how changes in key factors may impact your business.</p>
        </div>
        <div className="flex items-center gap-1.5">
          <button onClick={saveCurrentScenario} disabled={!result} className={btnGhost} aria-label="Save scenario">Save Scenario</button>
          <button onClick={resetScenario} className={btnGhost} aria-label="Reset scenario"><RotateCcw size={13} /> Reset</button>
        </div>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card className="p-4 md:p-5">
          <h3 className="text-[14px] font-semibold text-foreground tracking-tight">Scenario Setup</h3>
          <div className="mt-3 space-y-3">
            <Field label="Variable to change" htmlFor="wi-var">
              <select id="wi-var" value={variable} onChange={(e) => setVariable(e.target.value)} className={selectCls} aria-label="Scenario variable">
                {variables.map((v) => <option key={v} value={v}>{humanize(v)}</option>)}
              </select>
            </Field>
            <Field label="Change Type" htmlFor="wi-type">
              <select id="wi-type" value={changeType} onChange={(e) => setChangeType(e.target.value)} className={selectCls} aria-label="Change type">
                {changeTypes.map((c) => <option key={c} value={c}>{humanize(c)}</option>)}
              </select>
            </Field>
            <div>
              <label htmlFor="wi-value" className="block text-[11px] font-medium text-muted-foreground mb-1">
                Change Value{range ? ` (supported range ${range.min}% to ${range.max}%)` : ""}
              </label>
              {range ? (
                <div className="flex items-center gap-3">
                  <span className="text-[11px] text-muted-foreground w-8 text-right tabular-nums">{range.min}%</span>
                  <div className="relative flex-1 h-6 flex items-center">
                    <div className="absolute inset-x-0 h-[3px] bg-border rounded-full" />
                    <div
                      className="absolute h-[3px] bg-blue-500 rounded-full"
                      style={{
                        left: `${((Math.max(range.min, Math.min(range.max, changeValue)) - range.min) / (range.max - range.min)) * 100}%`,
                        right: 0,
                      }}
                    />
                    <input
                      id="wi-value" type="range" min={range.min} max={range.max} step={range.step} value={changeValue}
                      onChange={(e) => setChangeValue(Number(e.target.value))}
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                      aria-label="Change value slider"
                    />
                    <div
                      className="absolute w-4 h-4 bg-card border-2 border-blue-600 rounded-full shadow-sm pointer-events-none z-20 transition-transform hover:scale-110"
                      style={{
                        left: `calc(${((Math.max(range.min, Math.min(range.max, changeValue)) - range.min) / (range.max - range.min)) * 100}% - 8px)`,
                      }}
                    />
                  </div>
                  <span className="text-[11px] text-muted-foreground w-8 tabular-nums">{range.max}%</span>
                  <span className="text-[13px] font-bold text-foreground tabular-nums w-14 text-right">{changeValue >= 0 ? "+" : ""}{changeValue}%</span>
                </div>
              ) : (
                <input id="wi-value" type="number" value={changeValue} onChange={(e) => setChangeValue(Number(e.target.value) || 0)} className={inputCls} aria-label="Change value" />
              )}
              {range && (changeValue < range.min || changeValue > range.max) && (
                <p className="text-[11px] text-amber-600 dark:text-amber-400 mt-1">Outside the supported range — results may be less reliable.</p>
              )}
            </div>
            <fieldset>
              <legend className="text-[11px] font-medium text-muted-foreground mb-1.5">Apply to</legend>
              <div className="space-y-1.5" role="radiogroup" aria-label="Apply scenario to">
                {applyOptions.map((t) => (
                  <label key={t} className="flex items-center gap-2 text-[12.5px] text-foreground cursor-pointer">
                    <input type="radio" name="wi-apply" checked={applyTo === t} onChange={() => setApplyTo(t)} className="accent-blue-700" />
                    {humanize(t)}
                  </label>
                ))}
              </div>
              {segLoading && <p className="text-[11px] text-muted-foreground mt-1">Loading live segments…</p>}
              {!segLoading && segValues.length > 0 && (
                <p className="text-[11px] text-muted-foreground mt-1">Segments from your live data — breakdown of {humanize(variable)} by {humanize(segDim)}.</p>
              )}
            </fieldset>
            <button onClick={runScenario} disabled={running || !variable} className={`${btnPrimary} w-full`} aria-label="Run scenario">
              {running ? "Running…" : "Run Scenario"} <Play size={13} />
            </button>
          </div>
        </Card>

        <div className="lg:col-span-2 space-y-4">
          {running && <LoadingPanel label="Modeling scenario…" />}
          {!running && error && <ErrorPanel message={error} requestId={errorId} onRetry={runScenario} retryLabel="Run again" />}
          {!running && !error && !result && (
            <EmptyState icon={SlidersHorizontal} title="No scenario yet." body="Choose a variable, set a change, and run the scenario. ARIA models the outcome from your historical data." />
          )}
          {!running && !error && result && (
            <>
              <Card className="p-4 md:p-5">
                <h3 className="text-[14px] font-semibold text-foreground tracking-tight">Projected Impact</h3>
                {impacts.length ? (
                  <div className="mt-2 grid grid-cols-1 sm:grid-cols-3 gap-2">
                    {impacts.map((k: any, i: number) => (
                      <div key={i} className="rounded-xl border border-border p-3">
                        <p className="text-[11px] text-muted-foreground">{strOf(k?.label || k?.metric || `Impact ${i + 1}`)}</p>
                        <p className="text-[20px] font-semibold text-foreground tabular-nums mt-0.5">{k?.display || (numOf(k?.value) !== null ? fmtBiz(numOf(k.value)) : "—")}</p>
                        {strOf(k?.delta || k?.change_note) && <p className="text-[11px] text-muted-foreground mt-0.5">{strOf(k.delta || k.change_note)}</p>}
                      </div>
                    ))}
                  </div>
                ) : <p className="text-[12px] text-muted-foreground mt-2">The backend returned no projected impact figures.</p>}
              </Card>
              <Card className="p-4 md:p-5">
                <h3 className="text-[14px] font-semibold text-foreground tracking-tight">{compTitle}</h3>
                {compSeries.length ? (
                  <div className="h-[220px] mt-2">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={compSeries} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                        <XAxis dataKey="label" tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} minTickGap={16} />
                        <YAxis tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} width={52} tickFormatter={(v: number) => fmtBiz(v)} />
                        <Tooltip formatter={(v: any, name: any) => [fmtMetric(numOf(v), variable), humanize(String(name))]} contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        <Bar dataKey="baseline" name="Baseline" fill="#C7D6E8" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="scenario" name={`Scenario (${changeValue >= 0 ? "+" : ""}${changeValue}%)`} fill="#4F46E5" radius={[4, 4, 0, 0]}>
                          {compSeries.map((_, i) => <Cell key={i} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : <p className="text-[12px] text-muted-foreground mt-2">The backend returned no comparison series.</p>}
              </Card>
              {(summary || assumptions.length > 0) && (
                <Card className="p-4 md:p-5">
                  <h3 className="text-[14px] font-semibold text-foreground tracking-tight">Scenario Summary</h3>
                  {summary && <p className="text-[12.5px] text-secondary-foreground leading-relaxed mt-2">{summary}</p>}
                  {assumptions.length > 0 && (
                    <ul className="mt-2 space-y-1">
                      {assumptions.map((a, i) => <li key={i} className="text-[11.5px] text-muted-foreground leading-relaxed">· {a}</li>)}
                    </ul>
                  )}
                  <p className="mt-2 text-[11px] text-muted-foreground">This is a scenario estimate based on your historical data. Actual results may vary.</p>
                  <div className="mt-3 flex items-center gap-2">
                    <button onClick={saveCurrentScenario} className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg text-[12px] font-semibold text-blue-700 dark:text-blue-400 border border-blue-200 dark:border-blue-500/25 hover:bg-blue-50 dark:hover:bg-blue-500/10" aria-label="Save scenario">
                      <Save size={13} /> Save Scenario
                    </button>
                    {saveMsg && <span className="text-[11.5px] text-secondary-foreground" role="status">{saveMsg}</span>}
                  </div>
                </Card>
              )}
            </>
          )}
        </div>
      </div>

      <Card className="p-4 md:p-5">
        <CardHead title="Saved Scenarios" sub="Scenarios you kept for later." right={<button onClick={loadSaved} className="text-[11px] font-semibold text-blue-700 dark:text-blue-400 hover:underline">View all</button>} />
        {savedError && <p className="mt-2 text-[12px] text-rose-600 dark:text-rose-400" role="alert">{savedError} <button onClick={loadSaved} className="font-semibold text-blue-700 dark:text-blue-400 hover:underline ml-1">Retry</button></p>}
        <div className="mt-2 grid grid-cols-1 sm:grid-cols-2 gap-2">
          {!savedError && saved.length === 0 && <p className="text-[12px] text-muted-foreground">No saved scenarios yet.</p>}
          {saved.map((s: any, i: number) => (
            <div key={strOf(s?.id || i)} className="rounded-xl border border-border p-3 flex items-center gap-2">
              <SlidersHorizontal size={14} className="text-indigo-600 dark:text-indigo-400 flex-shrink-0" />
              <div className="min-w-0 flex-1">
                <p className="text-[12.5px] font-medium text-foreground truncate">{strOf(s?.title || s?.name || `${humanize(strOf(s?.variable))} ${strOf(s?.change_value)}%`)}</p>
                <p className="text-[11px] text-muted-foreground">{strOf(s?.created_at || "")}</p>
              </div>
              <button onClick={() => loadSavedScenario(s)} className="text-[11px] font-semibold text-blue-700 dark:text-blue-400 hover:underline" aria-label={`Reload scenario ${strOf(s?.title || "")}`}>Load</button>
              {strOf(s?.id) && <button onClick={() => deleteSavedScenario(strOf(s.id))} className="text-[11px] font-medium text-muted-foreground hover:text-rose-600 dark:hover:text-rose-400" aria-label={`Delete scenario ${strOf(s?.title || "")}`}>Delete</button>}
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

/* =====================================================================
   REPORTS
   ===================================================================== */

const FALLBACK_SECTIONS = ["Executive Summary", "Key Metrics", "Analysis & Charts", "Insights", "Hypotheses", "Forecast", "Goals", "What-If Scenarios", "Recommendations"];

function ReportsTab({ meta }: { meta: any }) {
  const [templates, setTemplates] = useState<any[]>([]);
  const [tplLoading, setTplLoading] = useState(true);
  const [tplError, setTplError] = useState("");
  const [templateId, setTemplateId] = useState("");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [dateRange, setDateRange] = useState("");
  const [sections, setSections] = useState<string[]>([]);
  const [genBusy, setGenBusy] = useState(false);
  const [genError, setGenError] = useState("");
  const [genOk, setGenOk] = useState("");
  const [actMsg, setActMsg] = useState("");
  const [reports, setReports] = useState<any[]>([]);
  const [repLoading, setRepLoading] = useState(false);
  const [repError, setRepError] = useState("");
  const [repBusyId, setRepBusyId] = useState<string | null>(null);
  /* Scheduled digests (Phase 4): per-user daily/weekly configs with
     enable/disable, last-run status, one-click re-run, View/Download via
     the existing report HTML path. */
  const [digests, setDigests] = useState<any[]>([]);
  const [digLoading, setDigLoading] = useState(false);
  const [digError, setDigError] = useState("");
  const [digName, setDigName] = useState("");
  const [digFreq, setDigFreq] = useState("daily");
  const [digBusy, setDigBusy] = useState(false);
  const [digBusyId, setDigBusyId] = useState<string | null>(null);
  /* Live preview data: latest verified outputs (analysis + hypotheses + goals). */
  const [pvAnalysis, setPvAnalysis] = useState<any>(null);
  const [pvHyps, setPvHyps] = useState<any[]>([]);
  const [pvGoals, setPvGoals] = useState<any[]>([]);
  const [pvScenarios, setPvScenarios] = useState<any[]>([]);
  /* Working report draft (hypothesis-to-report): [{ id, title }]. */
  const [draftItems, setDraftItems] = useState<ReportDraftItem[]>(() => loadReportDraft());

  const reloadDraft = useCallback(() => setDraftItems(loadReportDraft()), []);
  useEffect(() => { reloadDraft(); }, [reloadDraft]);

  const loadPreviewData = useCallback(async () => {
    try {
      const [a, h, g, s] = await Promise.all([
        intelligenceApi.listAnalyses({ page: 1, page_size: 1, sort: "recent" }).catch(() => ({ data: [] as any[] })),
        intelligenceApi.listHypotheses({ page: 1, page_size: 3 }).catch(() => ({ data: [] as any[] })),
        intelligenceApi.listGoals({ status: "all" }).catch(() => ({ data: [] as any[] })),
        intelligenceApi.listScenarios({ page_size: 5 }).catch(() => ({ data: [] as any[] })),
      ]);
      const al = asList((a as any)?.data);
      setPvAnalysis(al[0] || null);
      setPvHyps(asList((h as any)?.data).slice(0, 3));
      setPvGoals(asList((g as any)?.data).slice(0, 3));
      setPvScenarios(asList((s as any)?.data).slice(0, 5));
    } catch { /* preview stays outline-only */ }
  }, []);
  useEffect(() => { loadPreviewData(); }, [loadPreviewData]);

  const downloadHtml = () => {
    const secs = sections.length ? sections : FALLBACK_SECTIONS;
    const esc = (s: string) => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const goalsRows = pvGoals.map((g: any) => {
      const cur = numOf(g?.current ?? g?.current_value);
      const tgt = numOf(g?.target ?? g?.target_value);
      const pct = numOf(g?.progress_pct ?? g?.progress);
      const prog = pct !== null
        ? `<div style="background:#e2e8f0;border-radius:4px;height:6px;width:160px"><div style="background:#10b981;height:6px;border-radius:4px;width:${Math.max(0, Math.min(100, pct))}%"></div></div>`
        : "";
      return `<li>${esc(strOf(g?.title || g?.name || "Goal"))} — ${esc(strOf(g?.status_label || g?.status || ""))}${cur !== null && tgt !== null ? ` (${esc(fmtBiz(cur))} / ${esc(fmtBiz(tgt))}${pct !== null ? `, ${pct}%` : ""})` : ""}${prog}</li>`;
    }).join("");
    const hypRows = pvHyps.map((x: any) => `<li>${esc(strOf(x?.title || x?.question || ""))}</li>`).join("");
    const draftRows = draftItems.map((d) => `<li>${esc(d.title)}</li>`).join("");
    const scenRows = pvScenarios.map((s: any) => `<li>${esc(strOf(s?.title || s?.name || ""))}</li>`).join("");
    const html = `<!doctype html><html><head><meta charset="utf-8"><title>${esc(name || "ARIA Business Review")}</title></head><body style="font-family:sans-serif;max-width:720px;margin:32px auto;color:#0F2440"><p style="color:#1d4ed8;font-weight:700;letter-spacing:2px">ARIA</p><h1>${esc(name || "Business Review")}</h1><p>Insight. Analysis. Action.</p>${description ? `<p>${esc(description)}</p>` : ""}<p>Prepared ${esc(dateRange || "—")} · ${esc(strOf(activeTemplate?.name || templateId || ""))}</p><h2>Sections</h2><ul>${secs.map((s) => `<li>${esc(s)}</li>`).join("")}</ul>${pvAnalysis ? `<h2>Latest analysis</h2><p><strong>${esc(strOf(pvAnalysis?.title || ""))}</strong> — ${esc(strOf(pvAnalysis?.result_summary || pvAnalysis?.summary || ""))}</p>` : ""}${hypRows ? `<h2>Hypotheses</h2><ul>${hypRows}</ul>` : ""}${draftRows ? `<h2>Report draft hypotheses</h2><ul>${draftRows}</ul>` : ""}${goalsRows ? `<h2>Goals</h2><ul>${goalsRows}</ul>` : ""}${scenRows ? `<h2>What-if scenarios</h2><ul>${scenRows}</ul>` : ""}<p style="color:#64748b;font-size:12px">Reports use verified outputs from your analyses, insights, forecasts and goals.</p></body></html>`;
    try {
      const blob = new Blob([html], { type: "text/html" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${(name || "aria-report").replace(/[^a-z0-9-_]+/gi, "-").toLowerCase()}.html`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 2000);
    } catch { /* ignore */ }
  };

  const loadReports = useCallback(async () => {
    setRepLoading(true);
    try {
      const r = await intelligenceApi.listReports();
      setReports(asList(r.data));
      setRepError("");
    } catch (e: any) {
      console.warn("Failed to load reports:", e?.message || e);
      setRepError(businessError(e?.message) || "ARIA couldn't load reports.");
    }
    finally { setRepLoading(false); }
  }, []);
  useEffect(() => { loadReports(); }, [loadReports]);

  /* ---- Scheduled digests (Phase 4) ---- */
  const loadDigests = useCallback(async () => {
    setDigLoading(true);
    try {
      const r: any = await apiFetch("/intelligence/digests", { method: "GET" });
      setDigests(asList(r?.data ?? r?.items));
      setDigError("");
    } catch (e: any) {
      setDigError(businessError(e?.message) || "ARIA couldn't load digest schedules.");
    } finally {
      setDigLoading(false);
    }
  }, []);
  useEffect(() => { loadDigests(); }, [loadDigests]);

  const createDigest = async () => {
    if (digBusy) return;
    setDigBusy(true);
    setDigError("");
    setActMsg("");
    try {
      await apiFetch("/intelligence/digests", {
        method: "POST",
        body: JSON.stringify({
          name: digName.trim() || (digFreq === "weekly" ? "Weekly Digest" : "Daily Digest"),
          frequency: digFreq,
          template: templateId || "professional",
          sections: sections.length ? sections : undefined,
        }),
      });
      setDigName("");
      setActMsg("Digest scheduled — it runs in the background and each run is kept in report history.");
      loadDigests();
    } catch (e: any) {
      setDigError(businessError(e?.message) || "ARIA couldn't schedule this digest.");
    } finally {
      setDigBusy(false);
    }
  };

  const toggleDigest = async (d: any) => {
    const id = strOf(d?.id);
    if (!id || digBusyId) return;
    setDigBusyId(id);
    try {
      await apiFetch(`/intelligence/digests/${encodeURIComponent(id)}`, {
        method: "PATCH",
        body: JSON.stringify({ enabled: !d?.enabled }),
      });
      loadDigests();
    } catch (e: any) {
      setDigError(businessError(e?.message) || "ARIA couldn't update this digest.");
    } finally {
      setDigBusyId(null);
    }
  };

  const runDigestNow = async (d: any) => {
    const id = strOf(d?.id);
    if (!id || digBusyId) return;
    setDigBusyId(id);
    setActMsg("");
    try {
      await apiFetch(`/intelligence/digests/${encodeURIComponent(id)}/run`, { method: "POST" });
      setActMsg("Digest ran with fresh numbers — see the newest report below.");
      loadDigests();
      loadReports();
    } catch (e: any) {
      setActMsg(businessError(e?.message) || "ARIA couldn't run this digest — the failed run was recorded.");
      loadDigests();
    } finally {
      setDigBusyId(null);
    }
  };

  const deleteDigest = async (d: any) => {
    const id = strOf(d?.id);
    if (!id || digBusyId) return;
    setDigBusyId(id);
    try {
      await apiFetch(`/intelligence/digests/${encodeURIComponent(id)}`, { method: "DELETE" });
      loadDigests();
    } catch (e: any) {
      setDigError(businessError(e?.message) || "ARIA couldn't delete this digest.");
    } finally {
      setDigBusyId(null);
    }
  };

  const viewDigestHtml = (reportId: string) => {
    if (!reportId) return;
    try {
      window.open(`/api/intelligence/reports/${encodeURIComponent(reportId)}/html`, "_blank", "noopener");
    } catch { /* ignore */ }
  };

  const digestStatusTone = (s: string) => {
    const t = String(s || "").toLowerCase();
    if (t === "ok" || t === "success" || t === "generated") return "bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-500/25";
    if (t === "failed" || t === "error") return "bg-rose-50 dark:bg-rose-500/10 text-rose-700 dark:text-rose-400 border-rose-200 dark:border-rose-500/25";
    return "bg-secondary text-secondary-foreground border-border";
  };

  const deleteReport = async (id: string) => {
    if (!id) return;
    try {
      await intelligenceApi.deleteReport(id);
      loadReports();
    } catch (e: any) {
      setActMsg(businessError(e?.message) || "ARIA couldn't delete this report.");
    }
  };

  /* Server-rendered HTML (Reporting Agent): styled report with findings,
     charts/tables, evidence SQL, generated-at + fingerprint. Download the
     HTML; print-to-PDF from the browser for a PDF copy. */
  const downloadServerHtml = async (id: string, fallbackName: string) => {
    if (!id || repBusyId) return;
    setRepBusyId(id);
    setActMsg("");
    try {
      const r = await intelligenceApi.getReportHtml(id);
      const html = strOf((r as any)?.data ?? r);
      if (!html) throw new Error("empty report");
      const blob = new Blob([html], { type: "text/html" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${(fallbackName || "aria-report").replace(/[^a-z0-9-_]+/gi, "-").toLowerCase()}.html`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 2000);
      setActMsg("Report downloaded as HTML — open it and Print → Save as PDF for a PDF copy.");
    } catch (e: any) {
      setActMsg(businessError(e?.message) || "ARIA couldn't download this report.");
    } finally {
      setRepBusyId(null);
    }
  };

  /* Re-run: same definition, fresh numbers (new history entry, per user). */
  const rerunOne = async (id: string) => {
    if (!id || repBusyId) return;
    setRepBusyId(id);
    setActMsg("");
    try {
      await intelligenceApi.rerunReport(id);
      setActMsg("Report re-ran with fresh numbers — see the newest entry.");
      loadReports();
    } catch (e: any) {
      setActMsg(businessError(e?.message) || "ARIA couldn't re-run this report.");
    } finally {
      setRepBusyId(null);
    }
  };

  const viewReport = async (id: string) => {
    if (!id) return;
    try {
      const r = await intelligenceApi.getReport(id);
      const d: any = r.data || {};
      if (strOf(d?.name)) setName(strOf(d.name));
      if (strOf(d?.description)) setDescription(strOf(d.description));
      if (strOf(d?.date_range)) setDateRange(strOf(d.date_range));
      const secs = asList(d?.sections).map((s) => strOf(typeof s === "string" ? s : s?.name || s?.label)).filter(Boolean);
      if (secs.length) setSections(secs);
      if (strOf(d?.template)) setTemplateId(strOf(d.template));
      setGenOk(`Loaded report ${strOf(d?.name || "")}. Review and Generate PDF to re-export.`);
    } catch (e: any) {
      setGenError(businessError(e?.message) || "ARIA couldn't open this report.");
    }
  };

  const dateRanges: string[] = useMemo(() => {
    const d = meta?.date_ranges ?? meta?.dateRanges ?? [];
    return asList(d).map((x) => strOf(typeof x === "string" ? x : x?.name || x?.label)).filter(Boolean);
  }, [meta]);

  const loadTemplates = useCallback(async () => {
    setTplLoading(true);
    setTplError("");
    try {
      const r = await intelligenceApi.listReportTemplates();
      const list = asList(r.data);
      setTemplates(list);
      if (list.length && !templateId) {
        const first = list[0];
        setTemplateId(strOf(first?.id || first?.name));
        const secs = asList(first?.sections).map((s) => strOf(typeof s === "string" ? s : s?.name || s?.label)).filter(Boolean);
        if (secs.length) setSections(secs);
      }
    } catch (e: any) {
      setTplError(businessError(e?.message) || "ARIA couldn't load report templates.");
    } finally {
      setTplLoading(false);
    }
  }, [templateId]);
  useEffect(() => { loadTemplates(); }, [loadTemplates]);

  useEffect(() => {
    if (!dateRange && dateRanges.length) setDateRange(dateRanges[0]);
    if (sections.length === 0) setSections(FALLBACK_SECTIONS.slice(0, 9));
  }, [dateRanges, dateRange, sections.length]);

  const activeTemplate = templates.find((t) => strOf(t?.id || t?.name) === templateId) || null;
  const toggleSection = (s: string) => setSections((prev) => (prev.includes(s) ? prev.filter((x) => x !== s) : [...prev, s]));

  const pickTemplate = (id: string) => {
    setTemplateId(id);
    const t = templates.find((x) => strOf(x?.id || x?.name) === id);
    const secs = asList(t?.sections).map((s) => strOf(typeof s === "string" ? s : s?.name || s?.label)).filter(Boolean);
    if (secs.length) setSections(secs);
  };

  const generatePdf = async () => {
    setGenBusy(true);
    setGenError("");
    setGenOk("");
    try {
      /* Working-draft hypotheses ride along as items; the backend resolves
         them against live hypotheses into a custom section. */
      await intelligenceApi.generateReport({
        name: name || undefined, description: description || undefined,
        date_range: dateRange || undefined, sections,
        template: templateId || undefined,
        ...(draftItems.length ? { items: draftItems } : {}),
      });
      if (draftItems.length) {
        clearReportDraft();
        setDraftItems([]);
      }
      setGenOk("Report generated. Opening print view — choose Save as PDF.");
      loadReports();
      setTimeout(() => { try { window.print(); } catch { /* ignore */ } }, 400);
    } catch (e: any) {
      setGenError(businessError(e?.message) || "ARIA couldn't generate this report.");
    } finally {
      setGenBusy(false);
    }
  };

  const reportAction = async (kind: "template" | "schedule" | "share" | "ppt") => {
    setActMsg("");
    try {
      if (kind === "template") {
        await intelligenceApi.saveReportTemplate({ name: name || "Untitled template", sections, template: templateId || undefined });
        setActMsg("Saved as a template.");
        loadTemplates();
      } else if (kind === "schedule") {
        await apiFetch("/intelligence/reports/schedule", { method: "POST", body: JSON.stringify({ name, date_range: dateRange, sections, template: templateId }) });
        setActMsg("Report scheduled.");
      } else if (kind === "share") {
        await apiFetch("/intelligence/reports/share", { method: "POST", body: JSON.stringify({ name, date_range: dateRange, sections, template: templateId }) });
        setActMsg("Share link created.");
      } else {
        await apiFetch("/intelligence/reports/export", { method: "POST", body: JSON.stringify({ format: "pptx", name, date_range: dateRange, sections, template: templateId }) });
        setActMsg("PowerPoint export started — check your downloads shortly.");
      }
    } catch (e: any) {
      setActMsg(businessError(e?.message) || "That action isn't available right now.");
    }
  };

  const hasDatasetRep = dateRanges.length > 0 || templates.length > 0;
  const noSchemaRep = getNoSchemaInfo(meta);
  const scrollPreview = () => { try { document.getElementById("intel-report-preview")?.scrollIntoView({ behavior: "smooth", block: "start" }); } catch {} };
  return (
    <div className="space-y-4">
      <div className="text-[11px] text-muted-foreground flex items-center gap-1">
        <span>Intelligence</span><ChevronRight size={11} /><span className="text-secondary-foreground font-medium">Reports</span>
      </div>
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div>
          <h3 className="text-[16px] font-semibold text-foreground tracking-tight">Report Builder</h3>
          <p className="text-[12px] text-muted-foreground">Create professional reports from your analysis.</p>
        </div>
        <div className="flex items-center gap-1.5">
          <button onClick={scrollPreview} className={btnGhost} aria-label="Preview report"><Eye size={13} /> Preview</button>
          <button onClick={generatePdf} disabled={genBusy} className={btnPrimary} aria-label="Generate PDF top"><Printer size={13} /> {genBusy ? "Generating..." : "Generate PDF"}</button>
        </div>
      </div>
      {!hasDatasetRep && !tplLoading && <NoDatasetPanel hint={noSchemaRep.isNoSchema ? noSchemaRep.hint : undefined} />}
      {draftItems.length > 0 && (
        <Card className="p-4 md:p-5 !border-blue-200 dark:border-blue-500/25 bg-blue-50/40 dark:bg-blue-500/10">
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <p className="text-[12.5px] font-semibold text-foreground">
              Draft includes {draftItems.length} item{draftItems.length === 1 ? "" : "s"} — added from Hypotheses. Generate includes them in the report sections.
            </p>
            <button
              onClick={() => { clearReportDraft(); setDraftItems([]); }}
              className="text-[11.5px] font-semibold text-muted-foreground hover:text-rose-600 dark:hover:text-rose-400 hover:underline"
              aria-label="Clear report draft">
              Clear draft
            </button>
          </div>
          <ul className="mt-2 space-y-1">
            {draftItems.map((d) => (
              <li key={d.id || d.title} className="flex items-center gap-2 text-[12px] text-secondary-foreground">
                <span className="w-1.5 h-1.5 rounded-full bg-blue-600 flex-shrink-0" />
                <span className="flex-1 truncate">{d.title}</span>
                <button
                  onClick={() => setDraftItems(removeFromReportDraft(d.id || d.title))}
                  className="text-[11px] font-medium text-muted-foreground hover:text-rose-600 dark:hover:text-rose-400 hover:underline flex-shrink-0"
                  aria-label={`Remove from draft: ${d.title}`}>
                  Remove
                </button>
              </li>
            ))}
          </ul>
        </Card>
      )}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card className="p-4 md:p-5">
          <h3 className="text-[14px] font-semibold text-foreground tracking-tight">Report Details</h3>
          <div className="mt-3 space-y-3">
            <Field label="Report Name" htmlFor="rep-name">
              <input id="rep-name" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Q2 Business Review" className={inputCls} aria-label="Report name" />
            </Field>
            <Field label="Description (optional)" htmlFor="rep-desc">
              <input id="rep-desc" value={description} onChange={(e) => setDescription(e.target.value)} placeholder="What this report covers…" className={inputCls} aria-label="Report description" />
            </Field>
            <Field label="Date Range" htmlFor="rep-range">
              <select id="rep-range" value={dateRange} onChange={(e) => setDateRange(e.target.value)} className={selectCls} aria-label="Report date range">
                {dateRanges.length === 0 && <option value="">Select…</option>}
                {dateRanges.map((d) => <option key={d} value={d}>{humanize(d)}</option>)}
              </select>
            </Field>
            <fieldset>
              <legend className="text-[11px] font-medium text-muted-foreground mb-1.5">Include Sections</legend>
              <div className="space-y-1.5">
                {(activeTemplate && asList(activeTemplate?.sections).length
                  ? asList(activeTemplate.sections).map((s) => strOf(typeof s === "string" ? s : s?.name || s?.label)).filter(Boolean)
                  : [...new Set([...sections, ...FALLBACK_SECTIONS])].slice(0, 9)
                ).map((s) => (
                  <label key={s} className="flex items-center gap-2 text-[12.5px] text-foreground cursor-pointer">
                    <input type="checkbox" checked={sections.includes(s)} onChange={() => toggleSection(s)} className="accent-blue-700" aria-label={`Include ${s}`} />
                    {s}
                  </label>
                ))}
              </div>
            </fieldset>
          </div>
        </Card>

        <Card className="p-4 md:p-5">
          <div className="flex items-center justify-between gap-2">
            <h3 className="text-[14px] font-semibold text-foreground tracking-tight">Report Preview</h3>
            <span className="text-[11px] text-muted-foreground">{sections.length > 0 ? `1 / ${sections.length}` : "No sections"}</span>
          </div>
          <div id="intel-report-preview" className="mt-3 rounded-xl border border-border bg-secondary/60 p-5 min-h-64" aria-label="Report preview">
            <p className="text-[11px] font-bold tracking-[0.14em] text-blue-700 dark:text-blue-400">ARIA</p>
            <h4 className="text-[17px] font-semibold text-foreground tracking-tight mt-1">{name || "Business Review"}</h4>
            <p className="text-[11px] text-muted-foreground mt-0.5">Insight. Analysis. Action.</p>
            {description && <p className="text-[12px] text-secondary-foreground mt-2 leading-relaxed">{description}</p>}
            <div className="mt-3 space-y-1.5">
              {sections.length === 0 && <p className="text-[11px] text-muted-foreground">Select sections on the left to preview the outline.</p>}
              {sections.map((s) => (
                <div key={s} className="flex items-center gap-2 text-[12px] text-secondary-foreground">
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-600" />{s}
                </div>
              ))}
            </div>
            {(pvAnalysis || pvHyps.length > 0 || pvGoals.length > 0 || pvScenarios.length > 0 || draftItems.length > 0) && (
              <div className="mt-3 border-t border-border pt-3 space-y-2">
                {pvAnalysis && (
                  <div>
                    <p className="text-[11px] font-semibold text-foreground">Latest analysis</p>
                    <p className="text-[11.5px] text-secondary-foreground leading-relaxed">{strOf(pvAnalysis?.title || pvAnalysis?.name || "")}{strOf(pvAnalysis?.result_summary || pvAnalysis?.summary) ? ` — ${strOf(pvAnalysis.result_summary || pvAnalysis.summary).slice(0, 160)}` : ""}</p>
                  </div>
                )}
                {pvHyps.length > 0 && (
                  <div>
                    <p className="text-[11px] font-semibold text-foreground">Hypotheses</p>
                    {pvHyps.map((x: any, i: number) => (
                      <p key={i} className="text-[11.5px] text-secondary-foreground leading-relaxed">· {strOf(x?.title || x?.question || "")}</p>
                    ))}
                  </div>
                )}
                {draftItems.length > 0 && (
                  <div>
                    <p className="text-[11px] font-semibold text-foreground">Report draft ({draftItems.length})</p>
                    {draftItems.slice(0, 5).map((d, i: number) => (
                      <p key={i} className="text-[11.5px] text-secondary-foreground leading-relaxed">· {d.title}</p>
                    ))}
                  </div>
                )}
                {pvGoals.length > 0 && (
                  <div>
                    <p className="text-[11px] font-semibold text-foreground">Goals</p>
                    {pvGoals.map((g: any, i: number) => {
                      const cur = numOf(g?.current ?? g?.current_value);
                      const tgt = numOf(g?.target ?? g?.target_value);
                      const pct = numOf(g?.progress_pct ?? g?.progress);
                      const status = strOf(g?.status_label || g?.status || "");
                      return (
                        <div key={i} className="mt-1.5">
                          <p className="text-[11.5px] text-secondary-foreground leading-relaxed">
                            · {strOf(g?.title || g?.name || "Goal")} — {status}
                            {cur !== null && tgt !== null ? ` (${fmtBiz(cur)} / ${fmtBiz(tgt)}${pct !== null ? `, ${pct}%` : ""})` : ""}
                          </p>
                          {pct !== null && (
                            <div className="h-1.5 rounded-full bg-border/70 overflow-hidden max-w-52 mt-1" role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100} aria-label={`Progress for ${strOf(g?.title)}`}>
                              <div className={`h-full rounded-full ${/behind|at risk|off track/i.test(status) ? "bg-rose-500" : /near|watch/i.test(status) ? "bg-amber-500" : "bg-emerald-500"}`} style={{ width: `${Math.max(0, Math.min(100, pct))}%` }} />
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
                {pvScenarios.length > 0 && (
                  <div>
                    <p className="text-[11px] font-semibold text-foreground">What-if scenarios</p>
                    {pvScenarios.map((s: any, i: number) => (
                      <p key={i} className="text-[11.5px] text-secondary-foreground leading-relaxed">
                        · {strOf(s?.title || s?.name || `${humanize(strOf(s?.variable))} scenario`)}{numOf(s?.change_value) !== null ? ` (${numOf(s.change_value)}%)` : ""}
                      </p>
                    ))}
                  </div>
                )}
              </div>
            )}
            <p className="mt-4 text-[10.5px] text-muted-foreground">Prepared {dateRange ? `for ${dateRange}` : "—"} · {activeTemplate ? humanize(strOf(activeTemplate?.name || templateId)) : "No template"}</p>
          </div>
          {genError && <p className="mt-2 text-[12px] text-rose-600 dark:text-rose-400" role="alert">{genError}</p>}
          {genOk && <p className="mt-2 text-[12px] text-emerald-700 dark:text-emerald-400" role="status">{genOk}</p>}
          <button onClick={generatePdf} disabled={genBusy} className={`${btnPrimary} w-full mt-3`} aria-label="Generate PDF">
            <Printer size={13} /> {genBusy ? "Generating…" : "Generate PDF"}
          </button>
          <button onClick={downloadHtml} className={`${btnGhost} w-full mt-2 justify-center`} aria-label="Download printable HTML">
            <Download size={13} /> Download HTML
          </button>
        </Card>

        <div className="space-y-4">
          <Card className="p-4 md:p-5">
            <h3 className="text-[14px] font-semibold text-foreground tracking-tight">Template</h3>
            {tplLoading && <div className="skeleton h-9 rounded-lg mt-2" aria-hidden />}
            {!tplLoading && tplError && <p className="text-[12px] text-rose-600 dark:text-rose-400 mt-2" role="alert">{tplError}</p>}
            {!tplLoading && !tplError && (
              <div className="mt-2 space-y-1.5" role="radiogroup" aria-label="Report template">
                {templates.length === 0 && (
                  <div className="mt-2 space-y-1.5">
                    {[["Professional", "Clean and modern"], ["Executive", "Focused on key insights"], ["Detailed", "Comprehensive analysis"]].map(([n, d]) => (
                      <button key={n} onClick={() => { setTemplateId(n); }} className={`w-full text-left rounded-xl border p-2.5 transition-colors ${templateId === n ? "border-blue-500 bg-blue-50/60 dark:bg-blue-500/10" : "border-border hover:border-border"}`}>
                        <span className="block text-[12.5px] font-semibold text-foreground">{n}</span>
                        <span className="block text-[11px] text-muted-foreground mt-0.5">{d}</span>
                      </button>
                    ))}
                  </div>
                )}
                {templates.map((t: any) => {
                  const id = strOf(t?.id || t?.name);
                  const on = templateId === id;
                  return (
                    <button key={id} role="radio" aria-checked={on} onClick={() => pickTemplate(id)}
                      className={`w-full text-left rounded-xl border p-2.5 transition-colors ${on ? "border-blue-500 bg-blue-50/60 dark:bg-blue-500/10" : "border-border hover:border-border"}`}>
                      <span className="block text-[12.5px] font-semibold text-foreground">{strOf(t?.name || id)}</span>
                      <span className="block text-[11px] text-muted-foreground mt-0.5">{strOf(t?.description || "")}</span>
                    </button>
                  );
                })}
              </div>
            )}
          </Card>
          <Card className="p-4 md:p-5">
            <h3 className="text-[14px] font-semibold text-foreground tracking-tight">Report Actions</h3>
            <div className="mt-2 space-y-1.5">
              {[
                { k: "template" as const, label: "Save as Template", Icon: Save },
                { k: "schedule" as const, label: "Schedule Report", Icon: CalendarDays },
                { k: "share" as const, label: "Share Report", Icon: Share2 },
                { k: "ppt" as const, label: "Export to PowerPoint", Icon: Download },
              ].map(({ k, label, Icon }) => (
                <button key={k} onClick={() => reportAction(k)} className="w-full flex items-center gap-2 text-left text-[12.5px] font-medium text-foreground hover:text-blue-700 dark:hover:text-blue-400 rounded-lg px-2 py-1.5 hover:bg-blue-50/60 dark:hover:bg-blue-500/10 transition-colors" aria-label={label}>
                  <Icon size={13} className="text-muted-foreground" />{label}
                </button>
              ))}
            </div>
            {actMsg && <p className="mt-2 text-[11.5px] text-secondary-foreground" role="status">{actMsg}</p>}
            <p className="mt-2 text-[11px] text-muted-foreground">Reports use verified outputs from your analyses, insights, forecasts and goals.</p>
          </Card>
        </div>
      </div>
      <Card className="p-4 md:p-5">
        <CardHead title="Scheduled Digests" sub="Daily or weekly background runs — each run is kept in report history." right={digLoading ? <span className="text-[11px] text-muted-foreground">Loading…</span> : <button onClick={loadDigests} className="text-[11px] font-semibold text-blue-700 dark:text-blue-400 hover:underline" aria-label="Refresh digests">Refresh</button>} />
        <div className="mt-3 flex items-center gap-2 flex-wrap">
          <label htmlFor="digest-name" className="sr-only">Digest name</label>
          <input id="digest-name" value={digName} onChange={(e) => setDigName(e.target.value)} placeholder={digFreq === "weekly" ? "e.g. Weekly Digest" : "e.g. Daily Digest"} className={`${inputCls} !w-52`} aria-label="Digest name" />
          <label htmlFor="digest-freq" className="sr-only">Digest frequency</label>
          <select id="digest-freq" value={digFreq} onChange={(e) => setDigFreq(e.target.value)} className={`${selectCls} !w-auto`} aria-label="Digest frequency">
            <option value="daily">Daily</option>
            <option value="weekly">Weekly</option>
          </select>
          <button onClick={createDigest} disabled={digBusy} className={btnPrimary} aria-label="Schedule digest">
            <CalendarDays size={13} /> {digBusy ? "Scheduling…" : "Schedule digest"}
          </button>
        </div>
        {digError && <p className="mt-2 text-[12px] text-rose-600 dark:text-rose-400" role="alert">{digError} <button onClick={loadDigests} className="font-semibold text-blue-700 dark:text-blue-400 hover:underline ml-1">Retry</button></p>}
        <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2">
          {!digError && digests.length === 0 && !digLoading && <p className="text-[12px] text-muted-foreground">No digests yet — schedule a daily or weekly digest above.</p>}
          {digests.map((d: any, i: number) => {
            const did = strOf(d?.id || i);
            const lastReportId = strOf(d?.last_report_id || "");
            return (
              <div key={did} className="rounded-xl border border-border p-3 flex items-center gap-2">
                <CalendarDays size={14} className="text-blue-600 dark:text-blue-400 flex-shrink-0" />
                <div className="min-w-0 flex-1">
                  <p className="text-[12.5px] font-medium text-foreground truncate">{strOf(d?.name || "Digest")} <span className="text-muted-foreground font-normal">· {strOf(d?.frequency || "daily")}</span></p>
                  <p className="text-[11px] text-muted-foreground">
                    {d?.last_run_at ? `Last run ${strOf(d.last_run_at).slice(0, 16).replace("T", " ")}` : "Never run yet"}
                    {d?.enabled ? "" : " · paused"}
                  </p>
                  <span className={`inline-block mt-1 text-[10.5px] font-semibold px-2 py-0.5 rounded-full border ${digestStatusTone(strOf(d?.last_status || "never"))}`}>
                    {strOf(d?.last_status || "never") === "never" ? "Not run yet" : strOf(d?.last_status)}
                  </span>
                  {strOf(d?.last_error) && <p className="text-[11px] text-rose-600 dark:text-rose-400 mt-1 truncate" title={strOf(d.last_error)}>{strOf(d.last_error)}</p>}
                </div>
                <div className="flex flex-col items-end gap-1 flex-shrink-0">
                  <button onClick={() => runDigestNow(d)} disabled={digBusyId === did} className="text-[11px] font-semibold text-blue-700 dark:text-blue-400 hover:underline disabled:opacity-40" aria-label={`Run digest now ${strOf(d?.name || "")}`}>{digBusyId === did ? "Running…" : "Run now"}</button>
                  <button onClick={() => toggleDigest(d)} disabled={digBusyId === did} className="text-[11px] font-medium text-muted-foreground hover:text-blue-700 dark:hover:text-blue-400 hover:underline disabled:opacity-40" aria-label={d?.enabled ? `Pause digest ${strOf(d?.name || "")}` : `Enable digest ${strOf(d?.name || "")}`}>{d?.enabled ? "Pause" : "Enable"}</button>
                  {lastReportId && <button onClick={() => viewDigestHtml(lastReportId)} className="text-[11px] font-semibold text-blue-700 dark:text-blue-400 hover:underline" aria-label={`View last digest report ${strOf(d?.name || "")}`}>View</button>}
                  {lastReportId && <button onClick={() => downloadServerHtml(lastReportId, strOf(d?.name || "digest"))} disabled={repBusyId === lastReportId} className="text-[11px] font-medium text-muted-foreground hover:text-blue-700 dark:hover:text-blue-400 hover:underline disabled:opacity-40" aria-label={`Download last digest report ${strOf(d?.name || "")}`}>Download</button>}
                  <button onClick={() => deleteDigest(d)} disabled={digBusyId === did} className="text-[11px] font-medium text-muted-foreground hover:text-rose-600 dark:hover:text-rose-400 disabled:opacity-40" aria-label={`Delete digest ${strOf(d?.name || "")}`}>Delete</button>
                </div>
              </div>
            );
          })}
        </div>
      </Card>
      <Card className="p-4 md:p-5">
        <CardHead title="Generated Reports" sub="Every report you generate is kept here." right={repLoading ? <span className="text-[11px] text-muted-foreground">Loading…</span> : <button onClick={loadReports} className="text-[11px] font-semibold text-blue-700 dark:text-blue-400 hover:underline" aria-label="Refresh reports">Refresh</button>} />
        {repError && <p className="mt-2 text-[12px] text-rose-600 dark:text-rose-400" role="alert">{repError} <button onClick={loadReports} className="font-semibold text-blue-700 dark:text-blue-400 hover:underline ml-1">Retry</button></p>}
        <div className="mt-2 grid grid-cols-1 sm:grid-cols-2 gap-2">
          {!repError && reports.length === 0 && <p className="text-[12px] text-muted-foreground">No reports yet — fill the details and choose Generate PDF.</p>}
          {reports.map((r: any, i: number) => (
            <div key={strOf(r?.id || i)} className="rounded-xl border border-border p-3 flex items-center gap-2">
              <FileText size={14} className="text-blue-600 dark:text-blue-400 flex-shrink-0" />
              <div className="min-w-0 flex-1">
                <p className="text-[12.5px] font-medium text-foreground truncate">{strOf(r?.name || "Report")}</p>
                <p className="text-[11px] text-muted-foreground">{strOf(r?.generated_at || r?.created_at || "")}{asList(r?.sections).length ? ` · ${asList(r.sections).length} sections` : ""}</p>
              </div>
              <button onClick={() => viewReport(strOf(r?.id))} disabled={!r?.id} className="text-[11px] font-semibold text-blue-700 dark:text-blue-400 hover:underline disabled:opacity-40" aria-label={`View report ${strOf(r?.name || "")}`}>View</button>
              {strOf(r?.id) && <button onClick={() => downloadServerHtml(strOf(r.id), strOf(r?.name || "aria-report"))} disabled={repBusyId === strOf(r.id)} className="text-[11px] font-semibold text-blue-700 dark:text-blue-400 hover:underline disabled:opacity-40" aria-label={`Download report HTML ${strOf(r?.name || "")}`}>{repBusyId === strOf(r.id) ? "Working…" : "Download"}</button>}
              {strOf(r?.id) && <button onClick={() => rerunOne(strOf(r.id))} disabled={repBusyId === strOf(r.id)} className="text-[11px] font-medium text-muted-foreground hover:text-blue-700 dark:hover:text-blue-400 hover:underline disabled:opacity-40" aria-label={`Re-run report ${strOf(r?.name || "")}`}>Re-run</button>}
              {strOf(r?.id) && <button onClick={() => deleteReport(strOf(r.id))} className="text-[11px] font-medium text-muted-foreground hover:text-rose-600 dark:hover:text-rose-400" aria-label={`Delete report ${strOf(r?.name || "")}`}>Delete</button>}
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

/* =====================================================================
   PAGE
   ===================================================================== */

export default function Intelligence({
  connectionStatus = "not-connected",
  onAskCopilot,
}: {
  connectionStatus?: string;
  onAskCopilot?: (question: string) => void;
}) {
  const theme = useTheme() as any;
  const setPage = theme?.setPage;
  const [tab, setTab] = useState<IntelTab>(() => tabFromHash());
  const [meta, setMeta] = useState<any>(null);
  const [metaLoading, setMetaLoading] = useState(true);
  const [metaError, setMetaError] = useState("");
  const [metaId, setMetaId] = useState("");
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);

  const loadMeta = useCallback(async () => {
    setMetaLoading(true);
    setMetaError("");
    setMetaId("");
    try {
      const r = await intelligenceApi.getMeta();
      setMeta(r.data || null);
    } catch (e: any) {
      const norm = normalizeApiError({ message: e?.message }, e?.status || 0, null);
      setMetaError(norm.message);
      setMetaId(norm.requestId);
      setMeta(null);
    } finally {
      setMetaLoading(false);
    }
  }, []);
  useEffect(() => { if (connectionStatus === "connected") loadMeta(); else { setMeta(null); setMetaLoading(false); } }, [connectionStatus, loadMeta]);

  /* Deep-link arrivals (Dashboard / bot link straight to a case):
     #/intelligence?tab=prescriptions&rx=<id> switches tabs without reload. */
  useEffect(() => {
    const onHash = () => {
      try {
        const t = tabFromHash();
        setTab((cur) => (cur === t ? cur : t));
      } catch { /* ignore */ }
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const goto = (t: IntelTab, extra?: Record<string, string | null | undefined>) => {
    setTab(t);
    try {
      const base = (window.location.hash || "#/intelligence").split("?")[0];
      const qs = Object.entries(extra || {})
        .filter(([, v]) => v !== null && v !== undefined && String(v).trim() !== "")
        .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
        .join("&");
      window.location.hash = qs ? `${base}?tab=${t}&${qs}` : `${base}?tab=${t}`;
    } catch { /* ignore */ }
  };

  const ask = (q: string) => {
    const question = (q || "").trim();
    if (onAskCopilot) { onAskCopilot(question); return; }
    try {
      sessionStorage.setItem("aria_copilot_context", JSON.stringify({ origin: "intelligence", question }));
    } catch { /* best-effort */ }
    if (setPage) setPage("copilot");
  };

  const quickPrefill = (_actionId: string) => { /* quick actions drive Explore defaults in place */ };

  const onTabKey = (e: React.KeyboardEvent, idx: number) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft" && e.key !== "Home" && e.key !== "End") return;
    e.preventDefault();
    let next = idx;
    if (e.key === "ArrowRight") next = (idx + 1) % TABS.length;
    if (e.key === "ArrowLeft") next = (idx - 1 + TABS.length) % TABS.length;
    if (e.key === "Home") next = 0;
    if (e.key === "End") next = TABS.length - 1;
    goto(TABS[next].id);
    tabRefs.current[next]?.focus();
  };

  if (connectionStatus !== "connected") {
    return (
      <div className="min-h-full bg-card text-foreground">
        <div className="p-6 max-w-screen-xl mx-auto">
          <p className="text-[11px] font-bold tracking-[0.14em] text-blue-700 dark:text-blue-400">ARIA INTELLIGENCE</p>
          <h2 className="text-xl font-semibold tracking-tight">ARIA Intelligence</h2>
          <p className="text-sm text-muted-foreground mt-0.5">Explore beyond the dashboard.</p>
          <div className="mt-8 text-center py-12 bg-card border border-border rounded-2xl shadow-sm">
            <Database size={22} className="mx-auto text-muted-foreground mb-3" />
            <p className="text-sm font-medium">No data source connected</p>
            <p className="text-xs text-muted-foreground mt-1">Connect PostgreSQL, MySQL, or upload a CSV to start analyzing with Intelligence.</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-full bg-card text-foreground">
      <div className="p-4 md:p-6 max-w-screen-xl mx-auto">
        {/* Header — gradient banner */}
        <div className="relative overflow-hidden rounded-2xl border border-border/80 bg-gradient-to-br from-blue-50 dark:from-blue-500/10 via-card to-indigo-50/60 dark:to-indigo-500/10 px-5 py-5 md:px-6">
          <div className="intel-hero-gradient" aria-hidden />
          <div className="intel-orb intel-orb-subtle-1" aria-hidden />
          <div className="intel-orb intel-orb-subtle-2" aria-hidden />
          <div className="intel-grid-overlay" aria-hidden />
          <div className="relative flex items-start justify-between gap-4 flex-wrap">
            <div className="min-w-0">
              <p className="text-[11px] font-bold tracking-[0.14em] text-blue-700 dark:text-blue-400">ARIA INTELLIGENCE</p>
              <h2 className="text-[22px] font-bold tracking-tight">ARIA Intelligence</h2>
              <p className="text-[14px] font-medium mt-0.5">Explore beyond the dashboard.</p>
              <p className="text-[12px] text-muted-foreground mt-0.5 max-w-xl">Investigate patterns, compare performance, test scenarios, and build business analysis from your connected data.</p>
            </div>
            <p className="hidden md:block text-[12px] text-muted-foreground italic text-right leading-relaxed">
              &ldquo;Better questions.<br />Deeper understanding.<br />Stronger decisions.&rdquo;
            </p>
          </div>
        </div>

        {/* Tab bar */}
        <div className="mt-4 flex items-center gap-1 border-b border-border overflow-x-auto" role="tablist" aria-label="Intelligence sections">
          {TABS.map(({ id, label, Icon }, i) => {
            const on = tab === id;
            return (
              <button
                key={id}
                ref={(el) => { tabRefs.current[i] = el; }}
                role="tab"
                aria-selected={on}
                aria-controls={`intel-panel-${id}`}
                id={`intel-tab-${id}`}
                tabIndex={on ? 0 : -1}
                onClick={() => goto(id)}
                onKeyDown={(e) => onTabKey(e, i)}
                className={`inline-flex items-center gap-1.5 px-3.5 py-2.5 text-[12.5px] font-medium border-b-2 -mb-px transition-colors whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${on ? "border-blue-700 text-blue-800 dark:text-blue-300" : "border-transparent text-muted-foreground hover:text-secondary-foreground"}`}
              >
                <Icon size={14} /> {label}
              </button>
            );
          })}
        </div>

        {/* Meta-level failure still allows tab browsing of cached/empty states,
            but Explore/Goals/What-If/Reports need meta options to function. */}
        {metaLoading && (
          <div className="mt-4" aria-busy="true" aria-label="Loading intelligence options">
            <p className="sr-only">Loading intelligence options…</p>
            <Card><SkeletonBlock lines={2} /></Card>
          </div>
        )}
        {!metaLoading && metaError && (tab === "explore" || tab === "goals" || tab === "whatif" || tab === "reports") && !meta && (
          <div className="mt-4">
            <ErrorPanel message={metaError} requestId={metaId} onRetry={loadMeta} retryLabel="Reload options" />
          </div>
        )}

        <div id={`intel-panel-${tab}`} role="tabpanel" aria-labelledby={`intel-tab-${tab}`} className="mt-4">
          <MotionConfig reducedMotion="user">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div
                key={tab}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.22, ease: "easeOut" }}
              >
                {tab === "explore" && <ExploreTab meta={meta || {}} onRunQuick={quickPrefill} onGoto={goto} onAsk={ask} />}
                {tab === "hypotheses" && <HypothesesTab onAsk={ask} onGoto={goto} />}
                {tab === "goals" && <GoalsTab meta={meta || {}} />}
                {tab === "prescriptions" && <PrescriptionsFeed />}
                {tab === "whatif" && <WhatIfTab />}
                {tab === "reports" && <ReportsTab meta={meta || {}} />}
              </motion.div>
            </AnimatePresence>
          </MotionConfig>
        </div>
      </div>
    </div>
  );
}
