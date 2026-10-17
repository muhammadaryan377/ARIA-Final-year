import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle, Bookmark, CalendarDays, Check, ChevronRight,
  RefreshCw, Shield, Sparkles, User, X,
} from "lucide-react";
import { prescriptionApi } from "../../../services/api";
import {
  asList, strOf, numOf, humanize, fmtMetric, businessError,
} from "../../lib/intelHelpers";
import {
  Card, EmptyState, LoadingPanel, ErrorPanel, btnGhost, btnPrimary,
} from "./cards";
import RxCaseDrawer, { type RxCaseInput } from "./RxCaseDrawer";

/* =====================================================================
   Prescriptions auto feed — "What needs attention".
   Zero clicks: on mount it reads the live prescription context
   (GET /api/prescriptions/context), takes the top 3 opportunities, and
   runs each through the existing engine (POST /api/prescriptions/analyze)
   in the background. Cached cards render instantly; the background run
   refreshes them. The frontend never invents actions or numbers — every
   headline, action, outcome and figure comes from the backend display
   contract (display.* + impact.*). Plain words only, no raw table.column.
   ===================================================================== */

const CACHE_KEY = "aria_rx_feed_cache_v1";
const CACHE_TTL_MS = 30 * 60 * 1000;
const MAX_CARDS = 3;
const MAX_ACTIONS_PER_CARD = 3;

type FeedCard = {
  key: string;
  metric: string;
  driver: string;
  table: string;
  options: any[];
  status: "ready" | "refused" | "error";
  refusalHeadline: string;
  refusalAlternatives: any[];
  error: string;
  verdict: any | null;
};

type RecheckState = {
  state: "checking" | "done" | "error";
  at: number;
  fresh: string;
};

function readRxFromHash(): string | null {
  try {
    const m = /[?&]rx=([^&]+)/.exec(window.location.hash || "");
    return m?.[1] ? decodeURIComponent(m[1]) : null;
  } catch {
    return null;
  }
}

function timeAgo(at: number | null): string {
  if (!at) return "never";
  const s = Math.max(0, Math.round((Date.now() - at) / 1000));
  if (s < 60) return "just now";
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h} hr ago`;
  return `${Math.round(h / 24)} day${Math.round(h / 24) === 1 ? "" : "s"} ago`;
}

function todayISO(): string {
  try {
    return new Date().toISOString().slice(0, 10);
  } catch {
    return "";
  }
}

function isPastDue(due: string): boolean {
  const d = strOf(due).slice(0, 10);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(d)) return false;
  const t = todayISO();
  return !!t && d < t;
}

/* Plain-words headline for a card: what slowed/dropped + segment + size.
   Backend problem_text first; legacy title/summary as fallback. Never a
   raw table.column — metric/driver names are humanized words only. */
function headlineOf(p: any, metric: string, driver: string): string {
  const disp = p?.display && typeof p.display === "object" ? p.display : {};
  const s = (v: unknown) => strOf(v).trim();
  const problem = s(disp.problem_text) || s(p?.problem_text)
    || s(p?.summary_plain) || s(p?.story) || s(p?.summary);
  if (problem) return problem;
  const title = s(disp.title) || s(p?.title);
  if (title) return title;
  const m = humanize(metric || "this measure");
  const d = humanize(driver || "");
  return d ? `Something to look at for ${m.toLowerCase()} around ${d.toLowerCase()}.`
    : `Something to look at for ${m.toLowerCase()}.`;
}

function confidenceDots(conf: unknown): { filled: number; label: string } {
  const s = strOf(conf).toLowerCase();
  if (/high|strong/.test(s)) return { filled: 3, label: strOf(conf) || "High" };
  if (/moderate|medium/.test(s)) return { filled: 2, label: strOf(conf) || "Moderate" };
  return { filled: 1, label: strOf(conf) || "Early signal" };
}

function impactOf(p: any): { baseline: number | null; scenario: number | null; delta: number | null } {
  const imp = p?.impact && typeof p.impact === "object" ? p.impact : {};
  return {
    baseline: numOf(imp?.baseline),
    scenario: numOf(imp?.scenario),
    delta: numOf(imp?.delta ?? imp?.rel_pct),
  };
}

function compactNum(v: number): string {
  if (Math.abs(v) >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
  if (Math.abs(v) >= 1_000) return `${(v / 1_000).toFixed(0)}K`;
  return strOf(Math.round(v * 10) / 10);
}

/* Mini baseline-vs-scenario bars: plain words, backend numbers only. */
function MiniBars({ baseline, scenario, metricLabel }: { baseline: number | null; scenario: number | null; metricLabel: string }) {
  if (baseline === null || scenario === null) return null;
  const max = Math.max(baseline, scenario, 0.0001);
  const rows: { label: string; value: number; cls: string }[] = [
    { label: "Where things stand", value: baseline, cls: "bg-slate-300" },
    { label: "If the action works", value: scenario, cls: scenario >= baseline ? "bg-emerald-500" : "bg-amber-500" },
  ];
  return (
    <div className="mt-2.5 rounded-xl bg-secondary border border-border px-3 py-2.5" role="img"
      aria-label={`Where things stand ${fmtMetric(baseline, metricLabel)} versus ${fmtMetric(scenario, metricLabel)} if the action works`}>
      {rows.map((r) => (
        <div key={r.label} className="flex items-center gap-2 py-1">
          <span className="w-28 flex-shrink-0 text-[11px] text-muted-foreground">{r.label}</span>
          <div className="flex-1 h-2 rounded-full bg-border/70 overflow-hidden">
            <div className={`h-full rounded-full ${r.cls}`} style={{ width: `${Math.max(3, Math.min(100, (r.value / max) * 100))}%` }} />
          </div>
          <span className="w-16 flex-shrink-0 text-right text-[11px] font-semibold text-foreground tabular-nums">{compactNum(r.value)}</span>
        </div>
      ))}
    </div>
  );
}

export default function PrescriptionsFeed() {
  const [ctxLoading, setCtxLoading] = useState(true);
  const [ctxError, setCtxError] = useState("");
  const [refreshing, setRefreshing] = useState(false);
  const [cards, setCards] = useState<FeedCard[]>([]);
  const [cacheAt, setCacheAt] = useState<number | null>(null);
  const [watchedMetrics, setWatchedMetrics] = useState<string[]>([]);
  const [watchedGroups, setWatchedGroups] = useState<string[]>([]);
  const [limitations, setLimitations] = useState<string[]>([]);
  const [noSource, setNoSource] = useState(false);

  const [saved, setSaved] = useState<any[]>([]);
  const [savedLoading, setSavedLoading] = useState(true);
  const [recheck, setRecheck] = useState<Record<string, RecheckState>>({});
  const [selectedId, setSelectedId] = useState<string | null>(() => readRxFromHash());
  const [drafts, setDrafts] = useState<Record<string, { owner: string; due: string }>>({});
  const [busySave, setBusySave] = useState<string | null>(null);
  const [saveFlash, setSaveFlash] = useState<Record<string, string>>({});
  const [patchBusy, setPatchBusy] = useState<string | null>(null);
  const autoRechecked = useRef<Set<string>>(new Set());

  /* Deep-link arrivals after mount (Dashboard / bot links). */
  useEffect(() => {
    const onHash = () => setSelectedId(readRxFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const openCase = useCallback((id: string) => {
    setSelectedId(id);
    try {
      const base = (window.location.hash || "#/intelligence").split("?")[0];
      window.location.hash = `${base}?tab=prescriptions&rx=${encodeURIComponent(id)}`;
    } catch { /* best-effort */ }
  }, []);

  const closeCase = useCallback(() => {
    setSelectedId(null);
    try {
      const base = (window.location.hash || "#/intelligence").split("?")[0];
      window.location.hash = `${base}?tab=prescriptions`;
    } catch { /* best-effort */ }
  }, []);

  /* Scroll a deep-linked card into view. */
  useEffect(() => {
    if (!selectedId) return;
    try {
      const el = document.getElementById(`rx-card-${selectedId}`);
      if (el) el.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch { /* ignore */ }
  }, [selectedId, cards.length]);

  const loadSaved = useCallback(async () => {
    setSavedLoading(true);
    try {
      const r = await prescriptionApi.listSaved();
      setSaved(asList((r as any)?.data?.saved ?? (r as any)?.data?.items));
    } catch {
      setSaved([]);
    } finally {
      setSavedLoading(false);
    }
  }, []);

  const recheckOne = useCallback(async (item: any) => {
    const id = strOf(item?.id);
    const metric = strOf(item?.metric);
    if (!id || !metric) return;
    setRecheck((p) => ({ ...p, [id]: { state: "checking", at: Date.now(), fresh: "" } }));
    try {
      const payload: Record<string, unknown> = { metric };
      if (strOf(item?.driver)) payload.driver = strOf(item.driver);
      const r = await prescriptionApi.analyze(payload);
      const d: any = (r as any)?.data ?? null;
      const top = asList(d?.prescriptions)[0];
      const fresh = strOf(
        top?.display?.outcome_text || top?.display?.action_text
        || d?.refusal?.headline || "",
      );
      setRecheck((p) => ({
        ...p,
        [id]: {
          state: "done", at: Date.now(),
          fresh: fresh || "Re-checked — still watching this one.",
        },
      }));
    } catch {
      setRecheck((p) => ({ ...p, [id]: { state: "error", at: Date.now(), fresh: "" } }));
    }
  }, []);

  /* Auto re-measure: past-due tracked items without a review outcome are
     re-run through the existing analyze endpoint in the background. */
  useEffect(() => {
    if (savedLoading || !saved.length) return;
    for (const s of saved.slice(0, 6)) {
      const id = strOf(s?.id);
      const outcome = strOf(s?.review_outcome).toLowerCase();
      if (!id || outcome || autoRechecked.current.has(id)) continue;
      if (!isPastDue(strOf(s?.due))) continue;
      autoRechecked.current.add(id);
      void recheckOne(s);
    }
  }, [saved, savedLoading, recheckOne]);

  const buildFeed = useCallback(async (opts?: { background?: boolean }) => {
    const background = !!opts?.background;
    if (!background) {
      setCtxLoading(true);
      setCtxError("");
    } else {
      setRefreshing(true);
    }
    try {
      const ctx = await prescriptionApi.context();
      const d: any = (ctx as any)?.data ?? {};
      if (d?.ok === false || d?.code === "NO_CONNECTION") {
        setNoSource(true);
        if (!background) setCards([]);
        setWatchedMetrics([]);
        setWatchedGroups([]);
        setLimitations([]);
        return;
      }
      setNoSource(false);
      const opps = asList(d?.opportunities).slice(0, MAX_CARDS);
      const mets: string[] = asList(d?.metrics)
        .map((m: any) => strOf(typeof m === "string" ? m : m?.name || m?.label || m?.column))
        .filter(Boolean).slice(0, 6);
      const dims: string[] = asList(d?.dimensions)
        .map((x: any) => strOf(typeof x === "string" ? x : x?.name || x?.label || x?.column))
        .filter(Boolean).slice(0, 6);
      setWatchedMetrics(mets);
      setWatchedGroups(dims);
      setLimitations(
        asList(d?.table_limitations ?? d?.limitations)
          .map((x: any) => strOf(typeof x === "string" ? x : x?.text)).filter(Boolean).slice(0, 4),
      );

      const next: FeedCard[] = await Promise.all(opps.map(async (o: any, i: number) => {
        const metric = strOf(o?.metric);
        const driver = strOf(o?.driver);
        const table = strOf(o?.table);
        const key = `${strOf(o?.metric)}|${strOf(o?.driver)}|${strOf(o?.table)}|${i}`;
        if (!metric) {
          return {
            key, metric, driver, table, options: [], status: "error" as const,
            refusalHeadline: "", refusalAlternatives: [], error: "This opening had no measure attached.",
            verdict: null,
          };
        }
        try {
          const payload: Record<string, unknown> = { metric };
          if (driver) payload.driver = driver;
          if (table) payload.table = table;
          const r = await prescriptionApi.analyze(payload);
          const out: any = (r as any)?.data ?? null;
          const list = asList(out?.prescriptions);
          if (!out || out.ok === false || out.status === "no_prescription" || !list.length) {
            const refusal = out?.refusal && typeof out.refusal === "object" ? out.refusal : null;
            return {
              key, metric, driver, table, options: [], status: "refused" as const,
              refusalHeadline: strOf(refusal?.headline || "Nothing here cleared the bar for an action — ARIA would rather say so than guess."),
              refusalAlternatives: asList(refusal?.alternatives).slice(0, 3),
              error: "",
              verdict: (out as any)?.verdict || null,
            };
          }
          const withIds = list.slice(0, MAX_ACTIONS_PER_CARD).map((p: any, j: number) => ({
            ...p,
            id: strOf(p?.id) || `rx-${metric}-${driver}-${j}`.replace(/[^a-zA-Z0-9-_]+/g, "-"),
          }));
          return {
            key: strOf(withIds[0]?.id) || key,
            metric, driver, table, options: withIds, status: "ready" as const,
            refusalHeadline: "", refusalAlternatives: [], error: "",
            verdict: (out as any)?.verdict || null,
          };
        } catch (e: any) {
          return {
            key, metric, driver, table, options: [], status: "error" as const,
            refusalHeadline: "", refusalAlternatives: [],
            error: businessError(e?.message) || "ARIA couldn't check this one just now.",
            verdict: null,
          };
        }
      }));

      setCards(next);
      const at = Date.now();
      setCacheAt(at);
      try {
        localStorage.setItem(CACHE_KEY, JSON.stringify({ at, cards: next }));
      } catch { /* best-effort */ }
    } catch (e: any) {
      if (!background && !cards.length) {
        setCtxError(businessError(e?.message) || "ARIA couldn't load what needs attention.");
      }
    } finally {
      setCtxLoading(false);
      setRefreshing(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* Zero-click start: cached cards paint instantly, then the background
     run refreshes from live context + engine. */
  useEffect(() => {
    try {
      const raw = localStorage.getItem(CACHE_KEY);
      if (raw) {
        const c = JSON.parse(raw);
        if (Array.isArray(c?.cards) && c.cards.length && Date.now() - Number(c.at || 0) < CACHE_TTL_MS) {
          setCards(c.cards);
          setCacheAt(Number(c.at));
        }
      }
    } catch { /* ignore */ }
    void buildFeed({ background: false });
    void loadSaved();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const refreshAll = useCallback(() => {
    autoRechecked.current = new Set();
    void buildFeed({ background: true });
    void loadSaved();
  }, [buildFeed, loadSaved]);

  const draftOf = (key: string) => drafts[key] || { owner: "", due: "" };
  const setDraft = (key: string, patch: Partial<{ owner: string; due: string }>) =>
    setDrafts((p) => ({ ...p, [key]: { ...draftOf(key), ...patch } }));

  const saveAction = useCallback(async (card: FeedCard, prescription: any, idx: number) => {
    const actionKey = `${card.key}:${idx}`;
    if (busySave) return;
    setBusySave(actionKey);
    setSaveFlash((p) => ({ ...p, [actionKey]: "" }));
    try {
      const headline = headlineOf(prescription, card.metric, card.driver);
      const r = await prescriptionApi.save(prescription, headline);
      const entry = asList((r as any)?.data?.saved)[0];
      const savedId = strOf(entry?.id);
      const { owner, due } = draftOf(actionKey);
      if (savedId && (owner.trim() || due)) {
        try {
          await prescriptionApi.updateSaved(savedId, {
            ...(owner.trim() ? { owner: owner.trim() } : {}),
            ...(due ? { due } : {}),
          });
        } catch { /* save itself succeeded; owner/due can be set later */ }
      }
      setSaveFlash((p) => ({ ...p, [actionKey]: "Tracked — see Tracked actions below." }));
      await loadSaved();
    } catch (e: any) {
      setSaveFlash((p) => ({ ...p, [actionKey]: businessError(e?.message) || "ARIA couldn't track this action." }));
    } finally {
      setBusySave(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [busySave, drafts, loadSaved]);

  const patchSaved = useCallback(async (id: string, patch: Record<string, string>) => {
    if (!id || patchBusy) return;
    setPatchBusy(id);
    try {
      await prescriptionApi.updateSaved(id, patch);
      await loadSaved();
    } catch (e: any) {
      setSaveFlash((p) => ({ ...p, [`saved:${id}`]: businessError(e?.message) || "ARIA couldn't update this item." }));
    } finally {
      setPatchBusy(null);
    }
  }, [patchBusy, loadSaved]);

  const unsave = useCallback(async (id: string) => {
    if (!id) return;
    try {
      await prescriptionApi.unsave(id);
      await loadSaved();
    } catch (e: any) {
      setSaveFlash((p) => ({ ...p, [`saved:${id}`]: businessError(e?.message) || "ARIA couldn't remove this item." }));
    }
  }, [loadSaved]);

  /* Deep-link target: a feed card or a tracked item. */
  const selectedInput: RxCaseInput | null = useMemo(() => {
    if (!selectedId) return null;
    const card = cards.find((c) => c.key === selectedId || c.options.some((o) => strOf(o?.id) === selectedId));
    if (card) {
      const p = card.options.find((o) => strOf(o?.id) === selectedId) || card.options[0];
      if (!p) return null;
      return {
        key: strOf(p?.id) || card.key,
        headline: headlineOf(p, card.metric, card.driver),
        metricLabel: humanize(card.metric),
        prescription: p,
        owner: "", due: "", outcome: "", savedId: null, recheckNote: "",
      };
    }
    const s = saved.find((x) => strOf(x?.id) === selectedId);
    if (!s) return null;
    const disp = s?.display && typeof s.display === "object" ? s.display : {};
    const rc = recheck[strOf(s?.id)];
    return {
      key: strOf(s?.id),
      headline: strOf(disp.problem_text || disp.title || s?.title || "Tracked action"),
      metricLabel: humanize(strOf(s?.metric)),
      prescription: s,
      owner: strOf(s?.owner), due: strOf(s?.due), outcome: strOf(s?.review_outcome),
      savedId: strOf(s?.id),
      recheckNote: rc?.state === "done" ? `Re-checked ${timeAgo(rc.at)} — ${rc.fresh}` : "",
    };
  }, [selectedId, cards, saved, recheck]);

  const readyCards = cards.filter((c) => c.status === "ready" && c.options.length);
  const showInitialLoading = ctxLoading && !cards.length;

  return (
    <div className="space-y-4">
      <div className="text-[11px] text-muted-foreground flex items-center gap-1">
        <span>Intelligence</span><ChevronRight size={11} /><span className="text-secondary-foreground font-medium">Prescriptions</span>
      </div>
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div>
          <h3 className="text-[16px] font-semibold text-foreground tracking-tight flex items-center gap-1.5">
            <Sparkles size={15} className="text-blue-700 dark:text-blue-400" /> What needs attention
          </h3>
          <p className="text-[12px] text-muted-foreground">
            Checked automatically from your live data
            {cacheAt ? ` · last check ${timeAgo(cacheAt)}` : ""} · checks again on your next visit.
          </p>
        </div>
        <button onClick={refreshAll} disabled={refreshing || ctxLoading} className={btnGhost} aria-label="Check again now">
          <RefreshCw size={13} className={refreshing ? "animate-spin" : ""} /> {refreshing ? "Checking…" : "Check again"}
        </button>
      </div>

      {showInitialLoading && <LoadingPanel label="Checking what needs attention…" />}
      {!showInitialLoading && ctxError && !cards.length && (
        <ErrorPanel message={ctxError} onRetry={() => void buildFeed()} />
      )}
      {!showInitialLoading && !ctxError && (noSource || (!readyCards.length && !refreshing)) && (
        <EmptyState
          icon={Sparkles}
          title={noSource ? "No data source connected yet." : "Nothing needs attention right now."}
          body={
            noSource
              ? "Connect your data and ARIA will watch it for slowdowns and drops worth acting on."
              : `ARIA is watching ${watchedMetrics.length ? watchedMetrics.map((m) => humanize(m)).join(", ") : "your measures"}${watchedGroups.length ? ` across ${watchedGroups.map((g) => humanize(g)).join(", ")}` : ""}. It checks again on your next visit — or press Check again any time.`
          }
          action={!noSource ? (
            <button onClick={refreshAll} className="text-xs font-semibold text-blue-700 dark:text-blue-400 hover:underline">Check again now</button>
          ) : undefined}
        />
      )}

      {readyCards.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
          {readyCards.map((card, ci) => {
            const primary = card.options[0];
            const disp = primary?.display && typeof primary.display === "object" ? primary.display : {};
            const headline = headlineOf(primary, card.metric, card.driver);
            const actionText = strOf(disp.action_text);
            const outcomeText = strOf(disp.outcome_text);
            const conf = confidenceDots(disp.confidence_label || primary?.confidence_label || primary?.confidence);
            const imp = impactOf(primary);
            return (
              <Card
                key={card.key}
                className={`p-4 flex flex-col ${selectedId === card.key ? "!border-blue-400 ring-1 ring-blue-200 dark:ring-blue-500/25" : "hover:border-blue-200 dark:hover:border-blue-500/25"} transition-colors`}
              >
                <div id={`rx-card-${card.key}`}>
                  <div className="flex items-center gap-2">
                    <span className="text-[10.5px] font-bold text-blue-700 dark:text-blue-400 bg-blue-50 dark:bg-blue-500/10 border border-blue-100 dark:border-blue-500/20 rounded-full px-2 py-0.5">
                      #{ci + 1} to look at
                    </span>
                    {(() => {
                      const v: any = (card as any)?.verdict;
                      if (!v) return null;
                      const ok = v.verified === true;
                      const fails = Array.isArray(v.checks) ? v.checks.filter((c: any) => !c?.pass).length : 0;
                      return (
                        <span
                          title={ok
                            ? "ARIA double-checked this recommendation against your data. Everything lines up."
                            : "ARIA couldn't fully double-check this recommendation yet. Treat it as a guide, not the final answer."}
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
                    <span className="inline-flex items-center gap-1 ml-auto" title={`Confidence: ${conf.label}`}>
                      {[0, 1, 2].map((dot) => (
                        <span key={dot} className={`inline-block w-1.5 h-1.5 rounded-full ${dot < conf.filled ? "bg-blue-600" : "bg-border"}`} />
                      ))}
                    </span>
                  </div>
                  <p className="mt-2 text-[13.5px] font-semibold text-foreground leading-snug">{headline}</p>
                  <p className="mt-1 text-[11px] text-muted-foreground">
                    {humanize(card.metric)}{card.driver ? ` · lever: ${humanize(card.driver)}` : ""}
                  </p>
                  <MiniBars baseline={imp.baseline} scenario={imp.scenario} metricLabel={card.metric} />
                  {card.options.slice(0, MAX_ACTIONS_PER_CARD).map((p: any, idx: number) => {
                    const pd = p?.display && typeof p.display === "object" ? p.display : {};
                    const aKey = `${card.key}:${idx}`;
                    const d = draftOf(aKey);
                    const savingThis = busySave === aKey;
                    return (
                      <div key={strOf(p?.id) || idx} className={idx === 0 ? "mt-3 rounded-xl bg-blue-50/60 dark:bg-blue-500/10 border border-blue-100 dark:border-blue-500/20 px-3 py-2.5" : "mt-2 rounded-xl border border-border px-3 py-2.5"}>
                        {idx > 0 && <p className="text-[10.5px] font-semibold text-muted-foreground">ALSO WORTH CONSIDERING</p>}
                        <p className="text-[12.5px] font-semibold text-foreground leading-snug">
                          {strOf(pd.action_text) || (idx === 0 ? actionText : "Another way to respond.")}
                        </p>
                        {(strOf(pd.outcome_text) || (idx === 0 ? outcomeText : "")) && (
                          <p className="mt-1 text-[12px] text-secondary-foreground leading-relaxed">
                            Expected: {strOf(pd.outcome_text) || outcomeText}
                          </p>
                        )}
                        <div className="mt-2 flex items-center gap-1.5 flex-wrap">
                          <label className="inline-flex items-center gap-1 text-[11px] text-muted-foreground">
                            <User size={11} />
                            <span className="sr-only">Owner for this action</span>
                            <input
                              value={d.owner}
                              onChange={(e) => setDraft(aKey, { owner: e.target.value })}
                              placeholder="Owner"
                              className="w-24 bg-card border border-border rounded-md px-1.5 py-1 text-[11px] text-foreground placeholder:text-muted-foreground focus:border-blue-500 outline-none"
                              aria-label="Action owner"
                            />
                          </label>
                          <label className="inline-flex items-center gap-1 text-[11px] text-muted-foreground">
                            <CalendarDays size={11} />
                            <span className="sr-only">Review date for this action</span>
                            <input
                              type="date"
                              value={d.due}
                              onChange={(e) => setDraft(aKey, { due: e.target.value })}
                              className="bg-card border border-border rounded-md px-1.5 py-1 text-[11px] text-foreground focus:border-blue-500 outline-none"
                              aria-label="Review date"
                            />
                          </label>
                          <button
                            onClick={() => void saveAction(card, p, idx)}
                            disabled={savingThis}
                            className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-[11.5px] font-semibold text-white bg-blue-700 hover:bg-blue-800 disabled:opacity-50"
                            aria-label={`Track action: ${strOf(pd.action_text).slice(0, 60)}`}
                          >
                            <Bookmark size={11} /> {savingThis ? "…" : "Track"}
                          </button>
                        </div>
                        {saveFlash[aKey] && (
                          <p className="mt-1.5 text-[11px] text-emerald-700 dark:text-emerald-400" role="status">{saveFlash[aKey]}</p>
                        )}
                      </div>
                    );
                  })}
                  <button
                    onClick={() => openCase(card.key)}
                    className="mt-2.5 inline-flex items-center gap-1 text-[12px] font-semibold text-blue-700 dark:text-blue-400 hover:underline"
                    aria-label={`Open case: ${headline.slice(0, 60)}`}
                  >
                    Open case <ChevronRight size={12} />
                  </button>
                </div>
              </Card>
            );
          })}
        </div>
      )}

      {/* Tracked actions — saved items with auto re-measure + outcome badges */}
      <Card className="p-4 md:p-5">
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <div>
            <h4 className="text-[13px] font-semibold text-foreground">Tracked actions</h4>
            <p className="text-[11px] text-muted-foreground">Anything past its review date is re-checked automatically.</p>
          </div>
          <button onClick={() => void loadSaved()} className="text-[11px] font-semibold text-blue-700 dark:text-blue-400 hover:underline" aria-label="Reload tracked actions">
            Reload
          </button>
        </div>
        {savedLoading && <p className="mt-2 text-[12px] text-muted-foreground">Loading tracked actions…</p>}
        {!savedLoading && !saved.length && (
          <p className="mt-2 text-[12px] text-muted-foreground">Nothing tracked yet — press Track on any action above and it will appear here with its owner and review date.</p>
        )}
        {!savedLoading && saved.length > 0 && (
          <div className="mt-2 space-y-2">
            {saved.map((s: any) => {
              const id = strOf(s?.id);
              const disp = s?.display && typeof s.display === "object" ? s.display : {};
              const title = strOf(disp.problem_text || disp.title || s?.title || "Tracked action");
              const outcome = strOf(s?.review_outcome).toLowerCase();
              const badge = outcome === "achieved"
                ? <span className="inline-flex items-center gap-1 text-[10.5px] font-semibold px-2 py-0.5 rounded-full border bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-500/25"><Check size={10} /> Achieved</span>
                : outcome === "missed"
                  ? <span className="inline-flex items-center gap-1 text-[10.5px] font-semibold px-2 py-0.5 rounded-full border bg-rose-50 dark:bg-rose-500/10 text-rose-700 dark:text-rose-400 border-rose-200 dark:border-rose-500/25"><X size={10} /> Missed</span>
                  : outcome === "inconclusive"
                    ? <span className="text-[10.5px] font-semibold px-2 py-0.5 rounded-full border bg-secondary text-secondary-foreground border-border">Needs another look</span>
                    : isPastDue(strOf(s?.due))
                      ? <span className="inline-flex items-center gap-1 text-[10.5px] font-semibold px-2 py-0.5 rounded-full border bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-500/25"><AlertTriangle size={10} /> Past review date</span>
                      : null;
              const rc = recheck[id];
              const owner = strOf(s?.owner);
              const due = strOf(s?.due).slice(0, 10);
              return (
                <div key={id || title} id={`rx-card-${id}`} className="rounded-xl border border-border/80 p-3">
                  <div className="flex items-start gap-2 flex-wrap">
                    <p className="text-[12.5px] font-semibold text-foreground leading-snug flex-1 min-w-40">{title}</p>
                    {badge}
                  </div>
                  <p className="mt-1 text-[11px] text-muted-foreground flex items-center gap-2 flex-wrap">
                    {owner && <span className="inline-flex items-center gap-1"><User size={11} /> {owner}</span>}
                    {due && <span className="inline-flex items-center gap-1"><CalendarDays size={11} /> Review {due}</span>}
                    {!owner && !due && <span>No owner or review date yet — add them below.</span>}
                  </p>
                  {rc?.state === "checking" && <p className="mt-1 text-[11px] text-blue-700 dark:text-blue-400">Re-checking against live data…</p>}
                  {rc?.state === "done" && <p className="mt-1 text-[11px] text-secondary-foreground">Re-checked {timeAgo(rc.at)} — {rc.fresh}</p>}
                  <div className="mt-2 flex items-center gap-1.5 flex-wrap">
                    <button onClick={() => openCase(id)} className="text-[11.5px] font-semibold text-blue-700 dark:text-blue-400 hover:underline" aria-label={`Open tracked case ${title.slice(0, 40)}`}>Open case</button>
                    <span className="text-slate-200 dark:text-muted-foreground">|</span>
                    <button
                      onClick={() => void patchSaved(id, { review_outcome: "achieved" })}
                      disabled={patchBusy === id}
                      className="text-[11.5px] font-medium text-emerald-700 dark:text-emerald-400 hover:underline disabled:opacity-40"
                      aria-label={`Mark achieved: ${title.slice(0, 40)}`}
                    >Mark achieved</button>
                    <button
                      onClick={() => void patchSaved(id, { review_outcome: "missed" })}
                      disabled={patchBusy === id}
                      className="text-[11.5px] font-medium text-rose-600 dark:text-rose-400 hover:underline disabled:opacity-40"
                      aria-label={`Mark missed: ${title.slice(0, 40)}`}
                    >Mark missed</button>
                    <button
                      onClick={() => void recheckOne(s)}
                      disabled={rc?.state === "checking"}
                      className="inline-flex items-center gap-1 text-[11.5px] font-medium text-muted-foreground hover:text-blue-700 dark:hover:text-blue-400 hover:underline disabled:opacity-40"
                      aria-label={`Re-check now: ${title.slice(0, 40)}`}
                    ><RefreshCw size={11} /> Re-check</button>
                    <button
                      onClick={() => void unsave(id)}
                      className="text-[11.5px] font-medium text-muted-foreground hover:text-rose-600 dark:hover:text-rose-400 hover:underline"
                      aria-label={`Stop tracking: ${title.slice(0, 40)}`}
                    >Stop tracking</button>
                  </div>
                  {saveFlash[`saved:${id}`] && (
                    <p className="mt-1 text-[11px] text-rose-600 dark:text-rose-400" role="alert">{saveFlash[`saved:${id}`]}</p>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </Card>

      {limitations.length > 0 && (
        <p className="text-[11px] text-muted-foreground leading-relaxed">
          Coverage notes: {limitations.join(" · ")}
        </p>
      )}

      <RxCaseDrawer
        item={selectedInput}
        onClose={closeCase}
        onTrack={(prescription) => {
          const card = cards.find((c) => c.options.some((o) => o === prescription) || c.options[0] === prescription)
            || cards.find((c) => strOf(c.key) === selectedId);
          if (card) void saveAction(card, prescription, Math.max(0, card.options.indexOf(prescription)));
        }}
        onPatch={async (patch) => {
          if (selectedInput?.savedId) await patchSaved(selectedInput.savedId, patch);
        }}
        onRecheck={() => {
          const s = saved.find((x) => strOf(x?.id) === selectedId);
          if (s) void recheckOne(s);
        }}
        rechecking={selectedId ? recheck[selectedId]?.state === "checking" : false}
      />
    </div>
  );
}
