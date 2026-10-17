/* =====================================================================
   Shared cross-module analysis context (AI Analytics <-> Forecast <->
   Insights -> Decision Center, chart drill-downs).
   - Stored in localStorage under `aria_analysis_context` so context
     survives reloads and hash navigation; per-handoff prefills also ride
     in the URL hash (e.g. #/predictive?metric=X) for deep-linking.
   - Shape is dynamic: metric/dataset/dateRange/filters/dimension/table
     are whatever the source analysis produced — nothing hardcoded per
     dataset. Missing fields stay null, never invented.
   - NEVER forces re-analysis: consumers prefill selectors from context
     but only fetch when they have no usable result yet. Context never
     clears or invalidates another page's results.
   - Legacy sessionStorage prefill keys (aria_forecast_prefill,
     aria_insight_handoff, aria_hypothesis_prefill, aria_copilot_context)
     are still written for backward compatibility and read as fallback.
   ===================================================================== */

export const ANALYSIS_CONTEXT_KEY = "aria_analysis_context";

export interface AnalysisContext {
  origin?: string;
  question?: string | null;
  metric?: string | null;
  dataset?: string | null;
  table?: string | null;
  dateRange?: string | null;
  period?: string | null;
  dimension?: string | null;
  driver?: string | null;
  segment?: string | null;
  filters?: Record<string, string> | null;
  source?: { tables?: string[]; columns?: string[] } | null;
  updatedAt?: string | null;
}

const pick = (v: any): string | null => {
  if (v === null || v === undefined) return null;
  if (typeof v === "object") return null;
  const s = String(v).trim();
  return s ? s : null;
};

export function saveAnalysisContext(patch: Partial<AnalysisContext>): AnalysisContext {
  let prev: Partial<AnalysisContext> = {};
  try {
    const raw = localStorage.getItem(ANALYSIS_CONTEXT_KEY);
    if (raw) prev = JSON.parse(raw) || {};
  } catch {
    prev = {};
  }
  const next: AnalysisContext = {
    ...prev,
    ...Object.fromEntries(
      Object.entries(patch).filter(([, v]) => v !== undefined),
    ),
    updatedAt: new Date().toISOString(),
  };
  try {
    localStorage.setItem(ANALYSIS_CONTEXT_KEY, JSON.stringify(next));
  } catch {
    /* storage full/blocked — context is best-effort */
  }
  // Mirror the focused metric into legacy prefill keys so older readers
  // (and in-flight sessions) keep working during the migration.
  try {
    if (next.metric) {
      sessionStorage.setItem(
        "aria_forecast_prefill",
        JSON.stringify({ metric: next.metric, period: next.period || next.dateRange || undefined }),
      );
    }
  } catch {
    /* ignore */
  }
  return next;
}

export function loadAnalysisContext(): AnalysisContext | null {
  try {
    const raw = localStorage.getItem(ANALYSIS_CONTEXT_KEY);
    if (!raw) return null;
    const ctx = JSON.parse(raw);
    if (!ctx || typeof ctx !== "object") return null;
    return ctx as AnalysisContext;
  } catch {
    return null;
  }
}

/** Read shared context first, fall back to a legacy sessionStorage prefill. */
export function consumeHandoff(legacyKey: string): AnalysisContext | null {
  const shared = loadAnalysisContext();
  let legacy: Record<string, any> | null = null;
  try {
    const raw = sessionStorage.getItem(legacyKey);
    if (raw) {
      legacy = JSON.parse(raw);
      sessionStorage.removeItem(legacyKey);
    }
  } catch {
    legacy = null;
  }
  if (!shared && !legacy) return null;
  const merged: AnalysisContext = {
    origin: (shared?.origin || (legacy as any)?.origin || legacyKey || "handoff") as string,
    question: shared?.question ?? pick((legacy as any)?.question),
    metric:
      shared?.metric ??
      pick((legacy as any)?.metric) ??
      pick(typeof (legacy as any)?.question === "string" ? (legacy as any).question : null),
    dataset: shared?.dataset ?? pick((legacy as any)?.dataset) ?? null,
    table: shared?.table ?? (Array.isArray((legacy as any)?.source?.tables) ? (legacy as any).source.tables[0] : null) ?? null,
    dateRange: shared?.dateRange ?? pick((legacy as any)?.dateRange) ?? null,
    period: shared?.period ?? pick((legacy as any)?.period) ?? null,
    dimension:
      shared?.dimension ??
      (Array.isArray((legacy as any)?.dimensions) ? pick((legacy as any).dimensions[0]) : null) ??
      pick((legacy as any)?.segment) ??
      null,
    driver: shared?.driver ?? pick((legacy as any)?.driver) ?? null,
    segment: shared?.segment ?? pick((legacy as any)?.segment) ?? null,
    filters: shared?.filters ?? ((legacy as any)?.filters && typeof (legacy as any).filters === "object" ? (legacy as any).filters : null),
    source: shared?.source ?? ((legacy as any)?.source && typeof (legacy as any).source === "object" ? (legacy as any).source : null),
    updatedAt: shared?.updatedAt || null,
  };
  return merged;
}

/** Navigate to a page carrying the shared context via hash params
 *  (deep-linkable) without clearing any page's existing results.
 *  Extra params (metric/table/period/...) travel in the hash query string. */
export function gotoWithContext(page: string, params?: Record<string, string | null | undefined>) {
  const qs = Object.entries(params || {})
    .filter(([, v]) => v !== null && v !== undefined && String(v).trim() !== "")
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
    .join("&");
  try {
    window.location.hash = qs ? `#/${page}?${qs}` : `#/${page}`;
  } catch {
    /* ignore */
  }
}

/** Read hash query params (e.g. #/predictive?metric=X&period=Y). */
export function readHashParams(): Record<string, string> {
  try {
    const h = window.location.hash || "";
    const q = h.includes("?") ? h.slice(h.indexOf("?") + 1) : "";
    const out: Record<string, string> = {};
    for (const part of q.split("&")) {
      if (!part || !part.includes("=")) continue;
      const [k, v] = part.split("=");
      out[decodeURIComponent(k)] = decodeURIComponent(v || "");
    }
    return out;
  } catch {
    return {};
  }
}
