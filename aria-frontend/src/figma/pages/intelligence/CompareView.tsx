/* Compare saved analyses side-by-side.
   Picker lists saved analyses via intelligenceApi.listAnalyses (title /
   metric / breakdown / date); exactly 2 can be selected. Each column
   renders from the stored result payload when the item carries one, else
   the saved params are re-run through intelligenceApi.runExplore and the
   column is marked "Recomputed live". No mocks — every number comes from
   a stored result or a live re-run. */

import { useCallback, useEffect, useMemo, useState } from "react";
import { Check, X } from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from "recharts";
import { intelligenceApi, normalizeApiError } from "../../../services/api";
import {
  asList, strOf, numOf, humanize, titleCase, fmtMetric, businessError,
} from "../../lib/intelHelpers";
import { fmtPctBiz } from "../../lib/format";
import { Card, LoadingPanel, ErrorPanel, EmptyState } from "./cards";

interface Loaded {
  item: any;
  result: any;
  live: boolean;
  error: string;
}

/* A stored payload counts as a full result when it carries trend points,
   breakdown rows, or backend totals — otherwise it is params-only. */
function storedResultOf(item: any): any | null {
  const cands = [item?.result, item?.explore_result, item?.data?.result];
  for (const c of cands) {
    if (!c || typeof c !== "object") continue;
    const hasTrend =
      asList(c?.trend?.points || c?.trend || c?.series || c?.points).length > 0;
    const hasRows =
      asList(c?.rows || c?.breakdown || c?.breakdown_rows || c?.table).length > 0;
    const hasTotal = numOf(c?.total) !== null;
    if (hasTrend || hasRows || hasTotal) return c;
  }
  return null;
}

/* Re-runnable params from a params-only saved item (same contract as the
   Explore builder payload — the backend joins/filters server-side). */
function paramsOf(item: any): Record<string, unknown> | null {
  const metric = strOf(item?.metric);
  if (!metric) return null;
  const p: Record<string, unknown> = { metric };
  if (strOf(item?.breakdown)) p.breakdown = strOf(item.breakdown);
  if (strOf(item?.secondary_breakdown)) p.secondary_breakdown = strOf(item.secondary_breakdown);
  if (strOf(item?.time_grain)) p.time_grain = strOf(item.time_grain);
  if (strOf(item?.comparison)) p.comparison = strOf(item.comparison);
  if (item?.date_range) p.date_range = item.date_range;
  const filters = asList(item?.filters)
    .map((f: any) => ({ column: strOf(f?.column), value: strOf(f?.value) }))
    .filter((f) => f.column && f.value);
  if (filters.length) p.filters = filters;
  return p;
}

function pointsOf(result: any): { label: string; current: number; previous: number | null }[] {
  const raw = asList(
    result?.trend?.points || result?.trend || result?.series || result?.points || result?.chart?.data,
  );
  return raw.slice(0, 60).map((p: any, i: number) => ({
    label: strOf(p?.label || p?.period || p?.x || p?.month || `P${i + 1}`),
    current: numOf(p?.current ?? p?.current_value ?? p?.value ?? p?.y) ?? 0,
    previous: numOf(p?.previous ?? p?.previous_value ?? p?.prev),
  }));
}

function rowsOf(result: any): any[] {
  return asList(result?.rows || result?.breakdown || result?.breakdown_rows || result?.table);
}

function summaryOf(result: any, item: any): string {
  const ob = result?.observation || result?.aria_observation || null;
  return strOf(
    typeof ob === "string"
      ? ob
      : ob?.summary || ob?.text || result?.summary || item?.result_summary || "",
  );
}

function titleOf(item: any): string {
  return strOf(
    item?.title || item?.name || (item?.metric ? `${humanize(item.metric)} by ${humanize(item.breakdown)}` : ""),
    "Analysis",
  );
}

function dateOf(item: any): string {
  const iso = strOf(item?.updated_at || item?.created_at || "");
  if (!iso) return "";
  try {
    return new Date(iso).toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" });
  } catch {
    return iso;
  }
}

export default function CompareView({ onClose }: { onClose: () => void }) {
  const [items, setItems] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [errorId, setErrorId] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [loaded, setLoaded] = useState<Record<string, Loaded>>({});
  const [resolving, setResolving] = useState(false);

  const fetchAll = useCallback(async () => {
    setLoading(true);
    setError("");
    setErrorId("");
    try {
      const r = await intelligenceApi.listAnalyses({ page: 1, page_size: 50, sort: "recent" });
      setItems(asList((r as any).data));
    } catch (e: any) {
      const norm = normalizeApiError({ message: e?.message }, e?.status || 0, null);
      setError(norm.message);
      setErrorId(norm.requestId);
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  /* Escape closes the dialog. */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  /* Keep selection valid when the list (re)loads. */
  useEffect(() => {
    setSelected((prev) => prev.filter((id) => items.some((a) => strOf(a?.id) === id)).slice(0, 2));
  }, [items]);

  const toggle = (id: string) => {
    setSelected((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      if (prev.length >= 2) return prev;
      return [...prev, id];
    });
  };

  /* Resolve both selections: stored result first, else re-run saved params. */
  useEffect(() => {
    if (selected.length !== 2) {
      setLoaded({});
      setResolving(false);
      return;
    }
    let cancelled = false;
    setResolving(true);
    (async () => {
      const out: Record<string, Loaded> = {};
      for (const id of selected) {
        const item = items.find((a) => strOf(a?.id) === id);
        if (!item) continue;
        const stored = storedResultOf(item);
        if (stored) {
          out[id] = { item, result: stored, live: false, error: "" };
          continue;
        }
        const params = paramsOf(item);
        if (!params) {
          out[id] = {
            item, result: null, live: false,
            error: "This saved analysis has no stored result or parameters, so there is nothing to compare.",
          };
          continue;
        }
        try {
          const r = await intelligenceApi.runExplore(params);
          const rd: any = (r as any)?.data ?? null;
          if (cancelled) return;
          if (rd && rd.ok !== false) out[id] = { item, result: rd, live: true, error: "" };
          else out[id] = { item, result: null, live: true, error: strOf(rd?.detail || rd?.message || "Recompute returned no result for this analysis.") };
        } catch (e: any) {
          if (cancelled) return;
          out[id] = { item, result: null, live: true, error: businessError(e?.message) || "ARIA couldn't recompute this analysis." };
        }
      }
      if (!cancelled) {
        setLoaded(out);
        setResolving(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selected, items]);

  const pair = useMemo(() => selected.map((id) => loaded[id] || null), [selected, loaded]);

  /* Overlay chart: both current series aligned by index (labels may differ
     between analyses, so the x-axis prefers A's labels, then B's). */
  const overlay = useMemo(() => {
    const [A, B] = pair;
    if (!A?.result || !B?.result || A.error || B.error) return [];
    const pa = pointsOf(A.result);
    const pb = pointsOf(B.result);
    const n = Math.max(pa.length, pb.length);
    if (!n) return [];
    return Array.from({ length: n }, (_, i) => ({
      label: pa[i]?.label ?? pb[i]?.label ?? `P${i + 1}`,
      a: pa[i] ? pa[i].current : null,
      b: pb[i] ? pb[i].current : null,
    }));
  }, [pair]);

  const nameA = pair[0] ? titleOf(pair[0].item) : "Analysis A";
  const nameB = pair[1] ? titleOf(pair[1].item) : "Analysis B";

  return (
    <div>
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div>
          <h3 className="text-[15px] font-semibold text-foreground tracking-tight">Compare saved analyses</h3>
          <p className="text-[11.5px] text-muted-foreground mt-0.5">Pick two saved analyses to compare side-by-side.</p>
        </div>
        <button
          onClick={onClose}
          className="inline-flex items-center justify-center w-8 h-8 rounded-lg hover:bg-secondary text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
          aria-label="Close compare view"
        >
          <X size={16} />
        </button>
      </div>

      {loading && <div className="mt-3"><LoadingPanel label="Loading saved analyses…" /></div>}
      {!loading && error && (
        <div className="mt-3"><ErrorPanel message={error} requestId={errorId} onRetry={fetchAll} /></div>
      )}
      {!loading && !error && items.length < 2 && (
        <div className="mt-3">
          <EmptyState
            title="Not enough saved analyses yet."
            body="Save at least two analyses from Explore (use Save Analysis on any result) — then pick two here to compare side-by-side."
            action={<button onClick={onClose} className="text-xs font-semibold text-blue-700 dark:text-blue-400 hover:underline">Back to Explore</button>}
          />
        </div>
      )}

      {!loading && !error && items.length >= 2 && (
        <>
          <div className="mt-3 flex items-center justify-between gap-2 flex-wrap">
            <p className="text-[11.5px] text-muted-foreground" role="status">
              {selected.length === 2
                ? "Two selected — remove one to pick another."
                : `Select ${2 - selected.length} more to compare.`}
            </p>
            {selected.length > 0 && (
              <button
                onClick={() => setSelected([])}
                className="text-[11.5px] font-semibold text-blue-700 dark:text-blue-400 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
              >
                Clear selection
              </button>
            )}
          </div>
          <div className="mt-2 max-h-56 overflow-y-auto divide-y divide-border border border-border rounded-xl" role="group" aria-label="Saved analyses picker">
            {items.map((a: any) => {
              const id = strOf(a?.id);
              const checked = selected.includes(id);
              const disabled = !checked && selected.length >= 2;
              return (
                <button
                  key={id || titleOf(a)}
                  onClick={() => id && toggle(id)}
                  disabled={!id || disabled}
                  role="checkbox"
                  aria-checked={checked}
                  className={`w-full text-left px-3 py-2 flex items-center gap-2.5 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-blue-500 ${checked ? "bg-blue-50/70 dark:bg-blue-500/10" : "hover:bg-secondary"} ${disabled ? "opacity-50" : ""}`}
                >
                  <span
                    className={`inline-flex items-center justify-center w-[18px] h-[18px] rounded border flex-shrink-0 ${checked ? "bg-blue-700 border-blue-700 text-white" : "border-border text-transparent"}`}
                    aria-hidden
                  >
                    <Check size={12} />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block text-[12.5px] font-medium text-foreground truncate">{titleOf(a)}</span>
                    <span className="block text-[11px] text-muted-foreground truncate">
                      {[humanize(a?.metric), a?.breakdown ? humanize(a.breakdown) : "", dateOf(a)].filter(Boolean).join(" · ")}
                    </span>
                  </span>
                </button>
              );
            })}
          </div>

          {selected.length < 2 && (
            <p className="mt-3 text-[12px] text-muted-foreground">Select two analyses above to compare their totals, trends, breakdowns, and observations.</p>
          )}
          {selected.length === 2 && resolving && (
            <div className="mt-3"><LoadingPanel label="Loading both analyses — using stored results, recomputing live where needed…" /></div>
          )}

          {selected.length === 2 && !resolving && (
            <div className="mt-3 space-y-4">
              {pair.some((p) => p?.error) && (
                <div className="space-y-2">
                  {pair.filter((p) => p?.error).map((p: Loaded | null, i: number) => (
                    <p key={i} className="text-[12px] text-rose-600 dark:text-rose-400 bg-rose-50 dark:bg-rose-500/10 border border-rose-100 dark:border-rose-500/25 rounded-xl px-3 py-2" role="alert">
                      {titleOf(p!.item)}: {p!.error}
                    </p>
                  ))}
                </div>
              )}

              {overlay.length > 0 && (
                <Card className="p-4">
                  <h4 className="text-[13px] font-semibold text-foreground">Trend comparison · current series</h4>
                  <p className="text-[11px] text-muted-foreground">Both current series aligned by point order.</p>
                  <div className="h-[220px] mt-2">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={overlay} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                        <XAxis dataKey="label" tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} minTickGap={24} />
                        <YAxis tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} width={52} />
                        <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        <Line type="monotone" dataKey="a" name={nameA} stroke="#2F6FAD" strokeWidth={2.5} dot={false} connectNulls />
                        <Line type="monotone" dataKey="b" name={nameB} stroke="#7C3AED" strokeWidth={2.5} dot={false} connectNulls />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </Card>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {pair.map((p: Loaded | null, i: number) => {
                  if (!p || p.error || !p.result) return null;
                  const r = p.result;
                  const metricName = strOf(r?.metric || p.item?.metric);
                  const total = numOf(r?.total);
                  const prevTotal = numOf(r?.previous_total);
                  const chg = numOf(r?.change);
                  const pct = numOf(r?.change_pct ?? r?.changePct);
                  const pts = pointsOf(r);
                  const top = rowsOf(r).slice(0, 5);
                  const summary = summaryOf(r, p.item);
                  return (
                    <Card key={selected[i]} className="p-4">
                      <p className="text-[13.5px] font-semibold text-foreground leading-snug">{titleOf(p.item)}</p>
                      <p className="text-[11px] text-muted-foreground mt-0.5">
                        {[humanize(p.item?.metric), p.item?.breakdown ? `by ${humanize(p.item.breakdown)}` : "", dateOf(p.item)].filter(Boolean).join(" · ")}
                      </p>
                      <p className="mt-1.5">
                        <span className={`inline-flex items-center text-[10.5px] font-semibold px-2 py-0.5 rounded-full border ${p.live ? "bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-500/25" : "bg-blue-50 dark:bg-blue-500/10 text-blue-700 dark:text-blue-400 border-blue-100 dark:border-blue-500/20"}`}>
                          {p.live ? "Recomputed live from saved parameters" : "Stored result"}
                        </span>
                      </p>

                      <div className="mt-2 grid grid-cols-3 gap-2 text-center">
                        <div className="rounded-xl bg-secondary border border-border px-2 py-2">
                          <p className="text-[10px] font-medium text-muted-foreground">Current</p>
                          <p className="text-[14px] font-bold text-foreground tabular-nums">{total === null ? "—" : fmtMetric(total, metricName)}</p>
                        </div>
                        <div className="rounded-xl bg-secondary border border-border px-2 py-2">
                          <p className="text-[10px] font-medium text-muted-foreground">Previous</p>
                          <p className="text-[14px] font-bold text-foreground tabular-nums">{prevTotal === null ? "—" : fmtMetric(prevTotal, metricName)}</p>
                        </div>
                        <div className="rounded-xl bg-secondary border border-border px-2 py-2">
                          <p className="text-[10px] font-medium text-muted-foreground">Change</p>
                          <p className={`text-[14px] font-bold tabular-nums ${pct === null ? "text-muted-foreground" : pct >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                            {pct === null ? "—" : fmtPctBiz(pct)}
                          </p>
                          {chg !== null && <p className="text-[10px] text-muted-foreground tabular-nums">{fmtMetric(chg, metricName)}</p>}
                        </div>
                      </div>

                      {pts.length > 0 ? (
                        <div className="h-[160px] mt-2">
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={pts} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                              <XAxis dataKey="label" tick={{ fontSize: 10, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} minTickGap={28} />
                              <YAxis tick={{ fontSize: 10, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} width={48} />
                              <Tooltip formatter={(v: any) => fmtMetric(numOf(v), metricName)} contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
                              <Line type="monotone" dataKey="current" name="Current" stroke={i === 0 ? "#2F6FAD" : "#7C3AED"} strokeWidth={2} dot={false} />
                              <Line type="monotone" dataKey="previous" name="Previous" stroke="#B9D0E8" strokeWidth={1.5} strokeDasharray="5 4" dot={false} connectNulls />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      ) : (
                        <p className="text-[11.5px] text-muted-foreground mt-2">No trend points in this result.</p>
                      )}

                      <h5 className="mt-2 text-[12px] font-semibold text-foreground">
                        Top segments{strOf(r?.breakdown || p.item?.breakdown) ? ` · ${titleCase(strOf(r.breakdown || p.item.breakdown))}` : ""}
                      </h5>
                      {top.length > 0 ? (
                        <table className="w-full text-[11.5px] mt-1">
                          <thead>
                            <tr className="text-left text-muted-foreground border-b border-border">
                              <th scope="col" className="py-1 pr-2 font-medium">Segment</th>
                              <th scope="col" className="py-1 pr-2 font-medium text-right">Current</th>
                              <th scope="col" className="py-1 font-medium text-right">% Change</th>
                            </tr>
                          </thead>
                          <tbody>
                            {top.map((row: any, j: number) => {
                              const cur = numOf(row?.current ?? row?.current_value ?? row?.value);
                              const pp = numOf(row?.change_pct ?? row?.pct_change ?? row?.pct ?? row?.percent_change);
                              return (
                                <tr key={j} className="border-b border-border last:border-0">
                                  <td className="py-1 pr-2 font-medium text-foreground truncate max-w-[140px]">
                                    {strOf(row?.category || row?.label || row?.name || `Row ${j + 1}`)}
                                  </td>
                                  <td className="py-1 pr-2 text-right tabular-nums">{cur === null ? "—" : fmtMetric(cur, metricName)}</td>
                                  <td className={`py-1 text-right tabular-nums font-medium ${pp === null ? "text-muted-foreground" : pp >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                                    {pp === null ? "—" : fmtPctBiz(pp)}
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      ) : (
                        <p className="text-[11.5px] text-muted-foreground mt-1">No breakdown rows in this result.</p>
                      )}

                      <h5 className="mt-2 text-[12px] font-semibold text-foreground">Observation</h5>
                      <p className="text-[11.5px] text-secondary-foreground leading-relaxed mt-0.5">
                        {summary || "No observation was stored for this analysis."}
                      </p>
                    </Card>
                  );
                })}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
