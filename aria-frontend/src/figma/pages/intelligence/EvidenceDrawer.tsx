/* Full evidence drawer for the Explore tab (hyp.txt #60): source, period,
   metric, comparison, segments, observations, limitations — from the live
   result, never mocked. Adds a "Copy summary" action over real text. */

import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence, MotionConfig } from "framer-motion";
import { AlertTriangle, Check, Copy, X } from "lucide-react";
import { asList, strOf, numOf, humanize, fmtMetric } from "../../lib/intelHelpers";

export interface EvidenceContext {
  metric: string;
  breakdown: string;
  comparison: string;
  trendTitle: string;
  trendSub: string;
  rows: any[];
  trendPoints: { label: string; current: number; previous: number | null }[];
  obsSummary: string;
  obsBullets: string[];
  obsBasis: string;
}

export function buildEvidenceSummary(result: any, ctx: EvidenceContext): string {
  const lines: string[] = [];
  lines.push(ctx.trendTitle);
  if (ctx.trendSub) lines.push(ctx.trendSub);
  const total = numOf(result?.total);
  const prevTotal = numOf(result?.previous_total);
  const changePct = numOf(result?.change_pct ?? result?.changePct);
  if (total !== null) {
    let line = `Total: ${fmtMetric(total, result?.metric || ctx.metric)}`;
    if (prevTotal !== null) line += ` (previous ${fmtMetric(prevTotal, result?.metric || ctx.metric)})`;
    if (changePct !== null) line += ` [${changePct >= 0 ? "+" : ""}${Math.round(changePct * 10) / 10}%]`;
    lines.push(line);
  }
  if (ctx.obsSummary) lines.push(`Observation: ${ctx.obsSummary}`);
  for (const b of ctx.obsBullets.slice(0, 8)) lines.push(`- ${b}`);
  if (ctx.obsBasis) lines.push(`Basis: ${ctx.obsBasis}`);
  const period = strOf(result?.analysis_period || ctx.trendSub);
  if (period) lines.push(`Period: ${period}`);
  lines.push(`Segments: ${ctx.rows.length}`);
  return lines.join("\n");
}

export default function EvidenceDrawer({
  open,
  onClose,
  result,
  context: ctx,
}: {
  open: boolean;
  onClose: () => void;
  result: any;
  context: EvidenceContext;
}) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!open) {
      setCopied(false);
      return;
    }
    try {
      closeRef.current?.focus();
    } catch {
      /* ignore */
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [open, onClose]);

  const copySummary = async () => {
    const text = buildEvidenceSummary(result, ctx);
    try {
      if (navigator?.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.setAttribute("readonly", "");
        ta.style.position = "absolute";
        ta.style.left = "-9999px";
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        ta.remove();
      }
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  const v: any = result?.validation || null;
  const scope: any = result?.breakdown_scope || null;
  const checks: string[] = asList(v?.checks)
    .map((c: any) => strOf(typeof c === "string" ? c : c?.name || c?.label))
    .filter(Boolean);
  const failed: string[] = asList(v?.failed)
    .map((c: any) => strOf(typeof c === "string" ? c : c?.name))
    .filter(Boolean);
  const detailRows: [string, string][] = [
    ["Metric", strOf(result?.metric || humanize(ctx.metric) || "—")],
    ["Breakdown", strOf(result?.breakdown || humanize(ctx.breakdown) || "—")],
    ["Period", strOf(result?.analysis_period || ctx.trendSub || "—")],
    ["Comparison", strOf(result?.comparison || ctx.comparison || "—")],
    ["Segments", String(ctx.rows.length)],
    ["Groups analyzed", strOf(result?.row_count ?? "") || "—"],
    ["Records", strOf(result?.record_count ?? result?.total_records ?? "") || "—"],
  ];
  const limits: string[] = [];
  if (ctx.obsBasis) limits.push(ctx.obsBasis);
  if (scope?.truncated)
    limits.push(
      `Showing top ${scope.returned} of ${scope.total_groups} groups — totals cover all groups.`,
    );
  if (ctx.trendPoints.some((p) => p.previous === null || p.previous === undefined))
    limits.push("Previous-period values unavailable for some points (shown as —).");
  if (failed.length > 0) limits.push(`Validation flags: ${failed.join(", ")}.`);

  return (
    <MotionConfig reducedMotion="user">
      <AnimatePresence>
        {open && (
          <div className="fixed inset-0 z-50" role="dialog" aria-modal="true" aria-label="Full evidence">
            <motion.div
              className="absolute inset-0 bg-slate-900/40"
              onClick={onClose}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
            />
            <motion.aside
              className="absolute right-0 top-0 h-full w-full max-w-md bg-card shadow-2xl flex flex-col"
              initial={{ x: 48, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: 48, opacity: 0 }}
              transition={{ duration: 0.28, ease: "easeOut" }}
            >
              <div className="flex items-start justify-between gap-3 px-5 py-4 border-b border-border">
                <div>
                  <h3 className="text-[15px] font-semibold text-foreground tracking-tight">
                    Full Evidence
                  </h3>
                  <p className="text-[11.5px] text-muted-foreground mt-0.5">
                    {ctx.trendTitle}
                    {ctx.trendSub ? ` · ${ctx.trendSub}` : ""}
                  </p>
                </div>
                <div className="flex items-center gap-1.5 flex-shrink-0">
                  <button
                    onClick={copySummary}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11.5px] font-semibold text-blue-700 dark:text-blue-400 border border-blue-200 dark:border-blue-500/25 hover:bg-blue-50 dark:hover:bg-blue-500/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                    aria-label="Copy evidence summary"
                    title="Copy a plain-text summary of this analysis"
                  >
                    <Copy size={13} /> {copied ? "Copied" : "Copy summary"}
                  </button>
                  <button
                    ref={closeRef}
                    onClick={onClose}
                    className="inline-flex items-center justify-center w-8 h-8 rounded-lg hover:bg-secondary text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                    aria-label="Close evidence"
                  >
                    <X size={16} />
                  </button>
                </div>
              </div>
              {copied && (
                <p className="px-5 pt-2 text-[11px] text-emerald-700 dark:text-emerald-400" role="status">
                  Summary copied to clipboard.
                </p>
              )}
              <div className="flex-1 overflow-y-auto px-5 py-4 space-y-5">
                <section>
                  <h4 className="text-[12px] font-semibold text-muted-foreground uppercase tracking-wide">
                    Evidence Quality
                  </h4>
                  {v ? (
                    <div
                      className="mt-2 flex items-start gap-2 rounded-xl border px-3 py-2.5 text-[12.5px]"
                      style={{
                        borderColor: v.passed ? "#BBF7D0" : "#FECACA",
                        background: v.passed ? "#F0FDF4" : "#FEF2F2",
                      }}
                    >
                      {v.passed ? (
                        <Check size={15} className="text-emerald-600 dark:text-emerald-400 flex-shrink-0 mt-px" />
                      ) : (
                        <AlertTriangle size={15} className="text-rose-600 dark:text-rose-400 flex-shrink-0 mt-px" />
                      )}
                      <div>
                        <p className="font-semibold text-foreground">
                          {v.passed ? "Verified — all checks passed" : "Needs attention"}
                        </p>
                        {checks.length > 0 && <p className="text-muted-foreground mt-0.5">{checks.join(" · ")}</p>}
                      </div>
                    </div>
                  ) : (
                    <p className="mt-2 text-[12.5px] text-muted-foreground">
                      No validation record on this result.
                    </p>
                  )}
                </section>
                <section>
                  <h4 className="text-[12px] font-semibold text-muted-foreground uppercase tracking-wide">
                    Analysis Details
                  </h4>
                  <dl className="mt-2 divide-y divide-border rounded-xl border border-border">
                    {detailRows.map(([k, val]) => (
                      <div key={k} className="flex items-center justify-between gap-3 px-3 py-2">
                        <dt className="text-[12px] text-muted-foreground">{k}</dt>
                        <dd className="text-[12.5px] font-medium text-foreground text-right truncate max-w-[60%]">
                          {val}
                        </dd>
                      </div>
                    ))}
                  </dl>
                </section>
                <section>
                  <h4 className="text-[12px] font-semibold text-muted-foreground uppercase tracking-wide">
                    Observations
                  </h4>
                  {ctx.obsBullets.length > 0 ? (
                    <ul className="mt-2 space-y-1.5">
                      {ctx.obsBullets.map((b, i) => (
                        <li
                          key={i}
                          className="flex items-start gap-1.5 text-[12.5px] text-secondary-foreground leading-relaxed"
                        >
                          <span className="w-1.5 h-1.5 rounded-full bg-blue-600 flex-shrink-0 mt-1.5" />
                          {b}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="mt-2 text-[12.5px] text-muted-foreground">
                      {ctx.obsSummary || "No observations recorded."}
                    </p>
                  )}
                </section>
                <section>
                  <h4 className="text-[12px] font-semibold text-muted-foreground uppercase tracking-wide">
                    Limitations
                  </h4>
                  {limits.length > 0 ? (
                    <ul className="mt-2 space-y-1.5">
                      {limits.map((l, i) => (
                        <li
                          key={i}
                          className="flex items-start gap-1.5 text-[12.5px] text-secondary-foreground leading-relaxed"
                        >
                          <AlertTriangle size={13} className="text-amber-500 dark:text-amber-400 flex-shrink-0 mt-px" />
                          {l}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="mt-2 text-[12.5px] text-muted-foreground">
                      No known limitations for this analysis.
                    </p>
                  )}
                </section>
                <section>
                  <h4 className="text-[12px] font-semibold text-muted-foreground uppercase tracking-wide">
                    Top Segments
                  </h4>
                  {ctx.rows.length > 0 ? (
                    <ul className="mt-2 space-y-1">
                      {ctx.rows.slice(0, 8).map((r: any, i: number) => {
                        const cur = numOf(r?.current ?? r?.current_value ?? r?.value);
                        return (
                          <li
                            key={i}
                            className="flex items-center justify-between gap-2 text-[12.5px] px-3 py-1.5 rounded-lg bg-secondary"
                          >
                            <span className="font-medium text-foreground truncate">
                              {strOf(r?.category || r?.label || r?.name || `Row ${i + 1}`)}
                            </span>
                            <span className="tabular-nums text-secondary-foreground">
                              {cur === null ? "—" : fmtMetric(cur, ctx.metric)}
                            </span>
                          </li>
                        );
                      })}
                    </ul>
                  ) : (
                    <p className="mt-2 text-[12.5px] text-muted-foreground">No segment rows in this result.</p>
                  )}
                </section>
              </div>
            </motion.aside>
          </div>
        )}
      </AnimatePresence>
    </MotionConfig>
  );
}
