import { useEffect, useState } from "react";
import { Bookmark, CalendarDays, RefreshCw, User, X } from "lucide-react";
import { asList, strOf, numOf, humanize, fmtMetric } from "../../lib/intelHelpers";
import { btnPrimary, btnGhost } from "./cards";

/* =====================================================================
   Simplified prescription case drawer. Shows ONE case: the detected
   problem, its action with the expected outcome, a small
   where-things-stand vs if-the-action-works visual, and confidence dots.
   Everything rendered comes from the backend display contract —
   the drawer never invents numbers or actions. Plain words only.
   ===================================================================== */

export type RxCaseInput = {
  key: string;
  headline: string;
  metricLabel: string;
  prescription: any;
  owner: string;
  due: string;
  outcome: string;
  savedId: string | null;
  recheckNote: string;
};

function dotsOf(conf: unknown): { filled: number; label: string } {
  const s = strOf(conf).toLowerCase();
  if (/high|strong/.test(s)) return { filled: 3, label: strOf(conf) || "High" };
  if (/moderate|medium/.test(s)) return { filled: 2, label: strOf(conf) || "Moderate" };
  return { filled: 1, label: strOf(conf) || "Early signal" };
}

export default function RxCaseDrawer({
  item, onClose, onTrack, onPatch, onRecheck, rechecking,
}: {
  item: RxCaseInput | null;
  onClose: () => void;
  onTrack: (prescription: any) => void;
  onPatch: (patch: Record<string, string>) => Promise<void>;
  onRecheck: () => void;
  rechecking: boolean;
}) {
  const [owner, setOwner] = useState("");
  const [due, setDue] = useState("");
  const [patchBusy, setPatchBusy] = useState(false);
  const [patchMsg, setPatchMsg] = useState("");

  useEffect(() => {
    setOwner(item?.owner || "");
    setDue(strOf(item?.due).slice(0, 10));
    setPatchMsg("");
  }, [item?.key, item?.owner, item?.due]);

  useEffect(() => {
    if (!item) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [item, onClose]);

  if (!item) return null;
  const p = item.prescription || {};
  const disp = p?.display && typeof p.display === "object" ? p.display : {};
  const det = p?.details && typeof p.details === "object" ? p.details : {};
  const s = (v: unknown) => strOf(v).trim();
  const actionText = s(disp.action_text);
  const outcomeText = s(disp.outcome_text);
  const conf = dotsOf(disp.confidence_label || p?.confidence_label || p?.confidence);
  const whyBullets: string[] = (() => {
    const raw = disp.why_text !== undefined ? disp.why_text : p?.why_text;
    const list = Array.isArray(raw) ? raw : raw ? [raw] : [];
    const bullets = list.map(s).filter(Boolean).slice(0, 3);
    if (bullets.length) return bullets;
    const legacy = s(disp.why_summary) || s(p?.why_summary);
    return legacy ? [legacy] : [];
  })();
  const limits: string[] = asList(det.limitations?.length ? det.limitations : p?.limitations)
    .map((x: any) => s(typeof x === "string" ? x : x?.text || x?.label))
    .filter(Boolean).slice(0, 3);
  const imp = p?.impact && typeof p.impact === "object" ? p.impact : {};
  const baseline = numOf(imp?.baseline);
  const scenario = numOf(imp?.scenario);
  const showBars = baseline !== null && scenario !== null;
  const maxBar = Math.max(baseline ?? 0, scenario ?? 0, 0.0001);
  const outcome = s(item.outcome).toLowerCase();
  const metricLabel = item.metricLabel || humanize(strOf(p?.metric));

  const saveOwnerDue = async () => {
    if (!item.savedId || patchBusy) return;
    setPatchBusy(true);
    setPatchMsg("");
    try {
      await onPatch({
        ...(owner.trim() ? { owner: owner.trim() } : {}),
        ...(due ? { due } : {}),
      });
      setPatchMsg("Saved.");
    } catch {
      setPatchMsg("ARIA couldn't save that just now.");
    } finally {
      setPatchBusy(false);
    }
  };

  const mark = async (review_outcome: string) => {
    if (!item.savedId || patchBusy) return;
    setPatchBusy(true);
    setPatchMsg("");
    try {
      await onPatch({ review_outcome });
      setPatchMsg(review_outcome === "achieved" ? "Marked as achieved." : "Marked as missed.");
    } catch {
      setPatchMsg("ARIA couldn't save that just now.");
    } finally {
      setPatchBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50" role="dialog" aria-modal="true" aria-label={`Case: ${item.headline.slice(0, 80)}`}>
      <button
        className="absolute inset-0 bg-[#0F2440]/40 cursor-default"
        onClick={onClose}
        aria-label="Close case"
        tabIndex={-1}
      />
      <aside className="absolute right-0 top-0 h-full w-full max-w-md bg-card shadow-2xl border-l border-border overflow-y-auto">
        <div className="p-5 space-y-4">
          <div className="flex items-start gap-2">
            <div className="min-w-0 flex-1">
              <p className="text-[11px] font-bold tracking-[0.14em] text-blue-700 dark:text-blue-400">PRESCRIPTION CASE</p>
              <h3 className="mt-1 text-[16px] font-semibold text-foreground leading-snug">{item.headline}</h3>
              {metricLabel && <p className="mt-0.5 text-[11px] text-muted-foreground">{metricLabel}</p>}
            </div>
            <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-secondary text-muted-foreground hover:text-foreground" aria-label="Close case">
              <X size={16} />
            </button>
          </div>

          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1" title={`Confidence: ${conf.label}`}>
              {[0, 1, 2].map((dot) => (
                <span key={dot} className={`inline-block w-2 h-2 rounded-full ${dot < conf.filled ? "bg-blue-600" : "bg-border"}`} />
              ))}
            </span>
            <span className="text-[11px] text-muted-foreground">{conf.label}</span>
            {outcome === "achieved" && (
              <span className="text-[10.5px] font-semibold px-2 py-0.5 rounded-full border bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-500/25">Achieved</span>
            )}
            {outcome === "missed" && (
              <span className="text-[10.5px] font-semibold px-2 py-0.5 rounded-full border bg-rose-50 dark:bg-rose-500/10 text-rose-700 dark:text-rose-400 border-rose-200 dark:border-rose-500/25">Missed</span>
            )}
          </div>

          {showBars && (
            <div className="rounded-xl bg-secondary border border-border px-3 py-2.5" role="img"
              aria-label={`Where things stand ${fmtMetric(baseline, metricLabel)} versus ${fmtMetric(scenario, metricLabel)} if the action works`}>
              {[
                { label: "Where things stand", value: baseline as number, cls: "bg-slate-300" },
                { label: "If the action works", value: scenario as number, cls: (scenario as number) >= (baseline as number) ? "bg-emerald-500" : "bg-amber-500" },
              ].map((r) => (
                <div key={r.label} className="flex items-center gap-2 py-1">
                  <span className="w-28 flex-shrink-0 text-[11px] text-muted-foreground">{r.label}</span>
                  <div className="flex-1 h-2 rounded-full bg-border/70 overflow-hidden">
                    <div className={`h-full rounded-full ${r.cls}`} style={{ width: `${Math.max(3, Math.min(100, (r.value / maxBar) * 100))}%` }} />
                  </div>
                  <span className="w-16 flex-shrink-0 text-right text-[11px] font-semibold text-foreground tabular-nums">
                    {fmtMetric(r.value, metricLabel)}
                  </span>
                </div>
              ))}
            </div>
          )}

          {actionText && (
            <div className="rounded-xl bg-blue-50/60 dark:bg-blue-500/10 border border-blue-100 dark:border-blue-500/20 px-3.5 py-3">
              <p className="text-[13px] font-semibold text-foreground leading-snug">{actionText}</p>
              {outcomeText && <p className="mt-1 text-[12px] text-secondary-foreground leading-relaxed">Expected: {outcomeText}</p>}
            </div>
          )}

          {whyBullets.length > 0 && (
            <div>
              <p className="text-[12px] font-semibold text-foreground">Why this is suggested</p>
              <ul className="mt-1 space-y-1">
                {whyBullets.map((b, i) => (
                  <li key={i} className="text-[12px] text-secondary-foreground leading-relaxed">· {b}</li>
                ))}
              </ul>
            </div>
          )}

          {limits.length > 0 && (
            <p className="text-[11px] text-muted-foreground leading-relaxed">Limits: {limits.join(" · ")}</p>
          )}

          {item.recheckNote && (
            <p className="text-[11.5px] text-secondary-foreground leading-relaxed" role="status">{item.recheckNote}</p>
          )}

          {item.savedId ? (
            <div className="rounded-xl border border-border p-3 space-y-2">
              <p className="text-[12px] font-semibold text-foreground">Owner & review</p>
              <div className="flex items-center gap-2 flex-wrap">
                <label className="inline-flex items-center gap-1.5 text-[12px] text-muted-foreground">
                  <User size={12} />
                  <input
                    value={owner}
                    onChange={(e) => setOwner(e.target.value)}
                    placeholder="Owner"
                    className="w-32 bg-card border border-border rounded-lg px-2 py-1.5 text-[12px] text-foreground placeholder:text-muted-foreground focus:border-blue-500 outline-none"
                    aria-label="Case owner"
                  />
                </label>
                <label className="inline-flex items-center gap-1.5 text-[12px] text-muted-foreground">
                  <CalendarDays size={12} />
                  <input
                    type="date"
                    value={due}
                    onChange={(e) => setDue(e.target.value)}
                    className="bg-card border border-border rounded-lg px-2 py-1.5 text-[12px] text-foreground focus:border-blue-500 outline-none"
                    aria-label="Review date"
                  />
                </label>
                <button onClick={saveOwnerDue} disabled={patchBusy} className={btnGhost} aria-label="Save owner and review date">
                  {patchBusy ? "Saving…" : "Save"}
                </button>
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                <button onClick={() => void mark("achieved")} disabled={patchBusy} className="text-[12px] font-medium text-emerald-700 dark:text-emerald-400 hover:underline disabled:opacity-40" aria-label="Mark case achieved">Mark achieved</button>
                <button onClick={() => void mark("missed")} disabled={patchBusy} className="text-[12px] font-medium text-rose-600 dark:text-rose-400 hover:underline disabled:opacity-40" aria-label="Mark case missed">Mark missed</button>
                <button onClick={onRecheck} disabled={rechecking} className="inline-flex items-center gap-1 text-[12px] font-medium text-muted-foreground hover:text-blue-700 dark:hover:text-blue-400 hover:underline disabled:opacity-40" aria-label="Re-check this case now">
                  <RefreshCw size={12} className={rechecking ? "animate-spin" : ""} /> {rechecking ? "Checking…" : "Re-check"}
                </button>
              </div>
              {patchMsg && <p className="text-[11.5px] text-secondary-foreground" role="status">{patchMsg}</p>}
            </div>
          ) : (
            <button onClick={() => onTrack(p)} className={`${btnPrimary} w-full`} aria-label="Track this action">
              <Bookmark size={13} /> Track this action
            </button>
          )}
        </div>
      </aside>
    </div>
  );
}
