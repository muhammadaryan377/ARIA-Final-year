import { Loader2, Check } from "lucide-react";

/* =====================================================================
   AIProcessingIndicator — subtle premium 4-stage progress for AI work.
   Stages (driven ONLY by the real request lifecycle, never timers):
     1. Understanding data   — request sent, no response yet
     2. Analyzing patterns   — response headers/first payload arrived
     3. Finding evidence     — payload received, post-processing locally
     4. Generating visualization — building charts/viz from real rows
   Props:
     stage: 0-3 (current stage index, derived by the caller from actual
            lifecycle milestones) | -1 hides the step list (label only)
     label: optional override for the headline (defaults per stage)
     sublabel: optional override for the supporting line
     compact: smaller inline variant (forecast/decision/copilot surfaces)
   No loops, no gimmicks: one spinner on the current step, checks on done
   steps, pending dots on future steps. Settles to static on completion
   (callers unmount it when loading flips false).
   ===================================================================== */

export const AI_STAGES = [
  "Understanding data",
  "Analyzing patterns",
  "Finding evidence",
  "Generating visualization",
] as const;

const STAGE_SUBS = [
  "Inspecting schema and scoping your question",
  "Running analysis against real data",
  "Validating supporting evidence",
  "Building charts from the results",
];

interface AIProcessingIndicatorProps {
  stage?: number;
  label?: string;
  sublabel?: string;
  compact?: boolean;
}

export default function AIProcessingIndicator({
  stage = 0,
  label,
  sublabel,
  compact = false,
}: AIProcessingIndicatorProps) {
  const current = Math.max(0, Math.min(3, stage ?? 0));
  const headline = label || AI_STAGES[current];
  const support = sublabel || STAGE_SUBS[current];

  if (compact) {
    return (
      <div className="flex items-center gap-2.5" role="status" aria-live="polite">
        <Loader2 size={15} className="animate-spin text-primary flex-shrink-0" />
        <div className="min-w-0">
          <p className="text-[13px] font-medium text-foreground truncate ai-step-current" key={current}>
            {headline}
          </p>
          <p className="text-[11px] text-muted-foreground truncate">{support}</p>
        </div>
        <div className="flex items-center gap-1 ml-1 flex-shrink-0" aria-hidden="true">
          {AI_STAGES.map((s, i) => (
            <span
              key={s}
              title={s}
              className={`w-1.5 h-1.5 rounded-full ${i < current ? "bg-positive" : i === current ? "bg-primary" : "bg-border"}`}
            />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div
      className="bg-card border border-border rounded-xl p-6 flex flex-col items-center gap-5"
      role="status"
      aria-live="polite"
    >
      <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
        <Loader2 size={20} className="text-primary animate-spin" />
      </div>
      <div className="text-center space-y-1">
        <p className="text-sm font-medium text-foreground ai-step-current" key={current}>
          {headline}
        </p>
        <p className="text-xs text-muted-foreground">{support}</p>
      </div>
      <ol className="flex items-center gap-2 text-[11px] text-muted-foreground">
        {AI_STAGES.map((s, i) => (
          <li key={s} className="flex items-center gap-2">
            {i > 0 && <span aria-hidden="true">→</span>}
            <span
              className={`inline-flex items-center gap-1.5 ${
                i < current ? "text-positive" : i === current ? "text-primary font-medium" : ""
              }`}
            >
              {i < current ? (
                <Check size={11} aria-label={`${s} done`} />
              ) : (
                <span
                  aria-hidden="true"
                  className={`w-1.5 h-1.5 rounded-full ${i === current ? "bg-primary" : "bg-border"}`}
                />
              )}
              {s}
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}
