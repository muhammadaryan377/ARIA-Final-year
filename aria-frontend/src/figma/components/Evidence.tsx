import { useState } from "react";
import type { ReactNode } from "react";
import { ChevronDown } from "lucide-react";

/* =====================================================================
   Shared Evidence disclosure (s.txt): ONE consistent component used by
   every AI surface. Collapsed by default, subtle and secondary; expands
   smoothly to reveal supporting sections. Never the dominant element.
   L1 answer + L2 explanation stay outside; L3 evidence + L4 technical
   details live inside.
   ===================================================================== */

export interface EvidenceSection {
  title: string;
  body: ReactNode;
}

export default function Evidence({
  sections,
  defaultOpen = false,
  label = "Evidence",
  technical,
}: {
  sections: EvidenceSection[];
  defaultOpen?: boolean;
  label?: string;
  technical?: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  if (!sections.length) return null;
  return (
    <div className="rounded-lg border border-border overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="w-full flex items-center gap-1.5 px-3 py-2 text-left text-[11px] font-semibold text-muted-foreground hover:text-foreground hover:bg-secondary/40 transition-colors"
      >
        <ChevronDown
          size={13}
          className={`transition-transform duration-200 ${open ? "rotate-180" : ""}`}
        />
        {label}
        <span className="ml-auto text-[10px] font-normal">
          {open ? "hide" : "why this conclusion"}
        </span>
      </button>
      {open && (
        <div className="border-t border-border px-3 py-3 space-y-3 bg-muted/20 animate-fade-in">
          {sections.map((s) => (
            <div key={s.title}>
              <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider">
                {s.title}
              </p>
              <div className="mt-1 text-[12px] text-foreground leading-relaxed">
                {s.body}
              </div>
            </div>
          ))}
          {technical && (
            <details className="rounded-md border border-border/60 bg-background/50 px-2.5 py-2">
              <summary className="cursor-pointer text-[11px] font-semibold text-muted-foreground hover:text-foreground">
                Technical details
              </summary>
              <div className="mt-2 text-[11px] text-muted-foreground leading-relaxed">
                {technical}
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
