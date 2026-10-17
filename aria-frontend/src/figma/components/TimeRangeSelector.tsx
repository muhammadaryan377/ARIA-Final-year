import React from "react";
import { CalendarDays } from "lucide-react";

export interface TimeRangeOption {
  key: string;
  label: string;
}

interface TimeRangeSelectorProps {
  options: TimeRangeOption[];
  value: string;
  onChange: (key: string) => void;
  className?: string;
}

export function TimeRangeSelector({ options, value, onChange, className = "" }: TimeRangeSelectorProps) {
  return (
    <div
      className={`inline-flex items-center gap-0.5 p-0.5 rounded-xl bg-secondary/80 border border-border/60 ${className}`}
      role="radiogroup"
      aria-label="Time range"
    >
      <CalendarDays size={12} className="text-muted-foreground ml-1.5 mr-0.5 flex-shrink-0 pointer-events-none" />
      {options.map((opt) => {
        const active = opt.key === value;
        return (
          <button
            key={opt.key}
            role="radio"
            aria-checked={active}
            onClick={() => onChange(opt.key)}
            className={`
              relative px-2.5 py-1 text-[11px] font-medium rounded-lg
              transition-all duration-200 ease-out
              ${active
                ? "bg-card text-foreground shadow-sm border border-border/80"
                : "text-muted-foreground hover:text-secondary-foreground border border-transparent"
              }
            `}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
