/* Business-English number formatting (s.txt): numbers stay when useful,
   rounded to human scale. No excessive precision anywhere in the UI. */

export function fmtBiz(v: number | null | undefined): string {
  if (v === null || v === undefined || isNaN(Number(v))) return "—";
  const n = Number(v);
  const a = Math.abs(n);
  if (a >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1)}B`;
  if (a >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (a >= 10_000) return `${(n / 1_000).toFixed(1)}K`;
  if (a >= 100) return `${Math.round(n).toLocaleString()}`;
  if (a >= 1) return `${Math.round(n * 10) / 10}`;
  if (a > 0) return `${Math.round(n * 100) / 100}`;
  return "0";
}

export function fmtMoneyBiz(v: number | null | undefined): string {
  if (v === null || v === undefined || isNaN(Number(v))) return "—";
  return `$${fmtBiz(v)}`;
}

export function fmtPctBiz(v: number | null | undefined): string {
  if (v === null || v === undefined || isNaN(Number(v))) return "—";
  const n = Number(v);
  return `${n >= 0 ? "+" : ""}${Math.round(n * 10) / 10}%`;
}

/** p-values become words by default; exact value only inside Evidence. */
export function confidenceWord(p: number | null | undefined): string {
  if (p === null || p === undefined || isNaN(Number(p))) return "Unknown";
  const n = Number(p);
  if (n < 0.01) return "High";
  if (n < 0.05) return "Moderate";
  return "Low";
}

/** Correlation |r| becomes words; exact value only inside Evidence. */
export function relationshipWord(r: number | null | undefined): string {
  if (r === null || r === undefined || isNaN(Number(r))) return "Unclear";
  const a = Math.abs(Number(r));
  if (a >= 0.7) return "Strong";
  if (a >= 0.4) return "Moderate";
  if (a >= 0.2) return "Weak";
  return "Very weak";
}

/** R² becomes words; exact value only inside Evidence technical details. */
export function patternWord(r2: number | null | undefined): string {
  if (r2 === null || r2 === undefined || isNaN(Number(r2))) return "Unclear";
  const n = Number(r2);
  if (n >= 0.7) return "Strong";
  if (n >= 0.4) return "Clear";
  if (n >= 0.2) return "Weak";
  return "Very weak";
}
