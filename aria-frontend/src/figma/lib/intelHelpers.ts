/* Shared Intelligence presentation helpers (single source of truth).
   Pure formatting/normalization only — the frontend never calculates truth
   here: every number rendered with these helpers comes from a backend
   result. Imported by Intelligence.tsx + intelligence/* (no duplication). */

import { fmtBiz, fmtMoneyBiz } from "./format";

export const asList = (v: unknown): any[] => (Array.isArray(v) ? v : []);

export const strOf = (v: unknown, fb = ""): string =>
  typeof v === "string" ? v : v === null || v === undefined ? fb : String(v);

export const numOf = (v: unknown): number | null => {
  if (v === null || v === undefined || v === "") return null;
  const n = Number(String(v).replace(/,/g, ""));
  return isNaN(n) ? null : n;
};

/* Human-scale metric formatting: money-like metrics get $, others plain.
   Prevents raw long decimals and misleading $ on counts. Token-based
   (never bare substrings): "Costello"/"marginal"/"discountinued" stay
   plain. Mirrors the backend _fmtm rule. */
export function isMoneyMetric(name: unknown): boolean {
  const spaced = String(name || "")
    .replace(/[_-]+/g, " ")
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .toLowerCase();
  const toks = new Set(spaced.split(/[^a-z0-9]+/).filter(Boolean));
  const money = new Set([
    "revenue", "sales", "price", "cost", "profit", "amount", "payment",
    "spend", "budget", "discount", "wage", "salary", "fee", "fare",
    "freight", "tuition", "premium", "margin", "income", "expense",
  ]);
  for (const t of toks) if (money.has(t)) return true;
  return false;
}

export function fmtMetric(v: number | null | undefined, metricName?: unknown): string {
  if (v === null || v === undefined) return "—";
  return isMoneyMetric(metricName) ? fmtMoneyBiz(v) : fmtBiz(v);
}

export function humanize(v: unknown): string {
  const s = String(v ?? "").trim();
  if (!s) return "";
  const spaced = s
    .replace(/[_-]+/g, " ")
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/\s+/g, " ")
    .trim();
  if (/^[A-Z0-9_()%. ]+$/.test(s) && s.length <= 10) return s.replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/* Title-case for table/heading labels ("Product Category",
   "Revenue Trend by Product Category" style). */
export function titleCase(v: unknown): string {
  return humanize(v)
    .split(" ")
    .map((w) => (w ? w.charAt(0).toUpperCase() + w.slice(1) : w))
    .join(" ");
}

/* Business-worded error copy: never a bare "Failed to fetch", never SQL. */
export function businessError(raw: string): string {
  const s = String(raw || "");
  if (!s) return "";
  if (/failed to fetch|network|load failed|request failed|fetch|backend unreachable|timed out|timeout/i.test(s)) {
    return "ARIA couldn't reach the server for this Intelligence request. Check that the backend is running, then retry.";
  }
  if (/session expired|sign in|unauthorized|401/i.test(s)) return s.slice(0, 200);
  if (/select\s.+\sfrom|traceback|exception|stack/i.test(s)) {
    return "ARIA couldn't complete this Intelligence request. Try again in a moment.";
  }
  return s.slice(0, 220);
}

/* Backend-driven empty reason: GET /intelligence/meta may return
   { reason: "NO_SCHEMA", hint/message } when no dataset is connected.
   Surface the backend hint + a Data-page link instead of a generic panel. */
export function getNoSchemaInfo(meta: any): { isNoSchema: boolean; hint: string } {
  const reason = meta?.reason ?? meta?.data?.reason ?? "";
  const hint = strOf(meta?.hint ?? meta?.message ?? meta?.data?.hint ?? "", "");
  if (String(reason).toUpperCase() === "NO_SCHEMA") {
    return {
      isNoSchema: true,
      hint:
        hint ||
        "No schema found for this data source. Connect PostgreSQL, MySQL, or upload a CSV on the Data page — then every section here will populate.",
    };
  }
  return { isNoSchema: false, hint };
}
