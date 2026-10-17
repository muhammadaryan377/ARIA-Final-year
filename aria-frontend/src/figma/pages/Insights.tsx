import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { Filter, Bookmark, ChevronRight, Share2, MessageSquare, FlaskConical, TrendingUp, RefreshCw, AlertTriangle, Database, Clock, Sparkles, XCircle, ArrowLeft, Send, Lightbulb, BarChart3, GitBranch, Shield } from "lucide-react";
import { useTheme } from "../App";
import { InsightChart, normalizeInsightViz } from "../components/charts";
import type { DrillSelection } from "../components/charts";
import { saveAnalysisContext } from "../lib/analysisContext";
import { apiFetch, normalizeApiError } from "../../services/api";

/* =====================================================================
   ARIA Insights — premium list + explain experience (PNG 1 + PNG 2).
   List page: hero banner, filter pills, data-source card, key-insight
   grid with REAL mini charts, recent sidebar, Ask ARIA bar.
   Explain page: breadcrumb, hero, status panel, large chart, takeaways,
   story tabs, preserved actions. All values come from GET /api/insights
   (deterministic detectors). LLM never calculates numbers here.
   ===================================================================== */

interface InsightEvidence {
  observations?: string[];
  supportingMetrics?: string[];
  sampleSize?: number;
  baseline?: number;
  currentValue?: number;
  changePercent?: number;
  limitations?: string[] | string;
}

interface InsightBusinessImpact {
  value?: number;
  direction?: "positive" | "negative" | "neutral";
  explanation?: string;
}

interface InsightRecommendation {
  question?: string;
  metric?: string;
  dimensions?: string[];
}

/* Backend 8-key controlled chart spec
   {type,x,y,series,title,interactive,data,insufficient,unsupported}.
   Validated by normalizeInsightViz before render — never invented here. */
interface InsightViz {
  type: string;
  x: string | null;
  y: string | null;
  series: string[];
  title: string;
  interactive: boolean;
  data: Record<string, unknown>[];
  insufficient?: boolean;
  unsupported?: boolean;
}

interface Insight {
  id: string;
  type: string;
  severity: string;
  title: string;
  description: string;
  metric: string;
  metricValue?: number;
  previousValue?: number;
  magnitude: string;
  confidence: number;
  detectedAt: string;
  affectedPeriod: string;
  tags: string[];
  evidence?: InsightEvidence;
  businessImpact?: InsightBusinessImpact;
  recommendedNextAnalysis?: InsightRecommendation;
  visualization?: InsightViz;
  source?: { tables: string[]; columns: string[] };
  limitations?: string[] | string;
  /* Optional backend-provided explain fields — read only when present,
     never synthesized. */
  takeaways?: string[];
  evidenceQuality?: string;
  associatedFactors?: string[];
  recommendation?: string | { text?: string; title?: string };
  /* New conversational prose fields (backend-built, evidence-backed). */
  explains?: string;
  recommendation_detail?: { title?: string; steps?: string[]; outcome?: string; honest?: boolean };
  scenario?: { available?: boolean; label?: string } | null;
  monitoring?: { available?: boolean; metric?: string; expectedRange?: string; nextReview?: string } | null;
}

/* ---- Spec §4/§9 taxonomy: one insight can surface under several filters
   (e.g. a Growth insight is both a Trend and a Performance signal), so no
   filter looks empty while sibling views have content. ---- */
type Category = "trends" | "anomalies" | "performance" | "relationships" | "opportunities" | "risks";

const FILTERS: { id: string; label: string; cat: Category | "all" }[] = [
  { id: "all", label: "All", cat: "all" },
  { id: "trends", label: "Trends", cat: "trends" },
  { id: "anomalies", label: "Anomalies", cat: "anomalies" },
  { id: "performance", label: "Performance", cat: "performance" },
  { id: "relationships", label: "Relationships", cat: "relationships" },
  { id: "opportunities", label: "Opportunities", cat: "opportunities" },
  { id: "risks", label: "Risks", cat: "risks" },
];

/* The five hero pills required by the design; every pill is wired to the
   same matchesFilter used everywhere else — never decorative. */
const HERO_PILLS = ["all", "trends", "anomalies", "opportunities", "risks"] as const;

function matchesFilter(type: string, filter: string): boolean {
  if (filter === "all") return true;
  const t = (type || "").toLowerCase();
  switch (filter) {
    case "trends": return t.includes("trend") || t.includes("growth") || t.includes("decline");
    case "performance": return t.includes("growth") || t.includes("decline");
    case "anomalies": return t.includes("anomal") || t.includes("outlier");
    case "relationships": return t.includes("correl") || t.includes("relation");
    case "opportunities": return t.includes("opportunit") || t.includes("growth");
    case "risks": return t.includes("risk") || t.includes("decline");
    default: return true;
  }
}

function primaryCategory(type: string): Category {
  const t = (type || "").toLowerCase();
  if (t.includes("anomal") || t.includes("outlier")) return "anomalies";
  if (t.includes("opportunit")) return "opportunities";
  if (t.includes("risk")) return "risks";
  if (t.includes("correl") || t.includes("relation")) return "relationships";
  if (t.includes("growth") || t.includes("decline") || t.includes("perform")) return "performance";
  return "trends";
}

function typeLabel(type: string): string {
  const t = (type || "").toUpperCase();
  if (t === "CORRELATION") return "RELATIONSHIP";
  return t || "TREND";
}

/* ---- Semantic business labels: never leak raw snake_case / SQL internals
   to the UI. humanize("total_amt") -> "Total amt" via schema-friendly words.
   Table-qualified refs ("orders.total_amt") resolve to the column label. ---- */
function humanize(v: unknown): string {
  const raw = String(v ?? "").trim();
  if (!raw) return "";
  const s = raw.includes(".") ? raw.slice(raw.lastIndexOf(".") + 1) : raw;
  if (!s) return "";
  const spaced = s.replace(/[_-]+/g, " ").replace(/([a-z0-9])([A-Z])/g, "$1 $2").replace(/\s+/g, " ").trim();
  if (/^[A-Z0-9_()%. ]+$/.test(s) && s.length <= 8) return s.replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/* ---- Plain business language: backend prose can carry detector jargon
   (correlation, p-value, slope, variance, std, z-score, confidence
   interval, statistical terms, table.column). Rewrite to what happened /
   why it matters / what to do next. Business numbers ($, %, counts) are
   preserved; math words/numbers are removed or reworded. Charts, tooltips,
   drill-downs and API calls are untouched — only wording changes. ---- */
function containsBannedTerms(s: string): boolean {
  const t = s || "";
  if (/[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*/.test(t)) return true;
  if (/[A-Za-z0-9]+_[A-Za-z0-9]+/.test(t)) return true;
  return /correl\w*|\bp[\s-]?values?|\bslopes?|\bvariance?s?|\bcovariance?s?|\bstd\b|\bstandard\s+deviations?|\bz[\s-]?scores?|\bconfidence\s+intervals?|\bstatistic\w*|\bregressions?|\bR\^2\b|\bR-squared\b|\bOLS\b|\bIQR\b|\bANOVA\b|\bchi[\s-]?squared?|\bintercepts?|\bcoefficients?/i.test(t);
}

function plainBusiness(v: unknown): string {
  if (typeof v !== "string") return "";
  let s = v.trim();
  if (!s) return "";
  // table.column -> column label ("orders.total_amt" -> "Total amt").
  s = s.replace(/([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)/g, (_m, _t, col) => humanize(col));
  // snake_case tokens -> human labels.
  s = s.replace(/\b[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+\b/g, (m) => humanize(m));
  // Statistical assignments -> plain words or removed (no math in prose).
  s = s.replace(/\bn\s*=\s*(\d+)\b/gi, "based on $1 observations");
  s = s.replace(/\b(r|p|slope|std|var|z)\s*=\s*-?\d+(\.\d+)?\b/gi, "");
  s = s.replace(/\bCV\s*=\s*[\d.]+\b/gi, "wide spread");
  s = s.replace(/\bSkew\s*=\s*[-\d.]+\b/gi, "lopsided spread");
  // Phrase-level rewrites first (natural verbs, not single-word swaps).
  s = s.replace(/\bcorrelates?\s+with\b/gi, "is linked to");
  s = s.replace(/\bcorrelated\s+with\b/gi, "linked to");
  s = s.replace(/\bnormal\s+distributions?\b/gi, "usual range");
  s = s.replace(/\bstatistically\s+significant\b/gi, "meaningful");
  // Word-level rewrites: math term -> business term.
  s = s.replace(/\bcorrelations?\b/gi, "link");
  s = s.replace(/\bcorrel\w*\b/gi, "link");
  s = s.replace(/\bp[\s-]?values?\b/gi, "reliability check");
  s = s.replace(/\bslopes?\b/gi, "trend");
  s = s.replace(/\bcovariance?s?\b/gi, "joint movement");
  s = s.replace(/\bvariance?s?\b/gi, "spread");
  s = s.replace(/\bstd\.?\s*dev\w*/gi, "typical spread");
  s = s.replace(/\bstandard\s+deviations?\b/gi, "typical spread");
  s = s.replace(/\bstd\b\.?/gi, "typical spread");
  s = s.replace(/\bz[\s-]?scores?\b/gi, "unusual reading");
  s = s.replace(/\bconfidence\s+intervals?\b/gi, "expected range");
  s = s.replace(/\bstatistically\b/gi, "reliably");
  s = s.replace(/\bstatistical\b/gi, "data-based");
  s = s.replace(/\bstatistic\w*\b/gi, "evidence");
  s = s.replace(/\bregressions?\b/gi, "trend analysis");
  s = s.replace(/\bR\^2\b|\bR-squared\b/gi, "pattern strength");
  s = s.replace(/\bOLS\b/g, "trend analysis");
  s = s.replace(/\bIQR\b/gi, "usual range");
  s = s.replace(/\bANOVA\b/gi, "comparison");
  s = s.replace(/\bchi[\s-]?squared?\b/gi, "comparison");
  s = s.replace(/\bintercepts?\b/gi, "starting level");
  s = s.replace(/\bcoefficients?\b/gi, "link");
  s = s.replace(/\bdistributions?\b/gi, "spread");
  // Cleanup artifacts left by removed numbers.
  s = s.replace(/\s*,\s*,\s*/g, ", ");
  s = s.replace(/\(\s*,/g, "(").replace(/,\s*\)/g, ")");
  s = s.replace(/\s+([.,;:!?])/g, "$1");
  s = s.replace(/\s{2,}/g, " ").trim();
  s = s.replace(/^[,;:\-–—\s]+/, "").trim();
  return s;
}

/* Display helpers: title / description / magnitude in plain business words.
   Data identical — only wording. Falls back to humanized raw when plain
   would empty the string, so cards never render blank. */
function displayTitle(ins: Insight): string {
  const p = plainBusiness(ins.title);
  if (p && !containsBannedTerms(p)) return p;
  const h = humanize(ins.title);
  if (h && !containsBannedTerms(h)) return h;
  return "Finding";
}

function displayDescription(ins: Insight): string {
  const p = plainBusiness(ins.description);
  if (!p || containsBannedTerms(p)) return "";
  return p;
}

function displayMagnitude(ins: Insight): string {
  const p = plainBusiness(ins.magnitude);
  if (!p || containsBannedTerms(p)) return "";
  return p;
}

function displayImpact(ins: Insight): string {
  const raw = ins.businessImpact?.explanation || "";
  const p = plainBusiness(raw);
  if (p && !containsBannedTerms(p) && !isRoboticText(p)) return p;
  return "";
}

function shortMetric(metric: string): string {
  const h = humanize((metric || "").split(" (")[0]);
  const p = plainBusiness(h);
  return p || h;
}

/* ---- Real timestamps only: detectedAt -> "Today, 09:18 AM" style. ---- */
function fmtWhen(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso.slice(0, 40);
  const now = new Date();
  const day = (x: Date) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const diffDays = Math.round((day(now) - day(d)) / 86400000);
  const time = d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  if (diffDays <= 0) return `Today, ${time}`;
  if (diffDays === 1) return `Yesterday, ${time}`;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

/* ---- Source freshness: relative time derived only from the real fetch
   timestamp — never hardcoded. "just now / 4m ago / 3h ago / 2d ago". ---- */
function relTime(iso: string): string {
  if (!iso) return "";
  const t = new Date(iso).getTime();
  if (isNaN(t)) return "";
  const secs = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (secs < 45) return "just now";
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 30) return `${days}d ago`;
  return fmtWhen(iso);
}

/* ---- Key-takeaway tone -> icon: keyword tone only, no dataset values.
   Positive (growth/opportunity) gets Lightbulb/emerald, caution
   (risk/decline/drop) gets AlertTriangle/rose, everything else Sparkles/blue. ---- */
function takeawayTone(text: string): { Icon: typeof Lightbulb; wrap: string } {
  const t = (text || "").toLowerCase();
  if (/(risk|declin|drop|fall|caution|watch|concern|threat|loss)/.test(t)) {
    return { Icon: AlertTriangle, wrap: "bg-rose-50 dark:bg-rose-500/10 text-rose-600 dark:text-rose-400" };
  }
  if (/(growth|opportunit|increas|improv|gain|uplift|strong|record)/.test(t)) {
    return { Icon: Lightbulb, wrap: "bg-emerald-50 dark:bg-emerald-500/10 text-emerald-600 dark:text-emerald-400" };
  }
  return { Icon: Sparkles, wrap: "bg-blue-50 dark:bg-blue-500/10 text-blue-600 dark:text-blue-400" };
}

/* ---- Trust (§12): confidence only from computed evidence ---- */
function confidenceInfo(conf?: number): { label: string; dots: number; tone: string } {
  const c = typeof conf === "number" ? conf : -1;
  if (c >= 75) return { label: "High confidence", dots: 3, tone: "text-emerald-600 dark:text-emerald-400" };
  if (c >= 50) return { label: "Moderate confidence", dots: 2, tone: "text-blue-600 dark:text-blue-400" };
  if (c >= 30) return { label: "Limited evidence", dots: 1, tone: "text-muted-foreground" };
  return { label: "Insufficient data", dots: 0, tone: "text-muted-foreground" };
}

/* Short status-panel wording for confidence (Strong / Moderate / Limited). */
function confidenceShort(conf?: number): string {
  const c = typeof conf === "number" ? conf : -1;
  if (c >= 75) return "Strong";
  if (c >= 50) return "Moderate";
  if (c >= 30) return "Limited";
  return "Not reported";
}

function evidenceQualityOf(ins: Insight): string {
  const raw = ins.evidenceQuality ?? (ins.evidence as unknown as Record<string, unknown> | undefined)?.quality;
  if (typeof raw === "string" && raw.trim()) return plainBusiness(humanize(raw)) || humanize(raw);
  if (typeof raw === "number" && !isNaN(raw)) return confidenceShort(raw);
  return "Not reported";
}

function ConfidenceDots({ conf }: { conf?: number }) {
  const info = confidenceInfo(conf);
  return (
    <span className="inline-flex items-center gap-1.5" title={typeof conf === "number" ? `${info.label} (${conf}%)` : info.label}>
      <span className="flex items-center gap-0.5" aria-hidden>
        {[0, 1, 2].map((i) => (
          <span key={i} className={`w-1.5 h-1.5 rounded-full ${i < info.dots ? "bg-emerald-500" : "bg-border"}`} />
        ))}
      </span>
      <span className={`text-[10px] font-medium ${info.tone}`}>{info.label}</span>
    </span>
  );
}

function severityAccent(sev: string): string {
  const s = (sev || "").toLowerCase();
  if (s === "high") return "#E11D48";
  if (s === "medium") return "#4F46E5";
  return "#2563EB";
}

const fmtMoney = (v?: number | null) => {
  if (v === null || v === undefined || isNaN(Number(v))) return "—";
  const n = Number(v);
  if (Math.abs(n) >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`;
  if (Math.abs(n) >= 1_000) return `$${(n / 1_000).toFixed(0)}K`;
  return `$${n.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
};

/* ---- Copilot-style error detail: backend `detail` verbatim when it adds
   context, capped + sanitized so SQL/traces never leak to the UI. ---- */
function cleanDetail(v: unknown, fallback: string): string {
  const s = typeof v === "string" ? v.trim() : "";
  if (!s || /select\s.+\sfrom|traceback|exception|stack trace/i.test(s)) return fallback;
  return s.slice(0, 200);
}

/* ---- Legacy POST /api/insight shape (keyInsights: {title, description,
   impact}) adapted into the full Insight shape so a legacy fallback can
   never render broken cards. Weak claims stay hedged: confidence 0 renders
   as "Insufficient data", no impact is invented. ---- */
function adaptLegacyInsight(item: unknown, index: number): Insight {
  const it = (item && typeof item === "object" ? item : {}) as Record<string, unknown>;
  const str = (v: unknown): string => (typeof v === "string" ? v : "");
  const impact = str(it.impact).toLowerCase();
  const severity = impact === "high" ? "high" : impact === "medium" ? "medium" : "low";
  const title = str(it.title) || `Finding ${index + 1}`;
  const description = str(it.description) || title;
  return {
    id: str(it.id) || `legacy_${index}`,
    type: str(it.type) || "Trend",
    severity,
    title,
    description,
    metric: str(it.metric),
    magnitude: str(it.magnitude),
    confidence: typeof it.confidence === "number" ? it.confidence : 0,
    detectedAt: str(it.detectedAt) || new Date().toISOString(),
    affectedPeriod: str(it.affectedPeriod),
    tags: Array.isArray(it.tags) ? it.tags.filter((t): t is string => typeof t === "string") : [],
    evidence: { observations: [description] },
  };
}

/* ---- Chart caption: business title + what it shows ---- */
function ChartCaption({ ins }: { ins: Insight }) {
  const cat = primaryCategory(ins.type);
  const map: Record<string, [string, string]> = {
    trends: ["Movement over time", "Read left to right — the direction is the finding."],
    anomalies: ["History with the unusual point marked", "Marked points sit outside the usual range."],
    performance: ["Breakdown by group", "Tallest bar leads — hover for exact values."],
    relationships: ["How the two numbers move together", "Each dot is one period."],
    opportunities: ["Breakdown by group", "Tallest bar leads — hover for exact values."],
    risks: ["History under watch", "Flat or falling stretches deserve attention."],
  };
  const [title, sub] = map[cat] || ["Evidence chart", "Hover any mark for its exact value."];
  return (
    <div className="mb-1.5">
      <p className="text-[12px] font-semibold text-foreground">{title}</p>
      <p className="text-[10px] text-muted-foreground">{sub}</p>
    </div>
  );
}

/* Category pill meta — premium light tints, no orange anywhere. */
const CAT_META: Record<Category, { label: string; Icon: typeof TrendingUp; pill: string; iconWrap: string }> = {
  trends: { label: "Trend", Icon: TrendingUp, pill: "bg-violet-50 dark:bg-violet-500/10 text-violet-700 dark:text-violet-400 border-violet-200 dark:border-violet-500/25", iconWrap: "bg-violet-50 dark:bg-violet-500/10 text-violet-600 dark:text-violet-400" },
  anomalies: { label: "Anomaly", Icon: AlertTriangle, pill: "bg-blue-50 dark:bg-blue-500/10 text-blue-700 dark:text-blue-400 border-blue-200 dark:border-blue-500/25", iconWrap: "bg-blue-50 dark:bg-blue-500/10 text-blue-600 dark:text-blue-400" },
  performance: { label: "Performance", Icon: BarChart3, pill: "bg-sky-50 text-sky-700 border-sky-200", iconWrap: "bg-sky-50 text-sky-600" },
  relationships: { label: "Relationship", Icon: GitBranch, pill: "bg-indigo-50 dark:bg-indigo-500/10 text-indigo-700 dark:text-indigo-400 border-indigo-200 dark:border-indigo-500/25", iconWrap: "bg-indigo-50 dark:bg-indigo-500/10 text-indigo-600 dark:text-indigo-400" },
  opportunities: { label: "Opportunity", Icon: Lightbulb, pill: "bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-500/25", iconWrap: "bg-emerald-50 dark:bg-emerald-500/10 text-emerald-600 dark:text-emerald-400" },
  risks: { label: "Risk", Icon: Shield, pill: "bg-rose-50 dark:bg-rose-500/10 text-rose-700 dark:text-rose-400 border-rose-200 dark:border-rose-500/25", iconWrap: "bg-rose-50 dark:bg-rose-500/10 text-rose-600 dark:text-rose-400" },
};

function CatPill({ ins }: { ins: Insight }) {
  const meta = CAT_META[primaryCategory(ins.type)];
  const Icon = meta.Icon;
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={`inline-flex items-center justify-center w-6 h-6 rounded-full ${meta.iconWrap}`}>
        <Icon size={13} />
      </span>
      <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${meta.pill}`}>{meta.label}</span>
    </span>
  );
}

/* ================= Interactive evidence chart (recharts, spec-only) ========
   The backend spec is validated by normalizeInsightViz before render; P2
   states (empty/insufficient/unsupported) render as graceful notes. */
function InteractiveEvidenceChart({ ins, height = 260, onDrillDown }: { ins: Insight; height?: number; onDrillDown?: (sel: DrillSelection) => void }) {
  const spec = useMemo(() => normalizeInsightViz(ins.visualization, ins.title), [ins]);
  if (!ins.visualization?.data?.length || !spec) {
    const unsupported = ins.visualization?.unsupported === true;
    return (
      <p className="text-xs text-muted-foreground">
        {unsupported
          ? "This chart type isn't supported for this data — supporting values below still describe the evidence."
          : "Chart unavailable for this insight — supporting values below still describe the evidence."}
      </p>
    );
  }
  if (spec.insufficient) {
    return (
      <div>
        <InsightChart spec={spec} height={height} onDrillDown={onDrillDown} />
        <p className="mt-1 text-[11px] text-muted-foreground">Limited history — treat the shape as indicative, not conclusive.</p>
      </div>
    );
  }
  return <InsightChart spec={spec} height={height} onDrillDown={onDrillDown} />;
}

/* Compact REAL mini-chart for list cards: same validated InsightChart at a
   small height, driven only by insight.visualization — never fixed data. */
function MiniInsightChart({ ins }: { ins: Insight }) {
  const spec = useMemo(() => normalizeInsightViz(ins.visualization, ins.title), [ins]);
  if (!spec || !ins.visualization?.data?.length) return null;
  return (
    <div className="h-[132px] min-w-0 overflow-hidden">
      <InsightChart spec={spec} height={132} />
    </div>
  );
}

/* ---- Data-table alternative for the evidence chart (a11y): the same
   backend values as text rows — no new numbers, capped at 8×4. ---- */
function EvidenceDataTable({ ins }: { ins: Insight }) {
  const rows = (ins.visualization?.data || []).slice(0, 8);
  if (!rows.length) return null;
  const first = rows[0] || {};
  const cols = Object.keys(first).slice(0, 4);
  if (!cols.length) return null;
  const cell = (v: unknown): string => {
    if (typeof v === "number" && !isNaN(v)) return fmtMoney(v);
    if (typeof v === "boolean") return v ? "yes" : "no";
    return String(v ?? "—").slice(0, 40);
  };
  return (
    <details className="mt-2">
      <summary className="cursor-pointer text-[11px] font-medium text-blue-600 dark:text-blue-400 hover:underline">
        View data table
      </summary>
      <div className="mt-1.5 overflow-x-auto rounded-lg border border-border">
        <table className="w-full text-[11px]">
          <thead>
            <tr className="bg-secondary">
              {cols.map((c) => (
                <th key={c} scope="col" className="px-2.5 py-1.5 text-left font-semibold text-foreground">
                  {humanize(c)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="border-t border-border">
                {cols.map((c) => (
                  <td key={c} className="px-2.5 py-1.5 text-muted-foreground">
                    {cell(r[c])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

/* ---- Copilot handoff context (CopilotRequest.context): metric, period,
   segment/dimension, source, question. Copilot consumes it once server-side
   so follow-ups stay on THIS insight — preserved across Why/investigate and
   drill-downs (drill label travels as the segment). ---- */
function writeCopilotContext(ins: Insight, question: string, segment?: string | null): void {
  try {
    const dims = ins.recommendedNextAnalysis?.dimensions || [];
    const seg = (segment || dims[0] || "").trim() || null;
    sessionStorage.setItem("aria_copilot_context", JSON.stringify({
      metric: (ins.metric || "").split(" (")[0] || null,
      period: ins.affectedPeriod || null,
      segment: seg,
      dimension: seg,
      source: ins.source || null,
      question,
    }));
  } catch { /* context is best-effort; the question still carries over */ }
}

/* ---- Normalize backend limitations (top-level or evidence-level) into
   display lines. Reworded to plain business language; renders only when the
   backend sent them — never invented. */
function limitationLines(ins: Insight): string[] {
  const raw = ins.limitations ?? ins.evidence?.limitations;
  const list = Array.isArray(raw) ? raw : typeof raw === "string" ? [raw] : [];
  return list
    .filter((l): l is string => typeof l === "string" && l.trim().length > 0)
    .map((l) => plainBusiness(l).slice(0, 200))
    .filter((l) => l && !isRoboticText(l) && !containsBannedTerms(l));
}

/* ---- Robotic-fallback guard: sections hide when content is empty or
   still carries raw schema / SQL / engine internals or math jargon.
   plainBusiness runs first so genuine findings show in business words;
   anything still technical hides instead of leaking. Never shown. ---- */
function isRoboticText(v: unknown): boolean {
  if (typeof v !== "string") return true;
  const s = v.trim();
  if (s.length < 12) return true;
  if (/[A-Za-z0-9]+_[A-Za-z0-9]+/.test(s)) return true; // raw snake_case
  if (/[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*/.test(s)) return true; // table.column
  // Raw SQL leaks are conventionally UPPERCASE; everyday words like "from"
  // must survive in conversational prose, so this match is case-sensitive.
  if (/\b(SELECT|FROM|WHERE|GROUP BY|ORDER BY|JOIN|SQL)\b/.test(s)) return true;
  if (/\bcorrel\w*|\bp[\s-]?values?|\bslopes?|\bvariance?s?|\bcovariance?s?|\bstd\b|\bstandard\s+deviations?|\bz[\s-]?scores?|\bconfidence\s+intervals?|\bstatistic\w*|\bregressions?|\bR\^2\b|\bR-squared\b|\bOLS\b|\bIQR\b|\bANOVA\b|\bchi[\s-]?squared?|\bCV\s*=\s*[\d.]|\bSkew\s*=/i.test(s)) return true;
  if (/\bFK\b|\bforeign key\b/i.test(s)) return true;
  if (/^Ranked \d+ .* by /i.test(s)) return true; // legacy robotic template
  return false;
}

/* Key takeaways: backend takeaways when provided, reworded to plain business
   language (what happened / why it matters / what to do next); otherwise ONLY
   items the backend actually sent (observations / supporting metrics) — never
   derived numbers or invented claims. Robotic, technical or empty lines are
   dropped so the section hides instead of showing filler. */
function takeawayLines(ins: Insight): string[] {
  const clean = (t: string): string => plainBusiness(t).trim().slice(0, 240);
  const keep = (t: string): boolean => {
    if (!t || t.length < 4) return false;
    if (isRoboticText(t) || containsBannedTerms(t)) return false;
    return true;
  };
  const direct = Array.isArray(ins.takeaways) ? ins.takeaways.filter((t): t is string => typeof t === "string" && t.trim().length > 0) : [];
  if (direct.length) return direct.map(clean).filter(keep).slice(0, 4);
  const obs = (ins.evidence?.observations || []).filter((o) => typeof o === "string" && o.trim());
  const sup = (ins.evidence?.supportingMetrics || []).filter((m) => typeof m === "string" && m.trim());
  return [...obs, ...sup].map(clean).filter(keep).slice(0, 4);
}

/* ARIA explains: prefer the backend's conversational `explains` prose,
   reworded to plain business language. Returns "" when nothing meaningful
   exists so the section hides. */
function explainsTextOf(ins: Insight): string {
  const direct = typeof (ins as unknown as Record<string, unknown>).explains === "string"
    ? String((ins as unknown as Record<string, unknown>).explains).trim()
    : "";
  if (!direct) return "";
  const plain = plainBusiness(direct).slice(0, 2000);
  if (!plain || isRoboticText(plain) || containsBannedTerms(plain)) return "";
  return plain;
}

/** Split explains text into paragraphs for rendering. */
function explainsParagraphs(text: string): string[] {
  if (!text) return [];
  return text
    .split(/\n{2,}|(?<=[.!?])\s{2,}/g)
    .map((p) => p.trim())
    .filter((p) => p.length > 10);
}

function associatedFactorsOf(ins: Insight): string[] {
  const dims = (ins.recommendedNextAnalysis?.dimensions || []).filter((d) => typeof d === "string" && d.trim());
  const extra = Array.isArray(ins.associatedFactors) ? ins.associatedFactors.filter((f): f is string => typeof f === "string" && f.trim().length > 0) : [];
  const seen = new Set<string>();
  const out: string[] = [];
  for (const f of [...extra, ...dims]) {
    const k = f.toLowerCase();
    if (!seen.has(k)) { seen.add(k); out.push(f); }
  }
  return out.slice(0, 6);
}

function recommendationTextOf(ins: Insight): string {
  const clean = (v: string): string => {
    const p = plainBusiness(v).trim().slice(0, 400);
    if (!p || isRoboticText(p) || containsBannedTerms(p)) return "";
    return p;
  };
  if (typeof ins.recommendation === "string" && ins.recommendation.trim()) {
    const c = clean(ins.recommendation);
    if (c) return c;
  }
  if (ins.recommendation && typeof ins.recommendation === "object") {
    const t = (ins.recommendation.text || ins.recommendation.title || "").trim();
    if (t) {
      const c = clean(t);
      if (c) return c;
    }
  }
  const q = (ins.recommendedNextAnalysis?.question || "").trim();
  if (q) {
    const c = clean(q);
    if (c) return c;
  }
  const det = ins.recommendation_detail;
  if (det?.title) {
    const bits = [det.title, ...(det.steps || []).slice(0, 1)];
    if (det.outcome) bits.push(`Expected: ${det.outcome}`);
    const joined = bits.join(" — ");
    const c = clean(joined);
    if (c) return c;
  }
  return "";
}

function recommendationDetailOf(ins: Insight): { title: string; steps: string[]; outcome: string } | null {
  const det = ins.recommendation_detail;
  if (!det || typeof det !== "object") return null;
  const clean = (v: string, cap: number): string => {
    const p = plainBusiness(v).trim().slice(0, cap);
    if (!p || isRoboticText(p) || containsBannedTerms(p)) return "";
    return p;
  };
  const titleRaw = typeof det.title === "string" ? det.title.trim() : "";
  const title = titleRaw ? clean(titleRaw, 300) : "";
  const steps = Array.isArray(det.steps)
    ? det.steps
        .filter((s): s is string => typeof s === "string" && s.trim().length > 0)
        .map((s) => clean(s, 300))
        .filter((s) => s.length > 0)
        .slice(0, 2)
    : [];
  const outcomeRaw = typeof det.outcome === "string" ? det.outcome.trim().slice(0, 300) : "";
  const outcome = outcomeRaw ? clean(outcomeRaw, 300) : "";
  if (!title && !steps.length && !outcome) return null;
  if (!title && !steps.length) return null;
  return { title, steps: steps.slice(0, 2), outcome };
}

/* ---- Helpfulness upgrades (wording/structure only — no data, chart or
   API changes). Every helper reads only fields the backend already sent
   and returns "" when data is missing so callers render nothing. All
   copy uses plain everyday words. ---- */
function isFiniteNum(v: unknown): v is number {
  return typeof v === "number" && isFinite(v);
}

/* Plain attention label for severity codes — never shows raw high/medium/low. */
function severityPlain(sev: string): string {
  const s = (sev || "").toLowerCase().trim();
  if (s === "high") return "Needs attention now";
  if (s === "medium") return "Worth a look";
  if (s === "low") return "For your information";
  return "";
}

/* Humanized monitoring range — plain words only, "" when missing/technical. */
function expectedRangePlain(ins: Insight): string {
  const raw = ins.monitoring?.expectedRange;
  if (typeof raw !== "string" || !raw.trim()) return "";
  const p = plainBusiness(raw).trim().slice(0, 160);
  if (!p || isRoboticText(p) || containsBannedTerms(p)) return "";
  return p;
}

/* 1) Decision sentence: what changed + by how much + what to do,
   all from the insight's own numbers and recommendation. */
function decisionSentenceOf(ins: Insight): string {
  const metricLabel = shortMetric(ins.metric || "");
  const titleLabel = displayTitle(ins);
  const what = metricLabel || titleLabel || "";
  if (!what) return "";
  let how = displayMagnitude(ins).trim();
  const cp = ins.evidence?.changePercent;
  if (!how && isFiniteNum(cp) && cp !== 0) {
    const dir = cp > 0 ? "up" : "down";
    how = `${dir} ${Math.abs(cp)}%`;
  }
  const base = ins.evidence?.baseline;
  const cur = ins.evidence?.currentValue;
  if (!how && isFiniteNum(base) && isFiniteNum(cur)) {
    how = `from ${fmtMoney(base)} to ${fmtMoney(cur)}`;
  }
  if (!how) return "";
  const det = recommendationDetailOf(ins);
  const action = (det?.title || det?.steps?.[0] || recommendationTextOf(ins) || "").trim().slice(0, 160);
  if (!action) return "";
  const periodRaw = (ins.affectedPeriod || "").trim().slice(0, 60);
  const periodPlain = periodRaw ? (plainBusiness(periodRaw) || periodRaw) : "";
  if (containsBannedTerms(periodPlain) || isRoboticText(periodPlain)) return "";
  const sentence = periodPlain
    ? `${what} changed ${how} in ${periodPlain} — ${action}`
    : `${what} changed ${how} — ${action}`;
  if (!sentence || isRoboticText(sentence) || containsBannedTerms(sentence)) return "";
  return sentence.slice(0, 320);
}

/* 2) Anomaly baseline from monitoring band + evidence fields. */
function anomalyBaselineOf(ins: Insight): string {
  const t = (ins.type || "").toLowerCase();
  if (!t.includes("anomal") && !t.includes("outlier")) return "";
  const usual = expectedRangePlain(ins)
    || (isFiniteNum(ins.evidence?.baseline) ? fmtMoney(ins.evidence.baseline) : "");
  const thisTime = isFiniteNum(ins.evidence?.currentValue) ? fmtMoney(ins.evidence.currentValue) : "";
  if (!usual || !thisTime || usual === "—" || thisTime === "—") return "";
  const started = fmtWhen(ins.detectedAt || "") || (ins.affectedPeriod || "").trim().slice(0, 60);
  const lasted = isFiniteNum(ins.evidence?.sampleSize) && (ins.evidence?.sampleSize as number) > 0
    ? `${ins.evidence?.sampleSize} periods`
    : "";
  let tail = "";
  if (started && lasted) tail = ` (started ${started}, lasted ${lasted})`;
  else if (started) tail = ` (started ${started})`;
  else if (lasted) tail = ` (lasted ${lasted})`;
  const sentence = `Usually ${usual}, this time ${thisTime}${tail}`;
  if (isRoboticText(sentence) || containsBannedTerms(sentence)) return "";
  return sentence.slice(0, 320);
}

/* 3) At-stake estimate: affected records x typical value, in plain words. */
function atStakeOf(ins: Insight): string {
  const records = ins.evidence?.sampleSize;
  const typical = isFiniteNum(ins.evidence?.baseline)
    ? ins.evidence?.baseline as number
    : isFiniteNum(ins.evidence?.currentValue)
      ? ins.evidence?.currentValue as number
      : isFiniteNum(ins.metricValue)
        ? ins.metricValue as number
        : null;
  if (!isFiniteNum(records) || (records as number) < 1 || typical === null || !isFinite(typical)) return "";
  const total = (records as number) * (typical as number);
  if (!isFinite(total) || total <= 0) return "";
  return `About ${Math.round(records as number).toLocaleString()} records at roughly ${fmtMoney(typical)} each — about ${fmtMoney(total)} at stake.`;
}

/* 4) One plain "why shown" clause per evidence row. */
function whyShownFor(kind: string): string {
  const k = (kind || "").toLowerCase();
  if (k.includes("observ")) return "shown because it is what actually happened.";
  if (k.includes("compar")) return "shown to put the change in context.";
  if (k.includes("scope") || k.includes("segment")) return "shown to say where this applies.";
  if (k.includes("quality") || k.includes("strength")) return "shown to say how much proof sits behind this.";
  if (k.includes("support") || k.includes("metric")) return "shown because it backs up the change.";
  return "shown because it supports this finding.";
}

/* Shared drill context — same localStorage pattern Dashboard uses
   (localStorage `aria_dashboard_drill`), then the caller navigates. */
function storeDrillContextLocal(ctx: Record<string, unknown>): void {
  try {
    localStorage.setItem("aria_dashboard_drill", JSON.stringify({ at: Date.now(), ...ctx }));
  } catch { /* private mode: navigation still works */ }
}

/* ============================ Main page ============================ */
export default function Insights({
  insightResult = null,
  insightStatus = "idle",
  insightError = "",
  connectionStatus = "not-connected",
  dbName: propDbName = "",
  selectedSourceType: propSourceType = "",
  onAskQuestion,
}: {
  insightResult?: any;
  insightStatus?: string;
  insightError?: string;
  connectionStatus?: string;
  dbName?: string;
  selectedSourceType?: string;
  onAskQuestion?: (question: string) => void;
}) {
  const theme = useTheme() as any;
  const setPage = theme?.setPage;
  const [activeFilter, setActiveFilter] = useState<string>("all");
  const [criticalOnly, setCriticalOnly] = useState<boolean>(false);
  const [metricFilter, setMetricFilter] = useState<string>("All");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string>("changed");
  const [askText, setAskText] = useState<string>("");
  const [recentTab, setRecentTab] = useState<"recent" | "saved">("recent");
  const [savedIds, setSavedIds] = useState<string[]>([]);
  const [insights, setInsights] = useState<Insight[]>([]);
  const [samplingNote, setSamplingNote] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [errorRequestId, setErrorRequestId] = useState("");
  const [errorKind, setErrorKind] = useState("");
  const [fetchedAt, setFetchedAt] = useState<string>("");
  const [sessionInfo, setSessionInfo] = useState<{ dbName: string; sourceType: string }>({ dbName: "", sourceType: "" });
  // data-aware drill-down context (e.g. click a bar to focus a segment).
  const [drill, setDrill] = useState<{ insightId: string; label: string; value: number | string | null } | null>(null);
  // 5) Owner + review-by per recommended steps — local state only, never sent.
  const [ownerById, setOwnerById] = useState<Record<string, { owner: string; reviewBy: string }>>({});
  const abortRef = useRef<AbortController | null>(null);

  const fetchInsights = useCallback(async () => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setLoading(true);
    setError("");
    setErrorRequestId("");
    setErrorKind("");
    // Centralized error object {message, code, requestId, status} -> UI copy.
    const fail = (data: any, status: number, headers?: Headers | null) => {
      const norm = normalizeApiError(data, status, headers as any);
      setInsights([]);
      setError(norm.message);
      setErrorRequestId(norm.requestId);
      setErrorKind(
        status === 401 ? "auth"
        : status === 0 ? "timeout"
        : status >= 500 ? "server"
        : /llm|explanation/i.test(norm.code + " " + norm.message) ? "llm"
        : /insufficient|no data|unavailable/i.test(norm.code + " " + norm.message) ? "analysis"
        : "server");
    };
    try {
      const data: any = await apiFetch("/insights", { signal: ctrl.signal });
      // Accept wrapped or bare payloads; only clear with an explicit error
      // otherwise so valid-but-unwrapped shapes never read as "no insights".
      const list = data?.insights ?? data?.data ?? data?.items ?? (Array.isArray(data) ? data : undefined);
      if (Array.isArray(list)) {
        setInsights(list);
        setFetchedAt(new Date().toISOString());
        setDrill(null);
        try {
          const tables = (data as any)?.sampling?.tables || {};
          const first = Object.values(tables)[0] as any;
          setSamplingNote(
            first && first.total_rows
              ? `Computed on a ~${Number(first.sample_rows || 0).toLocaleString()}-row sample of ${Number(first.total_rows).toLocaleString()} records`
              : "",
          );
        } catch {
          setSamplingNote("");
        }
        if (!data?.ok && data?.detail) setError(data.detail);
      } else {
        setInsights([]);
        if (data?.detail) setError(data.detail);
        else setError("Insights came back in an unexpected shape — retry to reload.");
      }
    } catch (e: any) {
      if (e?.name === "AbortError") return; // cancelled refresh — not an error state
      if (e?.status === 401) { fail({ message: e?.message }, 401, null); return; }
      const status = /timed out|timeout|abort/i.test(String(e?.message || "")) ? 0 : 0;
      const norm = normalizeApiError({ message: e?.message }, status, null);
      setError(norm.message || "Failed to load insights");
      setErrorRequestId(norm.requestId);
      setErrorKind(/timed out|timeout/i.test(String(e?.message || "")) ? "timeout" : "server");
      setInsights([]);
    } finally {
      if (abortRef.current === ctrl) setLoading(false);
    }
  }, []);

  useEffect(() => () => abortRef.current?.abort(), []);

  const fetchSaved = useCallback(async () => {
    try {
      const data: any = await apiFetch("/insights/saved");
      if (data.ok && data.saved) setSavedIds(data.saved.map((s: any) => s.insight_id));
    } catch { /* non-fatal */ }
  }, []);

  useEffect(() => {
    if (connectionStatus === "connected") fetchInsights();
    fetchSaved();
  }, [connectionStatus, fetchInsights, fetchSaved]);

  /* Real session source for the data-source card: GET /api/session/status
     (props first when the host app provides them). Never hardcoded. */
  useEffect(() => {
    if (connectionStatus !== "connected") return;
    if (propDbName || propSourceType) {
      setSessionInfo({ dbName: propDbName || "", sourceType: propSourceType || "" });
      return;
    }
    (async () => {
      try {
        const body: any = await apiFetch("/session/status");
        setSessionInfo({
          dbName: String(body?.db_name || ""),
          sourceType: String(body?.source_type || body?.db_type || ""),
        });
      } catch { /* card falls back to insight source labels */ }
    })();
  }, [connectionStatus, propDbName, propSourceType]);

  // Canonical path is GET /api/insights (InsightsEngine-shaped cards with
  // {id,type,title,description,evidence,visualization}). A legacy POST
  // /api/insight prop fallback is adapted into the full shape so it can
  // never render broken cards. Refresh re-analyzes via GET (AbortController).
  const propInsights: Insight[] = Array.isArray(insightResult?.keyInsights)
    ? insightResult.keyInsights.map(adaptLegacyInsight)
    : [];
  const allInsights: Insight[] = insights.length > 0 ? insights : propInsights;

  const metrics = useMemo(() => ["All", ...Array.from(new Set(allInsights.map((i) => (i.metric || "").split(" (")[0]).filter(Boolean))).slice(0, 12)], [allInsights]);

  const filtered = useMemo(() => allInsights.filter((ins) => {
    const catOk = matchesFilter(ins.type, activeFilter);
    const critOk = !criticalOnly || (ins.severity || "").toLowerCase() === "high";
    const metricOk = metricFilter === "All" || (ins.metric || "").startsWith(metricFilter) || (ins.metric || "") === metricFilter;
    return catOk && critOk && metricOk;
  }), [allInsights, activeFilter, criticalOnly, metricFilter]);

  const selected: Insight | null = allInsights.find((i) => i.id === selectedId) || null;

  const counts = useMemo(() => {
    const c: Record<string, number> = { critical: 0 };
    for (const f of FILTERS) {
      if (f.id === "all") continue;
      c[f.id] = allInsights.filter((ins) => matchesFilter(ins.type, f.id)).length;
    }
    for (const ins of allInsights) {
      if ((ins.severity || "").toLowerCase() === "high") c.critical += 1;
    }
    return c;
  }, [allInsights]);

  const openInsight = (id: string) => {
    setSelectedId(id);
    setActiveTab("changed");
    setDrill(null);
  };

  /* ---- Real module handoff (§14): never dummy. Persist structured context
     so Analytics / Forecast open with it, primary path goes
     through Copilot (page isolation rule) or direct setPage navigation. ---- */
  const handoffContext = (ins: Insight) => ({
    insight_id: ins.id,
    insight_type: ins.type,
    title: ins.title,
    description: ins.description,
    metric: ins.metric,
    magnitude: ins.magnitude,
    confidence: ins.confidence,
    affected_period: ins.affectedPeriod,
    dimensions: ins.recommendedNextAnalysis?.dimensions || [],
    question: ins.recommendedNextAnalysis?.question || ins.description,
    source: ins.source || { tables: [], columns: [] },
  });

  const persistHandoff = (ins: Insight) => {
    try {
      sessionStorage.setItem("aria_insight_handoff", JSON.stringify(handoffContext(ins)));
    } catch { /* storage full: navigation still works */ }
    // Shared cross-module context (metric/dataset/date/filters/dimension
    // preserved; targets preselect without re-analyzing).
    try {
      saveAnalysisContext({
        origin: "insights",
        question: ins.recommendedNextAnalysis?.question || ins.description,
        metric: ins.metric || null,
        dataset: ins.source?.tables?.[0] || null,
        table: ins.source?.tables?.[0] || null,
        dateRange: ins.affectedPeriod || null,
        period: ins.affectedPeriod || null,
        dimension: ins.recommendedNextAnalysis?.dimensions?.[0] || null,
        source: ins.source || null,
      });
    } catch { /* best-effort */ }
  };

  const askCopilot = async (ins: Insight, question: string, segment?: string | null) => {
    persistHandoff(ins);
    writeCopilotContext(ins, question, segment);
    try {
      // Cheap path: send the card the user already sees so the backend can
      // skip regenerating all insights (id-only calls still work server-side).
      await apiFetch(`/insights/${ins.id}/ask`, {
        method: "POST",
        body: JSON.stringify({ question, insight_id: ins.id, insight: ins }),
      });
    } catch (e: any) {
      // Backend trace is best-effort; Copilot still runs. Log Copilot-style
      // detail instead of swallowing it silently.
      if (e?.name !== "AbortError") console.debug("[ARIA] insight ask trace failed:", e?.message);
    }
    if (onAskQuestion) onAskQuestion(question);
  };

  // Hypothesis capability lives inside AI Analytics as Possible drivers
  // (no standalone page): hand the observed metric/segment/period over via
  // the shared context key and open the analysis there.
  const openHypothesis = (ins: Insight) => {
    persistHandoff(ins);
    const dims = ins.recommendedNextAnalysis?.dimensions || [];
    const q0 = ins.recommendedNextAnalysis?.question || `Why did ${shortMetric(ins.metric) || "this metric"} change${ins.magnitude ? ` (${ins.magnitude})` : ""}?`;
    const q = ins.affectedPeriod && !q0.includes(ins.affectedPeriod) ? `${q0} Focus on ${ins.affectedPeriod}.` : q0;
    // Copilot context preserved across Why/investigate (CopilotRequest.context).
    writeCopilotContext(ins, q, dims[0] || null);
    try {
      const prev = JSON.parse(sessionStorage.getItem("aria_shared_context") || "{}");
      sessionStorage.setItem("aria_shared_context", JSON.stringify({
        ...prev, question: q, metric: ins.metric, dimensions: dims,
        period: ins.affectedPeriod || prev.period, origin: "insights",
        updatedAt: Date.now(),
      }));
      sessionStorage.setItem("aria_insight_handoff", JSON.stringify({
        origin: "insights", question: q, metric: ins.metric,
      }));
    } catch { /* ignore */ }
    if (setPage) setPage("analytics");
    else askCopilot(ins, `Why did ${shortMetric(ins.metric) || "this metric"} change? ${q}`);
  };

  const openForecast = (ins: Insight) => {
    persistHandoff(ins);
    try {
      sessionStorage.setItem("aria_forecast_prefill", JSON.stringify({ metric: ins.metric, period: ins.affectedPeriod, source: ins.source }));
      // Shared Forecast context: metric + dataset + date preselected on the
      // Forecast page (localStorage key, consumed once).
      localStorage.setItem("aria_forecast_context", JSON.stringify({
        metric: ins.metric,
        dataset: ins.source?.tables?.[0] || undefined,
        date: ins.affectedPeriod || undefined,
        filters: [],
        question: ins.recommendedNextAnalysis?.question || ins.description,
      }));
    } catch { /* ignore */ }
    if (setPage) setPage("predictive");
    else askCopilot(ins, `Forecast ${shortMetric(ins.metric) || "this metric"} over the next periods given: ${ins.description}`);
  };

  /* 7) Cross-links per insight — all reuse existing handoff mechanisms:
     dashboard via the drill-context pattern (localStorage
     `aria_dashboard_drill`), forecast via the existing forecast prefill. */
  const openDashboard = (ins: Insight) => {
    persistHandoff(ins);
    const dims = ins.recommendedNextAnalysis?.dimensions || [];
    storeDrillContextLocal({
      kind: "insight",
      insight_id: ins.id,
      ...(ins.metric ? { metric: ins.metric } : {}),
      ...(dims[0] ? { dimension: dims[0], breakdown: dims[0] } : {}),
      ...(ins.affectedPeriod ? { period: ins.affectedPeriod } : {}),
      ...(ins.source?.tables?.[0] ? { dataset: ins.source.tables[0], table: ins.source.tables[0] } : {}),
    });
    if (setPage) setPage("dashboard");
  };

  const toggleSave = async (ins: Insight) => {
    const isSaved = savedIds.includes(ins.id);
    try {
      if (isSaved) {
        await apiFetch(`/insights/save/${ins.id}`, { method: "DELETE" });
        setSavedIds((s) => s.filter((id) => id !== ins.id));
      } else {
        await apiFetch("/insights/save", {
          method: "POST",
          body: JSON.stringify({ insight_id: ins.id, title: ins.title, insight_type: ins.type, severity: ins.severity, metric: ins.metric, insight_payload: ins }),
        });
        setSavedIds((s) => [...s, ins.id]);
      }
    } catch { /* non-fatal */ }
  };

  const handleShare = async (ins: Insight) => {
    const text = `${displayTitle(ins)}\n\n${displayDescription(ins)}\n\nMetric: ${shortMetric(ins.metric) || "—"}\nConfidence: ${typeof ins.confidence === "number" ? `${ins.confidence}%` : "not reported"}`;
    try {
      if (navigator.share) await navigator.share({ title: displayTitle(ins), text });
      else await navigator.clipboard.writeText(text);
    } catch { /* dismissed */ }
  };

  /* Ask ARIA Anything (bottom bar): general question goes through the
     existing Copilot path with an insights-origin context. */
  const submitAskBar = () => {
    const q = askText.trim();
    if (!q) return;
    try {
      sessionStorage.setItem("aria_copilot_context", JSON.stringify({ origin: "insights", question: q }));
    } catch { /* best-effort */ }
    setAskText("");
    if (onAskQuestion) onAskQuestion(q);
  };

  const displayDbName = sessionInfo.dbName || propDbName || "";
  const displaySourceType = sessionInfo.sourceType || propSourceType || "";

  const recentItems = useMemo(() => {
    const list = recentTab === "saved" ? allInsights.filter((i) => savedIds.includes(i.id)) : filtered;
    return list.slice(0, 6);
  }, [recentTab, allInsights, filtered, savedIds]);

  /* ------------------------- Failure / empty states ------------------------- */
  if (connectionStatus !== "connected") {
    return (
      <div className="min-h-full bg-card text-foreground">
        <div className="p-6 max-w-screen-xl mx-auto">
          <p className="text-[11px] font-bold tracking-[0.14em] text-blue-600 dark:text-blue-400">ARIA INSIGHTS</p>
          <h2 className="text-xl font-semibold tracking-tight">Insights</h2>
          <p className="text-sm text-muted-foreground mt-0.5">ARIA automatically finds meaningful patterns in your connected data.</p>
          <div className="mt-8 text-center py-12 bg-card border border-border rounded-2xl shadow-sm">
            <Database size={22} className="mx-auto text-muted-foreground mb-3" />
            <p className="text-sm font-medium">No data source connected</p>
            <p className="text-xs text-muted-foreground mt-1">Connect PostgreSQL, MySQL, or upload a CSV to let ARIA investigate your data.</p>
          </div>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="min-h-full bg-card text-foreground" aria-busy="true">
        <div className="p-6 max-w-screen-xl mx-auto">
          <p className="sr-only">Analyzing your data — loading insights…</p>
          <div className="rounded-2xl border border-border bg-gradient-to-br from-blue-50 dark:from-blue-500/10 via-card to-indigo-50 dark:to-indigo-500/10 p-6 animate-pulse" aria-hidden>
            <div className="h-3 w-24 rounded bg-border" />
            <div className="mt-3 h-6 w-2/3 rounded bg-border" />
            <div className="mt-2 h-4 w-1/2 rounded bg-secondary" />
            <div className="mt-4 flex gap-2">
              <div className="h-7 w-20 rounded-full bg-border" />
              <div className="h-7 w-20 rounded-full bg-secondary" />
              <div className="h-7 w-24 rounded-full bg-secondary" />
            </div>
          </div>
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4" aria-hidden>
            {[0, 1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="rounded-2xl border border-border bg-card p-4 shadow-sm animate-pulse">
                <div className="flex items-center gap-2">
                  <div className="w-6 h-6 rounded-full bg-border" />
                  <div className="h-3 w-20 rounded-full bg-border" />
                </div>
                <div className="mt-3 h-4 w-3/4 rounded bg-border" />
                <div className="mt-2 h-3 w-full rounded bg-secondary" />
                <div className="mt-2 h-3 w-5/6 rounded bg-secondary" />
                <div className="mt-3 h-28 rounded-xl bg-secondary" />
                <div className="mt-3 flex items-center gap-2">
                  <div className="h-3 w-24 rounded bg-secondary" />
                  <div className="ml-auto h-3 w-20 rounded bg-border" />
                </div>
              </div>
            ))}
          </div>
          <div className="text-center py-6">
            <button onClick={() => abortRef.current?.abort()} className="text-xs text-muted-foreground hover:text-foreground underline">Cancel loading</button>
          </div>
        </div>
      </div>
    );
  }

  if ((error || insightError) && !allInsights.length) {
    const msg = error || insightError;
    const kindLabel = !error ? "" : errorKind === "auth" ? "Session expired — sign in again."
      : errorKind === "timeout" ? "Request timed out."
      : errorKind === "server" ? "Server error."
      : errorKind === "llm" ? "Explanation unavailable."
      : errorKind === "analysis" ? "Analysis unavailable."
      : "";
    return (
      <div className="min-h-full bg-card text-foreground">
        <div className="p-6 max-w-screen-xl mx-auto">
          <p className="text-[11px] font-bold tracking-[0.14em] text-blue-600 dark:text-blue-400">ARIA INSIGHTS</p>
          <h2 className="text-xl font-semibold tracking-tight">Insights</h2>
          <p className="text-sm text-muted-foreground mt-0.5">ARIA automatically finds meaningful patterns in your connected data.</p>
          <div className="mt-6 bg-card border border-border rounded-2xl p-5 flex gap-3 shadow-sm">
            <AlertTriangle size={18} className="text-rose-500 dark:text-rose-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium">ARIA couldn&apos;t generate insights for this dataset{kindLabel ? ` — ${kindLabel}` : ""}</p>
              <p className="text-xs text-muted-foreground mt-1">{msg}</p>
              {errorRequestId && (
                <p className="text-[11px] text-muted-foreground mt-1 font-mono">Request ID: {errorRequestId}</p>
              )}
              <p className="text-xs text-muted-foreground mt-1">ARIA needs a measurable metric with enough historical observations. Check for empty tables, missing date fields, or excessive nulls.</p>
              <button onClick={fetchInsights} className="mt-2 text-xs text-blue-600 dark:text-blue-400 hover:underline">Try again</button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!allInsights.length) {
    return (
      <div className="min-h-full bg-card text-foreground">
        <div className="p-6 max-w-screen-xl mx-auto">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-[11px] font-bold tracking-[0.14em] text-blue-600 dark:text-blue-400">ARIA INSIGHTS</p>
              <h2 className="text-xl font-semibold tracking-tight">What did ARIA learn today?</h2>
              <p className="text-sm text-muted-foreground mt-0.5">ARIA automatically finds meaningful patterns in your connected data.</p>
            </div>
            <button onClick={fetchInsights} className="p-2 rounded-lg hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors" title="Refresh Analysis">
              <RefreshCw size={14} />
            </button>
          </div>
          <div className="mt-8 text-center py-12 bg-card border border-border rounded-2xl shadow-sm">
            <Sparkles size={22} className="mx-auto text-muted-foreground mb-3" />
            <p className="text-sm font-medium">ARIA hasn&apos;t identified any strong patterns yet.</p>
            <p className="text-xs text-muted-foreground mt-1 max-w-md mx-auto">Weak evidence is never promoted into a business claim. Try a broader question in AI Analytics.</p>
            <button onClick={() => setPage?.("analytics")} className="mt-3 text-xs font-medium text-blue-600 dark:text-blue-400 hover:underline">
              Is there anything unusual about this dataset?
            </button>
          </div>
        </div>
      </div>
    );
  }

  /* ============================ Shared pieces ============================ */
  const heroPills = (
    <div className="mt-5 flex items-center gap-2 flex-wrap">
      {HERO_PILLS.map((id) => {
        const f = FILTERS.find((x) => x.id === id)!;
        const n = id === "all" ? allInsights.length : counts[id] ?? 0;
        const on = activeFilter === id;
        return (
          <button
            key={id}
            onClick={() => { setActiveFilter(id); }}
            aria-pressed={on}
            className={`text-[11px] font-semibold px-3.5 py-1.5 rounded-full border transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-1 ${on ? "bg-blue-700 text-white border-blue-700 shadow-md ring-1 ring-blue-700/30" : "bg-white/80 text-secondary-foreground border-border hover:border-blue-300 dark:hover:border-blue-500/25 hover:text-foreground hover:shadow-sm"}`}
          >
            {f.label}{id !== "all" ? ` · ${n}` : ""}
          </button>
        );
      })}
    </div>
  );

  const toolbar = (
    <div className="mt-4 flex items-center gap-2 flex-wrap">
      <p className="text-[12px] text-secondary-foreground mr-1">
        <span className="font-semibold text-foreground">{filtered.length} of {allInsights.length}</span> insights
        {criticalOnly ? " · critical only" : ""}
        {metricFilter !== "All" ? ` · ${humanize(metricFilter)}` : ""}
      </p>
      <button
        onClick={() => setCriticalOnly((v) => !v)}
        aria-pressed={criticalOnly}
        className={`text-[11px] font-semibold px-2.5 py-1 rounded-full border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-400 ${criticalOnly ? "bg-rose-600 text-white border-rose-600 shadow-sm" : "bg-card text-muted-foreground border-border hover:text-foreground hover:border-rose-200 dark:hover:border-rose-500/25"}`}
      >
        Critical · {counts.critical}
      </button>
      <div className="ml-auto flex items-center gap-1.5 min-w-0">
        <Filter size={12} className="text-muted-foreground flex-shrink-0" />
        <select value={metricFilter} onChange={(e) => setMetricFilter(e.target.value)} className="text-[11px] bg-card border border-border rounded-lg px-2 py-1.5 text-foreground max-w-[180px]" aria-label="Filter by metric">
          {metrics.map((m) => <option key={m} value={m}>{m === "All" ? "All metrics" : humanize(m)}</option>)}
        </select>
        <button onClick={fetchInsights} className="p-1.5 rounded-lg hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors" title="Refresh Analysis" aria-label="Refresh analysis">
          <RefreshCw size={13} />
        </button>
      </div>
    </div>
  );

  const dataSourceCard = (
    <div className="rounded-2xl border border-border bg-card p-4 shadow-sm">
      <div className="flex items-center gap-2">
        <span className="inline-flex items-center justify-center w-8 h-8 rounded-xl bg-blue-50 dark:bg-blue-500/10 text-blue-600 dark:text-blue-400">
          <Database size={15} />
        </span>
        <p className="text-[12px] font-semibold flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 inline-block" />
          {displayDbName ? "Data Connected" : "Connected source"}
        </p>
      </div>
      <button onClick={() => setPage?.("data")} className="mt-3 w-full text-left rounded-xl hover:bg-secondary p-2 -m-1 transition-colors" title="Open data center">
        <p className="text-[13px] font-semibold truncate">{displayDbName || selected?.source?.tables?.[0] || "Connected source"}</p>
        {displaySourceType && <p className="text-[11px] text-muted-foreground">{humanize(displaySourceType)}</p>}
      </button>
      <div className="mt-2 pt-2 border-t border-border flex items-center justify-between text-[11px] text-muted-foreground">
        <span>Last analyzed</span>
        <span className="font-medium text-secondary-foreground" title={fetchedAt ? fmtWhen(fetchedAt) : undefined}>{fetchedAt ? relTime(fetchedAt) : "—"}</span>
      </div>
      {samplingNote && (
        <p className="mt-1.5 text-[10.5px] text-muted-foreground" title="Large datasets are analyzed on a bounded sample; conclusions are indicative, not row-exact.">{samplingNote} · sample-based</p>
      )}
      <button onClick={() => setPage?.("data")} className="mt-1.5 text-[11px] font-semibold text-blue-600 dark:text-blue-400 hover:underline inline-flex items-center gap-1">
        View data center <ChevronRight size={12} />
      </button>
    </div>
  );

  const recentCard = (
    <div className="rounded-2xl border border-border bg-card shadow-sm overflow-hidden">
      <div className="flex items-center gap-4 px-4 pt-3 text-[12px] font-semibold border-b border-border">
        {(["recent", "saved"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setRecentTab(t)}
            className={`pb-2.5 -mb-px border-b-2 transition-colors ${recentTab === t ? "border-blue-600 text-foreground" : "border-transparent text-muted-foreground hover:text-secondary-foreground"}`}
          >
            {t === "recent" ? "Recent Insights" : `Saved${savedIds.length ? ` · ${savedIds.length}` : ""}`}
          </button>
        ))}
      </div>
      <div className="divide-y divide-border">
        {recentItems.length === 0 && (
          <p className="px-4 py-5 text-[11px] text-muted-foreground">{recentTab === "saved" ? "No saved insights yet — tap the bookmark on any insight." : "No insights under the current filters."}</p>
        )}
        {recentItems.map((ins) => {
          const meta = CAT_META[primaryCategory(ins.type)];
          const Icon = meta.Icon;
          return (
            <button key={ins.id} onClick={() => openInsight(ins.id)} className="w-full text-left px-4 py-2.5 flex items-center gap-2.5 hover:bg-blue-50/50 dark:hover:bg-blue-500/10 transition-colors group">
              <span className={`inline-flex items-center justify-center w-7 h-7 rounded-full flex-shrink-0 ${meta.iconWrap}`}>
                <Icon size={13} />
              </span>
              <span className="min-w-0 flex-1">
                <span className="block text-[12px] font-medium text-foreground leading-snug line-clamp-2 group-hover:text-blue-700 dark:group-hover:text-blue-400">{displayTitle(ins)}</span>
                <span className="block text-[10px] text-muted-foreground mt-0.5">{fmtWhen(ins.detectedAt)}</span>
              </span>
              <ChevronRight size={13} className="text-slate-300 dark:text-muted-foreground group-hover:text-blue-500 dark:group-hover:text-blue-400 flex-shrink-0" />
            </button>
          );
        })}
      </div>
      <button onClick={() => { setActiveFilter("all"); setMetricFilter("All"); setCriticalOnly(false); setRecentTab("recent"); }} className="w-full text-left px-4 py-2.5 text-[11px] font-semibold text-blue-600 dark:text-blue-400 hover:underline">
        View all insights →
      </button>
    </div>
  );

  const askBar = (
    <div className="rounded-2xl border border-blue-100 dark:border-blue-500/20 bg-gradient-to-r from-blue-50/80 dark:from-blue-500/10 to-indigo-50/80 dark:to-indigo-500/10 p-4 flex flex-col sm:flex-row sm:items-center gap-3">
      <div className="flex items-center gap-3 min-w-0 sm:w-64 flex-shrink-0">
        <span className="inline-flex items-center justify-center w-10 h-10 rounded-full bg-blue-100 dark:bg-blue-500/15 text-blue-700 dark:text-blue-400 flex-shrink-0">
          <Sparkles size={16} />
        </span>
        <span className="min-w-0">
          <span className="block text-[13px] font-semibold">Ask ARIA Anything</span>
          <span className="block text-[11px] text-muted-foreground leading-snug">Get deeper insights, ask follow-up questions, or explore your data in natural language.</span>
        </span>
      </div>
      <div className="flex items-center gap-2 flex-1 min-w-0 bg-card border border-border rounded-xl pl-3 pr-1.5 py-1.5 shadow-sm focus-within:border-blue-400">
        <MessageSquare size={14} className="text-muted-foreground flex-shrink-0" />
        <input
          value={askText}
          onChange={(e) => setAskText(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") submitAskBar(); }}
          placeholder="Ask a question about your data…"
          className="flex-1 min-w-0 bg-transparent text-[12px] text-foreground placeholder:text-muted-foreground outline-none"
          aria-label="Ask ARIA anything"
        />
        <button onClick={submitAskBar} disabled={!askText.trim()} className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-40 transition-colors flex-shrink-0" title="Ask Copilot" aria-label="Send question">
          <Send size={14} />
        </button>
      </div>
    </div>
  );

  /* ============================ EXPLAIN VIEW ============================ */
  if (selected) {
    const selChange = selected.evidence?.changePercent;
    const impact = selected.businessImpact;
    const takeaways = takeawayLines(selected);
    const factors = associatedFactorsOf(selected);
    const recoText = recommendationTextOf(selected);
    const recoDetail = recommendationDetailOf(selected);
    const explains = explainsTextOf(selected);
    const explainsParas = explainsParagraphs(explains);
    const dims = selected.recommendedNextAnalysis?.dimensions || [];
    const affectedSegment = dims[0] || selected.tags?.[0] || "";
    const scenarioAvailable = selected.scenario?.available === true;
    const monitorAvailable = selected.monitoring?.available === true;
    const lims = limitationLines(selected);
    const meta = CAT_META[primaryCategory(selected.type)];
    // Helpfulness derivations — each guards on its own data, "" hides.
    const decisionSentence = decisionSentenceOf(selected);
    const anomalyBaseline = anomalyBaselineOf(selected);
    const atStake = atStakeOf(selected);
    const attentionLabel = severityPlain(selected.severity);
    const rangePlain = expectedRangePlain(selected);
    const ownerState = ownerById[selected.id] || { owner: "", reviewBy: "" };
    const canSeeDashboard = Boolean((selected.metric || "").trim() || selected.source?.tables?.length);
    const canForecast = Boolean((selected.metric || "").trim());

    const tabs: { id: string; label: string }[] = [
      { id: "changed", label: "What Changed" },
      { id: "why", label: "Why It Matters" },
      { id: "evidence", label: "Evidence" },
      ...(factors.length ? [{ id: "factors", label: "Associated Factors" }] : []),
      ...(recoDetail || recoText ? [{ id: "reco", label: "Recommendation" }] : []),
      { id: "whatif", label: "What If" },
      { id: "monitor", label: "Monitor" },
    ];
    const tabId = tabs.some((t) => t.id === activeTab) ? activeTab : "changed";

    return (
      <div className="min-h-full bg-card text-foreground overflow-x-hidden overflow-x-clip scroll-smooth">
        <div className="p-4 md:p-6 max-w-[1400px] mx-auto min-w-0 max-w-full">
          {/* Breadcrumb + back */}
          <div className="flex items-center gap-2 text-[12px] flex-wrap">
            <button onClick={() => setSelectedId(null)} className="inline-flex items-center gap-1.5 font-semibold text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 transition-colors" aria-label="Back to insights">
              <ArrowLeft size={13} /> Insights
            </button>
            <span className="text-slate-300 dark:text-muted-foreground">›</span>
            <span className="text-muted-foreground truncate max-w-[60vw]">{displayTitle(selected)}</span>
            <span className="ml-auto flex items-center gap-1">
              <button onClick={() => toggleSave(selected)} className={`p-1.5 rounded-lg transition-colors ${savedIds.includes(selected.id) ? "text-blue-600 dark:text-blue-400" : "text-muted-foreground hover:text-foreground"} hover:bg-secondary`} title={savedIds.includes(selected.id) ? "Unsave insight" : "Save insight"} aria-label="Save insight">
                <Bookmark size={15} fill={savedIds.includes(selected.id) ? "currentColor" : "none"} />
              </button>
              <button onClick={() => handleShare(selected)} className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors" title="Share" aria-label="Share insight">
                <Share2 size={15} />
              </button>
            </span>
          </div>

          <div className="mt-3 grid grid-cols-1 lg:grid-cols-[300px_minmax(0,1fr)] gap-4 items-start">
            {/* Left: compact insight list */}
            <div className="rounded-2xl border border-border bg-card shadow-sm overflow-hidden">
              <div className="px-4 pt-4 pb-3 border-b border-border">
                <p className="text-[11px] font-bold tracking-[0.14em] text-blue-600 dark:text-blue-400">ARIA INSIGHTS</p>
                <p className="text-[15px] font-semibold leading-snug mt-1">What did ARIA learn today?</p>
                <div className="mt-2.5 flex items-center gap-1.5 flex-wrap">
                  {HERO_PILLS.map((id) => (
                    <button
                      key={id}
                      onClick={() => setActiveFilter(id)}
                      aria-pressed={activeFilter === id}
                      className={`text-[10px] font-semibold px-2.5 py-1 rounded-full border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${activeFilter === id ? "bg-blue-700 text-white border-blue-700 shadow-sm" : "bg-card text-muted-foreground border-border hover:text-foreground hover:border-blue-300 dark:hover:border-blue-500/25"}`}
                    >
                      {FILTERS.find((x) => x.id === id)!.label}
                    </button>
                  ))}
                </div>
              </div>
              <div className="divide-y divide-border max-h-[70vh] overflow-y-auto">
                {filtered.map((ins) => {
                  const m = CAT_META[primaryCategory(ins.type)];
                  const Icon = m.Icon;
                  const on = ins.id === selected.id;
                  return (
                    <button key={ins.id} onClick={() => openInsight(ins.id)} className={`w-full text-left px-4 py-3 transition-colors ${on ? "bg-blue-50/70 dark:bg-blue-500/10 border-l-2 border-l-blue-600" : "hover:bg-secondary border-l-2 border-l-transparent"}`}>
                      <span className="flex items-center gap-2">
                        <span className={`inline-flex items-center justify-center w-6 h-6 rounded-full flex-shrink-0 ${m.iconWrap}`}>
                          <Icon size={12} />
                        </span>
                        <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${m.pill}`}>{m.label}</span>
                      </span>
                      <span className="block mt-1.5 text-[12px] font-semibold leading-snug">{displayTitle(ins)}</span>
                      <span className="block mt-0.5 text-[11px] text-muted-foreground leading-snug line-clamp-2">{displayDescription(ins)}</span>
                      <span className="mt-1.5 flex items-center gap-1 text-[10px] text-muted-foreground">
                        <Clock size={10} /> {fmtWhen(ins.detectedAt)}
                        <ChevronRight size={11} className="ml-auto" />
                      </span>
                    </button>
                  );
                })}
                {filtered.length === 0 && (
                  <p className="px-4 py-6 text-[11px] text-muted-foreground">No insights match these filters.</p>
                )}
              </div>
            </div>

            {/* Right: explain panel */}
            <article key={selected.id} className="min-w-0 max-w-full rounded-2xl border border-border bg-card shadow-sm overflow-hidden overflow-x-clip">
              {/* Hero */}
              <div className="px-5 md:px-6 pt-6 pb-5 border-b border-border bg-gradient-to-br from-blue-50/60 dark:from-blue-500/10 via-card to-indigo-50/40 dark:to-indigo-500/10">
                <div className="flex flex-col md:flex-row gap-4 md:gap-5">
                  <div className="min-w-0 flex-1 max-w-full">
                    <CatPill ins={selected} />
                    <h1 className="mt-2.5 text-[22px] md:text-[24px] font-semibold leading-[1.2] tracking-tight text-balance break-words">{displayTitle(selected)}</h1>
                    <p className="mt-2 text-[13px] md:text-[14px] text-secondary-foreground leading-relaxed break-words">{displayDescription(selected)}</p>
                    {decisionSentence ? (
                      <p className="mt-2.5 rounded-xl border border-blue-100 dark:border-blue-500/20 bg-blue-50/60 dark:bg-blue-500/10 px-3 py-2 text-[12.5px] font-medium leading-relaxed text-foreground">{decisionSentence}</p>
                    ) : null}
                    {anomalyBaseline ? (
                      <p className="mt-1.5 text-[11.5px] text-secondary-foreground leading-relaxed">{anomalyBaseline}</p>
                    ) : null}
                    <div className="mt-2.5"><ConfidenceDots conf={selected.confidence} /></div>
                  </div>
                  {/* Status panel — real fields only, semantic labels */}
                  <dl className="flex-shrink-0 md:w-56 rounded-xl border border-border bg-card p-3.5 grid grid-cols-2 md:grid-cols-1 gap-x-4 gap-y-2.5 text-[11px] shadow-sm">
                    <div className="flex md:justify-between gap-2"><dt className="text-muted-foreground">Confidence</dt><dd className="font-semibold ml-auto">{confidenceShort(selected.confidence)}</dd></div>
                    <div className="flex md:justify-between gap-2"><dt className="text-muted-foreground">Evidence Quality</dt><dd className="font-semibold ml-auto">{evidenceQualityOf(selected)}</dd></div>
                    {attentionLabel ? (
                      <div className="flex md:justify-between gap-2"><dt className="text-muted-foreground">Attention</dt><dd className="font-semibold ml-auto text-right">{attentionLabel}</dd></div>
                    ) : null}
                    <div className="flex md:justify-between gap-2"><dt className="text-muted-foreground">Period</dt><dd className="font-semibold ml-auto text-right">{selected.affectedPeriod || "—"}</dd></div>
                    <div className="flex md:justify-between gap-2"><dt className="text-muted-foreground">Affected Segment</dt><dd className="font-semibold ml-auto text-right">{affectedSegment ? plainBusiness(humanize(affectedSegment)) || humanize(affectedSegment) : "—"}</dd></div>
                  </dl>
                </div>
              </div>

              <div className="px-5 py-4 space-y-4 min-w-0 max-w-full overflow-x-clip">
                {/* Main visual + takeaways — stacks on mobile, side-by-side on xl */}
                <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_260px] gap-4 items-start min-w-0 max-w-full">
                  <div className="min-w-0 max-w-full overflow-hidden rounded-xl border border-border bg-card p-4">
                    <div className="flex items-center gap-2">
                      <span className={`inline-flex items-center justify-center w-7 h-7 rounded-lg ${meta.iconWrap}`}>
                        <meta.Icon size={14} />
                      </span>
                      <p className="text-[13px] font-semibold truncate">{plainBusiness(selected.visualization?.title) || selected.visualization?.title || `${shortMetric(selected.metric) || "Evidence"} vs ${humanize(selected.affectedPeriod) || "baseline"}`}</p>
                    </div>
                    <div className="mt-2 min-w-0">
                      <ChartCaption ins={selected} />
                      {drill && drill.insightId === selected.id && (
                        <div className="mb-2 flex items-center gap-2 rounded-lg border border-blue-200 dark:border-blue-500/25 bg-blue-50 dark:bg-blue-500/10 px-2.5 py-1.5 text-[11px]">
                          <span>
                            Context: <span className="font-semibold">{drill.label}</span>
                            {typeof drill.value === "number" ? ` · ${fmtMoney(drill.value)}` : ""}
                          </span>
                          <button
                            onClick={() => askCopilot(selected, `Break down "${drill.label}" further: trend and top contributors within this segment for ${shortMetric(selected.metric) || "this metric"}.`, drill.label)}
                            className="ml-auto font-medium text-blue-600 dark:text-blue-400 hover:underline"
                          >
                            Investigate {drill.label}
                          </button>
                          <button onClick={() => setDrill(null)} className="text-muted-foreground hover:text-foreground" title="Clear drill-down" aria-label="Clear drill-down">
                            <XCircle size={13} />
                          </button>
                        </div>
                      )}
                      <InteractiveEvidenceChart
                        ins={selected}
                        height={300}
                        onDrillDown={(sel) => setDrill({ insightId: selected.id, label: sel.label, value: sel.value })}
                      />
                      <p className="mt-1.5 text-[10px] text-muted-foreground">Tip: hover for exact values · drag the mini-map to zoom a time range · click legend entries to toggle series · click a mark to drill down.</p>
                    </div>
                  </div>

                  {/* Key Takeaways — only items the backend actually sent */}
                  <aside className="rounded-xl border border-border bg-card p-4 min-w-0 max-w-full">
                    <p className="text-[13px] font-semibold flex items-center gap-1.5">
                      <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-blue-50 dark:bg-blue-500/10 text-blue-600 dark:text-blue-400"><TrendingUp size={12} /></span>
                      Key Takeaways
                    </p>
                    {takeaways.length === 0 && (
                      <p className="mt-2 text-[11px] text-muted-foreground">No additional takeaways reported for this insight.</p>
                    )}
                    <ul className="mt-2 space-y-3">
                      {takeaways.map((t, i) => {
                        const tone = takeawayTone(t);
                        const ToneIcon = tone.Icon;
                        return (
                          <li key={i} className="flex gap-2.5 text-[11px] leading-snug">
                            <span className={`inline-flex items-center justify-center w-6 h-6 rounded-full flex-shrink-0 ${tone.wrap}`}>
                              <ToneIcon size={12} />
                            </span>
                            <span className="text-secondary-foreground break-words min-w-0">{t}</span>
                          </li>
                        );
                      })}
                    </ul>
                  </aside>
                </div>

                {/* ARIA explains — backend conversational prose (`explains`)
                    when present; hidden entirely when empty or robotic.
                    Renders multi-paragraph explanations when available. */}
                {explains && explainsParas.length > 0 ? (
                  <div className="rounded-xl border border-blue-100 dark:border-blue-500/20 bg-blue-50/40 dark:bg-blue-500/10 p-4">
                    <p className="text-[13px] font-semibold flex items-center gap-1.5 text-foreground">
                      <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-blue-600 text-white"><Sparkles size={12} /></span>
                      ARIA explains
                    </p>
                    <div className="mt-2 space-y-2">
                      {explainsParas.map((para, i) => (
                        <p key={i} className="text-[12.5px] text-secondary-foreground leading-relaxed">
                          {para}
                        </p>
                      ))}
                    </div>
                  </div>
                ) : null}

                {/* Tabs — sticky, smooth-scroll aware, no horizontal page overflow */}
                <div className="sticky top-0 z-10 -mx-5 px-5 bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/85 border-b border-border flex gap-4 overflow-x-auto scroll-smooth" role="tablist" aria-label="Insight explanation sections">
                  {tabs.map((t) => (
                    <button
                      key={t.id}
                      role="tab"
                      aria-selected={tabId === t.id}
                      onClick={() => {
                        setActiveTab(t.id);
                        requestAnimationFrame(() => {
                          document.getElementById("insight-tab-panel")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
                        });
                      }}
                      className={`pb-2 pt-2.5 text-[12px] font-semibold whitespace-nowrap border-b-2 -mb-px transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded-t ${tabId === t.id ? "border-blue-600 text-blue-700 dark:text-blue-400" : "border-transparent text-muted-foreground hover:text-secondary-foreground"}`}
                    >
                      {t.label}
                    </button>
                  ))}
                </div>
                <div id="insight-tab-panel" className="scroll-mt-16 min-w-0 max-w-full"></div>

                {tabId === "changed" && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div className="rounded-xl border border-border p-4 bg-gradient-to-br from-blue-50/60 dark:from-blue-500/10 to-card">
                      <p className="text-[11px] font-semibold text-muted-foreground flex items-center gap-1.5"><TrendingUp size={12} className="text-blue-600 dark:text-blue-400" /> Change in {shortMetric(selected.metric) || "metric"}</p>
                      {displayMagnitude(selected) ? (
                        <p className={`mt-1 text-[26px] font-bold ${selChange === null || selChange === undefined ? "text-foreground" : selChange >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>{displayMagnitude(selected)}</p>
                      ) : (
                        <p className="mt-1 text-[13px] text-muted-foreground">Change not quantified by the backend.</p>
                      )}
                      {selected.affectedPeriod && <p className="text-[11px] text-muted-foreground">vs. {selected.affectedPeriod}</p>}
                      {(selected.evidence?.baseline !== null && selected.evidence?.baseline !== undefined) ||
                      (selected.evidence?.currentValue !== null && selected.evidence?.currentValue !== undefined) ? (
                        <p className="mt-2 text-[11px] text-secondary-foreground">
                          {selected.evidence?.baseline !== null && selected.evidence?.baseline !== undefined && <>Baseline <span className="font-semibold text-foreground">{fmtMoney(selected.evidence.baseline)}</span></>}
                          {selected.evidence?.baseline !== null && selected.evidence?.baseline !== undefined && selected.evidence?.currentValue !== null && selected.evidence?.currentValue !== undefined && " → "}
                          {selected.evidence?.currentValue !== null && selected.evidence?.currentValue !== undefined && <>Current <span className="font-semibold text-foreground">{fmtMoney(selected.evidence.currentValue)}</span></>}
                        </p>
                      ) : null}
                    </div>
                    <div className="rounded-xl border border-border p-4">
                      <p className="text-[11px] font-semibold text-muted-foreground flex items-center gap-1.5"><Lightbulb size={12} className="text-indigo-500 dark:text-indigo-400" /> Why It Matters</p>
                      <p className="mt-1.5 text-[12px] text-secondary-foreground leading-relaxed">
                        {displayImpact(selected) || "The available evidence does not yet support a business interpretation — no causal claim is made."}
                      </p>
                      {atStake ? (
                        <p className="mt-1.5 text-[11px] text-secondary-foreground leading-relaxed">{atStake}</p>
                      ) : null}
                    </div>
                  </div>
                )}

                {tabId === "why" && (
                  <div className="rounded-xl border border-border p-4 max-w-3xl">
                    <p className="text-[12px] font-semibold">Why It Matters</p>
                    <p className="mt-1.5 text-[13px] text-secondary-foreground leading-relaxed">
                      {displayImpact(selected) || "The available evidence does not yet support a business interpretation — no causal claim is made."}
                    </p>
                    {atStake ? (
                      <p className="mt-1.5 text-[12px] text-secondary-foreground leading-relaxed">{atStake}</p>
                    ) : null}
                    {!displayImpact(selected) && (
                      <p className="mt-1 text-[11px] text-muted-foreground">Interpretation appears only when the underlying evidence supports it.</p>
                    )}
                    {impact?.value !== undefined && impact.value !== null && impact.direction !== "neutral" && impact.explanation && (
                      <p className="mt-2 text-xs text-muted-foreground">
                        Estimated impact: <span className={`font-semibold ${impact.direction === "positive" ? "text-emerald-600 dark:text-emerald-400" : impact.direction === "negative" ? "text-rose-600 dark:text-rose-400" : "text-foreground"}`}>{fmtMoney(Math.abs(impact.value))}</span>
                      </p>
                    )}
                  </div>
                )}

                {tabId === "evidence" && (
                  <div className="space-y-3 min-w-0 max-w-full">
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px]">
                      {selected.evidence?.currentValue !== null && selected.evidence?.currentValue !== undefined && (
                        <div className="rounded-lg border border-border border-l-4 border-l-blue-600 px-2.5 py-2 bg-card"><p className="text-muted-foreground">Current</p><p className="font-semibold break-words">{fmtMoney(selected.evidence.currentValue)}</p></div>
                      )}
                      {selected.evidence?.baseline !== null && selected.evidence?.baseline !== undefined && (
                        <div className="rounded-lg border border-border border-l-4 border-l-indigo-500 px-2.5 py-2 bg-card"><p className="text-muted-foreground">Baseline</p><p className="font-semibold break-words">{fmtMoney(selected.evidence.baseline)}</p></div>
                      )}
                      {selected.evidence?.sampleSize !== null && selected.evidence?.sampleSize !== undefined && (
                        <div className="rounded-lg border border-border border-l-4 border-l-blue-600 px-2.5 py-2 bg-card"><p className="text-muted-foreground">Sample</p><p className="font-semibold break-words">{selected.evidence.sampleSize} periods</p></div>
                      )}
                      <div className="rounded-lg border border-border border-l-4 border-l-indigo-500 px-2.5 py-2 bg-card"><p className="text-muted-foreground">Strength</p><p className="font-semibold break-words">{typeof selected.confidence === "number" ? `${selected.confidence}%` : "—"}</p></div>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
                      {[
                        { k: "Observed", v: selected.evidence?.observations?.[0] ? plainBusiness(selected.evidence.observations[0]) || undefined : undefined },
                        { k: "Comparison", v: selected.affectedPeriod ? plainBusiness(selected.affectedPeriod) || selected.affectedPeriod : (selected.evidence?.supportingMetrics?.[0] ? plainBusiness(selected.evidence.supportingMetrics[0]) || undefined : undefined) },
                        { k: "Scope", v: affectedSegment ? plainBusiness(humanize(affectedSegment)) || humanize(affectedSegment) : undefined },
                        { k: "Evidence Quality", v: evidenceQualityOf(selected) === "Not reported" ? undefined : evidenceQualityOf(selected) },
                      ].filter((b) => b.v).map((b) => (
                        <div key={b.k} className="rounded-xl border border-border border-l-4 border-l-blue-600 p-3 bg-card min-w-0">
                          <p className="text-[10px] font-bold uppercase tracking-wide text-muted-foreground">{b.k}</p>
                          <p className="mt-1 text-[11px] text-secondary-foreground leading-snug line-clamp-3 break-words">{b.v}</p>
                          <p className="mt-1 text-[10px] text-muted-foreground leading-snug">Why shown: {whyShownFor(b.k)}</p>
                        </div>
                      ))}
                    </div>
                    {!!selected.evidence?.supportingMetrics?.length && (
                      <ul className="space-y-1">
                        {selected.evidence.supportingMetrics.slice(0, 4).map((m, i) => {
                          const text = plainBusiness(m) || humanize(m);
                          if (!text) return null;
                          return (
                            <li key={i} className="text-xs"><span className="text-muted-foreground">• </span>{text} <span className="text-muted-foreground">— {whyShownFor("supporting")}</span></li>
                          );
                        })}
                      </ul>
                    )}
                    {lims.length > 0 && (
                      <div className="rounded-lg border border-border bg-secondary px-2.5 py-2" role="note" aria-label="Limitations">
                        <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-wide">Limitations</p>
                        <p className="mt-0.5 text-[11px] text-muted-foreground leading-relaxed">{lims[0]}</p>
                      </div>
                    )}
                    <EvidenceDataTable ins={selected} />
                  </div>
                )}

                {tabId === "factors" && (
                  <div className="rounded-xl border border-border p-4 max-w-3xl">
                    <p className="text-[12px] font-semibold">Associated Factors</p>
                    <p className="text-[11px] text-muted-foreground mt-0.5">Shown only because the analysis independently supports them.</p>
                    <ul className="mt-2 space-y-1.5">
                      {factors.map((f, i) => (
                        <li key={i} className="text-[12px] text-secondary-foreground"><span className="text-blue-500 dark:text-blue-400">• </span><span className="font-medium text-foreground">{plainBusiness(humanize(f)) || humanize(f)}</span></li>
                      ))}
                    </ul>
                  </div>
                )}

                {tabId === "reco" && (recoDetail || recoText) && (
                  <div className="rounded-xl border border-blue-100 dark:border-blue-500/20 bg-blue-50/50 dark:bg-blue-500/10 p-4 max-w-3xl">
                    <p className="text-[12px] font-semibold">What ARIA recommends</p>
                    {recoDetail ? (
                      <div className="mt-1">
                        {recoDetail.title && (
                          <p className="text-[13px] font-semibold text-foreground leading-relaxed">{recoDetail.title}</p>
                        )}
                        {recoDetail.steps.length > 0 && (
                          <ul className="mt-2 space-y-1.5">
                            {recoDetail.steps.map((s, i) => (
                              <li key={i} className="text-[12.5px] text-secondary-foreground leading-relaxed">
                                <span className="font-semibold text-blue-700 dark:text-blue-400">{i + 1}. </span>{s}
                              </li>
                            ))}
                          </ul>
                        )}
                        {recoDetail.outcome && (
                          <p className="mt-2 text-[12px] text-secondary-foreground leading-relaxed">
                            <span className="font-semibold text-foreground">Expected outcome: </span>{recoDetail.outcome}
                          </p>
                        )}
                      </div>
                    ) : (
                      <p className="mt-1 text-[13px] text-foreground leading-relaxed">{recoText}</p>
                    )}
                    <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2 border-t border-blue-100 dark:border-blue-500/20 pt-3">
                      <label className="block text-[11px]">
                        <span className="text-muted-foreground font-medium">Owner</span>
                        <input
                          value={ownerState.owner}
                          onChange={(e) => setOwnerById((p) => ({ ...p, [selected.id]: { owner: e.target.value.slice(0, 80), reviewBy: ownerState.reviewBy } }))}
                          placeholder="Name who will act"
                          className="mt-1 w-full rounded-lg border border-border bg-card px-2.5 py-1.5 text-[12px] text-foreground outline-none focus:border-blue-400"
                          aria-label="Recommendation owner"
                        />
                      </label>
                      <label className="block text-[11px]">
                        <span className="text-muted-foreground font-medium">Review by</span>
                        <input
                          value={ownerState.reviewBy}
                          onChange={(e) => setOwnerById((p) => ({ ...p, [selected.id]: { owner: ownerState.owner, reviewBy: e.target.value.slice(0, 80) } }))}
                          placeholder="Date to check back"
                          className="mt-1 w-full rounded-lg border border-border bg-card px-2.5 py-1.5 text-[12px] text-foreground outline-none focus:border-blue-400"
                          aria-label="Recommendation review-by date"
                        />
                      </label>
                    </div>
                  </div>
                )}

                {tabId === "whatif" && (
                  <div className="rounded-xl border border-border p-4 max-w-3xl">
                    <p className="text-[12px] font-semibold">What If</p>
                    {scenarioAvailable ? (
                      <p className="mt-1 text-[12px] text-secondary-foreground">{plainBusiness(selected.scenario?.label) || selected.scenario?.label || "Scenario modeling is available for this insight — ask ARIA to model a change."}</p>
                    ) : (
                      <p className="mt-1 text-[12px] text-muted-foreground">Scenario modeling isn&apos;t available for this insight yet.</p>
                    )}
                  </div>
                )}

                {tabId === "monitor" && (
                  <div className="rounded-xl border border-border p-4 max-w-3xl">
                    <p className="text-[12px] font-semibold">Watch this</p>
                    {monitorAvailable ? (
                      <div className="mt-2 grid grid-cols-1 sm:grid-cols-3 gap-2 text-[11px]">
                        {selected.monitoring?.metric && <div className="rounded-lg border border-border px-2.5 py-2"><p className="text-muted-foreground">Metric</p><p className="font-semibold">{plainBusiness(humanize(selected.monitoring.metric)) || humanize(selected.monitoring.metric)}</p></div>}
                        {rangePlain ? <div className="rounded-lg border border-border px-2.5 py-2"><p className="text-muted-foreground">Usual range</p><p className="font-semibold">{rangePlain}</p></div> : null}
                        {selected.monitoring?.nextReview && <div className="rounded-lg border border-border px-2.5 py-2"><p className="text-muted-foreground">Next review</p><p className="font-semibold">{selected.monitoring.nextReview}</p></div>}
                      </div>
                    ) : (
                      <p className="mt-1 text-[12px] text-muted-foreground">Monitoring isn&apos;t configured for this insight yet.</p>
                    )}
                  </div>
                )}

                {/* Actions */}
                <div className="rounded-xl border border-blue-100 dark:border-blue-500/20 bg-blue-50/40 dark:bg-blue-500/10 p-3.5">
                  <p className="text-[12px] font-semibold">Investigate this insight</p>
                  <p className="text-[11px] text-muted-foreground">Explore the drivers, forecast the impact, or hand it to a decision.</p>
                  <div className="mt-2.5 flex flex-wrap gap-1.5">
                    <button onClick={() => openHypothesis(selected)} className="inline-flex items-center gap-1 text-[11px] font-semibold px-3 py-1.5 rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition-colors">
                      <FlaskConical size={12} /> Investigate
                    </button>
                    <button onClick={() => askCopilot(selected, selected.recommendedNextAnalysis?.question || selected.description, drill && drill.insightId === selected.id ? drill.label : undefined)} className="inline-flex items-center gap-1 text-[11px] font-semibold px-3 py-1.5 rounded-lg border border-border bg-card hover:border-blue-400 transition-colors">
                      <MessageSquare size={12} /> Ask ARIA
                    </button>
                    <button onClick={() => toggleSave(selected)} className="inline-flex items-center gap-1 text-[11px] font-semibold px-3 py-1.5 rounded-lg border border-border bg-card hover:border-blue-400 transition-colors">
                      <Bookmark size={12} fill={savedIds.includes(selected.id) ? "currentColor" : "none"} /> {savedIds.includes(selected.id) ? "Saved" : "Save"}
                    </button>
                    <button onClick={() => openForecast(selected)} className="inline-flex items-center gap-1 text-[11px] font-semibold px-3 py-1.5 rounded-lg border border-border bg-card hover:border-blue-400 transition-colors">
                      <TrendingUp size={12} /> Forecast
                    </button>
                  </div>
                  {(canSeeDashboard || canForecast) ? (
                    <div className="mt-2.5 flex flex-wrap gap-1.5 border-t border-blue-100 dark:border-blue-500/20 pt-2.5">
                      {canSeeDashboard ? (
                        <button onClick={() => openDashboard(selected)} className="inline-flex items-center gap-1 text-[11px] font-semibold px-3 py-1.5 rounded-lg border border-border bg-card hover:border-blue-400 transition-colors">
                          See on dashboard
                        </button>
                      ) : null}
                      {canForecast ? (
                        <button onClick={() => openForecast(selected)} className="inline-flex items-center gap-1 text-[11px] font-semibold px-3 py-1.5 rounded-lg border border-border bg-card hover:border-blue-400 transition-colors">
                          Forecast this
                        </button>
                      ) : null}
                    </div>
                  ) : null}
                </div>

                {/* Prev / Next */}
                <div className="flex items-center justify-between pt-1">
                  <button
                    onClick={() => { const i = filtered.findIndex((x) => x.id === selected.id); if (i > 0) openInsight(filtered[i - 1].id); }}
                    disabled={filtered.findIndex((x) => x.id === selected.id) <= 0}
                    className="text-[11px] text-muted-foreground hover:text-foreground disabled:opacity-40 inline-flex items-center gap-1">
                    ← Previous finding
                  </button>
                  <button
                    onClick={() => askCopilot(selected, `What is driving ${shortMetric(selected.metric) || "this metric"}? Break down ${selected.magnitude || "the change"} by the top contributing dimensions${selected.affectedPeriod ? ` in ${selected.affectedPeriod}` : ""}.`, drill && drill.insightId === selected.id ? drill.label : undefined)}
                    className="text-[11px] text-blue-600 dark:text-blue-400 hover:underline inline-flex items-center gap-1">
                    See What&apos;s Driving This <ChevronRight size={12} />
                  </button>
                  <button
                    onClick={() => { const i = filtered.findIndex((x) => x.id === selected.id); if (i < filtered.length - 1) openInsight(filtered[i + 1].id); }}
                    disabled={filtered.findIndex((x) => x.id === selected.id) >= filtered.length - 1}
                    className="text-[11px] text-muted-foreground hover:text-foreground disabled:opacity-40 inline-flex items-center gap-1">
                    Next finding →
                  </button>
                </div>
              </div>
            </article>
          </div>
        </div>
      </div>
    );
  }

  /* ============================ LIST VIEW ============================ */
  return (
    <div className="min-h-full bg-card text-foreground overflow-x-hidden">
      <style>{`@keyframes insightCardIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}.insight-card-in{animation:insightCardIn .45s ease both}`}</style>
      <div className="p-4 md:p-6 max-w-[1400px] mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_300px] gap-5 items-start">
          {/* Main column */}
          <div className="min-w-0 space-y-5">
            {/* Hero banner */}
            <section className="relative overflow-hidden rounded-2xl border border-blue-100 dark:border-blue-500/20 bg-gradient-to-br from-blue-50 dark:from-blue-500/10 via-card to-indigo-100 dark:to-indigo-500/10 px-6 py-7 md:px-8 md:py-9">
              <div className="absolute -right-16 -top-24 w-72 h-72 rounded-full bg-indigo-200/40 dark:bg-indigo-500/15 blur-2xl pointer-events-none" aria-hidden />
              <div className="absolute right-10 bottom-0 w-56 h-32 rounded-t-full bg-blue-200/30 dark:bg-blue-500/15 blur-xl pointer-events-none" aria-hidden />
              <div className="relative min-w-0 max-w-full">
                <p className="text-[11px] font-bold tracking-[0.18em] text-blue-700 dark:text-blue-400">ARIA INSIGHTS</p>
                <h1 className="mt-2 text-[26px] md:text-[30px] font-semibold tracking-tight leading-[1.12] text-balance text-foreground">What did ARIA learn today?</h1>
                <p className="mt-2 text-[13px] md:text-[14px] text-secondary-foreground leading-relaxed max-w-2xl">
                  Here are the key patterns, trends and opportunities found in your data. Explore deeper, ask questions, or see what might happen next.
                </p>
                {heroPills}
                {toolbar}
              </div>
            </section>

            {/* Key Insights */}
            <section className="min-w-0">
              <div className="flex items-center gap-2">
                <Sparkles size={15} className="text-blue-600 dark:text-blue-400" />
                <h2 className="text-[15px] font-semibold tracking-tight">Key Insights</h2>
                <button onClick={() => { setActiveFilter("all"); setMetricFilter("All"); setCriticalOnly(false); }} className="ml-auto text-[11px] font-semibold text-blue-600 dark:text-blue-400 hover:underline">View all →</button>
              </div>
              {filtered.length === 0 ? (
                <div className="mt-3 text-center py-14 px-6 bg-card border border-dashed border-border rounded-2xl shadow-sm max-w-full">
                  <Filter size={20} className="mx-auto text-slate-300 dark:text-muted-foreground mb-3" />
                  <p className="text-sm font-semibold">No insights match these filters</p>
                  <p className="mt-1 text-xs text-muted-foreground">Try a different category, metric, or clear everything to see the full set.</p>
                  <button onClick={() => { setActiveFilter("all"); setMetricFilter("All"); setCriticalOnly(false); }} className="mt-3 inline-flex items-center gap-1 text-xs font-semibold px-3.5 py-1.5 rounded-full bg-blue-600 text-white hover:bg-blue-700 transition-colors">Clear filters</button>
                </div>
              ) : (
                <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4">
                  {filtered.map((ins, idx) => {
                    const isSaved = savedIds.includes(ins.id);
                    return (
                      <article
                        key={ins.id}
                        className="insight-card-in min-w-0 max-w-full rounded-2xl border border-border bg-card p-4 shadow-sm hover:shadow-lg hover:border-blue-200 dark:hover:border-blue-500/25 hover:-translate-y-0.5 transition-all duration-200 flex flex-col"
                        style={{ borderTop: `3px solid ${severityAccent(ins.severity)}`, animationDelay: `${Math.min(idx, 6) * 60}ms` }}
                      >
                        <div className="flex items-center gap-2">
                          <CatPill ins={ins} />
                          <button onClick={() => toggleSave(ins)} className={`ml-auto p-1.5 rounded-lg transition-colors ${isSaved ? "text-blue-600 dark:text-blue-400" : "text-slate-300 dark:text-muted-foreground hover:text-muted-foreground"} hover:bg-secondary`} title={isSaved ? "Unsave insight" : "Save insight"} aria-label="Save insight">
                            <Bookmark size={14} fill={isSaved ? "currentColor" : "none"} />
                          </button>
                        </div>
                        <h3 className="mt-2 text-[14px] font-semibold leading-snug tracking-tight break-words">{displayTitle(ins)}</h3>
                        <p className="mt-1 text-[12px] text-muted-foreground leading-relaxed line-clamp-3 break-words">{displayDescription(ins)}</p>
                        <div className="mt-2 min-w-0 max-w-full overflow-hidden rounded-xl border border-border bg-secondary/50 px-1">
                          <MiniInsightChart ins={ins} />
                        </div>
                        <div className="mt-auto pt-3 flex items-center gap-1.5 text-[11px] text-muted-foreground">
                          <Clock size={11} className="flex-shrink-0" />
                          <span className="truncate">{fmtWhen(ins.detectedAt)}</span>
                          <button onClick={() => openInsight(ins.id)} className="ml-auto flex-shrink-0 inline-flex items-center gap-1 font-semibold text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 hover:underline">
                            View details <ChevronRight size={12} />
                          </button>
                        </div>
                      </article>
                    );
                  })}
                </div>
              )}
            </section>

            {askBar}
          </div>

          {/* Right sidebar */}
          <aside className="min-w-0 space-y-4 lg:sticky lg:top-4">
            {dataSourceCard}
            {recentCard}
          </aside>
        </div>
      </div>
    </div>
  );
}
