/* AI Analytics — Dashboard home + inline analysis results.
   Search bar and popular actions call /api/ask (Goal Agent) and display
   chart + insights + table directly on the page. No copilot redirect. */

import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import {
  BarChart3, LineChart as LineChartIcon, TrendingUp, Search, Send,
  ChevronRight, Plus, Bookmark, Clock, FlaskConical, SlidersHorizontal,
  Target, FileText, Sparkles, ArrowUpRight, ArrowDownRight, Database,
  Copy, MoreHorizontal, Brain, Lightbulb, GitCompareArrows, Zap,
  Compass, Share2, X, Info, CheckCircle2, Shield,
} from "lucide-react";
import {
  BarChart, Bar, LineChart, Line, AreaChart, Area,
  PieChart, Pie, Cell, ScatterChart, Scatter, ZAxis,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine,
} from "recharts";
import { useTheme } from "../App";
import { colorForName } from "../lib/charts";
import { fmtPctBiz } from "../lib/format";
import {
  asList, strOf, numOf, humanize, titleCase, fmtMetric, getNoSchemaInfo,
} from "../lib/intelHelpers";
import {
  Card, CardHead, NoDatasetPanel, ErrorPanel, LoadingPanel, Field, StatPill,
  CountUp, Sparkline, selectCls, inputCls, btnPrimary, btnGhost,
} from "./intelligence/cards";
import { useExplore, RANGE_OPTIONS, type RangeKey } from "./intelligence/useExplore";
import { intelligenceApi, analyzeQuestion, normalizeApiError, apiFetch } from "../../services/api";
import EvidenceDrawer, { type EvidenceContext } from "./intelligence/EvidenceDrawer";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface AIAnalyticsProps {
  initialQuery?: string;
  onAsk?: (question: string) => void;
  analysisResult?: any;
  goalStatus?: string;
  goalResult?: any;
  goalError?: string;
  loading?: boolean;
  onNavigate?: (page: string) => void;
  onAskCopilot?: (question: string) => void;
  connectionStatus?: string;
  dbName?: string;
}

type HomeTab = "recent" | "saved" | "collections" | "shared";

/* ------------------------------------------------------------------ */
/*  Popular actions                                                    */
/* ------------------------------------------------------------------ */

const POPULAR_ACTIONS: { label: string; icon: any; query?: string; navigate?: string }[] = [
  { label: "Compare segments", icon: GitCompareArrows, query: "Compare segments breakdown" },
  { label: "Analyze a trend", icon: TrendingUp, query: "Analyze trend over time" },
];

/* ------------------------------------------------------------------ */
/*  Sidebar actions                                                    */
/* ------------------------------------------------------------------ */

const SIDEBAR_ACTIONS = [
  { label: "Explain this analysis", sub: "Get a plain language summary", icon: Brain, query: "Explain this analysis in plain language" },
  { label: "Investigate possible factors", sub: "Find what might be behind this", icon: Compass, query: "What factors might be driving these results?" },
  { label: "Compare with previous period", sub: "See how it changed over time", icon: GitCompareArrows, query: "Compare this with the previous period" },
  { label: "Test a scenario", sub: "Try different values or conditions", icon: Zap, query: "What-if scenario analysis" },
];

const CONTINUE_EXPLORING = [
  { label: "Drill down", sub: "View detailed breakdowns", icon: BarChart3, tab: "explore" },
  { label: "Apply a filter", sub: "Narrow down your data", icon: SlidersHorizontal, tab: "explore" },
  { label: "Create a goal", sub: "Set a target and track progress", icon: Target, tab: "goals" },
  { label: "Save this analysis", sub: "Keep it for later", icon: Bookmark, tab: "explore" },
];

/* ------------------------------------------------------------------ */
/*  Chart palette                                                      */
/* ------------------------------------------------------------------ */

const CHART_PALETTE = [
  "#2F6FAD", "#5B9BD5", "#70AD47", "#FFC000", "#ED7D31",
  "#A855F7", "#06B6D4", "#F43F5E", "#10B981", "#8B5CF6",
  "#EC4899", "#14B8A6", "#F97316", "#6366F1", "#84CC16",
];

function palette(n: number): string[] {
  const out: string[] = [];
  for (let i = 0; i < n; i++) out.push(CHART_PALETTE[i % CHART_PALETTE.length]);
  return out;
}

/* ------------------------------------------------------------------ */
/*  Dynamic chart — renders based on chart_spec from backend           */
/* ------------------------------------------------------------------ */

function DynamicChart({ data, chartSpec, metric }: { data: any[]; chartSpec: any; metric: string }) {
  const { isDark } = useTheme();
  const spec: any = chartSpec || {};
  const chartType: string = spec.chart_type || "bar";
  const xField: string = spec.x || "";
  const yField: string = spec.y || "";

  const specYCols: string[] = useMemo(() => {
    if (spec.y_columns && spec.y_columns.length) return spec.y_columns;
    if (Array.isArray(spec.y)) return spec.y;
    return [yField];
  }, [spec, yField]);

  const points = useMemo(() => {
    if (!data || !data.length) return [];
    return data.slice(0, 60).map((r: any) => ({
      label: strOf(r?.label ?? r?.[xField] ?? ""),
      value: numOf(r?.value ?? r?.[yField]) ?? 0,
    }));
  }, [data, xField, yField]);

  if (!data || !data.length) return null;

  const colors = palette(Math.max(data.length, 10));
  /* Stable per-series color so single-metric charts differ by metric instead
     of all rendering the same blue. */
  const lineColor = colorForName(strOf(metric || yField || "value") || "value");
  const fmtVal = (v: number) => fmtMetric(v, metric);
  const tickStyle = { fontSize: 11, fill: "var(--muted-foreground)" };
  const gridStyle = { strokeDasharray: "3 3", stroke: isDark ? "#2A2A35" : "#E5E7EB" };
  const tooltipStyle = { fontSize: 12, borderRadius: 8, background: "var(--card)", border: "1px solid var(--border)", color: "var(--foreground)" };

  /* ---- Bar family (per-category colors) ---- */
  if (chartType === "bar" || chartType === "column") {
    const barData = data.slice(0, 60).map((r: any) => ({ label: strOf(r?.[xField] ?? r?.label ?? ""), value: numOf(r?.[yField] ?? r?.value) ?? 0 }));
    return (
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={barData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid {...gridStyle} vertical={false} />
          <XAxis dataKey="label" tick={tickStyle} axisLine={false} tickLine={false} minTickGap={24} />
          <YAxis tick={tickStyle} axisLine={false} tickLine={false} width={52} tickFormatter={fmtVal} />
          <Tooltip formatter={(v: number) => [fmtVal(v), humanize(metric)]} contentStyle={tooltipStyle} />
          <Bar dataKey="value" name="Current" radius={[4, 4, 0, 0]} isAnimationActive animationDuration={500}>
            {barData.map((d: any, i: number) => <Cell key={i} fill={colorForName(d.label || String(i))} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (chartType === "bar_horizontal" || chartType === "horizontal_bar") {
    const hbarData = data.slice(0, 60).map((r: any) => ({ label: strOf(r?.[xField] ?? r?.label ?? ""), value: numOf(r?.[yField] ?? r?.value) ?? 0 }));
    return (
      <ResponsiveContainer width="100%" height="100%">
        <BarChart layout="vertical" data={hbarData} margin={{ top: 8, right: 8, bottom: 0, left: 80 }}>
          <CartesianGrid {...gridStyle} horizontal={false} />
          <XAxis type="number" tick={tickStyle} axisLine={false} tickLine={false} tickFormatter={fmtVal} />
          <YAxis type="category" dataKey="label" tick={tickStyle} axisLine={false} tickLine={false} width={76} />
          <Tooltip formatter={(v: number) => [fmtVal(v), humanize(metric)]} contentStyle={tooltipStyle} />
          <Bar dataKey="value" name="Current" radius={[0, 4, 4, 0]} isAnimationActive animationDuration={500}>
            {hbarData.map((d: any, i: number) => <Cell key={i} fill={colorForName(d.label || String(i))} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (chartType === "bar_stacked" || chartType === "stacked_bar") {
    const stackedData = data.slice(0, 60).map((r: any) => {
      const row: any = { label: strOf(r?.[xField] ?? r?.label ?? "") };
      specYCols.forEach((yc: string) => { row[yc] = numOf(r?.[yc]) ?? 0; });
      return row;
    });
    return (
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={stackedData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid {...gridStyle} vertical={false} />
          <XAxis dataKey="label" tick={tickStyle} axisLine={false} tickLine={false} minTickGap={24} />
          <YAxis tick={tickStyle} axisLine={false} tickLine={false} width={52} tickFormatter={fmtVal} />
          <Tooltip formatter={(v: number) => [fmtVal(v)]} contentStyle={tooltipStyle} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {specYCols.map((yc: string, i: number) => (
            <Bar key={yc} dataKey={yc} stackId="a" name={humanize(yc)} fill={colors[i]} isAnimationActive animationDuration={500} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (chartType === "bar_grouped" || chartType === "grouped_bar") {
    const groupedData = data.slice(0, 60).map((r: any) => {
      const row: any = { label: strOf(r?.[xField] ?? r?.label ?? "") };
      specYCols.forEach((yc: string) => { row[yc] = numOf(r?.[yc]) ?? 0; });
      return row;
    });
    return (
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={groupedData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid {...gridStyle} vertical={false} />
          <XAxis dataKey="label" tick={tickStyle} axisLine={false} tickLine={false} minTickGap={24} />
          <YAxis tick={tickStyle} axisLine={false} tickLine={false} width={52} tickFormatter={fmtVal} />
          <Tooltip formatter={(v: number) => [fmtVal(v)]} contentStyle={tooltipStyle} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {specYCols.map((yc: string, i: number) => (
            <Bar key={yc} dataKey={yc} name={humanize(yc)} fill={colors[i]} radius={[4, 4, 0, 0]} isAnimationActive animationDuration={500} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (chartType === "bar_normalized" || chartType === "normalized_bar") {
    const normData = data.slice(0, 60).map((r: any) => {
      const row: any = { label: strOf(r?.[xField] ?? r?.label ?? "") };
      let total = 0;
      specYCols.forEach((yc: string) => { total += numOf(r?.[yc]) ?? 0; });
      specYCols.forEach((yc: string) => { row[yc] = total > 0 ? ((numOf(r?.[yc]) ?? 0) / total) * 100 : 0; });
      return row;
    });
    return (
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={normData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid {...gridStyle} vertical={false} />
          <XAxis dataKey="label" tick={tickStyle} axisLine={false} tickLine={false} minTickGap={24} />
          <YAxis tick={tickStyle} axisLine={false} tickLine={false} width={52} tickFormatter={(v: number) => `${Math.round(v)}%`} />
          <Tooltip formatter={(v: number) => [`${v.toFixed(1)}%`]} contentStyle={tooltipStyle} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {specYCols.map((yc: string, i: number) => (
            <Bar key={yc} dataKey={yc} stackId="a" name={humanize(yc)} fill={colors[i]} isAnimationActive animationDuration={500} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (chartType === "bar_diverging" || chartType === "diverging_bar") {
    return (
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data.slice(0, 60).map((r: any) => ({ label: strOf(r?.[xField] ?? r?.label ?? ""), value: numOf(r?.[yField] ?? r?.value) ?? 0 }))} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid {...gridStyle} vertical={false} />
          <XAxis dataKey="label" tick={tickStyle} axisLine={false} tickLine={false} minTickGap={24} />
          <YAxis tick={tickStyle} axisLine={false} tickLine={false} width={52} tickFormatter={fmtVal} />
          <Tooltip formatter={(v: number) => [fmtVal(v), humanize(metric)]} contentStyle={tooltipStyle} />
          <ReferenceLine y={0} stroke="var(--border)" />
          <Bar dataKey="value" name="Current" radius={[4, 4, 0, 0]} isAnimationActive animationDuration={500}>
            {data.slice(0, 60).map((r: any, i: number) => {
              const val = numOf(r?.[yField] ?? r?.value) ?? 0;
              return <Cell key={i} fill={val >= 0 ? "#10B981" : "#F43F5E"} />;
            })}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (chartType === "lollipop") {
    const lolliData = data.slice(0, 60).map((r: any) => ({ label: strOf(r?.[xField] ?? r?.label ?? ""), value: numOf(r?.[yField] ?? r?.value) ?? 0 }));
    return (
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={lolliData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid {...gridStyle} vertical={false} />
          <XAxis dataKey="label" tick={tickStyle} axisLine={false} tickLine={false} minTickGap={24} />
          <YAxis tick={tickStyle} axisLine={false} tickLine={false} width={52} tickFormatter={fmtVal} />
          <Tooltip formatter={(v: number) => [fmtVal(v), humanize(metric)]} contentStyle={tooltipStyle} />
          <Bar dataKey="value" name="Current" barSize={8} radius={[4, 4, 0, 0]} isAnimationActive animationDuration={500}>
            {lolliData.map((d: any, i: number) => <Cell key={i} fill={colorForName(d.label || String(i))} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (chartType === "pareto") {
    const sorted = data.slice(0, 60).map((r: any) => ({ label: strOf(r?.[xField] ?? r?.label ?? ""), value: numOf(r?.[yField] ?? r?.value) ?? 0 })).sort((a: any, b: any) => b.value - a.value);
    const total = sorted.reduce((s: number, p: any) => s + p.value, 0);
    let cum = 0;
    const paretoData = sorted.map((p: any) => { cum += p.value; return { ...p, cumulative: total > 0 ? (cum / total) * 100 : 0 }; });
    return (
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={paretoData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid {...gridStyle} vertical={false} />
          <XAxis dataKey="label" tick={tickStyle} axisLine={false} tickLine={false} minTickGap={24} />
          <YAxis tick={tickStyle} axisLine={false} tickLine={false} width={52} tickFormatter={fmtVal} />
          <Tooltip contentStyle={tooltipStyle} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="value" name="Value" radius={[4, 4, 0, 0]} isAnimationActive animationDuration={500}>
            {paretoData.map((p: any, i: number) => <Cell key={i} fill={colorForName(p.label || String(i))} />)}
          </Bar>
          <Line type="monotone" dataKey="cumulative" name="Cumulative %" stroke="#ED7D31" strokeWidth={2} dot={false} yAxisId={0} />
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (chartType === "waterfall") {
    const wfData = data.slice(0, 60).map((r: any, i: number) => {
      const val = numOf(r?.[yField] ?? r?.value) ?? 0;
      return { label: strOf(r?.[xField] ?? r?.label ?? ""), value: Math.abs(val), fill: val >= 0 ? "#10B981" : "#F43F5E", base: i === 0 ? 0 : undefined };
    });
    return (
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={wfData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid {...gridStyle} vertical={false} />
          <XAxis dataKey="label" tick={tickStyle} axisLine={false} tickLine={false} minTickGap={24} />
          <YAxis tick={tickStyle} axisLine={false} tickLine={false} width={52} tickFormatter={fmtVal} />
          <Tooltip formatter={(v: number) => [fmtVal(v), humanize(metric)]} contentStyle={tooltipStyle} />
          <Bar dataKey="value" name="Value" radius={[4, 4, 0, 0]} isAnimationActive animationDuration={500}>
            {wfData.map((entry: any, i: number) => <Cell key={i} fill={entry.fill} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (chartType === "histogram") {
    const histData = data.slice(0, 60).map((r: any) => ({ label: strOf(r?.[xField] ?? r?.label ?? ""), value: numOf(r?.[yField] ?? r?.value) ?? 0 }));
    return (
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={histData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid {...gridStyle} vertical={false} />
          <XAxis dataKey="label" tick={tickStyle} axisLine={false} tickLine={false} minTickGap={24} />
          <YAxis tick={tickStyle} axisLine={false} tickLine={false} width={52} tickFormatter={fmtVal} />
          <Tooltip formatter={(v: number) => [fmtVal(v), humanize(metric)]} contentStyle={tooltipStyle} />
          <Bar dataKey="value" name="Frequency" radius={[2, 2, 0, 0]} isAnimationActive animationDuration={500}>
            {histData.map((d: any, i: number) => <Cell key={i} fill={colorForName(d.label || String(i))} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (chartType === "box_plot") {
    const boxData = data.slice(0, 60).map((r: any) => ({ label: strOf(r?.[xField] ?? r?.label ?? ""), value: numOf(r?.[yField] ?? r?.value ?? r?.median ?? 0) }));
    return (
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={boxData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid {...gridStyle} vertical={false} />
          <XAxis dataKey="label" tick={tickStyle} axisLine={false} tickLine={false} minTickGap={24} />
          <YAxis tick={tickStyle} axisLine={false} tickLine={false} width={52} tickFormatter={fmtVal} />
          <Tooltip formatter={(v: number) => [fmtVal(v), humanize(metric)]} contentStyle={tooltipStyle} />
          <Bar dataKey="value" name="Median" radius={[4, 4, 0, 0]} isAnimationActive animationDuration={500}>
            {boxData.map((d: any, i: number) => <Cell key={i} fill={colorForName(d.label || String(i))} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (chartType === "bullet") {
    return (
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data.slice(0, 60).map((r: any) => ({ label: strOf(r?.[xField] ?? r?.label ?? ""), value: numOf(r?.[yField] ?? r?.value) ?? 0, target: numOf(r?.target) ?? 0 }))} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid {...gridStyle} vertical={false} />
          <XAxis dataKey="label" tick={tickStyle} axisLine={false} tickLine={false} minTickGap={24} />
          <YAxis tick={tickStyle} axisLine={false} tickLine={false} width={52} tickFormatter={fmtVal} />
          <Tooltip contentStyle={tooltipStyle} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="value" name="Actual" fill={CHART_PALETTE[0]} radius={[4, 4, 0, 0]} isAnimationActive animationDuration={500} />
          <Bar dataKey="target" name="Target" fill="#CBD5E1" radius={[4, 4, 0, 0]} isAnimationActive animationDuration={500} />
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (chartType === "combo") {
    const comboData = data.slice(0, 60).map((r: any) => {
      const row: any = { label: strOf(r?.[xField] ?? r?.label ?? "") };
      specYCols.forEach((yc: string) => { row[yc] = numOf(r?.[yc]) ?? 0; });
      return row;
    });
    return (
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={comboData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid {...gridStyle} vertical={false} />
          <XAxis dataKey="label" tick={tickStyle} axisLine={false} tickLine={false} minTickGap={24} />
          <YAxis tick={tickStyle} axisLine={false} tickLine={false} width={52} tickFormatter={fmtVal} />
          <Tooltip contentStyle={tooltipStyle} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {specYCols.map((yc: string, i: number) => {
            if (i === 0) return <Bar key={yc} dataKey={yc} name={humanize(yc)} fill={colors[i]} radius={[4, 4, 0, 0]} isAnimationActive animationDuration={500} />;
            return <Line key={yc} type="monotone" dataKey={yc} name={humanize(yc)} stroke={colors[i]} strokeWidth={2} dot={false} />;
          })}
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (chartType === "funnel") {
    const funnelData = data.slice(0, 20).map((r: any) => ({ label: strOf(r?.[xField] ?? r?.label ?? ""), value: numOf(r?.[yField] ?? r?.value) ?? 0 }));
    return (
      <div className="flex flex-col gap-1.5 py-2">
        {funnelData.map((p: any, i: number) => {
          const maxVal = Math.max(...funnelData.map((f: any) => f.value), 1);
          const pct = (p.value / maxVal) * 100;
          return (
            <div key={i} className="flex items-center gap-2">
              <span className="w-[120px] text-right text-[11px] text-secondary-foreground truncate flex-shrink-0">{p.label}</span>
              <div className="flex-1 bg-secondary rounded-md h-6 relative overflow-hidden">
                <div className="absolute inset-y-0 left-0 rounded-md transition-all duration-500" style={{ width: `${pct}%`, backgroundColor: colors[i % colors.length] }} />
                <span className="absolute inset-0 flex items-center justify-center text-[10.5px] font-medium text-foreground z-10">{fmtVal(p.value)}</span>
              </div>
            </div>
          );
        })}
      </div>
    );
  }

  if (chartType === "treemap") {
    const tmData = data.slice(0, 40).map((r: any, i: number) => ({
      name: strOf(r?.[xField] ?? r?.label ?? `Item ${i + 1}`),
      value: numOf(r?.[yField] ?? r?.value) ?? 0,
      fill: colors[i % colors.length],
    }));
    return (
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={tmData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid {...gridStyle} vertical={false} />
          <XAxis dataKey="name" tick={tickStyle} axisLine={false} tickLine={false} minTickGap={24} />
          <YAxis tick={tickStyle} axisLine={false} tickLine={false} width={52} tickFormatter={fmtVal} />
          <Tooltip formatter={(v: number) => [fmtVal(v)]} contentStyle={tooltipStyle} />
          <Bar dataKey="value" name="Value" isAnimationActive animationDuration={500}>
            {tmData.map((entry: any, i: number) => <Cell key={i} fill={entry.fill} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    );
  }

  /* ---- Line family ---- */
  if (chartType === "line" || chartType === "multi_line") {
    if (chartType === "multi_line" && specYCols.length > 1) {
      const mlData = data.slice(0, 60).map((r: any) => {
        const row: any = { label: strOf(r?.[xField] ?? r?.label ?? "") };
        specYCols.forEach((yc: string) => { row[yc] = numOf(r?.[yc]) ?? 0; });
        return row;
      });
      return (
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={mlData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid {...gridStyle} vertical={false} />
            <XAxis dataKey="label" tick={tickStyle} axisLine={false} tickLine={false} minTickGap={24} />
            <YAxis tick={tickStyle} axisLine={false} tickLine={false} width={52} tickFormatter={fmtVal} />
            <Tooltip formatter={(v: number) => [fmtVal(v)]} contentStyle={tooltipStyle} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {specYCols.map((yc: string, i: number) => (
              <Line key={yc} type="monotone" dataKey={yc} name={humanize(yc)} stroke={colors[i]} strokeWidth={2.5} dot={false} isAnimationActive animationDuration={500} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      );
    }
    return (
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={points} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid {...gridStyle} vertical={false} />
          <XAxis dataKey="label" tick={tickStyle} axisLine={false} tickLine={false} minTickGap={24} />
          <YAxis tick={tickStyle} axisLine={false} tickLine={false} width={52} tickFormatter={fmtVal} />
          <Tooltip formatter={(v: number) => [fmtVal(v), humanize(metric)]} contentStyle={tooltipStyle} />
          <Line type="monotone" dataKey="value" name="Current" stroke={lineColor} strokeWidth={2.5} dot={false} isAnimationActive animationDuration={500} />
        </LineChart>
      </ResponsiveContainer>
    );
  }

  if (chartType === "step_line" || chartType === "step") {
    return (
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={points} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid {...gridStyle} vertical={false} />
          <XAxis dataKey="label" tick={tickStyle} axisLine={false} tickLine={false} minTickGap={24} />
          <YAxis tick={tickStyle} axisLine={false} tickLine={false} width={52} tickFormatter={fmtVal} />
          <Tooltip formatter={(v: number) => [fmtVal(v), humanize(metric)]} contentStyle={tooltipStyle} />
          <Line type="stepAfter" dataKey="value" name="Current" stroke={lineColor} strokeWidth={2.5} dot={false} isAnimationActive animationDuration={500} />
        </LineChart>
      </ResponsiveContainer>
    );
  }

  if (chartType === "slope") {
    return (
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={points} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid {...gridStyle} vertical={false} />
          <XAxis dataKey="label" tick={tickStyle} axisLine={false} tickLine={false} minTickGap={24} />
          <YAxis tick={tickStyle} axisLine={false} tickLine={false} width={52} tickFormatter={fmtVal} />
          <Tooltip formatter={(v: number) => [fmtVal(v), humanize(metric)]} contentStyle={tooltipStyle} />
          <Line type="linear" dataKey="value" name="Current" stroke={lineColor} strokeWidth={2.5} dot={{ r: 4, fill: lineColor }} isAnimationActive animationDuration={500} />
        </LineChart>
      </ResponsiveContainer>
    );
  }

  if (chartType === "control_chart") {
    const mean = points.length > 0 ? points.reduce((s: number, p: any) => s + p.value, 0) / points.length : 0;
    return (
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={points} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid {...gridStyle} vertical={false} />
          <XAxis dataKey="label" tick={tickStyle} axisLine={false} tickLine={false} minTickGap={24} />
          <YAxis tick={tickStyle} axisLine={false} tickLine={false} width={52} tickFormatter={fmtVal} />
          <Tooltip formatter={(v: number) => [fmtVal(v), humanize(metric)]} contentStyle={tooltipStyle} />
          <ReferenceLine y={mean} stroke="#ED7D31" strokeDasharray="5 4" label={{ value: "Mean", position: "right", fontSize: 10, fill: "#ED7D31" }} />
          <Line type="monotone" dataKey="value" name="Current" stroke={lineColor} strokeWidth={2.5} dot={{ r: 3, fill: lineColor }} isAnimationActive animationDuration={500} />
        </LineChart>
      </ResponsiveContainer>
    );
  }

  if (chartType === "sparkline") {
    return (
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={points} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
          <Line type="monotone" dataKey="value" stroke={lineColor} strokeWidth={1.5} dot={false} isAnimationActive animationDuration={500} />
        </LineChart>
      </ResponsiveContainer>
    );
  }

  /* ---- Area family ---- */
  if (chartType === "area" || chartType === "stacked_area") {
    if (chartType === "stacked_area" && specYCols.length > 1) {
      const saData = data.slice(0, 60).map((r: any) => {
        const row: any = { label: strOf(r?.[xField] ?? r?.label ?? "") };
        specYCols.forEach((yc: string) => { row[yc] = numOf(r?.[yc]) ?? 0; });
        return row;
      });
      return (
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={saData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <defs>
              {specYCols.map((_: string, i: number) => (
                <linearGradient key={i} id={`areaGrad${i}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={colors[i]} stopOpacity={0.3} />
                  <stop offset="100%" stopColor={colors[i]} stopOpacity={0.05} />
                </linearGradient>
              ))}
            </defs>
            <CartesianGrid {...gridStyle} vertical={false} />
            <XAxis dataKey="label" tick={tickStyle} axisLine={false} tickLine={false} minTickGap={24} />
            <YAxis tick={tickStyle} axisLine={false} tickLine={false} width={52} tickFormatter={fmtVal} />
            <Tooltip formatter={(v: number) => [fmtVal(v)]} contentStyle={tooltipStyle} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {specYCols.map((yc: string, i: number) => (
              <Area key={yc} type="monotone" dataKey={yc} stackId="a" name={humanize(yc)} stroke={colors[i]} fill={`url(#areaGrad${i})`} strokeWidth={2} isAnimationActive animationDuration={500} />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      );
    }
    return (
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={points} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="areaGradSingle" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={CHART_PALETTE[0]} stopOpacity={0.3} />
              <stop offset="100%" stopColor={CHART_PALETTE[0]} stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid {...gridStyle} vertical={false} />
          <XAxis dataKey="label" tick={tickStyle} axisLine={false} tickLine={false} minTickGap={24} />
          <YAxis tick={tickStyle} axisLine={false} tickLine={false} width={52} tickFormatter={fmtVal} />
          <Tooltip formatter={(v: number) => [fmtVal(v), humanize(metric)]} contentStyle={tooltipStyle} />
          <Area type="monotone" dataKey="value" name="Current" stroke={CHART_PALETTE[0]} fill="url(#areaGradSingle)" strokeWidth={2.5} isAnimationActive animationDuration={500} />
        </AreaChart>
      </ResponsiveContainer>
    );
  }

  /* ---- Scatter family ---- */
  if (chartType === "scatter" || chartType === "bubble") {
    const scatterData = data.slice(0, 200).map((r: any) => ({
      x: numOf(r?.[xField] ?? r?.x ?? r?.value) ?? 0,
      y: numOf(r?.[yField] ?? r?.y) ?? 0,
      z: numOf(r?.z ?? r?.size ?? r?.value) ?? 10,
      name: strOf(r?.label ?? r?.name ?? ""),
    }));
    const isBubble = chartType === "bubble";
    return (
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid {...gridStyle} />
          <XAxis type="number" dataKey="x" name={humanize(xField)} tick={tickStyle} axisLine={false} tickLine={false} tickFormatter={fmtVal} />
          <YAxis type="number" dataKey="y" name={humanize(yField)} tick={tickStyle} axisLine={false} tickLine={false} tickFormatter={fmtVal} />
          {isBubble && <ZAxis type="number" dataKey="z" range={[40, 400]} name="Size" />}
          <Tooltip cursor={{ strokeDasharray: "3 3" }} formatter={(v: any) => [typeof v === "number" ? fmtVal(v) : v]} contentStyle={tooltipStyle} />
          <Scatter name="Data" fill={CHART_PALETTE[0]} isAnimationActive animationDuration={500}>
            {scatterData.map((_: any, i: number) => <Cell key={i} fill={colors[i % colors.length]} />)}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    );
  }

  /* ---- Pie family ---- */
  if (chartType === "pie" || chartType === "doughnut" || chartType === "polar_area") {
    const pieData = data.slice(0, 20).map((r: any, i: number) => ({
      name: strOf(r?.[xField] ?? r?.label ?? r?.name ?? `Item ${i + 1}`),
      value: numOf(r?.[yField] ?? r?.value) ?? 0,
    }));
    const inner = chartType === "doughnut" ? 60 : 0;
    return (
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie data={pieData} cx="50%" cy="50%" outerRadius={100} innerRadius={inner} dataKey="value" nameKey="name" label={({ name, percent }: any) => `${name} ${(percent * 100).toFixed(0)}%`} labelLine={{ strokeWidth: 1 }} isAnimationActive animationDuration={500}>
            {pieData.map((_: any, i: number) => <Cell key={i} fill={colors[i % colors.length]} />)}
          </Pie>
          <Tooltip formatter={(v: number) => [fmtVal(v)]} contentStyle={tooltipStyle} />
        </PieChart>
      </ResponsiveContainer>
    );
  }

  /* ---- Radar ---- */
  if (chartType === "radar") {
    const radarData = data.slice(0, 20).map((r: any) => {
      const row: any = { subject: strOf(r?.[xField] ?? r?.label ?? "") };
      specYCols.forEach((yc: string) => { row[yc] = numOf(r?.[yc]) ?? 0; });
      if (!specYCols.length) row.value = numOf(r?.[yField] ?? r?.value) ?? 0;
      return row;
    });
    const radarKeys = specYCols.length ? specYCols : ["value"];
    return (
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
          <PolarGrid stroke="var(--border)" />
          <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} />
          <PolarRadiusAxis tick={{ fontSize: 10, fill: "var(--muted-foreground)" }} />
          <Tooltip contentStyle={tooltipStyle} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {radarKeys.map((rk: string, i: number) => (
            <Radar key={rk} name={humanize(rk)} dataKey={rk} stroke={colors[i]} fill={colors[i]} fillOpacity={0.2} strokeWidth={2} isAnimationActive animationDuration={500} />
          ))}
        </RadarChart>
      </ResponsiveContainer>
    );
  }

  /* ---- Heatmap / Calendar Heatmap (CSS grid) ---- */
  if (chartType === "heatmap" || chartType === "calendar_heatmap") {
    const hmData = data.slice(0, 100).map((r: any) => ({
      x: strOf(r?.[xField] ?? r?.x ?? r?.label ?? ""),
      y: strOf(r?.[yField] ?? r?.y ?? r?.category ?? ""),
      value: numOf(r?.value ?? r?.count) ?? 0,
    }));
    const allVals = hmData.map((d: any) => d.value);
    const minV = Math.min(...allVals, 0);
    const maxV = Math.max(...allVals, 1);
    const range = maxV - minV || 1;
    const getColor = (v: number) => {
      const t = (v - minV) / range;
      const r = Math.round(47 + (16 - 47) * t);
      const g = Math.round(111 + (185 - 111) * t);
      const b = Math.round(173 + (164 - 173) * t);
      return `rgb(${r},${g},${b})`;
    };
    const xLabels = [...new Set(hmData.map((d: any) => d.x))];
    const yLabels = [...new Set(hmData.map((d: any) => d.y))];
    return (
      <div className="overflow-auto">
        <div className="inline-grid gap-px" style={{ gridTemplateColumns: `60px repeat(${xLabels.length}, minmax(28px, 1fr))` }}>
          <div />
          {xLabels.map((xl: string) => <div key={xl} className="text-[9px] text-muted-foreground text-center truncate px-0.5">{xl}</div>)}
          {yLabels.map((yl: string) => {
            const rowKey = `row-${yl}`;
            return [
              <div key={`y-${rowKey}`} className="text-[9px] text-muted-foreground text-right pr-1.5 truncate flex items-center justify-end">{yl}</div>,
              ...xLabels.map((xl: string) => {
                const cell = hmData.find((d: any) => d.x === xl && d.y === yl);
                const v = cell?.value ?? 0;
                return <div key={`${rowKey}-${xl}`} className="w-full aspect-square rounded-sm flex items-center justify-center text-[8px] text-white font-medium" style={{ backgroundColor: getColor(v) }} title={`${yl}: ${xl} = ${fmtVal(v)}`} />;
              }),
            ];
          })}
        </div>
      </div>
    );
  }

  /* ---- KPI Card ---- */
  if (chartType === "kpi_card" || chartType === "kpi") {
    const kpiVal = numOf(data[0]?.[yField] ?? data[0]?.value) ?? 0;
    return (
      <div className="flex flex-col items-center justify-center py-6">
        <span className="text-[36px] font-bold text-foreground tabular-nums">{fmtVal(kpiVal)}</span>
        <span className="text-[12px] text-muted-foreground mt-1">{spec.label || humanize(metric)}</span>
      </div>
    );
  }

  /* ---- Gauge ---- */
  if (chartType === "gauge") {
    const gaugeVal = numOf(data[0]?.[yField] ?? data[0]?.value) ?? 0;
    const gaugeMax = numOf(data[0]?.max) ?? 100;
    const pct = gaugeMax > 0 ? (gaugeVal / gaugeMax) * 100 : 0;
    return (
      <div className="flex flex-col items-center justify-center py-6">
        <div className="relative w-32 h-32">
          <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
            <circle cx="60" cy="60" r="50" fill="none" stroke="var(--border)" strokeWidth="10" />
            <circle cx="60" cy="60" r="50" fill="none" stroke={CHART_PALETTE[0]} strokeWidth="10" strokeDasharray={`${Math.PI * 100}`} strokeDashoffset={`${Math.PI * 100 * (1 - pct / 100)}`} strokeLinecap="round" className="transition-all duration-1000" />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-[20px] font-bold text-foreground tabular-nums">{fmtVal(gaugeVal)}</span>
            <span className="text-[10px] text-muted-foreground">/ {fmtVal(gaugeMax)}</span>
          </div>
        </div>
        <span className="text-[12px] text-muted-foreground mt-2">{spec.label || humanize(metric)}</span>
      </div>
    );
  }

  /* ---- Ecdf ---- */
  if (chartType === "ecdf") {
    const sorted = [...data].sort((a: any, b: any) => (numOf(a?.[yField] ?? a?.value) ?? 0) - (numOf(b?.[yField] ?? b?.value) ?? 0));
    let cumN = 0;
    const ecdfData = sorted.map((r: any) => {
      cumN++;
      return { value: numOf(r?.[yField] ?? r?.value) ?? 0, ecdf: (cumN / sorted.length) * 100 };
    });
    return (
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={ecdfData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid {...gridStyle} vertical={false} />
          <XAxis dataKey="value" tick={tickStyle} axisLine={false} tickLine={false} tickFormatter={fmtVal} name={humanize(metric)} />
          <YAxis tick={tickStyle} axisLine={false} tickLine={false} width={52} tickFormatter={(v: number) => `${Math.round(v)}%`} name="ECDF" />
          <Tooltip formatter={(v: any, name: string) => [name === "ECDF" ? `${(v as number).toFixed(1)}%` : fmtVal(v as number), name]} contentStyle={tooltipStyle} />
          <Line type="stepAfter" dataKey="ecdf" name="ECDF" stroke={CHART_PALETTE[0]} strokeWidth={2.5} dot={false} isAnimationActive animationDuration={500} />
        </LineChart>
      </ResponsiveContainer>
    );
  }

  /* ---- Density ---- */
  if (chartType === "density") {
    return (
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={points} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="densityGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={CHART_PALETTE[0]} stopOpacity={0.4} />
              <stop offset="100%" stopColor={CHART_PALETTE[0]} stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid {...gridStyle} vertical={false} />
          <XAxis dataKey="label" tick={tickStyle} axisLine={false} tickLine={false} minTickGap={24} />
          <YAxis tick={tickStyle} axisLine={false} tickLine={false} width={52} tickFormatter={fmtVal} />
          <Tooltip formatter={(v: number) => [fmtVal(v), "Density"]} contentStyle={tooltipStyle} />
          <Area type="monotone" dataKey="value" name="Density" stroke={CHART_PALETTE[0]} fill="url(#densityGrad)" strokeWidth={2.5} isAnimationActive animationDuration={500} />
        </AreaChart>
      </ResponsiveContainer>
    );
  }

  /* ---- Violin (approximate with mirrored area) ---- */
  if (chartType === "violin") {
    const viData = points.map((p: any) => ({ ...p, neg: -p.value }));
    return (
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={viData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="violinGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={CHART_PALETTE[0]} stopOpacity={0.4} />
              <stop offset="100%" stopColor={CHART_PALETTE[0]} stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid {...gridStyle} vertical={false} />
          <XAxis dataKey="label" tick={tickStyle} axisLine={false} tickLine={false} minTickGap={24} />
          <YAxis tick={tickStyle} axisLine={false} tickLine={false} width={52} tickFormatter={fmtVal} />
          <Tooltip formatter={(v: number) => [fmtVal(v), humanize(metric)]} contentStyle={tooltipStyle} />
          <Area type="monotone" dataKey="value" stroke={CHART_PALETTE[0]} fill="url(#violinGrad)" strokeWidth={2} isAnimationActive animationDuration={500} />
        </AreaChart>
      </ResponsiveContainer>
    );
  }

  /* ---- Fallback: simple bar (per-category colors) ---- */
  const fallbackData = data.slice(0, 60).map((r: any) => ({ label: strOf(r?.[xField] ?? r?.label ?? ""), value: numOf(r?.[yField] ?? r?.value) ?? 0 }));
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={fallbackData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
        <CartesianGrid {...gridStyle} vertical={false} />
        <XAxis dataKey="label" tick={tickStyle} axisLine={false} tickLine={false} minTickGap={24} />
        <YAxis tick={tickStyle} axisLine={false} tickLine={false} width={52} tickFormatter={fmtVal} />
        <Tooltip formatter={(v: number) => [fmtVal(v), humanize(metric)]} contentStyle={tooltipStyle} />
        <Bar dataKey="value" name="Current" radius={[4, 4, 0, 0]} isAnimationActive animationDuration={500}>
          {fallbackData.map((d: any, i: number) => <Cell key={i} fill={colorForName(d.label || String(i))} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/* ------------------------------------------------------------------ */
/*  Time-ago                                                           */
/* ------------------------------------------------------------------ */

function formatTimeAgo(ts: string): string {
  try {
    const d = new Date(ts);
    const diff = Date.now() - d.getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return "Just now";
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    const days = Math.floor(hrs / 24);
    if (days < 7) return `${days}d ago`;
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  } catch { return ""; }
}

/* ------------------------------------------------------------------ */
/*  Analysis card (Recent / Saved)                                      */
/* ------------------------------------------------------------------ */

function AnalysisCard({ item, onOpen }: { item: any; onOpen: () => void }) {
  const title = strOf(item?.title || item?.name, "Untitled Analysis");
  const metric = strOf(item?.metric);
  const summary = strOf(item?.result_summary || item?.summary);
  const ts = item?.updated_at || item?.created_at;
  const timeAgo = ts ? formatTimeAgo(ts) : "";
  const changePct = numOf(item?.change_pct ?? item?.result?.change_pct);
  const total = numOf(item?.result?.total);

  return (
    <div className="bg-card border border-border/80 rounded-2xl shadow-[0_1px_3px_rgba(0,0,0,0.04)] hover:shadow-md hover:border-blue-200 dark:hover:border-blue-500/25 transition-all group">
      <div className="p-4">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2.5 min-w-0">
            <span className="inline-flex items-center justify-center w-9 h-9 rounded-xl bg-blue-50 dark:bg-blue-500/10 text-blue-700 dark:text-blue-400 flex-shrink-0">
              <BarChart3 size={16} />
            </span>
            <div className="min-w-0">
              <h4 className="text-[13px] font-semibold text-foreground truncate">{title}</h4>
              <p className="text-[11px] text-muted-foreground flex items-center gap-1">
                {timeAgo && <><Clock size={10} />{timeAgo}</>}
                {!timeAgo && metric && <>{humanize(metric)}</>}
              </p>
            </div>
          </div>
          <button onClick={onOpen} className="text-[11px] font-medium text-blue-700 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 px-2 py-1 rounded-md hover:bg-blue-50 dark:hover:bg-blue-500/10 opacity-0 group-hover:opacity-100 transition-opacity" aria-label="Open analysis">Open</button>
        </div>
        {total !== null && (
          <div className="mt-3 flex items-baseline gap-2">
            <span className="text-[20px] font-bold text-foreground tabular-nums">{fmtMetric(total, metric)}</span>
            {changePct !== null && (
              <span className={`inline-flex items-center gap-0.5 text-[11px] font-semibold ${changePct >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                {changePct >= 0 ? <ArrowUpRight size={11} /> : <ArrowDownRight size={11} />}
                {fmtPctBiz(changePct)}
              </span>
            )}
          </div>
        )}
        {summary && <p className="mt-2 text-[11.5px] text-muted-foreground line-clamp-2 leading-relaxed">{summary}</p>}
      </div>
      <div className="px-4 pb-3">
        <button onClick={onOpen} className="w-full text-left text-[12px] font-semibold text-blue-700 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 flex items-center justify-between py-1.5 border-t border-border mt-1">
          Explore this analysis <ChevronRight size={13} />
        </button>
      </div>
    </div>
  );
}

/* ================================================================== */
/*  Main component                                                     */
/* ================================================================== */

export default function AIAnalytics({
  onNavigate,
  onAskCopilot,
  connectionStatus,
  dbName,
}: AIAnalyticsProps) {
  /* ---- Meta / schema ---- */
  const [meta, setMeta] = useState<any>(null);
  const [metaLoading, setMetaLoading] = useState(true);
  const [metaError, setMetaError] = useState("");

  const loadMeta = useCallback(async () => {
    setMetaLoading(true); setMetaError("");
    try { const r = await intelligenceApi.getMeta(); setMeta(r.data || null); }
    catch (e: any) { setMetaError(normalizeApiError({ message: e?.message }, e?.status || 0, null).message); setMeta(null); }
    finally { setMetaLoading(false); }
  }, []);

  useEffect(() => {
    if (connectionStatus === "connected" || connectionStatus === undefined) loadMeta();
    else { setMeta(null); setMetaLoading(false); }
  }, [connectionStatus, dbName, loadMeta]);

  const ex = useExplore(meta);
  const { recent, savedCols, recentError, loadRecent } = ex;

  /* ---- UI state ---- */
  const [homeTab, setHomeTab] = useState<HomeTab>("recent");
  const [searchQuery, setSearchQuery] = useState("");
  const [sidebarCtx, setSidebarCtx] = useState<{ metric?: string; breakdown?: string }>({});
  /* The question the user asked, kept visible until they dismiss it. */
  const [askedQuestion, setAskedQuestion] = useState("");

  /* ---- Analysis results (on-page) ---- */
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState("");
  const [reportMsg, setReportMsg] = useState("");
  const [followUpQuery, setFollowUpQuery] = useState("");
  const [history, setHistory] = useState<{ q: string; r: any }[]>([]);
  const abortRef = useRef<AbortController | null>(null);
  const conversationRef = useRef<string>(crypto.randomUUID());

  /* ---- Autocomplete ---- */
  const [acOptions, setAcOptions] = useState<string[]>([]);
  const [acOpen, setAcOpen] = useState(false);
  const acTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fetchAc = useCallback((partial: string) => {
    if (acTimer.current) clearTimeout(acTimer.current);
    if (!partial || partial.length < 1) { setAcOptions([]); return; }
    acTimer.current = setTimeout(async () => {
      try {
        const data: any = await apiFetch(`/autocomplete?partial=${encodeURIComponent(partial)}&limit=6`);
        const list = Array.isArray(data?.suggestions) ? data.suggestions.map((s: any) => typeof s === "string" ? s : s?.text || "").filter(Boolean) : [];
        setAcOptions(list);
      } catch { setAcOptions([]); }
    }, 200);
  }, []);

  /* ---- Suggestions (starter chips) — refreshed for the connected
     database, and stale analysis is cleared on dataset switch so old
     results (old columns) can never linger into a new database. ---- */
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const datasetKey = dbName || "";
  const prevDatasetRef = useRef<string | null>(null);
  const loadSuggestions = useCallback(async () => {
    try {
      const data: any = await apiFetch("/suggestions", { method: "POST", body: JSON.stringify({ limit: 4 }) });
      if (Array.isArray(data?.suggestions) && data.suggestions.length) setSuggestions(data.suggestions.slice(0, 4));
      else setSuggestions([]);
    } catch { /* keep previous chips on transient errors */ }
  }, []);
  useEffect(() => {
    loadSuggestions();
    if (prevDatasetRef.current !== null && prevDatasetRef.current !== datasetKey) {
      abortRef.current?.abort();
      conversationRef.current = crypto.randomUUID();
      setRunning(false);
      setResult(null); setError(""); setHistory([]); setAskedQuestion("");
      setSearchQuery(""); setFollowUpQuery(""); setAcOptions([]); setAcOpen(false);
      setSidebarCtx({});
    }
    prevDatasetRef.current = datasetKey;
  }, [connectionStatus, datasetKey, loadSuggestions]);

  /* ---- Navigation ---- */
  const gotoTab = useCallback((tab: string) => {
    try { window.location.hash = `#/intelligence?tab=${tab}`; } catch { /* ignore */ }
    if (onNavigate) { try { onNavigate("intelligence"); } catch { /* ignore */ } }
  }, [onNavigate]);

  /* ---- Hypothesis handoffs (existing mechanisms, dynamic values only) ----
     Test-this: Explore handoff via sessionStorage `aria_explore_prefill`
     plus the drill-context pattern (localStorage `aria_dashboard_drill`). */
  const hypothesisHandoff = useCallback(() => {
    try {
      const data: any = (result as any)?.data || result || {};
      const env: any = data?.analytics_envelope || data || {};
      const hyp: any = env?.hypothesis || {};
      const drv: any = ((env?.possible_drivers || [])[0]) || {};
      const ex: any = hyp?.explore_handoff || {};
      const metric = strOf(ex?.metric || env?.metric || chartMetricFallback());
      const breakdown = strOf(ex?.breakdown || (env?.dimensions || [])[0] || drv?.dimension || "");
      const segment = strOf(ex?.segment || drv?.value || "");
      return { metric, breakdown, segment };
    } catch {
      return { metric: "", breakdown: "", segment: "" };
    }
    function chartMetricFallback(): string {
      try {
        const d: any = (result as any)?.data || result || {};
        return strOf(d?.metric || d?.display?.metric || "");
      } catch {
        return "";
      }
    }
  }, [result]);

  const testHypothesis = useCallback(() => {
    const h = hypothesisHandoff();
    try {
      if (h.metric || h.breakdown) {
        sessionStorage.setItem("aria_explore_prefill", JSON.stringify({
          ...(h.metric ? { metric: h.metric } : {}),
          ...(h.breakdown ? { breakdown: h.breakdown } : {}),
        }));
      }
    } catch { /* best-effort */ }
    try {
      localStorage.setItem("aria_dashboard_drill", JSON.stringify({
        at: Date.now(), kind: "hypothesis",
        ...(h.metric ? { metric: h.metric } : {}),
        ...(h.breakdown ? { dimension: h.breakdown, breakdown: h.breakdown } : {}),
        ...(h.segment ? { segment: h.segment, value: h.segment } : {}),
      }));
    } catch { /* best-effort */ }
    gotoTab("explore");
  }, [hypothesisHandoff, gotoTab]);

  /* One-click "Add to report": feeds the shared report draft
     (localStorage `aria_report_draft`) consumed by Intelligence Reports. */
  const addResultToReport = useCallback(() => {
    setReportMsg("");
    try {
      const data: any = (result as any)?.data || result || {};
      const title = strOf(askedQuestion)
        || strOf(data?.title) || strOf(data?.display?.title) || "AI Analytics result";
      const key = "aria_report_draft";
      let cur: any[] = [];
      try {
        const v = JSON.parse(localStorage.getItem(key) || "[]");
        if (Array.isArray(v)) cur = v;
      } catch { cur = []; }
      if (cur.some((x) => String(x?.title || "") === title)) {
        setReportMsg("Already in the report draft. Open Intelligence → Reports to generate.");
        return;
      }
      cur.push({ id: "", title });
      localStorage.setItem(key, JSON.stringify(cur));
      setReportMsg(`Added to the report draft (${cur.length}). Open Intelligence → Reports to generate.`);
    } catch { setReportMsg("Couldn't update the report draft."); }
  }, [result, askedQuestion]);

  /* ---- Run analysis on-page ---- */
  const runOnPage = useCallback(async (question: string) => {
    const q = question.trim();
    if (!q || running) return;
    setAskedQuestion(q);
    setSearchQuery(""); setFollowUpQuery(""); setAcOptions([]); setAcOpen(false);
    setRunning(true); setError(""); setResult(null);
    abortRef.current?.abort();
    const ctrl = new AbortController(); abortRef.current = ctrl;
    try {
      const data = await analyzeQuestion(q, null, { dataset: datasetKey || undefined }, { conversationId: conversationRef.current }, { signal: ctrl.signal, timeoutMs: 120000, useSharedContext: false });
      if (ctrl.signal.aborted) return;
      setResult(data);
      setSidebarCtx({ metric: data?.analytics_envelope?.metric || data?.display?.metric, breakdown: (data?.analytics_envelope?.dimensions || [])[0] });
      setHistory((h) => [...h.slice(-9), { q, r: data }]);
    } catch (e: any) {
      if (e?.name === "AbortError") return;
      setError(e?.message || "Could not analyze this question.");
    } finally { if (abortRef.current === ctrl) setRunning(false); }
  }, [running, datasetKey]);

  /* ---- Follow-up submit ---- */
  const submitFollowUp = () => {
    const q = followUpQuery.trim();
    if (!q) return;
    runOnPage(q);
  };

  /* ---- Popular action click ---- */
  const handlePopularAction = (a: typeof POPULAR_ACTIONS[number]) => {
    if (a.navigate) { gotoTab(a.navigate); return; }
    if (a.query) runOnPage(a.query);
  };

  /* ---- Sidebar action click ---- */
  const handleSidebarAction = (a: typeof SIDEBAR_ACTIONS[number]) => {
    const ctx = sidebarCtx.metric ? ` about ${humanize(sidebarCtx.metric)}` : "";
    runOnPage(`${a.query}${ctx}`);
  };

  /* ---- Tab items ---- */
  const tabItems = useMemo(() => {
    if (homeTab === "recent") return asList(recent);
    if (homeTab === "saved") return asList(savedCols);
    return [];
  }, [homeTab, recent, savedCols]);

  /* ---- Open saved analysis from card ---- */
  const openAnalysis = (item: any) => {
    const q = `Analyze ${item?.metric || "data"}${item?.breakdown ? ` by ${item.breakdown}` : ""}`;
    runOnPage(q);
  };

  /* ---- Back to home ---- */
  const goHome = () => {
    abortRef.current?.abort(); abortRef.current = null;
    conversationRef.current = crypto.randomUUID();
    setRunning(false); setResult(null); setError(""); setHistory([]); setAskedQuestion("");
    setSearchQuery(""); setFollowUpQuery(""); setSidebarCtx({});
  };

  /* ---- Result normalization (analyzeQuestion returns {success, data: goalData}) ---- */
  const gd = result?.data || result;  // goalData or raw
  const csvResults: any[] = Array.isArray(gd?.analysis_results) ? gd.analysis_results : [];
  const answerMessage = strOf(gd?.answer || gd?.message || "");
  const findings: string[] = asList(gd?.findings || []).map((f: any) => strOf(typeof f === "string" ? f : f?.text)).filter(Boolean);
  const vizData = gd?.visualization?.data || gd?.data || [];
  const vizType = gd?.visualization?.type || "bar";
  const chartMetric = strOf(gd?.metric || gd?.display?.metric || gd?.processedDataSummary?.columns?.[0] || "");
  const columns: string[] = gd?.processedDataSummary?.columns || gd?.columns || [];
  const followUps: string[] = asList(gd?.followUpQuestions || gd?.suggestedQuestions || (gd as any)?.follow_up_questions || (gd as any)?.suggested_questions);
  const story = strOf(gd?.story || "");
  const limitations: string[] = asList(gd?.limitations || gd?.why?.limitations || []);
  const b360: any = (gd as any)?.business360 || (result as any)?.business360 || null;
  /* Structured insight fields (backend envelope): possible drivers with
     shares, hypothesis, and recommendations. Rendered in plain words —
     raw table.column identifiers are humanized, statistician phrasing
     ("statistical association") is reworded. Generic, no schema names. */
  const drivers: any[] = asList((gd as any)?.possible_drivers || (gd as any)?.possibleDrivers || []);
  const hypothesis: any = (gd as any)?.hypothesis || null;
  const recommendations: any[] = asList((gd as any)?.recommendations || []).map((r: any) =>
    typeof r === "string" ? { text: r } : r);
  const plainWords = (s: string): string => {
    let t = strOf(s);
    if (!t) return "";
    t = t.replace(/statistical association/gi, "pattern in the data");
    t = t.replace(/Statistical association only,? not proven cause\.?/gi,
      "Things moving together does not prove what causes what.");
    t = t.replace(/not a proven cause/gi, "not proof of what causes it");
    return t;
  };
  // Short identifier-safe labeler (columns/dims only — never sentences).
  const identLabel = (s: any): string => {
    const t = strOf(s);
    if (!t || t.length > 80 || /\s{2,}/.test(t)) return t;
    try {
      return humanize(t.split(".").pop() || t);
    } catch {
      return t;
    }
  };
  const hasInsightsBase = findings.length > 0 || drivers.length > 0 || !!hypothesis || recommendations.length > 0;
  void hasInsightsBase;
  /* Question-aware Key Insights ordering + temporal gating — derived only
     from existing gd fields (interpreted_intent, comparison_period
     granularity, findings, drivers). No new fetches, no fabricated numbers.
     - ranking intent leads with winner + share (from drivers[0])
     - trend intent leads with the direction/change finding (from findings)
     - the backend comparison blurb ("Compared current month vs previous
       month…") renders only when the answer is temporal. */
  const intentPrimary: string = (() => {
    try {
      const raw: any = (gd as any)?.interpreted_intent ?? (gd as any)?.interpretedIntent ?? "";
      const p = typeof raw === "string" ? raw : raw?.primary || "";
      return strOf(p).toLowerCase();
    } catch { return ""; }
  })();
  const comparisonPeriod: any = (gd as any)?.comparison_period ?? (gd as any)?.comparisonPeriod ?? null;
  const comparisonGranularity = strOf(comparisonPeriod?.granularity || "").toLowerCase();
  void comparisonGranularity;
  const comparisonDescription = strOf(comparisonPeriod?.description || "");
  const isRankingIntent = /rank/.test(intentPrimary);
  const isTrendIntent = /trend|forecast|change_detection|anomaly|period_over_period/.test(intentPrimary);
  const isComparisonIntent = /compar|period_over_period|contribution/.test(intentPrimary);
  const temporalHint = /day|week|month|quarter|year|over time|previous|current period|trend|growth|declin|increas|decreas|\bmom\b|\byoy\b|\bqoq\b/i;
  const answerTemporal = temporalHint.test(`${answerMessage} ${askedQuestion} ${findings.join(" ")}`);
  const isTemporalAnswer = isTrendIntent || isComparisonIntent || intentPrimary === "period_over_period" || answerTemporal;
  const isComparisonFinding = (f: string): boolean => {
    const t = strOf(f);
    if (!t) return false;
    if (comparisonDescription && t.trim() === comparisonDescription.trim()) return true;
    return /compared\s+current\s+\w+.*vs\s+previous\s+\w+|period-over-period\s+on\s+\w+\s+granularity|current\s+month\s+vs\s+previous\s+month/i.test(t);
  };
  const visibleFindings: string[] = findings.filter((f) => (isTemporalAnswer ? true : !isComparisonFinding(f)));
  /* Ranking lead (winner + share) synthesized from the existing drivers[0]
     entry — real share only, humanized identifiers, never fabricated. */
  const topDriverVal = strOf((drivers[0] as any)?.value || "");
  const topDriverShare = numOf((drivers[0] as any)?.contribution_pct ?? (drivers[0] as any)?.share_pct);
  const rankingLead: string | null = (isRankingIntent && topDriverVal)
    ? (topDriverShare !== null
      ? `${humanize(topDriverVal.split(".").pop() || topDriverVal)} leads with about ${Math.round(topDriverShare)}% of the total`
      : `${humanize(topDriverVal.split(".").pop() || topDriverVal)} leads`)
    : null;
  const trendSignal = /increas|decreas|\bgrowth\b|\bdeclin|\brose\b|\bfell\b|\bdropped\b|\btrend\b|change[sd]?\s+by|\+?-?\d+(\.\d+)?%/i;
  let orderedFindings: string[] = [...visibleFindings];
  if (isTrendIntent) {
    const idx = orderedFindings.findIndex((f) => trendSignal.test(f));
    if (idx > 0) {
      const [t] = orderedFindings.splice(idx, 1);
      orderedFindings.unshift(t);
    }
  }
  if (rankingLead) {
    const winnerLc = topDriverVal.split(".").pop()?.toLowerCase() || "";
    const alreadyLeads = orderedFindings.length > 0 && winnerLc
      ? orderedFindings[0].toLowerCase().includes(winnerLc.slice(0, Math.min(12, winnerLc.length)))
      : false;
    if (!alreadyLeads) orderedFindings = [rankingLead, ...orderedFindings.filter((f) => f !== rankingLead)];
  }
  /* Leak lockdown: humanize raw identifiers inside insight sentences
     (table.column, snake_case) without rewriting prose. SQL never renders
     in this card — it lives only in the evidence drawer. */
  const humanizeIdentifiers = (s: string): string => {
    let t = strOf(s);
    if (!t) return "";
    t = t.replace(/([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_.]*)/g, (m) => {
      try { return humanize((m.split(".").pop() || m).replace(/["'`]/g, "")); } catch { return m; }
    });
    t = t.replace(/\b([A-Za-z]+_[A-Za-z0-9_]+)\b/g, (m) => {
      if (m.length > 80) return m;
      try { return humanize(m); } catch { return m; }
    });
    return t;
  };
  const insightText = (s: string): string => humanizeIdentifiers(plainWords(s));
  const hasInsights = orderedFindings.length > 0 || drivers.length > 0 || !!hypothesis || recommendations.length > 0;

  const hasDataset = meta && (meta?.metrics?.length > 0 || meta?.measures?.length > 0);
  const showResults = result || running || error;

  /* ---- Chart spec: chart_spec first, visualization envelope as fallback.
      Both carry REAL column names (x/y) pointing into the data keys — never
      positional guesses, so charts show names (artists, regions…) not blanks. */
  const _cs: any = (gd as any)?.chart_spec || {};
  const _vz: any = (gd as any)?.visualization || {};
  const chartSpec = {
    chart_type: _cs.chart_type || _vz.type || "bar",
    x: _cs.x || _vz.x || "",
    y: _cs.y || _vz.y || "",
    label: _cs.label || "",
  };
  const hasChartData = vizData.length >= 2;
  const hasAxes = !!chartSpec.x && !!chartSpec.y;
  const isChartType = chartSpec.chart_type !== "table" && chartSpec.chart_type !== "text";
  const showChart = hasChartData && hasAxes && isChartType;
  /* Is the answer text-only (no chart, no findings, just narrative)? */
  const isTextOnly = !showChart && visibleFindings.length === 0 && !story;

  /* ------------------------------------------------------------------ */
  /*  Not connected / Loading / Error / No dataset                       */
  /* ------------------------------------------------------------------ */

  if (connectionStatus !== undefined && connectionStatus !== "connected") {
    return (
      <div className="min-h-full bg-background">
        <div className="p-6 max-w-screen-xl mx-auto">
          <p className="text-[11px] font-bold tracking-[0.14em] text-blue-700 dark:text-blue-400">ARIA</p>
          <h1 className="text-[28px] font-bold text-foreground tracking-tight mt-1">Analytics</h1>
          <p className="text-[14px] text-muted-foreground mt-0.5">Turn your data into answers.</p>
          <div className="mt-8 text-center py-16 bg-card border border-border rounded-2xl">
            <Database size={28} className="mx-auto text-muted-foreground mb-3" />
            <p className="text-sm font-medium">No data source connected</p>
            {onNavigate && <button onClick={() => onNavigate("data")} className="mt-4 inline-flex items-center gap-2 px-4 py-2 rounded-xl text-[13px] font-semibold text-white bg-blue-700 hover:bg-blue-800">Go to Data</button>}
          </div>
        </div>
      </div>
    );
  }
  if (metaLoading) return <div className="min-h-full bg-background"><div className="p-6 max-w-screen-xl mx-auto"><LoadingPanel label="Loading analytics…" /></div></div>;
  if (metaError && !meta) return <div className="min-h-full bg-background"><div className="p-6 max-w-screen-xl mx-auto"><ErrorPanel message={metaError} onRetry={loadMeta} retryLabel="Reload" /></div></div>;
  if (meta && !hasDataset) return <div className="min-h-full bg-background"><div className="p-6 max-w-screen-xl mx-auto"><NoDatasetPanel onRetry={loadMeta} hint={(() => { try { return (getNoSchemaInfo(meta).hint as string) || undefined; } catch { return undefined; } })()} /></div></div>;

  /* ------------------------------------------------------------------ */
  /*  Render                                                             */
  /* ------------------------------------------------------------------ */

  return (
    <div className="min-h-full bg-background">
      <div className="flex gap-6 p-4 md:p-6 max-w-[1440px] mx-auto">
        {/* ---- Main column ---- */}
        <div className="flex-1 min-w-0 space-y-5">
          {/* Header */}
          <div>
            <p className="text-[11px] font-bold tracking-[0.14em] text-blue-700 dark:text-blue-400">ARIA <span className="text-muted-foreground font-normal tracking-normal">/ Analytics</span></p>
            <div className="flex items-start justify-between gap-3 mt-1">
              <div>
                <h1 className="text-[28px] font-bold text-foreground tracking-tight">Analytics</h1>
                <p className="text-[14px] text-muted-foreground mt-0.5">Turn your data into answers. Explore, compare, and uncover what matters.</p>
              </div>
              <div className="flex items-center gap-2 flex-shrink-0">
                {!showResults && (
                  <button onClick={() => gotoTab("explore")} className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg text-[12.5px] font-medium text-secondary-foreground border border-border hover:text-foreground hover:border-border bg-card transition-colors">
                    <Bookmark size={13} /> Saved analyses
                  </button>
                )}
                {showResults ? (
                  <button onClick={goHome} className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg text-[12.5px] font-semibold text-white bg-blue-700 hover:bg-blue-800 transition-colors">
                    <Plus size={14} /> New analysis
                  </button>
                ) : (
                  <button onClick={goHome} className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg text-[12.5px] font-semibold text-white bg-blue-700 hover:bg-blue-800 transition-colors">
                    <Plus size={14} /> New analysis
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Compact search bar — always visible when results are shown */}
          {showResults && (
            <div className="relative">
              <div className="flex items-center gap-2 rounded-xl border border-border bg-card px-3 py-2.5 focus-within:border-blue-400 transition-colors shadow-[0_1px_3px_rgba(0,0,0,0.04)]">
                <Search size={15} className="text-muted-foreground flex-shrink-0" />
                <input
                  value={searchQuery}
                  onChange={(e) => { setSearchQuery(e.target.value); fetchAc(e.target.value); setAcOpen(true); }}
                  onFocus={() => setAcOpen(true)}
                  onBlur={() => setTimeout(() => setAcOpen(false), 200)}
                  onKeyDown={(e) => { if (e.key === "Enter") { setAcOpen(false); runOnPage(searchQuery); } }}
                  placeholder="Ask another question..."
                  className="flex-1 bg-transparent outline-none text-[13px] text-foreground placeholder:text-muted-foreground"
                  aria-label="Ask a question"
                />
                <button onClick={() => { setAcOpen(false); runOnPage(searchQuery); }} className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-blue-700 text-white hover:bg-blue-800 transition-colors flex-shrink-0" aria-label="Submit">
                  <Send size={14} />
                </button>
              </div>
              {acOpen && acOptions.length > 0 && (
                <div className="absolute top-full left-0 right-0 mt-1 bg-card border border-border rounded-xl shadow-lg py-1.5 z-30 max-h-[240px] overflow-y-auto">
                  {acOptions.map((s, i) => (
                    <button key={s + i} onMouseDown={() => { setAcOpen(false); setSearchQuery(s); runOnPage(s); }}
                      className="w-full flex items-center gap-3 px-4 py-2.5 text-[12px] text-foreground hover:bg-secondary transition-colors text-left">
                      <Zap size={13} className="text-blue-600 dark:text-blue-400 flex-shrink-0" />
                      <span className="truncate">{s}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Start a new analysis — only show when no results */}
          {!showResults && (
            <div className="bg-card border border-border/80 rounded-2xl shadow-[0_1px_3px_rgba(0,0,0,0.04)] p-5">
              <div className="flex items-center gap-2.5 mb-1">
                <span className="inline-flex items-center justify-center w-8 h-8 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 text-white"><Sparkles size={15} /></span>
                <div>
                  <h3 className="text-[14px] font-semibold text-foreground">Start a new analysis</h3>
                  <p className="text-[11.5px] text-muted-foreground">Ask a question or choose from common analysis types.</p>
                </div>
              </div>
              <div className="mt-3 relative">
                <div className="flex items-center gap-2 rounded-xl border border-border bg-secondary/60 px-3 py-2.5 focus-within:border-blue-400 focus-within:bg-card transition-colors">
                  <Search size={15} className="text-muted-foreground flex-shrink-0" />
                  <input
                    value={searchQuery}
                    onChange={(e) => { setSearchQuery(e.target.value); fetchAc(e.target.value); setAcOpen(true); }}
                    onFocus={() => setAcOpen(true)}
                    onBlur={() => setTimeout(() => setAcOpen(false), 200)}
                    onKeyDown={(e) => { if (e.key === "Enter") { setAcOpen(false); runOnPage(searchQuery); } }}
                    placeholder="E.g. Why did revenue drop last month?"
                    className="flex-1 bg-transparent outline-none text-[13px] text-foreground placeholder:text-muted-foreground"
                    aria-label="Ask a question"
                  />
                  <button onClick={() => { setAcOpen(false); runOnPage(searchQuery); }} className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-blue-700 text-white hover:bg-blue-800 transition-colors flex-shrink-0" aria-label="Submit">
                    <ChevronRight size={15} />
                  </button>
                </div>
                {acOpen && acOptions.length > 0 && (
                  <div className="absolute top-full left-0 right-0 mt-1 bg-card border border-border rounded-xl shadow-lg py-1.5 z-30 max-h-[240px] overflow-y-auto">
                    {acOptions.map((s, i) => (
                      <button key={s + i} onMouseDown={() => { setAcOpen(false); setSearchQuery(s); runOnPage(s); }}
                        className="w-full flex items-center gap-3 px-4 py-2.5 text-[12px] text-foreground hover:bg-secondary transition-colors text-left">
                        <Zap size={13} className="text-blue-600 dark:text-blue-400 flex-shrink-0" />
                        <span className="truncate">{s}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <div className="mt-3">
                <p className="text-[11px] font-medium text-muted-foreground mb-2">Popular actions</p>
                <div className="flex items-center gap-2 flex-wrap">
                  {POPULAR_ACTIONS.map((a) => (
                    <button key={a.label} onClick={() => handlePopularAction(a)} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[12px] font-medium text-secondary-foreground border border-border hover:border-blue-300 dark:hover:border-blue-500/25 hover:text-blue-700 dark:hover:text-blue-400 hover:bg-blue-50/50 dark:hover:bg-blue-500/10 transition-colors">
                      <a.icon size={13} /> {a.label}
                    </button>
                  ))}
                </div>
              </div>
              {suggestions.length > 0 && (
                <div className="mt-3">
                  <p className="text-[11px] font-medium text-muted-foreground mb-2">Suggested questions</p>
                  <div className="flex items-center gap-2 flex-wrap">
                    {suggestions.map((s, i) => (
                      <button key={s + i} onClick={() => runOnPage(s)} className="px-3 py-1.5 rounded-lg text-[12px] font-medium text-blue-700 dark:text-blue-400 border border-blue-200 dark:border-blue-500/25 hover:bg-blue-50 dark:hover:bg-blue-500/10 transition-colors">
                        {s}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Results — inline on-page */}
          {showResults && (
            <div className="space-y-4">
              {/* Running */}
              {running && <LoadingPanel label="Analyzing your question…" />}

              {/* Error */}
              {!running && error && (
                <Card className="p-5">
                  <p className="text-[13px] text-rose-600 dark:text-rose-400 font-medium">{error}</p>
                  <button onClick={() => runOnPage(searchQuery || history[history.length - 1]?.q || "")} className="mt-2 text-[12px] font-semibold text-blue-700 dark:text-blue-400 hover:underline">Try again</button>
                </Card>
              )}

              {/* Answer + Chart + Insights */}
              {!running && !error && result && (
                <div className="space-y-4 chart-reveal">
                  {history.length > 1 && <details className="rounded-xl border border-border p-4 text-sm">
                    <summary className="cursor-pointer font-semibold">Current chat · {history.length} turns</summary>
                    {history.slice(0, -1).map((turn, index) => <div key={index} className="mt-3 border-t pt-3">
                      <p className="font-medium">You: {turn.q}</p>
                      <p className="mt-1 text-muted-foreground">ARIA: {turn.r?.data?.answer || turn.r?.data?.message || turn.r?.answer || turn.r?.message}</p>
                      <button type="button" className="mt-1 text-xs underline" onClick={() => { setResult(turn.r); setAskedQuestion(turn.q); }}>View this result</button>
                    </div>)}
                    <p className="mt-3 text-xs text-muted-foreground">Follow-ups use the latest conversation context. New analysis starts a separate chat.</p>
                  </details>}
                  {/* Asked question — stays until the user dismisses it */}
                  {askedQuestion && (
                    <div className="flex items-start gap-2.5 rounded-xl border border-border bg-secondary/70 px-4 py-3">
                      <Search size={14} className="text-muted-foreground flex-shrink-0 mt-0.5" />
                      <div className="flex-1 min-w-0">
                        <p className="text-[10.5px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">Your question</p>
                        <p className="text-[13px] font-medium text-foreground leading-snug break-words">{askedQuestion}</p>
                      </div>
                      <button onClick={() => setAskedQuestion("")} className="inline-flex items-center justify-center w-6 h-6 rounded-md text-muted-foreground hover:text-secondary-foreground hover:bg-border/70 transition-colors flex-shrink-0" aria-label="Dismiss question" title="Dismiss">
                        <X size={13} />
                      </button>
                    </div>
                  )}
                  {/* One-click report handoff — feeds the Intelligence Reports draft */}
                  <div className="flex items-center gap-2 flex-wrap">
                    <button
                      onClick={addResultToReport}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[12px] font-semibold text-blue-700 dark:text-blue-400 border border-blue-200 dark:border-blue-500/25 hover:bg-blue-50 dark:hover:bg-blue-500/10"
                      aria-label="Add this analysis to the report draft">
                      <FileText size={13} /> Add to report
                    </button>
                    {reportMsg && <span className="text-[11.5px] text-secondary-foreground" role="status">{reportMsg}</span>}
                  </div>
                  {/* Answer message */}
                  {answerMessage && (
                    <Card className="p-5">
                      <div className="flex items-start gap-3">
                        <span className="inline-flex items-center justify-center w-8 h-8 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 text-white flex-shrink-0"><Sparkles size={15} /></span>
                        <div className="flex-1 min-w-0">
                          <p className="text-[13px] text-foreground leading-relaxed">{answerMessage}</p>
                          {(() => {
                            const v: any = (gd as any)?.verdict;
                            if (!v) return null;
                            const ok = v.verified === true;
                            const fails = Array.isArray(v.checks) ? v.checks.filter((c: any) => !c?.pass).length : 0;
                            return (
                              <span
                                title={ok
                                  ? "ARIA double-checked this answer against your data. Everything lines up."
                                  : "ARIA couldn't fully double-check this answer yet. Treat it as a guide, not the final answer."}
                                className={`mt-2 inline-flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-full border ${ok
                                  ? "bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-500/25"
                                  : "bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-500/25"}`}
                              >
                                <Shield size={11} />
                                {ok ? "Checked" : "Not fully checked"}
                                {!ok && fails > 0 ? ` · ${fails} to review` : ""}
                              </span>
                            );
                          })()}
                        </div>
                      </div>
                    </Card>
                  )}

                  {/* Follow-up suggestion chips */}
                  {(() => {
                    const fus = asList(gd?.followUpQuestions || gd?.suggestedQuestions || (gd as any)?.follow_up_questions || (gd as any)?.suggested_questions);
                    if (!fus.length) return null;
                    return (
                      <div className="flex items-center gap-2 flex-wrap">
                        {fus.slice(0, 4).map((fq: string, i: number) => (
                          <button key={fq + i} onClick={() => runOnPage(fq)} className="px-3 py-1.5 rounded-lg text-[12px] font-medium text-blue-700 dark:text-blue-400 border border-blue-200 dark:border-blue-500/25 hover:bg-blue-50 dark:hover:bg-blue-500/10 transition-colors">
                            {humanize(fq)}
                          </button>
                        ))}
                      </div>
                    );
                  })()}

                  {csvResults.map((item: any, index: number) => {
                    const rows: any[] = item.data || [];
                    const spec = item.chart_spec || {};
                    const chartRows = rows.filter(row => typeof row[spec.y] === "number");
                    return <Card key={index} className="p-5">
                      <CardHead title={item.title} sub={item.reason} />
                      {spec.chart_type !== "table" && spec.x && spec.y && chartRows.length > 0 && <div className="h-[260px] mt-3">
                        <DynamicChart data={chartRows} chartSpec={spec} metric={spec.y} />
                      </div>}
                      {chartRows.length !== rows.length && spec.chart_type !== "table" && <p className="text-xs mt-2 text-muted-foreground">Unavailable values are omitted from the chart; see the table for all values.</p>}
                      {rows.length > 60 && spec.chart_type !== "table" && <p className="text-xs mt-2 text-muted-foreground">Chart shows the first 60 available points. The table contains all returned rows.</p>}
                      <div className="overflow-auto max-h-80 mt-3">
                        {rows.length ? <table className="w-full text-xs"><thead><tr>{item.columns.map((col: string) => <th key={col} className="text-left p-2 border-b">{humanize(col)}</th>)}</tr></thead>
                          <tbody>{rows.map((row: any, i: number) => <tr key={i}>{item.columns.map((col: string) => <td key={col} className="p-2 border-b">{row[col] == null ? "Unavailable (NULL)" : typeof row[col] === "number" ? row[col].toLocaleString(undefined, { maximumSignificantDigits: 12 }) : String(row[col])}</td>)}</tr>)}</tbody></table>
                          : <p className="text-sm text-muted-foreground">No matching rows. This does not imply a zero total.</p>}
                      </div>
                      <details className="mt-3 text-xs"><summary className="cursor-pointer">How this was calculated</summary><pre className="overflow-auto whitespace-pre-wrap mt-2">{item.sql}</pre></details>
                    </Card>;
                  })}
                  {gd?.dataset_understanding && <Card className="p-5 text-sm space-y-2">
                    <p>{gd.dataset_understanding.description}</p>
                    {(gd.dataset_understanding.assumptions || []).map((text: string, i: number) => <p key={i}>Assumption: {text}</p>)}
                    {(gd.limitations || []).map((text: string, i: number) => <p key={i} className="text-muted-foreground">{text}</p>)}
                    {gd.dataset_profile?.profile && <p className="text-xs text-muted-foreground">Data quality profile: {gd.dataset_profile.profile.scope}. Analysis queries use the connected dataset.</p>}
                  </Card>}
                  {gd?.llm_trace && <details className="text-xs text-muted-foreground rounded-xl border p-4">
                    <summary className="cursor-pointer">AI execution details</summary>
                    <p className="mt-2">Provider: {gd.llm_trace.provider} · {gd.llm_trace.cached ? "Cached result" : "Fresh analysis"} · Provider invocations: {gd.llm_trace.cached ? 0 : gd.llm_trace.provider_invocation_count}</p>
                    {(gd.llm_trace.calls || []).map((call: any, i: number) => <p key={i}>{call.stage} · {call.model} · {call.duration_ms} ms · {call.status}</p>)}
                  </details>}

                  {/* Chart — adapts to result type: chart for visual data, table for tabular, nothing for text-only */}
                  {!csvResults.length && showChart && (
                    <Card className="p-5">
                      <CardHead title={chartSpec.label || (chartMetric ? `${humanize(chartMetric)} Analysis` : "Analysis Results")} sub={`${vizData.length} data points`} />
                      <div className="h-[260px] mt-3">
                        <DynamicChart data={vizData} chartSpec={chartSpec} metric={chartMetric} />
                      </div>
                    </Card>
                  )}

                  {/* Table — for table-type results or when chart axes are missing but data exists */}
                  {!csvResults.length && !showChart && vizData.length > 0 && (() => {
                    const keys = vizData.length > 0 ? Object.keys(vizData[0]).filter((k) => k !== "_idx") : [];
                    if (keys.length === 0) return null;
                    return (
                      <Card className="p-5">
                        <CardHead title={chartSpec.label || (chartMetric ? `${humanize(chartMetric)} Data` : "Data")} sub={`${vizData.length} rows`} />
                        <div className="mt-3 overflow-x-auto">
                          <table className="w-full text-[12px]">
                            <thead>
                              <tr className="border-b border-border">
                                {keys.map((k) => (
                                  <th key={k} className="text-left font-semibold text-secondary-foreground px-3 py-2 whitespace-nowrap">{humanize(k)}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {vizData.slice(0, 20).map((row: any, i: number) => (
                                <tr key={i} className="border-b border-border hover:bg-secondary/50">
                                  {keys.map((k) => (
                                    <td key={k} className="px-3 py-2 text-foreground tabular-nums whitespace-nowrap">{typeof row[k] === "number" ? row[k].toLocaleString() : String(row[k] ?? "—")}</td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                          {vizData.length > 20 && <p className="mt-2 text-[11px] text-muted-foreground">Showing first 20 of {vizData.length} rows.</p>}
                        </div>
                      </Card>
                    );
                  })()}

                  {/* Key Insights: findings + what stands out + what it might
                      mean + what to do — plain words, never raw identifiers
                      or statistician phrasing. Sections appear only when the
                      backend actually provided them. */}
                  {hasInsights && (
                    <Card className="p-5">
                      <h3 className="text-[14px] font-semibold text-foreground tracking-tight flex items-center gap-1.5">
                        <Sparkles size={14} className="text-blue-700 dark:text-blue-400" /> Key Insights
                      </h3>
                      {orderedFindings.length > 0 && (
                        <ul className="mt-2 space-y-2">
                          {orderedFindings.map((f, i) => (
                            <li key={i} className="flex items-start gap-2 text-[12.5px] text-secondary-foreground leading-relaxed">
                              <span className="w-1.5 h-1.5 rounded-full bg-blue-600 flex-shrink-0 mt-1.5" />{insightText(f)}
                            </li>
                          ))}
                        </ul>
                      )}
                      {drivers.length > 0 && (
                        <div className="mt-3">
                          <p className="text-[12px] font-semibold text-foreground">What stands out</p>
                          <ul className="mt-1.5 space-y-1.5">
                            {drivers.slice(0, 3).map((d: any, i: number) => {
                              const share = numOf(d?.contribution_pct ?? d?.share_pct);
                              return (
                                <li key={i} className="flex items-start gap-2 text-[12.5px] text-secondary-foreground leading-relaxed">
                                  <span className="inline-flex items-center justify-center min-w-5 h-5 px-1 rounded-md bg-blue-50 dark:bg-blue-500/10 text-blue-700 dark:text-blue-400 text-[11px] font-bold flex-shrink-0 mt-0.5">{i + 1}</span>
                                  <span>
                                    <span className="font-semibold text-foreground">{identLabel(d?.value)}</span>
                                    {strOf(d?.dimension) && (
                                      <span className="text-muted-foreground"> ({identLabel(d?.dimension)})</span>
                                    )}
                                    {share !== null && (
                                      <span> — about <span className="font-semibold text-foreground tabular-nums">{Math.round(share)}%</span> of the total</span>
                                    )}
                                  </span>
                                </li>
                              );
                            })}
                          </ul>
                        </div>
                      )}
                      {(strOf(hypothesis?.text) || (hypothesis && strOf(hypothesis))) && (
                        <div className="mt-3 rounded-xl bg-secondary border border-border px-3 py-2.5">
                          <div className="flex items-center gap-2 flex-wrap">
                            <p className="text-[12px] font-semibold text-foreground">What this might mean</p>
                            {(() => {
                              const g = strOf((hypothesis as any)?.evidence_grade || "").toLowerCase();
                              if (!g) return null;
                              const tone = /strong/.test(g)
                                ? "bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-500/25"
                                : /mixed/.test(g)
                                  ? "bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-500/25"
                                  : "bg-secondary text-secondary-foreground border-border";
                              const label = /strong/.test(g) ? "Strong evidence" : /mixed/.test(g) ? "Mixed evidence" : "Weak evidence";
                              return (
                                <span className={`text-[10.5px] font-semibold px-2 py-0.5 rounded-full border ${tone}`} title={strOf((hypothesis as any)?.evidence_reason || "")}>
                                  {label}
                                </span>
                              );
                            })()}
                          </div>
                          <p className="mt-0.5 text-[12.5px] text-secondary-foreground leading-relaxed">
                            {insightText(strOf(hypothesis?.text) || strOf(hypothesis))}
                          </p>
                          {strOf((hypothesis as any)?.evidence_reason) && (
                            <p className="mt-1 text-[11.5px] text-muted-foreground leading-relaxed">
                              {insightText(strOf((hypothesis as any)?.evidence_reason))}
                            </p>
                          )}
                          {(() => {
                            const rivals: string[] = asList((hypothesis as any)?.rivals)
                              .map((r: any) => strOf(typeof r === "string" ? r : r?.text))
                              .filter(Boolean)
                              .slice(0, 2);
                            if (!rivals.length) return null;
                            return (
                              <div className="mt-2">
                                <p className="text-[11.5px] font-semibold text-foreground">Other possibilities checked</p>
                                <ul className="mt-1 space-y-1">
                                  {rivals.map((r, i) => (
                                    <li key={i} className="text-[11.5px] text-secondary-foreground leading-relaxed">· {insightText(r)}</li>
                                  ))}
                                </ul>
                              </div>
                            );
                          })()}
                          {strOf((hypothesis as any)?.data_needed) && (
                            <p className="mt-2 text-[11.5px] text-muted-foreground leading-relaxed">
                              <span className="font-semibold text-secondary-foreground">What would settle it: </span>
                              {insightText(strOf((hypothesis as any)?.data_needed))}
                            </p>
                          )}
                          <div className="mt-2 pt-2 border-t border-border/70 flex items-center gap-3 flex-wrap">
                            <button
                              onClick={() => testHypothesis()}
                              className="inline-flex items-center gap-1 text-[11.5px] font-semibold text-blue-700 dark:text-blue-400 hover:underline"
                              aria-label="Test this hypothesis in Explore">
                              <FlaskConical size={12} /> Test this
                            </button>
                          </div>
                        </div>
                      )}
                      {recommendations.length > 0 && (
                        <div className="mt-3">
                          <p className="text-[12px] font-semibold text-foreground">What to do next</p>
                          <ul className="mt-1.5 space-y-1.5">
                              {recommendations.slice(0, 3).map((r: any, i: number) => (
                                <li key={i} className="flex items-start gap-2 text-[12.5px] text-secondary-foreground leading-relaxed">
                                  <CheckCircle2 size={13} className="text-emerald-600 dark:text-emerald-400 flex-shrink-0 mt-0.5" />
                                  <span>{insightText(strOf(r?.text ?? r))}</span>
                                </li>
                              ))}
                          </ul>
                        </div>
                      )}
                    </Card>
                  )}

                  {/* Limitations */}
                  {limitations.length > 0 && (
                    <div className="flex items-start gap-1.5 text-[11px] text-muted-foreground bg-blue-50/60 dark:bg-blue-500/10 border border-blue-100 dark:border-blue-500/20 rounded-xl px-3 py-2">
                      <Info size={13} className="text-blue-600 dark:text-blue-400 flex-shrink-0 mt-px" />
                      <span>{limitations[0]}</span>
                    </div>
                  )}
                </div>
              )}

              {/* Follow-up input — always visible when results shown */}
              {showResults && !running && (
                <Card className="p-4">
                  <div className="relative">
                    <div className="flex items-center gap-2 rounded-xl border border-border bg-secondary/60 px-3 py-2.5 focus-within:border-blue-400 focus-within:bg-card transition-colors">
                      <Search size={15} className="text-muted-foreground flex-shrink-0" />
                      <input
                        value={followUpQuery}
                        onChange={(e) => { setFollowUpQuery(e.target.value); fetchAc(e.target.value); setAcOpen(true); }}
                        onFocus={() => setAcOpen(true)}
                        onBlur={() => setTimeout(() => setAcOpen(false), 200)}
                        onKeyDown={(e) => { if (e.key === "Enter") { setAcOpen(false); submitFollowUp(); } }}
                        placeholder="Ask a follow-up..."
                        className="flex-1 bg-transparent outline-none text-[13px] text-foreground placeholder:text-muted-foreground"
                        aria-label="Ask a follow-up question"
                      />
                      <button onClick={() => { setAcOpen(false); submitFollowUp(); }} className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-blue-700 text-white hover:bg-blue-800 transition-colors flex-shrink-0" aria-label="Send">
                        <Send size={14} />
                      </button>
                    </div>
                    {acOpen && acOptions.length > 0 && (
                      <div className="absolute top-full left-0 right-0 mt-1 bg-card border border-border rounded-xl shadow-lg py-1.5 z-30 max-h-[240px] overflow-y-auto">
                        {acOptions.map((s, i) => (
                          <button key={s + i} onMouseDown={() => { setAcOpen(false); setFollowUpQuery(s); runOnPage(s); }}
                            className="w-full flex items-center gap-3 px-4 py-2.5 text-[12px] text-foreground hover:bg-secondary transition-colors text-left">
                            <Zap size={13} className="text-blue-600 dark:text-blue-400 flex-shrink-0" />
                            <span className="truncate">{s}</span>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </Card>
              )}
            </div>
          )}

          {/* Tab bar — only when no results */}
          {!showResults && (
            <>
              <div className="flex items-center gap-1 border-b border-border" role="tablist" aria-label="Analysis sections">
                {([ { id: "recent" as const, label: "Recent" }, { id: "saved" as const, label: "Saved" }, { id: "collections" as const, label: "Collections" }, { id: "shared" as const, label: "Shared" } ]).map((t) => {
                  const on = homeTab === t.id;
                  return (
                    <button key={t.id} role="tab" aria-selected={on} onClick={() => setHomeTab(t.id)}
                      className={`px-3.5 py-2.5 text-[12.5px] font-medium border-b-2 -mb-px transition-colors whitespace-nowrap ${on ? "border-blue-700 text-blue-800 dark:text-blue-300" : "border-transparent text-muted-foreground hover:text-secondary-foreground"}`}>
                      {t.label}
                    </button>
                  );
                })}
              </div>

              {/* Tab content */}
              {(homeTab === "recent" || homeTab === "saved") ? (
                <div>
                  {recentError && homeTab === "recent" && <p className="text-[12px] text-rose-600 dark:text-rose-400 mb-3">{recentError}</p>}
                  {tabItems.length > 0 ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                      {tabItems.map((item: any, i: number) => (
                        <AnalysisCard key={item?.id || i} item={item} onOpen={() => openAnalysis(item)} />
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-12 bg-card border border-border rounded-2xl">
                      <BarChart3 size={24} className="mx-auto text-slate-300 dark:text-muted-foreground mb-2" />
                      <p className="text-[13px] font-medium text-muted-foreground">{homeTab === "recent" ? "No recent analyses yet." : "No saved analyses yet."}</p>
                      <button onClick={() => gotoTab("explore")} className="mt-3 inline-flex items-center gap-1.5 px-4 py-2 rounded-lg text-[12.5px] font-semibold text-white bg-blue-700 hover:bg-blue-800"><Plus size={13} /> New analysis</button>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-12 bg-card border border-border rounded-2xl">
                  <Compass size={24} className="mx-auto text-slate-300 dark:text-muted-foreground mb-2" />
                  <p className="text-[13px] font-medium text-muted-foreground">{homeTab === "collections" ? "No collections yet." : "No shared analyses yet."}</p>
                </div>
              )}

              {/* Saved Analyses section */}
              {homeTab === "recent" && asList(savedCols).length > 0 && (
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-[14px] font-semibold text-foreground">Saved Analyses</h3>
                    <button onClick={() => setHomeTab("saved")} className="text-[12px] font-medium text-blue-700 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 flex items-center gap-0.5">View all <ChevronRight size={12} /></button>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
                    {asList(savedCols).slice(0, 4).map((item: any, i: number) => (
                      <AnalysisCard key={item?.id || i} item={item} onOpen={() => openAnalysis(item)} />
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* ---- Right sidebar ---- */}
        <aside className="w-[300px] flex-shrink-0 hidden lg:block space-y-4">
          {/* 360 degree business view — side panel keeps the main column short */}
          {b360 && (
            <div className="bg-card border border-blue-200/80 dark:border-blue-500/25 rounded-2xl shadow-[0_1px_3px_rgba(0,0,0,0.04)] p-4 bg-gradient-to-b from-blue-50/60 dark:from-blue-500/10 to-card">
              <p className="text-[11px] font-bold tracking-[0.14em] text-blue-700 dark:text-blue-400">360 DEGREE BUSINESS VIEW</p>
              <h3 className="text-[14px] font-bold text-foreground tracking-tight mt-1 leading-snug">{strOf(b360?.headline || "Here's the full picture")}</h3>
              <p className="mt-1.5 text-[12px] text-secondary-foreground leading-relaxed">{strOf(b360?.what_happened?.summary || "")}</p>
              {asList(b360?.likely_causes).length > 0 && (
                <div className="mt-2.5">
                  <p className="text-[11px] font-semibold text-foreground">Most likely causes</p>
                  <ul className="mt-1 space-y-1">
                    {asList(b360.likely_causes).slice(0, 3).map((c: any, i: number) => (
                      <li key={i} className="text-[12px] text-secondary-foreground leading-relaxed">· <span className="font-semibold text-foreground">{strOf(c?.cause)}</span> — {strOf(c?.why_plain)}</li>
                    ))}
                  </ul>
                </div>
              )}
              {asList(b360?.what_to_do).length > 0 && (
                <div className="mt-2.5">
                  <p className="text-[11px] font-semibold text-foreground">What to do next</p>
                  <div className="mt-1 space-y-1.5">
                    {asList(b360.what_to_do).slice(0, 3).map((a: any, i: number) => (
                      <details key={i} className="rounded-xl bg-emerald-50/60 dark:bg-emerald-500/10 border border-emerald-200 dark:border-emerald-500/25 px-2.5 py-1.5">
                        <summary className="text-[12px] font-bold text-emerald-900 dark:text-emerald-300 cursor-pointer leading-snug">{strOf(a?.title)}</summary>
                        <ul className="mt-1 space-y-0.5">
                          {asList(a?.steps).slice(0, 3).map((s: string, j: number) => (
                            <li key={j} className="text-[11.5px] text-secondary-foreground leading-relaxed">✓ {strOf(s)}</li>
                          ))}
                        </ul>
                        {strOf(a?.expected) && <p className="mt-1 text-[11.5px] text-secondary-foreground">{strOf(a.expected)}</p>}
                      </details>
                    ))}
                  </div>
                </div>
              )}
              {strOf(b360?.if_do_nothing) && (
                <div className="mt-2.5 rounded-xl bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/25 px-2.5 py-2">
                  <p className="text-[11px] font-semibold text-amber-900 dark:text-amber-300">If you do nothing</p>
                  <p className="text-[11.5px] text-amber-800 dark:text-amber-400 leading-relaxed">{strOf(b360.if_do_nothing)}</p>
                </div>
              )}
            </div>
          )}
          {/* ARIA Assistant */}
          <div className="bg-card border border-border/80 rounded-2xl shadow-[0_1px_3px_rgba(0,0,0,0.04)] p-4">
            <div className="flex items-center gap-2.5 mb-4">
              <span className="inline-flex items-center justify-center w-9 h-9 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 text-white"><Sparkles size={16} /></span>
              <div>
                <h3 className="text-[13px] font-semibold text-foreground">ARIA</h3>
                <p className="text-[11px] text-muted-foreground">Your analysis assistant</p>
              </div>
            </div>
            <div className="space-y-0.5">
              {SIDEBAR_ACTIONS.map((a) => (
                <button key={a.label} onClick={() => handleSidebarAction(a)}
                  className="w-full text-left py-2.5 px-2 -mx-2 flex items-center gap-2.5 hover:bg-blue-50/50 dark:hover:bg-blue-500/10 rounded-lg transition-colors group">
                  <span className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-blue-50 dark:bg-blue-500/10 text-blue-700 dark:text-blue-400 flex-shrink-0"><a.icon size={14} /></span>
                  <span className="min-w-0 flex-1">
                    <span className="block text-[12px] font-medium text-foreground group-hover:text-blue-700 dark:group-hover:text-blue-400">{a.label}</span>
                    <span className="block text-[11px] text-muted-foreground">{a.sub}</span>
                  </span>
                  <ChevronRight size={13} className="text-slate-300 dark:text-muted-foreground group-hover:text-blue-500 dark:group-hover:text-blue-400 flex-shrink-0" />
                </button>
              ))}
            </div>
          </div>

          {/* Context */}
          {sidebarCtx.metric && (
            <div className="bg-card border border-border/80 rounded-2xl shadow-[0_1px_3px_rgba(0,0,0,0.04)] p-4">
              <h4 className="text-[11px] font-medium text-muted-foreground mb-2">Context</h4>
              <div className="flex items-center gap-2.5 p-2.5 bg-secondary rounded-xl">
                <span className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-blue-50 dark:bg-blue-500/10 text-blue-700 dark:text-blue-400 flex-shrink-0"><BarChart3 size={14} /></span>
                <div className="min-w-0">
                  <p className="text-[12px] font-medium text-foreground truncate">{humanize(sidebarCtx.metric)}</p>
                  <p className="text-[11px] text-muted-foreground">{sidebarCtx.breakdown ? `By ${humanize(sidebarCtx.breakdown)}` : "Overall"}</p>
                </div>
              </div>
            </div>
          )}

          {/* Continue exploring */}
          <div className="bg-card border border-border/80 rounded-2xl shadow-[0_1px_3px_rgba(0,0,0,0.04)] p-4">
            <h4 className="text-[11px] font-medium text-muted-foreground mb-2">Continue exploring</h4>
            <div className="space-y-0.5">
              {CONTINUE_EXPLORING.map((a) => (
                <button key={a.label} onClick={() => gotoTab(a.tab)}
                  className="w-full text-left py-2.5 px-2 -mx-2 flex items-center gap-2.5 hover:bg-blue-50/50 dark:hover:bg-blue-500/10 rounded-lg transition-colors group">
                  <span className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-secondary text-secondary-foreground flex-shrink-0"><a.icon size={14} /></span>
                  <span className="min-w-0 flex-1">
                    <span className="block text-[12px] font-medium text-foreground group-hover:text-blue-700 dark:group-hover:text-blue-400">{a.label}</span>
                    <span className="block text-[11px] text-muted-foreground">{a.sub}</span>
                  </span>
                  <ChevronRight size={13} className="text-slate-300 dark:text-muted-foreground group-hover:text-blue-500 dark:group-hover:text-blue-400 flex-shrink-0" />
                </button>
              ))}
            </div>
          </div>

          {/* Follow-up input */}
          <div className="bg-card border border-border/80 rounded-2xl shadow-[0_1px_3px_rgba(0,0,0,0.04)] p-4">
            <div className="flex items-center gap-2 rounded-xl border border-border bg-secondary/60 px-3 py-2 focus-within:border-blue-400 transition-colors">
              <input
                value={followUpQuery}
                onChange={(e) => setFollowUpQuery(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") submitFollowUp(); }}
                placeholder="Ask a follow-up..."
                className="flex-1 bg-transparent outline-none text-[12px] text-foreground placeholder:text-muted-foreground"
                aria-label="Ask a follow-up question"
              />
              <button onClick={submitFollowUp} className="inline-flex items-center justify-center w-7 h-7 rounded-lg bg-blue-700 text-white hover:bg-blue-800 transition-colors flex-shrink-0" aria-label="Send">
                <Send size={12} />
              </button>
            </div>
            <p className="text-[10.5px] text-muted-foreground mt-2 leading-relaxed">ARIA can help you analyze, compare, and go deeper.</p>
          </div>
        </aside>
      </div>
    </div>
  );
}
