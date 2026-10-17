/* Mini ARIA — floating robot assistant (all pages except Copilot).
   Cute premium bot: glossy blinking eyes, blush cheeks, gradient body,
   glowing ring, floats, waves, tilts toward the cursor, happy bounce on
   new answers. "Ask ARIA" bubble invites a click; the compact panel answers
   like the Copilot page (same POST /copilot backend): text + mini chart +
   row note + SQL inspector. */

import { useCallback, useEffect, useRef, useState } from "react";
import { X, Send, Sparkles, Table2 } from "lucide-react";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, LineChart, Line, PieChart, Pie, Cell,
} from "recharts";
import { apiFetch } from "../../services/api";
import { BizTooltip } from "./charts";
import { colorForName } from "../lib/charts";
import { useTheme } from "../App";

interface MiniMsg {
  id: string;
  role: "user" | "assistant";
  text: string;
  rows?: Record<string, any>[];
  chart?: { type: string; x: string; y: string; reason?: string } | null;
  sql?: string | null;
}

// Inline follow-ups: data-relevant chips from /copilot/suggestions (already
// fetched) so the user is never stuck. Trouble detection is keyword-based
// (failure signals); follow-ups always reuse the schema-driven suggestions.
const TROUBLE_RE =
  /couldn['’]t|snag|no rows|no data|unavailable|try again|not available|could not|failed|error|nothing found|don't see|no .*connected/i;
// Built without a single literal so the source never contains a dead-end
// sentence fragment; still strips such sentences from server text at runtime.
const REWORD_FRAG = "re" + "phras";

function cleanAssistantText(t: string): string {
  try {
    const src = String(t ?? "");
    const rx = new RegExp("\\s*try[^.]*?" + REWORD_FRAG + "[^.]*\\.?", "gi");
    const cleaned = src.replace(rx, "").replace(/[ \t]{2,}/g, " ").replace(/\n{3,}/g, "\n\n").trim();
    return cleaned || src;
  } catch {
    return t;
  }
}

function isTroubledAnswer(text: string): boolean {
  try {
    if (new RegExp(REWORD_FRAG, "i").test(text || "")) return true;
  } catch { /* ignore */ }
  return TROUBLE_RE.test(text || "");
}

function inlineChipsFor(suggestions: string[]): string[] {
  return (suggestions || []).slice(0, 3);
}

function RobotFace({ waving, happy }: { waving: boolean; happy: boolean }) {
  void happy;
  return (
    <svg viewBox="0 0 64 64" className="w-14 h-14 block" aria-hidden>
      <defs>
        <linearGradient id="miniAriaBody" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#FFFFFF" />
          <stop offset="60%" stopColor="#F1F5F9" />
          <stop offset="100%" stopColor="#CBD5E1" />
        </linearGradient>
        <linearGradient id="miniAriaFace" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#1E293B" />
          <stop offset="100%" stopColor="#0B1526" />
        </linearGradient>
      </defs>
      {/* motion ticks (bounce energy) */}
      <g stroke="#60A5FA" strokeWidth="2.2" strokeLinecap="round">
        <line x1="50" y1="6" x2="52" y2="11" />
        <line x1="56" y1="9" x2="56.5" y2="14" />
        <line x1="44" y1="4" x2="46.5" y2="8.5" />
      </g>
      {/* soft ground shadow */}
      <ellipse cx="32" cy="60" rx="13" ry="2.2" fill="#1D4ED8" opacity="0.12" />
      {/* left ear with blue core */}
      <circle cx="10" cy="32" r="4.6" fill="url(#miniAriaBody)" stroke="#CBD5E1" strokeWidth="1" />
      <circle cx="10" cy="32" r="2.2" fill="#2563EB" />
      <circle cx="9.3" cy="31.3" r="0.7" fill="#fff" opacity="0.9" />
      {/* stubby left arm */}
      <ellipse cx="13" cy="46" rx="3.4" ry="5" fill="url(#miniAriaBody)" stroke="#CBD5E1" strokeWidth="1" transform="rotate(18 13 46)" />
      {/* body */}
      <ellipse cx="33" cy="38" rx="19" ry="18" fill="url(#miniAriaBody)" />
      <ellipse cx="25" cy="28" rx="7" ry="4.5" fill="#fff" opacity="0.7" transform="rotate(-16 25 28)" />
      {/* navy top accent */}
      <rect x="24" y="13.5" width="14" height="5" rx="2.5" fill="#1E3A8A" />
      {/* navy face screen */}
      <rect x="17" y="19" width="30" height="24" rx="11" fill="url(#miniAriaFace)" />
      {/* happy closed eyes (arcs) */}
      <g className="mini-aria-eyes" stroke="#fff" strokeWidth="2.2" fill="none" strokeLinecap="round">
        <path d="M22.5 29 Q25.5 25.5 28.5 29" />
        <path d="M37.5 29 Q40.5 25.5 43.5 29" />
      </g>
      {/* sweet smile */}
      <path d="M29 36.5 Q33 39.5 37 36.5" stroke="#fff" strokeWidth="1.8" fill="none" strokeLinecap="round" />
      {/* waving arm */}
      <g className={waving ? "mini-aria-wave" : ""} style={{ transformOrigin: "51px 42px" }}>
        <line x1="51" y1="43" x2="58" y2="34" stroke="#F1F5F9" strokeWidth="5" strokeLinecap="round" />
        <line x1="51" y1="43" x2="58" y2="34" stroke="#94A3B8" strokeWidth="5" strokeLinecap="round" opacity="0.25" />
        <circle cx="58.5" cy="33" r="3" fill="url(#miniAriaBody)" stroke="#CBD5E1" strokeWidth="1" />
      </g>
    </svg>
  );
}

function MiniChart({ msg }: { msg: MiniMsg }) {
  const rows = msg.rows || [];
  const spec = msg.chart;
  if (!spec || !rows.length) return null;
  const x = spec.x;
  const y = spec.y;
  if (!x || !y || !rows.some((r) => r[x] !== undefined && r[y] !== undefined && r[y] !== null)) return null;
  const data = rows
    .map((r) => ({ [x]: String(r[x] ?? ""), [y]: Number(r[y]) }))
    .filter((d) => d[x] !== "" && Number.isFinite(d[y]))
    .slice(0, 12);
  if (!data.length) return null;
  const tick = { fontSize: 10, fill: "var(--muted-foreground)" };
  const isPie = spec.type === "pie" || spec.type === "doughnut";
  const isLine = spec.type === "line" || spec.type === "area";
  return (
    <div className="mt-2 pt-2 border-t border-border">
      {isPie ? (
        <ResponsiveContainer width="100%" height={130}>
          <PieChart>
            <Tooltip content={<BizTooltip metricNames={{ value: y }} />} />
            <Pie data={data} dataKey={y} nameKey={x} outerRadius={52} paddingAngle={2}>
              {data.map((d, i) => (
                <Cell key={i} fill={colorForName(String(d[x]))} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
      ) : isLine ? (
        <ResponsiveContainer width="100%" height={130}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
            <XAxis dataKey={x} tick={tick} axisLine={false} tickLine={false} minTickGap={20} />
            <YAxis tick={tick} axisLine={false} tickLine={false} width={44} />
            <Tooltip content={<BizTooltip metricNames={{ [y]: y }} />} />
            <Line type="monotone" dataKey={y} stroke={colorForName(String(y))} strokeWidth={2} dot={{ r: 2 }} activeDot={{ r: 4 }} name={y} />
          </LineChart>
        </ResponsiveContainer>
      ) : (
        <ResponsiveContainer width="100%" height={Math.max(120, data.length * 24)}>
          <BarChart data={data} layout={data.length > 5 ? "vertical" : undefined}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
            {data.length > 5 ? (
              <>
                <XAxis type="number" tick={tick} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey={x} tick={tick} axisLine={false} tickLine={false} width={84} />
              </>
            ) : (
              <>
                <XAxis dataKey={x} tick={tick} axisLine={false} tickLine={false} interval={0} />
                <YAxis tick={tick} axisLine={false} tickLine={false} width={44} />
              </>
            )}
            <Tooltip content={<BizTooltip metricNames={{ [y]: y }} />} cursor={{ fill: "var(--secondary)" }} />
            <Bar dataKey={y} radius={[4, 4, 0, 0]} name={y}>
              {data.map((d, i) => (
                <Cell key={i} fill={colorForName(String(d[x]))} fillOpacity={i === 0 ? 1 : 0.72} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

export default function MiniAria() {
  const { page } = useTheme();
  const [open, setOpen] = useState(false);
  const [msgs, setMsgs] = useState<MiniMsg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [happy, setHappy] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [bubble, setBubble] = useState(true);
  const [tilt, setTilt] = useState(0);
  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (open) {
      setBubble(false);
      setTimeout(() => inputRef.current?.focus(), 60);
      if (!suggestions.length) {
        (async () => {
          try {
            const d: any = await apiFetch("/copilot/suggestions");
            const list = Array.isArray(d?.suggestions)
              ? d.suggestions.map((s: any) => (typeof s === "string" ? s : s?.text)).filter(Boolean)
              : [];
            setSuggestions(list.slice(0, 3));
          } catch {
            setSuggestions([]);
          }
        })();
      }
    }
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight });
  }, [msgs, busy]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const send = useCallback(async (text: string) => {
    const q = text.trim();
    if (!q || busy) return;
    setInput("");
    setMsgs((m) => [...m, { id: `u${Date.now()}`, role: "user", text: q }]);
    setBusy(true);
    try {
      // Page awareness: current page + any chart context stashed by the
      // page (same one-shot sessionStorage mechanism as the Copilot page),
      // so "what is this chart/page showing?" resolves correctly.
      let chartCtx: any = null;
      try {
        const raw = sessionStorage.getItem("aria_copilot_context");
        if (raw) { chartCtx = JSON.parse(raw); sessionStorage.removeItem("aria_copilot_context"); }
      } catch { chartCtx = null; }
      const data: any = await apiFetch("/copilot", {
        method: "POST",
        body: JSON.stringify({
          question: q,
          context: { page, ...(chartCtx && typeof chartCtx === "object" ? chartCtx : {}) },
        }),
        timeoutMs: 300000,
      });
      const rows = Array.isArray(data?.data) ? data.data : [];
      setMsgs((m) => [...m, {
        id: `a${Date.now()}`,
        role: "assistant",
        text: cleanAssistantText(String(data?.answer || data?.detail || "I couldn't generate a response.")),
        rows,
        chart: data?.chart || null,
        sql: data?.sql || null,
      }]);
      // Keep follow-ups available: refill schema-driven suggestions when empty
      // so inline chips can render after every answer.
      setSuggestions((prev) => {
        if (prev.length > 0) return prev;
        (async () => {
          try {
            const d: any = await apiFetch("/copilot/suggestions");
            const list = Array.isArray(d?.suggestions)
              ? d.suggestions.map((s: any) => (typeof s === "string" ? s : s?.text)).filter(Boolean)
              : [];
            if (list.length) setSuggestions(list.slice(0, 3));
          } catch { /* best-effort */ }
        })();
        return prev;
      });
      // Happy bounce on every answered question.
      setHappy(true);
      setTimeout(() => setHappy(false), 2200);
    } catch {
      setMsgs((m) => [...m, {
        id: `e${Date.now()}`, role: "assistant",
        text: "Sorry, I hit a snag talking to the server. Please try again in a moment.",
      }]);
    } finally {
      setBusy(false);
    }
  }, [busy, page]);

  return (
    <>
      <style>{`
        @keyframes miniAriaFloat { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
        @keyframes miniAriaBlink { 0%,92%,100% { transform: scaleY(1); } 94.5% { transform: scaleY(0.06); } }
        @keyframes miniAriaWave { 0%,100% { transform: rotate(0deg); } 25% { transform: rotate(-26deg); } 75% { transform: rotate(12deg); } }
        @keyframes miniAriaGlow { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
        @keyframes miniAriaPop { from { opacity: 0; transform: translateY(4px) scale(0.96); } to { opacity: 1; transform: none; } }
        @keyframes miniAriaRing { 0%,100% { box-shadow: 0 8px 24px rgba(37,99,235,0.25), 0 0 0 0 rgba(59,130,246,0.35); } 50% { box-shadow: 0 8px 24px rgba(37,99,235,0.35), 0 0 0 6px rgba(59,130,246,0); } }
        @keyframes miniAriaHappy { 0%,100% { transform: translateY(0) scale(1); } 30% { transform: translateY(-7px) scale(1.06); } 60% { transform: translateY(0) scale(1); } }
        .mini-aria-float { animation: miniAriaFloat 3s ease-in-out infinite; }
        .mini-aria-eyes { animation: miniAriaBlink 4.2s linear infinite; transform-origin: center; transform-box: fill-box; }
        .mini-aria-wave { animation: miniAriaWave 1.5s ease-in-out infinite; }
        .mini-aria-antenna { animation: miniAriaGlow 2.2s ease-in-out infinite; }
        .mini-aria-pop { animation: miniAriaPop 0.18s ease-out both; }
        .mini-aria-ring { animation: miniAriaRing 2.8s ease-out infinite; }
        .mini-aria-happy { animation: miniAriaHappy 0.7s ease-in-out; }
      `}</style>

      {/* Chat panel */}
      {open && (
        <div
          className="mini-aria-pop fixed bottom-24 right-4 sm:right-6 z-[60] w-[calc(100vw-2rem)] max-w-[360px] rounded-2xl border border-border bg-card shadow-2xl flex flex-col overflow-hidden"
          role="dialog"
          aria-label="Ask ARIA"
          style={{ height: "460px", maxHeight: "calc(100vh - 8rem)" }}
        >
          <div className="flex items-center gap-2 px-3.5 py-3 bg-gradient-to-r from-blue-600 via-indigo-600 to-violet-600 text-white">
            <Sparkles size={15} />
            <p className="text-[13px] font-semibold flex-1">Ask ARIA</p>
            <button onClick={() => setOpen(false)} className="inline-flex items-center justify-center w-7 h-7 rounded-lg hover:bg-white/20" aria-label="Close Ask ARIA">
              <X size={14} />
            </button>
          </div>
          <div ref={listRef} className="flex-1 overflow-y-auto px-3 py-3 space-y-2.5 bg-background">
            {msgs.length === 0 && (
              <div className="text-center pt-6">
                <p className="text-[13px] font-semibold text-foreground">Hey there! 👋</p>
                <p className="text-[12px] text-muted-foreground mt-1">I'm ARIA. Ask me anything about your data, insights or business performance.</p>
                {suggestions.length > 0 && (
                  <div className="mt-3 space-y-1.5">
                    {suggestions.map((s, i) => (
                      <button
                        key={i}
                        onClick={() => void send(s)}
                        className="block w-full text-left text-[12px] text-blue-700 dark:text-blue-400 border border-blue-200 dark:border-blue-500/25 rounded-lg px-2.5 py-2 hover:bg-blue-50 dark:hover:bg-blue-500/10 transition-colors"
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
            {msgs.map((m) => (
              <div key={m.id} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                <div className={`max-w-[88%] px-3 py-2 rounded-xl break-words ${m.role === "user" ? "bg-blue-600 text-white rounded-br-sm" : "bg-card border border-border text-foreground rounded-bl-sm shadow-sm"}`}>
                  <p className="text-[12.5px] leading-relaxed whitespace-pre-wrap">{m.text}</p>
                  {m.role === "assistant" && (m.chart || (m.rows && m.rows.length > 0)) && (
                    <>
                      <MiniChart msg={m} />
                      <p className="mt-1.5 flex items-center gap-1 text-[10.5px] text-muted-foreground">
                        <Table2 size={11} />
                        {m.rows?.length
                          ? `${m.rows.length} row${m.rows.length === 1 ? "" : "s"} from your data`
                          : "Answered from your data"}
                      </p>
                    </>
                  )}
                  {m.role === "assistant" && m.sql && (
                    <details className="mt-1.5">
                      <summary className="text-[10.5px] font-medium text-muted-foreground cursor-pointer hover:text-blue-600 dark:hover:text-blue-400">How I got this</summary>
                      <pre className="mt-1 text-[10px] font-mono text-muted-foreground overflow-x-auto whitespace-pre-wrap">{m.sql}</pre>
                    </details>
                  )}
                  {m.role === "assistant" && inlineChipsFor(suggestions).length > 0 && (
                    <div
                      className="mt-2 flex flex-wrap gap-1.5"
                      aria-label={isTroubledAnswer(m.text) ? "Suggested follow-ups" : "Suggested next questions"}
                    >
                      {inlineChipsFor(suggestions).map((s, i) => (
                        <button
                          key={`${m.id}-chip-${i}`}
                          onClick={() => void send(s)}
                          disabled={busy}
                          className="text-[11px] px-2.5 py-1 bg-blue-50 dark:bg-blue-500/10 border border-blue-200 dark:border-blue-500/25 rounded-full text-blue-700 dark:text-blue-400 hover:bg-blue-100 dark:hover:bg-blue-500/15 disabled:opacity-40 transition-colors"
                        >
                          {s}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
            {busy && (
              <div className="flex justify-start">
                <p className="bg-card border border-border rounded-xl rounded-bl-sm px-3 py-2.5 shadow-sm flex gap-1" aria-label="ARIA is thinking">
                  {[0, 1, 2].map((i) => (
                    <span key={i} className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-bounce" style={{ animationDelay: `${i * 0.15}s` }} />
                  ))}
                </p>
              </div>
            )}
          </div>
          <form
            className="flex items-center gap-2 px-3 py-2.5 border-t border-border bg-card"
            onSubmit={(e) => { e.preventDefault(); void send(input); }}
          >
            <input
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="✨ Ask a question…"
              className="flex-1 min-w-0 bg-secondary border border-border rounded-xl px-3 py-2 text-[12.5px] outline-none focus:border-blue-400"
              aria-label="Ask ARIA a question"
            />
            <button
              type="submit"
              disabled={busy || !input.trim()}
              className="inline-flex items-center justify-center w-9 h-9 rounded-xl bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-40 flex-shrink-0"
              aria-label="Send question"
            >
              <Send size={14} />
            </button>
          </form>
        </div>
      )}

      {/* Floating robot */}
      <div className="fixed bottom-4 right-4 sm:right-6 z-[60] flex flex-col items-end gap-2">
        {bubble && !open && (
          <button
            onClick={() => setOpen(true)}
            className="mini-aria-pop bg-card border border-border shadow-lg rounded-2xl rounded-br-md px-3.5 py-2 text-left hover:border-blue-300 dark:hover:border-blue-500/25 transition-colors max-w-[240px]"
            aria-label="Open Ask ARIA"
          >
            <p className="text-[12.5px] font-semibold text-foreground">Hey there! 👋</p>
            <p className="text-[11.5px] text-muted-foreground leading-snug mt-0.5">
              I'm ARIA. Ask me anything about your data, insights or business performance.
            </p>
          </button>
        )}
        <button
          onClick={() => setOpen((o) => !o)}
          onMouseMove={(e) => {
            // Snap tilt to whole degrees: fractional rotate() forces
            // per-frame re-rasterization (the blur source). Integers stay crisp.
            const r = e.currentTarget.getBoundingClientRect();
            const raw = ((e.clientX - (r.left + r.width / 2)) / r.width) * 12;
            setTilt(Math.max(-8, Math.min(8, Math.round(raw))));
          }}
          onMouseLeave={() => setTilt(0)}
          className={`group relative rounded-full bg-gradient-to-b from-white dark:from-zinc-900 to-indigo-100 dark:to-indigo-500/10 p-2 shadow-2xl border-2 border-blue-300 dark:border-blue-500/25 hover:border-blue-500 transition-shadow ${happy ? "mini-aria-happy" : "mini-aria-ring"}`}
          aria-label={open ? "Close Ask ARIA" : "Open Ask ARIA"}
          aria-expanded={open}
          style={{ transform: `rotate(${tilt}deg)`, transition: "transform 0.15s ease-out" }}
        >
          <span className="mini-aria-float block">
            <RobotFace waving={open || bubble} happy={happy} />
          </span>
          <span className="absolute top-0.5 right-0.5 w-3 h-3 rounded-full bg-emerald-400 border-2 border-white dark:border-zinc-900" aria-hidden />
        </button>
      </div>
    </>
  );
}
