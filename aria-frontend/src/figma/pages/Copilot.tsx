import { useState, useRef, useEffect, useMemo } from "react";
import { Send, Plus, MoreHorizontal, Sparkles } from "lucide-react";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, LineChart, Line, PieChart, Pie, Cell } from "recharts";
import { useTheme } from "../App";
import Evidence from "../components/Evidence";
import AIProcessingIndicator from "../components/AIProcessingIndicator";
import { BizTooltip } from "../components/charts";
import { colorForName } from "../lib/charts";
import { fmtBiz } from "../lib/format";
import { apiFetch } from "../../services/api";
import { KEYS, storage } from "../../utils/storage";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  sql?: string | null;
  provider?: string | null;
  providerFailed?: boolean;
  rows?: Record<string, any>[];
  chart?: { type: string; x: string; y: string; reason?: string } | null;
};

/** Provider/LLM failure signals in a copilot payload or error text. */
function isProviderFailure(data: any, errText = ""): boolean {
  const hay = `${data?.error || ""} ${data?.detail || ""} ${data?.answer || ""} ${errText}`;
  return /provider|llm|model|groq|ollama|deepseek|omniroute|\b403\b|\b404\b|access denied|could not reach|timed out/i.test(hay);
}

interface Conversation {
  id: string;
  title: string;
  subtitle: string;
  date: string;
  messages: Message[];
}

const FALLBACK_SUGGESTIONS = [
  "What are the main metrics in this dataset?",
  "Show me a summary of the data",
  "Which categories have the most records?",
  "What trends do you see in the data?",
];

// Render the chart-ready payload returned by the backend as a fully
// interactive chart (hover tooltips, axis formatting). The type is chosen
// dynamically from the question + returned data (bar/line/pie/scatter).
function renderChart(msg: Message, isDark?: boolean) {
  const rows = msg.rows || [];
  const spec = msg.chart;
  if (!spec || !rows.length) return null;
  const x = spec.x;
  const y = spec.y;
  // Guard shape drift: if the spec keys don't exist in the rows, render
  // nothing instead of a flat zero chart. Never invent zeros.
  if (!x || !y || !rows.some((r) => r[x] !== undefined && r[y] !== undefined && r[y] !== null)) return null;
  const data = rows
    .map((r) => ({ [x]: String(r[x] ?? ""), [y]: Number(r[y]) }))
    .filter((d) => d[x] !== "" && Number.isFinite(d[y]))
    .slice(0, 24);
  if (!data.length) return null;
  const grid = isDark ? "#2A2A35" : "#E5E7EB";
  const axis = isDark ? "#8889A0" : "#9CA3AF";
  const tick = { fontSize: 11, fill: axis };
  const isPie = spec.type === "pie" || spec.type === "doughnut";
  const isLine = spec.type === "line" || spec.type === "area";
  const horizontal = spec.type === "bar_horizontal" || (!isLine && !isPie && data.length > 8);
  return (
    <div className="mt-2 pt-2 border-t border-border">
      {isPie ? (
        <ResponsiveContainer width="100%" height={180}>
          <PieChart>
            <Tooltip content={<BizTooltip metricNames={{ value: y }} />} />
            <Pie data={data} dataKey={y} nameKey={x} outerRadius={70} paddingAngle={2}>
              {data.map((d, i) => (
                <Cell key={i} fill={colorForName(String(d[x]))} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
      ) : isLine ? (
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke={grid} vertical={false} />
            <XAxis dataKey={x} tick={tick} axisLine={false} tickLine={false} minTickGap={24} />
            <YAxis tick={tick} axisLine={false} tickLine={false} width={52} tickFormatter={(v: number) => fmtBiz(v)} />
            <Tooltip content={<BizTooltip metricNames={{ [y]: y }} />} />
            <Line type="monotone" dataKey={y} stroke={colorForName(String(y))} strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} name={y} />
          </LineChart>
        </ResponsiveContainer>
      ) : horizontal ? (
        <ResponsiveContainer width="100%" height={Math.max(160, data.length * 32)}>
          <BarChart data={data} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke={grid} horizontal={false} />
            <XAxis type="number" tick={tick} axisLine={false} tickLine={false} tickFormatter={(v: number) => fmtBiz(v)} />
            <YAxis type="category" dataKey={x} tick={tick} axisLine={false} tickLine={false} width={110} />
            <Tooltip content={<BizTooltip metricNames={{ [y]: y }} />} cursor={{ fill: "var(--secondary)" }} />
            <Bar dataKey={y} radius={[0, 5, 5, 0]} name={y}>
              {data.map((d, i) => (
                <Cell key={i} fill={colorForName(String(d[x]))} fillOpacity={i === 0 ? 1 : 0.7} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      ) : (
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke={grid} vertical={false} />
            <XAxis dataKey={x} tick={tick} axisLine={false} tickLine={false} interval={data.length > 12 ? Math.floor(data.length / 12) : 0} />
            <YAxis tick={tick} axisLine={false} tickLine={false} width={52} tickFormatter={(v: number) => fmtBiz(v)} />
            <Tooltip content={<BizTooltip metricNames={{ [y]: y }} />} cursor={{ fill: "var(--secondary)" }} />
            <Bar dataKey={y} radius={[5, 5, 0, 0]} name={y}>
              {data.map((d, i) => (
                <Cell key={i} fill={colorForName(String(d[x]))} fillOpacity={i === 0 ? 1 : 0.7} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
      <p className="text-[10px] text-muted-foreground mt-1.5">{spec.reason || spec.type}</p>
    </div>
  );
}

export default function Copilot({ initialQuestion = "", analysisResult = null, goalResult = null }: { initialQuestion?: string; analysisResult?: any; goalResult?: any }) {
  void analysisResult; void goalResult;
  const { isDark } = useTheme();
  // Data-driven starter questions (4, from the connected schema).
  // Refreshed on mount and whenever a new chat starts (source may change).
  const [suggestions, setSuggestions] = useState<string[]>(FALLBACK_SUGGESTIONS);
  const refreshSuggestions = async () => {
    try {
      const data: any = await apiFetch("/copilot/suggestions");
      if (Array.isArray(data.suggestions) && data.suggestions.length) {
        setSuggestions(data.suggestions.slice(0, 4));
      }
    } catch { /* fallback list stays */ }
  };
  useEffect(() => { refreshSuggestions(); }, []);
  // Per-user persistence (§13): conversations survive refresh. Rows are
  // trimmed on save so one chat cannot exhaust localStorage quota.
  // The key + initial chats resolve once (useMemo/useState initializer —
  // never parsed inline during render).
  const chatsKey = useMemo(() => {
    const u: any = storage.get(KEYS.authUser, null);
    return storage.scoped(KEYS.copilotChats, u);
  }, []);
  const loadChats = (): Conversation[] => {
    const parsed = storage.get(chatsKey, []);
    return Array.isArray(parsed) ? (parsed as Conversation[]).slice(0, 10) : [];
  };
  const [conversations, setConversations] = useState<Conversation[]>(loadChats);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  // Lifecycle-driven AI stage (real request milestones, never timers):
  // 0 = request sent · 1 = response headers arrived · 2 = payload parsed,
  // rendering the answer. Unmounted when isTyping flips false.
  const [copilotStage, setCopilotStage] = useState(0);
  const endRef = useRef<HTMLDivElement>(null);
  const consumedPrefill = useRef("");

  const active = conversations.find((c) => c.id === activeId) || conversations[0] || null;

  // Persist on every change (trimmed: 10 chats, 30 messages each, 24 rows).
  // Versioned key + quota-guarded write; overflow stays session-only.
  useEffect(() => {
    const trimmed = conversations.slice(0, 10).map((c) => ({
      ...c,
      messages: c.messages.slice(-30).map((m) => ({
        ...m,
        rows: Array.isArray(m.rows) ? m.rows.slice(0, 24) : m.rows,
      })),
    }));
    storage.set(chatsKey, trimmed);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversations]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [active?.messages.length, isTyping]);

  // Handoff from other pages ("Ask ARIA" buttons): prefill once per
  // distinct question and send it, so the conversation continues here.
  useEffect(() => {
    const q = (initialQuestion || "").trim();
    if (q && consumedPrefill.current !== q) {
      consumedPrefill.current = q;
      setInput(q);
      send(q);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialQuestion]);

  const send = async (text?: string) => {
    const content = text || input;
    if (!content.trim()) return;
    setInput("");

    let targetId = activeId;
    if (!targetId || !conversations.find(c => c.id === targetId)) {
      const id = `c${Date.now()}`;
      const newConv: Conversation = {
        id,
        title: content.length > 30 ? content.substring(0, 30) + "..." : content,
        subtitle: "Just now",
        date: "Today",
        messages: [],
      };
      setConversations(prev => [newConv, ...prev]);
      targetId = id;
      setActiveId(id);
    }

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setConversations((prev) =>
      prev.map((c) => c.id === targetId ? { ...c, messages: [...c.messages, userMsg] } : c)
    );
    setIsTyping(true);
    setCopilotStage(0);

    try {
      // Chart context stashed by Decision Center / other pages (§38): measured
      // evidence travels with the question so Copilot answers about THIS chart.
      let chartCtx: any = null;
      try {
        const raw = sessionStorage.getItem("aria_copilot_context");
        if (raw) { chartCtx = JSON.parse(raw); sessionStorage.removeItem("aria_copilot_context"); }
      } catch { chartCtx = null; }
      const data: any = await apiFetch("/copilot", {
        method: "POST",
        body: JSON.stringify(chartCtx ? { question: content, context: chartCtx } : { question: content }),
        // Cold local models can take minutes on first load — don't cut them off.
        timeoutMs: 300000,
      });
      // Response headers arrived — analysis running server-side.
      setCopilotStage(1);
      // Payload parsed — building the answer rendering.
      setCopilotStage(2);
      const reply: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.answer || data.detail || "I couldn't generate a response.",
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        sql: data.sql || null,
        provider: data.provider || null,
        providerFailed: !data.ok && isProviderFailure(data),
        rows: Array.isArray(data.data) ? data.data : [],
        chart: data.chart || null,
      };
      setConversations((prev) =>
        prev.map((c) => c.id === targetId ? { ...c, messages: [...c.messages, reply] } : c)
      );
    } catch {
      const reply: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "Sorry, I encountered an error. Please try again.",
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      };
      setConversations((prev) =>
        prev.map((c) => c.id === targetId ? { ...c, messages: [...c.messages, reply] } : c)
      );
    } finally {
      setIsTyping(false);
    }
  };

  const newChat = () => {
    const id = `c${Date.now()}`;
    setConversations((prev) => [{ id, title: "New conversation", subtitle: "Just now", date: "Today", messages: [] }, ...prev]);
    setActiveId(id);
    setMenuOpen(false);
    refreshSuggestions();
  };

  const deleteActiveChat = () => {
    if (!activeId) return;
    setConversations((prev) => prev.filter((c) => c.id !== activeId));
    setActiveId(null);
    setMenuOpen(false);
  };

  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <div className="flex h-full">
      {/* Conversation list */}
      <div className="w-64 border-r border-border bg-card/80 backdrop-blur-sm flex flex-col flex-shrink-0">
        <div className="px-3 py-3 border-b border-border">
          <div className="flex items-center justify-between mb-2">
            <p className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">Conversations</p>
            <button onClick={newChat} className="premium-btn p-1 rounded-md hover:bg-secondary text-muted-foreground hover:text-foreground transition-all duration-150">
              <Plus size={13} />
            </button>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto custom-scroll">
          {conversations.map((conv) => (
            <button key={conv.id} onClick={() => setActiveId(conv.id)}
              className={`w-full text-left px-3 py-3 border-b border-border/50 transition-all duration-150 ${conv.id === activeId ? "bg-primary/8 border-l-2 border-l-primary" : "hover:bg-secondary/50 border-l-2 border-l-transparent"}`}>
              <p className="text-sm font-medium text-foreground truncate">{conv.title}</p>
              <p className="text-[11px] text-muted-foreground truncate mt-0.5">{conv.subtitle}</p>
              <p className="text-[10px] text-muted-foreground/60 mt-0.5">{conv.date}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {active ? (
          <>
            <div className="px-5 py-3 border-b border-border flex items-center justify-between flex-shrink-0">
              <div>
                <p className="text-sm font-semibold text-foreground">{active.title}</p>
                <p className="text-[11px] text-muted-foreground">
                  {active.subtitle}
                  {(() => {
                    const lastAssistant = [...active.messages].reverse().find(m => m.role === "assistant" && m.provider);
                    return lastAssistant?.provider ? ` · ${lastAssistant.provider} mode` : "";
                  })()}
                </p>
              </div>
              <div className="relative">
                <button
                  onClick={() => setMenuOpen((o) => !o)}
                  aria-label="Conversation options"
                  className="p-1.5 rounded-lg hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
                >
                  <MoreHorizontal size={14} />
                </button>
                {menuOpen && (
                  <div className="absolute right-0 top-9 w-44 bg-card border border-border rounded-xl shadow-lg py-1 z-50">
                    <button
                      onClick={newChat}
                      className="w-full flex items-center gap-2.5 px-3 py-2 text-xs text-foreground hover:bg-secondary transition-colors"
                    >
                      <Plus size={13} /> New chat
                    </button>
                    <button
                      onClick={deleteActiveChat}
                      className="w-full flex items-center gap-2.5 px-3 py-2 text-xs text-negative hover:bg-secondary transition-colors"
                    >
                      Delete conversation
                    </button>
                  </div>
                )}
              </div>
            </div>

            <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4 custom-scroll">
              {active.messages.length === 0 && !isTyping && (
                <div className="flex flex-col items-center justify-center h-full text-center animate-fade-in">
                  <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center mb-3">
                    <Sparkles size={18} className="text-primary" />
                  </div>
                  <p className="text-sm font-medium text-foreground">ARIA Copilot</p>
                  <p className="text-xs text-muted-foreground mt-1">Ask me anything about your data</p>
                </div>
              )}

              {active.messages.map((msg) =>
                msg.role === "assistant" ? (
                  <div key={msg.id} className="flex items-start gap-2.5 max-w-2xl animate-slide-in-left">
                    <div className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center mt-0.5">
                      <Sparkles size={11} className="text-primary" />
                    </div>
                    <div className="flex-1 space-y-1.5">
                      <div className="bg-card/90 border border-border/80 rounded-2xl rounded-tl-md px-4 py-3 shadow-sm backdrop-blur-sm">
                        <p className="text-[13px] text-foreground leading-relaxed">{msg.content}</p>
                        {(msg as any).providerFailed && (
                          <button
                            onClick={async () => {
                              try {
                                await apiFetch("/provider/switch", {
                                  method: "POST",
                                  body: JSON.stringify({ provider: "local" }),
                                });
                              } catch { /* best-effort; user can retry regardless */ }
                            }}
                            className="mt-2 inline-flex items-center gap-1.5 text-[12px] font-semibold text-blue-700 dark:text-blue-400 border border-blue-200 dark:border-blue-500/25 rounded-lg px-3 py-1.5 hover:bg-blue-50 dark:hover:bg-blue-500/10"
                          >
                            Switch to on-device AI and retry your question
                          </button>
                        )}
                        {renderChart(msg, isDark)}
                        {msg.sql && (
                          <div className="mt-2">
                            <Evidence
                              label="Evidence"
                              sections={[
                                { title: "Data analyzed", body: `Connected data${Array.isArray(msg.rows) ? ` — ${msg.rows.length} row${msg.rows.length === 1 ? "" : "s"} returned` : ""}.` },
                                { title: "How ARIA answered", body: "Ran a read-only query against the connected source and summarized the result in plain language." },
                              ]}
                              technical={(
                                <pre className="text-[10px] font-mono overflow-x-auto whitespace-pre-wrap">{msg.sql}</pre>
                              )}
                            />
                          </div>
                        )}
                      </div>
                      <p className="text-[10px] text-muted-foreground px-1">{msg.timestamp}</p>
                    </div>
                  </div>
                ) : (
                  <div key={msg.id} className="flex items-start gap-2.5 max-w-2xl ml-auto flex-row-reverse animate-slide-in-right">
                    <div className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/20 flex items-center justify-center text-[10px] font-semibold text-primary mt-0.5">A</div>
                    <div className="space-y-1">
                      <div
                        className="rounded-2xl rounded-tr-md px-4 py-3 shadow-md text-white"
                        style={{ background: "var(--gradient-primary)" }}
                      >
                        <p className="text-[13px] leading-relaxed">{msg.content}</p>
                      </div>
                      <p className="text-[10px] text-muted-foreground px-1 text-right">{msg.timestamp}</p>
                    </div>
                  </div>
                )
              )}

              {isTyping && (
                <div className="px-1 py-1 max-w-2xl">
                  <AIProcessingIndicator compact stage={copilotStage} />
                </div>
              )}
              {isTyping && (
                <div className="flex items-start gap-2.5 max-w-2xl animate-slide-in-left">
                  <div className="w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center">
                    <Sparkles size={11} className="text-primary" />
                  </div>
                  <div className="bg-card/90 border border-border/80 rounded-2xl rounded-tl-md px-4 py-3 shadow-sm backdrop-blur-sm flex items-center gap-1.5">
                    <span className="typing-dot" />
                    <span className="typing-dot" />
                    <span className="typing-dot" />
                  </div>
                </div>
              )}
              <div ref={endRef} />
            </div>
          </>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-center animate-fade-in">
            <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mb-4">
              <Sparkles size={22} className="text-primary" />
            </div>
            <p className="text-base font-medium text-foreground">ARIA Copilot</p>
            <p className="text-sm text-muted-foreground mt-1.5 max-w-xs">Ask me anything about your connected data. I can help you analyze, explore, and understand your information.</p>
          </div>
        )}

        {/* Suggestion chips */}
        <div className="px-5 pb-2 flex flex-wrap gap-1.5">
          {suggestions.map((s) => (
            <button key={s} onClick={() => send(s)}
              className="text-[11px] px-2.5 py-1 bg-secondary border border-border rounded-full text-muted-foreground hover:text-foreground hover:border-primary/40 transition-colors">
              {s}
            </button>
          ))}
        </div>

        {/* Input */}
        <div className="px-5 pb-4 flex-shrink-0">
          <div className="flex items-end gap-2 bg-card border border-border rounded-xl px-3.5 py-2.5 focus-within:border-primary/50 transition-colors">
            <input value={input} onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
              placeholder="Ask ARIA anything about your data..."
              className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground outline-none" />
            <button onClick={() => send()}
              className={`p-1.5 rounded-lg transition-all ${input.trim() ? "bg-primary text-primary-foreground hover:opacity-90" : "bg-secondary text-muted-foreground"}`}>
              <Send size={13} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
