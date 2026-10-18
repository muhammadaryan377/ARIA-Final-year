import { useEffect, useRef, useState } from "react";
import { apiFetch } from "../../services/api";

type Source = { source_id?: string; filename?: string; page?: number; content?: string; snippet?: string };
type Turn = { question: string; answer: string; sources: Source[]; evidence: string; warnings: string[] };

export default function DocumentChat({ documents }: { documents: { id: string; name: string; status?: string }[] }) {
  const [selected, setSelected] = useState("");
  const [question, setQuestion] = useState("");
  const [conversation, setConversation] = useState<string | undefined>();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const requestVersion = useRef(0);
  const ready = documents.filter(d => d.status === "ready");
  const validSelection = ready.some(d => d.id === selected);
  const reset = () => { requestVersion.current += 1; setConversation(undefined); setTurns([]); setError(""); setQuestion(""); setBusy(false); };
  useEffect(() => {
    if (!ready.some(d => d.id === selected)) {
      reset();
      setSelected(ready[0]?.id || "");
    }
  }, [documents, selected]);
  useEffect(() => () => { requestVersion.current += 1; }, []);

  const ask = async () => {
    const q = question.trim();
    if (busy || !validSelection || q.length < 2) return;
    const version = ++requestVersion.current;
    setBusy(true); setError("");
    try {
      const response = await apiFetch("/insight/pdf/chat", {
        method: "POST", timeoutMs: 150000,
        body: JSON.stringify({ question: q, document_ids: [selected], conversation_id: conversation }),
      });
      if (version !== requestVersion.current) return;
      if (response.ok === false) throw new Error(response.detail || "Document chat could not complete.");
      setConversation(response.conversation_id);
      setTurns(previous => [...previous, {
        question: q, answer: response.answer || "No answer was returned.",
        sources: response.sources || [], evidence: response.evidence_status || "unverified",
        warnings: response.warnings || [],
      }]);
      setQuestion("");
    } catch (e: unknown) {
      if (version === requestVersion.current) setError(e instanceof Error ? e.message : "Document chat failed. Please retry.");
    } finally {
      if (version === requestVersion.current) setBusy(false);
    }
  };

  return <section className="premium-card rounded-2xl p-5 space-y-4" aria-label="Document chat">
    <div className="flex items-start justify-between gap-3">
      <div><h2 className="text-xl font-semibold">Ask your documents</h2>
        <p className="text-sm text-muted-foreground mt-1">Choose a PDF, ask a question, then continue with follow-ups in this chat.</p></div>
      <button type="button" onClick={reset} disabled={busy} className="text-sm underline disabled:opacity-50">New chat</button>
    </div>
    <label className="block text-sm font-medium">PDF for this chat
      <select value={selected} disabled={busy || !ready.length} onChange={e => { reset(); setSelected(e.target.value); }}
        className="mt-2 block w-full rounded-xl border border-border bg-card p-3 text-foreground">
        {!ready.length && <option value="">Upload a text PDF from the Data page first</option>}
        {ready.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
      </select>
    </label>
    <div className="space-y-4 max-h-[560px] overflow-y-auto" aria-live="polite">
      {turns.map((turn, i) => <article key={i} className="rounded-xl border border-border p-4 space-y-3">
        <p className="font-semibold whitespace-pre-wrap">{turn.question}</p>
        <div className="text-sm leading-6 whitespace-pre-wrap break-words">{turn.answer}</div>
        <p className="text-xs text-muted-foreground">Evidence status: {turn.evidence.replaceAll("_", " ")}</p>
        {turn.warnings.map((w, j) => <p key={j} className="text-xs text-amber-700 dark:text-amber-400">{w}</p>)}
        {turn.sources.length > 0 && <details className="text-sm"><summary className="cursor-pointer font-medium">Sources and page references ({turn.sources.length})</summary>
          <ul className="mt-2 space-y-2">{turn.sources.map((source, j) => <li key={j}>
            <span className="font-medium">[{source.source_id || `S${j + 1}`}] {source.filename || "Selected PDF"}{source.page != null ? ` — page ${source.page}` : ""}</span>
            {(source.snippet || source.content) && <p className="text-muted-foreground whitespace-pre-wrap">{source.snippet || source.content}</p>}
          </li>)}</ul>
        </details>}
      </article>)}
    </div>
    <form onSubmit={e => { e.preventDefault(); void ask(); }} className="space-y-3">
      <textarea aria-label="Question about selected PDF" value={question} maxLength={4000} disabled={busy || !validSelection}
        onChange={e => setQuestion(e.target.value)} rows={3} placeholder="Summarize this report with page citations."
        className="w-full rounded-xl border border-border bg-card p-3 text-sm disabled:opacity-50" />
      {error && <p role="alert" className="text-sm text-red-600 dark:text-red-400">{error}</p>}
      <button type="submit" disabled={busy || !validSelection || question.trim().length < 2} className="premium-btn px-5 py-2.5 rounded-xl text-sm disabled:opacity-50">
        {busy ? "Reading evidence and answering…" : "Ask document"}
      </button>
    </form>
  </section>;
}
