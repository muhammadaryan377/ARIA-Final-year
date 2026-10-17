import { useEffect, useState } from "react";
import { listDocuments, deleteDocument, retryDocument, searchDocuments } from "../../services/api";
import { useToast } from "../../components/Toast";

function StatusPill({ status }: { status?: string }) {
  const s = status || "unknown";
  const palette: Record<string, string> = {
    ready: "from-emerald-500/20 to-emerald-500/10 text-emerald-600 dark:text-emerald-400 dark:text-emerald-400 border-emerald-500/30",
    partial: "from-amber-500/20 to-amber-500/10 text-amber-600 dark:text-amber-400 dark:text-amber-400 border-amber-500/30",
    failed: "from-rose-500/20 to-rose-500/10 text-rose-600 dark:text-rose-400 dark:text-rose-400 border-rose-500/30",
    uploading: "from-sky-500/20 to-sky-500/10 text-sky-600 dark:text-sky-400 border-sky-500/30",
  };
  const grad = palette[s] || "from-slate-500/20 to-slate-500/10 text-secondary-foreground dark:text-slate-300 border-slate-500/30";
  return (
    <span className={`px-2.5 py-1 rounded-full text-[10px] font-semibold uppercase tracking-wider border bg-gradient-to-r ${grad}`}>
      {s}
    </span>
  );
}

function formatDate(d: string | undefined) {
  if (!d) return "—";
  try { return new Date(d).toLocaleString(); } catch { return d; }
}

function formatSize(bytes?: number) {
  if (!bytes) return "—";
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  return `${(kb / 1024).toFixed(1)} MB`;
}

export default function Documents() {
  const [documents, setDocuments] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [deleting, setDeleting] = useState<string | null>(null);
  const [retrying, setRetrying] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<any[] | null>(null);
  const [searching, setSearching] = useState(false);
  const { error: toastError, success: toastSuccess, info: toastInfo } = useToast() as any;

  const fetchAll = async () => {
    setLoading(true);
    setLoadError("");
    try {
      const r: any = await listDocuments();
      if (r.success) setDocuments(r.data || []);
    } catch (e: any) {
      setLoadError(e?.message || "Failed to load documents");
      toastError("Failed to load documents");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchAll(); }, []);

  const onDelete = async (id: string) => {
    if (!window.confirm("Delete this document? This cannot be undone.")) return;
    setDeleting(id);
    try {
      await deleteDocument(id);
      setDocuments((d) => d.filter((x) => x.id !== id));
      toastSuccess("Document deleted");
    } catch {
      toastError("Delete failed");
    } finally {
      setDeleting(null);
    }
  };

  const onRetry = async (doc: any) => {
    setRetrying(doc.id);
    try {
      await retryDocument(doc.id);
      toastSuccess("Retry started");
      await fetchAll();
    } catch {
      toastError("Retry failed");
    } finally {
      setRetrying(null);
    }
  };

  const onSearch = async () => {
    if (!query.trim()) {
      setResults(null);
      return;
    }
    setSearching(true);
    try {
      const r: any = await searchDocuments(query, undefined, 20);
      if (r.success) setResults(r.data || []);
    } catch {
      toastError("Search failed");
    } finally {
      setSearching(false);
    }
  };

  const clearSearch = () => {
    setQuery("");
    setResults(null);
  };

  const list = results ?? documents;

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-6">
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 mb-3">
            <span className="w-1.5 h-1.5 rounded-full bg-primary" />
            <span className="text-[10px] font-bold uppercase tracking-widest text-primary">Knowledge Base</span>
          </div>
          <h1 className="text-3xl font-bold tracking-tight">Documents</h1>
          <p className="text-sm text-muted-foreground mt-1">Manage and search your uploaded PDF documents</p>
        </div>
        <div className="text-xs text-muted-foreground premium-card px-4 py-2 rounded-xl">
          {documents.length} {documents.length === 1 ? "document" : "documents"}
        </div>
      </div>

      <div className="premium-card rounded-2xl p-4">
        <div className="flex gap-2 items-center">
          <div className="relative flex-1">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && onSearch()}
              placeholder="Search across all documents…"
              className="w-full px-4 py-2.5 rounded-xl bg-card/40 dark:bg-slate-900/40 border border-white/20 dark:border-slate-700/30 focus:outline-none focus:ring-2 focus:ring-primary/30 text-sm"
            />
          </div>
          <button
            onClick={onSearch}
            disabled={searching || !query.trim()}
            className="premium-btn px-5 py-2.5 rounded-xl font-medium text-sm disabled:opacity-50"
          >
            {searching ? "Searching…" : "Search"}
          </button>
          {results && (
            <button onClick={clearSearch} className="px-4 py-2.5 rounded-xl text-sm font-medium border border-white/20 dark:border-slate-700/30 hover:bg-card/40 dark:hover:bg-slate-800/40">
              Clear
            </button>
          )}
        </div>
      </div>

      {loading && list.length === 0 && !loadError && (
        <div className="premium-card rounded-2xl p-12 text-center text-sm text-muted-foreground">Loading documents…</div>
      )}

      {!loading && loadError && list.length === 0 && (
        <div className="premium-card rounded-2xl p-12 text-center">
          <h3 className="text-lg font-semibold mb-1">Documents couldn't be loaded</h3>
          <p className="text-sm text-muted-foreground">{loadError}</p>
          <button onClick={fetchAll} className="premium-btn px-5 py-2.5 rounded-xl font-medium text-sm mt-4">
            Retry
          </button>
        </div>
      )}

      {!loading && !loadError && list.length === 0 && (
        <div className="premium-card rounded-2xl p-12 text-center">
          <div className="text-5xl mb-4">{results ? "🔍" : "📄"}</div>
          <h3 className="text-lg font-semibold mb-1">
            {results ? `No results for "${query}"` : "No documents yet"}
          </h3>
          <p className="text-sm text-muted-foreground">
            {results ? "Try a different search term." : "Upload PDF documents from the Data Sources page to get started."}
          </p>
        </div>
      )}

      {list.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {list.map((doc) => (
            <article key={doc.id} className="premium-card rounded-2xl p-5 space-y-4 hover:shadow-lg transition-shadow">
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1 min-w-0">
                  <h3 className="font-semibold text-base truncate">{doc.name || doc.original_name || "Untitled"}</h3>
                  <div className="flex items-center gap-2 mt-1.5 flex-wrap text-[11px] text-muted-foreground">
                    <span>{formatSize(doc.file_size)}</span>
                    <span>·</span>
                    <span>{doc.page_count || 0} pages</span>
                    <span>·</span>
                    <span>{formatDate(doc.created_at)}</span>
                  </div>
                </div>
                <StatusPill status={doc.status} />
              </div>

              <div className="grid grid-cols-3 gap-2 text-center">
                <div className="rounded-lg bg-card/30 dark:bg-slate-800/30 p-2">
                  <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Elements</div>
                  <div className="text-sm font-semibold mt-0.5">{doc.element_count || 0}</div>
                </div>
                <div className="rounded-lg bg-card/30 dark:bg-slate-800/30 p-2">
                  <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Chunks</div>
                  <div className="text-sm font-semibold mt-0.5">{doc.chunk_count || 0}</div>
                </div>
                <div className="rounded-lg bg-card/30 dark:bg-slate-800/30 p-2">
                  <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Tables</div>
                  <div className="text-sm font-semibold mt-0.5">{doc.table_count || 0}</div>
                </div>
              </div>

              {doc.error_message && (
                <div className="text-[11px] rounded-lg bg-rose-500/10 border border-rose-500/20 px-3 py-2 text-rose-600 dark:text-rose-400 dark:text-rose-400">
                  {doc.error_message}
                </div>
              )}

              <div className="flex gap-2 pt-1">
                {(doc.status === "ready" || doc.status === "partial") && (
                  <button
                    className="flex-1 px-3 py-2 rounded-lg text-xs font-medium premium-btn"
                    onClick={() => toastInfo("Use the Ask ARIA page to query this document.")}
                  >
                    Ask Questions
                  </button>
                )}
                {(doc.status === "failed" || doc.status === "partial") && (
                  <button
                    onClick={() => onRetry(doc)}
                    disabled={retrying === doc.id}
                    className="flex-1 px-3 py-2 rounded-lg text-xs font-medium border border-white/30 dark:border-slate-700/30 hover:bg-card/40 dark:hover:bg-slate-800/40 disabled:opacity-50"
                  >
                    {retrying === doc.id ? "Retrying…" : "Retry"}
                  </button>
                )}
                <button
                  onClick={() => onDelete(doc.id)}
                  disabled={deleting === doc.id}
                  className="px-3 py-2 rounded-lg text-xs font-medium bg-rose-500/10 text-rose-600 dark:text-rose-400 dark:text-rose-400 border border-rose-500/20 hover:bg-rose-500/20 disabled:opacity-50"
                >
                  {deleting === doc.id ? "…" : "Delete"}
                </button>
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}
