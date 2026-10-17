import { useState, useEffect } from "react";
import { Search, Trash2, Bookmark, ChevronDown, Clock } from "lucide-react";
import { apiFetch } from "../../services/api";

// Standalone Hypothesis/Prescription pages were removed (hypothesis now
// surfaces as Possible drivers; prescriptions are historical).
// Legacy history rows keep their stored type; the filter maps them to
// Analytics so old records stay findable.
type Category = "All" | "Analytics" | "Insight" | "Prediction";

const LEGACY_TYPE_MAP: Record<string, string> = {
  Prescription: "Analytics",
  Hypothesis: "Analytics",
};

const typeStyle: Record<string, string> = {
  Analytics: "text-primary bg-primary/10",
  Insight: "text-positive bg-positive/10",
  Prediction: "text-warning bg-warning/10",
};

const categories: Category[] = ["All", "Analytics", "Insight", "Prediction"];

interface HistoryItem {
  id: string;
  type: string;
  title: string;
  date: string;
  createdAt?: string;
  source: string;
  summary: string;
  has_insights?: boolean;
}

interface Props {
  historyRefreshKey?: number;
  onDeleteHistory?: (id: string) => void;
}

export default function History({ historyRefreshKey = 0, onDeleteHistory }: Props) {
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState<Category>("All");
  const [sort, setSort] = useState("newest");
  const [deleted, setDeleted] = useState<string[]>([]);
  const [saved, setSaved] = useState<string[]>([]);
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState("");
  // Inline detail stays on this page (pages are isolated — no navigation).
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [details, setDetails] = useState<Record<string, any>>({});
  const [detailLoading, setDetailLoading] = useState<string | null>(null);

  const toggleDetail = async (id: string) => {
    if (expandedId === id) {
      setExpandedId(null);
      return;
    }
    setExpandedId(id);
    if (details[id]) return;
    setDetailLoading(id);
    try {
      // Single client: apiFetch handles API_BASE + credentials:include +
      // X-Request-ID + 401 (handleUnauthorized). Throws on 401/error.
      const data = await apiFetch(`/history/${id}`);
      if (data?.record) {
        setDetails((d) => ({ ...d, [id]: data.record }));
      }
    } catch (err) {
      console.debug("[ARIA] history detail failed:", err);
    } finally {
      setDetailLoading(null);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [historyRefreshKey]);

  const fetchHistory = async () => {
    setLoading(true);
    setLoadError("");
    try {
      const data = await apiFetch(`/history?limit=100`);
      // Shared normalizer (sessions|items|history|data): valid payloads in
      // any of these shapes render; anything else is an error, not empty.
      const rawList = data?.sessions ?? data?.items ?? data?.history ?? data?.data ?? data;
      const list = Array.isArray(rawList) ? rawList : [];
      if (!Array.isArray(rawList)) {
        setLoadError("History came back in an unexpected shape — retry to reload.");
        setItems([]);
        return;
      }
      const mapped = list.map((h: any) => ({
        id: String(h.id),
        type: h.source_type || "Analytics",
        title: h.goal || "Untitled analysis",
        date: h.created_at ? new Date(h.created_at).toLocaleDateString() : "Unknown",
        createdAt: h.created_at || "",
        source: h.db_name || "Unknown",
        summary: h.sql_used ? `Query executed (${h.row_count || 0} rows)` : "Analysis completed",
        has_insights: h.has_insights,
      }));
      setItems(mapped);
    } catch (err: any) {
      console.error("Failed to fetch history:", err);
      setLoadError(err?.message || "History couldn't be loaded. Check your connection and retry.");
      setItems([]);
    } finally {
      setLoading(false);
    }
  };

  const filtered = items.filter((h) => {
    if (deleted.includes(h.id)) return false;
    const mappedType = LEGACY_TYPE_MAP[h.type] || h.type;
    if (category !== "All" && mappedType !== category) return false;
    if (search && !h.title.toLowerCase().includes(search.toLowerCase()) && !h.summary.toLowerCase().includes(search.toLowerCase())) return false;
    return true;
  });

  const sorted = [...filtered].sort((a, b) => {
    // created_at desc (id tiebreak): non-numeric IDs sort stably.
    const aT = Date.parse((a as any).createdAt || "") || 0;
    const bT = Date.parse((b as any).createdAt || "") || 0;
    if (sort === "newest") return bT - aT || String(b.id).localeCompare(String(a.id));
    if (sort === "oldest") return aT - bT || String(a.id).localeCompare(String(b.id));
    return a.type.localeCompare(b.type);
  });

  const handleDelete = async (id: string) => {
    try {
      await apiFetch(`/history/${id}`, { method: "DELETE" });
    } catch (err) {
      console.error("Delete failed:", err);
    }
    setDeleted((d) => [...d, id]);
    onDeleteHistory?.(id);
  };

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <div className="animate-fade-in">
        <h2 className="text-xl font-bold text-foreground tracking-tight">History</h2>
        <p className="text-sm text-muted-foreground mt-0.5">Your previous analyses, insights, and queries.</p>
      </div>

      <div className="space-y-3">
        <div className="flex items-center gap-3 bg-card border border-border rounded-xl px-4 py-3 focus-within:border-primary transition-colors">
          <Search size={14} className="text-muted-foreground flex-shrink-0" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search history..."
            className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground outline-none"
          />
          {search && (
            <button onClick={() => setSearch("")} className="text-xs text-muted-foreground hover:text-foreground transition-colors">Clear</button>
          )}
        </div>

        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex gap-1.5 flex-wrap">
            {categories.map((c) => (
              <button key={c} onClick={() => setCategory(c)}
                className={`text-xs px-3 py-1.5 rounded-full border transition-colors ${
                  category === c ? "bg-primary text-primary-foreground border-primary" : "bg-card text-muted-foreground border-border hover:text-foreground"
                }`}>{c}</button>
            ))}
          </div>
          <div className="ml-auto flex items-center gap-2">
            <span className="text-xs text-muted-foreground">{sorted.length} items</span>
            <div className="relative">
              <select value={sort} onChange={(e) => setSort(e.target.value)}
                className="appearance-none bg-card border border-border rounded-lg pl-3 pr-7 py-1.5 text-xs text-foreground outline-none cursor-pointer">
                <option value="newest">Newest first</option>
                <option value="oldest">Oldest first</option>
                <option value="type">By type</option>
              </select>
              <ChevronDown size={11} className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground pointer-events-none" />
            </div>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center mb-4">
            <Clock size={20} className="text-muted-foreground" />
          </div>
          <p className="text-sm text-muted-foreground">Loading history...</p>
        </div>
      ) : sorted.length > 0 ? (
        <div className="space-y-2 stagger-children">
          {sorted.map((item) => {
            const isSaved = saved.includes(item.id);
            return (
              <div key={item.id} className="premium-card bg-card border border-border rounded-xl p-4 group cursor-default">
                <div className="flex items-center gap-4">
                <div className="flex-1 min-w-0 space-y-1.5">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${typeStyle[LEGACY_TYPE_MAP[item.type] || item.type] || "text-muted-foreground bg-secondary"}`}>
                      {LEGACY_TYPE_MAP[item.type] || item.type}
                    </span>
                    <span className="text-xs text-muted-foreground bg-secondary px-2 py-0.5 rounded-full">{item.source}</span>
                  </div>
                  <p className="text-sm font-medium text-foreground truncate">{item.title}</p>
                  <p className="text-xs text-muted-foreground truncate">{item.summary}</p>
                </div>

                <div className="flex-shrink-0 flex items-center gap-2">
                  <span className="text-xs text-muted-foreground hidden md:block whitespace-nowrap">{item.date}</span>
                  <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={() => toggleDetail(item.id)}
                      aria-label={expandedId === item.id ? "Collapse details" : "Expand details"}
                      className="p-1.5 rounded-lg hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors">
                      <ChevronDown size={13} className={`transition-transform ${expandedId === item.id ? "rotate-180" : ""}`} />
                    </button>
                    <button
                      onClick={() => setSaved((s) => isSaved ? s.filter((x) => x !== item.id) : [...s, item.id])}
                      className={`p-1.5 rounded-lg hover:bg-secondary transition-colors ${isSaved ? "text-primary" : "text-muted-foreground hover:text-foreground"}`}>
                      <Bookmark size={13} fill={isSaved ? "currentColor" : "none"} />
                    </button>
                    <button
                      onClick={() => handleDelete(item.id)}
                      className="p-1.5 rounded-lg hover:bg-secondary text-muted-foreground hover:text-negative transition-colors">
                      <Trash2 size={13} />
                    </button>
                  </div>
                </div>
                </div>
                {expandedId === item.id && (
                  <div className="mt-3 pt-3 border-t border-border space-y-2">
                    {detailLoading === item.id ? (
                      <p className="text-xs text-muted-foreground">Loading details…</p>
                    ) : details[item.id] ? (
                      <>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                          <div>
                            <p className="text-muted-foreground">Rows</p>
                            <p className="font-semibold text-foreground">{details[item.id].row_count ?? "—"}</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">Source</p>
                            <p className="font-semibold text-foreground">{details[item.id].db_name || item.source}</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">Date</p>
                            <p className="font-semibold text-foreground">{details[item.id].created_at ? new Date(details[item.id].created_at).toLocaleString() : item.date}</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">Columns</p>
                            <p className="font-semibold text-foreground truncate">{(details[item.id].columns || []).join(", ") || "—"}</p>
                          </div>
                        </div>
                        {details[item.id].sql_used && (
                          <pre className="font-mono text-[11px] text-foreground bg-muted/50 rounded-lg px-3 py-2 overflow-x-auto whitespace-pre-wrap">
                            {details[item.id].sql_used}
                          </pre>
                        )}
                      </>
                    ) : (
                      <p className="text-xs text-muted-foreground">Details unavailable for this entry.</p>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ) : loadError ? (
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <p className="text-sm font-medium text-foreground">History needs a reload</p>
          <p className="text-sm text-muted-foreground mt-1">{loadError}</p>
          <button onClick={() => { setLoadError(""); fetchHistory(); }} className="mt-4 text-xs text-primary hover:underline">Retry</button>
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center mb-4">
            <Clock size={20} className="text-muted-foreground" />
          </div>
          <p className="text-sm font-medium text-foreground">
            {search ? "No results found" : "No history yet"}
          </p>
          <p className="text-sm text-muted-foreground mt-1">
            {search ? `No items match "${search}"` : "Start analyzing your data to see history here."}
          </p>
          {search && (
            <button onClick={() => setSearch("")} className="mt-4 text-xs text-primary hover:underline">Clear search</button>
          )}
        </div>
      )}
    </div>
  );
}
