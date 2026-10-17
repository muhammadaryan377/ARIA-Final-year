import { useState, useEffect, useCallback, useRef } from "react";
import {
  Database, FileSpreadsheet, FileText, Server, Plug, CheckCircle,
  AlertTriangle, RefreshCw, Table, XCircle, Upload, Loader2,
  Eye, ChevronRight, Zap, Search, ArrowUpDown, ArrowUp, ArrowDown,
  Columns3, Network, ChevronDown, ChevronLeft, Info, Shield,
  X, AlertCircle, TrendingDown, BarChart3, GitBranch, Key,
  Link, Unlink, Check, Minus, Maximize2, Minimize2,
  Target, Activity, Clock, CheckSquare, XSquare, Filter,
  ChevronUp, Download, MoreHorizontal, Copy, Trash2,
} from "lucide-react";

import { apiFetch } from "../../services/api";

type SourceType = "postgresql" | "mysql" | "csv" | "pdf" | "";
type Tab = "sources" | "schema" | "preview" | "quality";

const SOURCES = [
  { id: "postgresql" as SourceType, label: "PostgreSQL", icon: Server, color: "#336791", bgColor: "rgba(51,103,145,0.08)", description: "Connect to a PostgreSQL database for full analytics.", category: "Database" },
  { id: "mysql" as SourceType, label: "MySQL", icon: Database, color: "#00758F", bgColor: "rgba(0,117,143,0.08)", description: "Connect to a MySQL database for full analytics.", category: "Database" },
  { id: "csv" as SourceType, label: "CSV File", icon: FileSpreadsheet, color: "#217346", bgColor: "rgba(33,115,70,0.08)", description: "Upload a structured CSV dataset for analysis.", category: "File Upload" },
  { id: "pdf" as SourceType, label: "PDF Document", icon: FileText, color: "#B30B00", bgColor: "rgba(179,11,0,0.08)", description: "Upload a PDF document for AI-powered extraction.", category: "File Upload" },
];

const TABS: { id: Tab; label: string; icon: typeof Database }[] = [
  { id: "sources", label: "Connected Sources", icon: Plug },
  { id: "schema", label: "Schema", icon: Columns3 },
  { id: "preview", label: "Data Preview", icon: Eye },
  { id: "quality", label: "Data Quality", icon: Shield },
];

// Single HTTP client (P0-2): all requests go through apiFetch from
// src/services/api.js (API_BASE + credentials:include + X-Request-ID +
// centralized 401 handling). Timeout uses the central default (60-120s
// clamped REQUEST_TIMEOUT_MS; apiFetch supports an external `signal`
// but no per-call timeout option).
const api = apiFetch;
const apiGet = (path: string) => apiFetch(path);

interface Props {
  connectionStatus?: string;
  selectedSourceType?: string;
  dbName?: string;
  schemaData?: any;
  schemaStatus?: string;
  schemaError?: string;
  onConnect?: (details: any) => Promise<void>;
  onDisconnect?: () => void;
  onSchemaRetry?: () => void;
  onSourceTypeSelect?: (type: string) => void;
  selectedFile?: File | null;
  onFileSelect?: (file: File) => void;
  onFileUpload?: (sourceType?: string) => Promise<void>;
  dbHost?: string;
  dbPort?: string;
  dbUsername?: string;
  dbPassword?: string;
  setDbHost?: (v: string) => void;
  setDbPort?: (v: string) => void;
  setDbName?: (v: string) => void;
  setDbUsername?: (v: string) => void;
  setDbPassword?: (v: string) => void;
  connectionError?: string;
}

export default function Data({
  connectionStatus = "not-connected",
  selectedSourceType = "",
  dbName = "",
  schemaData,
  schemaStatus = "idle",
  schemaError = "",
  onConnect,
  onDisconnect,
  onSchemaRetry,
  onSourceTypeSelect,
  selectedFile,
  onFileSelect,
  onFileUpload,
  dbHost: propDbHost = "localhost",
  dbPort: propDbPort = "5432",
  dbUsername: propDbUsername = "postgres",
  dbPassword: propDbPassword = "",
  setDbHost: propSetDbHost,
  setDbPort: propSetDbPort,
  setDbName: propSetDbName,
  setDbUsername: propSetDbUsername,
  setDbPassword: propSetDbPassword,
  connectionError = "",
}: Props) {
  const [activeTab, setActiveTab] = useState<Tab>("sources");
  const [localSourceType, setLocalSourceType] = useState<SourceType>("" as SourceType);
  const [localFile, setLocalFile] = useState<File | null>(null);
  const [localError, setLocalError] = useState("");
  const [localConnecting, setLocalConnecting] = useState(false);
  const [lHost, setLHost] = useState(propDbHost);
  const [lPort, setLPort] = useState(propDbPort);
  const [lDbName, setLDbName] = useState("");
  const [lUser, setLUser] = useState(propDbUsername);
  const [lPass, setLPass] = useState(propDbPassword);
  const [showPassword, setShowPassword] = useState(false);
  const [selectedTable, setSelectedTable] = useState("");
  const [schemaSearch, setSchemaSearch] = useState("");
  const [showERD, setShowERD] = useState(false);
  const [columnDetail, setColumnDetail] = useState<any>(null);
  const [showColumnDetail, setShowColumnDetail] = useState<string | null>(null);
  const [overrideMsg, setOverrideMsg] = useState("");

  const fileInputRef = useRef<HTMLInputElement>(null);

  const sourceType = selectedSourceType || localSourceType;
  const file = selectedFile !== undefined ? selectedFile : localFile;
  const error = connectionError || localError;
  const connecting = connectionStatus === "connecting" || localConnecting;
  const isConnected = connectionStatus === "connected";
  const tables = schemaData?.tables || {};
  const tableNames = Object.keys(tables);
  const sourceMeta = SOURCES.find((s) => s.id === sourceType);
  const SourceIcon = sourceMeta?.icon || Plug;

  useEffect(() => { setLPass(propDbPassword || ""); }, [propDbPassword]);

  // Connection health state — fetched only on mount / connect, not on every render
  const [connHealth, setConnHealth] = useState<any>(null);
  const [healthError, setHealthError] = useState("");
  const connHealthFetched = useRef(false);
  useEffect(() => {
    if (isConnected && !connHealthFetched.current) {
      connHealthFetched.current = true;
      setHealthError("");
      apiGet("/api/data/connection").then(setConnHealth).catch((e: any) => {
        setHealthError(e?.message || "Could not load connection health.");
      });
    }
    if (!isConnected) { connHealthFetched.current = false; setConnHealth(null); setHealthError(""); }
  }, [isConnected]);

  // Load overrides — only once per connection
  const [overrides, setOverrides] = useState<any>({});
  const overridesFetched = useRef(false);
  useEffect(() => {
    if (isConnected && !overridesFetched.current) {
      overridesFetched.current = true;
      apiGet("/api/data/overrides").then((d) => setOverrides(d.overrides || {})).catch(() => {});
    }
    if (!isConnected) { overridesFetched.current = false; setOverrides({}); }
  }, [isConnected]);

  // Fetch table row counts from summary endpoint (schema mapping doesn't have row_count)
  const [tableRowCounts, setTableRowCounts] = useState<Record<string, number>>({});
  useEffect(() => {
    if (isConnected && schemaStatus === "success") {
      apiGet("/api/data/summary").then((d) => {
        const map: Record<string, number> = {};
        (d.tables || []).forEach((t: any) => { map[t.name] = t.row_count; });
        setTableRowCounts(map);
      }).catch(() => {});
    }
  }, [isConnected, schemaStatus]);

  const handleSourceSelect = (id: SourceType) => {
    if (connecting) return;
    setLocalSourceType(id);
    setLocalFile(null);
    setLocalError("");
    onSourceTypeSelect?.(id);
    if (id === "mysql") { setLPort("3306"); setLUser("root"); setLDbName(""); propSetDbPort?.("3306"); propSetDbUsername?.("root"); propSetDbName?.(""); }
    if (id === "postgresql") { setLPort("5432"); setLUser("postgres"); setLDbName(""); propSetDbPort?.("5432"); propSetDbUsername?.("postgres"); propSetDbName?.(""); }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const ext = f.name.split(".").pop()?.toLowerCase() ?? "";
    const allowed = sourceType === "pdf" ? "pdf" : "csv";
    if (ext !== allowed) { setLocalError(`Only .${allowed} files are supported.`); return; }
    setLocalError("");
    setLocalFile(f);
    onFileSelect?.(f);
  };

  const handleConnect = async () => {
    setLocalError("");
    if (!sourceType) { setLocalError("Select a data source first."); return; }
    if (sourceType === "postgresql" || sourceType === "mysql") {
      if (!lHost.trim()) { setLocalError("Host is required."); return; }
      if (!lDbName.trim()) { setLocalError("Database name is required."); return; }
      if (!lUser.trim()) { setLocalError("Username is required."); return; }
      if (!lPass.trim()) { setLocalError("Password is required."); return; }
      if (!/^\d+$/.test(lPort)) { setLocalError("Port must be numbers only."); return; }
      propSetDbHost?.(lHost); propSetDbPort?.(lPort); propSetDbName?.(lDbName);
      propSetDbUsername?.(lUser); propSetDbPassword?.(lPass);
    }
    setLocalConnecting(true);
    try {
      if (sourceType === "csv" || sourceType === "pdf") {
        if (!file) { setLocalError("Please select a file first."); return; }
        await onFileUpload?.(sourceType!);
      } else {
        await onConnect?.({ host: lHost, port: lPort, db: lDbName, user: lUser, password: lPass, sourceType });
      }
    } catch (err: any) {
      setLocalError(err?.message || "Connection failed.");
    } finally {
      setLocalConnecting(false);
    }
  };

  const handleDisconnect = () => {
    setLocalSourceType("" as SourceType); setLocalFile(null); setLocalError("");
    setLHost("localhost"); setLPort("5432"); setLDbName(""); setLUser("postgres"); setLPass("");
    propSetDbHost?.("localhost"); propSetDbPort?.("5432"); propSetDbName?.("");
    propSetDbUsername?.("postgres"); propSetDbPassword?.("");
    onDisconnect?.();
  };

  const handleRefreshSchema = async () => {
    setOverrideMsg("");
    onSchemaRetry?.();
  };

  const handleRefreshConnection = async () => {
    try {
      const r = await api("/api/data/refresh", { method: "POST" });
      setOverrideMsg(r.message || "Refreshed");
      setTimeout(() => setOverrideMsg(""), 3000);
    } catch (e: any) {
      setOverrideMsg(e.message || "Refresh failed");
    }
  };

  const handleOverride = async (action: string, table: string, column = "", references_table = "", references_column = "") => {
    try {
      const r = await api("/api/data/schema/override", {
        method: "POST",
        body: JSON.stringify({ action, table, column, references_table, references_column }),
      });
      if (r.ok) {
        setOverrideMsg(r.message || "Done");
        setTimeout(() => setOverrideMsg(""), 3000);
        // Reload overrides
        const ov = await apiGet("/api/data/overrides");
        setOverrides(ov.overrides || {});
        // Reload schema
        onSchemaRetry?.();
      } else {
        setOverrideMsg(r.detail || r.message || "Failed");
      }
    } catch (e: any) {
      setOverrideMsg(e.message || "Override failed");
    }
  };

  // ── SOURCES TAB (inline JSX, not a component — prevents remount/focus loss) ──
  const sourcesTabJSX = (
    <div className="space-y-5">
      {isConnected && healthError && !connHealth && (
        <div className="bg-warning/10 border border-warning/30 text-warning text-sm rounded-xl px-4 py-2.5 flex items-center gap-2">
          <AlertTriangle size={14} /> {healthError}
        </div>
      )}
      {isConnected && connHealth && (
        <div className="premium-card bg-positive/5 border border-positive/20 rounded-xl p-4 animate-fade-in">
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-center gap-3">
              <div key={String(connHealth.database || dbName)} className="w-9 h-9 rounded-lg bg-positive/10 flex items-center justify-center connect-success-pop">
                <SourceIcon size={16} className="text-positive" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <p className="text-sm font-semibold text-foreground">{connHealth.database || dbName}</p>
                  <span className="flex items-center gap-1 text-[11px] text-positive font-medium">
                    <span className="w-1.5 h-1.5 rounded-full bg-positive" /> {connHealth.status === "connected" ? "Connected" : connHealth.status}
                  </span>
                </div>
                <p className="text-[11px] text-muted-foreground mt-0.5">
                  {connHealth.db_type?.toUpperCase()} · {connHealth.host} · Schema: {connHealth.schema || "default"} · {connHealth.table_count} tables · {connHealth.column_count} columns
                </p>
                <p className="text-[10px] text-muted-foreground mt-0.5">
                  Last scan: {connHealth.last_schema_scan ? new Date(connHealth.last_schema_scan).toLocaleString() : "Never"} · Provider: {connHealth.provider}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button onClick={handleRefreshConnection}
                className="flex items-center gap-1.5 text-[11px] text-muted-foreground hover:text-foreground px-3 py-1.5 rounded-lg hover:bg-secondary transition-all">
                <RefreshCw size={11} /> Refresh
              </button>
              <button onClick={handleRefreshSchema}
                className="flex items-center gap-1.5 text-[11px] text-muted-foreground hover:text-foreground px-3 py-1.5 rounded-lg hover:bg-secondary transition-all">
                <Columns3 size={11} /> Rescan Schema
              </button>
              <button onClick={handleDisconnect}
                className="flex items-center gap-1.5 text-[11px] text-negative hover:text-negative/80 px-3 py-1.5 rounded-lg hover:bg-negative/5 transition-all">
                <XCircle size={11} /> Disconnect
              </button>
            </div>
          </div>
          <div className="mt-3 bg-secondary/60 rounded-full h-1">
            <div className="h-1 rounded-full bg-positive connect-progress"
              style={{ width: schemaStatus === "success" ? "100%" : schemaStatus === "running" ? "65%" : "25%" }} />
          </div>
          <div className="flex items-center justify-between mt-1.5">
            <span className="text-[10px] text-muted-foreground">
              {schemaStatus === "success" ? "Schema ready" : schemaStatus === "running" ? "Loading schema..." : "Schema pending"}
            </span>
            {connHealth.tables_without_pk?.length > 0 && (
              <span className="text-[10px] text-amber-600 dark:text-amber-400 flex items-center gap-1">
                <AlertTriangle size={10} /> {connHealth.tables_without_pk.length} tables without PK
              </span>
            )}
          </div>
          {overrideMsg && (
            <div className="mt-2 text-[11px] text-primary bg-primary/5 px-3 py-1.5 rounded-lg">{overrideMsg}</div>
          )}
        </div>
      )}

      {!isConnected && (
        <div>
          <p className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider mb-3 flex items-center gap-2">
            <Zap size={11} /> Choose a data source
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {SOURCES.map((s) => {
              const Icon = s.icon;
              return (
                <button key={s.id} onClick={() => handleSourceSelect(s.id)}
                  className={`relative premium-card rounded-xl p-4 text-left transition-all duration-200 group cursor-pointer
                    ${sourceType === s.id ? "border-primary/50 shadow-glow" : "border-border"}`}
                  style={{ background: sourceType === s.id ? s.bgColor : "var(--card)" }}>
                  <div className="absolute top-0 left-0 right-0 h-[3px] rounded-t-xl transition-opacity duration-200"
                    style={{ background: `linear-gradient(90deg, ${s.color}, ${s.color}88)`, opacity: sourceType === s.id ? 1 : 0 }} />
                  <div className="flex items-start gap-3">
                    <div className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0"
                      style={{ background: `${s.color}18` }}>
                      <Icon size={16} style={{ color: s.color }} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-0.5">
                        <p className="text-sm font-semibold text-foreground">{s.label}</p>
                        <span className="text-[10px] px-1.5 py-0.5 rounded-full font-medium"
                          style={{ background: `${s.color}15`, color: s.color }}>{s.category}</span>
                      </div>
                      <p className="text-[11px] text-muted-foreground leading-relaxed">{s.description}</p>
                    </div>
                    {sourceType === s.id && <CheckCircle size={14} className="flex-shrink-0 text-primary mt-0.5" />}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {sourceType && !isConnected && (
        <div className="animate-slide-in-bottom">
          <div className="divider-gradient mb-5" />
          {(sourceType === "postgresql" || sourceType === "mysql") && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 mb-1">
                <SourceIcon size={14} style={{ color: sourceMeta?.color }} />
                <p className="text-sm font-semibold text-foreground">{sourceMeta?.label} Connection</p>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <Field label="Host" id="db-host" value={lHost} onChange={setLHost} placeholder="localhost" disabled={connecting} />
                <Field label="Port" id="db-port" value={lPort} onChange={setLPort} placeholder={sourceType === "mysql" ? "3306" : "5432"} disabled={connecting} />
                <Field label="Database Name" id="db-name" value={lDbName} onChange={setLDbName} placeholder="my_database" disabled={connecting} />
                <Field label="Username" id="db-user" value={lUser} onChange={setLUser} placeholder={sourceType === "mysql" ? "root" : "postgres"} disabled={connecting} />
              </div>
              <Field label="Password" id="db-pass" type="password-show" value={lPass} onChange={setLPass} placeholder="Enter password" disabled={connecting}>
                <button type="button" onClick={() => setShowPassword((s) => !s)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground text-[10px]">
                  {showPassword ? "Hide" : "Show"}
                </button>
              </Field>
              {error && (
                <div className="flex items-center gap-2 bg-negative/8 border border-negative/20 rounded-lg px-3 py-2.5">
                  <AlertTriangle size={13} className="text-negative flex-shrink-0" />
                  <p className="text-xs text-negative">{error}</p>
                </div>
              )}
              <button onClick={handleConnect} disabled={connecting}
                className="premium-btn btn-glow w-full sm:w-auto px-6 py-2.5 rounded-xl text-sm font-semibold text-white flex items-center justify-center gap-2 disabled:opacity-50"
                style={{ background: "var(--gradient-primary)" }}>
                {connecting ? <><Loader2 size={13} className="animate-spin" /> Connecting...</> : <><Plug size={13} /> Connect {sourceMeta?.label}</>}
              </button>
            </div>
          )}
          {sourceType === "csv" && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 mb-1">
                <FileSpreadsheet size={14} style={{ color: sourceMeta?.color }} />
                <p className="text-sm font-semibold text-foreground">Upload CSV File</p>
              </div>
              <div onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200
                  ${file ? "border-primary/40 bg-primary/4" : "border-border hover:border-primary/40 hover:bg-primary/2"}`}>
                <input ref={fileInputRef} type="file" accept=".csv,text/csv" onChange={handleFileChange} disabled={connecting} className="hidden" />
                <FileSpreadsheet size={28} className={`mx-auto mb-2 transition-colors ${file ? "text-primary" : "text-muted-foreground"}`} />
                {file ? (
                  <><p className="text-sm font-medium text-foreground">{file.name}</p><p className="text-[11px] text-muted-foreground mt-1">{(file.size / 1024).toFixed(1)} KB</p></>
                ) : (
                  <><p className="text-sm font-medium text-foreground">Drop a CSV here or click to browse</p><p className="text-[11px] text-muted-foreground mt-1">.csv only</p></>
                )}
              </div>
              {error && <div className="flex items-center gap-2 bg-negative/8 border border-negative/20 rounded-lg px-3 py-2.5"><AlertTriangle size={13} className="text-negative flex-shrink-0" /><p className="text-xs text-negative">{error}</p></div>}
              <button onClick={handleConnect} disabled={!file || connecting}
                className="premium-btn btn-glow w-full sm:w-auto px-6 py-2.5 rounded-xl text-sm font-semibold text-white flex items-center justify-center gap-2 disabled:opacity-50"
                style={{ background: "var(--gradient-primary)" }}>
                {connecting ? <><Loader2 size={13} className="animate-spin" /> Processing...</> : <><Upload size={13} /> Upload & Analyze CSV</>}
              </button>
            </div>
          )}
          {sourceType === "pdf" && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 mb-1">
                <FileText size={14} style={{ color: sourceMeta?.color }} />
                <p className="text-sm font-semibold text-foreground">Upload PDF Document</p>
              </div>
              <div onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200
                  ${file ? "border-primary/40 bg-primary/4" : "border-border hover:border-primary/40 hover:bg-primary/2"}`}>
                <input ref={fileInputRef} type="file" accept=".pdf,application/pdf" onChange={handleFileChange} disabled={connecting} className="hidden" />
                <FileText size={28} className={`mx-auto mb-2 transition-colors ${file ? "text-primary" : "text-muted-foreground"}`} />
                {file ? (
                  <><p className="text-sm font-medium text-foreground">{file.name}</p><p className="text-[11px] text-muted-foreground mt-1">{(file.size / 1024).toFixed(1)} KB</p></>
                ) : (
                  <><p className="text-sm font-medium text-foreground">Drop a PDF here or click to browse</p><p className="text-[11px] text-muted-foreground mt-1">.pdf only</p></>
                )}
              </div>
              {error && <div className="flex items-center gap-2 bg-negative/8 border border-negative/20 rounded-lg px-3 py-2.5"><AlertTriangle size={13} className="text-negative flex-shrink-0" /><p className="text-xs text-negative">{error}</p></div>}
              <button onClick={handleConnect} disabled={!file || connecting}
                className="premium-btn btn-glow w-full sm:w-auto px-6 py-2.5 rounded-xl text-sm font-semibold text-white flex items-center justify-center gap-2 disabled:opacity-50"
                style={{ background: "var(--gradient-primary)" }}>
                {connecting ? <><Loader2 size={13} className="animate-spin" /> Processing...</> : <><Upload size={13} /> Upload & Analyze PDF</>}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );

  // ── SCHEMA TAB ──
  const SchemaTab = () => {
    const [tableSearch, setTableSearch] = useState("");
    const filteredTables = tableSearch
      ? tableNames.filter((n) => n.toLowerCase().includes(tableSearch.toLowerCase()))
      : tableNames;

    const currentTable = tables[selectedTable];
    const currentCols = currentTable?.columns || [];
    const pkList = overrides?.pk?.[selectedTable] || currentTable?.primary_key || currentTable?.inferred_primary_key || [];
    const declaredRels = schemaData?.declared_fk_list?.filter((r: any) => r.from?.startsWith(selectedTable + ".")) || [];
    const inferredRels = schemaData?.inferred_fk_list?.filter((r: any) => r.from?.startsWith(selectedTable + ".")) || [];

    const relFor = (from: string, to: string) => {
      const relId = from + "->" + to;
      const rejected = overrides?.rejected?.includes(relId);
      const confirmed = overrides?.fk?.[relId];
      return { rejected, confirmed, relId };
    };

    return (
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <div className="flex-1 relative">
            <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
            <input value={tableSearch} onChange={(e) => setTableSearch(e.target.value)}
              placeholder="Search tables..."
              className="w-full bg-secondary border border-border rounded-lg pl-8 pr-3 py-2 text-xs text-foreground placeholder:text-muted-foreground outline-none focus:border-primary" />
          </div>
          <button onClick={() => setShowFKModal(true)}
            className="flex items-center gap-1.5 text-xs font-medium text-primary hover:text-primary/80 px-3 py-2 rounded-lg bg-primary/10 hover:bg-primary/15 transition-all">
            <Link size={12} /> Create Relationship
          </button>
          <button onClick={() => setShowERD(true)}
            className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground px-3 py-2 rounded-lg bg-secondary hover:bg-secondary/80 transition-all">
            <Network size={12} /> View ERD
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="bg-card border border-border rounded-xl overflow-hidden max-h-[600px] flex flex-col">
            <div className="px-3 py-2.5 border-b border-border flex items-center justify-between">
              <p className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">Tables ({filteredTables.length})</p>
            </div>
            <div className="overflow-y-auto flex-1">
              {filteredTables.map((name) => {
                const t = tables[name];
                const hasPk = (overrides?.pk?.[name] || t?.primary_key || t?.inferred_primary_key || []).length > 0;
                return (
                  <button key={name} onClick={() => { setSelectedTable(name); setShowColumnDetail(null); }}
                    className={`w-full flex items-center justify-between px-3 py-2 hover:bg-secondary/40 transition-colors border-b border-border/50 last:border-0
                      ${selectedTable === name ? "bg-secondary/60" : ""}`}>
                    <div className="flex items-center gap-2 text-left">
                      <Table size={12} className="text-muted-foreground flex-shrink-0" />
                      <div>
                        <p className="text-xs font-medium text-foreground font-mono">{name}</p>
                        <p className="text-[10px] text-muted-foreground">{t?.columns?.length ?? "?"} cols · {tableRowCounts[name] ?? t?.row_count ?? "?"} rows</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-1.5">
                      {hasPk ? <Key size={10} className="text-positive" /> : <Key size={10} className="text-muted-foreground/40" />}
                      <ChevronRight size={11} className="text-muted-foreground" />
                    </div>
                  </button>
                );
              })}
              {filteredTables.length === 0 && (
                <div className="p-6 text-center">
                  <Table size={20} className="mx-auto text-muted-foreground mb-2" />
                  <p className="text-xs text-muted-foreground">No tables found.</p>
                </div>
              )}
            </div>
          </div>

          <div className="md:col-span-2 bg-card border border-border rounded-xl overflow-hidden">
            {!selectedTable ? (
              <div className="p-8 text-center">
                <Table size={24} className="mx-auto text-muted-foreground mb-2" />
                <p className="text-sm font-medium text-foreground">Select a table</p>
                <p className="text-xs text-muted-foreground mt-1">Choose a table from the list to view columns and relationships.</p>
              </div>
            ) : (
              <div>
                <div className="px-4 py-3 border-b border-border flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <p className="text-sm font-semibold text-foreground font-mono">{selectedTable}</p>
                    {pkList.length > 0 && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-positive/10 text-positive font-medium flex items-center gap-1">
                        <Key size={9} /> PK: {pkList.join(", ")}
                      </span>
                    )}
                    {pkList.length === 0 && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-amber-500/10 text-amber-600 dark:text-amber-400 font-medium">No PK detected</span>
                    )}
                    {(tableRowCounts[selectedTable] ?? currentTable?.row_count) != null && (
                      <span className="text-[10px] text-muted-foreground">{tableRowCounts[selectedTable] ?? currentTable?.row_count} rows</span>
                    )}
                  </div>
                </div>

                {/* Columns table */}
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="text-left text-[11px] font-medium text-muted-foreground px-4 py-2.5">Column</th>
                        <th className="text-left text-[11px] font-medium text-muted-foreground px-4 py-2.5">Type</th>
                        <th className="text-center text-[11px] font-medium text-muted-foreground px-2 py-2.5">PK</th>
                        <th className="text-center text-[11px] font-medium text-muted-foreground px-2 py-2.5">FK</th>
                        <th className="text-center text-[11px] font-medium text-muted-foreground px-2 py-2.5">Null</th>
                        <th className="text-center text-[11px] font-medium text-muted-foreground px-2 py-2.5">Unique</th>
                        <th className="text-left text-[11px] font-medium text-muted-foreground px-2 py-2.5">Null %</th>
                        <th className="text-left text-[11px] font-medium text-muted-foreground px-2 py-2.5">Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {currentCols.map((col: any, i: number) => {
                        const colName = col.column;
                        const isPk = pkList.includes(colName);
                        const fkInfo = declaredRels.find((r: any) => r.from?.endsWith("." + colName)) ||
                          inferredRels.find((r: any) => r.from?.endsWith("." + colName));
                        const nullPct = currentTable?.null_stats?.[colName]?.null_pct ?? 0;
                        const relInfo = fkInfo ? relFor(fkInfo.from, fkInfo.to) : null;
                        return (
                          <tr key={colName}
                            className={`${i < currentCols.length - 1 ? "border-b border-border/50" : ""} hover:bg-secondary/30 transition-colors cursor-pointer`}
                            onClick={() => setShowColumnDetail(showColumnDetail === colName ? null : colName)}>
                            <td className="px-4 py-2 font-mono text-xs text-foreground font-medium">{colName}</td>
                            <td className="px-4 py-2 font-mono text-xs text-primary">{col.data_type || "?"}</td>
                            <td className="px-2 py-2 text-center">
                              {isPk ? <Key size={11} className="text-positive inline" /> : <Minus size={11} className="text-muted-foreground/30 inline" />}
                            </td>
                            <td className="px-2 py-2 text-center">
                              {fkInfo ? <Link size={11} className="text-blue-500 dark:text-blue-400 inline" /> : <Minus size={11} className="text-muted-foreground/30 inline" />}
                            </td>
                            <td className="px-2 py-2 text-center text-[10px] text-muted-foreground">{col.nullable !== false ? "Yes" : "No"}</td>
                            <td className="px-2 py-2 text-center">
                              {currentTable?.unique_keys?.includes(colName) ? <Check size={11} className="text-positive inline" /> : <Minus size={11} className="text-muted-foreground/30 inline" />}
                            </td>
                            <td className="px-2 py-2 text-[10px] text-muted-foreground">{nullPct > 0 ? `${nullPct.toFixed(0)}%` : "0%"}</td>
                            <td className="px-2 py-2">
                              <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                                {!isPk && pkList.length === 0 && (
                                  <button onClick={() => handleOverride("set_pk", selectedTable, colName)}
                                    className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary hover:bg-primary/20 transition-colors flex items-center gap-1">
                                    <Key size={9} /> Set PK
                                  </button>
                                )}
                                {isPk && (
                                  <button onClick={() => handleOverride("remove_pk", selectedTable, colName)}
                                    className="text-[10px] px-1.5 py-0.5 rounded bg-negative/10 text-negative hover:bg-negative/20 transition-colors flex items-center gap-1">
                                    <Key size={9} /> Remove PK
                                  </button>
                                )}
                                {!isPk && !fkInfo && (
                                  <button onClick={() => { setFkSourceTable(selectedTable); setFkSourceCol(colName); setShowFKModal(true); }}
                                    className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-600 dark:text-blue-400 hover:bg-blue-500/20 transition-colors flex items-center gap-1">
                                    <Link size={9} /> Set FK
                                  </button>
                                )}
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                {/* Column detail panel */}
                {showColumnDetail && (
                  <div className="border-t border-border px-4 py-3 bg-secondary/30 animate-fade-in">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-xs font-semibold text-foreground flex items-center gap-2">
                        <Info size={12} /> {showColumnDetail} — Column Detail
                      </p>
                      <button onClick={() => setShowColumnDetail(null)} className="text-muted-foreground hover:text-foreground">
                        <X size={12} />
                      </button>
                    </div>
                    {(() => {
                      const col = currentCols.find((c: any) => c.column === showColumnDetail);
                      if (!col) return null;
                      const ns = currentTable?.null_stats?.[showColumnDetail] || {};
                      return (
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-[11px]">
                          <div><span className="text-muted-foreground">Type:</span> <span className="font-mono text-foreground">{col.data_type}</span></div>
                          <div><span className="text-muted-foreground">Nullable:</span> <span className="text-foreground">{col.nullable !== false ? "Yes" : "No"}</span></div>
                          <div><span className="text-muted-foreground">Nulls:</span> <span className="text-foreground">{ns.nulls ?? 0} ({ns.null_pct?.toFixed(0) ?? 0}%)</span></div>
                          <div><span className="text-muted-foreground">Total:</span> <span className="text-foreground">{ns.total_rows ?? "?"}</span></div>
                          {col.is_primary_key && <div><span className="text-muted-foreground">Role:</span> <span className="text-foreground flex items-center gap-1"><Key size={10} className="text-positive" /> Primary Key</span></div>}
                          {col.is_foreign_key && col.references && <div><span className="text-muted-foreground">Role:</span> <span className="text-foreground flex items-center gap-1"><Link size={10} className="text-blue-500 dark:text-blue-400" /> FK → {col.references.table}.{col.references.column}</span></div>}
                          {col.distinct_count != null && <div><span className="text-muted-foreground">Distinct:</span> <span className="text-foreground">{col.distinct_count}</span></div>}
                          {col.min != null && <div><span className="text-muted-foreground">Min:</span> <span className="text-foreground">{col.min}</span></div>}
                          {col.max != null && <div><span className="text-muted-foreground">Max:</span> <span className="text-foreground">{col.max}</span></div>}
                          {col.avg != null && <div><span className="text-muted-foreground">Avg:</span> <span className="text-foreground">{col.avg}</span></div>}
                          {col.sample_values?.length > 0 && (
                            <div className="col-span-2 sm:col-span-4"><span className="text-muted-foreground">Samples:</span> <span className="font-mono text-foreground">{col.sample_values.slice(0, 5).join(", ")}</span></div>
                          )}
                        </div>
                      );
                    })()}
                  </div>
                )}

                {/* Relationships */}
                {(declaredRels.length > 0 || inferredRels.length > 0) && (
                  <div className="border-t border-border px-4 py-3">
                    <p className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider mb-2">Relationships</p>
                    <div className="space-y-1.5">
                      {declaredRels.map((r: any, i: number) => (
                        <div key={`decl-${i}`} className="flex items-center gap-2 text-[11px]">
                          <Link size={10} className="text-positive flex-shrink-0" />
                          <span className="font-mono text-foreground">{r.from}</span>
                          <span className="text-muted-foreground">→</span>
                          <span className="font-mono text-foreground">{r.to}</span>
                          <span className="text-[9px] px-1 py-0.5 rounded bg-positive/10 text-positive font-medium">DB</span>
                        </div>
                      ))}
                      {inferredRels.map((r: any, i: number) => {
                        const info = relFor(r.from, r.to);
                        return (
                          <div key={`inf-${i}`} className="flex items-center gap-2 text-[11px]">
                            <GitBranch size={10} className="text-amber-500 dark:text-amber-400 flex-shrink-0" />
                            <span className="font-mono text-foreground">{r.from}</span>
                            <span className="text-muted-foreground">→</span>
                            <span className="font-mono text-foreground">{r.to}</span>
                            {r.confidence && <span className="text-[9px] text-muted-foreground">{r.confidence}</span>}
                            {info.rejected ? (
                              <span className="text-[9px] px-1 py-0.5 rounded bg-negative/10 text-negative font-medium">Rejected</span>
                            ) : info.confirmed ? (
                              <span className="text-[9px] px-1 py-0.5 rounded bg-positive/10 text-positive font-medium">Confirmed</span>
                            ) : (
                              <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                                <button onClick={() => handleOverride("confirm_relationship", selectedTable, r.from?.split(".")?.[1], r.to?.split(".")?.[0], r.to?.split(".")?.[1])}
                                  className="text-[9px] px-1 py-0.5 rounded bg-positive/10 text-positive hover:bg-positive/20 transition-colors">Confirm</button>
                                <button onClick={() => handleOverride("reject_relationship", selectedTable, r.from?.split(".")?.[1], r.to?.split(".")?.[0], r.to?.split(".")?.[1])}
                                  className="text-[9px] px-1 py-0.5 rounded bg-negative/10 text-negative hover:bg-negative/20 transition-colors">Reject</button>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  // ── MANUAL FK MODAL ──
  const [showFKModal, setShowFKModal] = useState(false);
  const [fkSourceTable, setFkSourceTable] = useState("");
  const [fkSourceCol, setFkSourceCol] = useState("");
  const [fkTargetTable, setFkTargetTable] = useState("");
  const [fkTargetCol, setFkTargetCol] = useState("");

  const handleCreateFK = async () => {
    if (!fkSourceTable || !fkSourceCol || !fkTargetTable || !fkTargetCol) return;
    await handleOverride("set_fk", fkSourceTable, fkSourceCol, fkTargetTable, fkTargetCol);
    setShowFKModal(false);
    setFkSourceTable(""); setFkSourceCol(""); setFkTargetTable(""); setFkTargetCol("");
  };

  const FKModal = () => {
    if (!showFKModal) return null;
    const srcCols = fkSourceTable ? (tables[fkSourceTable]?.columns || []).map((c: any) => c.column) : [];
    const tgtCols = fkTargetTable ? (tables[fkTargetTable]?.columns || []).map((c: any) => c.column) : [];
    return (
      <div className="fixed inset-0 z-[60] bg-black/50 flex items-center justify-center animate-fade-in" onClick={() => setShowFKModal(false)}>
        <div className="bg-card border border-border rounded-xl p-5 w-full max-w-md shadow-xl" onClick={(e) => e.stopPropagation()}>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-foreground flex items-center gap-2"><Link size={14} /> Create Relationship</h3>
            <button onClick={() => setShowFKModal(false)} className="text-muted-foreground hover:text-foreground"><X size={14} /></button>
          </div>
          <div className="space-y-3">
            <div>
              <label className="text-[11px] font-medium text-muted-foreground">Source Table</label>
              <select value={fkSourceTable} onChange={(e) => { setFkSourceTable(e.target.value); setFkSourceCol(""); }}
                className="w-full bg-secondary border border-border rounded-lg px-3 py-2 text-xs text-foreground outline-none focus:border-primary mt-1">
                <option value="">Select source table</option>
                {tableNames.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </div>
            <div>
              <label className="text-[11px] font-medium text-muted-foreground">Source Column</label>
              <select value={fkSourceCol} onChange={(e) => setFkSourceCol(e.target.value)}
                className="w-full bg-secondary border border-border rounded-lg px-3 py-2 text-xs text-foreground outline-none focus:border-primary mt-1" disabled={!fkSourceTable}>
                <option value="">Select source column</option>
                {srcCols.map((c: string) => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <div className="flex items-center justify-center text-muted-foreground"><Link size={14} className="rotate-90" /></div>
            <div>
              <label className="text-[11px] font-medium text-muted-foreground">Target Table</label>
              <select value={fkTargetTable} onChange={(e) => { setFkTargetTable(e.target.value); setFkTargetCol(""); }}
                className="w-full bg-secondary border border-border rounded-lg px-3 py-2 text-xs text-foreground outline-none focus:border-primary mt-1">
                <option value="">Select target table</option>
                {tableNames.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </div>
            <div>
              <label className="text-[11px] font-medium text-muted-foreground">Target Column</label>
              <select value={fkTargetCol} onChange={(e) => setFkTargetCol(e.target.value)}
                className="w-full bg-secondary border border-border rounded-lg px-3 py-2 text-xs text-foreground outline-none focus:border-primary mt-1" disabled={!fkTargetTable}>
                <option value="">Select target column</option>
                {tgtCols.map((c: string) => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
          </div>
          <div className="flex justify-end gap-2 mt-5">
            <button onClick={() => setShowFKModal(false)} className="text-xs px-3 py-1.5 rounded-lg bg-secondary text-muted-foreground hover:text-foreground transition-colors">Cancel</button>
            <button onClick={handleCreateFK} disabled={!fkSourceTable || !fkSourceCol || !fkTargetTable || !fkTargetCol}
              className="text-xs px-3 py-1.5 rounded-lg bg-primary text-white font-medium disabled:opacity-40 transition-colors">
              Create Relationship
            </button>
          </div>
        </div>
      </div>
    );
  };

  // ── ERD VIEW ──
  const ERDView = () => {
    const [zoom, setZoom] = useState(1);
    const [pan, setPan] = useState({ x: 0, y: 0 });
    const [erdSearch, setErdSearch] = useState("");
    const [dragging, setDragging] = useState(false);
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
    const svgRef = useRef<SVGSVGElement>(null);

    const BOX_W = 220;
    const HEADER_H = 28;
    const ROW_H = 18;
    const MAX_ROWS = 10;

    const tableList = tableNames.map((name, i) => ({
      name,
      x: 80 + (i % 4) * (BOX_W + 40),
      y: 60 + Math.floor(i / 4) * 300,
      cols: tables[name]?.columns || [],
      pk: overrides?.pk?.[name] || tables[name]?.primary_key || tables[name]?.inferred_primary_key || [],
    }));

    const allRels = [
      ...(schemaData?.declared_fk_list || []).map((r: any) => ({ ...r, source: "db" })),
      ...(schemaData?.inferred_fk_list || []).map((r: any) => ({ ...r, source: "aria" })),
    ];

    const getTableCenter = (name: string) => {
      const t = tableList.find((t) => t.name === name);
      if (!t) return { x: 0, y: 0 };
      return { x: t.x + BOX_W / 2, y: t.y + HEADER_H + Math.min(t.cols.length, MAX_ROWS) * ROW_H / 2 };
    };

    const handleMouseDown = (e: React.MouseEvent) => {
      if (e.target === svgRef.current || (e.target as Element)?.tagName === "rect") {
        setDragging(true);
        setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
      }
    };
    const handleMouseMove = (e: React.MouseEvent) => {
      if (dragging) setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    };
    const handleMouseUp = () => setDragging(false);

    const C = {
      card: "var(--card)",
      border: "var(--border)",
      fg: "var(--foreground)",
      muted: "var(--muted-foreground)",
      secondary: "var(--secondary)",
      primary: "#2F6FAD",
      positive: "#059669",
      amber: "#F59E0B",
      negative: "#DC2626",
    };

    return (
      <div className="fixed inset-0 z-50 bg-card flex flex-col animate-fade-in">
        <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-card shrink-0">
          <div className="flex items-center gap-3">
            <button onClick={() => setShowERD(false)}
              className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground px-2.5 py-1.5 rounded-lg hover:bg-secondary transition-all">
              <ChevronLeft size={14} /> Back to Schema
            </button>
            <div className="w-px h-5 bg-border" />
            <h3 className="text-sm font-semibold text-foreground">Entity Relationship Diagram</h3>
            <span className="text-[10px] text-muted-foreground">{tableList.length} tables · {allRels.length} relationships</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground" />
              <input value={erdSearch} onChange={(e) => setErdSearch(e.target.value)}
                placeholder="Search tables..."
                className="bg-secondary border border-border rounded-lg pl-7 pr-3 py-1.5 text-[11px] text-foreground placeholder:text-muted-foreground outline-none focus:border-primary w-40" />
            </div>
            <button onClick={() => setZoom((z) => Math.min(z + 0.15, 2))}
              className="p-1.5 rounded-lg hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors" title="Zoom in"><Maximize2 size={13} /></button>
            <button onClick={() => setZoom((z) => Math.max(z - 0.15, 0.3))}
              className="p-1.5 rounded-lg hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors" title="Zoom out"><Minimize2 size={13} /></button>
            <button onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}
              className="text-[10px] px-2 py-1 rounded-lg hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors">Fit</button>
            <button onClick={() => setShowERD(false)}
              className="p-1.5 rounded-lg hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors" title="Close"><X size={14} /></button>
          </div>
        </div>
        {/* Legend */}
        <div className="flex items-center gap-4 px-4 py-2 border-b border-border/50 bg-card/50 shrink-0">
          <span className="text-[10px] text-muted-foreground font-medium">Legend:</span>
          <span className="flex items-center gap-1 text-[10px]"><span className="w-3 h-0.5 rounded" style={{ background: C.positive }} /> DB FK</span>
          <span className="flex items-center gap-1 text-[10px]"><span className="w-3 h-0.5 rounded border border-dashed" style={{ borderColor: C.amber }} /> ARIA Inferred</span>
          <span className="flex items-center gap-1 text-[10px]"><span className="w-3 h-0.5 rounded" style={{ background: C.negative, opacity: 0.3 }} /> Rejected</span>
          <span className="flex items-center gap-1 text-[10px]"><span className="w-2.5 h-2.5 rounded" style={{ background: C.primary }} /> PK</span>
        </div>
        <div className="flex-1 overflow-hidden relative"
          style={{ cursor: dragging ? "grabbing" : "grab" }}
          onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}>
          <svg ref={svgRef} width="100%" height="100%"
            style={{ transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`, transformOrigin: "0 0" }}>
            <defs>
              <marker id="arrow" viewBox="0 0 10 6" refX="10" refY="3" markerWidth="8" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 3 L 0 6 z" fill={C.muted} />
              </marker>
            </defs>
            {/* Relationship lines */}
            {allRels.map((rel: any, i: number) => {
              const fromName = rel.from?.split(".")[0];
              const toName = rel.to?.split(".")[0];
              const from = getTableCenter(fromName);
              const to = getTableCenter(toName);
              if (!from.x && !from.y) return null;
              if (!to.x && !to.y) return null;
              const isRejected = overrides?.rejected?.includes(`${rel.from}->${rel.to}`);
              const isConfirmed = overrides?.fk?.[`${rel.from}->${rel.to}`];
              const color = rel.source === "db" ? C.positive : isRejected ? C.negative : isConfirmed ? C.positive : C.amber;
              const opacity = isRejected ? 0.25 : 0.6;
              const dash = rel.source === "aria" && !isRejected && !isConfirmed ? "5 4" : undefined;
              const dx = to.x - from.x;
              const dy = to.y - from.y;
              const cx1 = from.x + dx * 0.1;
              const cy1 = from.y;
              const cx2 = to.x - dx * 0.1;
              const cy2 = to.y;
              return (
                <g key={i} opacity={opacity}>
                  <path d={`M ${from.x} ${from.y} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${to.x} ${to.y}`}
                    stroke={color} strokeWidth={1.5} fill="none" strokeDasharray={dash}
                    markerEnd="url(#arrow)" />
                </g>
              );
            })}
            {/* Table boxes */}
            {tableList.map((t) => {
              const isHighlighted = erdSearch && t.name.toLowerCase().includes(erdSearch.toLowerCase());
              const visibleCols = t.cols.slice(0, MAX_ROWS);
              const colHeight = visibleCols.length * ROW_H;
              const totalH = HEADER_H + colHeight + (t.cols.length > MAX_ROWS ? 16 : 4);
              return (
                <g key={t.name} transform={`translate(${t.x}, ${t.y})`}>
                  {/* Shadow */}
                  <rect width={BOX_W} height={totalH} rx={8} y={2} fill="rgba(0,0,0,0.06)" />
                  {/* Body */}
                  <rect width={BOX_W} height={totalH} rx={8} fill={isHighlighted ? "var(--primary-light)" : C.card}
                    stroke={isHighlighted ? C.primary : C.border} strokeWidth={isHighlighted ? 1.5 : 1} />
                  {/* Header */}
                  <rect width={BOX_W} height={HEADER_H} rx={8} fill={C.secondary} />
                  <rect width={BOX_W} height={HEADER_H / 2} fill={C.secondary} />
                  <text x={10} y={19} fontSize={11} fontWeight={700} fontFamily="Inter, sans-serif" fill={C.fg}>{t.name}</text>
                  <text x={BOX_W - 10} y={19} fontSize={9} textAnchor="end" fontFamily="Inter, sans-serif" fill={C.muted}>{t.cols.length} cols</text>
                  {/* Columns */}
                  {visibleCols.map((col: any, ci: number) => {
                    const isPk = t.pk.includes(col.column);
                    const y = HEADER_H + 12 + ci * ROW_H;
                    return (
                      <g key={col.column}>
                        {isPk && <rect x={6} y={y - 8} width={14} height={12} rx={3} fill={C.primary} opacity={0.12} />}
                        <text x={10} y={y} fontSize={10} fontFamily="monospace" fill={isPk ? C.primary : C.fg} fontWeight={isPk ? 600 : 400}>
                          {isPk ? "PK " : ""}{col.column}
                        </text>
                        <text x={BOX_W - 10} y={y} fontSize={9} textAnchor="end" fontFamily="monospace" fill={C.muted}>
                          {(col.data_type || "").slice(0, 10)}
                        </text>
                      </g>
                    );
                  })}
                  {t.cols.length > MAX_ROWS && (
                    <text x={BOX_W / 2} y={totalH - 5} fontSize={9} textAnchor="middle" fontFamily="Inter, sans-serif" fill={C.muted}>
                      +{t.cols.length - MAX_ROWS} more
                    </text>
                  )}
                </g>
              );
            })}
          </svg>
        </div>
      </div>
    );
  };

  // ── PREVIEW TAB ──
  const PreviewTab = () => {
    const [previewTable, setPreviewTable] = useState("");
    const [rows, setRows] = useState<any[]>([]);
    const [columns, setColumns] = useState<string[]>([]);
    const [total, setTotal] = useState(0);
    const [page, setPage] = useState(0);
    const [limit] = useState(50);
    const [search, setSearch] = useState("");
    const [sortCol, setSortCol] = useState("");
    const [sortDir, setSortDir] = useState<"ASC" | "DESC">("ASC");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [inspectCol, setInspectCol] = useState<string | null>(null);

    const fetchPreview = useCallback(async (tbl: string, off: number, s: string, sc: string, sd: string) => {
      if (!tbl) return;
      setLoading(true); setError("");
      try {
        const r = await api("/api/data/preview", {
          method: "POST",
          body: JSON.stringify({ table: tbl, limit, offset: off, search: s, sort_column: sc, sort_direction: sd }),
        });
        setRows(r.rows || []);
        setColumns(r.columns || []);
        setTotal(r.total || 0);
      } catch (e: any) {
        setError(e.message || "Preview failed");
        setRows([]); setColumns([]);
      } finally {
        setLoading(false);
      }
    }, [limit]);

    useEffect(() => {
      if (previewTable) fetchPreview(previewTable, page * limit, search, sortCol, sortDir);
    }, [previewTable, page, sortCol, sortDir]);

    const handleSearch = () => { setPage(0); fetchPreview(previewTable, 0, search, sortCol, sortDir); };
    const toggleSort = (col: string) => {
      if (sortCol === col) { setSortDir((d) => d === "ASC" ? "DESC" : "ASC"); }
      else { setSortCol(col); setSortDir("ASC"); setPage(0); }
    };
    const totalPages = Math.ceil(total / limit);

    return (
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <select value={previewTable} onChange={(e) => { setPreviewTable(e.target.value); setPage(0); setSortCol(""); setSearch(""); }}
            className="bg-secondary border border-border rounded-lg px-3 py-2 text-xs text-foreground outline-none focus:border-primary min-w-[180px]">
            <option value="">Select a table</option>
            {tableNames.map((n) => <option key={n} value={n}>{n} ({tables[n]?.columns?.length ?? "?"} cols)</option>)}
          </select>
          {previewTable && (
            <>
              <div className="flex-1 relative">
                <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
                <input value={search} onChange={(e) => setSearch(e.target.value)} onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                  placeholder="Search in table..."
                  className="w-full bg-secondary border border-border rounded-lg pl-8 pr-3 py-2 text-xs text-foreground placeholder:text-muted-foreground outline-none focus:border-primary" />
              </div>
              <button onClick={handleSearch}
                className="text-xs px-3 py-2 rounded-lg bg-primary/10 text-primary hover:bg-primary/20 transition-colors font-medium">Search</button>
              <span className="text-[10px] text-muted-foreground">{total} rows</span>
            </>
          )}
        </div>

        {!previewTable && (
          <div className="bg-card border border-border rounded-xl p-8 text-center">
            <Eye size={24} className="mx-auto text-muted-foreground mb-3" />
            <p className="text-sm font-medium text-foreground">Data Preview</p>
            <p className="text-xs text-muted-foreground mt-1">Select a table to preview actual records.</p>
          </div>
        )}

        {previewTable && (
          <div className="bg-card border border-border rounded-xl overflow-hidden">
            {loading ? (
              <div className="p-8 text-center"><Loader2 size={20} className="mx-auto animate-spin text-primary mb-2" /><p className="text-xs text-muted-foreground">Loading...</p></div>
            ) : error ? (
              <div className="p-8 text-center"><AlertTriangle size={20} className="mx-auto text-negative mb-2" /><p className="text-xs text-negative">{error}</p></div>
            ) : rows.length === 0 ? (
              <div className="p-8 text-center"><Table size={20} className="mx-auto text-muted-foreground mb-2" /><p className="text-xs text-muted-foreground">No rows found.</p></div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border">
                      {columns.map((col) => (
                        <th key={col} onClick={() => toggleSort(col)}
                          className="text-left text-[11px] font-medium text-muted-foreground px-3 py-2.5 cursor-pointer hover:text-foreground transition-colors whitespace-nowrap">
                          <span className="flex items-center gap-1">
                            {col}
                            {sortCol === col ? (sortDir === "ASC" ? <ArrowUp size={10} /> : <ArrowDown size={10} />) : <ArrowUpDown size={10} className="opacity-30" />}
                          </span>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row, ri) => (
                      <tr key={ri} className="border-b border-border/50 last:border-0 hover:bg-secondary/30 transition-colors">
                        {columns.map((col) => (
                          <td key={col} className="px-3 py-2 text-xs font-mono text-foreground truncate max-w-[200px]"
                            onClick={() => setInspectCol(inspectCol === col ? null : col)}>
                            {row[col] != null ? String(row[col]) : <span className="text-muted-foreground/40">NULL</span>}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-between px-4 py-2.5 border-t border-border">
                <span className="text-[10px] text-muted-foreground">Page {page + 1} of {totalPages}</span>
                <div className="flex items-center gap-1.5">
                  <button onClick={() => setPage(0)} disabled={page === 0}
                    className="text-[10px] px-2 py-1 rounded bg-secondary text-muted-foreground hover:text-foreground disabled:opacity-40 transition-colors">First</button>
                  <button onClick={() => setPage((p) => Math.max(0, p - 1))} disabled={page === 0}
                    className="text-[10px] px-2 py-1 rounded bg-secondary text-muted-foreground hover:text-foreground disabled:opacity-40 transition-colors">Prev</button>
                  <button onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))} disabled={page >= totalPages - 1}
                    className="text-[10px] px-2 py-1 rounded bg-secondary text-muted-foreground hover:text-foreground disabled:opacity-40 transition-colors">Next</button>
                  <button onClick={() => setPage(totalPages - 1)} disabled={page >= totalPages - 1}
                    className="text-[10px] px-2 py-1 rounded bg-secondary text-muted-foreground hover:text-foreground disabled:opacity-40 transition-colors">Last</button>
                </div>
              </div>
            )}
            {/* Column inspection overlay */}
            {inspectCol && (
              <div className="border-t border-border px-4 py-3 bg-secondary/30 animate-fade-in">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-xs font-semibold text-foreground flex items-center gap-2">
                    <Info size={12} /> {inspectCol} — Column Profile
                  </p>
                  <button onClick={() => setInspectCol(null)} className="text-muted-foreground hover:text-foreground"><X size={12} /></button>
                </div>
                <div className="text-[11px] text-muted-foreground">
                  Click any column header to sort. Column profiling shows type, null %, distinct count, min/max/avg for numeric types.
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  // ── QUALITY TAB ──
  const QualityTab = () => {
    const [quality, setQuality] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [expandedIssue, setExpandedIssue] = useState<number | null>(null);

    const runScan = async () => {
      setLoading(true); setError("");
      try {
        const r = await apiGet("/api/data/quality");
        setQuality(r.quality || null);
      } catch (e: any) {
        setError(e.message || "Quality scan failed");
      } finally {
        setLoading(false);
      }
    };

    useEffect(() => {
      if (isConnected) runScan();
    }, [isConnected, schemaStatus]);

    const issues = quality?.issues || [];
    const metrics = quality?.metrics || {};
    const overall = quality?.overall ?? 0;

    const severityColor = (s: string) => s === "high" ? "text-negative bg-negative/10" : s === "medium" ? "text-amber-600 dark:text-amber-400 bg-amber-500/10" : "text-muted-foreground bg-secondary";

    return (
      <div className="space-y-5">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-foreground">Data Quality</h3>
            <p className="text-[11px] text-muted-foreground mt-0.5">
              {quality?.last_scan ? `Last scanned: ${new Date(quality.last_scan).toLocaleString()}` : "No scans yet"}
            </p>
          </div>
          <button onClick={runScan} disabled={loading || !isConnected}
            className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground px-3 py-2 rounded-lg bg-secondary hover:bg-secondary/80 transition-all disabled:opacity-50">
            {loading ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
            {loading ? "Scanning..." : "Run Quality Scan"}
          </button>
        </div>

        {!isConnected ? (
          <div className="bg-card border border-border rounded-xl p-8 text-center">
            <Shield size={24} className="mx-auto text-muted-foreground mb-3" />
            <p className="text-sm font-medium text-foreground">Data Quality</p>
            <p className="text-xs text-muted-foreground mt-1">Connect a data source to run quality checks.</p>
          </div>
        ) : loading && !quality ? (
          <div className="bg-card border border-border rounded-xl p-8 text-center">
            <Loader2 size={24} className="mx-auto animate-spin text-primary mb-3" />
            <p className="text-xs text-muted-foreground">Scanning tables...</p>
            <p className="text-[10px] text-muted-foreground mt-1">Checking null values, duplicates, relationships...</p>
          </div>
        ) : error ? (
          <div className="bg-card border border-border rounded-xl p-8 text-center">
            <AlertTriangle size={24} className="mx-auto text-negative mb-3" />
            <p className="text-xs text-negative">{error}</p>
          </div>
        ) : quality ? (
          <>
            {/* Overall score */}
            <div className="premium-card rounded-xl p-5 bg-card border border-border">
              <div className="flex items-center gap-4">
                <div className="w-16 h-16 rounded-xl flex items-center justify-center relative"
                  style={{ background: overall >= 80 ? "rgba(34,197,94,0.1)" : overall >= 50 ? "rgba(245,158,11,0.1)" : "rgba(239,68,68,0.1)" }}>
                  <span className="text-2xl font-bold" style={{ color: overall >= 80 ? "#22c55e" : overall >= 50 ? "#f59e0b" : "#ef4444" }}>
                    {overall}
                  </span>
                  <span className="text-[9px] text-muted-foreground absolute -bottom-0.5">/100</span>
                </div>
                <div>
                  <p className="text-sm font-semibold text-foreground">Overall Quality Score</p>
                  <p className="text-[11px] text-muted-foreground mt-0.5">
                    {overall >= 80 ? "Good quality — data is reliable" : overall >= 50 ? "Moderate quality — review issues below" : "Low quality — significant issues detected"}
                  </p>
                </div>
              </div>
              <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-3">
                <MetricCard label="Completeness" value={metrics.completeness} suffix="%" />
                <MetricCard label="PK Coverage" value={metrics.pk_coverage} suffix="%" />
                <MetricCard label="Avg Null %" value={metrics.avg_null_percentage} suffix="%" invert />
                <MetricCard label="Empty Tables" value={metrics.empty_tables} suffix={` / ${metrics.total_tables}`} />
              </div>
            </div>

            {/* Issues list */}
            <div>
              <div className="flex items-center justify-between mb-3">
                <p className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">
                  Issues ({issues.length})
                </p>
                {issues.length > 0 && (
                  <div className="flex items-center gap-2">
                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-negative/10 text-negative font-medium">
                      {issues.filter((i: any) => i.severity === "high").length} high
                    </span>
                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-600 dark:text-amber-400 font-medium">
                      {issues.filter((i: any) => i.severity === "medium").length} medium
                    </span>
                  </div>
                )}
              </div>
              <div className="space-y-2">
                {issues.map((issue: any, i: number) => (
                  <div key={i}
                    className={`bg-card border rounded-xl overflow-hidden transition-all cursor-pointer
                      ${issue.severity === "high" ? "border-negative/20" : issue.severity === "medium" ? "border-amber-500/20" : "border-border"}`}
                    onClick={() => setExpandedIssue(expandedIssue === i ? null : i)}>
                    <div className="px-4 py-3 flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        {issue.severity === "high" ? <AlertCircle size={14} className="text-negative flex-shrink-0" /> :
                          issue.severity === "medium" ? <AlertTriangle size={14} className="text-amber-500 dark:text-amber-400 flex-shrink-0" /> :
                            <Info size={14} className="text-muted-foreground flex-shrink-0" />}
                        <div>
                          <p className="text-xs font-medium text-foreground">
                            {issue.table}{issue.column ? `.${issue.column}` : ""}
                          </p>
                          <p className="text-[11px] text-muted-foreground">{issue.message}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className={`text-[9px] px-1.5 py-0.5 rounded font-medium ${severityColor(issue.severity)}`}>
                          {issue.severity}
                        </span>
                        <ChevronDown size={12} className={`text-muted-foreground transition-transform ${expandedIssue === i ? "rotate-180" : ""}`} />
                      </div>
                    </div>
                    {expandedIssue === i && (
                      <div className="px-4 py-3 border-t border-border bg-secondary/20 animate-fade-in">
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-[11px]">
                          <div><span className="text-muted-foreground">Check:</span> <span className="text-foreground">{issue.check}</span></div>
                          <div><span className="text-muted-foreground">Affected:</span> <span className="text-foreground">{issue.affected_rows} rows</span></div>
                          <div><span className="text-muted-foreground">Percentage:</span> <span className="text-foreground">{issue.percentage?.toFixed(1) ?? 0}%</span></div>
                          <div><span className="text-muted-foreground">Severity:</span> <span className={`font-medium ${severityColor(issue.severity)?.split(" ")[0]}`}>{issue.severity}</span></div>
                        </div>
                        {issue.details && (
                          <div className="mt-2 text-[11px]">
                            <span className="text-muted-foreground">References:</span>{" "}
                            <span className="font-mono text-foreground">{issue.details.references_table}.{issue.details.references_column}</span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
                {issues.length === 0 && (
                  <div className="bg-card border border-border rounded-xl p-8 text-center">
                    <CheckCircle size={24} className="mx-auto text-positive mb-3" />
                    <p className="text-sm font-medium text-foreground">No issues detected</p>
                    <p className="text-xs text-muted-foreground mt-1">All quality checks passed.</p>
                  </div>
                )}
              </div>
            </div>
          </>
        ) : null}
      </div>
    );
  };

  return (
    <div className="p-6 max-w-screen-xl mx-auto space-y-5">
      {showERD && <ERDView />}
      <FKModal />

      <div className="animate-fade-in">
        <h2 className="text-base font-bold text-foreground tracking-tight">Data</h2>
        <p className="text-xs text-muted-foreground mt-0.5">
          Inspect your database structure, preview data, and monitor quality.
        </p>
      </div>

      <div className="flex gap-1 bg-secondary rounded-lg p-0.5 w-fit">
        {TABS.map((t) => {
          const Icon = t.icon;
          return (
            <button key={t.id} onClick={() => setActiveTab(t.id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors
                ${activeTab === t.id ? "bg-card text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}>
              <Icon size={12} /> {t.label}
            </button>
          );
        })}
      </div>

      {activeTab === "sources" && sourcesTabJSX}
      {activeTab === "schema" && <SchemaTab />}
      {activeTab === "preview" && <PreviewTab />}
      {activeTab === "quality" && <QualityTab />}
    </div>
  );
}

function Field({ label, id, type, value, onChange, placeholder, disabled, children }: {
  label: string; id: string; type?: string; value: string;
  onChange: (v: string) => void; placeholder?: string;
  disabled?: boolean; children?: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <label htmlFor={id} className="text-[11px] font-medium text-muted-foreground">{label}</label>
      <div className="relative">
        <input id={id} type={type === "password-show" ? "text" : type} value={value}
          onChange={(e) => onChange(e.target.value)} placeholder={placeholder} disabled={disabled}
          className={`w-full bg-secondary border border-border rounded-lg px-3 py-2 text-sm text-foreground
            placeholder:text-muted-foreground outline-none focus:border-primary focus:shadow-[0_0_0_3px_var(--primary-light)] transition-all duration-150
            ${disabled ? "opacity-50 cursor-not-allowed" : ""}`} />
        {children}
      </div>
    </div>
  );
}

function MetricCard({ label, value, suffix = "", invert = false }: { label: string; value?: number; suffix?: string; invert?: boolean }) {
  const v = value ?? 0;
  const color = invert
    ? (v <= 20 ? "text-positive" : v <= 50 ? "text-amber-600 dark:text-amber-400" : "text-negative")
    : (v >= 80 ? "text-positive" : v >= 50 ? "text-amber-600 dark:text-amber-400" : "text-negative");
  return (
    <div className="bg-secondary/50 rounded-lg px-3 py-2">
      <p className="text-[10px] text-muted-foreground">{label}</p>
      <p className={`text-lg font-bold ${color}`}>{v}{suffix}</p>
    </div>
  );
}
