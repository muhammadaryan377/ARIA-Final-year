import { createContext, useContext, useState, useEffect, useRef, lazy, Suspense } from "react";
import type { ReactNode } from "react";
import "./index.css";
import "../aria-enhance.css";
import AppShell from "./components/AppShell";
import ErrorBoundary from "./components/ErrorBoundary";
import MiniAria from "./components/MiniAria";
import Login from "./pages/Login";
// Route pages are code-split: each hash route loads its own chunk on
// first visit (vendor/charts/motion stay in shared manualChunks — see
// vite.config.js). The hash router itself is untouched.
const LLMSelection = lazy(() => import("./pages/LLMSelection"));
const Dashboard = lazy(() => import("./pages/Dashboard"));
const AIAnalytics = lazy(() => import("./pages/AIAnalytics"));
const Insights = lazy(() => import("./pages/Insights"));
const Predictive = lazy(() => import("./pages/Predictive"));
const Intelligence = lazy(() => import("./pages/Intelligence"));
const Data = lazy(() => import("./pages/Data"));
const Copilot = lazy(() => import("./pages/Copilot"));
const Documents = lazy(() => import("./pages/Documents"));
const History = lazy(() => import("./pages/History"));
const Settings = lazy(() => import("./pages/Settings"));
function PageFallback() {
  return (
    <div className="p-6 max-w-screen-xl mx-auto" aria-busy="true" aria-label="Loading page">
      <div className="rounded-2xl border border-border bg-card p-6 animate-pulse" aria-hidden>
        <div className="h-4 w-40 rounded bg-secondary" />
        <div className="mt-3 h-6 w-2/3 rounded bg-secondary" />
        <div className="mt-4 h-40 rounded bg-secondary" />
      </div>
    </div>
  );
}
import {
  connectDatabase,
  fetchSchema,
  analyzeQuestion,
  generateAnalyticsResult,
  fetchSessionStatus,
  disconnectSource,
  apiFetch,
} from "../services/api";
import { clearAuthSession, getAuthUser } from "../utils/authStorage";
import { KEYS, storage } from "../utils/storage";
import { ToastProvider } from "../components/Toast";

export type Page =
  | "dashboard"
  | "analytics"
  | "insights"
  | "predictive"
  | "intelligence"
  | "data"
  | "documents"
  | "copilot"
  | "history"
  | "settings";

const VALID_PAGES: Page[] = ["dashboard","analytics","insights","predictive","intelligence","data","documents","copilot","history","settings"];

// Legacy aliases: "hypothesis" -> AI Analytics (the standalone Hypothesis
// page was removed; its capability surfaces as Possible drivers inside
// AI Analytics / Insights / Forecast). Old #/hypothesis hashes land on
// analytics harmlessly.
const normalizePage = (h: string): Page | null => {
  if (h === "hypothesis") return "analytics";
  return (VALID_PAGES as string[]).includes(h) ? (h as Page) : null;
};

const parseHash = (hash: string): Page | null => {
  const h = (hash || "").replace(/^#\/?/, "");
  return normalizePage(h.split("?")[0]);
};

type AppStage = "login" | "llm-selection" | "app";

interface ThemeCtx {
  isDark: boolean;
  toggleTheme: () => void;
  page: Page;
  setPage: (p: Page) => void;
}

export const ThemeContext = createContext<ThemeCtx>({
  isDark: false,
  toggleTheme: () => {},
  page: "dashboard",
  setPage: () => {},
});

export const useTheme = () => useContext(ThemeContext);

export default function App() {
  // ---- Auth stage ----
  const [appStage, setAppStage] = useState<AppStage>("login");
  const [currentUser, setCurrentUser] = useState<{ email?: string } | null>(null);
  const [selectedLLM, setSelectedLLM] = useState("");

  // ---- Theme ----
  const [isDark, setIsDark] = useState(() => {
    return storage.getString(KEYS.theme, "light") === "dark";
  });
  const [page, setPageState] = useState<Page>(() => {
    return parseHash(window.location.hash) || "dashboard";
  });
  const setPage = (p: Page) => {
    setPageState(p);
    try {
      window.location.hash = `#/${p}`;
    } catch { /* ignore */ }
  };

  useEffect(() => {
    const onHash = () => {
      const parsed = parseHash(window.location.hash);
      if (parsed) setPageState(parsed);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", isDark);
    storage.set(KEYS.theme, isDark ? "dark" : "light");
  }, [isDark]);

  // Cookie auth: the session lives in the httpOnly `aria_token` cookie.
  // On mount, probe the live backend session (credentials:include). 401
  // forces login; network blips keep the current stage (pages retry).
  // Also listen for centralized "aria:unauthorized" (any 401) to sign out.
  useEffect(() => {
    const toLogin = () => {
      clearAuthSession();
      setCurrentUser(null);
      setAppStage("login");
    };
    const onUnauthorized = () => toLogin();
    window.addEventListener("aria:unauthorized", onUnauthorized);
    (async () => {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 10000);
      try {
        // Single client: apiFetch carries credentials + 401 cleanup.
        // A 10s caller signal keeps the boot probe snappy; slower
        // failures stay AbortErrors so blips never log the user out.
        const body: any = await apiFetch("/session/status", { signal: ctrl.signal }).catch((e: any) => {
          if (e?.status === 401) throw e;
          return null;
        });
        if (!body) return;
        // Restore connection state from the live server session so a page
        // reload does not drop every page to "Connect a data source" while
        // the backend still holds the connection (connectionStatus is
        // memory-only state and otherwise resets on each load).
        if (body?.connection_status === "connected" || body?.display_state === "ready") {
          if (body?.display_state === "schema-loading" || body?.schema_status === "pending") {
            setConnectionStatus("schema-loading");
          } else if (body?.display_state === "error" || body?.schema_status === "failed" || body?.schema_status === "empty") {
            setConnectionStatus("error");
          } else {
            setConnectionStatus("connected");
          }
          if (body.db_name) setDbName(body.db_name);
          if (body.source_type) setSelectedSourceType(body.source_type);
        }
      } catch (e: any) {
        // 401 returns to login; blips/offline keep the current stage so
        // a flaky network does not log the user out. Pages retry.
        if (e?.status === 401) toLogin();
      } finally {
        clearTimeout(t);
      }
    })();
    return () => window.removeEventListener("aria:unauthorized", onUnauthorized);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const toggleTheme = () => setIsDark((d) => !d);

  // ---- Connection state ----
  const [connectionStatus, setConnectionStatus] = useState("not-connected");
  const [dbName, setDbName] = useState("");
  const [schemaData, setSchemaData] = useState<any>(null);
  const [schemaStatus, setSchemaStatus] = useState("idle");
  const [schemaError, setSchemaError] = useState("");
  const [connectionError, setConnectionError] = useState("");
  const [selectedSourceType, setSelectedSourceType] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // ---- DB credentials ----
  const [dbHost, setDbHost] = useState("localhost");
  const [dbPort, setDbPort] = useState("5432");
  const [dbUsername, setDbUsername] = useState("postgres");
  const [dbPassword, setDbPassword] = useState("");

  // ---- Question / Query ----
  const [question, setQuestion] = useState("");
  const [submittedQuestion, setSubmittedQuestion] = useState("");
  // Cancel obsolete analyses when the user asks again (§25 performance).
  const askAbortRef = useRef<AbortController | null>(null);
  const [questionSuggestions, setQuestionSuggestions] = useState<string[]>([]);
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);

  // ---- Goal Agent ----
  const [goalStatus, setGoalStatus] = useState("idle");
  const [goalResult, setGoalResult] = useState<any>(null);
  const [goalError, setGoalError] = useState("");
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  // ---- Insight Agent ----
  const [insightStatus, setInsightStatus] = useState("idle");
  const [insightResult, setInsightResult] = useState<any>(null);
  const [insightError, setInsightError] = useState("");

  // ---- History ----
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0);
  const [currentHistorySessionId, setCurrentHistorySessionId] = useState<string | null>(null);

  // =========================================================
  // AUTH HANDLERS
  // =========================================================

  const handleLogin = () => {
    const user = getAuthUser();
    setCurrentUser(user ? { email: user.email || user.username } : { email: "user" });
    setAppStage("llm-selection");
  };

  const handleLLMSelect = async (mode: string, apiKey?: string) => {
    setSelectedLLM(mode);
    storage.set(KEYS.llmMode, mode);
    // Materialize the choice server-side so the session provider matches
    // the UI immediately (prevents "choose an LLM provider" right after).
    // Provider isolation: the Groq key is sent ONLY when Cloud is selected;
    // Local AI switches keyless and never touches stored keys.
    try {
      await apiFetch("/provider/switch", {
        method: "POST",
        body: JSON.stringify(mode === "cloud" && apiKey ? { provider: mode, api_key: apiKey } : { provider: mode }),
      });
    } catch (err: any) {
      if (err?.status === 401) {
        clearAuthSession();
        setCurrentUser(null);
        setAppStage("login");
        throw new Error("Session expired. Please sign in again.");
      }
      // Surface the failure so the user knows the switch did not happen.
      throw new Error(err?.message || `Could not switch to ${mode}.`);
    }
    // Restore connection state from the stored session so the Dashboard
    // shows KPIs/charts without requiring a fresh connect.
    (async () => {
      const result = await fetchSessionStatus();
      if (result.success && result.data) {
        if (result.data.connection_status === "connected") {
          setConnectionStatus("connected");
          setDbName(result.data.db_name || "");
          setSelectedSourceType(result.data.source_type || "relational");
          // Load schema from the server when reconnecting so the Data
          // section never shows "pending" — the DB was never disconnected.
          // Load schema directly (skip extraction) when reconnecting so the
          // Data section never shows "pending" — the DB was never disconnected.
          try {
            const schemaJson: any = await apiFetch("/schema");
            if (schemaJson?.ok && schemaJson.schema) {
              setSchemaData(schemaJson.schema);
              setSchemaStatus("success");
            } else {
              setSchemaStatus("idle");
              setSchemaData(null);
            }
          } catch (e: any) {
            if (e?.status === 401) {
              clearAuthSession();
              setCurrentUser(null);
              setAppStage("login");
              return;
            }
            setSchemaStatus("idle");
            setSchemaData(null);
          }
        }
      }
    })();
    setAppStage("app");
  };

  const handleSignOut = async () => {
    try {
      await apiFetch("/auth/logout", {
        method: "POST",
        body: JSON.stringify({}),
      });
    } catch {
      // Clear the local session even when the server is unavailable.
    }
    clearAuthSession();
    setCurrentUser(null);
    setSelectedLLM("");
    setAppStage("login");
    setConnectionStatus("not-connected");
    setDbName("");
    setSchemaData(null);
    setGoalResult(null);
    setAnalysisResult(null);
    setInsightResult(null);
    setQuestion("");
    setSubmittedQuestion("");
    setPage("dashboard");
  };

  // =========================================================
  // CONNECTION HANDLER
  // =========================================================
  // P3 §19: Disconnected/Connecting/Connected/Schema-loading/Ready/Error
  // reflect GET /api/session/status (backend truth), never button success
  // alone. "connected" is kept as the canonical ready value so existing
  // pages (`=== "connected"`) keep working; "ready" is normalized to it.

  const refreshConnectionFromBackend = async () => {
    try {
      const st = await fetchSessionStatus();
      const body: any = (st as any)?.data || {};
      const display = body?.display_state || body?.connection_status || "";
      const schemaSt = body?.schema_status || "";
      if (display === "ready" || body?.connection_status === "connected") {
        if (schemaSt === "pending" || display === "schema-loading") {
          setConnectionStatus("schema-loading");
        } else {
          setConnectionStatus("connected");
        }
        if (body.db_name) setDbName(body.db_name);
        if (body.source_type) setSelectedSourceType(body.source_type);
        return "connected";
      }
      if (display === "schema-loading") {
        setConnectionStatus("schema-loading");
        return "schema-loading";
      }
      if (display === "error") {
        setConnectionStatus("error");
        return "error";
      }
      return display || "not-connected";
    } catch {
      return null;
    }
  };

  const handleConnect = async (details: any) => {
    setConnectionError("");
    setConnectionStatus("connecting");
    try {
      const result = await connectDatabase(details);
      if (result?.success) {
        setConnectionStatus("connected");
        setDbName(result?.data?.database || details?.db || "");
        setSelectedSourceType("relational");
        // Auto-fetch schema after connect
        await handleSchemaFetch();
        // Verify against backend truth; downgrade to error when the
        // server reports empty/failed schema instead of staying "connected".
        const verified = await refreshConnectionFromBackend();
        if (verified === "error") {
          setConnectionError("Connected, but schema extraction returned no usable tables. Check the schema/database selection, then retry.");
        }
      } else {
        setConnectionStatus("error");
        setConnectionError(result?.message || "Connection failed");
      }
    } catch (err: any) {
      setConnectionStatus("error");
      setConnectionError(err?.message || "Connection failed");
    }
  };

  const handleDisconnect = async () => {
    try {
      await disconnectSource();
    } catch (err: any) {
      setConnectionError(err?.message || "Could not disconnect the data source");
      return;
    }
    setConnectionStatus("not-connected");
    setDbName("");
    setSchemaData(null);
    setSchemaStatus("idle");
  };

  const handleSchemaFetch = async () => {
    setSchemaStatus("loading");
    setConnectionStatus((prev) => (prev === "connected" || prev === "error" ? "schema-loading" : prev));
    setSchemaError("");
    try {
      const result = await fetchSchema();
      if (result?.success) {
        setSchemaData(result?.data || null);
        setSchemaStatus("success");
        setConnectionStatus((prev) => (prev === "schema-loading" ? "connected" : prev));
      } else {
        setSchemaError("Failed to fetch schema");
        setSchemaStatus("error");
        setConnectionStatus("error");
      }
    } catch (err: any) {
      setSchemaError(err?.message || "Schema fetch failed");
      setSchemaStatus("error");
      setConnectionStatus("error");
    }
  };

  // =========================================================
  // FILE UPLOAD HANDLER
  // =========================================================

  const handleFileUpload = async (sourceType: string) => {
    const file = selectedFile;
    if (!file) return;
    setConnectionError("");
    setConnectionStatus("connecting");
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("provider", selectedLLM || "local");
      
      let path = sourceType === 'csv' ? `/connect_file` : (sourceType === 'pdf' ? `/connect_pdf` : `/connect_${sourceType}`);
      const data: any = await apiFetch(path, {
        method: "POST",
        body: formData,
      }).catch((e: any) => {
        if (e?.status === 401) {
          clearAuthSession();
          setCurrentUser(null);
          setAppStage("login");
          setConnectionStatus("error");
          setConnectionError("Session expired. Please sign in again.");
          return { __handled401: true };
        }
        throw e;
      });
      if ((data as any)?.__handled401) return;
      if (!data) {
        setConnectionStatus("error");
        setConnectionError("File upload failed");
        return;
      }
      if (data.ok) {
        if ((data as any)?.accepted) {
          /* Large file: backend processes in background. Poll progress
             (30s cap per round, up to ~20 min) until ready or error. */
          setConnectionStatus("schema-loading");
          const deadline = Date.now() + 20 * 60 * 1000;
          let done = false;
          while (Date.now() < deadline && !done) {
            await new Promise((r) => setTimeout(r, 3000));
            try {
              const prog: any = await apiFetch("/upload/progress");
              if (prog?.error) {
                setConnectionStatus("error");
                setConnectionError(String(prog.error).slice(0, 200));
                return;
              }
              if (prog?.done && prog?.result) {
                const res: any = prog.result;
                await handleSchemaFetch();
                setConnectionStatus("connected");
                setDbName(res?.table || res?.file || file.name.replace(/\.[^.]+$/, ""));
                setSelectedSourceType(res?.source_type || "semi-structured");
                return;
              }
              if (prog?.done) {
                await handleSchemaFetch();
                setConnectionStatus("connected");
                return;
              }
            } catch {
              /* transient poll failure — keep waiting */
            }
          }
          setConnectionStatus("error");
          setConnectionError("Processing is taking longer than expected — check Data for progress.");
          return;
        }
        // After successful file connect, fetch schema and mark connected
        setConnectionStatus("schema-loading");
        await handleSchemaFetch();
        setConnectionStatus("connected");
        setDbName(data?.table || data?.file || file.name.replace(/\.[^.]+$/, ""));
        setSelectedSourceType(data?.source_type || "semi-structured");
      } else {
        setConnectionStatus("error");
        setConnectionError(data?.message || "File upload failed");
      }
    } catch (err: any) {
      setConnectionStatus("error");
      setConnectionError(err?.message || "File upload failed");
    }
  };

  const handleAsk = async (q: string) => {
    if (!q?.trim()) return;
    // Abort any in-flight analysis before starting a new one.
    askAbortRef.current?.abort();
    const controller = new AbortController();
    askAbortRef.current = controller;
    setLoading(true);
    setGoalStatus("loading");
    setSubmittedQuestion(q);
    setGoalResult(null);
    setGoalError("");
    setInsightResult(null);
    try {
      // Pass multi-turn context when available (follow-up questions)
      const multiTurnContext = goalResult?.generatedSql && submittedQuestion
        ? { previousGoal: submittedQuestion, previousResult: { data: goalResult?.data || [], columns: goalResult?.processedDataSummary?.columns || [], data_selection: goalResult?.data_selection }, conversationId: goalResult?.conversationId }
        : goalResult?.conversationId ? { conversationId: goalResult.conversationId } : null;
      const result = await analyzeQuestion(q, selectedLLM, {
        sourceType: selectedSourceType,
        dbName,
      }, multiTurnContext as any, { signal: controller.signal });
      if (controller.signal.aborted) return;
      if (result?.success) {
        setGoalResult(result?.data || null);
        setGoalStatus("success");
        const analytics = await generateAnalyticsResult(result?.data);
        if (controller.signal.aborted) return;
        setAnalysisResult(analytics?.data || null);
      } else {
        setGoalStatus("error");
        setGoalError(result?.message || "Analysis failed");
      }
    } catch (err: any) {
      if (err?.name === "AbortError" || controller.signal.aborted) {
        setGoalStatus("idle");
        return;
      }
      setGoalStatus("error");
      setGoalError(err?.message || "Analysis failed");
    } finally {
      if (!controller.signal.aborted) setLoading(false);
      else setLoading(false);
    }
  };

  // =========================================================
  // INSIGHT STATE (populated by GET /api/insights inside the Insights page;
  // the legacy POST /api/insight retry was removed — Refresh re-analyzes.)
  // =========================================================

  // =========================================================
  // COPILOT HANDOFF (page isolation rule)
  // =========================================================
  // Pages are isolated: the ONLY cross-page wiring allowed is Ask ARIA
  // buttons → Copilot (question carried over), plus Dashboard View-all →
  // Insights. Everything else happens in place.
  const [copilotPrefill, setCopilotPrefill] = useState("");

  const goCopilot = (question: string) => {
    const q = (question || "").trim();
    if (!q) {
      setPage("copilot");
      return;
    }
    setCopilotPrefill(q);
    setPage("copilot");
  };

  const handleDeleteHistory = (_id: string) => {
    setHistoryRefreshKey((k) => k + 1);
  };

  // =========================================================
  // SCREENS (each wrapped so one bad page never blanks the app)
  // =========================================================

  // Per-page boundary helper (defined below use to avoid TDZ issues
  // with the page switch).
  const wrapPage = (key: string, node: ReactNode) => (
    <ErrorBoundary key={`${key}-${page}`} page={key}>{node}</ErrorBoundary>
  );

  if (appStage === "login") {
    return <Login onLogin={handleLogin} />;
  }

  if (appStage === "llm-selection") {
    return (
      <Suspense fallback={<PageFallback />}>
        <LLMSelection onSelectLLM={handleLLMSelect} />
      </Suspense>
    );
  }

  return (
    <ThemeContext.Provider value={{ isDark, toggleTheme, page, setPage }}>
      {/* ToastProvider mounts the toast region: figma/pages/Documents.tsx
          already calls useToast(), which throws without a provider. */}
      <ToastProvider>
      <AppShell
        user={currentUser}
        selectedLLM={selectedLLM}
        dbName={dbName}
        onChangeLLM={() => setAppStage("llm-selection")}
        onSignOut={handleSignOut}
      >
        <div key={page} className="page-enter"><Suspense fallback={<PageFallback />}>{wrapPage(page, (() => {
          switch (page) {
            case "dashboard":
              return (
                <Dashboard
                  setPage={setPage}
                  onAsk={goCopilot}
                  user={currentUser}
                  connectionStatus={connectionStatus}
                  dbName={dbName}
                  insightResult={insightResult}
                  analysisResult={analysisResult}
                  connectionError={connectionError}
                />
              );
            case "analytics":
              return (
                <AIAnalytics
                  initialQuery={question}
                  onAsk={handleAsk}
                  analysisResult={analysisResult}
                  goalStatus={goalStatus}
                  goalResult={goalResult}
                  goalError={goalError}
                  loading={loading}
                  onNavigate={(p) => setPage(p as Page)}
                  onAskCopilot={goCopilot}
                  connectionStatus={connectionStatus}
                  dbName={dbName}
                />
              );
            case "insights":
              return (
                <Insights
                  insightResult={insightResult}
                  insightStatus={insightStatus}
                  insightError={insightError}
                  connectionStatus={connectionStatus}
                  onAskQuestion={goCopilot}
                />
              );
            case "predictive":
              return <Predictive insightResult={insightResult} goalResult={goalResult} connectionStatus={connectionStatus} dbName={dbName} />;
            case "intelligence":
              return <Intelligence connectionStatus={connectionStatus} onAskCopilot={goCopilot} />;
            case "data":
              return (
                <Data
                  connectionStatus={connectionStatus}
                  selectedSourceType={selectedSourceType}
                  dbName={dbName}
                  schemaData={schemaData}
                  schemaStatus={schemaStatus}
                  schemaError={schemaError}
                  onConnect={handleConnect}
                  onDisconnect={handleDisconnect}
                  onSchemaRetry={handleSchemaFetch}
                  onSourceTypeSelect={(t: string) => setSelectedSourceType(t)}
                  selectedFile={selectedFile}
                  onFileSelect={(f: File) => setSelectedFile(f)}
                  onFileUpload={(st?: string) => handleFileUpload(st || selectedSourceType || "csv")}
                  dbHost={dbHost}
                  dbPort={dbPort}
                  dbUsername={dbUsername}
                  dbPassword={dbPassword}
                  setDbHost={setDbHost}
                  setDbPort={setDbPort}
                  setDbName={setDbName}
                  setDbUsername={setDbUsername}
                  setDbPassword={setDbPassword}
                  connectionError={connectionError}
                />
              );
            case "documents":
              return <Documents />;
            case "copilot":
              return <Copilot analysisResult={analysisResult} goalResult={goalResult} initialQuestion={copilotPrefill} />;
            case "history":
              return (
                <History
                  historyRefreshKey={historyRefreshKey}
                  onDeleteHistory={handleDeleteHistory}
                />
              );
            case "settings":
              return (
                <Settings
                  selectedLLM={selectedLLM}
                  onChangeLLM={() => setAppStage("llm-selection")}
                   onProviderSwitched={(p) => {
                     setSelectedLLM(p);
                     storage.set(KEYS.llmMode, p);
                   }}
                  user={currentUser}
                  connectionStatus={connectionStatus}
                  dbName={dbName}
                  selectedSourceType={selectedSourceType}
                />
              );
            default:
              return null;
          }
        })())}</Suspense></div>
      </AppShell>
      {/* Mini ARIA robot: every page except the Copilot page itself. */}
      {page !== "copilot" && <MiniAria />}
      </ToastProvider>
    </ThemeContext.Provider>
  );
}
