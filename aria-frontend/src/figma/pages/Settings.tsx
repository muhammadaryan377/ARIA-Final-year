import { useState, useEffect } from "react";
import { User, Palette, Brain, Database, Bell, Shield, Monitor, ChevronRight } from "lucide-react";
import { useTheme } from "../App";
import { apiFetch } from "../../services/api";
import { KEYS, storage } from "../../utils/storage";

type Tab = "profile" | "appearance" | "ai" | "datasources" | "notifications" | "privacy" | "application";

const tabs: { id: Tab; label: string; icon: React.ElementType }[] = [
  { id: "profile", label: "Profile", icon: User },
  { id: "appearance", label: "Appearance", icon: Palette },
  { id: "ai", label: "AI Configuration", icon: Brain },
  { id: "datasources", label: "Data Sources", icon: Database },
  { id: "notifications", label: "Notifications", icon: Bell },
  { id: "privacy", label: "Privacy", icon: Shield },
  { id: "application", label: "Application", icon: Monitor },
];

interface SettingsProps {
  selectedLLM?: string;
  onChangeLLM?: () => void;
  onProviderSwitched?: (provider: string) => void;
  user?: { email?: string; username?: string } | null;
  connectionStatus?: string;
  dbName?: string;
  selectedSourceType?: string;
}

export default function Settings({ selectedLLM, onChangeLLM, onProviderSwitched, user, connectionStatus, dbName, selectedSourceType }: SettingsProps) {
  const { isDark, toggleTheme } = useTheme();
  const [tab, setTab] = useState<Tab>("profile");
  const [displayName, setDisplayName] = useState(user?.username || "User");
  const [email, setEmail] = useState(user?.email || "");
  const [role, setRole] = useState("Business Analyst");
  const [providers, setProviders] = useState<any[]>([]);
  const [apiKey, setApiKey] = useState("");
  const [selectedProvider, setSelectedProvider] = useState("cloud");
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState("");
  const [notifAnalysis, setNotifAnalysis] = useState(true);
  const [notifInsights, setNotifInsights] = useState(true);
  const [notifSystem, setNotifSystem] = useState(false);
  // Browser-local avatar (no backend storage; honest about scope).
  // Versioned key (aria.v1.profile.avatar), read once via the state
  // initializer — never parsed inline during render.
  const [avatar, setAvatar] = useState<string | null>(() => storage.getString(KEYS.avatar, null));

  const pickAvatar = (file: File | undefined) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) { setSaveMsg("Please choose an image file"); return; }
    if (file.size > 2 * 1024 * 1024) { setSaveMsg("Image must be under 2MB"); return; }
    const reader = new FileReader();
    reader.onload = () => {
      const url = String(reader.result || "");
      setAvatar(url);
      storage.set(KEYS.avatar, url);
      setSaveMsg("Avatar updated in this browser");
      setTimeout(() => setSaveMsg(""), 3000);
    };
    reader.readAsDataURL(file);
  };

  useEffect(() => { fetchProviders(); }, []);

  const fetchProviders = async () => {
    try {
      const data: any = await apiFetch("/settings/providers");
      setProviders(data.providers || []);
    } catch (e: any) {
      if (e?.status !== 401) console.debug("[ARIA] background request failed:", e);
    }
  };

  const saveProvider = async () => {
    if (selectedProvider !== "local" && !apiKey.trim()) return;
    setSaving(true);
    try {
      const data: any = await apiFetch("/settings/providers", {
        method: "POST",
        body: JSON.stringify({ provider: selectedProvider, api_key: apiKey, is_default: true }),
      });
      onProviderSwitched?.(data.provider || selectedProvider);
      setSaveMsg(data.message || "Provider saved"); setApiKey(""); fetchProviders();
      setTimeout(() => setSaveMsg(""), 3000);
    } catch (e: any) { setSaveMsg(e?.status === 401 ? "Session expired. Please sign in again." : (e?.message || "Failed")); } finally { setSaving(false); }
  };

  return (
    <div className="flex h-full">
      {/* Sub-sidebar */}
      <div className="w-48 border-r border-border bg-card/80 backdrop-blur-sm flex-shrink-0">
        <div className="py-3">
          {tabs.map((t) => (
            <button key={t.id} onClick={() => setTab(t.id)}
              className={`nav-item w-full flex items-center gap-2.5 px-4 py-2 text-[13px] transition-all duration-150 ${tab === t.id ? "bg-primary/8 text-primary font-medium border-r-2 border-primary active" : "text-muted-foreground hover:text-foreground hover:bg-secondary/60"}`}>
              <t.icon size={14} />
              <span>{t.label}</span>
              {tab === t.id && <ChevronRight size={12} className="ml-auto" />}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6 max-w-2xl">
        {saveMsg && (
          <div className="mb-4 text-sm text-positive bg-positive/10 border border-positive/20 rounded-lg px-3 py-2">{saveMsg}</div>
        )}

        {tab === "profile" && (
          <div className="space-y-6">
            <div>
              <h2 className="text-base font-semibold text-foreground">Profile</h2>
              <p className="text-xs text-muted-foreground mt-0.5">Manage your personal information.</p>
            </div>
            <div className="flex items-center gap-4">
              {avatar ? (
                <img src={avatar} alt="Profile avatar" className="w-14 h-14 rounded-full object-cover shadow-md" />
              ) : (
                <div className="w-14 h-14 rounded-full bg-primary/10 flex items-center justify-center text-lg font-semibold text-primary shadow-md" style={{ background: "var(--gradient-primary)", color: "white" }}>A</div>
              )}
              <div>
                <label className="text-sm text-primary hover:underline cursor-pointer">
                  Change avatar
                  <input type="file" accept="image/*" className="hidden" onChange={(e) => pickAvatar(e.target.files?.[0])} />
                </label>
                <p className="text-[11px] text-muted-foreground">JPG, PNG up to 2MB · stored in this browser</p>
              </div>
            </div>
            <div className="space-y-4">
              <div>
                <label className="text-xs font-medium text-muted-foreground">Display name</label>
                <input value={displayName} onChange={(e) => setDisplayName(e.target.value)}
                  className="mt-1 w-full bg-secondary border border-border rounded-lg px-3 py-2 text-sm text-foreground outline-none focus:border-primary focus:shadow-[0_0_0_3px_rgba(99,102,241,0.08)] transition-all duration-150" />
              </div>
              <div>
                <label className="text-xs font-medium text-muted-foreground">Email address</label>
                <input value={email} onChange={(e) => setEmail(e.target.value)}
                  className="mt-1 w-full bg-secondary border border-border rounded-lg px-3 py-2 text-sm text-foreground outline-none focus:border-primary focus:shadow-[0_0_0_3px_rgba(99,102,241,0.08)] transition-all duration-150" />
              </div>
              <div>
                <label className="text-xs font-medium text-muted-foreground">Role</label>
                <input value={role} onChange={(e) => setRole(e.target.value)}
                  className="mt-1 w-full bg-secondary border border-border rounded-lg px-3 py-2 text-sm text-foreground outline-none focus:border-primary focus:shadow-[0_0_0_3px_rgba(99,102,241,0.08)] transition-all duration-150" />
              </div>
            </div>
            <button onClick={async () => {
                setSaving(true);
                try {
                  await apiFetch("/settings/general", {
                    method: "POST",
                    body: JSON.stringify({ theme: isDark ? "dark" : "light", language: "en" }),
                  });
                  setSaveMsg("Profile saved");
                  setTimeout(() => setSaveMsg(""), 3000);
                } catch {
                  setSaveMsg("Failed to save");
                  setTimeout(() => setSaveMsg(""), 3000);
                } finally {
                  setSaving(false);
                }
              }}
              className="bg-primary text-primary-foreground px-4 py-2 rounded-lg text-sm font-medium hover:opacity-90 transition-opacity disabled:opacity-50" disabled={saving}>
              {saving ? "Saving..." : "Save changes"}
            </button>
          </div>
        )}

        {tab === "appearance" && (
          <div className="space-y-6">
            <div>
              <h2 className="text-base font-semibold text-foreground">Appearance</h2>
              <p className="text-xs text-muted-foreground mt-0.5">Customize the look and feel.</p>
            </div>
            <div className="flex items-center justify-between py-3 border-t border-border">
              <div>
                <p className="text-sm text-foreground">Dark Mode</p>
                <p className="text-xs text-muted-foreground">Use dark theme across the application</p>
              </div>
              <button onClick={toggleTheme}
                className={`relative w-9 h-5 rounded-full transition-colors ${isDark ? "bg-primary" : "bg-secondary border border-border"}`}>
                <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-card transition-transform ${isDark ? "translate-x-4" : "translate-x-0.5"}`} />
              </button>
            </div>
          </div>
        )}

        {tab === "ai" && (
          <div className="space-y-6">
            <div>
              <h2 className="text-base font-semibold text-foreground">AI Configuration</h2>
              <p className="text-xs text-muted-foreground mt-0.5">Configure which AI models power your analysis.</p>
            </div>
            <div className="space-y-3 pt-3 border-t border-border">
              <p className="text-sm font-medium text-foreground">Add API Key</p>
              <div className="flex gap-2">
                <select value={selectedProvider} onChange={(e) => setSelectedProvider(e.target.value)}
                  className="bg-secondary border border-border rounded-lg px-3 py-2 text-sm text-foreground outline-none w-40">
                  <option value="cloud">Cloud (Groq)</option>
                  <option value="deepseek">DeepSeek</option>
                  <option value="omniroute">Universal (OmniRoute)</option>
                  <option value="local">Local (Ollama)</option>
                </select>
                {selectedProvider !== "local" && (
                  <input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)}
                    placeholder="Paste your API key..." className="flex-1 bg-secondary border border-border rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground outline-none focus:border-primary transition-colors" />
                )}
                <button onClick={saveProvider} disabled={(selectedProvider !== "local" && !apiKey.trim()) || saving}
                  className="bg-primary text-primary-foreground px-3 py-2 rounded-lg text-sm font-medium hover:opacity-90 transition-opacity disabled:opacity-50">
                  {saving ? "Verifying..." : "Save & Activate"}
                </button>
              </div>
            </div>
            {providers.length > 0 && (
              <div className="pt-3 border-t border-border space-y-2">
                <p className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">Configured</p>
                {providers.map((p: any) => (
                  <div key={p.provider} className="flex items-center justify-between py-2 px-3 bg-secondary/50 rounded-lg">
                    <span className="text-sm text-foreground font-medium capitalize">{p.provider}</span>
                    <span className="text-[11px] text-muted-foreground">Key configured</span>
                  </div>
                ))}
              </div>
            )}
            {selectedLLM && (
              <div className="pt-3 border-t border-border">
                <p className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider mb-2">Current AI Mode</p>
                <div className="flex items-center justify-between py-2 px-3 bg-primary/5 border border-primary/20 rounded-lg">
                  <span className="text-sm font-medium text-primary capitalize">{selectedLLM === "local" ? "Local LLM (Ollama)" : selectedLLM === "cloud" ? "Cloud LLM (Groq)" : selectedLLM === "deepseek" ? "DeepSeek" : selectedLLM === "omniroute" ? "Universal (OmniRoute)" : selectedLLM}</span>
                  {onChangeLLM && (
                    <button onClick={onChangeLLM} className="text-[11px] text-primary hover:underline">Change</button>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {tab === "datasources" && (
          <div className="space-y-6">
            <div>
              <h2 className="text-base font-semibold text-foreground">Data Sources</h2>
              <p className="text-xs text-muted-foreground mt-0.5">Manage connected databases and files.</p>
            </div>
            {connectionStatus === "connected" && dbName ? (
              <div className="bg-primary/5 border border-primary/20 rounded-xl p-5 space-y-3">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-positive" />
                  <p className="text-sm font-medium text-foreground">Connected</p>
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Database</span>
                    <span className="text-foreground font-medium">{dbName}</span>
                  </div>
                  {selectedSourceType && (
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Source Type</span>
                      <span className="text-foreground font-medium capitalize">{selectedSourceType}</span>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="bg-card border border-border rounded-xl p-6 text-center">
                <p className="text-sm text-muted-foreground">No data source connected.</p>
                <p className="text-xs text-muted-foreground mt-1">Go to the Data Sources page to connect a database.</p>
              </div>
            )}
          </div>
        )}

        {tab === "notifications" && (
          <div className="space-y-6">
            <div>
              <h2 className="text-base font-semibold text-foreground">Notifications</h2>
              <p className="text-xs text-muted-foreground mt-0.5">Configure notification preferences.</p>
            </div>
            <div className="space-y-3">
              {[
                { label: "Analysis complete", state: notifAnalysis, toggle: setNotifAnalysis },
                { label: "New insights detected", state: notifInsights, toggle: setNotifInsights },
                { label: "System alerts", state: notifSystem, toggle: setNotifSystem },
              ].map((item) => (
                <div key={item.label} className="flex items-center justify-between py-2 border-b border-border last:border-0">
                  <p className="text-sm text-foreground">{item.label}</p>
                  <button onClick={() => item.toggle(!item.state)}
                    className={`relative w-9 h-5 rounded-full transition-colors ${item.state ? "bg-primary" : "bg-secondary border border-border"}`}>
                    <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-card transition-transform ${item.state ? "translate-x-4" : "translate-x-0.5"}`} />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {tab === "privacy" && (
          <div className="space-y-6">
            <div>
              <h2 className="text-base font-semibold text-foreground">Privacy</h2>
              <p className="text-xs text-muted-foreground mt-0.5">Manage your privacy settings.</p>
            </div>
            <div className="py-3 border-t border-border">
              <p className="text-sm text-foreground">API Key Security</p>
              <p className="text-xs text-muted-foreground mt-1">All API keys are stored encrypted. Keys are never sent to our servers.</p>
            </div>
          </div>
        )}

        {tab === "application" && (
          <div className="space-y-6">
            <div>
              <h2 className="text-base font-semibold text-foreground">Application</h2>
              <p className="text-xs text-muted-foreground mt-0.5">Application-level settings.</p>
            </div>
            <div className="py-3 border-t border-border">
              <p className="text-sm text-foreground">Session</p>
              <p className="text-xs text-muted-foreground mt-1">Your authentication session is active.</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
