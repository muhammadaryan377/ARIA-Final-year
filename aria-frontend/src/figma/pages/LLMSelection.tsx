import { useState, useEffect } from "react";
import { Zap, Shield, CheckCircle2 } from "lucide-react";
import { apiFetch } from "../../services/api";

interface LLMSelectionProps {
  onSelectLLM: (mode: string, apiKey?: string) => Promise<void> | void;
}

const LLM_OPTIONS = [
  {
    id: "cloud",
    label: "Cloud AI",
    sublabel: "Groq-powered",
    icon: Zap,
    badge: "Fast",
    description: "Lightning-fast cloud processing with GPT-4 class models.",
    speed: "Instant",
    privacy: "Cloud-processed",
  },
  {
    id: "local",
    label: "Local AI",
    sublabel: "On-device",
    icon: Shield,
    badge: "Private",
    description: "Keep all data on your machine. Full privacy guaranteed.",
    speed: "Hardware-dependent",
    privacy: "100% local",
  },
];

export default function LLMSelection({ onSelectLLM }: LLMSelectionProps) {
  const [selected, setSelected] = useState<string | null>(null);
  const [groqKey, setGroqKey] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<Record<string, { available: boolean; reason: string }>>({});
  const [statusLoading, setStatusLoading] = useState(true);

  const loadStatus = async () => {
    setStatusLoading(true);
    try {
      // Single client: apiFetch handles API_BASE + credentials:include +
      // X-Request-ID + 401. Throws on error; status stays advisory.
      const data = await apiFetch(`/providers/status`);
      if (Array.isArray(data.providers)) {
        const map: Record<string, { available: boolean; reason: string }> = {};
        for (const p of data.providers) {
          map[p.provider] = { available: !!p.available, reason: p.reason || "" };
        }
        setStatus(map);
      }
    } catch {
      // Status is advisory; selection still works and the switch validates.
    } finally {
      setStatusLoading(false);
    }
  };

  useEffect(() => { loadStatus(); }, []);

  const confirm = async () => {
    if (!selected || busy) return;
    setError("");
    if (selected === "cloud" && !groqKey.trim()) {
      const st = status["cloud"];
      if (st && !st.available && st.reason) {
        setError(st.reason);
      } else {
        setError("Paste your Groq API key to use Cloud AI — Local AI needs no key.");
      }
      return;
    }
    const st = selected ? status[selected] : undefined;
    // Availability describes the saved key; validate a newly entered key on the backend.
    const hasNewCloudKey = selected === "cloud" && Boolean(groqKey.trim());
    if (st && !st.available && !hasNewCloudKey) {
      setError(`${st.reason} You can retry the status check or choose the other engine.`);
      return;
    }
    setBusy(true);
    try {
      await onSelectLLM(selected, selected === "cloud" ? groqKey.trim() : undefined);
    } catch (err: any) {
      setError(err?.message || "Could not switch provider. Check the key and try again.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-6" style={{ background: "var(--background)" }}>
      <div className="w-full max-w-lg animate-fade-in-scale">
        {/* Logo */}
        <div className="flex items-center justify-center gap-2 mb-8">
          <div
            className="w-9 h-9 rounded-lg flex items-center justify-center text-white font-bold text-sm"
            style={{ background: "var(--gradient-primary)" }}
          >
            A
          </div>
          <span className="text-xl font-semibold" style={{ color: "var(--foreground)" }}>
            ARIA
          </span>
        </div>

        <h1
          className="text-center text-2xl font-bold mb-1"
          style={{ color: "var(--foreground)" }}
        >
          Choose your AI engine
        </h1>
        <p className="text-center text-sm mb-8" style={{ color: "var(--muted-foreground)" }}>
          Select how ARIA processes your analytics. You can change this later.
        </p>

        <div className="space-y-3 mb-6">
          {LLM_OPTIONS.map((opt) => (
            <button
              key={opt.id}
              type="button"
              onClick={() => setSelected(opt.id)}
              className="w-full text-left rounded-xl p-5 transition-all animate-fade-in"
              style={{
                background: "var(--card)",
                border: selected === opt.id
                  ? "2px solid var(--primary)"
                  : "1px solid var(--border)",
                boxShadow: selected === opt.id ? "var(--shadow-glow)" : "var(--shadow-sm)",
              }}
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div
                    className="w-10 h-10 rounded-lg flex items-center justify-center"
                    style={{
                      background: selected === opt.id ? "var(--primary-light)" : "var(--muted)",
                      color: selected === opt.id ? "var(--primary)" : "var(--muted-foreground)",
                    }}
                  >
                    <opt.icon size={20} />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-semibold" style={{ color: "var(--foreground)" }}>
                        {opt.label}
                      </span>
                      <span
                        className="badge-gradient text-xs px-2 py-0.5 rounded-full font-medium"
                      >
                        {opt.badge}
                      </span>
                    </div>
                    <span className="text-xs" style={{ color: "var(--muted-foreground)" }}>
                      {opt.sublabel}
                    </span>
                  </div>
                </div>
                <div
                  className="w-5 h-5 rounded-full flex items-center justify-center mt-1"
                  style={{
                    border: selected === opt.id
                      ? "2px solid var(--primary)"
                      : "2px solid var(--border)",
                    background: selected === opt.id ? "var(--primary)" : "transparent",
                  }}
                >
                  {selected === opt.id && <CheckCircle2 size={12} color="white" />}
                </div>
              </div>

              <p className="text-sm mb-3" style={{ color: "var(--muted-foreground)" }}>
                {opt.description}
              </p>

              <div className="flex gap-6 text-xs">
                <div>
                  <span className="font-medium" style={{ color: "var(--foreground)" }}>Speed: </span>
                  <span style={{ color: "var(--muted-foreground)" }}>{opt.speed}</span>
                </div>
                <div>
                  <span className="font-medium" style={{ color: "var(--foreground)" }}>Privacy: </span>
                  <span style={{ color: "var(--muted-foreground)" }}>{opt.privacy}</span>
                </div>
              </div>

              <div className="mt-3 flex items-center gap-2 text-[11px]">
                {statusLoading ? (
                  <span style={{ color: "var(--muted-foreground)" }}>Checking availability…</span>
                ) : status[opt.id] ? (
                  <>
                    <span
                      className="inline-block w-1.5 h-1.5 rounded-full"
                      style={{ background: status[opt.id].available ? "var(--positive)" : "var(--negative)" }}
                    />
                    <span style={{ color: "var(--muted-foreground)" }}>
                      {status[opt.id].available ? "Available" : "Unavailable"} — {status[opt.id].reason}
                    </span>
                  </>
                ) : null}
              </div>
            </button>
          ))}
        </div>

        {selected === "cloud" && (
          <div className="mb-4 animate-fade-in">
            <label htmlFor="llm-groq-key" className="block text-xs font-semibold mb-1.5" style={{ color: "var(--foreground)" }}>
              Groq API key (used only for Cloud AI)
            </label>
            <input
              id="llm-groq-key"
              type="password"
              value={groqKey}
              onChange={(e) => setGroqKey(e.target.value)}
              placeholder="gsk_..."
              autoComplete="off"
              className="w-full px-4 py-2.5 rounded-xl text-sm outline-none"
              style={{ background: "var(--card)", border: "1px solid var(--border)", color: "var(--foreground)" }}
            />
            <p className="text-[11px] mt-1.5" style={{ color: "var(--muted-foreground)" }}>
              Your key is stored encrypted and never used when Local AI is selected.
            </p>
          </div>
        )}

        {error && (
          <p role="alert" className="text-xs font-medium mb-3 text-center" style={{ color: "var(--negative)" }}>
            {error}{" "}
            <button onClick={loadStatus} className="underline font-semibold">
              Retry status check
            </button>
          </p>
        )}

        <button
          disabled={!selected || busy}
          onClick={confirm}
          className="premium-btn btn-glow w-full py-2.5 rounded-lg text-sm font-semibold text-white transition-all disabled:opacity-40"
          style={{ background: "var(--gradient-primary)" }}
        >
          {busy ? "Connecting…" : "Continue"}
        </button>
      </div>
    </div>
  );
}
