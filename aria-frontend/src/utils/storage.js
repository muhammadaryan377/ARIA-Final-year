// =========================================================
// ARIA VERSIONED STORAGE HELPER
// =========================================================
//
// Single namespaced localStorage layer. All keys live under the
// `aria.v1.*` namespace so future migrations can bump the version
// without colliding with legacy unversioned keys (which are still
// read once and migrated forward).
//
// Features: JSON-safe get/set, per-key TTL expiry, quota try/catch
// (QuotaExceededError -> evict expired entries once, then retry),
// and in-memory fallback when storage is unavailable (private mode).

const NS = "aria.v1.";

export const KEYS = {
  theme: `${NS}theme`, // "dark" | "light"
  llmMode: `${NS}ui.llm_mode`,
  ui: `${NS}ui`, // generic UI prefs object
  avatar: `${NS}profile.avatar`,
  authUser: `${NS}auth.user`,
  copilotChats: `${NS}copilot.chats`, // base; per-user suffix appended
  forecastContext: `${NS}forecast.context`,
  analysisContext: `${NS}analysis.context`,
  sharedContext: `${NS}shared.context`,
  history: `${NS}history.sessions`,
  dashboardDrill: `${NS}dashboard.drill`,
  sidebarCollapsed: `${NS}ui.sidebar_collapsed`,
};

// Legacy (unversioned) keys still honored as read-fallbacks.
const LEGACY_FALLBACK = {
  [KEYS.theme]: ["aria_theme"],
  [KEYS.llmMode]: ["aria_llm_mode"],
  [KEYS.avatar]: ["aria_avatar"],
  [KEYS.authUser]: ["aria_auth_user"],
  [KEYS.forecastContext]: ["aria_forecast_context"],
  [KEYS.analysisContext]: ["aria_analysis_context"],
  [KEYS.history]: ["aria_analysis_history"],
  [KEYS.sidebarCollapsed]: ["aria_sidebar_collapsed"],
};

const memory = new Map();

const canUseStorage = () => {
  try {
    return typeof localStorage !== "undefined";
  } catch {
    return false;
  }
};

const isQuotaError = (e) =>
  e?.name === "QuotaExceededError" ||
  e?.code === 22 ||
  e?.code === 1014 ||
  String(e?.message || "").toLowerCase().includes("quota");

const readRaw = (key) => {
  if (!canUseStorage()) return memory.has(key) ? memory.get(key) : null;
  try {
    const v = localStorage.getItem(key);
    if (v !== null) return v;
  } catch {
    return memory.has(key) ? memory.get(key) : null;
  }
  // Legacy fallback (migrate forward on next set).
  const legacyKeys = LEGACY_FALLBACK[key] || [];
  for (const lk of legacyKeys) {
    try {
      const v = localStorage.getItem(lk);
      if (v !== null) return v;
    } catch {
      /* ignore */
    }
  }
  return null;
};

/** Parse the `{ v, exp }` envelope; returns { hit, value, expired }. */
const parseEnvelope = (raw) => {
  if (raw === null || raw === undefined) return { hit: false };
  try {
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object" && "__aria_v1" in parsed) {
      if (parsed.exp && Date.now() > parsed.exp) return { hit: false, expired: true };
      return { hit: true, value: parsed.v };
    }
    // Unenveloped legacy value (plain string or JSON).
    try {
      return { hit: true, value: JSON.parse(raw) };
    } catch {
      return { hit: true, value: raw };
    }
  } catch {
    return { hit: true, value: raw };
  }
};

export const storage = {
  /**
   * @param {string} key
   * @param {any} [fallback]
   * @param {{ ttlMs?: number | null }} [opts]
   * @returns {any}
   */
  get(key, fallback = null, { ttlMs = null } = {}) {
    const raw = readRaw(key);
    const parsed = parseEnvelope(raw);
    if (parsed.expired) {
      try {
        storage.remove(key);
      } catch {
        /* ignore */
      }
      return fallback;
    }
    if (!parsed.hit) return fallback;
    void ttlMs;
    return parsed.value === undefined ? fallback : parsed.value;
  },

  /**
   * @param {string} key
   * @param {any} [fallback]
   * @returns {string | null}
   */
  getString(key, fallback = null) {
    const v = storage.get(key, fallback);
    if (v === null || v === undefined) return fallback;
    return typeof v === "string" ? v : String(v);
  },

  /**
   * @param {string} key
   * @param {any} value
   * @param {{ ttlMs?: number | null }} [opts]
   * @returns {boolean}
   */
  set(key, value, { ttlMs = null } = {}) {
    const envelope = JSON.stringify({
      __aria_v1: 1,
      exp: ttlMs ? Date.now() + ttlMs : 0,
      v: value,
    });
    const write = () => {
      if (!canUseStorage()) {
        memory.set(key, envelope);
        return true;
      }
      try {
        localStorage.setItem(key, envelope);
        return true;
      } catch (e) {
        if (isQuotaError(e)) {
          // Evict expired versioned keys once, then retry.
          try {
            const doomed = [];
            for (let i = 0; i < localStorage.length; i++) {
              const k = localStorage.key(i);
              if (k && k.startsWith(NS)) doomed.push(k);
            }
            for (const k of doomed) {
              try {
                const p = JSON.parse(localStorage.getItem(k) || "null");
                if (p && p.exp && Date.now() > p.exp) localStorage.removeItem(k);
              } catch {
                /* keep */
              }
            }
            localStorage.setItem(key, envelope);
            return true;
          } catch {
            memory.set(key, envelope);
            return false;
          }
        }
        memory.set(key, envelope);
        return false;
      }
    };
    return write();
  },

  /** @param {string} key */
  remove(key) {
    memory.delete(key);
    if (!canUseStorage()) return;
    try {
      localStorage.removeItem(key);
    } catch {
      /* ignore */
    }
  },

  /** Per-user key suffix (copilot chats are isolated per identity). */
  /**
   * @param {string} base
   * @param {{ username?: string, email?: string } | null} [user]
   * @returns {string}
   */
  scoped(base, user) {
    const id = user?.username || user?.email || "anon";
    return `${base}.${id}`;
  },
};

/** @param {string} key @param {any} [fallback] @returns {any} */
export const getStored = (key, fallback) => storage.get(key, fallback);
/** @param {string} key @param {any} value @returns {boolean} */
export const setStored = (key, value, opts) => storage.set(key, value, opts);
/** @param {string} key */
export const removeStored = (key) => storage.remove(key);
