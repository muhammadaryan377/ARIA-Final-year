// =========================================================
// DAY 9 - HISTORY STORAGE
// =========================================================
//
// Preferred source: backend GET /history (FastAPI -> PostgreSQL).
// Fallback: versioned localStorage (aria.v1.history.sessions, 7-day
// TTL) so the History page still works offline / on backend blips.
// Legacy unversioned key `aria_analysis_history` is migrated forward
// on read. All writes are quota-guarded (try/catch, boolean result).

import { apiFetch } from "../services/api.js";
import { KEYS, storage } from "./storage.js";

const HISTORY_TTL_MS = 7 * 24 * 60 * 60 * 1000;

// =========================================================
// SYNC LOCAL READ (localStorage fallback path)
// =========================================================

export const getHistorySessions = () => {
  try {
    const sessions = storage.get(KEYS.history, null);
    if (Array.isArray(sessions)) return sessions;
    return [];
  } catch (error) {
    console.error("Could not load ARIA history:", error);
    return [];
  }
};

// =========================================================
// SAVE ALL HISTORY (local fallback; quota-guarded)
// =========================================================

const saveHistorySessions = (sessions) => {
  try {
    const ok = storage.set(KEYS.history, sessions, { ttlMs: HISTORY_TTL_MS });
    return ok !== false;
  } catch (error) {
    console.error("Could not save ARIA history:", error);
    return false;
  }
};

// =========================================================
// BACKEND-FIRST ASYNC READ (prefers /history, falls back local)
// =========================================================

export const getHistorySessionsAsync = async () => {
  try {
    const data = await apiFetch("/history?limit=100");
    const list = data?.sessions || data?.history || data?.data || data;
    if (Array.isArray(list)) {
      // Refresh the local fallback copy (best-effort, quota-guarded).
      try {
        storage.set(KEYS.history, list, { ttlMs: HISTORY_TTL_MS });
      } catch {
        /* offline cache is best-effort */
      }
      return list;
    }
  } catch {
    /* backend unreachable: fall through to local copy */
  }
  return getHistorySessions();
};

// =========================================================
// ADD ONE SESSION
// =========================================================

export const addHistorySession = (session) => {
  const currentHistory = getHistorySessions();

  const updatedHistory = [session, ...currentHistory];

  saveHistorySessions(updatedHistory);

  return updatedHistory;
};

// =========================================================
// UPDATE ONE SESSION
// =========================================================

export const updateHistorySession = (sessionId, updates) => {
  const currentHistory = getHistorySessions();

  const updatedHistory = currentHistory.map((session) =>
    session.id === sessionId
      ? {
          ...session,
          ...updates,
        }
      : session,
  );

  saveHistorySessions(updatedHistory);

  return updatedHistory;
};

// =========================================================
// DELETE ONE SESSION
// =========================================================

export const deleteHistorySession = (sessionId) => {
  const currentHistory = getHistorySessions();

  const updatedHistory = currentHistory.filter(
    (session) => session.id !== sessionId,
  );

  saveHistorySessions(updatedHistory);

  return updatedHistory;
};
