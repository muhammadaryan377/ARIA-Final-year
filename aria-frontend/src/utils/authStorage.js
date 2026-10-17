// =========================================================
// ARIA FRONTEND AUTH STORAGE
// =========================================================
//
// Connected to the ARIA FastAPI backend:
//
// React (this file)
//   â†“  POST /api/auth/register | /api/auth/login
// FastAPI
//   â†“
// SQLite + JWT
//
// Cookie auth: the backend sets an httpOnly `aria_token` cookie on
// login/register. Only the display identity is kept in localStorage;
// no JWT is stored and no Authorization header is sent.

import { API_BASE } from "../services/api.js";

const TOKEN_KEY = "aria_auth_token"; // legacy: kept only to clear old sessions; cookie auth now
const USER_KEY = "aria_auth_user";

// =========================================================
// BACKEND USERNAME
// =========================================================
//
// The backend accepts usernames of letters, digits and [._-]
// only (Pydantic pattern constraint). An email address contains
// characters such as "@" that are not valid, so a deterministic,
// reversible username is derived from the email. Register and
// login must use the exact same derivation.
// =========================================================

const toBackendUsername = (email) => {
  // Collision-safe: "@" -> "_at_", "+" -> "_plus_", "." preserved.
  // "a.b@gmail.com" and "ab@gmail.com" now map to distinct usernames.
  const cleaned = normalizeEmail(email)
    .replace(/@/g, "_at_")
    .replace(/\+/g, "_plus_")
    .replace(/[^a-z0-9_.-]/g, "");

  return cleaned.length >= 3 ? cleaned : `user_${cleaned}`;
};

export const isTokenExpired = (token) => {
  try {
    const parts = (token || "").split(".");
    if (parts.length !== 3) return false;
    const payload = JSON.parse(atob(parts[1].replace(/-/g, "+").replace(/_/g, "/")));
    if (!payload.exp) return false;
    return Date.now() / 1000 > payload.exp;
  } catch {
    return false;
  }
};

// =========================================================
// NORMALIZE EMAIL
// =========================================================

const normalizeEmail = (email) => {
  return (email || "").trim().toLowerCase();
};

// =========================================================
// TOKEN / SESSION STORAGE
// =========================================================

// Cookie auth: the backend httpOnly `aria_token` cookie is the session.
// No JWT is kept in localStorage; getAuthToken stays as a deprecated stub
// returning "" so legacy callers never emit `Bearer null`/empty.
export const getAuthToken = () => {
  return "";
};

export const getAuthUser = () => {
  try {
    return JSON.parse(localStorage.getItem(USER_KEY) || "null");
  } catch (error) {
    console.error("Could not read ARIA auth session:", error);

    return null;
  }
};

export const setAuthSession = (token, username, email) => {
  // token ignored: session lives in the httpOnly cookie. Keep only the
  // display identity; drop any legacy JWT copy.
  try {
    localStorage.removeItem(TOKEN_KEY);
  } catch { /* ignore */ }

  localStorage.setItem(
    USER_KEY,
    JSON.stringify({
      username: username || email || "",
      email: email || username || "",
    }),
  );
};

export const clearAuthSession = () => {
  localStorage.removeItem(TOKEN_KEY);

  localStorage.removeItem(USER_KEY);
};

// =========================================================
// BACKEND AUTH REQUEST
// =========================================================

const authRequest = async (path, payload) => {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",

    // Include cookies so the backend's httpOnly `aria_token` cookie
    // (set on login/register) is stored and sent automatically.
    // The localStorage copy below is kept for expiry pre-checks.
    credentials: "include",

    headers: {
      "Content-Type": "application/json",
    },

    body: JSON.stringify(payload),
  });

  let data;

  try {
    data = await response.json();
  } catch {
    data = {};
  }

  if (!response.ok) {
    const message =
      (typeof data?.detail === "string" && data.detail) ||
      data?.message ||
      `Request failed (${response.status}).`;

    const error = new Error(message);

    error.status = response.status;

    throw error;
  }

  return data;
};

// =========================================================
// REGISTER USER
// =========================================================

export const registerLocalUser = async (email, password) => {
  const currentEmail = normalizeEmail(email);

  try {
    const data = await authRequest("/auth/register", {
      username: toBackendUsername(currentEmail),

      password,
    });

    const username = data?.username || toBackendUsername(currentEmail);

    setAuthSession(data?.token, username, currentEmail);

    return {
      success: true,

      user: {
        id: username,

        email: currentEmail,
      },
    };
  } catch (error) {
    console.error("Registration error:", error);

    return {
      success: false,

      message: error?.message || "Could not create your account.",
    };
  }
};

// =========================================================
// LOGIN USER
// =========================================================

export const authenticateLocalUser = async (email, password) => {
  const currentEmail = normalizeEmail(email);

  try {
    const data = await authRequest("/auth/login", {
      username: toBackendUsername(currentEmail),

      password,
    });

    const username = data?.username || toBackendUsername(currentEmail);

    setAuthSession(data?.token, username, currentEmail);

    return {
      success: true,

      user: {
        id: username,

        email: currentEmail,
      },
    };
  } catch (error) {
    console.error("Login error:", error);

    const isMissing = error?.status === 401;

    return {
      success: false,

      code: isMissing ? "USER_NOT_FOUND" : undefined,

      message:
        error?.message || "Incorrect password. Please try again.",
    };
  }
};

