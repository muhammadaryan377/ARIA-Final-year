# ARIA Security & Quality Audit Report

**Audit Date:** 2025-09-15
**Files Reviewed:** ~100 Python files across core/, api/, agents/, copilot/, insight_rag/, document_processor/, hypothesis_engine/, prediction/, prescription_engine/, goal_agent/, models/, memory/, tests/
**Coverage:** Security, architecture, code quality, error handling, dependencies, auth

---

## Summary

Overall posture: **GOOD**. This is a well-engineered application with serious attention to security. Multiple defense layers exist. The findings below are prioritized for actionability.

---

## CRITICAL Findings (Action Required)

### 1. Session Store Race Condition — Data Loss Risk
**File:** `core/config.py:449-502`
**Severity:** CRITICAL
**Location:** `save_session()`

```python
with SESSION_LOCK:
    SESSION_STORE[user_id] = (now, session)
# Also write to disk so sessions survive server restarts.
try:
    # ... disk write ...
    tmp_path.write_text(...)
    tmp_path.replace(SESSION_STORE_PATH)   # ← not atomic w/ SESSION_STORE update
```

The in-memory store and the disk write are not atomic. A crash between the in-memory update and the disk write causes the session to reappear on next restart with stale data. The `replace()` swap is atomic only on POSIX; on Windows it can fail. Additionally, `save_session()` releases the lock before the disk write, allowing concurrent reads of `SESSION_STORE` while the JSON is being written and parsed.

**Recommendation:** Move the disk write inside the lock, or use an atomic rename with a proper tmp-then-swap pattern that also updates a version number atomically.

---

### 2. Shared Mutable Rate-Limit Dictionary — Process-Safe Violation
**Files:** `core/ratelimit.py:27`, `api/auth_routes.py:61`
**Severity:** HIGH (CRITICAL for multi-worker)
**Location:** `_hits`, `_LOGIN_ATTEMPTS`

Both `_hits` (rate limiter) and `_LOGIN_ATTEMPTS` (login brute-force guard) are **process-local `dict`s** shared across all requests in a single-process server. The code explicitly documents that multi-worker deployments need Redis, but:

1. `core/ratelimit.py` has a warning mechanism (`_check_shared_store`) that **only logs** unless `ARIA_REQUIRE_SHARED_STORE=1` is set.
2. `api/auth_routes.py:61-73` has **no such check** — `_LOGIN_ATTEMPTS` silently breaks under multi-worker with each process having independent counters.

**Recommendation:** Enable `ARIA_REQUIRE_SHARED_STORE=1` in production, or implement a shared-store-backed rate limiter for `_LOGIN_ATTEMPTS`.

---

### 3. `__import__` with User-Controlled String — Potential Code Execution
**File:** `tests/test_scorecard.py:168`
**Severity:** HIGH (but test-only file)
**Location:** `__import__("numpy")`

```python
r = _compute_correlation(__import__("numpy").array([1.0, 2.0]), ...)
```

While this is in a test file, using `__import__` with a hardcoded string is a pattern that can migrate to production. If this pattern appears in any non-test code with user-influenced input, it enables arbitrary code execution.

**Recommendation:** Audit the codebase for any dynamic `__import__`, `eval`, or `exec` with non-constant arguments. Currently clean in production code.

---

### 4. In-Memory Session Without Worker Isolation
**File:** `core/config.py:291`
**Severity:** HIGH

```python
SESSION_STORE: dict[int, tuple[float, dict]] = {}
```

The in-memory session store has no protection against a malicious or buggy user session exhausting memory (e.g., a session dict grows unboundedly). Combined with the `SESSION_MAX_USERS = 100` cap, cleanup is opportunistic rather than enforced.

**Recommendation:** Add per-session size limits or move session data to a bounded external store (Redis) in production.

---

## HIGH Findings

### 5. `except Exception` Swallows Errors Broadly Across API Routes
**Files:** `api/query_routes.py` (~20 instances), `api/kpi_routes.py` (~50 instances), `api/prescription_routes.py`, `api/predictive_routes.py`
**Severity:** HIGH

These bare `except Exception:` blocks hide real failures and make debugging difficult. Many of them are in mission-critical SQL generation and analysis pipelines.

**Example (api/query_routes.py:60):**
```python
try:
    # ... critical analysis ...
except Exception:
    # silently continues with potentially corrupt state
```

**Impact:** Silent failures in the Goal Agent / Insight Agent pipeline mean users get incomplete or incorrect results with no error message.

**Recommendation:** Catch specific exception types. At minimum, log every exception at ERROR level with sanitized details.

---

### 6. PBKDF2 Iterations Below OWASP 2023 Recommendation
**File:** `auth.py:47`
**Severity:** MEDIUM (promoted to HIGH due to password storage)
**Location:** `PBKDF2_ITERATIONS = 120_000`

OWASP 2023 recommends **minimum 600,000** iterations for PBKDF2-HMAC-SHA256. Current value is 120,000 — adequate for 2020-era hardware but below current guidance.

**Recommendation:** Increase to at least 600,000. Note: this changes the password hash format, but existing hashes will fail verification and users will need to reset passwords. Plan migration carefully.

---

### 7. SQLite History DB in `data/` Directory
**File:** `auth.py:20`
**Severity:** MEDIUM
**Location:** `DB_PATH = BASE_DIR / "data" / "aria.db"`

The SQLite auth/history database stores user credentials and query history. It has no file-level encryption and relies on filesystem permissions alone. If the `data/` directory is compromised, both password hashes and query history are exposed.

**Recommendation:** Consider encrypting the SQLite database, or ensure the `data/` directory has strict filesystem permissions (0600). On Windows, the DPAPI protection on the session store doesn't extend to this file.

---

### 8. No Token Rotation on Password Change
**File:** `auth.py`
**Severity:** MEDIUM
**Location:** `auth.py` — no `register()` password change path

When a user changes their password, existing valid JWTs remain active until natural expiry. An attacker with an old token can continue using the account. There's no `revoke_all_user_tokens()` function.

**Recommendation:** Add a `revoke_all_user_tokens(user_id)` function and call it on password change.

---

### 9. Session Store Encryption Keys Stored Alongside Encrypted Data
**File:** `core/config.py:454-495`
**Severity:** MEDIUM
**Location:** `save_session()` — `_encrypt_api_keys`

The session store on disk contains DPAPI-encrypted API keys. On Windows, DPAPI binds decryption to the Windows user account — so the `.env` and session store are protected together. On Linux/macOS, `secret_box.py` falls back to plaintext with a `RuntimeWarning`. This is documented but the production deployment guidance should explicitly require Windows or a proper secret manager.

---

## MEDIUM Findings

### 10. SQL Injection — Parameterized But Unenforced
**Files:** `auth.py`, `core/db.py`, `copilot/sql_engine.py`, `hypothesis_engine/engine.py`, `hypothesis_engine/investigate.py`, `prescription_engine/engine.py`
**Severity:** MEDIUM

SQLite connection strings use Python string formatting (`f"{scheme}://..."` with `quote_plus`). SQL execution uses SQLAlchemy's `text()` with parameters. The code appears safe, but:

- `core/db.py:19-22` uses f-string interpolation for the URI scheme from `_URI_SCHEMES` dict (safe, but fragile).
- `hypothesis_engine/investigate.py:547,607,725,912` and `prescription_engine/engine.py:99,115,485,662,700` use a local `_exec()` helper that wraps SQLAlchemy `text()`. Need to verify all parameters are properly bound.

**Recommendation:** Audit each `_exec()` call site to confirm all SQL values are parameterized, not string-interpolated.

---

### 11. SSRF Protection — DNS Rebinding Window
**File:** `api/connect_routes.py:62-86`
**Severity:** MEDIUM
**Location:** `_BLOCKED_DB_HOSTS`, `_validate_db_endpoint`

The SSRF blocklist covers `169.254.169.254` and `metadata.google.internal` but DNS resolution happens at connection time. An attacker with control of a DNS name can:
1. Point the hostname to a blocked IP after validation
2. Use IPv6 variants (`::ffff:169.254.169.254`)
3. Use URL-encoded or IDN-homograph variants

**Recommendation:** Resolve the hostname during validation and block the resulting IP. Add IPv6 checks. Consider adding a timeout to the connection attempt.

---

### 12. JWT Secret File Created With Insecure Permissions
**File:** `auth.py:62-68`
**Severity:** MEDIUM
**Location:** `_load_secret()`

```python
secret_path.write_text(secret, encoding="utf-8")
try:
    os.chmod(secret_path, 0o600)
except OSError:
    pass   # ← Windows silently fails this
```

On Windows, `os.chmod(path, 0o600)` is silently ignored (no-op). The `.jwt_secret` file is created with default permissions inheriting from the parent directory's ACL. While DPAPI encryption of the session store helps, the JWT secret itself could be readable by other users on a shared Windows system.

---

### 13. CORS Allows `localhost` Origins in Production
**File:** `app.py:120-125`, `.env.example:69`
**Severity:** MEDIUM (LOW if intentional dev-only)
**Location:** `CORS_ORIGINS`

Default CORS includes `http://127.0.0.1:8000,http://localhost:8000` which would allow requests from any localhost port. If a user has malware on their machine listening on port 8000, it can read ARIA API responses.

**Recommendation:** Require explicit `CORS_ORIGINS` configuration in production. Log a warning if localhost is in the origins list.

---

### 14. No Request Body Size Limit on API Endpoints
**File:** `app.py`
**Severity:** MEDIUM
**Location:** Global

While `connect_routes.py` has stream-limited file uploads (100MB CSV, 50MB PDF), general API endpoints have no explicit body size limits. A malicious user could send extremely large JSON payloads to any endpoint.

**Recommendation:** Add a global body size limit via FastAPI middleware or uvicorn `--limit-max-bytes`.

---

### 15. Login Rate Limit Uses In-Memory Dictionary
**File:** `api/auth_routes.py:61-73`
**Severity:** MEDIUM (for multi-user/multi-worker)
**Location:** `_LOGIN_ATTEMPTS`

```python
_LOGIN_ATTEMPTS: dict[str, list[float]] = {}
```

This dictionary is process-local. Under uvicorn with multiple workers, each worker has its own independent counter. An attacker can bypass the rate limit by distributing requests across workers.

**Recommendation:** Either use `ARIA_REQUIRE_SHARED_STORE=1` enforcement (not currently applied to this module) or implement a Redis-backed version.

---

### 16. Missing Type Hints Across Agent Modules
**Files:** `insight_agent/agent.py`, `goal_agent/`, `hypothesis_engine/`, `prescription_engine/`
**Severity:** MEDIUM (code quality)
**Location:** Widespread

Many functions lack return type annotations, making it difficult to verify correctness with static analysis. The `GoalAgent`, `InsightAgent`, and `HypothesisEngine` classes have complex return contracts that benefit greatly from type hints.

---

### 17. Error Messages Potentially Reveal Internal State
**File:** `api/connect_routes.py:417-423`
**Severity:** LOW-MEDIUM
**Location:** `_do_connect()`

The connection error message includes a hint that could reveal internal setup (e.g., "is MySQL server running"). While `sanitize_error()` is called, the hint structure itself reveals server software.

---

### 18. No LLM Output Sanitization for Stored Insights
**Files:** `insight_agent/agent.py`, `insight_rag/`
**Severity:** LOW-MEDIUM
**Location:** Throughout insight generation

LLM-generated insights are stored and served to the frontend. While `insight_rag/enterprise_quality.py` has prompt injection detection, the output (stored in `insights.json`) is not sanitized for XSS or similar injection when rendered by the frontend.

---

## LOW Findings

### 19. Hardcoded Magic Numbers Without Constants
**Files:** Throughout
**Severity:** LOW
**Examples:**
- `app.py:64` — `docs_url="/docs"` — hardcoded path
- `llm_provider.py:343` — `timeout=900` for story generation — unexplained
- `llm_provider.py:439-447` — varied timeout defaults per provider/role

**Recommendation:** Consolidate timeout and limit constants into `core/config.py`.

---

### 20. Multiple Duplicate Constants Across Modules
**Severity:** LOW
**Examples:**
- `CHART_WHITELIST` defined in both `core/validation.py:437` and `insight_agent/agent.py:62` with slightly different values
- `CHART_SPEC_VOCAB` in `insight_agent/agent.py:62` vs `CHART_WHITELIST` in `core/validation.py`

---

### 21. Dead Code — Legacy CSV Upload Without Session Isolation Check
**File:** `api/connect_routes.py:625-712`
**Severity:** LOW
**Location:** `connect_file()` — CSV upload path

The CSV upload creates per-user artifact paths but the validation (`csv_orchestrator`) still references global paths in some code paths. Confusing and potentially inconsistent.

---

### 22. No Structured Logging — Uses `logging.getLogger` Directly
**Files:** Throughout
**Severity:** LOW
**Location:** `logging.getLogger("aria.copilot")`, etc.

Loggers are created with module names but there's no centralized logging configuration. The project mixes `logging` module calls with `print()` in some places. No structured fields for log aggregation tools (JSON logging not used).

---

### 23. `get_session(None)` Returns Default Session — Misuse Risk
**File:** `core/config.py:377-378`
**Severity:** LOW
**Location:** `get_session()`

```python
if user_id is None:
    return dict(DEFAULT_SESSION)
```

This fallback exists for backward compatibility but could be accidentally called with `None` from a bug, returning an empty default session that bypasses user isolation. The function is well-guarded in `get_session_for_request()`, but any direct call with `None` silently returns shared state.

---

## Architecture & Code Quality Observations

### Strengths
1. **Defense in depth:** Multiple layers of SQL injection protection (parameterized queries + `is_read_only_sql()` guard + table/column validation + LIMIT enforcement).
2. **Secret management:** DPAPI encryption for at-rest secrets on Windows, clear documentation of Linux fallback risks.
3. **Cloud consent model:** `_require_cloud_consent()` creates a two-factor gate (env var + consent flag) that prevents accidental cloud data transmission.
4. **Security headers:** CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy all properly configured.
5. **Auth cookie security:** HttpOnly, Secure (default), SameSite=Lax, `__Host-` prefix support.
6. **Input validation:** Goal text validation (`core/validation.py`) catches gibberish and noise before it reaches LLM.
7. **Error sanitization:** `core/errors.py` strips credentials/paths/hosts from all client-facing error messages.
8. **JWT token revocation:** `revoked_tokens` table prevents reuse of logged-out tokens.
9. **User enumeration resistance:** Login/register return generic errors regardless of failure cause.
10. **Path traversal protection:** User artifacts use validated integer IDs, `pathlib.resolve()` checks, and `relative_to()` validation.
11. **Document type detection:** PDF magic bytes check prevents script-executable uploads.
12. **Per-user directory isolation:** All user data (uploads, processed, insights, schemas) stored under `data/users/user_{id}/`.
13. **Prompt injection detection:** `insight_rag/enterprise_quality.py` has a comprehensive blocklist for prompt injection attempts.

### Concerns
1. **Session store is not atomic** — see CRITICAL finding #1.
2. **In-memory state in multi-worker deployments** — rate limits and login guards documented as non-working but only warnings issued.
3. **Large except blocks** — many API routes swallow exceptions silently.
4. **No pagination on history** — `list_history()` defaults to 200 but returns all columns; a large history could be a data exposure vector.
5. **LLM provider model hardcoding** — switching models requires code changes, not runtime configuration.

---

## Dependency Audit

### High or Critical CVEs to Address
- **pydantic v2.13.4** — Current, stable. Verify against latest v2.
- **fastapi v0.141.1** — Verify against latest v0.x or v1.x. v0.115+ has important security fixes.
- **uvicorn v0.52.2** — Verify against latest.
- **langgraph v1.2.11** — Check for known vulnerabilities in graph execution.
- **pillow v12.2.0** — Check CVE database; older versions had RCE vulnerabilities in image processing.
- **sentence-transformers v5.5.0** — Verify; models loaded from disk could be attack vectors.
- **faiss-cpu v1.13.2** — Verify CVE status.

### Missing Critical Dependencies
- **No Redis client** — Multi-worker production requires Redis but it's not in requirements.txt.
- **No explicit pytest** — Test runner not in requirements-dev.txt.
- **No Bandit/Semgrep** — No security linting in dev dependencies.

### Properly Pinned
All packages in `requirements.txt` have exact versions pinned. Good practice.

---

## Test Coverage Observations

### Strengths
- E2E test (`tests/test_e2e_full.py`) covers the full user journey.
- Prediction/forecast tests (`tests/test_prediction_forecast.py`) cover statistical edge cases.
- Insight RAG reliability tests cover outage scenarios.
- Schema validation tests.

### Gaps
- No security-specific tests (SQL injection, SSRF, auth bypass, rate limit bypass).
- No fuzz testing for LLM input validation.
- No multi-worker concurrency tests for session store.
- `tests/test_scorecard.py` uses `__import__()` which is a code smell.

---

## Recommendations (Prioritized)

### Immediately (Before Production)
1. **[CRITICAL]** Fix session store atomicity — move disk write inside the lock.
2. **[HIGH]** Enable `ARIA_REQUIRE_SHARED_STORE=1` in production.
3. **[HIGH]** Audit all `except Exception:` blocks in API routes; add specific exception types and logging.
4. **[MEDIUM]** Increase PBKDF2 iterations to 600,000+ (plan migration).
5. **[MEDIUM]** Add `revoke_all_user_tokens()` for password change.

### Before Next Release
6. **[MEDIUM]** Implement SSRF hostname resolution at validation time (not just blocklist).
7. **[MEDIUM]** Add global request body size limit.
8. **[MEDIUM]** Audit `_exec()` calls in `hypothesis_engine/` and `prescription_engine/`.
9. **[MEDIUM]** Consolidate duplicate chart whitelist constants.
10. **[MEDIUM]** Add Redis to `requirements.txt` for production multi-worker.

### Good Practices Already in Place
- DPAPI encryption for secrets on Windows
- Multi-layer SQL injection protection
- Cloud consent two-factor gate
- Security headers (CSP, X-Frame-Options, etc.)
- User enumeration resistance
- Path traversal prevention
- Document magic byte validation
- Prompt injection detection
- Read-only SQL enforcement
- Secret redaction in logs

---

*Report generated by automated code analysis. Manual review of CRITICAL and HIGH findings recommended before production deployment.*
