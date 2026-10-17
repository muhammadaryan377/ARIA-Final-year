"""Safe error shaping: never leak credentials, paths, or driver internals."""

from __future__ import annotations

import re

_PATTERNS = [
    # scheme://user:password@...
    (re.compile(r"://([^:/@\s]+):[^@/\s]*@"), r"://\1:***@"),
    # quoted passwords
    (re.compile(r"(?i)(password\s*[=:]\s*)['\"]?[^\s'\",;]+['\"]?"), r"\1***"),
    # connection strings
    (re.compile(r"(?i)\b(postgresql|mysql|sqlite)(\+\w+)?://[^\s]+"), "[connection-string]"),
    # absolute filesystem paths
    (re.compile(r"(?:[A-Za-z]:\\|\/)[^\s\"']*"), "[path]"),
    # IPv4 addresses
    (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "[host]"),
]


def sanitize_error(exc_or_msg: object, max_len: int = 300) -> str:
    """Redact secrets/paths/hosts from an error message for client responses."""
    try:
        msg = str(exc_or_msg)
    except Exception:
        return "An error occurred."
    for pattern, repl in _PATTERNS:
        try:
            msg = pattern.sub(repl, msg)
        except Exception:
            continue
    msg = " ".join(msg.split())
    return msg[:max_len] if len(msg) > max_len else msg


# Canonical ARIA API error shape (converged contract):
#   {ok: False, success: False, code, detail, request_id}
# Both guarded router errors ({ok:false,...}) and global FastAPI errors
# (HTTPException / validation / unhandled 500 / unknown-API 404) use this
# shape: ok is always False (never success-like), request_id is always
# present (mirrors X-Request-ID), detail is sanitize_error-cleaned.
def error_payload(request_or_id=None, code: str = "REQUEST_FAILED",
                  detail: str = "Request failed.") -> dict:
    """Build the canonical error body for API error responses."""
    if isinstance(request_or_id, str):
        rid = request_or_id or "unknown"
    else:
        try:
            rid = (getattr(getattr(request_or_id, "state", None), "request_id", None)
                   or request_or_id.headers.get("X-Request-ID"))
        except Exception:
            rid = None
        rid = rid or "unknown"
    return {
        "ok": False,
        "success": False,
        "code": code or "REQUEST_FAILED",
        "detail": sanitize_error(detail or "Request failed."),
        "request_id": rid,
    }


# ── Shared typed-error helpers (maintainability) ──────────────────────────
# Central tuple of *typed* errors for DB-touching guards so routers/services
# never need broad `except Exception`. Covers stdlib + installed drivers
# (SQLAlchemy, psycopg2, pymysql, sqlite3). Import-time safe: every optional
# dependency is guarded by `except ImportError` (never bare).

import sqlite3 as _sqlite3


def _db_error_types() -> tuple:
    errs: list = [OSError, ValueError, KeyError, TypeError, AttributeError,
                  RuntimeError, ConnectionError, TimeoutError]
    try:
        errs.append(_sqlite3.Error)
    except Exception:  # pragma: no cover - sqlite3 always ships
        pass
    try:
        from sqlalchemy.exc import SQLAlchemyError as _SAE
        errs.append(_SAE)
    except ImportError:
        pass
    try:
        import psycopg2 as _pg
        errs.append(_pg.Error)
    except ImportError:
        pass
    try:
        import pymysql as _my
        errs.append(_my.MySQLError)
    except ImportError:
        pass
    seen = []
    for e in errs:
        if isinstance(e, type) and e not in seen:
            seen.append(e)
    return tuple(seen)


DB_ERRORS: tuple = _db_error_types()

#: Pure-Python (non-DB) typed errors for parsers/mappers/scorers.
PURE_ERRORS: tuple = (ValueError, KeyError, TypeError, AttributeError)


def get_request_id(request_or_id=None) -> str:
    """Best-effort request id for structured logs (never raises)."""
    if isinstance(request_or_id, str) and request_or_id:
        return request_or_id
    try:
        rid = getattr(getattr(request_or_id, "state", None), "request_id", None)
        if rid:
            return str(rid)
        headers = getattr(request_or_id, "headers", None)
        if headers is not None:
            try:
                rid = headers.get("X-Request-ID")
            except (ValueError, KeyError, TypeError, AttributeError):
                rid = None
            if rid:
                return str(rid)
    except (ValueError, KeyError, TypeError, AttributeError):
        pass
    return "unknown"


#: Canonical Decision Center error codes (hyp.txt #100, PHASE 4).
#: Routes return these inside the shared {ok, success, code, detail}
#: shape; helpers below never raise and never leak internals.
DECISION_ERROR_CODES = frozenset({
    "NOT_FOUND", "NOT_A_DECISION", "QUARANTINED", "INVALID_STATUS",
    "INVALID_SCENARIO", "SCENARIO_UNSUPPORTED", "INSUFFICIENT_DATA",
    "ANALYSIS_FAILED", "UNAUTHORIZED", "FORBIDDEN",
    "CAPABILITY NOT FOUND",
})


def decision_error(code: str, detail: str,
                   request_or_id=None) -> dict:
    """Build a canonical Decision Center error body (additive helper)."""
    code = code if code in DECISION_ERROR_CODES else "REQUEST_FAILED"
    return error_payload(request_or_id, code=code, detail=detail)


def log_and_sanitize(logger, exc: BaseException, request_or_id=None,
                     msg: str = "operation failed") -> str:
    """Log full traceback once; return a sanitized client-safe message."""
    try:
        rid = get_request_id(request_or_id)
    except (ValueError, KeyError, TypeError, AttributeError):
        rid = "unknown"
    try:
        logger.exception("%s request_id=%s: %s", msg, rid, exc)
    except (ValueError, KeyError, TypeError, AttributeError):
        pass
    return sanitize_error(exc)
