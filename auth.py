"""User store, JWT auth, and per-user history for ARIA.

Passwords are hashed with PBKDF2-HMAC-SHA256 (stdlib only, no bcrypt wheels).
Tokens are HS256 JWTs valid for TOKEN_TTL_DAYS. History is kept per user in a
local SQLite database (data/aria.db).
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "aria.db"
JWT_ALGO = "HS256"


def _token_ttl_days() -> int:
    """Resolve JWT lifetime in days from ``TOKEN_TTL_DAYS``.

    Env contract (documented):
      - ``TOKEN_TTL_DAYS``: integer ``1..7`` days, default ``1``.
      - Unset or empty/whitespace -> ``1`` (intended default).
      - Out-of-range integers are clamped to ``1..7``.
      - Non-integer values raise ``ValueError`` (fail clearly, no silent
        fallback to an unsafe value).
    """
    raw = os.getenv("TOKEN_TTL_DAYS")
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        return 1
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid TOKEN_TTL_DAYS={raw!r}: must be an integer 1..7 days."
        )
    return min(7, max(1, value))


TOKEN_TTL_DAYS = _token_ttl_days()
PBKDF2_ITERATIONS = 120_000

_USERNAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]+")


def _is_production() -> bool:
    """Production mode flag (existing convention: ARIA_ENV=production)."""
    return (os.getenv("ARIA_ENV") or "").strip().lower() == "production"


def _load_secret() -> str:
    # JWT_SECRET env override wins (container/prod secret injection); the
    # file fallback preserves existing single-machine deployments.
    env_secret = (os.getenv("JWT_SECRET") or "").strip()
    if _is_production():
        # Fail fast in production: never autogenerate/persist a secret and
        # never fall back to a known/default value. Inject JWT_SECRET.
        if not env_secret:
            raise RuntimeError(
                "JWT_SECRET is required when ARIA_ENV=production. "
                "Inject a strong random secret (e.g. 64 hex chars) via the "
                "JWT_SECRET environment variable."
            )
        if len(env_secret) < 32:
            raise RuntimeError(
                "JWT_SECRET is too short for ARIA_ENV=production "
                f"({len(env_secret)} chars; need >= 32). Inject a strong "
                "random secret."
            )
        return env_secret
    if env_secret:
        return env_secret
    secret_path = BASE_DIR / "data" / ".jwt_secret"
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    if secret_path.exists():
        return secret_path.read_text(encoding="utf-8").strip()
    secret = secrets.token_hex(32)
    secret_path.write_text(secret, encoding="utf-8")
    try:
        os.chmod(secret_path, 0o600)
    except OSError:
        pass
    return secret


SECRET = _load_secret()
# Previous secret accepted for verification only (rotation: set JWT_SECRET to
# the new value and JWT_SECRET_PREVIOUS to the old one until tokens expire).
SECRET_PREVIOUS = (os.getenv("JWT_SECRET_PREVIOUS") or "").strip() or None


_SCHEMA_READY_PATHS: set[str] = set()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create auth tables once per database path (not on every request)."""
    key = str(DB_PATH)
    if key in _SCHEMA_READY_PATHS:
        return
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            goal TEXT NOT NULL,
            sql_used TEXT,
            row_count INTEGER,
            columns TEXT,
            source_type TEXT,
            db_name TEXT,
            dialect TEXT,
            processed_json TEXT,
            insights_json TEXT,
            created_at TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS revoked_tokens (
            token_hash TEXT PRIMARY KEY,
            expires_at TEXT NOT NULL
        )"""
    )
    conn.commit()
    _SCHEMA_READY_PATHS.add(key)


def _decode_token(token: str):
    """Decode with the current secret, falling back to the previous secret."""
    try:
        return jwt.decode(token, SECRET, algorithms=[JWT_ALGO])
    except jwt.InvalidSignatureError:
        if SECRET_PREVIOUS:
            return jwt.decode(token, SECRET_PREVIOUS, algorithms=[JWT_ALGO])
        raise


def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Busy-timeout + WAL: concurrent logins/history writes must not 500
    # with "database is locked". Retries handled by callers' short loops;
    # the timeout below covers the common contention window.
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    except Exception:
        pass
    _ensure_schema(conn)
    return conn


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        _algo, iters, salt_hex, digest_hex = stored.split("$")
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters)
        )
        return hmac.compare_digest(digest.hex(), digest_hex)
    except Exception:
        return False


def register(username: str, password: str) -> int:
    username = (username or "").strip()
    if len(username) < 3 or not _USERNAME_RE.fullmatch(username):
        raise ValueError("Username must be at least 3 characters using letters, digits, . _ -.")
    if len(password) < 8 or len(password) > 128:
        raise ValueError("Password must be 8-128 characters.")
    conn = _get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, _hash_password(password), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        return row["id"]
    except sqlite3.IntegrityError:
        raise ValueError("Username already taken.")
    finally:
        conn.close()


def authenticate(username: str, password: str) -> int | None:
    if not username or not isinstance(username, str):
        return None
    if not isinstance(password, str) or len(password) > 128:
        return None
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username.strip(),)
        ).fetchone()
    finally:
        conn.close()
    if row and _verify_password(password, row["password_hash"]):
        return row["id"]
    # Dummy hash to reduce user-enumeration timing oracle.
    _verify_password("dummy-password-for-timing", "pbkdf2_sha256$120000$" + "00" * 16 + "$" + "00" * 32)
    return None


def issue_token(user_id: int, username: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": now,
        "exp": now + timedelta(days=TOKEN_TTL_DAYS),
    }
    return jwt.encode(payload, SECRET, algorithm=JWT_ALGO)


def verify_token(token: str) -> dict | None:
    try:
        if not token or not isinstance(token, str):
            return None
        payload = _decode_token(token)
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        conn = _get_db()
        try:
            revoked = conn.execute(
                "SELECT 1 FROM revoked_tokens WHERE token_hash = ?", (token_hash,)
            ).fetchone()
        finally:
            conn.close()
        if revoked:
            return None
        # Strict `sub` validation: accept only a positive int user id or a
        # digit string (never bool/float/None/injection strings). int(True)
        # == 1 and int(3.5) == 3 would otherwise alias to another user.
        sub = payload.get("sub")
        if isinstance(sub, bool):
            return None
        if isinstance(sub, int):
            user_id = sub
        elif isinstance(sub, str) and sub.strip().isdigit():
            user_id = int(sub.strip())
        else:
            return None
        if user_id <= 0:
            return None
        return {"user_id": user_id, "username": payload.get("username")}
    except (jwt.PyJWTError, KeyError, TypeError, ValueError):
        return None


def revoke_token(token: str) -> None:
    """Invalidate this JWT until its own expiry without storing the token itself."""
    try:
        payload = _decode_token(token)
        if isinstance(payload, dict) and payload.get("exp") is None:
            return
        expires_at = datetime.fromtimestamp(int(payload["exp"]), timezone.utc).isoformat()
    except (jwt.PyJWTError, KeyError, TypeError, ValueError):
        return
    conn = _get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("DELETE FROM revoked_tokens WHERE expires_at <= ?", (now,))
        conn.execute(
            "INSERT OR REPLACE INTO revoked_tokens (token_hash, expires_at) VALUES (?, ?)",
            (hashlib.sha256(token.encode("utf-8")).hexdigest(), expires_at),
        )
        conn.commit()
    finally:
        conn.close()


def add_history(user_id: int, *, goal: str, sql_used=None, row_count=None,
                columns=None, source_type=None, db_name=None, dialect=None,
                processed=None) -> int:
    conn = _get_db()
    try:
        cur = conn.execute(
            """INSERT INTO history
               (user_id, goal, sql_used, row_count, columns, source_type, db_name, dialect, processed_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id, goal, sql_used, row_count,
                json.dumps(columns) if columns else None,
                source_type, db_name, dialect,
                json.dumps(processed) if processed is not None else None,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def set_history_insights(history_id: int, user_id: int, insights) -> None:
    conn = _get_db()
    try:
        conn.execute(
            "UPDATE history SET insights_json = ? WHERE id = ? AND user_id = ?",
            (json.dumps(insights), history_id, user_id),
        )
        conn.commit()
    finally:
        conn.close()


def list_history(user_id: int, limit: int = 200) -> list[dict]:
    conn = _get_db()
    try:
        rows = conn.execute(
            """SELECT id, goal, sql_used, row_count, columns, source_type, db_name, dialect, created_at, insights_json
               FROM history WHERE user_id = ? ORDER BY id DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
    finally:
        conn.close()
    items = []
    for r in rows:
        items.append(
            {
                "id": r["id"],
                "goal": r["goal"],
                "sql_used": r["sql_used"],
                "row_count": r["row_count"],
                "columns": json.loads(r["columns"]) if r["columns"] else [],
                "source_type": r["source_type"],
                "db_name": r["db_name"],
                "dialect": r["dialect"],
                "created_at": r["created_at"],
                "has_insights": bool(r["insights_json"]),
            }
        )
    return items


def get_history(user_id: int, history_id: int) -> dict | None:
    conn = _get_db()
    try:
        r = conn.execute(
            "SELECT * FROM history WHERE id = ? AND user_id = ?",
            (history_id, user_id),
        ).fetchone()
    finally:
        conn.close()
    if not r:
        return None
    return {
        "id": r["id"],
        "goal": r["goal"],
        "sql_used": r["sql_used"],
        "row_count": r["row_count"],
        "columns": json.loads(r["columns"]) if r["columns"] else [],
        "source_type": r["source_type"],
        "db_name": r["db_name"],
        "dialect": r["dialect"],
        "created_at": r["created_at"],
        "processed": json.loads(r["processed_json"]) if r["processed_json"] else None,
        "insights": json.loads(r["insights_json"]) if r["insights_json"] else None,
    }


def delete_history(user_id: int, history_id: int) -> bool:
    conn = _get_db()
    try:
        cur = conn.execute(
            "DELETE FROM history WHERE id = ? AND user_id = ?",
            (history_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()
