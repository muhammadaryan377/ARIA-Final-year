"""Shared application configuration and per-user runtime session state.

Environment variables are loaded exactly once here (from the project root's
``.env``) so every other module reads them without calling load_dotenv itself.
"""

import logging
import os
import threading
import time
from copy import deepcopy
from pathlib import Path

_logger = logging.getLogger("uvicorn.error")

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Secrets in .env may be stored DPAPI-encrypted (`enc1:<base64>`) so the file
# is unreadable at rest. Decrypt them back into the environment now that dotenv
# has loaded, before any module reads os.getenv for credentials.
from core.secret_box import decrypt, encrypt, PREFIX  # noqa: E402

for _name, _value in list(os.environ.items()):
    if isinstance(_value, str) and _value.startswith(PREFIX):
        try:
            os.environ[_name] = decrypt(_value)
        except Exception:
            pass


# Strict provider isolation: each hosted provider resolves its key ONLY from
# its own saved key or its own env var. Local (Ollama) never uses any API key.
# There is intentionally no cross-provider fallback (e.g. DeepSeek must never
# silently pick up GROQ_API_KEY).
PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "cloud": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "omniroute": "OMNIROUTE_API_KEY",
    "grok": "GROK_API_KEY",
    "local": None,
}


def resolve_provider_key(provider_name, api_keys=None):
    """Return the API key for exactly this provider, or None.

    Order: per-provider saved key -> provider's own env var. Local always
    returns None (Ollama needs no key and must never receive one).
    Grok additionally honours XAI_API_KEY as a fallback alias for GROK_API_KEY.
    """
    if provider_name == "local":
        return None
    saved = (api_keys or {}).get(provider_name)
    if saved:
        return saved
    env_var = PROVIDER_ENV_KEYS.get(provider_name)
    if env_var:
        val = os.getenv(env_var) or None
        if val:
            return val
        if provider_name == "grok":
            return os.getenv("XAI_API_KEY") or None
        return None
    return None


def _encrypt_api_keys(api_keys):
    """Encrypt per-provider API keys for at-rest storage.

    Plaintext values get `enc1:` prefix via DPAPI; already-encrypted
    values pass through unchanged. Non-string values are left as-is.
    """
    if not isinstance(api_keys, dict):
        return api_keys
    out = {}
    for k, v in api_keys.items():
        if isinstance(v, str) and v and not v.startswith(PREFIX):
            try:
                out[k] = encrypt(v)
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out


def _decrypt_api_keys(api_keys):
    """Decrypt at-rest API keys, preserving plaintext for migration.

    Existing plaintext entries (pre-migration) are returned as-is so
    they remain usable; they are re-encrypted on next save_session.
    """
    if not isinstance(api_keys, dict):
        return api_keys
    out = {}
    for k, v in api_keys.items():
        if isinstance(v, str) and v.startswith(PREFIX):
            try:
                out[k] = decrypt(v)
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out


# Per-process cache of decrypted DB passwords, keyed by the enc blob.
# Keeps the per-request rebuild in get_session cheap (one DPAPI decrypt per
# unique blob, then memory hit). Holds plaintext only in memory; never
# logged or persisted to disk/Redis.
_PWD_CACHE: dict = {}
_PWD_CACHE_LOCK = threading.RLock()
_PWD_CACHE_MAX = 256


def _ensure_relational_uri(session):
    """Rebuild a relational db_uri from the encrypted `db` config (schema-agnostic).

    Only when db_type is postgresql/mysql, db_uri is falsy, and the db
    config carries host+db. Decrypt failures leave db_uri as None (never a
    wrong URI). Plaintext (pre-migration) passwords keep working. Never
    logs or persists secrets; plaintext is cached per-process keyed by the
    enc blob.
    """
    try:
        if not isinstance(session, dict):
            return session
        if session.get("db_type") not in ("postgresql", "mysql"):
            return session
        if session.get("db_uri"):
            return session
        cfg = session.get("db")
        if not isinstance(cfg, dict):
            return session
        if not (cfg.get("host") and cfg.get("db")):
            return session
        pwd = cfg.get("password")
        if isinstance(pwd, str) and pwd.startswith(PREFIX):
            try:
                with _PWD_CACHE_LOCK:
                    _cached = _PWD_CACHE.get(pwd)
            except Exception:
                _cached = None
            if _cached:
                pwd = _cached
            else:
                try:
                    from core.secret_box import decrypt as _dec
                    _plain = _dec(pwd)
                except Exception:
                    return session
                if not _plain:
                    return session
                try:
                    with _PWD_CACHE_LOCK:
                        if len(_PWD_CACHE) >= _PWD_CACHE_MAX:
                            _PWD_CACHE.clear()
                        _PWD_CACHE[pwd] = _plain
                except Exception:
                    pass
                pwd = _plain
        if not pwd:
            return session
        from urllib.parse import quote_plus as _qp, quote as _q
        scheme = "mysql+pymysql" if session.get("db_type") == "mysql" else "postgresql+psycopg2"
        session["db_uri"] = (
            f"{scheme}://{_qp(str(cfg.get('user', '')))}:{_qp(str(pwd))}"
            f"@{_q(str(cfg.get('host', '')), safe='')}:{cfg.get('port')}/{_qp(str(cfg.get('db', '')))}"
        )
    except Exception:
        pass
    return session


def _decrypt_session_keys(session):
    """Return a copy of session with api_keys decrypted for runtime use."""
    if not isinstance(session, dict):
        return session
    keys = session.get("api_keys")
    if isinstance(keys, dict):
        session = dict(session)
        session["api_keys"] = _decrypt_api_keys(keys)
    # Scrub legacy persisted password-bearing URIs (older sessions stored
    # user:password in db_uri on disk). CSV sqlite URIs carry no secret.
    try:
        _uri = session.get("db_uri")
        if (isinstance(_uri, str) and "@" in _uri
                and session.get("db_type") in ("postgresql", "mysql")):
            session = dict(session)
            session["db_uri"] = None
    except Exception:
        pass
    # Rebuild relational db_uri (no longer persisted to disk) from the
    # encrypted `db` config so restarted sessions keep working.
    return _ensure_relational_uri(session)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
# Dedicated folder for all schema mapping JSON files, keeping them out of the
# code tree (everything under data/ is gitignored and regenerable).
SCHEMA_DIR = DATA_DIR / "schemas"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PROCESSED_DIR = ARTIFACTS_DIR / "processed_data"
INSIGHTS_DIR = ARTIFACTS_DIR / "insights"
FLOW_REPORTS_DIR = ARTIFACTS_DIR / "flow_reports"

for _dir in (ARTIFACTS_DIR, DATA_DIR, SCHEMA_DIR, PROCESSED_DIR, INSIGHTS_DIR, FLOW_REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

SCHEMA_PATH = SCHEMA_DIR / "schema_mapping_latest.json"
# Global artifact paths below are legacy defaults only (template for
# DEFAULT_SESSION). They MUST NOT be used for authenticated user data —
# per-user sessions always resolve to isolated per-user paths via
# get_user_artifact_paths(). Kept for backward compatibility.
PROCESSED_PATH = PROCESSED_DIR / "processed_data.json"
INSIGHTS_PATH = INSIGHTS_DIR / "insights.json"
SESSION_STORE_PATH = DATA_DIR / "session_store.json"
# Canonical isolated root for per-user artifacts.
USERS_DIR = DATA_DIR / "users"
USERS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared session store (Redis). File persistence is for local deployments.
#
# * ARIA_REDIS_URL (e.g. redis://localhost:6379/0): when set, sessions are
#   written through to Redis (preferred) and read from Redis first, so
#   multi-worker deployments share state. Configured Redis failures return 503.
# * ARIA_REQUIRE_SHARED_STORE=1: fail closed elsewhere (see
#   core.ratelimit._check_shared_store); reads/writes fail closed on outages.
# * File-only mode remains supported when no shared store is configured or required.
# ---------------------------------------------------------------------------
_REDIS_CLIENT = None
_REDIS_INIT_WARNED = False
_REDIS_MISSING_WARNED = False


def redis_url() -> str:
    """Return the configured Redis URL (empty string when not configured)."""
    try:
        return (os.getenv("ARIA_REDIS_URL", "") or "").strip()
    except Exception:
        return ""


def require_shared_store() -> bool:
    """Sessions and rate limits use the same deployment policy."""
    from core.ratelimit import shared_store_required
    return shared_store_required()


def _session_store_unavailable():
    from fastapi import HTTPException
    raise HTTPException(status_code=503, detail="Shared session store is unavailable. Try again shortly.")


def get_redis_client():
    """Return a shared Redis client, or None when not configured/unusable.

    Graceful fallback: None when ARIA_REDIS_URL is unset, when the `redis`
    package is missing (warn once), or when the server is unreachable (warn
    once). Callers enforce the fail-closed policy on None.
    """
    global _REDIS_CLIENT, _REDIS_INIT_WARNED, _REDIS_MISSING_WARNED
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT
    url = redis_url()
    if not url:
        return None
    try:
        import redis as _redis_mod  # type: ignore
    except ImportError:
        if not _REDIS_MISSING_WARNED:
            _REDIS_MISSING_WARNED = True
            _logger.warning(
                "ARIA_REDIS_URL is set but the 'redis' package is missing; "
                "shared session operations will fail (pip install redis)."
            )
        return None
    try:
        client = _redis_mod.StrictRedis.from_url(
            url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2
        )
        client.ping()
    except Exception as exc:
        if not _REDIS_INIT_WARNED:
            _REDIS_INIT_WARNED = True
            _logger.warning(
                "Redis session store unreachable (%s); shared session operations will fail.",
                type(exc).__name__,
            )
        return None
    _REDIS_CLIENT = client
    return _REDIS_CLIENT


def _reset_redis_client_for_tests() -> None:
    """Clear the cached Redis client (tests only)."""
    global _REDIS_CLIENT, _REDIS_INIT_WARNED, _REDIS_MISSING_WARNED
    _REDIS_CLIENT = None
    _REDIS_INIT_WARNED = False
    _REDIS_MISSING_WARNED = False


def _redis_session_key(user_id: int) -> str:
    return f"aria:session:{int(user_id)}"


# ---------------------------------------------------------------------------
# Atomic file I/O with 0600 perms + cross-platform file locking.
#
# Writes are atomic via .tmp + os.replace; tmp files are created 0600 and the
# final file is chmodded 0600 (POSIX) / ensure_private_file. Reads and the
# read-modify-write in save_session take a best-effort lock: fcntl.flock on
# POSIX, msvcrt.locking on Windows, via a sidecar .lock file so the lock
# survives the atomic replace of the data file.
# ---------------------------------------------------------------------------
def _lock_handle(fh, exclusive: bool) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            # msvcrt.locking locks a byte range; lock the first byte.
            fh.seek(0)
            mode = msvcrt.LK_LOCK if exclusive else msvcrt.LK_LOCK
            try:
                msvcrt.locking(fh.fileno(), mode, 1)
            except OSError:
                pass
        else:
            import fcntl

            fcntl.flock(
                fh.fileno(),
                fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
            )
    except Exception:
        pass


def _unlock_handle(fh) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            try:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass


def _read_text_locked(path: Path) -> str:
    """Read a file under a shared lock (best-effort; falls back to plain read)."""
    try:
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+b") as lock_fh:
            _lock_handle(lock_fh, exclusive=False)
            try:
                return path.read_text(encoding="utf-8")
            finally:
                _unlock_handle(lock_fh)
    except Exception:
        return path.read_text(encoding="utf-8")


def _atomic_write_text(target: Path, text: str) -> None:
    """Atomically write text to target via .tmp + replace with 0600 perms."""
    import os as _os

    target.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    tmp_path = Path(tmp_name)
    try:
        with _os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            try:
                fh.flush()
                _os.fsync(fh.fileno())
            except Exception:
                pass
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    try:
        try:
            if os.name != "nt":
                _os.chmod(tmp_path, 0o600)
        except Exception:
            pass
        try:
            from core.secret_box import ensure_private_file as _ensure_private

            _ensure_private(tmp_path)
        except Exception:
            pass
        # Hold an exclusive sidecar lock across the replace to reduce races.
        try:
            lock_path = target.with_suffix(target.suffix + ".lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with open(lock_path, "a+b") as lock_fh:
                _lock_handle(lock_fh, exclusive=True)
                try:
                    _os.replace(tmp_path, target)
                finally:
                    _unlock_handle(lock_fh)
        except Exception:
            _os.replace(tmp_path, target)
        try:
            if os.name != "nt":
                _os.chmod(target, 0o600)
        except Exception:
            pass
        try:
            from core.secret_box import ensure_private_file as _ensure_private2

            _ensure_private2(target)
        except Exception:
            pass
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _coerce_user_id(user_id) -> int:
    """Validate user_id as an int (prevents path traversal via crafted ids)."""
    if isinstance(user_id, bool) or not isinstance(user_id, (int, str)):
        raise ValueError("Invalid user id.")
    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        raise ValueError("Invalid user id.")
    if uid <= 0:
        raise ValueError("Invalid user id.")
    return uid


def _safe_user_dir(base: Path, user_id) -> Path:
    """Return base/user_{id}/, creating it deterministically.

    user_id is coerced to int so names are always `user_<int>` (no
    traversal). The resolved path is verified to stay within base.
    """
    uid = _coerce_user_id(user_id)
    base_resolved = base.resolve()
    candidate = (base_resolved / f"user_{uid}").resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError:
        raise ValueError("Invalid user artifact path.")
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def get_user_root(user_id) -> Path:
    """Canonical isolated root: data/users/user_{id}/ (created on demand)."""
    return _safe_user_dir(USERS_DIR, user_id)


def get_user_artifact_paths(user_id) -> dict:
    """Deterministic per-user artifact locations (all dirs created).

    Layout (compatible with existing per-user convention):
      schemas:   data/schemas/user_{id}/
      processed: artifacts/processed_data/user_{id}/
      insights:  artifacts/insights/user_{id}/
      uploads:   data/uploads/user_{id}/
      documents: artifacts/processed_data/user_{id}/documents/
      root:      data/users/user_{id}/ (+ processed/, insights/, documents/)
    Returns dict with dirs + default file paths for a fresh session.
    """
    uid = _coerce_user_id(user_id)
    schema_dir = _safe_user_dir(SCHEMA_DIR, uid)
    processed_dir = _safe_user_dir(PROCESSED_DIR, uid)
    insights_dir = _safe_user_dir(INSIGHTS_DIR, uid)
    upload_dir = _safe_user_dir(DATA_DIR / "uploads", uid)
    documents_dir = processed_dir / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)
    # Canonical root with matching subdirs (future-proof isolation).
    root = get_user_root(uid)
    for sub in ("processed", "insights", "documents"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "schema_dir": schema_dir,
        "processed_dir": processed_dir,
        "insights_dir": insights_dir,
        "upload_dir": upload_dir,
        "documents_dir": documents_dir,
        "schema_path": schema_dir / "schema_mapping_latest.json",
        "processed_path": processed_dir / "processed_data.json",
        "insights_path": insights_dir / "insights.json",
    }


def _ensure_user_paths(session: dict, user_id) -> dict:
    """Point a per-user session at isolated paths (never global shared files).

    Keeps existing per-DB filenames (e.g. processed_data_<db>.json) when they
    already live under the user's own dir; otherwise resets to the per-user
    defaults. Creates parent dirs deterministically.
    """
    try:
        paths = get_user_artifact_paths(user_id)
    except ValueError:
        return session
    uid = _coerce_user_id(user_id)
    tag = f"user_{uid}"
    for key, default_path, user_dir in (
        ("schema_path", paths["schema_path"], paths["schema_dir"]),
        ("processed_path", paths["processed_path"], paths["processed_dir"]),
        ("insights_path", paths["insights_path"], paths["insights_dir"]),
    ):
        cur = session.get(key)
        try:
            cur_s = str(cur) if cur is not None else ""
            # Keep per-user per-DB files already under this user's dir.
            # Segment match (never bare substring: "user_1" is inside
            # "user_10"). Resolve + verify the parent dir name instead.
            _keep = False
            if cur_s:
                try:
                    _keep = Path(cur_s).resolve().parent == user_dir.resolve()
                except Exception:
                    _keep = False
            if _keep:
                Path(cur_s).parent.mkdir(parents=True, exist_ok=True)
                continue
            session[key] = default_path
            default_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            try:
                session[key] = default_path
            except Exception:
                pass
    return session

SOURCE_TYPES = {"relational", "semi-structured", "document"}
PROVIDER_NAMES = {"local", "cloud", "deepseek", "omniroute", "grok", "openai", "anthropic"}
DB_TYPES = {"postgresql", "mysql"}

DEFAULT_SESSION = {
    "provider": None,       # LLMProvider instance
    "provider_name": None,  # 'local' | 'cloud'
    "source_type": None,    # 'relational' | 'semi-structured'
    "db": None,             # relational DB config
    "db_type": "postgresql",  # 'postgresql' | 'mysql' (relational sources only)
    "db_uri": None,
    "db_name": None,        # name of the connected database / table
    "schema_path": SCHEMA_PATH,   # schema file for the active database
    "processed_path": PROCESSED_PATH,
    "insights_path": INSIGHTS_PATH,
    "dialect": "postgresql",     # 'postgresql'|'mysql' for relational, 'sqlite' for CSV/PDF
    "pending_history_id": None,  # most recent history row awaiting insights
}

SESSION = deepcopy(DEFAULT_SESSION)
SESSION_STORE: dict[int, tuple[float, dict]] = {}
SESSION_LOCK = threading.RLock()
_SESSION_FILE_MTIME: float | None = None

# Load persisted sessions from disk (survives server restarts).
def _session_to_disk_dict(session: dict) -> dict:
    """Return a JSON-serializable copy for disk/Redis (encrypted keys)."""
    sess_disk = dict(session)
    if isinstance(sess_disk.get("api_keys"), dict):
        sess_disk["api_keys"] = _encrypt_api_keys(sess_disk["api_keys"])
    # Never persist live provider objects; reconstructed via _ensure_provider.
    if not isinstance(sess_disk.get("provider"), (str, type(None))):
        sess_disk["provider"] = None
    # Never persist relational db_uri (embeds DB password in plaintext).
    try:
        if sess_disk.get("db_type") in ("postgresql", "mysql"):
            sess_disk.pop("db_uri", None)
        _linked = sess_disk.get("linked_db")
        if isinstance(_linked, dict):
            _linked = dict(_linked)
            if _linked.get("db_type") in ("postgresql", "mysql"):
                _linked.pop("db_uri", None)
            sess_disk["linked_db"] = _linked
    except Exception:
        pass
    return sess_disk


def _disk_dict_to_session(loaded: dict) -> dict:
    """Restore Path types + decrypt keys for a disk/Redis payload."""
    from pathlib import Path as _P

    loaded = dict(loaded)
    for k in ("schema_path", "processed_path", "insights_path"):
        if k in loaded and isinstance(loaded[k], str):
            loaded[k] = _P(loaded[k])
    return _decrypt_session_keys(loaded)


def _redis_save_one(user_id: int, created: float, session: dict) -> None:
    """Write shared state; never acknowledge a failed required write."""
    import json
    try:
        client = get_redis_client()
        if client is None:
            if require_shared_store() or redis_url():
                _session_store_unavailable()
            return
        payload = json.dumps({"created": float(created),
                              "session": _session_to_disk_dict(session)}, default=str)
        client.setex(_redis_session_key(user_id), int(SESSION_TTL_SECONDS), payload)
    except Exception as exc:
        _logger.warning("Session store write failed (%s)", type(exc).__name__)
        if require_shared_store() or redis_url():
            _session_store_unavailable()


def _redis_load_one(user_id: int):
    """Distinguish a missing session from an unavailable shared backend."""
    import json
    try:
        client = get_redis_client()
        if client is None:
            if require_shared_store() or redis_url():
                _session_store_unavailable()
            return None
        raw = client.get(_redis_session_key(user_id))
        if raw is None:
            return None
        data = json.loads(raw)
        return float(data["created"]), _disk_dict_to_session(data["session"])
    except Exception as exc:
        _logger.warning("Session store read failed (%s)", type(exc).__name__)
        if require_shared_store() or redis_url():
            _session_store_unavailable()
        return None


def _load_session_store():
    """Load persisted session store from disk (locked read; file fallback)."""
    try:
        import json as _json
        if SESSION_STORE_PATH.exists():
            data = _json.loads(_read_text_locked(SESSION_STORE_PATH))
            for uid_str, (created, session) in data.items():
                uid = int(uid_str)
                loaded = _disk_dict_to_session(dict(session))
                SESSION_STORE[uid] = (float(created), loaded)
    except Exception:
        pass

_load_session_store()

# Regenerable-output hygiene (see RETENTION.md): per-user forecast/run
# outputs are a bounded cache, never active user data. Only files matching
# regenerable patterns are evicted, oldest-first by mtime.
REGENERABLE_PATTERNS = ("forecast_input_*.json",)
REGENERABLE_KEEP_PER_USER = 20


def prune_regenerable_outputs(processed_dir, patterns=None, keep=None) -> int:
    """Evict oldest regenerable run outputs in `processed_dir`, keeping `keep`.

    Only matches `patterns` (default forecast_input_*.json); never touches
    active files (processed_data*.json, insights*.json, uploads, documents).
    Returns the number of files removed. Never raises.
    """
    import fnmatch as _fn
    try:
        base = Path(processed_dir)
        if not base.is_dir():
            return 0
        pats = patterns or REGENERABLE_PATTERNS
        limit = REGENERABLE_KEEP_PER_USER if keep is None else int(keep)
        victims = []
        for p in base.iterdir():
            try:
                if not p.is_file():
                    continue
                if any(_fn.fnmatch(p.name, pat) for pat in pats):
                    victims.append(p)
            except Exception:
                continue
        victims.sort(key=lambda p: p.stat().st_mtime)
        excess = len(victims) - max(0, limit)
        removed = 0
        for p in victims[:max(0, excess)]:
            try:
                p.unlink(missing_ok=True)
                removed += 1
            except Exception:
                continue
        return removed
    except Exception:
        return 0


# Session store hygiene: cap the number of cached per-user sessions and drop
# sessions that have not been touched for SESSION_TTL_SECONDS. Prevents
# unbounded memory growth in long-running multi-user servers.
#
# Storage backend is explicit: the in-memory SESSION_STORE plus the JSON file
# below are LOCAL/DEV only (single process). A multi-worker production
# deployment needs a shared store (Redis/DB): set ARIA_WORKERS>1 only with
# ARIA_REDIS_URL configured, or ARIA_REQUIRE_SHARED_STORE=1 to fail closed
# (see core.ratelimit._check_shared_store). The session file carries
# encrypted-at-rest secrets and is written 0600 (see save_session).
SESSION_MAX_USERS = 100
SESSION_TTL_SECONDS = 4 * 60 * 60  # 4 hours


def check_session_store_shared() -> None:
    """Fail closed when workers > 1 without a shared session/rate-limit store."""
    try:
        from core.ratelimit import _check_shared_store
        _check_shared_store()
    except Exception:
        raise


def _ensure_provider(session):
    """Reconstruct only the user's selected provider, without a silent switch."""
    prov = session.get("provider")
    if prov is not None and not isinstance(prov, str):
        return
    name = session.get("provider_name")
    if not name:
        return
    try:
        from llm_provider import create_provider
        api_key = resolve_provider_key(name, session.get("api_keys", {}))
        session["provider"] = create_provider(provider=name, api_key=api_key, models=session.get("provider_models", {}).get(name))
        session.pop("_provider_error", None)
    except Exception as exc:
        _logger.warning("Selected provider initialization failed (%s)", type(exc).__name__)
        session["provider"] = None
        session["_provider_error"] = "Selected AI provider is unavailable. Check its settings and credentials."


def _copy_session(session):
    """Copy the session so callers can mutate without leaking changes.

    Deep-copies nested plain values (db config, api_keys, lists) so
    concurrent requests never share mutable state. The live LLM provider
    object holds thread locks and cannot be deep-copied — it is re-attached
    by reference.
    """
    if not isinstance(session, dict):
        return session
    prov = session.get("provider")
    try:
        import copy as _copy
        out = _copy.deepcopy({k: v for k, v in session.items() if k != "provider"})
    except Exception:
        out = dict(session)
        out.pop("provider", None)
    if prov is not None:
        out["provider"] = prov
    return out


def get_session(user_id=None):
    """Return the current session for a user, or a fresh default copy.

    Returns a copy of the session dict so each request gets its
    own mutable state (nested configs deep-copied; only the live LLM
    provider object is shared by reference). Use ``save_session`` to
    persist modifications back to the store.

    NOTE: get_session(None) returns the DEFAULT_SESSION template only and
    must never be used to serve authenticated user data. Per-user sessions
    always resolve to isolated per-user artifact paths (never the shared
    global PROCESSED_PATH/INSIGHTS_PATH).
    """
    if user_id is None:
        return dict(DEFAULT_SESSION)

    try:
        user_id = _coerce_user_id(user_id)
    except ValueError:
        raise ValueError("Invalid user id.")

    now = time.time()
    global _SESSION_FILE_MTIME
    with SESSION_LOCK:
        # Redis-first when configured: shared store is authoritative across
        # workers. File/memory mode applies only when Redis is not configured.
        _redis_hit = _redis_load_one(user_id)
        # Redis misses must not resurrect stale worker/file snapshots.
        if _redis_hit is None and (redis_url() or require_shared_store()):
            fresh = dict(DEFAULT_SESSION)
            _ensure_user_paths(fresh, user_id)
            return fresh
        if _redis_hit is not None:
            try:
                _created, _sess = _redis_hit
                if now - float(_created) <= SESSION_TTL_SECONDS:
                    _ensure_user_paths(_sess, user_id)
                    _ensure_relational_uri(_sess)
                    SESSION_STORE[user_id] = (now, _sess)
                    _ensure_provider(_sess)
                    return _copy_session(_sess)
            except Exception:
                pass
        if redis_url() or require_shared_store():
            # Expired or invalid shared records cannot revive file snapshots.
            fresh = dict(DEFAULT_SESSION)
            _ensure_user_paths(fresh, user_id)
            return fresh
        # mtime-gated reconcile: avoid re-reading/parsing session_store.json
        # on every request. Only reload when the file actually changed.
        # save_session updates _SESSION_FILE_MTIME so our own writes never
        # trigger a redundant reload; external writes still get picked up.
        try:
            import json as _json
            if SESSION_STORE_PATH.exists():
                mtime = SESSION_STORE_PATH.stat().st_mtime
                if _SESSION_FILE_MTIME is None or mtime != _SESSION_FILE_MTIME:
                    data = _json.loads(_read_text_locked(SESSION_STORE_PATH))
                    for uid_str, (created, session) in data.items():
                        uid = int(uid_str)
                        if uid not in SESSION_STORE or SESSION_STORE[uid][0] < float(created):
                            loaded = _disk_dict_to_session(dict(session))
                            SESSION_STORE[uid] = (float(created), loaded)
                    _SESSION_FILE_MTIME = mtime
        except Exception:
            pass

        # Opportunistic cleanup of expired sessions.
        if len(SESSION_STORE) >= SESSION_MAX_USERS or (len(SESSION_STORE) > 10 and len(SESSION_STORE) % 5 == 0):
            expired = [
                uid for uid, (created, _session) in SESSION_STORE.items()
                if now - created > SESSION_TTL_SECONDS
            ]
            for uid in expired:
                del SESSION_STORE[uid]
            # If still at capacity, drop the oldest entry.
            while len(SESSION_STORE) >= SESSION_MAX_USERS:
                oldest = min(SESSION_STORE, key=lambda uid: SESSION_STORE[uid][0])
                del SESSION_STORE[oldest]

        entry = SESSION_STORE.get(user_id)
        if entry is None:
            fresh = dict(DEFAULT_SESSION)
            _ensure_user_paths(fresh, user_id)
            entry = (now, fresh)
            SESSION_STORE[user_id] = entry
        else:
            _ensure_user_paths(entry[1], user_id)
            SESSION_STORE[user_id] = (now, entry[1])
        _ensure_relational_uri(entry[1])
        _ensure_provider(entry[1])
        # Warm the shared store on file/memory hits so the next worker
        # finds this session in Redis (best-effort, never raises).
        if redis_url():
            _redis_save_one(user_id, SESSION_STORE[user_id][0], SESSION_STORE[user_id][1])
        return _copy_session(entry[1])


def save_session(user_id, session):
    """Persist a modified session copy back to the store and to disk.

    In-memory SESSION_STORE keeps decrypted runtime keys; only the
    on-disk JSON copy carries encrypted `api_keys`. When ARIA_REDIS_URL is
    configured the session is also written through to Redis (preferred for
    multi-worker); Redis failures raise 503. File writes
    are atomic (.tmp + replace) with 0600 perms and a sidecar lock.
    """
    user_id = _coerce_user_id(user_id)
    _ensure_user_paths(session, user_id)
    now = time.time()
    with SESSION_LOCK:
        _redis_save_one(user_id, now, session)
        SESSION_STORE[user_id] = (now, _copy_session(session))
        # Also write to disk so sessions survive server restarts.
        try:
            import json as _json
            disk_store = {}
            for uid, (created, sess) in SESSION_STORE.items():
                disk_store[str(uid)] = (created, _session_to_disk_dict(sess))
            _atomic_write_text(
                SESSION_STORE_PATH, _json.dumps(disk_store, default=str)
            )
            try:
                global _SESSION_FILE_MTIME
                _SESSION_FILE_MTIME = SESSION_STORE_PATH.stat().st_mtime
            except Exception:
                pass
        except Exception:
            pass



def get_session_for_request(request):
    """Resolve the authenticated user's session or raise 401.

    Never returns DEFAULT_SESSION for missing/invalid/expired tokens.
    Accepts Authorization Bearer header (preferred) or the `aria_token` /
    `__Host-aria_token` httpOnly cookies set at login. Genuinely public
    routes (login/register/health/providers) must not use this helper.
    """
    from fastapi import HTTPException

    token = None
    try:
        authorization = request.headers.get("authorization", "") or ""
    except Exception:
        authorization = ""
    if isinstance(authorization, str) and authorization.lower().startswith("bearer "):
        candidate = authorization[7:].strip()
        # Guard against frontend sending literal "null"/"undefined"/empty.
        if candidate and candidate.lower() not in ("null", "undefined", "none"):
            token = candidate
    if not token:
        try:
            cookies = getattr(request, "cookies", {}) or {}
            token = cookies.get("aria_token") or cookies.get("__Host-aria_token") or None
        except Exception:
            token = None
    if not token or not isinstance(token, str):
        raise HTTPException(status_code=401, detail="Not authenticated. Provide a valid Bearer token.")
    try:
        import auth
        payload = auth.verify_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Not authenticated. Provide a valid Bearer token.")
    if not payload:
        raise HTTPException(status_code=401, detail="Not authenticated. Provide a valid Bearer token.")
    try:
        user_id = _coerce_user_id(payload.get("user_id"))
    except ValueError:
        raise HTTPException(status_code=401, detail="Not authenticated. Provide a valid Bearer token.")
    return get_session(user_id)
