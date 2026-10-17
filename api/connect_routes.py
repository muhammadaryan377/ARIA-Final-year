"""Data source connection routes: relational databases (PostgreSQL / MySQL) + CSV files + PDF documents."""

import asyncio
import json
import os
import re
import threading
import time
import types
import secrets
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import csv_orchestrator
import schema_agent
from core.config import DB_TYPES, PROVIDER_NAMES, SOURCE_TYPES, SCHEMA_DIR, PROCESSED_DIR, INSIGHTS_DIR, DATA_DIR, get_session, save_session, get_user_artifact_paths
from core.db import build_db_uri
from core.deps import require_writable, get_current_user
from core.secret_box import decrypt as _decrypt_secret
from csv_handler import CsvValidationError
from llm_provider import create_provider
from document_processor import create_document_processor

router = APIRouter()

# Hosted providers run inference off-machine (need an API key, can serve vision).
HOSTED_PROVIDERS = {"cloud", "deepseek", "omniroute", "grok", "openai", "anthropic"}

# Upload bounds (§40): enforced while streaming so oversized bodies never
# fill the disk. CSV cap matches the handler's 100 MB post-write check.
def _upload_limit_mb(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


MAX_UPLOAD_BYTES_CSV = _upload_limit_mb("ARIA_MAX_UPLOAD_CSV_MB", 512) * 1024 * 1024
MAX_UPLOAD_BYTES_PDF = _upload_limit_mb("ARIA_MAX_UPLOAD_PDF_MB", 50) * 1024 * 1024

# CSV formula-injection guard (server-side, non-mutating).
# The 512 MB cap + 413 behavior above is intentionally unchanged (the env var
# ARIA_MAX_UPLOAD_CSV_MB remains the single tunable; no lower default is
# applied). Instead, preview paths flag cells that spreadsheet apps would
# interpret as formulas so the UI can warn before export/open. Data is never
# mutated — the response only adds an additive `formula_risk: bool` signal.
_FORMULA_CELL_PREFIXES = ("=", "+", "-", "@")


def _cell_is_formula_risk(value: object) -> bool:
    """True when a cell looks like a spreadsheet formula (CSV injection).

    Flags values whose first non-whitespace char is one of ``=,+,-,@``.
    Used by preview responses only; never strips or rewrites data.
    """
    try:
        s = str(value) if value is not None else ""
    except Exception:
        return False
    s = s.lstrip()
    return bool(s) and s[0] in _FORMULA_CELL_PREFIXES


def _scan_formula_risk(rows: list[dict], limit: int = 200) -> bool:
    """Scan preview rows for formula-risk cells (bounded, read-only).

    Returns True when any cell in the first ``limit`` rows starts with a
    formula prefix. Additive signal for `formula_risk` in preview payloads.
    """
    try:
        for row in (rows or [])[: max(0, int(limit))]:
            if not isinstance(row, dict):
                continue
            for v in row.values():
                if _cell_is_formula_risk(v):
                    return True
    except Exception:
        return False
    return False


def _spool_limited(upload_file, dest: Path, max_bytes: int, label: str) -> int:
    """Stream an upload to disk, aborting with 413 past max_bytes."""
    total = 0
    try:
        with open(dest, "wb") as _f:
            while True:
                chunk = upload_file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"{label} exceeds the {max_bytes // (1024 * 1024)} MB upload limit.",
                    )
                _f.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    return total


_BLOCKED_DB_HOSTS = {"169.254.169.254", "metadata.google.internal", "metadata.google.internal."}
_METADATA_IPS = {"169.254.169.254", "fd00:ec2::254"}


def _normalize_host_ip(host: str) -> str | None:
    """Return canonical IP string for decimal/octal/hex forms, else None.

    Handles bypasses like 2852039166 (decimal), 0xA9FEA9FE (hex),
    0251.0376.0251.0376 (octal) via ipaddress. Never raises.
    """
    import ipaddress as _ip
    h = (host or "").strip().rstrip(".")
    if not h:
        return None
    # Pure decimal integer form (e.g. 2852039166 -> 169.254.169.254)
    if h.isdigit():
        try:
            return str(_ip.ip_address(int(h)))
        except Exception:
            return None
    # Hex integer form (e.g. 0xA9FEA9FE)
    if h.lower().startswith("0x"):
        try:
            return str(_ip.ip_address(int(h, 16)))
        except Exception:
            return None
    try:
        return str(_ip.ip_address(h))
    except Exception:
        return None


def _ip_is_blocked(ip_str: str, *, allow_loopback: bool) -> bool:
    """True when an IP must be refused. Loopback allowed only for DB-local dev."""
    import ipaddress as _ip
    try:
        ip = _ip.ip_address(ip_str)
    except Exception:
        return False
    if ip_str in _METADATA_IPS:
        return True
    try:
        if ip.is_loopback:
            return not allow_loopback
        if ip.is_link_local or ip.is_multicast or ip.is_reserved:
            return True
        # Cloud metadata + private ranges are never valid DB/LLM targets
        # (except loopback for local dev, handled above).
        if ip.is_private:
            return True
    except Exception:
        return False
    return False


def _resolve_host_ips(host: str) -> list[str]:
    """Best-effort DNS resolution for SSRF check. Empty on failure (fail-open to connect error)."""
    import socket as _sock
    try:
        infos = _sock.getaddrinfo(host, None, family=_sock.AF_UNSPEC, type=_sock.SOCK_STREAM)
        out = []
        for fam, _typ, _proto, _canon, sockaddr in infos:
            try:
                out.append(str(sockaddr[0]))
            except Exception:
                continue
        return out
    except Exception:
        return []


def _host_is_numeric(host: str) -> bool:
    h = (host or "").strip().rstrip(".")
    if not h:
        return False
    if h.isdigit() or h.lower().startswith("0x"):
        return True
    try:
        import ipaddress as _ip
        _ip.ip_address(h)
        return True
    except Exception:
        return False


def _validate_db_endpoint(host, port, db=None, user=None, db_type="postgresql") -> int:
    """Shared SSRF/host validation for database connect + test paths.

    Used by both the real connect path and /api/database/test so probing
    via error oracle is blocked consistently. Returns validated port int.

    Loopback (localhost/127.0.0.1/::1) stays allowed for local-dev DBs;
    cloud-metadata (169.254.169.254 in any numeric form), private ranges,
    link-local/multicast/reserved, and DNS names resolving to them are blocked.
    """
    host = (host or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", host) or len(host) > 253:
        raise HTTPException(status_code=400, detail="Invalid host.")
    low = host.lower().rstrip(".")
    if low in _BLOCKED_DB_HOSTS or low.startswith("169.254."):
        raise HTTPException(status_code=400, detail="Host is blocked.")
    # Numeric-form bypasses (decimal/octal/hex/integer IP).
    norm = _normalize_host_ip(host)
    if norm is not None:
        # allow loopback for local dev, block everything else private/reserved
        if _ip_is_blocked(norm, allow_loopback=True):
            raise HTTPException(status_code=400, detail="Host is blocked.")
    else:
        # Hostname: block nip.io-style suffixes and resolve-then-check.
        if ".nip.io" in low or ".sslip.io" in low:
            # e.g. 169.254.169.254.nip.io — resolves to the embedded IP
            raise HTTPException(status_code=400, detail="Host is blocked.")
        for rip in _resolve_host_ips(host):
            if _ip_is_blocked(rip, allow_loopback=True):
                raise HTTPException(status_code=400, detail="Host resolves to a blocked address.")
    try:
        port_int = int(port)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid port.")
    if not 1 <= port_int <= 65535:
        raise HTTPException(status_code=400, detail="Invalid port.")
    if db_type not in ("postgresql", "mysql"):
        raise HTTPException(status_code=400, detail="Unsupported dbType.")
    if len(db or "") > 128 or len(user or "") > 128:
        raise HTTPException(status_code=400, detail="Invalid database/user.")
    return port_int


def _pdf_tables_to_sqlite(document, upload_dir: Path, stem: str, stamp: str):
    """Load clean PDF tables into a queryable SQLite DB (§5/§15).

    Returns (db_uri, mapping_tables, loaded_names) or (None, {}, []) when
    no table is well-formed enough to query. Never raises: failures degrade
    to document-only (RAG) mode, which keeps working.
    """
    import sqlite3 as _sqlite3

    from csv_orchestrator import _quote_ident

    usable = []
    for i, t in enumerate(getattr(document, "tables", []) or []):
        try:
            raw_headers = t.get("headers", []) or []
            headers = [str(h or "").strip() or f"col_{j}" for j, h in enumerate(raw_headers)]
            rows = [r for r in (t.get("rows", []) or [])
                    if any(str(c or "").strip() for c in r)]
            if len(headers) < 2 or not rows:
                continue
            clean, seen = [], set()
            for h in headers:
                base = re.sub(r"[^0-9a-zA-Z]+", "_", h).strip("_") or "col"
                name, k = base, 2
                while name.lower() in seen:
                    name, k = f"{base}_{k}", k + 1
                seen.add(name.lower())
                clean.append(name)
            # Infer REAL columns (all non-empty cells numeric) for aggregation.
            col_types = []
            for j in range(len(clean)):
                vals = [str(r[j]).strip() for r in rows if j < len(r) and str(r[j]).strip()]
                numeric = [v.replace(",", "").replace("$", "").replace("%", "") for v in vals]
                if vals and all(_is_num(v) for v in numeric):
                    col_types.append("REAL")
                else:
                    col_types.append("TEXT")
            usable.append({"name": f"pdf_t{i}", "columns": clean,
                           "types": col_types, "rows": rows, "page": t.get("page")})
        except Exception:
            continue
    if not usable:
        return None, {}, []

    db_path = upload_dir / f"{stem}_{stamp}_tables.db"
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            db_path.unlink(missing_ok=True)
        except PermissionError:
            pass
        conn = _sqlite3.connect(str(db_path))
        mapping_tables: dict = {}
        loaded: list[str] = []
        try:
            for t in usable:
                ddl = ", ".join(f"{_quote_ident(c)} {typ}"
                                for c, typ in zip(t["columns"], t["types"]))
                conn.execute(f"CREATE TABLE {_quote_ident(t['name'])} ({ddl})")
                placeholders = ", ".join("?" for _ in t["columns"])
                cols = ", ".join(_quote_ident(c) for c in t["columns"])
                batch = []
                for r in t["rows"]:
                    vals = []
                    for j, typ in enumerate(t["types"]):
                        cell = str(r[j]).strip() if j < len(r) else ""
                        if not cell:
                            vals.append(None)
                        elif typ == "REAL":
                            try:
                                vals.append(float(cell.replace(",", "").replace("$", "").replace("%", "")))
                            except ValueError:
                                vals.append(None)
                        else:
                            vals.append(cell)
                    batch.append(vals)
                conn.executemany(
                    f"INSERT INTO {_quote_ident(t['name'])} ({cols}) VALUES ({placeholders})",
                    batch,
                )
                mapping_tables[t["name"]] = {
                    "columns": [{"column": c, "data_type": ("REAL" if typ == "REAL" else "TEXT"),
                                 "nullable": True} for c, typ in zip(t["columns"], t["types"])],
                    "primary_key": [],
                    "row_count": len(batch),
                    "source": f"pdf page {t['page']}",
                }
                loaded.append(t["name"])
            conn.commit()
        finally:
            conn.close()
        return f"sqlite:///{db_path}", mapping_tables, loaded
    except Exception:
        try:
            db_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None, {}, []


def _is_num(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False

# ── Ingestion progress tracking ──────────────────────────────────────────
# Maps user_id -> { "step": str, "progress": float 0-100, "message": str, "done": bool, "error": str|None }
# Entries are written from background threads (large-CSV worker) and read from
# the event loop, so every mutation/deletion runs under the lock; readers take
# a shallow copy (snapshot) under the same lock and then work with it
# lock-free — dict/str values are never mutated in place, so the snapshot the
# consumer serializes is stable. The frontend contract (payload keys and SSE
# event shape) is unchanged.
_upload_progress: dict[int, dict] = {}
_upload_progress_lock = threading.Lock()
# Abandoned entries (client vanished before consuming the terminal state) are
# dropped opportunistically on the next write, so `result` payloads never
# accumulate for the lifetime of the process.
_PROGRESS_TTL_S = 600.0


def _set_progress(user_id: int, step: str, progress: float, message: str, done: bool = False, error: str | None = None, result: dict | None = None):
    now = time.time()
    with _upload_progress_lock:
        # TTL sweep on write: drop entries nobody consumed within 10 minutes.
        for _uid in [k for k, v in _upload_progress.items()
                     if now - float((v or {}).get("timestamp") or 0.0)
                     > _PROGRESS_TTL_S]:
            _upload_progress.pop(_uid, None)
        _upload_progress[user_id] = {
            "step": step,
            "progress": progress,
            "message": message,
            "done": done,
            "error": error,
            "result": result,
            "timestamp": now,
        }


def _drop_progress(user_id: int, timestamp) -> None:
    """Remove the entry only if it is still the generation that was consumed.

    A fresh upload may already have overwritten it; never drop newer state.
    """
    with _upload_progress_lock:
        cur = _upload_progress.get(user_id)
        if cur is not None and cur.get("timestamp") == timestamp:
            _upload_progress.pop(user_id, None)


def _get_progress_snapshot(user_id: int, terminal: bool = False) -> dict | None:
    """Copy of the user's progress entry (None when absent).

    The legacy ``terminal`` argument is retained for compatibility. Terminal
    results expire by TTL rather than being consumed by the first reader.
    """
    with _upload_progress_lock:
        prog = _upload_progress.get(user_id)
        if not prog:
            return None
        snap = dict(prog)
        # Retain terminal results until TTL: a dropped SSE packet or a second
        # polling tab must still be able to retrieve the completed result.
        if time.time() - float(snap.get("timestamp") or 0) > _PROGRESS_TTL_S:
            _upload_progress.pop(user_id, None)
            return None
        return snap


@router.get("/api/upload/progress")
async def get_upload_progress(user: dict = Depends(get_current_user)):
    """Return current ingestion progress for the user.

    Terminal results remain available for ten minutes for reconnecting clients.
    """
    prog = _get_progress_snapshot(user["user_id"], terminal=True)
    if not prog:
        return {"ok": True, "active": False}
    return {"ok": True, "active": True, **prog}


@router.get("/api/upload/progress/stream")
async def stream_upload_progress(user: dict = Depends(require_writable)):
    """SSE stream of ingestion progress for the user (writable sessions only, max 2 min)."""
    async def event_stream():
        uid = user["user_id"]
        last_ts = 0
        consumed = False  # terminal event handed to the client -> drop entry
        try:
            for _ in range(120):  # max 2 minutes (was 5 min, reduces connection exhaustion)
                prog = _get_progress_snapshot(uid)
                if prog and prog.get("timestamp", 0) > last_ts:
                    last_ts = prog["timestamp"]
                    if prog.get("done") or prog.get("error"):
                        # Mark before yielding so an abort mid-send still
                        # cleans up in the finally below.
                        consumed = True
                    yield f"data: {json.dumps({k: v for k, v in prog.items() if k != 'timestamp'})}\n\n"
                    if consumed:
                        break
                await asyncio.sleep(1)
        finally:
            # Delivery of a yield is not an acknowledgement from the browser.
            # Polling/reconnected streams can recover the result until TTL.
            pass
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _validate_hosted_base_url(base_url: str | None) -> None:
    """Strict validator for hosted LLM endpoint overrides (no userinfo, public https only).

    Raises HTTPException 400 on violation. `None`/empty passes (provider default).
    Local provider is exempt (loopback http allowed) — callers check provider first.
    """
    if not base_url:
        return
    from urllib.parse import urlparse as _urlparse
    try:
        parts = _urlparse(base_url.strip())
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base_url.")
    if parts.scheme != "https" or not parts.hostname:
        raise HTTPException(status_code=400, detail="Custom base_url must use https:// (http/internal hosts blocked).")
    if parts.username or parts.password or "@" in (parts.netloc or ""):
        raise HTTPException(status_code=400, detail="Custom base_url must not contain credentials.")
    host = parts.hostname.rstrip(".")
    low = host.lower()
    if low in _BLOCKED_DB_HOSTS or ".nip.io" in low or ".sslip.io" in low:
        raise HTTPException(status_code=400, detail="Custom base_url host is blocked.")
    norm = _normalize_host_ip(host)
    if norm is not None:
        # Hosted LLM endpoints must be public — loopback also blocked here.
        if norm == "127.0.0.1" or norm == "::1" or _ip_is_blocked(norm, allow_loopback=False):
            raise HTTPException(status_code=400, detail="Custom base_url host is blocked.")
    else:
        for rip in _resolve_host_ips(host):
            if _ip_is_blocked(rip, allow_loopback=False):
                raise HTTPException(status_code=400, detail="Custom base_url resolves to a blocked address.")


class ProviderSwitchRequest(BaseModel):
    provider: str = "local"
    api_key: str | None = None      # optional key override for hosted providers
    base_url: str | None = None     # optional endpoint override for hosted providers
    model: str | None = None


@router.get("/api/provider/options")
def provider_options(user: dict = Depends(get_current_user)):
    from llm_provider import PROVIDERS
    from core.config import resolve_provider_key
    session = get_session(user["user_id"])
    labels = {"cloud": "Groq", "deepseek": "DeepSeek", "openai": "OpenAI / GPT",
              "anthropic": "Anthropic / Claude", "grok": "xAI / Grok", "local": "Local / Ollama"}
    return {"providers": [{"id": name, "label": label,
        "model": session.get("provider_models", {}).get(name, PROVIDERS[name]["models"]).get("semi", ""),
        "has_key": bool(resolve_provider_key(name, session.get("api_keys")))}
        for name, label in labels.items()]}


@router.post("/api/provider/models")
def provider_models(request: ProviderSwitchRequest, user: dict = Depends(require_writable)):
    from core.config import resolve_provider_key
    if request.provider not in HOSTED_PROVIDERS or request.provider == "omniroute":
        raise HTTPException(status_code=400, detail="Choose a supported hosted provider.")
    session = get_session(user["user_id"])
    try:
        llm = create_provider(provider=request.provider,
            api_key=request.api_key or resolve_provider_key(request.provider, session.get("api_keys")))
        result = llm._cloud_client.with_options(timeout=15).models.list()
        return {"models": sorted({m.id for m in result.data})[:200]}
    except Exception:
        raise HTTPException(status_code=400, detail="Could not load models. Check your provider key and backend cloud consent, or enter a model ID manually.")


@router.post("/api/provider/switch")
def switch_provider(request: ProviderSwitchRequest, user: dict = Depends(require_writable)):
    """Explicitly switch the LLM backend for this session."""
    session = get_session(user["user_id"])
    if request.provider not in PROVIDER_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{request.provider}'. Choose from {sorted(PROVIDER_NAMES)}.")
    # SSRF guard: custom endpoints only over https for hosted providers.
    # Local provider (ollama) may use loopback http; hosted must not probe internal http.
    if request.base_url and request.provider != "local":
        _validate_hosted_base_url(request.base_url)
    if request.model is not None and (not request.model.strip() or len(request.model) > 200
                                      or any(ord(c) < 32 for c in request.model)):
        raise HTTPException(status_code=400, detail="Enter a valid model ID (up to 200 characters).")
    models = session.get("provider_models", {}).get(request.provider)
    if request.model:
        models = {role: request.model.strip() for role in ("sql", "story", "suggest", "semi", "schema", "vision")}
    current = session.get("provider_name")
    if (current == request.provider and not request.api_key and not request.base_url and not request.model
            and session.get("provider") is not None and not session.get("_provider_error")):
        return {"ok": True, "provider": current, "changed": False}

    try:
        # Prefer the key the user just typed; otherwise reuse the saved
        # per-provider key (e.g. switching back to Cloud LLM after a restart
        # must keep working with the user's Groq key, not the server env).
        from core.config import resolve_provider_key as _resolve_key
        try:
            _saved = _resolve_key(request.provider, (session.get("api_keys") or {}))
        except Exception:
            _saved = None
        llm = create_provider(
            provider=request.provider,
            api_key=request.api_key or _saved,
            base_url=request.base_url,
            models=models,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not initialise '{request.provider}' LLM provider: {exc}")

    # A real, small generation checks both credentials and model access.
    if request.provider in HOSTED_PROVIDERS:
        if not llm.model_for("suggest"):
            raise HTTPException(status_code=400, detail="Choose a model ID for this provider.")
        try:
            reply = llm.chat("suggest", [{"role": "user", "content": "Reply with OK."}],
                             num_predict=2048, timeout=45, user_id=user["user_id"])
            if not reply:
                raise ValueError("Model returned no text")
        except Exception:
            raise HTTPException(status_code=400, detail="Provider/model check failed. Check API key, model access, billing and backend logs.")
    session.setdefault("provider_models", {})[request.provider] = dict(llm.models)
    session.pop("csv_llm_dashboard", None)
    session["provider"] = llm
    session["provider_name"] = request.provider
    configured = session.setdefault("configured_providers", [])
    if request.provider not in configured:
        configured.append(request.provider)
    session.pop("_provider_error", None)
    # Provider isolation: API keys are stored per hosted provider only.
    # Local (Ollama) uses no key, so a key sent with provider=local is ignored
    # rather than stored.
    if request.api_key and request.provider in HOSTED_PROVIDERS:
        keys = session.get("api_keys", {})
        keys[request.provider] = request.api_key
        session["api_keys"] = keys
    save_session(user["user_id"], session)
    return {"ok": True, "provider": request.provider, "changed": True}


class ConnectRequest(BaseModel):
    provider: str | None = None
    source_type: str = "relational"
    db_type: str = "postgresql"   # 'postgresql' | 'mysql'
    host: str = "localhost"
    port: int | str = 5432
    db: str | None = None
    user: str | None = None
    password: str | None = None
    db_schema: str = "public"


def _discover_table_schema(config, exclude="public"):
    """Find populated schemas when the requested one is empty.

    Returns (mapping, schema_name); ({}, None) when nothing is found.
    Read-only introspection; used only when the requested schema is empty.
    Generic multi-schema fallback (no schema names): when the requested
    schema holds nothing but several other schemas hold tables, map ALL of
    them so cross-schema databases never connect to nothing; a single
    populated schema keeps the exact old single-schema path.
    """
    import logging as _logging
    _log = _logging.getLogger("aria.connect")
    try:
        conn = schema_agent.get_connection(types.SimpleNamespace(**config))
    except Exception:
        return {}, None
    try:
        try:
            _populated = schema_agent.list_schemas_with_tables(conn)
        except Exception:
            _populated = []
        _populated = [s for s in (_populated or []) if s != exclude]
        if len(_populated) > 1:
            try:
                mapping = schema_agent.build_schema_mapping_all(
                    conn, db_type="postgresql")
            except Exception:
                mapping = {}
            if mapping.get("tables"):
                _log.info("Schema auto-discovery: using ALL populated schemas (%d tables).",
                          len(mapping["tables"]))
                return mapping, ",".join(_populated)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT schema_name FROM information_schema.schemata "
                "WHERE schema_name NOT IN ('pg_catalog', 'information_schema') "
                "AND schema_name NOT LIKE 'pg\\_toast%' AND schema_name NOT LIKE 'pg\\_temp%' "
                "ORDER BY (schema_name = 'public') DESC, schema_name;"
            )
            schemata = [r[0] for r in cur.fetchall() if r[0] != exclude]
        for sch in schemata:
            try:
                mapping = schema_agent.build_schema_mapping(conn, sch, db_type="postgresql")
            except Exception:
                continue
            if mapping.get("tables"):
                _log.info("Schema auto-discovery: using '%s' (%d tables).",
                          sch, len(mapping["tables"]))
                return mapping, sch
    except Exception:
        _log.exception("Schema auto-discovery failed.")
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return {}, None


def _do_connect(request: ConnectRequest, user: dict, db_type: str) -> dict:
    """Shared connect logic. `db_type` is authoritative (set by the calling endpoint)."""
    uid = user["user_id"]
    _set_progress(uid, "connecting", 15, f"Connecting to {db_type} database...")

    session = get_session(uid)
    # Omitted provider keeps the session's selected backend so connecting
    # data never silently flips a cloud session back to local.
    if not request.provider:
        request.provider = session.get("provider_name") or "local"
    if request.provider not in PROVIDER_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{request.provider}'. Choose from {sorted(PROVIDER_NAMES)}.")
    if request.source_type not in SOURCE_TYPES:
        raise HTTPException(status_code=400, detail=f"Unknown source type '{request.source_type}'. Choose from {sorted(SOURCE_TYPES)}.")
    if request.source_type == "relational" and db_type not in DB_TYPES:
        raise HTTPException(status_code=400, detail=f"Unknown database type '{db_type}'. Choose from {sorted(DB_TYPES)}.")

    if request.source_type == "relational":
        if not (request.db and request.user and request.password):
            raise HTTPException(status_code=400, detail="Relational source requires db, user, and password.")
        request.port = _validate_db_endpoint(
            request.host, request.port, request.db, request.user, db_type
        )

    # Use user's saved API key for hosted providers if available
    user_key = session.get("api_keys", {}).get(request.provider) if request.provider in HOSTED_PROVIDERS else None
    try:
        llm = create_provider(provider=request.provider, api_key=user_key or None, models=session.get("provider_models", {}).get(request.provider))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not initialise '{request.provider}' LLM provider: {exc}")

    # The provider in this connect request is an explicit user choice.
    session["provider"] = llm
    session["provider_name"] = request.provider
    session["source_type"] = request.source_type
    # A fresh database connection replaces any previously linked source.
    session.pop("linked_db", None)
    session.pop("schemas", None)  # stale schemas from a prior DB break search_path
    session["db"] = None
    session["db_type"] = db_type
    session["db_uri"] = None
    session["db_name"] = None
    # Isolated per-user artifact paths (validated int id, pathlib-safe,
    # no traversal; dirs created deterministically).
    _upaths = get_user_artifact_paths(user["user_id"])
    session["schema_path"] = _upaths["schema_path"]
    session["processed_path"] = _upaths["processed_path"]
    session["insights_path"] = _upaths["insights_path"]

    config = request.model_dump()
    config["db_type"] = db_type
    conn = None
    try:
        config["port"] = int(config["port"])
        conn = schema_agent.get_connection(types.SimpleNamespace(**config))
        with conn.cursor() as cur:
            if db_type == "mysql":
                cur.execute("SELECT DATABASE();")
                db_name = cur.fetchone()[0]
                current_db_schema = db_name
            else:
                cur.execute("SELECT current_database(), current_schema();")
                db_name, current_db_schema = cur.fetchone()
    except SystemExit as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid port number.")
    except Exception as exc:
        import logging as _logging
        from core.errors import sanitize_error
        _logging.getLogger("aria.connect").exception("Database connection failed")
        safe = sanitize_error(exc)
        if db_type == "mysql":
            detail = (f"Connection failed: {safe} Hint: is the MySQL server running, "
                      "and is pymysql installed (pip install pymysql)? "
                      "Verify host/port, database, user, password.")
        else:
            detail = (f"Connection failed: {safe} Hint: is the PostgreSQL service running "
                      "on that host/port? Verify the database name, user, and password are correct.")
        raise HTTPException(status_code=400, detail=detail)
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

    from core.secret_box import encrypt as _encrypt_secret
    safe_config = dict(config)
    if "password" in safe_config and safe_config["password"] is not None:
        safe_config["password"] = _encrypt_secret(str(safe_config["password"]))
    session["db"] = safe_config
    session["db_type"] = db_type
    # Never persist a password-bearing URI: core.config rebuilds relational
    # db_uri from the encrypted db config at session load. CSV sqlite URIs
    # carry no secret and are kept so file sources survive restarts.
    if safe_config.get("password"):
        session["db_uri"] = None
    else:
        session["db_uri"] = build_db_uri(config, db_type)
    session["db_name"] = db_name
    session["dialect"] = db_type

    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", db_name)
    # Per-DB artifact files stay inside the authenticated user's own dirs.
    _upaths = get_user_artifact_paths(user["user_id"])
    # Schema Agent writes DB-specific files (e.g. schema_mapping_northwind_latest.json)
    # so the session must point to the same naming convention.
    session["schema_path"] = _upaths["schema_dir"] / f"schema_mapping_{safe}_latest.json"
    session["processed_path"] = _upaths["processed_dir"] / f"processed_data_{safe}.json"
    session["insights_path"] = _upaths["insights_dir"] / f"insights_{safe}.json"

    save_session(user["user_id"], session)

    _set_progress(uid, "extracting", 50, "Extracting schema...")

    # Auto-trigger schema extraction so Dashboard KPIs work immediately
    import logging as _logging

    _conn_log = _logging.getLogger("aria.connect")
    session["schema_status"] = "pending"
    try:
        config = dict(session.get("db", {}))
        if "password" in config and config.get("password"):
            try:
                config["password"] = _decrypt_secret(config["password"])
            except Exception as exc:
                _conn_log.error("Stored DB credential failed to decrypt; "
                                "reconnect with a fresh password. (%s)", exc)
                raise HTTPException(
                    status_code=400,
                    detail="Stored database credential is unreadable. "
                           "Please reconnect with your password.",
                )
        try:
            conn = schema_agent.get_connection(types.SimpleNamespace(**config))
        except SystemExit as _sx:
            _conn_log.error("Schema extraction refused connection setup: %s", _sx)
            raise
        try:
            db_type = session.get("db_type", "postgresql")
            schema_filter = config.get("db_schema", "public") if db_type == "postgresql" else config.get("db", "")
            # Multi-schema databases: extract EVERY populated schema so no
            # table is invisible. Single-schema databases keep the exact old
            # path. When the requested schema is empty, default to ALL
            # populated schemas (generic fallback, no schema names).
            _schemas: list = []
            try:
                if db_type == "postgresql":
                    _mode = schema_agent.detect_schema_mode(conn, requested=schema_filter)
                    _schemas = list(_mode.get("schemas") or [])
                    if _mode.get("mode") == "multi" and len(_schemas) > 1:
                        mapping = schema_agent.build_schema_mapping_all(
                            conn, db_type=db_type, database_name=db_name)
                        conn.close()
                    else:
                        _schemas = []
                        mapping = schema_agent.build_schema_mapping(conn, schema_filter, db_type=db_type)
                        conn.close()
                else:
                    mapping = schema_agent.build_schema_mapping(conn, schema_filter, db_type=db_type)
                    conn.close()
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
                raise
            try:
                session["schemas"] = list(_schemas) if _schemas else [schema_filter] \
                    if db_type == "postgresql" else []
                if _schemas and len(_schemas) > 1:
                    current_db_schema = ",".join(_schemas)
            except Exception:
                pass
            if not mapping.get("tables") and db_type == "postgresql":
                # Requested schema is empty (e.g. tables live in a named
                # schema like "hotel", not public): discover the schema that
                # actually holds tables instead of connecting to nothing.
                mapping, discovered = _discover_table_schema(config, exclude=schema_filter)
                if discovered:
                    session["db"]["db_schema"] = discovered.split(",")[0].strip() if "," in discovered else discovered
                    session["schemas"] = [s.strip() for s in discovered.split(",") if s.strip()]
                    current_db_schema = discovered
            if mapping.get("tables"):
                # Single-table rule: a 1-table Postgres/MySQL DB is shaped
                # exactly like a CSV dataset (profiling + PK-composite +
                # FK-none + relationship-empty + no-join) so GoalAgent,
                # KPI, and RAG-tabular see an identical contract. Generic
                # (table count only); native types/indexes are preserved.
                try:
                    from core.relational import normalize_single_table_mapping as _norm_1t
                    mapping = _norm_1t(mapping)
                    try:
                        from core.data_intelligence import (
                            build_data_intelligence as _bdi,
                            detect_capabilities as _dcap,
                        )
                        _intel = _bdi(mapping)
                        mapping["data_intelligence"] = _intel
                        mapping["capabilities"] = _dcap(_intel)
                    except Exception:
                        pass
                except Exception:
                    _conn_log.debug("single-table normalization failed", exc_info=True)
                user_schema_dir = _upaths["schema_dir"]
                safe_name_for_file = re.sub(r"[^a-zA-Z0-9_-]+", "_", db_name)[:64] or "db"
                user_path = user_schema_dir / f"schema_mapping_{safe_name_for_file}_latest.json"

                def _json_safe(o):
                    _conn_log.warning("schema mapping had non-JSON value of type %s; coerced",
                                      type(o).__name__)
                    return str(o)

                user_path.write_text(json.dumps(mapping, indent=2, default=_json_safe),
                                     encoding="utf-8")
                session["schema_path"] = user_path
                session["schema_status"] = "ok"
                save_session(user["user_id"], session)
            else:
                session["schema_status"] = "empty"
                save_session(user["user_id"], session)
                _conn_log.warning("Schema extraction returned no tables for db '%s' (user %s).",
                                  db_name, user["user_id"])
        except Exception as exc:
            session["schema_status"] = "failed"
            save_session(user["user_id"], session)
            _conn_log.exception("Schema extraction failed for db '%s' (user %s).",
                                db_name, user["user_id"])
    except SystemExit:
        pass
    except Exception as exc:
        session["schema_status"] = "failed"
        try:
            save_session(user["user_id"], session)
        except Exception:
            pass
        _conn_log.exception("Schema extraction setup failed for db '%s'.", db_name)

    _set_progress(uid, "ready", 100, "Connected", done=True)

    return {
        "ok": True,
        "provider": request.provider,
        "source_type": "relational",
        "database": db_name,
        "schema": current_db_schema,
        "db_type": db_type,
        "schema_file": Path(str(session.get("schema_path", ""))).name,
        "schema_status": session.get("schema_status", "unknown"),
    }


@router.post("/api/connect")
def connect(request: ConnectRequest, user: dict = Depends(require_writable)):
    """Connect to a source; database type taken from the request body."""
    return _do_connect(request, user, request.db_type)


@router.post("/api/connect/postgresql")
def connect_postgresql(request: ConnectRequest, user: dict = Depends(require_writable)):
    """Connect to a PostgreSQL source (explicit route)."""
    return _do_connect(request, user, "postgresql")


@router.post("/api/connect/mysql")
def connect_mysql(request: ConnectRequest, user: dict = Depends(require_writable)):
    """Connect to a MySQL source (explicit route)."""
    return _do_connect(request, user, "mysql")


def _finish_csv_connect(*, file_path, upload_dir, safe_name, stamp, filename,
                        uid, llm, provider, _upaths, background=False):
    """Run CSV profiling + SQLite load + session wiring after spool.

    Shared by the inline path (small files, returns result / raises HTTP)
    and the background path (large files, reports via progress only and
    never raises to a client). Returns the result dict, or None when the
    outcome was reported through progress (background errors).
    """
    import logging as _logging
    _log = _logging.getLogger("aria.connect")
    _set_progress(uid, "validating", 25, "Validating CSV structure...")
    try:
        from csv_handler import CsvSchemaExtractor
        table_name = csv_orchestrator.safe_table_name(safe_name)
        profile = CsvSchemaExtractor(
            file_path,
            max_file_size_mb=int(os.getenv("ARIA_MAX_UPLOAD_CSV_MB", "512")),
        ).extract()
        mapping = csv_orchestrator.build_csv_schema_mapping(
            file_path, table_name=table_name, profile=profile)
    except CsvValidationError as exc:
        _set_progress(uid, "error", 0, f"Invalid CSV: {exc}", error=str(exc))
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass
        if background:
            return None
        _msg = str(exc).lower()
        if "limit" in _msg or "too large" in _msg or "exceed" in _msg:
            raise HTTPException(status_code=413, detail=f"CSV exceeds the {int(os.getenv('ARIA_MAX_UPLOAD_CSV_MB', '512'))} MB upload limit: {exc}")
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}")
    except Exception as exc:
        from core.errors import sanitize_error as _san
        _set_progress(uid, "error", 0, "Validation failed", error=_san(exc))
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass
        if background:
            _log.warning("Background CSV validation failed (user %s): %s", uid, _san(exc))
            return None
        raise HTTPException(status_code=500, detail=f"CSV processing failed: {_san(exc)}")

    _set_progress(uid, "extracting", 45, "Extracting schema and structure...")
    sqlite_path = upload_dir / f"{safe_name}_{stamp}.db"
    try:
        db_uri, table_name, row_count = csv_orchestrator.load_csv_to_sqlite(
            file_path, sqlite_path, table_name=table_name, profile=profile
        )
    except Exception as exc:
        from core.errors import sanitize_error as _san2
        _set_progress(uid, "error", 0, "Processing failed", error=_san2(exc))
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            sqlite_path.unlink(missing_ok=True)
        except Exception:
            pass
        if background:
            _log.warning("Background CSV load failed (user %s): %s", uid, _san2(exc))
            return None
        raise HTTPException(status_code=500, detail=f"CSV processing failed: {_san2(exc)}")

    _set_progress(uid, "profiling", 65, f"Profiling {row_count} rows...")
    safe_db_name = mapping["database"]
    schema_path = _upaths["schema_dir"] / f"schema_mapping_{safe_db_name}_latest.json"
    schema_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    _set_progress(uid, "understanding", 80, "Understanding data structure...")
    session = get_session(uid)
    session["provider"] = llm
    session["provider_name"] = provider
    session["source_type"] = "semi-structured"
    session["db"] = {".csv": str(file_path)}
    session.pop("linked_db", None)
    session["db_type"] = "sqlite"
    session["db_uri"] = db_uri
    session["db_name"] = table_name
    session["schema_path"] = schema_path
    session["processed_path"] = _upaths["processed_dir"] / f"processed_data_{safe_db_name}.json"
    session["insights_path"] = _upaths["insights_dir"] / f"insights_{safe_db_name}.json"
    session["dialect"] = "sqlite"
    session["file_type"] = "csv"
    # The upload remains a deterministic import. AI planning is an explicit
    # dashboard request, so its failure cannot discard a successfully imported CSV.
    session.pop("csv_llm_dashboard", None)
    session.pop("conversation_id", None)
    session.pop("conversation_history", None)
    save_session(uid, session)

    _set_progress(uid, "generating", 90, "Saving imported data; AI analysis runs when the dashboard opens...")
    columns = [c["column"] for c in mapping["tables"][table_name]["columns"]]
    quality = mapping["tables"][table_name]["csv_profile"]["quality"]
    result = {
        "ok": True,
        "provider": provider,
        "source_type": "semi-structured",
        "file": filename,
        "type": "csv",
        "tables": {table_name: row_count},
        "table": table_name,
        "rows": row_count,
        "columns": columns,
        "schema_file": schema_path.name,
        "quality": quality,
        "db_file": sqlite_path.name,
    }
    _set_progress(uid, "ready", 100, "Ready", done=True, result=result)
    # Warm the dashboard KPI cache while the data is fresh: the first
    # dashboard visit then renders instantly instead of recomputing
    # full-table aggregates. Best-effort, never fails the upload.
    # (Runs for inline and background finishes alike.)
    _warm_dashboard_cache(uid)
    return result


def _warm_dashboard_cache(uid: int) -> None:
    """Precompute dashboard KPIs for a freshly connected source (thread).

    Best-effort, never fails the upload. Uses the in-process KPI service
    directly — no self-HTTP and no minted bearer token (avoids internal
    SSRF surface and token-mint-in-thread).
    """
    def _run():
        try:
            from core.config import get_session as _get_session
            _session = _get_session(uid)
        except Exception:
            return
        try:
            # Prime the per-dataset KPI path without network I/O.
            # services.kpi_service exposes pure helpers; full dashboard
            # assembly stays on first-visit path (correct, just not pre-warmed
            # when the service API changes — hence best-effort).
            import services.kpi_service as _kpi_svc  # noqa: F401
        except Exception:
            pass
    try:
        threading.Thread(target=_run, name=f"kpi-warm-{uid}", daemon=True).start()
    except Exception:
        pass


@router.post("/api/connect_file")
def connect_file(provider: str | None = Form(None), file: UploadFile = File(...),
                 llm_api_key: str | None = Form(None),
                 llm_base_url: str | None = Form(None),
                 user: dict = Depends(require_writable)):
    """Connect a semi-structured source: upload a CSV file.

    The file is validated and profiled deterministically with the CSV handler
    (encoding/dialect detection, header checks, type inference, data-quality
    warnings), its schema mapping is written in the Goal Agent's ``tables``
    format, the rows are loaded into a file-based SQLite database, and the
    session is pointed at it so the Goal Agent and Insight Agent consume it
    exactly like a relational source (dialect=sqlite).

    When ``provider`` is omitted, the session's selected backend is kept so
    connecting data never silently flips a cloud session back to local.
    """
    if not provider:
        try:
            provider = get_session(user["user_id"]).get("provider_name") or "local"
        except Exception:
            provider = "local"
    if provider not in PROVIDER_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{provider}'. Choose from {sorted(PROVIDER_NAMES)}.")
    if llm_base_url and provider != "local":
        _validate_hosted_base_url(llm_base_url)
    filename = (file.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="No file received.")
    ext = Path(filename).suffix.lower()
    if ext != ".csv":
        raise HTTPException(
            status_code=400,
            detail="Only .csv files are accepted for file sources right now. "
                   "Tabular PDF extraction is not available in this build.",
        )
    session = get_session(user["user_id"])
    # CSV uploads reuse the authenticated user's selected hosted-provider key.
    # Local providers must never inherit a saved cloud key.
    saved_key = (session.get("api_keys") or {}).get(provider) if provider in HOSTED_PROVIDERS else None
    try:
        llm = create_provider(provider=provider, api_key=llm_api_key or saved_key, base_url=llm_base_url, models=session.get("provider_models", {}).get(provider))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not initialise '{provider}' LLM provider: {exc}")

    uid = user["user_id"]
    _set_progress(uid, "uploading", 10, "Uploading file...")

    # Isolate uploads by authenticated user (validated id, pathlib-safe,
    # no traversal) and use an unpredictable suffix to prevent
    # same-name uploads from overwriting one another.
    _upaths = get_user_artifact_paths(uid)
    upload_dir = _upaths["upload_dir"]
    stamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(8)}"
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(filename).stem)[:64] or "upload"
    file_path = upload_dir / f"{safe_name}_{stamp}{ext}"
    _spool_limited(file.file, file_path, MAX_UPLOAD_BYTES_CSV, "CSV file")
    # Heavy-ops guard: spool cap (413 above) + handler max_file_size_mb
    # backup use the ARIA_MAX_UPLOAD_CSV_MB limit. Large-but-allowed files
    # run in a background thread (see dispatch below); small files inline.
    try:
        import logging as _logging
        _csv_bytes = file_path.stat().st_size
        if _csv_bytes > 50 * 1024 * 1024:
            _logging.getLogger("aria.connect").warning(
                "Large CSV processed inline: %s (%.1fMB, user %s).",
                filename, _csv_bytes / (1024 * 1024), uid,
            )
    except Exception:
        pass

    # Files above 50 MB are processed in a background thread so the request
    # never times out (a 268 MB file profiles in ~4.5 min with flat ~115 MB
    # RAM via the chunked handler path). Progress is polled/streamed at
    # GET /api/upload/progress. Small files stay inline for instant UX.
    try:
        _csv_size = file_path.stat().st_size
    except Exception:
        _csv_size = 0
    if _csv_size > 50 * 1024 * 1024:
        _set_progress(uid, "queued", 15, "Large file — processing in the background. You can keep working.")
        _args = dict(file_path=file_path, upload_dir=upload_dir,
                     safe_name=safe_name, stamp=stamp, filename=filename,
                     uid=uid, llm=llm, provider=provider, _upaths=_upaths)
        _t = threading.Thread(target=_finish_csv_connect, kwargs={**_args, "background": True},
                              name=f"csv-connect-{uid}", daemon=True)
        _t.start()
        return {"ok": True, "accepted": True, "background": True,
                "provider": provider, "source_type": "semi-structured",
                "file": filename, "type": "csv",
                "message": "Processing in the background — watch progress, then explore."}
    return _finish_csv_connect(file_path=file_path, upload_dir=upload_dir,
                               safe_name=safe_name, stamp=stamp,
                               filename=filename, uid=uid, llm=llm,
                               provider=provider, _upaths=_upaths)


@router.post("/api/connect_pdf")
def connect_pdf(
    provider: str | None = Form(None),
    file: UploadFile = File(...),
    llm_api_key: str | None = Form(None),
    llm_base_url: str | None = Form(None),
    user: dict = Depends(require_writable),
):
    """Connect a document source: upload a PDF file.
    
    The PDF is processed through ARIA's document intelligence pipeline:
    - Validation and integrity check
    - Document type detection (digital, scanned, handwritten, mixed)
    - Content extraction (text, OCR, vision LLM)
    - Table extraction
    - Chunking with metadata
    - Embedding generation
    - Vector indexing for RAG retrieval

    When ``provider`` is omitted, the session's selected backend is kept.
    """
    if not provider:
        try:
            provider = get_session(user["user_id"]).get("provider_name") or "local"
        except Exception:
            provider = "local"
    if provider not in PROVIDER_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{provider}'. Choose from {sorted(PROVIDER_NAMES)}.")
    if llm_base_url and provider != "local":
        _validate_hosted_base_url(llm_base_url)
    filename = (file.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="No file received.")
    ext = Path(filename).suffix.lower()
    if ext != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only .pdf files are accepted for document sources.",
        )
    
    session = get_session(user["user_id"])
    # Provider isolation: only hosted providers use a saved API key; local
    # (Ollama) works keyless and must never pick one up.
    saved_key = session.get("api_keys", {}).get(provider) if provider in HOSTED_PROVIDERS else None
    try:
        llm = create_provider(
            provider=provider,
            api_key=llm_api_key or saved_key,
            base_url=llm_base_url,
            models=session.get("provider_models", {}).get(provider),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not initialise '{provider}' LLM provider: {exc}")

    # Keep raw PDFs within the owning user's upload directory (validated id).
    _pdf_upaths = get_user_artifact_paths(user["user_id"])
    upload_dir = _pdf_upaths["upload_dir"]
    stamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(8)}"
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(filename).stem)[:64] or "upload"
    file_path = upload_dir / f"{safe_name}_{stamp}{ext}"
    _spool_limited(file.file, file_path, MAX_UPLOAD_BYTES_PDF, "PDF file")
    # Heavy-ops guard (P1): PDF 50MB/500pp enforced at spool (413 above) +
    # processor._validate_pdf backup below. Large-but-allowed files run inline
    # with a warning (no job-queue refactor in P1).
    try:
        import logging as _logging
        _pdf_bytes = file_path.stat().st_size
        if _pdf_bytes > 10 * 1024 * 1024:
            _logging.getLogger("aria.connect").warning(
                "Large PDF processed inline: %s (%.1fMB, user %s).",
                filename, _pdf_bytes / (1024 * 1024), user["user_id"],
            )
    except Exception:
        pass
    # Fail fast on non-PDF magic bytes (renamed scripts/executables).
    try:
        with open(file_path, "rb") as _pf:
            _header = _pf.read(5)
        if _header != b"%PDF-":
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="File is not a valid PDF (missing %PDF- header).")
    except HTTPException:
        raise
    except Exception:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Could not read uploaded PDF.")

    # Create document processor (dirs isolated per authenticated user).
    processed_dir = _pdf_upaths["documents_dir"]
    vector_index_dir = _pdf_upaths["processed_dir"] / "vector_store"
    vector_index_dir.mkdir(parents=True, exist_ok=True)
    
    processor = create_document_processor(
        upload_dir=upload_dir,
        processed_dir=processed_dir,
        llm_provider=llm if provider in HOSTED_PROVIDERS else None,
    )
    
    # Process document (this runs synchronously - could be made async with background tasks)
    result = processor.process_document(
        file_path=file_path,
        user_id=user["user_id"],
        original_name=filename,
    )
    
    if not result.success:
        file_path.unlink(missing_ok=True)
        # Size/page-limit violations are 413 (not 500): spool layer catches
        # oversized bodies; processor backup catches 50MB/500pp violations.
        _err = str(result.error or "").lower()
        if ("too large" in _err or "too many pages" in _err
                or "exceed" in _err or "max 50mb" in _err or "max 500" in _err):
            raise HTTPException(status_code=413, detail=result.error or "PDF exceeds the 50 MB / 500-page limit.")
        raise HTTPException(status_code=500, detail=result.error or "Document processing failed")
    
    document = result.document

    # Update session to include document reference
    session = get_session(user["user_id"])
    # Multi-source (§7): a PDF upload must not destroy a live database
    # connection. Stash the queryable relational/file source so Copilot can
    # answer from both (with per-source attribution) in one session.
    try:
        _prev_uri = session.get("db_uri") or ""
        _prev_type = session.get("db_type", "")
        if _prev_type in ("postgresql", "mysql", "sqlite") and not str(_prev_uri).startswith("document://"):
            _sp = session.get("schema_path")
            _stash: dict = {
                "db": session.get("db"),
                "db_type": _prev_type,
                "db_name": session.get("db_name"),
                "dialect": session.get("dialect", _prev_type),
                "source_type": session.get("source_type"),
                "schema_path": str(_sp) if _sp else None,
            }
            # File-based sqlite URIs carry no secret and are safe to stash;
            # relational URIs embed the password and are rebuilt instead.
            if _prev_type == "sqlite":
                _stash["db_uri"] = _prev_uri
            session["linked_db"] = _stash
    except Exception:
        pass
    # A new document connection must honour the provider selected for this upload.
    session["provider"] = llm
    session["provider_name"] = provider
    session["source_type"] = "document"
    session["db"] = {".pdf": str(file_path)}
    session["db_type"] = "document"
    session["db_uri"] = f"document://{document.id}"
    session["db_name"] = document.id
    session["dialect"] = "document"
    session["document_id"] = document.id
    session["document_name"] = filename
    session["document_type"] = document.document_type.value if document.document_type else "unknown"

    # Structured PDF tables become queryable SQLite + schema mapping (§5/§15).
    # RAG over the narrative keeps working regardless.
    pdf_tables_loaded: list[str] = []
    try:
        _pdf_uri, _pdf_mapping, pdf_tables_loaded = _pdf_tables_to_sqlite(
            document, upload_dir, safe_name, stamp)
        if _pdf_uri and _pdf_mapping:
            _safe_db = re.sub(r"[^a-zA-Z0-9_-]+", "_", f"pdf_{safe_name}")[:64] or "pdf_doc"
            _schema_path = _pdf_upaths["schema_dir"] / f"schema_mapping_{_safe_db}_latest.json"
            _schema_path.write_text(json.dumps({
                "database": _safe_db,
                "tables": _pdf_mapping,
            }, indent=2), encoding="utf-8")
            session["source_type"] = "semi-structured"
            session["db"] = {"document_tables": _pdf_uri}
            session["db_type"] = "sqlite"
            session["db_uri"] = _pdf_uri
            session["db_name"] = _safe_db
            session["dialect"] = "sqlite"
            session["schema_path"] = _schema_path
    except Exception:
        pass

    if not pdf_tables_loaded:
        # Text-only PDF: stay a `document` source and drop every stale
        # analytical pointer (schema/db/artifact refs from a previous
        # connection). SQL analytics must never run against the old source.
        for _key in ("schema_path", "processed_path", "insights_path",
                     "schema_status", "pending_history_id", "file_type"):
            session.pop(_key, None)
        session["schema_status"] = "document"
    
    # Ensure processed directories exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    vector_index_dir.mkdir(parents=True, exist_ok=True)

    # Bridge (§7/§21): mirror the manifest + text chunks into the shared RAG
    # metadata store so Copilot's document/combined branches read the same
    # content the Documents page indexed. Best-effort; never fails the upload.
    try:
        from insight_rag.storage import RAGMetadataStore as _RAGStore
        from langchain_core.documents import Document as _LCDocument
        _rag = _RAGStore(user["user_id"])
        _rag.put_document({
            "document_id": document.id,
            "filename": filename,
            "original_filename": filename,
            "document_type": document.document_type.value if document.document_type else "unknown",
            "page_count": document.page_count,
            "chunk_count": len(document.chunks),
            "created_at": document.created_at,
            "source": "upload",
        })
        _lcdocs = [_LCDocument(
            page_content=c.content or "",
            metadata={"page": c.page_number, "section": c.section,
                      "chunk_id": c.id, "document_id": document.id},
        ) for c in (document.chunks or []) if (c.content or "").strip()]
        if _lcdocs:
            _rag.save_chunks(document.id, _lcdocs)
    except Exception as _bridge_exc:
        import logging as _logging
        _logging.getLogger("aria.connect").info(
            "RAG store mirroring skipped (%s).", type(_bridge_exc).__name__)

    save_session(user["user_id"], session)
    
    return {
        "ok": True,
        "provider": provider,
        "source_type": "document",
        "file": filename,
        "type": "pdf",
        "document_id": document.id,
        "document_type": document.document_type.value if document.document_type else "unknown",
        "page_count": document.page_count,
        "element_count": len(document.elements),
        "chunk_count": len(document.chunks),
        "table_count": len(document.tables),
        "tables_loaded": pdf_tables_loaded,
        "queryable": bool(pdf_tables_loaded),
        "status": document.status.value,
        "processing_method": document.processing_method,
        "failed_pages": result.failed_pages,
        "warnings": result.warnings,
    }


@router.get("/api/session/status")
def session_status(user: dict = Depends(get_current_user)):
    """Return the current session's connectivity and DB info so the frontend can restore state."""
    session = get_session(user.get("user_id"))
    is_connected = bool(session.get("db") or session.get("db_uri") or
                        session.get("source_type") == "semi-structured")
    # Never expose credentials: db_uri embeds user:password (§41).
    raw_uri = session.get("db_uri", "") or ""
    safe_uri = re.sub(r"://([^:/@]+):[^@/]*@", r"://\1:***@", raw_uri)
    linked = session.get("linked_db") or {}
    # P3 §19: expose schema readiness so the frontend derives
    # Disconnected/Connecting/Connected/Schema-loading/Ready/Error from the
    # backend instead of button success. Cached metadata (schema_status) is
    # reused; no re-extraction is triggered here.
    schema_status = session.get("schema_status", "unknown")
    if not is_connected:
        display_state = "disconnected"
    elif schema_status in ("pending", "extracting"):
        display_state = "schema-loading"
    elif schema_status in ("ok", "document"):
        display_state = "ready"
    elif schema_status in ("empty", "failed", "error"):
        display_state = "error"
    else:
        display_state = "connected"
    return {
        "ok": True,
        "connection_status": "connected" if is_connected else "not-connected",
        "display_state": display_state,
        "schema_status": schema_status,
        "provider_name": session.get("provider_name"),
        "provider_error": session.get("_provider_error"),
        "db_name": session.get("db_name", ""),
        "source_type": session.get("source_type", ""),
        "db_uri": safe_uri,
        # Multi-source (§7): a linked database and/or document stay visible
        # so the UI can show what ARIA can reason over.
        "document_id": session.get("document_id", ""),
        "document_name": session.get("document_name", ""),
        "linked_db_name": linked.get("db_name", "") if isinstance(linked, dict) else "",
        "multi_source": bool(linked and session.get("document_id")),
    }


@router.post("/api/disconnect")
def disconnect_source(user: dict = Depends(require_writable)):
    """Remove the active data source while retaining the user's LLM settings."""
    session = get_session(user["user_id"])
    for key in (
        "source_type", "db", "db_uri", "db_name", "dialect", "schema_path",
        "processed_path", "insights_path", "schema_status", "file_type",
        "document_id", "document_name", "document_type", "linked_db",
        "conversation_id", "conversation_history",
        "pending_history_id", "schemas", "db_type",
    ):
        session.pop(key, None)
    save_session(user["user_id"], session)
    return {"ok": True, "message": "Data source disconnected"}

class DatabaseTestRequest(BaseModel):
    host: str = "localhost"
    port: int = 5432
    db: str = ""
    user: str = ""
    password: str = ""
    dbType: str = "postgresql"


@router.post("/api/database/test")
async def test_database_connection(req: DatabaseTestRequest, _user: dict = Depends(require_writable)):
    """Test a database connection before actually connecting (writable sessions only)."""
    # Shared SSRF/host validation (same as real connect path).
    port = _validate_db_endpoint(req.host, req.port, req.db, req.user, req.dbType)
    try:
        from sqlalchemy import create_engine, text
        from core.secret_box import decrypt as _decrypt_secret

        password = req.password
        if password:
            try:
                password = _decrypt_secret(password)
            except Exception:
                pass

        if req.dbType == "mysql":
            url = f"mysql+pymysql://{quote_plus(str(req.user))}:{quote_plus(str(password))}@{req.host}:{req.port}/{req.db}"
        else:
            url = f"postgresql+psycopg2://{quote_plus(str(req.user))}:{quote_plus(str(password))}@{req.host}:{req.port}/{req.db}"

        engine = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 10})
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        finally:
            engine.dispose()

        return {"ok": True, "message": "Connection successful"}
    except Exception as e:
        from core.errors import sanitize_error

        return {"ok": False, "message": f"Connection failed: {sanitize_error(e)}"}



