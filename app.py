"""ARIA FastAPI application entry point.

Wires together the route modules under ``api/`` and exposes the public
endpoints (``/``, ``/api/providers``). Run with:

    python app.py
    uvicorn app:app

Environment variables honoured here:

* ARIA_ENABLE_DOCS (default "0", production-safe): set to "1" to expose
  /docs, /redoc and /openapi.json. Docs stay OFF unless explicitly enabled.
* ARIA_TRUSTED_HOSTS (comma-separated, optional): extra hostnames accepted
  by TrustedHostMiddleware. Local loopback (127.0.0.1, localhost) plus the
  test client ("testserver") are always allowed so dev never breaks.
* CORS_ORIGINS (comma-separated): allowed browser origins for the API.
  "*" is rejected (incompatible with allow_credentials).
* ARIA_PORT: preferred listen port (falls forward to the next free port).
"""

import os
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Make the project root importable regardless of the launching directory.
# core.config loads .env centrally on first import.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config import STATIC_DIR, BASE_DIR  # noqa: E402
from core.checks import run_checks  # noqa: E402
from llm_provider import PROVIDERS  # noqa: E402
from api import auth_routes, connect_routes, query_routes, schema_routes, document_routes, kpi_routes, copilot_routes, hypothesis_routes, hypothesis_investigation_routes, settings_routes, feedback_routes, autocomplete_routes, data_routes, insight_rag_routes, insights_routes, predictive_routes, prescription_routes, rl_routes, intelligence_routes, verification_routes  # noqa: E402
from core.ratelimit import limit, limit_heavy, limit_ip, limit_llm  # noqa: E402
from fastapi import Depends as _Depends  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup checks; failures stay visible but never crash import."""
    import logging
    import auth
    # Secret resolution is lazy at import, but invalid production credentials
    # must still fail startup before accepting requests.
    auth._signing_secret()
    report = run_checks()
    critical = report["critical"]
    if critical:
        logging.getLogger("uvicorn.error").error(
            "ARIA startup check: missing required packages -> %s", ", ".join(critical)
        )
    if not report["ollama"]["ok"]:
        logging.getLogger("uvicorn.error").warning(
            "ARIA startup check: %s", report["ollama"]["detail"]
        )
    # Single-worker safety: multi-worker without a shared store diverges
    # sessions (file) and rate limits (memory). Warn loudly but never crash
    # dev (single worker default). See ARIA_REDIS_URL / ARIA_REQUIRE_SHARED_STORE.
    # Single-worker storage model (P0-3): in-memory/file stores are LOCAL only.
    # Choice: fail fast with RuntimeError when workers > 1 without
    # ARIA_REDIS_URL AND (ARIA_REQUIRE_SHARED_STORE is truthy/not-"0" OR
    # ARIA_ENV=production without an explicit opt-out). Single-worker
    # default (workers <= 1) always starts unchanged. An explicit
    # ARIA_REQUIRE_SHARED_STORE=0/false/no/off opts back into warn-only
    # mode for non-production dev (e.g. multi-worker toy experiments).
    try:
        _workers = 1
        for _var in ("UVICORN_WORKERS", "WEB_CONCURRENCY", "ARIA_WORKERS"):
            try:
                _n = int(os.getenv(_var, "1") or 1)
                if _n > _workers:
                    _workers = _n
            except (TypeError, ValueError):
                continue
        _redis_url = (os.getenv("ARIA_REDIS_URL", "") or "").strip()
        if _workers > 1 and not _redis_url:
            # Unset is NOT an opt-out: only an explicit 0/false/no/off
            # downgrades production to warn-only. Default (unset) in prod
            # fails fast; in non-prod unset stays warn-only (dev compat).
            _require_env = os.getenv("ARIA_REQUIRE_SHARED_STORE")
            _require_raw = (_require_env or "").strip().lower()
            _explicit_opt_out = _require_raw in {"0", "false", "no", "off"}
            _require_shared = _require_raw in {"1", "true", "yes", "on"}
            _is_prod = (os.getenv("ARIA_ENV", "") or "").strip().lower() == "production"
            _fail_fast = _require_shared or (_is_prod and not _explicit_opt_out)
            _msg = (
                f"Multi-worker deployment ({_workers} workers via "
                "UVICORN_WORKERS/WEB_CONCURRENCY/ARIA_WORKERS) without "
                "ARIA_REDIS_URL: sessions and rate limits would diverge "
                "per-process. Set ARIA_REDIS_URL to a shared Redis, or run a "
                "single worker."
            )
            if _fail_fast:
                raise RuntimeError(
                    _msg + " (fail-fast: ARIA_REQUIRE_SHARED_STORE=1 or "
                    "ARIA_ENV=production without explicit opt-out)"
                )
            logging.getLogger("uvicorn.error").error(
                "Multi-worker deployment (%d workers via UVICORN_WORKERS/WEB_CONCURRENCY) "
                "without ARIA_REDIS_URL: sessions and rate limits will diverge "
                "per-process. Set ARIA_REDIS_URL to a shared Redis, or run a "
                "single worker. Set ARIA_REQUIRE_SHARED_STORE=1 to fail closed.",
                _workers,
            )
    except RuntimeError:
        raise
    except Exception:
        pass
    logging.getLogger("uvicorn.error").info("ARIA startup checks complete.")
    try:
        _eff_workers = _resolve_workers()
    except Exception:
        _eff_workers = 1
    logging.getLogger("uvicorn.error").info(
        "ARIA effective workers=%d (UVICORN_WORKERS/WEB_CONCURRENCY/ARIA_WORKERS).",
        _eff_workers,
    )
    yield


_enable_docs = os.getenv("ARIA_ENABLE_DOCS", "0").strip().lower() in {"1", "true", "yes", "on"}
app = FastAPI(
    title="ARIA Testing UI",
    lifespan=lifespan,
    docs_url="/docs" if _enable_docs else None,
    redoc_url="/redoc" if _enable_docs else None,
    openapi_url="/openapi.json" if _enable_docs else None,
)

# Local-dev host allowlist: loopback + test client always pass; operators add
# public hostnames via ARIA_TRUSTED_HOSTS. Never breaks `python app.py`.
_trusted_hosts = ["127.0.0.1", "localhost", "testserver"]
for _extra in (os.getenv("ARIA_TRUSTED_HOSTS", "") or "").split(","):
    _extra = _extra.strip().lower()
    if _extra and _extra not in _trusted_hosts:
        _trusted_hosts.append(_extra)
try:
    from starlette.middleware.trustedhost import TrustedHostMiddleware as _TrustedHost
    # allow 127.0.0.1/localhost (+ testserver for TestClient) without breaking dev
    app.add_middleware(_TrustedHost, allowed_hosts=_trusted_hosts)
except Exception:  # pragma: no cover - middleware must never break startup
    pass
try:
    from starlette.middleware.gzip import GZipMiddleware as _GZip
    # Compress JSON/bundles over 1KB (intelligence payloads, dist assets).
    app.add_middleware(_GZip, minimum_size=1000)
except Exception:  # pragma: no cover - middleware must never break startup
    pass


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    # CSP is report-only friendly: allow same-origin + data/blob for charts/exports.
    if request.url.path.startswith("/docs") or request.url.path.startswith("/redoc"):
        pass
    else:
        resp.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; font-src 'self' data:",
        )
    return resp


@app.middleware("http")
async def _request_id(request: Request, call_next):
    """Attach a stable request ID (t.txt-7): full tracebacks are logged
    server-side against it; clients only ever see the ID."""
    import uuid as _uuid

    rid = request.headers.get("X-Request-ID") or _uuid.uuid4().hex[:12]
    request.state.request_id = rid
    resp = await call_next(request)
    resp.headers["X-Request-ID"] = rid
    return resp

# Single-origin deployment: the backend serves the built frontend on :8000,
# so API calls are same-origin (/api). CORS exists only for operator
# overrides via CORS_ORIGINS.
_cors_origins = [
    o.strip() for o in os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000",
    ).split(",") if o.strip()
]
if "*" in _cors_origins:
    # "*" + allow_credentials is rejected by browsers anyway; drop it loudly
    # rather than running an insecure no-op config.
    import logging as _logging

    _logging.getLogger("uvicorn.error").warning(
        "CORS_ORIGINS contains '*', which is incompatible with allow_credentials; ignoring '*'."
    )
    _cors_origins = [o for o in _cors_origins if o != "*"] or ["http://localhost:8000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-Client-Type"],
)

app.include_router(auth_routes.router)  # has its own login/register limiter
app.include_router(connect_routes.router, dependencies=[_Depends(limit_heavy())])
app.include_router(schema_routes.router, dependencies=[_Depends(limit_heavy())])
app.include_router(query_routes.router, dependencies=[_Depends(limit_llm())])
app.include_router(document_routes.router, dependencies=[_Depends(limit_heavy())])
app.include_router(kpi_routes.router, dependencies=[_Depends(limit_heavy())])
app.include_router(copilot_routes.router, dependencies=[_Depends(limit_llm())])
app.include_router(hypothesis_routes.router, dependencies=[_Depends(limit_heavy())])
app.include_router(settings_routes.router, dependencies=[_Depends(limit_heavy())])
app.include_router(feedback_routes.router, dependencies=[_Depends(limit("feedback", 60))])
app.include_router(autocomplete_routes.router, dependencies=[_Depends(limit("autocomplete", 180))])
app.include_router(data_routes.router, dependencies=[_Depends(limit_heavy())])
app.include_router(insight_rag_routes.router, dependencies=[_Depends(limit_llm())])
app.include_router(insights_routes.router, dependencies=[_Depends(limit_heavy())])
app.include_router(predictive_routes.router, dependencies=[_Depends(limit_llm())])
app.include_router(prescription_routes.router, dependencies=[_Depends(limit_heavy())])
app.include_router(hypothesis_investigation_routes.router, dependencies=[_Depends(limit_heavy())])
app.include_router(rl_routes.router, dependencies=[_Depends(limit("rl", 60))])
app.include_router(intelligence_routes.router, dependencies=[_Depends(limit_heavy())])
app.include_router(verification_routes.router, dependencies=[_Depends(limit_heavy())])


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Converge FastAPI HTTP errors toward the canonical shape.

    Canonical error body: {ok:false, success:false, code, detail, request_id}.
    Never looks like success (ok is always False); request_id always present
    (mirrors X-Request-ID); detail is sanitized so no stack/SQL/secret leaks.
    """
    from core.errors import sanitize_error as _san
    rid = getattr(request.state, "request_id", None) or request.headers.get("X-Request-ID") or "unknown"
    code = "UNAUTHORIZED" if exc.status_code == 401 else (
        "FORBIDDEN" if exc.status_code == 403 else (
            "NOT_FOUND" if exc.status_code == 404 else "REQUEST_FAILED"))
    detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "ok": False,
            "success": False,
            "code": code,
            "detail": _san(detail),
            "request_id": rid,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """422 bodies carry the same canonical shape (no raw trace, has request_id)."""
    rid = getattr(request.state, "request_id", None) or request.headers.get("X-Request-ID") or "unknown"
    return JSONResponse(
        status_code=422,
        content={
            "ok": False,
            "success": False,
            "code": "INVALID_REQUEST",
            "detail": "Invalid request. Check the highlighted fields and retry.",
            "request_id": rid,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    """Return a clean JSON 500 with guidance instead of a raw stack trace page."""
    import logging
    import traceback

    rid = getattr(request.state, "request_id", "unknown")
    logger = logging.getLogger("uvicorn.error")
    logger.exception("Unhandled error on %s [rid=%s]", request.url.path, rid)
    tb = traceback.format_exc()
    logger.error("Full traceback [rid=%s]:\n%s", rid, tb)
    # Generic message to the client (detail is logged server-side only).
    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "success": False,
            "code": "INTERNAL_ERROR",
            "detail": "Something went wrong. Reference ID: " + rid,
            "request_id": rid,
        },
    )


@app.get("/api/health", dependencies=[_Depends(limit_ip("health", 120))])
def health():
    """Live dependency/health status for ops and the UI."""
    return run_checks()


@app.get("/api/providers", dependencies=[_Depends(limit_ip("providers", 120))])
def providers():
    """Return provider metadata (labels + privacy context) for the UI."""
    return {
        "providers": {
            name: {"label": meta["label"], "privacy": meta["privacy"], "models": meta["models"]}
            for name, meta in PROVIDERS.items()
        }
    }


# Serve the built React frontend (aria-frontend/dist/) at root
DIST_DIR = BASE_DIR / "aria-frontend" / "dist"
# Cache policy: index.html must NEVER cache (it references hashed bundles —
# a stale shell is why users see old pages after a deploy). Hashed /assets
# are immutable and cache for a year.
_INDEX_NO_STORE = {"Cache-Control": "no-store, must-revalidate"}


class _ImmutableStatic(StaticFiles):
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        try:
            if resp.status_code == 200:
                resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        except Exception:
            pass
        return resp


# Mount dist contents so /assets/ resolves correctly (vite builds to /assets/)
# Guarded: missing build dir must not crash the backend (API still works).
if (DIST_DIR / "assets").exists():
    app.mount("/assets", _ImmutableStatic(directory=str(DIST_DIR / "assets")), name="assets")
# Also serve root files (favicon, index.html) via direct response
def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or request.headers.get("X-Request-ID") or "unknown"


def _serve_existing_file(rel_path: str):
    """Return a FileResponse for rel_path if it exists under dist/ (first)
    or static/, else None. Blocks path traversal; never returns a
    directory. index.html gets no-store; hashed /assets get immutable."""
    for base in (DIST_DIR, STATIC_DIR):
        try:
            root = base.resolve()
            candidate = (root / rel_path).resolve()
            if not candidate.is_relative_to(root) or not candidate.is_file():
                continue
        except Exception:
            continue
        headers = None
        if candidate.name == "index.html":
            headers = dict(_INDEX_NO_STORE)
        elif rel_path.startswith("assets/"):
            headers = {"Cache-Control": "public, max-age=31536000, immutable"}
        return FileResponse(candidate, headers=headers)
    return None


def _frontend_index_response(request: Request):
    """Serve the built SPA shell. dist/index.html only — if the build is
    missing, return a clear 503 instead of broken/missing HTML."""
    prod_index = DIST_DIR / "index.html"
    if prod_index.exists():
        return FileResponse(prod_index, headers=dict(_INDEX_NO_STORE))
    return JSONResponse(status_code=503, content={
        "ok": False, "success": False, "code": "SERVICE_UNAVAILABLE",
        "detail": "frontend build missing", "request_id": _request_id(request)})


@app.get("/")
def index(request: Request):
    return _frontend_index_response(request)

# Fallback: any unmatched non-API route -> SPA routing.
# * existing file (dist first, then static) -> serve it (never index.html
#   for an asset request)
# * missing path WITH an extension -> 404 (never HTML in an asset slot)
# * extensionless path -> SPA index (hash router uses /#/..., kept for
#   robustness); if dist/ is absent -> 503 "frontend build missing"
# /api/* must never silently return HTML; unknown API paths return the
# canonical error shape (ok:false + request_id, never success-like).
@app.get("/{full_path:path}")
def catch_all(full_path: str, request: Request):
    rid = _request_id(request)
    if full_path == "api" or full_path.startswith("api/"):
        return JSONResponse(status_code=404, content={
            "ok": False, "success": False, "code": "NOT_FOUND",
            "detail": "Unknown API endpoint.", "request_id": rid})
    served = _serve_existing_file(full_path)
    if served is not None:
        return served
    if Path(full_path).suffix:
        return JSONResponse(status_code=404, content={
            "ok": False, "success": False, "code": "NOT_FOUND",
            "detail": "Static file not found.", "request_id": rid})
    return _frontend_index_response(request)


def _resolve_workers() -> int:
    """Return effective uvicorn worker count (default 1, clamp 1-16).

    Reads UVICORN_WORKERS / WEB_CONCURRENCY / ARIA_WORKERS (max wins);
    unparseable values are ignored. Single-worker default unchanged.
    """
    _workers = 1
    for _var in ("UVICORN_WORKERS", "WEB_CONCURRENCY", "ARIA_WORKERS"):
        try:
            _n = int(os.getenv(_var, "1") or 1)
            if _n > _workers:
                _workers = _n
        except (TypeError, ValueError):
            continue
    try:
        _workers = int(_workers)
    except (TypeError, ValueError):
        return 1
    return max(1, min(16, _workers))


def _resolve_port(start_port: int = 8000, max_tries: int = 20) -> int:
    """Return the first free local port starting from the requested value."""
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free local port found between {start_port} and {start_port + max_tries - 1}.")


if __name__ == "__main__":
    import uvicorn

    try:
        requested_port = int(os.getenv("ARIA_PORT", "8000"))
    except (ValueError, TypeError):
        requested_port = 8000
    port = _resolve_port(requested_port)
    _workers = _resolve_workers()
    print(f"ARIA Testing UI running at http://127.0.0.1:{port} (workers={_workers})")
    uvicorn.run(app, host="127.0.0.1", port=port, workers=_workers)
