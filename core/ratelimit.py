"""Shared sliding-window rate limiter (no extra dependencies).

Backend is explicit via ARIA_RATE_LIMIT_STORE (default "memory"):

* "memory" — single-process in-memory store. LOCAL/DEV ONLY. Each worker
  keeps its own counters, so limits do not compose across processes.
* Production multi-worker deployments need a shared store (Redis/DB).
  Set ARIA_REQUIRE_SHARED_STORE=1 to fail closed instead of running with
  per-process limits: any request then raises 503 when workers > 1 without
  a shared store configured (ARIA_REDIS_URL).

Budgets are generous: normal users and E2E never trip them; only bursts
(credential-stuffing, LLM-cost DoS) get 429.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict

from fastapi import Depends, HTTPException, Request

from core.deps import get_current_user

_hits: dict[tuple[str, str], list[float]] = defaultdict(list)
_warned_multi_worker = False
_last_cleanup: float = 0.0
_redis_client = None
_redis_init_warned = False
_redis_missing_warned = False


def redis_url() -> str:
    """Return the configured Redis URL (empty string when not configured)."""
    try:
        return (os.getenv("ARIA_REDIS_URL", "") or "").strip()
    except Exception:
        return ""


def get_redis_client():
    """Return a shared Redis client, or None with graceful in-memory fallback.

    None when ARIA_REDIS_URL is unset, when the `redis` package is missing
    (warn once), or when Redis is unreachable (warn once). Callers must fall
    back to the in-memory limiter on None.
    """
    global _redis_client, _redis_init_warned, _redis_missing_warned
    if _redis_client is not None:
        return _redis_client
    url = redis_url()
    if not url:
        return None
    try:
        import redis as _redis_mod  # type: ignore
    except ImportError:
        if not _redis_missing_warned:
            _redis_missing_warned = True
            logging.getLogger("uvicorn.error").warning(
                "ARIA_REDIS_URL is set but the 'redis' package is missing; "
                "falling back to in-memory rate limits (pip install redis)."
            )
        return None
    try:
        client = _redis_mod.StrictRedis.from_url(
            url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2
        )
        client.ping()
    except Exception as exc:
        if not _redis_init_warned:
            _redis_init_warned = True
            logging.getLogger("uvicorn.error").warning(
                "Redis rate-limit store unreachable (%s); "
                "falling back to in-memory limits.",
                type(exc).__name__,
            )
        return None
    _redis_client = client
    return _redis_client


def _reset_redis_client_for_tests() -> None:
    """Clear the cached Redis client (tests only)."""
    global _redis_client, _redis_init_warned, _redis_missing_warned
    _redis_client = None
    _redis_init_warned = False
    _redis_missing_warned = False


def store_backend() -> str:
    """Return the configured rate-limit store backend name.

    Returns "redis" when ARIA_REDIS_URL is set (shared store preferred) and
    ARIA_RATE_LIMIT_STORE is still the default "memory"; an explicit
    ARIA_RATE_LIMIT_STORE value is honoured verbatim for backward compat.
    """
    raw = (os.getenv("ARIA_RATE_LIMIT_STORE", "") or "").strip().lower()
    if not raw or raw == "memory":
        if redis_url():
            return "redis"
        return "memory"
    return raw


def worker_count() -> int:
    """Return the configured worker/entrypoint count (best-effort)."""
    for var in ("ARIA_WORKERS", "WEB_CONCURRENCY", "UVICORN_WORKERS"):
        try:
            n = int(os.getenv(var, "1") or 1)
            if n > 1:
                return n
        except (TypeError, ValueError):
            continue
    return 1


def _check_shared_store() -> None:
    """Fail closed for multi-worker runs without a shared store.

    Single-process dev always passes. With workers > 1 and the "memory"
    backend, log a loud warning; raise (503) instead of enforcing
    silently-divided per-process limits when:
      * ARIA_REQUIRE_SHARED_STORE is truthy (1/true/yes/on), OR
      * ARIA_ENV=production without an explicit opt-out
        (ARIA_REQUIRE_SHARED_STORE=0/false/no/off).
    Choice (P0-3): explicit opt-in fails closed everywhere; production
    fails closed by default so a missing flag can never silently split
    limits. Single-worker default is unchanged (never raises).
    """
    global _warned_multi_worker
    if store_backend() != "memory":
        return
    if worker_count() <= 1:
        return
    if os.getenv("ARIA_REDIS_URL", "").strip():
        return
    if not _warned_multi_worker:
        _warned_multi_worker = True
        logging.getLogger("uvicorn.error").warning(
            "Rate limiter uses per-process memory with %d workers and no shared "
            "store (set ARIA_REDIS_URL / ARIA_RATE_LIMIT_STORE). Limits are "
            "per-process; use a shared store in production.",
            worker_count(),
        )
    _require_env = os.getenv("ARIA_REQUIRE_SHARED_STORE")
    _require_raw = (_require_env or "").strip().lower()
    _explicit_opt_out = _require_raw in {"0", "false", "no", "off"}
    _require_shared = _require_raw in {"1", "true", "yes", "on"}
    _is_prod = (os.getenv("ARIA_ENV", "") or "").strip().lower() == "production"
    if _require_shared or (_is_prod and not _explicit_opt_out):
        raise HTTPException(
            status_code=503,
            detail="Multi-worker deployment requires a shared rate-limit store. "
                    "Set ARIA_REDIS_URL or run a single worker.",
        )


def _enabled() -> bool:
    # Kill-switch safety: ARIA_RATE_LIMIT=off is honoured in dev only.
    # In production the limiter stays enabled (fail-closed) so an env typo
    # cannot silently disable brute-force / LLM-cost protection.
    try:
        raw = (os.getenv("ARIA_RATE_LIMIT", "on") or "on").lower()
    except Exception:
        return True
    if raw not in ("off", "0", "false"):
        return True
    try:
        is_prod = (os.getenv("ARIA_ENV", "") or "").strip().lower() == "production"
    except Exception:
        is_prod = True
    if is_prod:
        try:
            logging.getLogger("uvicorn.error").warning(
                "ARIA_RATE_LIMIT=off ignored in production (fail-closed).")
        except Exception:
            pass
        return True
    return False


def _redis_check_limit(bucket: str, key: str, max_calls: int, window_s: int) -> bool:
    """Try a Redis sliding-window check; return True if handled.

    Returns False when Redis is not configured or any Redis error occurs so
    the caller falls back to the in-memory limiter (backward compat).
    Raises 429 when the shared budget is exceeded.
    """
    try:
        client = get_redis_client()
        if client is None:
            return False
        import time as _time
        import uuid as _uuid

        now = _time.time()
        rkey = f"aria:rl:{bucket}:{key}"
        member = f"{now:.6f}:{_uuid.uuid4().hex}"
        try:
            pipe = client.pipeline()
            pipe.zremrangebyscore(rkey, 0, now - window_s)
            pipe.zadd(rkey, {member: now})
            pipe.zcard(rkey)
            pipe.expire(rkey, int(window_s))
            res = pipe.execute()
            count = int(res[2]) if len(res) >= 3 else 0
        except Exception:
            return False
        if count > max_calls:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Slow down and try again shortly.",
            )
        return True
    except HTTPException:
        raise
    except Exception:
        return False


def check_limit(bucket: str, key: str, max_calls: int, window_s: int) -> None:
    if not _enabled():
        return
    _check_shared_store()
    # Prefer the shared Redis sliding window when configured; fall back to
    # the in-memory limiter when Redis is unset/unreachable (dev compat).
    try:
        if _redis_check_limit(bucket, key, max_calls, window_s):
            return
    except HTTPException:
        raise
    except Exception:
        pass
    now = time.monotonic()
    k = (bucket, key)
    hits = [t for t in _hits[k] if now - t < window_s]
    if len(hits) >= max_calls:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Slow down and try again shortly.",
        )
    hits.append(now)
    _hits[k] = hits
    # Bound memory: drop buckets idle past 2x the longest window and cap total keys.
    # Only run cleanup periodically (every ~60 s) to avoid O(n) scan on every request.
    global _last_cleanup
    if len(_hits) > 5000 and (now - _last_cleanup) > 60:
        _last_cleanup = now
        stale = [kk for kk, vv in _hits.items() if not vv or now - vv[-1] > window_s * 2]
        for kk in stale[:1000]:
            _hits.pop(kk, None)


def limit(bucket: str, max_calls: int, window_s: int = 60):
    """FastAPI dependency: throttle an endpoint per authenticated user."""

    def _dep(request: Request, user: dict = Depends(get_current_user)) -> dict:
        ident = str(user.get("user_id", "?"))
        ip = request.client.host if request.client else "unknown"
        check_limit(f"{bucket}:u", ident, max_calls, window_s)
        check_limit(f"{bucket}:ip", ip, max_calls * 4, window_s)
        return user

    return _dep


def limit_ip(bucket: str, max_calls: int, window_s: int = 60):
    """FastAPI dependency: throttle an unauthenticated endpoint per client IP."""

    def _dep(request: Request) -> None:
        ip = request.client.host if request.client else "unknown"
        check_limit(f"{bucket}:ip", ip, max_calls, window_s)

    return _dep


# Presets: expensive LLM/data endpoints get tighter budgets.
limit_llm = lambda: limit("llm", int(os.getenv("ARIA_RL_LLM_MAX", "30")), 60)  # noqa: E731
limit_heavy = lambda: limit("heavy", int(os.getenv("ARIA_RL_HEAVY_MAX", "60")), 60)  # noqa: E731
