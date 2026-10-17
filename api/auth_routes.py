"""Authentication and per-user history routes."""

import os
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import auth
from core.deps import get_current_user, is_mobile

router = APIRouter()

COOKIE_NAME = "aria_token"
SECURE_COOKIE_NAME = "__Host-aria_token"


def _cookie_secure() -> bool:
    """Secure flag for the auth cookie. Default True (prod-safe).

    Explicit opt-out for local HTTP dev only via ARIA_COOKIE_SECURE=0
    (0/false/no/off). An explicit ARIA_ENV=dev/development/local also
    opts out. Any other/unset value stays Secure so prod can never
    accidentally disable it.
    """
    explicit = (os.getenv("ARIA_COOKIE_SECURE") or "").strip().lower()
    if explicit in ("0", "false", "no", "off"):
        return False
    if explicit in ("1", "true", "yes", "on"):
        return True
    if (os.getenv("ARIA_ENV") or "").strip().lower() in ("dev", "development", "local"):
        return False
    return True


def _cookie_name() -> str:
    # __Host- prefix requires Secure, Path=/, and no Domain — all true
    # when secure. Use plain name for explicit local-HTTP dev.
    return SECURE_COOKIE_NAME if _cookie_secure() else COOKIE_NAME


def _cookie_kwargs() -> dict:
    return {
        "httponly": True,
        "samesite": "lax",
        "secure": _cookie_secure(),
        "path": "/",
        "max_age": auth.TOKEN_TTL_DAYS * 24 * 3600,
    }


def _auth_response(token: str, username: str) -> JSONResponse:
    # HttpOnly cookie flow: never return the raw JWT in JSON.
    resp = JSONResponse({"ok": True, "username": username})
    resp.set_cookie(_cookie_name(), token, **_cookie_kwargs())
    return resp

# Brute-force guard: sliding-window per client IP (generous so real users
# and E2E never trip it; only credential-stuffing bursts get 429).
# Storage model (P0-3): in-memory per-process, LOCAL/DEV ONLY. Multi-worker
# production must set ARIA_REDIS_URL + ARIA_REQUIRE_SHARED_STORE=1 (see
# core/ratelimit._check_shared_store); otherwise each worker enforces its
# own per-process login budget.
_LOGIN_ATTEMPTS: dict[str, list[float]] = {}
_LOGIN_WINDOW_S = 300
_LOGIN_MAX = 30
# Bound for the per-process login map: prune entries idle past the window
# and cap distinct IPs so a long-running server cannot grow unbounded.
_LOGIN_MAX_IPS = 10_000
_LOGIN_LAST_PRUNE: float = 0.0


def _prune_login_attempts(now: float | None = None) -> int:
    """Expire stale login entries and enforce the IP cap. Returns removed IPs.

    Drops per-IP timestamps older than _LOGIN_WINDOW_S, removes empty IPs,
    then evicts oldest-last-seen IPs while over _LOGIN_MAX_IPS. Never raises.
    """
    global _LOGIN_LAST_PRUNE
    try:
        current = float(now) if now is not None else time.monotonic()
        removed = 0
        for ip in list(_LOGIN_ATTEMPTS.keys()):
            try:
                hits = [t for t in _LOGIN_ATTEMPTS.get(ip, []) if current - t < _LOGIN_WINDOW_S]
            except Exception:
                hits = []
            if hits:
                _LOGIN_ATTEMPTS[ip] = hits
            else:
                _LOGIN_ATTEMPTS.pop(ip, None)
                removed += 1
        if len(_LOGIN_ATTEMPTS) > _LOGIN_MAX_IPS:
            # Evict oldest-last-seen first (deterministic, bounded memory).
            try:
                ordered = sorted(
                    _LOGIN_ATTEMPTS.items(),
                    key=lambda kv: max(kv[1]) if kv[1] else 0.0,
                )
                excess = len(_LOGIN_ATTEMPTS) - _LOGIN_MAX_IPS
                for ip, _ in ordered[:excess]:
                    _LOGIN_ATTEMPTS.pop(ip, None)
                    removed += 1
            except Exception:
                while len(_LOGIN_ATTEMPTS) > _LOGIN_MAX_IPS:
                    _LOGIN_ATTEMPTS.pop(next(iter(_LOGIN_ATTEMPTS)), None)
                    removed += 1
        _LOGIN_LAST_PRUNE = current
        return removed
    except Exception:
        return 0


def _check_login_rate_limit(request: Request) -> None:
    now = time.monotonic()
    ip = request.client.host if request.client else "unknown"
    # Use the same fail-closed enforcement as protected routes. Never swallow
    # a shared 429/503: that would split the budget across workers.
    from core.ratelimit import check_limit
    check_limit("login", ip, _LOGIN_MAX, _LOGIN_WINDOW_S)
    hits = [t for t in _LOGIN_ATTEMPTS.get(ip, []) if now - t < _LOGIN_WINDOW_S]
    if len(hits) >= _LOGIN_MAX:
        raise HTTPException(status_code=429, detail="Too many login attempts. Try again in a few minutes.")
    hits.append(now)
    _LOGIN_ATTEMPTS[ip] = hits
    # Opportunistic global prune: cheap per-request for this IP is already
    # done above; run the full sweep periodically or when over the cap so
    # the map cannot grow unbounded across many distinct IPs.
    global _LOGIN_LAST_PRUNE
    if len(_LOGIN_ATTEMPTS) > _LOGIN_MAX_IPS or (now - _LOGIN_LAST_PRUNE) > 60:
        _prune_login_attempts(now)


class AuthRequest(BaseModel):
    username: str
    password: str


@router.post("/api/auth/register")
def register(auth_req: AuthRequest, request: Request):
    """Create a user account and return a JWT (generic error avoids enumeration)."""
    _check_login_rate_limit(request)
    try:
        user_id = auth.register(auth_req.username, auth_req.password)
    except ValueError:
        # Generic message: don't reveal whether the username is taken.
        raise HTTPException(status_code=400, detail="Could not create account. Try a different username.")
    username = auth_req.username.strip()
    return _auth_response(auth.issue_token(user_id, username), username)


@router.post("/api/auth/login")
def login(auth_req: AuthRequest, request: Request):
    """Exchange username/password for a JWT."""
    _check_login_rate_limit(request)
    user_id = auth.authenticate(auth_req.username, auth_req.password)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    username = auth_req.username.strip()
    return _auth_response(auth.issue_token(user_id, username), username)


@router.get("/api/auth/me")
def me(user: dict = Depends(get_current_user)):
    return {"ok": True, "username": user["username"], "user_id": user["user_id"]}


@router.post("/api/auth/logout")
def logout(request: Request, user: dict = Depends(get_current_user)):
    """Revoke the presented JWT and acknowledge logout."""
    seen: set[str] = set()
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        seen.add(authorization[7:].strip())
    # Revoke whichever cookie(s) the client presented (backward compat:
    # accept both __Host- prefixed and legacy plain names).
    for name in (SECURE_COOKIE_NAME, COOKIE_NAME):
        tok = request.cookies.get(name)
        if tok:
            seen.add(tok.strip())
    for tok in seen:
        if tok and tok.lower() not in ("null", "undefined", '""', "''"):
            auth.revoke_token(tok)
    resp = JSONResponse({"ok": True, "username": user["username"]})
    for name in (SECURE_COOKIE_NAME, COOKIE_NAME):
        resp.delete_cookie(name, path="/")
    return resp


@router.get("/api/history")
def history(request: Request, user: dict = Depends(get_current_user), limit: int = 200):
    """List the current user's saved analyses (metadata only, newest first)."""
    items = auth.list_history(user["user_id"], limit=max(1, min(limit, 500)))
    return {
        "ok": True,
        "mode": "read-only" if is_mobile(request) else "full",
        "items": items,
    }


@router.get("/api/history/{history_id}")
def history_detail(history_id: int, user: dict = Depends(get_current_user)):
    """Return one saved analysis including processed data and insights."""
    record = auth.get_history(user["user_id"], history_id)
    if not record:
        raise HTTPException(status_code=404, detail="History record not found.")
    return {"ok": True, "record": record}


@router.delete("/api/history/{history_id}")
def history_delete(history_id: int, user: dict = Depends(get_current_user)):
    """Delete a saved analysis."""
    deleted = auth.delete_history(user["user_id"], history_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="History record not found.")
    return {"ok": True}
