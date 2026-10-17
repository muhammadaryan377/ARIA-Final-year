"""Shared FastAPI dependencies: session guards, authentication, mobile read-only."""

import re

from fastapi import Depends, Header, HTTPException, Request

import auth
from core.config import get_session


def get_current_user(request: Request, authorization: str | None = Header(default=None)) -> dict:
    """Resolve the Bearer token into a {user_id, username} dict or 401.

    Accepts the Authorization header (preferred) or the httpOnly auth
    cookie set at login/register (``__Host-aria_token`` when Secure,
    ``aria_token`` legacy/dev fallback). Header-only legacy ?token=
    query fallback was removed because tokens in URLs leak into logs.
    Any missing/malformed/invalid/expired token raises 401 — never falls
    back to a default/anonymous session.
    """
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if not token or token.lower() in ("null", "undefined", '""', "''", "none"):
        token = None
    if not token:
        # Prefer __Host- prefixed cookie; fall back to legacy plain name.
        for _name in ("__Host-aria_token", "aria_token"):
            _tok = request.cookies.get(_name)
            if _tok and _tok.strip() and _tok.strip().lower() not in ("null", "undefined", '""', "''", "none"):
                token = _tok.strip()
                break
    user = auth.verify_token(token) if token else None
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated. Provide a valid Bearer token.")
    return user


def _authenticated_user(request: Request, user) -> dict:
    """Resolve authenticated user for guards supporting both Depends and manual calls.

    When used as a FastAPI dependency, `user` is injected via
    Depends(get_current_user) (401 on bad token). When called manually as
    require_data(request), `user` is a Depends marker (not a dict), so
    resolve explicitly from the request's Authorization header/cookies.
    """
    if isinstance(user, dict) and "user_id" in user:
        return user
    authz = None
    try:
        authz = request.headers.get("authorization")
    except Exception:
        authz = None
    return get_current_user(request, authorization=authz)

_MOBILE_RE = re.compile(
    r"Mobile|Android|iPhone|iPad|iPod|Windows Phone|BlackBerry|IEMobile|Opera Mini|Silk",
    re.IGNORECASE,
)


def is_mobile(request: Request) -> bool:
    """Detect a mobile client via an explicit header (preferred) or the UA string."""
    client = request.headers.get("x-client-type", "")
    if client:
        return client.lower() == "mobile"
    return bool(_MOBILE_RE.search(request.headers.get("user-agent", "")))


def require_provider(request: Request, user: dict = Depends(get_current_user)):
    """Fail unless an LLM provider is active in the authenticated user's session."""
    user = _authenticated_user(request, user)
    session = get_session(user["user_id"])
    if not session.get("provider"):
        raise HTTPException(status_code=400, detail="Choose an LLM provider and connect a source first (POST /api/connect).")
    return session.get("provider")


def require_data(request: Request, user: dict = Depends(get_current_user)):
    """Require an authenticated session plus a usable source (DB or CSV/PDF)."""
    user = _authenticated_user(request, user)
    session = get_session(user["user_id"])
    provider = session.get("provider")
    if not provider:
        raise HTTPException(status_code=400, detail="Choose an LLM provider and connect a source first (POST /api/connect).")
    if session.get("source_type") == "relational" and not session.get("db"):
        raise HTTPException(status_code=400, detail="Connect to a relational database first (POST /api/connect).")
    if session.get("source_type") == "semi-structured" and not session.get("db_uri"):
        raise HTTPException(status_code=400, detail="Upload a CSV or PDF file first (POST /api/connect_file).")
    return provider


def reject_document_sql(session) -> None:
    """Block SQL analytics on text-only document sources (t.txt-4).

    Text PDFs stay `document` sources with a document:// URI and no tabular
    session. SQL engines must refuse them with a pointer to Document Chat
    instead of failing obscurely downstream.
    """
    if session.get("source_type") == "document" and str(
            session.get("db_uri") or "").startswith("document://"):
        raise HTTPException(
            status_code=400,
            detail="This source is a text document — use Document Chat "
                   "(Copilot) instead of SQL analytics.",
        )


def require_writable(request: Request, user: dict = Depends(get_current_user)) -> dict:
    """Allow an authenticated user; block write actions on mobile (view history only)."""
    if is_mobile(request):
        raise HTTPException(
            status_code=403,
            detail="Read-only on mobile: you can view your saved history, but connecting sources and running analyses are disabled on this device.",
        )
    return user


def ensure_decision_owner(session_user_id, decision: dict | None) -> dict:
    """Ownership gate for Decision Center records (hyp.txt #84-85).

    Sessions are already per-user (core.config.get_session), so a lookup
    inside the caller's own session IS the ownership check: ids alone are
    never trusted across users. Raises 404 (not 403) for missing/foreign
    ids so one user cannot probe another user's records. Raises 410 for
    quarantined records (invalidated by migration, history preserved).
    """
    if not isinstance(decision, dict):
        raise HTTPException(status_code=404, detail="That decision was not found.")
    owner = decision.get("owner_user_id")
    if owner is not None and session_user_id is not None:
        try:
            if int(owner) != int(session_user_id):
                raise HTTPException(status_code=404, detail="That decision was not found.")
        except (TypeError, ValueError):
            pass
    if decision.get("quarantined"):
        raise HTTPException(
            status_code=410,
            detail="That decision was invalidated during migration and is no longer active.")
    return decision


def get_request_id(request) -> str:
    """Best-effort request id for structured logs (never raises, no imports)."""
    try:
        rid = getattr(getattr(request, "state", None), "request_id", None)
        if rid:
            return str(rid)
        headers = getattr(request, "headers", None)
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
