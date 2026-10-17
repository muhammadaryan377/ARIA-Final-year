"""Dashboard verification routes — Phase 1 (additive only).

Exposes POST /api/verify/dashboard: accepts the dashboard config payload,
runs the Verification Agent, and returns its verdict. Phase 2 adds
POST /api/verify/{ask,forecast,prescription}, mirroring the dashboard
pattern. No existing routes are modified here; wiring happens in app.py.
"""

from fastapi import APIRouter, Depends, Request

from core.config import get_session
from core.deps import get_current_user

router = APIRouter(tags=["verification"])


def _payload_of(body) -> dict:
    """Unwrap {config/payload} envelopes; anything else verifies as-is."""
    if isinstance(body, dict):
        for key in ("config", "payload"):
            if isinstance(body.get(key), dict):
                return body[key]
        return body
    return {}


@router.post("/api/verify/dashboard")
def verify_dashboard_endpoint(body: dict, request: Request, user: dict = Depends(get_current_user)):
    """Verify a dashboard config payload without touching stored data."""
    from verification.agent import verify_dashboard

    session = get_session(user["user_id"])
    verdict = verify_dashboard(_payload_of(body), dict(session or {}))
    return {"ok": True, "verdict": verdict}


@router.post("/api/verify/ask")
def verify_ask_endpoint(body: dict, request: Request, user: dict = Depends(get_current_user)):
    """Verify an ask (/api/ask) answer payload without touching stored data."""
    from verification.agent import verify_ask

    session = get_session(user["user_id"])
    verdict = verify_ask(_payload_of(body), dict(session or {}))
    return {"ok": True, "verdict": verdict}


@router.post("/api/verify/forecast")
def verify_forecast_endpoint(body: dict, request: Request, user: dict = Depends(get_current_user)):
    """Verify a forecast payload without touching stored data."""
    from verification.agent import verify_forecast

    session = get_session(user["user_id"])
    verdict = verify_forecast(_payload_of(body), dict(session or {}))
    return {"ok": True, "verdict": verdict}


@router.post("/api/verify/prescription")
def verify_prescription_endpoint(body: dict, request: Request, user: dict = Depends(get_current_user)):
    """Verify a prescription payload without touching stored data."""
    from verification.agent import verify_prescription

    session = get_session(user["user_id"])
    verdict = verify_prescription(_payload_of(body), dict(session or {}))
    return {"ok": True, "verdict": verdict}
