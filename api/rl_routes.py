"""RL observability + feedback API (additive; no analytics logic here)."""

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from core.config import get_session  # noqa: F401  (session scope parity)
from core.deps import get_current_user
from core.rl import IMPACT_UNKNOWN_NOTE, load_recent_events, record_event

router = APIRouter()


class FeedbackEvent(BaseModel):
    event: str
    state: dict = {}
    action: dict = {}
    outcome: dict = {}
    reward: dict = {}


@router.post("/api/rl/event")
def post_event(body: FeedbackEvent, request: Request, user: dict = Depends(get_current_user)):
    """Record a structured UI feedback event (chart kept/changed, insight saved/dismissed...)."""
    record_event(
        user["user_id"],
        event=body.event[:64],
        state=body.state,
        action=body.action,
        outcome=body.outcome,
        reward=body.reward,
    )
    return {"ok": True, "impact_note": IMPACT_UNKNOWN_NOTE}


@router.get("/api/rl/events")
def get_events(request: Request, user: dict = Depends(get_current_user), limit: int = 100):
    """Observability: recent RL telemetry for this user (debug/admin)."""
    limit = max(1, min(int(limit), 500))
    return {"ok": True, "events": load_recent_events(user["user_id"], limit=limit)}
