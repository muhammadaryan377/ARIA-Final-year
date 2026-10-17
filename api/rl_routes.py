"""RL observability + feedback API (additive; no analytics logic here)."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from core.config import get_session  # noqa: F401  (session scope parity)
from core.deps import get_current_user
from core.rl import IMPACT_UNKNOWN_NOTE, load_recent_events, record_event

router = APIRouter()


class FeedbackEvent(BaseModel):
    event: str = Field(min_length=1, max_length=64)
    state: dict = Field(default_factory=dict)
    action: dict = Field(default_factory=dict)
    outcome: dict = Field(default_factory=dict)
    reward: dict = Field(default_factory=dict)


@router.post("/api/rl/event")
def post_event(body: FeedbackEvent, request: Request, user: dict = Depends(get_current_user)):
    """Record a structured UI feedback event (chart kept/changed, insight saved/dismissed...)."""
    import json
    try:
        blob = json.dumps(body.model_dump(), allow_nan=False)
    except (ValueError, TypeError, RecursionError):
        raise HTTPException(status_code=422, detail="Feedback must contain finite JSON values.")
    if len(blob.encode("utf-8")) > 7000:
        raise HTTPException(status_code=413, detail="Feedback is too large. Send summaries only.")
    record_event(
        user["user_id"],
        event=body.event[:64],
        state=body.state,
        action=body.action,
        outcome=body.outcome,
        reward=body.reward,
    )
    return {"ok": True, "impact_note": IMPACT_UNKNOWN_NOTE,
            "persistence": "best_effort", "reward_source": "client_feedback"}


@router.get("/api/rl/events")
def get_events(request: Request, user: dict = Depends(get_current_user), limit: int = 100):
    """Observability: recent RL telemetry for this user (debug/admin)."""
    limit = max(1, min(int(limit), 500))
    return {"ok": True, "events": load_recent_events(user["user_id"], limit=limit)}
