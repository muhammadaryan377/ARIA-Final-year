"""Feedback + Personalization routes.

Captures user signals (insight ratings, suggestion clicks, chart preferences)
and uses them to improve presentation-layer ranking without affecting
analytical truth. All personalization operates on the decision/experience
layer only.
"""

import json
import time
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from core.config import get_session, save_session
from core.deps import get_current_user

router = APIRouter()

# ── Preference storage path ─────────────────────────────────────────────
_PREFS_DIR = Path(__file__).resolve().parent.parent / "data" / "preferences"


def _prefs_path(user_id: str) -> Path:
    _PREFS_DIR.mkdir(parents=True, exist_ok=True)
    return _PREFS_DIR / f"user_{user_id}_prefs.json"


def _load_prefs(user_id: str) -> dict:
    p = _prefs_path(user_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "chart_preferences": {},
        "suggestion_clicks": {},
        "insight_ratings": [],
        "chart_type_history": {},
        "preferred_palette": None,
        "preferred_density": "auto",
    }


def _save_prefs(user_id: str, prefs: dict):
    p = _prefs_path(user_id)
    p.write_text(json.dumps(prefs, indent=2, default=str), encoding="utf-8")


# ── Models ──────────────────────────────────────────────────────────────

class InsightRating(BaseModel):
    insight_id: str = ""
    question: str = ""
    rating: str  # "useful" | "not_useful"
    category: str = "analytics"  # analytics | insight | prediction | prescription


class SuggestionClick(BaseModel):
    suggestion_text: str
    source: str = "dashboard"  # dashboard | analytics | idle


class ChartPreference(BaseModel):
    chart_type: str
    context: str = ""  # e.g. "bar_chart_default", "time_series"


class GeneralFeedback(BaseModel):
    message: str = ""
    category: str = "general"


# ── Endpoints ───────────────────────────────────────────────────────────

@router.post("/api/feedback")
def submit_feedback(req: GeneralFeedback, user: dict = Depends(get_current_user)):
    """General feedback (legacy endpoint)."""
    return {"ok": True, "message": "Thanks for your feedback."}


@router.post("/api/feedback/insight")
def rate_insight(payload: InsightRating, user: dict = Depends(get_current_user)):
    """Rate an insight as useful or not useful. Used for RL reward signal."""
    prefs = _load_prefs(user["user_id"])
    prefs["insight_ratings"].append({
        "question": payload.question,
        "rating": payload.rating,
        "category": payload.category,
        "timestamp": time.time(),
    })
    # Keep last 200 ratings
    prefs["insight_ratings"] = prefs["insight_ratings"][-200:]
    _save_prefs(user["user_id"], prefs)
    return {"ok": True}


@router.post("/api/feedback/suggestion")
def track_suggestion_click(payload: SuggestionClick, user: dict = Depends(get_current_user)):
    """Track when user clicks a suggestion. Used for suggestion ranking."""
    prefs = _load_prefs(user["user_id"])
    text = payload.suggestion_text.strip()
    clicks = prefs.get("suggestion_clicks", {})
    clicks[text] = clicks.get(text, 0) + 1
    prefs["suggestion_clicks"] = clicks
    _save_prefs(user["user_id"], prefs)
    return {"ok": True}


@router.post("/api/feedback/chart")
def track_chart_preference(payload: ChartPreference, user: dict = Depends(get_current_user)):
    """Track chart type selection. Used for chart recommendation."""
    prefs = _load_prefs(user["user_id"])
    history = prefs.get("chart_type_history", {})
    key = payload.context or "general"
    if key not in history:
        history[key] = {}
    history[key][payload.chart_type] = history[key].get(payload.chart_type, 0) + 1
    prefs["chart_type_history"] = history
    _save_prefs(user["user_id"], prefs)
    return {"ok": True}


@router.get("/api/preferences")
def get_preferences(user: dict = Depends(get_current_user)):
    """Get user preferences for personalization."""
    return {"ok": True, "preferences": _load_prefs(user["user_id"])}


@router.post("/api/preferences")
def save_preferences(payload: dict, user: dict = Depends(get_current_user)):
    """Save user preferences (palette, density, etc.)."""
    prefs = _load_prefs(user["user_id"])
    if "preferred_palette" in payload:
        prefs["preferred_palette"] = payload["preferred_palette"]
    if "preferred_density" in payload:
        prefs["preferred_density"] = payload["preferred_density"]
    if "preferred_explanation_style" in payload:
        prefs["preferred_explanation_style"] = payload["preferred_explanation_style"]
    _save_prefs(user["user_id"], prefs)
    return {"ok": True}
