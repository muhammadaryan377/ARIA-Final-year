"""ARIA Intelligence Module — API routes.

Provides Explore / Hypotheses / Goals / What-If / Reports / Coverage endpoints.
All persistence is per-user per-dataset under data/users/user_{id}/intelligence/.

Router prefix: /api/intelligence
"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from core.config import get_session
from core.deps import get_current_user, require_data, require_writable, get_request_id
from services import intelligence_service as intel

log = logging.getLogger("aria.intelligence.api")

router = APIRouter(prefix="/api/intelligence", tags=["intelligence"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ExploreRequest(BaseModel):
    metric: str
    breakdown: str | None = None
    secondary_breakdown: str | None = None
    secondaryBreakdown: str | None = None
    time: str | None = None
    time_grain: str | None = None
    timeGrain: str | None = None
    compare: str | None = None
    comparison: str | None = None
    date_range: dict | str | None = None
    dateRange: dict | str | None = None
    filters: list | None = None

    class Config:
        extra = "allow"


class WhatIfRequest(BaseModel):
    variable: str | None = None
    metric: str | None = None
    column: str | None = None
    change_type: str | None = None
    changeType: str | None = None
    change_value: float | None = None
    changeValue: float | None = None
    apply_to: str | None = None
    applyTo: str | None = None
    scope: str | None = None

    class Config:
        extra = "allow"


class GoalRequest(BaseModel):
    title: str | None = None
    name: str | None = None
    metric: str | None = None
    target_value: float | str | None = None
    target: float | str | None = None
    targetValue: float | str | None = None
    current_value: float | str | None = None
    current: float | str | None = None
    currentValue: float | str | None = None
    due_date: str | None = None
    dueDate: str | None = None
    due: str | None = None
    status: str | None = None

    class Config:
        extra = "allow"


class ReportRequest(BaseModel):
    name: str = "ARIA Report"
    description: str = ""
    date_range: str | None = None
    dateRange: str | None = None
    sections: list | None = None
    template: str = "professional"

    class Config:
        extra = "allow"


class AnalysisRequest(BaseModel):
    title: str = "Untitled Analysis"
    name: str | None = None
    metric: str | None = None
    breakdown: str | None = None
    time_grain: str | None = None
    result: dict | None = None
    result_summary: str | None = None
    summary: str | None = None

    class Config:
        extra = "allow"


class ScenarioRequest(BaseModel):
    title: str | None = None
    name: str | None = None
    variable: str | None = None
    change_value: float | None = None
    change_type: str | None = None
    result: dict | None = None

    class Config:
        extra = "allow"


class TemplateRequest(BaseModel):
    name: str = "Untitled template"
    description: str = ""
    sections: list | None = None
    template: str | None = None

    class Config:
        extra = "allow"


class DigestRequest(BaseModel):
    name: str = "Daily Digest"
    description: str = ""
    frequency: str = "daily"
    enabled: bool = True
    template: str = "professional"
    sections: list | None = None

    class Config:
        extra = "allow"


class DigestPatch(BaseModel):
    name: str | None = None
    description: str | None = None
    frequency: str | None = None
    enabled: bool | None = None
    template: str | None = None
    sections: list | None = None

    class Config:
        extra = "allow"


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def _get_session_and_user(request: Request, user: dict = Depends(get_current_user)):
    """Resolve session + ownership for Intelligence endpoints."""
    session = get_session(user["user_id"])
    provider = session.get("provider")
    if not provider:
        provider_error = session.get("_provider_error", "")
        detail = "No LLM provider available."
        if provider_error:
            detail += f" ({provider_error})"
        detail += " Connect a source or configure a provider in Settings."
        raise HTTPException(status_code=400, detail=detail)
    if session.get("source_type") == "relational" and not session.get("db"):
        raise HTTPException(status_code=400, detail="Connect to a database first.")
    if session.get("source_type") == "semi-structured" and not session.get("db_uri"):
        raise HTTPException(status_code=400, detail="Upload a CSV or PDF first.")
    return session, user


def _dataset_id(session: dict) -> str:
    return intel._dataset_id(session)


# ---------------------------------------------------------------------------
# GET /meta
# ---------------------------------------------------------------------------

@router.get("/meta")
def get_meta(request: Request, user: dict = Depends(get_current_user)):
    """Return available metrics, dimensions, date columns for Explore."""
    try:
        session, user = _get_session_and_user(request, user)
    except HTTPException:
        return {"ok": True, "metrics": [], "measures": [], "dimensions": [], "breakdowns": [], "date_columns": [], "tables": [], "time_grains": ["Monthly"], "comparisons": ["Previous Period"], "date_ranges": ["Last 12 months"], "filterable_columns": [], "variables": [], "change_types": ["Increase", "Decrease"], "apply_targets": ["All data"], "ranges": {}, "reason": "NO_SCHEMA", "hint": "Connect a data source or run Extract Schema (POST /api/extract_schema)"}
    return intel.get_intelligence_meta(session)


# ---------------------------------------------------------------------------
# POST /explore
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Short-TTL result cache for explore/meta (single-process; Redis covers
# multi-worker via the session store). Key = (user, schema mtime, params):
# same dataset + same question -> identical result. TTL 120s, cap 200
# entries oldest-first. Never caches errors/refusals.
# ---------------------------------------------------------------------------
_EXPLORE_CACHE: dict = {}
_EXPLORE_CACHE_TTL = 120
_EXPLORE_CACHE_MAX = 200


def _explore_cache_get(key):
    try:
        ts, val = _EXPLORE_CACHE.get(key, (0, None))
        if val is not None and __import__("time").time() - ts < _EXPLORE_CACHE_TTL:
            return val
        _EXPLORE_CACHE.pop(key, None)
    except Exception:
        pass
    return None


def _explore_cache_put(key, val):
    try:
        if len(_EXPLORE_CACHE) >= _EXPLORE_CACHE_MAX:
            _EXPLORE_CACHE.pop(next(iter(_EXPLORE_CACHE)))
        _EXPLORE_CACHE[key] = (__import__("time").time(), val)
    except Exception:
        pass


def _schema_cache_tag(session: dict) -> str:
    try:
        import os as _os
        p = session.get("schema_path")
        s = str(p) if p is not None else ""
        return f"{s}:{_os.path.getmtime(s)}" if s and _os.path.exists(s) else s
    except Exception:
        return ""


@router.post("/explore")
def run_explore(req: ExploreRequest, request: Request, user: dict = Depends(get_current_user)):
    """Run a server-side aggregation analysis."""
    try:
        session, user = _get_session_and_user(request, user)
    except HTTPException as exc:
        return {"ok": False, "detail": exc.detail, "code": "NO_DATA"}
    payload = req.model_dump()
    # normalize frontend aliases for service
    if not payload.get("time") and (payload.get("time_grain") or payload.get("timeGrain")):
        payload["time"] = payload.get("time_grain") or payload.get("timeGrain")
    if not payload.get("compare") and payload.get("comparison"):
        payload["compare"] = payload["comparison"]
    if payload.get("dateRange") and not payload.get("date_range"):
        payload["date_range"] = payload["dateRange"]
    try:
        _ckey = (user["user_id"], _schema_cache_tag(session),
                 __import__("json").dumps(payload, sort_keys=True, default=str))
        _hit = _explore_cache_get(_ckey)
        if _hit is not None:
            return _hit
    except Exception:
        _ckey = None
    result = intel.run_explore(session, payload)
    rid = get_request_id(request)
    if result.get("ok"):
        log.info("Explore OK [rid=%s] metric=%s rows=%s", rid, req.metric, result.get("row_count"))
    else:
        log.warning("Explore rejected [rid=%s] reason=%s", rid, result.get("detail"))
    # wrap so api.js `data?.data || data` works
    result["data"] = {k: v for k, v in result.items() if k != "data"}
    if result.get("ok") and _ckey is not None:
        _explore_cache_put(_ckey, result)
    return result


# ---------------------------------------------------------------------------
# POST /hypotheses
# ---------------------------------------------------------------------------

@router.get("/hypotheses")
@router.post("/hypotheses")
def list_hypotheses(request: Request, user: dict = Depends(get_current_user)):
    """Return hypothesis candidates from the connected dataset."""
    try:
        session, user = _get_session_and_user(request, user)
    except HTTPException:
        return {"ok": True, "candidates": [], "items": [], "data": [], "total_candidates": 0, "total": 0, "count": 0, "reason": "NO_SCHEMA", "hint": "Connect a data source or run Extract Schema (POST /api/extract_schema)"}
    params = dict(request.query_params) if request else {}
    result = intel.generate_hypotheses(session, params)
    result["data"] = result.get("data", [])
    result["items"] = result.get("items", result["data"])
    return result


@router.get("/hypotheses/{hypothesis_id}")
def get_hypothesis(hypothesis_id: str, request: Request, user: dict = Depends(get_current_user)):
    """Single hypothesis detail for the Hypotheses tab."""
    try:
        session, user = _get_session_and_user(request, user)
    except HTTPException as exc:
        return {"ok": False, "detail": exc.detail, "code": "NO_DATA"}
    detail = intel.get_hypothesis_detail(session, hypothesis_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Hypothesis not found.")
    return {"ok": True, "data": detail, **detail}


# ---------------------------------------------------------------------------
# Goals CRUD
# ---------------------------------------------------------------------------

@router.get("/goals")
def list_goals(request: Request, user: dict = Depends(get_current_user), status: str = "active"):
    """List goals filtered by dataset and status."""
    qp = dict(request.query_params) if request else {}
    status = qp.get("status", status) or "active"
    session = get_session(user["user_id"])
    did = _dataset_id(session)
    items = intel.list_goals(user["user_id"], did, status)
    return {"ok": True, "items": items, "data": items, "total": len(items), "count": len(items)}


@router.post("/goals")
def create_goal(req: GoalRequest, request: Request, user: dict = Depends(get_current_user)):
    session = get_session(user["user_id"])
    did = _dataset_id(session)
    data = req.model_dump(exclude_unset=True)
    if not (data.get("title") or data.get("name")):
        raise HTTPException(status_code=422, detail="Give the goal a title.")
    if data.get("target_value") is None and data.get("target") is None and data.get("targetValue") is None:
        raise HTTPException(status_code=422, detail="Give the goal a target value.")
    goal = intel.create_goal(user["user_id"], did, data)
    return {"ok": True, "item": goal, "data": goal}


@router.patch("/goals/{goal_id}")
def update_goal(goal_id: str, req: GoalRequest, request: Request, user: dict = Depends(get_current_user)):
    """Update a goal's status or fields."""
    session = get_session(user["user_id"])
    did = _dataset_id(session)
    updated = intel.update_goal(user["user_id"], goal_id, req.model_dump(exclude_unset=True))
    if not updated:
        raise HTTPException(status_code=404, detail="Goal not found.")
    return {"ok": True, "item": updated, "data": updated}


@router.delete("/goals/{goal_id}")
def delete_goal(goal_id: str, request: Request, user: dict = Depends(get_current_user)):
    """Delete a goal."""
    session = get_session(user["user_id"])
    if not intel.delete_goal(user["user_id"], goal_id):
        raise HTTPException(status_code=404, detail="Goal not found.")
    return {"ok": True}


@router.post("/goals/{goal_id}/refresh")
def refresh_goal(goal_id: str, request: Request, user: dict = Depends(get_current_user)):
    """Recompute one goal's current value from live data (auto-progress).

    Loads the stored goal, re-runs the run_explore-equivalent aggregation
    for goal.metric (overall, no breakdown) on the user's current dataset,
    stores the live total as current, recomputes progress + status pill,
    and returns the updated goal. Honest ok:False (goal unchanged) when the
    metric is missing or the aggregation cannot run.
    """
    session = get_session(user["user_id"])
    did = _dataset_id(session)
    result = intel.refresh_goal(user["user_id"], did, goal_id, session)
    if result.get("ok"):
        return {"ok": True, "item": result.get("item"), "data": result.get("data")}
    if result.get("code") == "NOT_FOUND":
        raise HTTPException(status_code=404, detail=result.get("detail") or "Goal not found.")
    return {"ok": False, "detail": result.get("detail"),
            "code": result.get("code"), "item": result.get("item"),
            "data": result.get("data")}


# ---------------------------------------------------------------------------
# What-If
# ---------------------------------------------------------------------------

@router.get("/what-if/meta")
def whatif_meta(request: Request, user: dict = Depends(get_current_user)):
    """Return available variables and slider ranges."""
    try:
        session, user = _get_session_and_user(request, user)
    except HTTPException:
        return {"ok": True, "variables": [], "change_types": ["Increase by", "Decrease by"], "changeTypes": ["Increase by", "Decrease by"], "apply_targets": ["All data"], "applyTargets": ["All data"], "scopes": ["All data"], "ranges": {}, "range": None, "reason": "NO_SCHEMA", "hint": "Connect a data source or run Extract Schema (POST /api/extract_schema)"}
    return intel.get_whatif_meta(session)


@router.post("/what-if/run")
def run_whatif(req: WhatIfRequest, request: Request, user: dict = Depends(get_current_user)):
    """Run a What-If scenario."""
    try:
        session, user = _get_session_and_user(request, user)
    except HTTPException as exc:
        return {"ok": False, "detail": exc.detail, "code": "NO_DATA"}
    payload = req.model_dump()
    if not payload.get("variable"):
        payload["variable"] = payload.get("metric") or payload.get("column")
    if not payload.get("change_type"):
        payload["change_type"] = payload.get("changeType") or "increase"
    if payload.get("change_value") is None:
        payload["change_value"] = payload.get("changeValue", 10)
    if not payload.get("apply_to"):
        payload["apply_to"] = payload.get("applyTo") or payload.get("scope") or "All data"
    result = intel.run_whatif(session, payload)
    rid = get_request_id(request)
    if result.get("ok"):
        result["data"] = {k: v for k, v in result.items() if k != "data"}
        log.info("What-If OK [rid=%s] variable=%s", rid, payload.get("variable"))
    else:
        log.warning("What-If rejected [rid=%s] reason=%s", rid, result.get("detail"))
    return result


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

@router.get("/analyses")
def list_analyses(request: Request, user: dict = Depends(get_current_user)):
    """List saved analyses."""
    session = get_session(user["user_id"])
    did = _dataset_id(session)
    params = dict(request.query_params) if request else {}
    items = intel.list_analyses(user["user_id"], did, params)
    return {"ok": True, "items": items, "data": items, "total": len(items), "count": len(items)}


@router.post("/analyses")
def save_analysis(req: AnalysisRequest, request: Request, user: dict = Depends(get_current_user)):
    """Save an analysis."""
    session = get_session(user["user_id"])
    did = _dataset_id(session)
    analysis = intel.save_analysis(user["user_id"], did, req.model_dump())
    return {"ok": True, "item": analysis, "data": analysis}


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@router.get("/what-if/scenarios")
def list_scenarios(request: Request, user: dict = Depends(get_current_user)):
    session = get_session(user["user_id"])
    did = _dataset_id(session)
    params = dict(request.query_params) if request else {}
    items = intel.list_scenarios(user["user_id"], did, params)
    return {"ok": True, "items": items, "data": items, "total": len(items)}


@router.post("/what-if/scenarios")
def save_scenario(req: ScenarioRequest, request: Request, user: dict = Depends(get_current_user)):
    session = get_session(user["user_id"])
    did = _dataset_id(session)
    item = intel.save_scenario(user["user_id"], did, req.model_dump())
    return {"ok": True, "item": item, "data": item}


@router.get("/reports/templates")
def list_report_templates(request: Request, user: dict = Depends(get_current_user)):
    items = intel.list_report_templates(user["user_id"])
    return {"ok": True, "items": items, "data": items, "total": len(items)}


@router.post("/reports/templates")
def save_report_template(req: TemplateRequest, request: Request, user: dict = Depends(get_current_user)):
    item = intel.save_report_template(user["user_id"], req.model_dump())
    return {"ok": True, "item": item, "data": item}


@router.post("/reports/schedule")
def schedule_report(req: ReportRequest, request: Request, user: dict = Depends(get_current_user)):
    session = get_session(user["user_id"])
    did = _dataset_id(session)
    payload = req.model_dump(); payload["scheduled"] = True
    item = intel.generate_report(user["user_id"], did, payload)
    return {"ok": True, "item": item, "data": item, "message": "Report scheduled."}


@router.post("/reports/share")
def share_report(req: ReportRequest, request: Request, user: dict = Depends(get_current_user)):
    session = get_session(user["user_id"])
    did = _dataset_id(session)
    payload = req.model_dump(); payload["shared"] = True
    item = intel.generate_report(user["user_id"], did, payload)
    return {"ok": True, "item": item, "data": item, "message": "Share link created.", "share_url": "#/intelligence?tab=reports"}


@router.post("/reports/export")
def export_report(request: Request, user: dict = Depends(get_current_user)):
    session = get_session(user["user_id"])
    did = _dataset_id(session)
    item = intel.generate_report(user["user_id"], did, {"name": "Export", "template": "professional", "sections": []})
    return {"ok": True, "item": item, "data": item, "message": "Export started."}


@router.get("/reports")
def list_reports(request: Request, user: dict = Depends(get_current_user)):
    """List generated reports."""
    session = get_session(user["user_id"])
    did = _dataset_id(session)
    items = intel.list_reports(user["user_id"], did)
    return {"ok": True, "items": items, "data": items, "total": len(items)}


@router.post("/reports/generate")
def generate_report(req: ReportRequest, request: Request, user: dict = Depends(get_current_user)):
    """Generate a structured report."""
    session = get_session(user["user_id"])
    did = _dataset_id(session)
    payload = req.model_dump()
    if payload.get("dateRange") and not payload.get("date_range"):
        payload["date_range"] = payload["dateRange"]
    report = intel.generate_report(user["user_id"], did, payload)
    return {"ok": True, "item": report, "data": report}




@router.delete("/analyses/{analysis_id}")
def delete_analysis(analysis_id: str, request: Request, user: dict = Depends(get_current_user)):
    """Delete a saved analysis."""
    if not intel.delete_analysis(user["user_id"], analysis_id):
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return {"ok": True}


@router.delete("/what-if/scenarios/{scenario_id}")
def delete_scenario(scenario_id: str, request: Request, user: dict = Depends(get_current_user)):
    """Delete a saved scenario."""
    if not intel.delete_scenario(user["user_id"], scenario_id):
        raise HTTPException(status_code=404, detail="Scenario not found.")
    return {"ok": True}


@router.delete("/reports/{report_id}")
def delete_report(report_id: str, request: Request, user: dict = Depends(get_current_user)):
    """Delete a generated report."""
    if not intel.delete_report(user["user_id"], report_id):
        raise HTTPException(status_code=404, detail="Report not found.")
    return {"ok": True}


@router.get("/reports/{report_id}")
def get_report(report_id: str, request: Request, user: dict = Depends(get_current_user)):
    """Single report for preview."""
    session = get_session(user["user_id"])
    did = _dataset_id(session)
    item = intel.get_report(user["user_id"], did, report_id)
    if not item:
        raise HTTPException(status_code=404, detail="Report not found.")
    return {"ok": True, "item": item, "data": item}


# ---------------------------------------------------------------------------
# Reporting Agent (Phase 3, additive-only): styled HTML render + re-run.
# PDF path is HTML download + print-to-PDF (pdfplumber is read-only).
# ---------------------------------------------------------------------------

@router.get("/reports/{report_id}/html")
def get_report_html(report_id: str, request: Request, user: dict = Depends(get_current_user)):
    """Render a stored report as styled, printable HTML (Reporting Agent)."""
    from fastapi.responses import HTMLResponse

    from reporting.agent import build_report, intel_report_to_payload

    session = get_session(user["user_id"])
    did = _dataset_id(session)
    item = intel.get_report(user["user_id"], did, report_id)
    if not item:
        raise HTTPException(status_code=404, detail="Report not found.")
    payload = intel_report_to_payload(item)
    built = build_report("intelligence", payload)
    return HTMLResponse(content=built["html"], media_type="text/html")


@router.post("/reports/{report_id}/rerun")
def rerun_report(report_id: str, request: Request, user: dict = Depends(get_current_user)):
    """Re-run a stored report definition with fresh numbers (new history entry)."""
    session = get_session(user["user_id"])
    did = _dataset_id(session)
    stored = intel.get_report(user["user_id"], did, report_id)
    if not stored:
        raise HTTPException(status_code=404, detail="Report not found.")
    content = stored.get("content") or {}
    definition = {
        "name": stored.get("name") or "ARIA Report",
        "description": stored.get("description") or "",
        "date_range": stored.get("date_range"),
        "sections": stored.get("sections") or [],
        "template": stored.get("template") or "professional",
    }
    draft_items = content.get("draft_items") or stored.get("draft_items") or []
    if draft_items:
        definition["items"] = draft_items
    report = intel.generate_report(user["user_id"], did, definition)
    return {"ok": True, "item": report, "data": report}


# ---------------------------------------------------------------------------
# Scheduled digests (Phase 4, additive-only)
#
# Per-user daily/weekly configs stored as JSON per user
# (reporting/digests.py). Background generation reuses the existing engines
# (generate_report over live dashboard content + top insights, forecast
# snapshot only where a saved forecast exists). Each run persists a normal
# report so digest history carries its definition for re-run. The scheduler
# is dependency-free (threading/timer, lazy start + atexit shutdown).
# ---------------------------------------------------------------------------

def _digests():
    from reporting import digests as _d
    try:
        _d.ensure_started()
    except Exception:
        pass
    return _d


@router.get("/digests")
def list_digests(request: Request, user: dict = Depends(get_current_user)):
    """List this user's digest schedules with last-run status."""
    items = _digests().list_digests(user["user_id"])
    return {"ok": True, "items": items, "data": items, "total": len(items)}


@router.post("/digests")
def create_digest(req: DigestRequest, request: Request, user: dict = Depends(get_current_user)):
    """Create a digest schedule (daily/weekly, enabled by default)."""
    freq = str(req.frequency or "daily").strip().lower()
    if freq not in ("daily", "weekly"):
        raise HTTPException(status_code=422, detail="Frequency must be daily or weekly.")
    item = _digests().create_digest(user["user_id"], req.model_dump())
    return {"ok": True, "item": item, "data": item}


@router.patch("/digests/{digest_id}")
def update_digest(digest_id: str, req: DigestPatch, request: Request, user: dict = Depends(get_current_user)):
    """Update a digest schedule (enable/disable, frequency, name, ...)."""
    patch = req.model_dump(exclude_unset=True)
    if "frequency" in patch and patch["frequency"] is not None:
        freq = str(patch["frequency"]).strip().lower()
        if freq not in ("daily", "weekly"):
            raise HTTPException(status_code=422, detail="Frequency must be daily or weekly.")
        patch["frequency"] = freq
    updated = _digests().update_digest(user["user_id"], digest_id, patch)
    if not updated:
        raise HTTPException(status_code=404, detail="Digest not found.")
    return {"ok": True, "item": updated, "data": updated}


@router.delete("/digests/{digest_id}")
def delete_digest(digest_id: str, request: Request, user: dict = Depends(get_current_user)):
    """Delete a digest schedule."""
    if not _digests().delete_digest(user["user_id"], digest_id):
        raise HTTPException(status_code=404, detail="Digest not found.")
    return {"ok": True}


@router.post("/digests/{digest_id}/run")
def run_digest(digest_id: str, request: Request, user: dict = Depends(get_current_user)):
    """One-click digest run: generate now from live data (new history entry).

    Failures record a failed run entry on the schedule and return
    ok:False — never a crash.
    """
    from reporting import digests as _mod

    _mod.ensure_started()
    digest = _mod.get_digest(user["user_id"], digest_id)
    if not digest:
        raise HTTPException(status_code=404, detail="Digest not found.")
    session = get_session(user["user_id"])
    did = _dataset_id(session)
    report = _mod.generate_digest_now(user["user_id"], digest, dataset_id=did)
    if isinstance(report, dict) and report.get("ok") is False and not report.get("id"):
        return {"ok": False, "detail": report.get("detail") or "Digest run failed.",
                "code": report.get("code") or "DIGEST_FAILED"}
    return {"ok": True, "item": report, "data": report}


# ---------------------------------------------------------------------------
# Coverage audit
# ---------------------------------------------------------------------------

@router.get("/coverage")
def get_coverage(request: Request, user: dict = Depends(get_current_user)):
    """Return what ARIA examined in the last analysis run."""
    try:
        session, user = _get_session_and_user(request, user)
    except HTTPException:
        return {"ok": True, "tables": [], "columns": [], "total_cells": 0, "reason": "NO_SCHEMA", "hint": "Connect a data source or run Extract Schema (POST /api/extract_schema)"}
    return intel.get_coverage(session)


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------

@router.post("/refresh")
def refresh(request: Request, user: dict = Depends(get_current_user)):
    """Re-run analytics and return updated Intelligence state."""
    try:
        session, user = _get_session_and_user(request, user)
    except HTTPException:
        return {"ok": True, "meta": {"ok": True, "metrics": [], "dimensions": [], "date_columns": [], "tables": [], "reason": "NO_SCHEMA", "hint": "Connect a data source or run Extract Schema (POST /api/extract_schema)"}, "coverage": {"ok": True, "tables": [], "columns": [], "total_cells": 0, "reason": "NO_SCHEMA", "hint": "Connect a data source or run Extract Schema (POST /api/extract_schema)"}, "active_goals": 0, "timestamp": time.time(), "reason": "NO_SCHEMA", "hint": "Connect a data source or run Extract Schema (POST /api/extract_schema)"}
    meta = intel.get_intelligence_meta(session)
    coverage = intel.get_coverage(session)
    did = _dataset_id(session)
    goals = intel.list_goals(user["user_id"], did, "active")
    return {
        "ok": True,
        "meta": meta,
        "coverage": coverage,
        "active_goals": len(goals),
        "timestamp": time.time(),
    }
