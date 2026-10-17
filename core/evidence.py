"""Centralized evidence sufficiency gate (spec sections 4-6, 10-12).

ONE canonical gate for every substantive conclusion (insights, decisions,
trends, anomalies, forecasts, recommendations, scenarios, monitoring,
hypothesis, relationships, narrative). No agent re-invents the minimum.

Global rule (reuses prescription_engine.MIN_N == hypothesis_engine.MIN_N == 10):
    rows < 10  -> INSUFFICIENT_EVIDENCE for ALL substantive conclusions.
    rows >= 10 -> per-capability requirements below (all REUSED, never invented).

Thresholds reused (with source):
- GLOBAL_MIN_ROWS = 10
    prescription_engine/engine.py MIN_N = 10 (fit_relationship, quality_gate,
    _stay_lengths, run_scenario)
    hypothesis_engine/engine.py MIN_N = 10 (run_correlation, run_trend,
    preview needs_more_data/min_required, _usable_count adequacy)
    api/prescription_routes.py _enrich_context "below the 10-observation minimum"
    api/decision_center_routes.py _forecast_eligibility n >= 10
- FORECAST_MIN_POINTS = 5
    prediction/agent.py PredictionAgent(min_points=5) default
    api/predictive_routes.py eligibility "min_points": 5, recommended
    distinct >= 5, "needs at least 5" copy
- TREND_MIN_PERIODS = 6
    api/insights_routes.py InsightsEngine._detect_trends len(rows) < 6
- ANOMALY_MIN_PERIODS = 6
    api/insights_routes.py InsightsEngine._detect_anomalies len(rows) < 6
- CORRELATION_MIN_SHARED_PERIODS = 6
    api/insights_routes.py InsightsEngine._detect_correlations len(rows) < 6
- DISTRIBUTION_MIN_ROWS = 20
    api/insights_routes.py InsightsEngine._detect_distributions len(vals) < 20
- OUTLIER_MIN_ROWS = 10
    api/insights_routes.py InsightsEngine._detect_value_outliers len(vals) < 10
- SCENARIO_MIN_OBS = 10 (== MIN_N, prescription run_scenario/fit_relationship)
- HYPOTHESIS_MIN_OBS = 10 (== MIN_N, hypothesis adequacy)
- GROUP_MIN_PER_GROUP = 5
    hypothesis_engine run_group_comparison "at least 5 each"
    prescription _investigate_cands groups len(vs) >= 5
- SEGMENT_MIN_GROUPS = 3 usable segments / GROUP comparison needs 2 groups
    hypothesis_engine run_segments "fewer than three usable segments"
- FORECAST_VALID_THRESHOLD = 70.0 (PredictionAgent valid_threshold)

Capability states (canonical, additive everywhere):
    supported | insufficient_evidence | unavailable | not_applicable | error
Each state carries a business-language `reason` (never technical jargon).

Copy contract:
- 0 rows:  "No usable records were found."
- 1-9:     "ARIA needs at least 10 usable records before it can reliably
             identify business patterns." + "Only N usable records were
             found..."
- 10+ but capability-lacking: capability-specific sentence naming what is
  needed (periods, baseline, shared periods, ...).

LLM rule: storytelling/recommendation paths must call gate FIRST via
`llm_allowed()` / `insufficient_narrative()`. When insufficient, return the
honest deterministic state (+ optional dataset facts: name, usable count,
columns, quality observations, what's needed). The LLM must never be asked
to compensate for missing evidence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Canonical thresholds (REUSED — see module docstring for each source).
# ---------------------------------------------------------------------------
GLOBAL_MIN_ROWS = 10
FORECAST_MIN_POINTS = 5
TREND_MIN_PERIODS = 6
ANOMALY_MIN_PERIODS = 6
CORRELATION_MIN_SHARED_PERIODS = 6
DISTRIBUTION_MIN_ROWS = 20
OUTLIER_MIN_ROWS = 10
SCENARIO_MIN_OBS = 10
HYPOTHESIS_MIN_OBS = 10
GROUP_MIN_PER_GROUP = 5
FORECAST_VALID_THRESHOLD = 70.0

SUPPORTED = "supported"
INSUFFICIENT = "insufficient_evidence"
UNAVAILABLE = "unavailable"
NOT_APPLICABLE = "not_applicable"
ERROR = "error"

VALID_STATUSES = (SUPPORTED, INSUFFICIENT, UNAVAILABLE, NOT_APPLICABLE, ERROR)

# Per-capability minimum history/periods (only checked when rows >= 10).
_CAPABILITY_PERIOD_MIN = {
    "forecast": FORECAST_MIN_POINTS,
    "trend": TREND_MIN_PERIODS,
    "anomaly": ANOMALY_MIN_PERIODS,
    "correlation": CORRELATION_MIN_SHARED_PERIODS,
    "distribution": DISTRIBUTION_MIN_ROWS,
    "outlier": OUTLIER_MIN_ROWS,
    "scenario": SCENARIO_MIN_OBS,
    "hypothesis": HYPOTHESIS_MIN_OBS,
    "prescription": GLOBAL_MIN_ROWS,
    "insights": GLOBAL_MIN_ROWS,
    "decisions": GLOBAL_MIN_ROWS,
    "relationships": CORRELATION_MIN_SHARED_PERIODS,
    "narrative": GLOBAL_MIN_ROWS,
    "monitoring": GLOBAL_MIN_ROWS,
    "recommendations": GLOBAL_MIN_ROWS,
}


def coerce_row_count(value: Any) -> Optional[int]:
    """Best-effort int coercion; None when unknown (missing evidence)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n >= 0 else 0


def _global_reason(row_count: int) -> str:
    if row_count <= 0:
        return "No usable records were found."
    return (
        "ARIA needs at least 10 usable records before it can reliably "
        "identify business patterns. "
        f"Only {row_count} usable record{'s' if row_count != 1 else ''} "
        "were found, so no pattern, forecast, recommendation or scenario "
        "can be supported yet."
    )


def _capability_reason(capability: str, have: Any, need: Any) -> str:
    cap = (capability or "").lower()
    if cap == "forecast":
        return (
            f"Only {have} usable time periods were found — a reliable "
            f"forecast needs at least {need}. Connect a longer history or "
            "pick a number with more recorded periods."
        )
    if cap == "trend":
        return (
            f"Only {have} usable periods were found — a trend needs at least "
            f"{need} periods to separate signal from noise."
        )
    if cap == "anomaly":
        return (
            f"Only {have} usable periods were found — anomaly detection needs "
            f"at least {need} periods to know what 'usual' looks like."
        )
    if cap in ("correlation", "relationships"):
        return (
            f"Only {have} shared periods were found — comparing two numbers "
            f"needs at least {need} shared periods."
        )
    if cap == "distribution":
        return (
            f"Only {have} usable values were found — describing a distribution "
            f"needs at least {need} values."
        )
    if cap == "outlier":
        return (
            f"Only {have} usable values were found — spotting unusual records "
            f"needs at least {need} values."
        )
    if cap in ("scenario", "monitoring"):
        return (
            "ARIA doesn't have enough historical evidence to estimate the "
            "impact yet — a measured baseline from at least "
            f"{need} usable observations is needed first."
        )
    if cap == "hypothesis":
        return (
            f"Only {have} usable observations were found — testing this idea "
            f"needs at least {need}."
        )
    return (
        f"Only {have} usable records were found for '{capability}' — at "
        f"least {need} are needed."
    )


def assess_evidence(
    row_count: Any,
    capability: Optional[str] = None,
    *,
    history_points: Any = None,
    shared_periods: Any = None,
    has_baseline: Optional[bool] = None,
    dataset_name: Optional[str] = None,
    columns: Optional[List[str]] = None,
    quality_notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Canonical gate. Pure + deterministic, never raises.

    Returns {status, reason, row_count, usable_records, min_required,
    capability, needed, ...}. `status` is one of the canonical five;
    `reason` is always a business sentence.
    """
    n = coerce_row_count(row_count)
    cap = (capability or "insights").strip().lower() or "insights"
    need_global = GLOBAL_MIN_ROWS
    out: Dict[str, Any] = {
        "capability": cap,
        "row_count": n,
        "usable_records": n,
        "min_required": need_global,
        "needed": None,
    }
    if dataset_name is not None:
        out["dataset"] = dataset_name
    # Unknown count: cannot support a conclusion, but distinguish from
    # measured-insufficient (error, not silent pass).
    if n is None:
        out.update({
            "status": ERROR,
            "reason": ("ARIA couldn't count the usable records for this "
                       "analysis, so no pattern can be reported yet."),
        })
        return out
    # GLOBAL RULE: rows < 10 -> INSUFFICIENT_EVIDENCE for everything.
    if n < need_global:
        out.update({
            "status": INSUFFICIENT,
            "reason": _global_reason(n),
            "needed": need_global - n,
        })
        return out
    # Rows >= 10: per-capability requirements (REUSED thresholds only).
    period_need = _CAPABILITY_PERIOD_MIN.get(cap)
    have_periods = shared_periods if cap in ("correlation", "relationships") \
        else history_points
    if period_need is not None and have_periods is not None:
        try:
            have = int(have_periods)
        except (TypeError, ValueError):
            have = None
        if have is not None and have < period_need:
            out.update({
                "status": INSUFFICIENT,
                "reason": _capability_reason(cap, have, period_need),
                "needed": period_need - have,
                "history_points": have,
                "min_required": period_need,
            })
            return out
    if cap in ("scenario", "monitoring") and has_baseline is False:
        out.update({
            "status": INSUFFICIENT,
            "reason": _capability_reason(cap, n, period_need or need_global),
            "needed": None,
        })
        return out
    out.update({
        "status": SUPPORTED,
        "reason": (f"{n} usable records meet the evidence bar for "
                   f"'{cap}'."),
        "needed": 0,
    })
    return out


def capability_state(capability: str, row_count: Any, **kwargs: Any) -> Dict[str, Any]:
    """Thin alias so call sites read as capability checks."""
    return assess_evidence(row_count, capability, **kwargs)


def dataset_facts(
    dataset_name: Optional[str] = None,
    usable_records: Any = None,
    columns: Optional[List[str]] = None,
    quality_notes: Optional[List[str]] = None,
    needed: Any = None,
) -> Dict[str, Any]:
    """Optional dataset facts the honest insufficient state may carry.

    Name, usable count, columns, quality observations, what's needed —
    never a pattern or estimate.
    """
    n = coerce_row_count(usable_records)
    return {
        "dataset": dataset_name,
        "usable_records": n,
        "columns": list(columns or [])[:20],
        "quality_notes": list(quality_notes or [])[:6],
        "needed": needed,
        "what_is_needed": (
            f"At least {GLOBAL_MIN_ROWS} usable records"
            if (n or 0) < GLOBAL_MIN_ROWS else None
        ),
    }


def llm_allowed(row_count: Any, capability: Optional[str] = None, **kwargs: Any) -> bool:
    """True only when the gate passes — LLM polish/narrative may run."""
    try:
        return assess_evidence(row_count, capability, **kwargs).get("status") == SUPPORTED
    except Exception:
        return False


def insufficient_narrative(
    row_count: Any,
    capability: Optional[str] = None,
    *,
    dataset_name: Optional[str] = None,
    columns: Optional[List[str]] = None,
    quality_notes: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Honest deterministic fallback for LLM-gated paths (no LLM content).

    Returns {text, evidence_state, dataset_facts}. Callers present `text`
    as-is; the LLM is never consulted for insufficient evidence.
    """
    state = assess_evidence(row_count, capability, dataset_name=dataset_name,
                            columns=columns, quality_notes=quality_notes, **kwargs)
    facts = dataset_facts(dataset_name, coerce_row_count(row_count),
                          columns, quality_notes, state.get("needed"))
    bits = [state.get("reason") or _global_reason(coerce_row_count(row_count) or 0)]
    if dataset_name:
        bits.append(f"Dataset: {dataset_name}.")
    if columns:
        bits.append(f"Columns seen: {', '.join(list(columns)[:8])}.")
    for note in list(quality_notes or [])[:3]:
        bits.append(str(note)[:200])
    return {"text": " ".join(bits), "evidence_state": state, "dataset_facts": facts}


# ---------------------------------------------------------------------------
# Mapping helpers: existing codes -> canonical states (additive, keep keys).
# ---------------------------------------------------------------------------
def quality_gate_status(code: Optional[str]) -> str:
    """Map prescription quality_gate codes onto canonical states."""
    c = str(code or "").upper()
    if c == "OK":
        return SUPPORTED
    if c in ("INSUFFICIENT_DATA", "WEAK_SIGNAL"):
        return INSUFFICIENT
    if c == "NOT_ACTIONABLE":
        return NOT_APPLICABLE
    return ERROR


def availability_to_status(availability: Optional[str]) -> str:
    """Map decision availability values onto canonical states."""
    a = str(availability or "").lower()
    if a == "available":
        return SUPPORTED
    if a == "insufficient_data":
        return INSUFFICIENT
    if a in ("not_supported", "unsupported"):
        return NOT_APPLICABLE
    return UNAVAILABLE


def status_to_availability(status: Optional[str]) -> str:
    """Back-map canonical states onto the stored availability contract."""
    s = str(status or "").lower()
    if s == SUPPORTED:
        return "available"
    if s == INSUFFICIENT:
        return "insufficient_data"
    if s in (NOT_APPLICABLE, UNAVAILABLE, ERROR):
        return "not_supported"
    return "insufficient_data"
