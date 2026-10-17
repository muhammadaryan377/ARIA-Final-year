"""Insights business logic (services layer — canonical home).

Moved verbatim from api/insights_routes.py: pure detectors/formatters/chart
spec builders. The router re-exports every name imported from here, so
existing imports keep working. No FastAPI imports in this module.
"""
from __future__ import annotations

import logging
import math
import sqlite3
import statistics
import time
from datetime import datetime

from core.errors import DB_ERRORS, PURE_ERRORS, get_request_id, log_and_sanitize

log = logging.getLogger("aria.insights.service")

__all__ = ["CHART_SPEC_VOCAB", "_MAX_SPEC_ROWS", "_MAX_RAW_VALUES", "_safe_float", "_fmt", "_fmt_currency", "_fmt_pct", "_iqr_bands", "_sanitize_spec_rows", "_normalize_chart_spec", "_confidence_level", "_evidence_quality", "_capability_flags", "_histogram_bins", "_build_takeaways", "_enrich_insight", "_normalize_legacy_insight", "_translate_sql_for_sqlite", "build_recommendation_detail", "build_explains"]


def _safe_float(v):
    try:
        if v is None:
            return None
        return float(v)
    except (ValueError, TypeError) as exc:
        log.debug("safe float skipped: %s", exc)
        return None


def _fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        if abs(v) >= 1_000_000:
            return f"{v / 1_000_000:.1f}M"
        if abs(v) >= 1_000:
            return f"{v / 1_000:.1f}K"
        if abs(v) < 0.01 and v != 0:
            return f"{v:.4f}"
        return f"{v:,.2f}"
    return str(v)


def _fmt_currency(v):
    if v is None:
        return "N/A"
    if abs(v) >= 1_000_000:
        return f"${v/1_000_000:,.1f}M"
    if abs(v) >= 1_000:
        return f"${v/1_000:,.0f}K"
    return f"${v:,.0f}"


def _fmt_pct(v):
    if v is None:
        return "N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.1f}%"


def _iqr_bands(values):
    """Generic IQR fences with linear-interpolated quartiles.

    Returns (q1, q3, iqr, lower, upper, severe_lower, severe_upper) with
    1.5xIQR regular fences + 3.0xIQR severe tiers, or None when the series
    cannot support a band (too few finite values, non-finite spread, or
    iqr <= 0 e.g. flat series). Callers skip silently on None — no fake band.
    Pure math, no dataset assumptions.
    """
    try:
        nums = [float(v) for v in (values or []) if v is not None]
    except (ValueError, TypeError) as exc:
        log.debug("iqr band input skipped: %s", exc)
        return None
    nums = [v for v in nums if math.isfinite(v)]
    n = len(nums)
    if n < 4:
        return None
    s = sorted(nums)

    def _pct(p):
        pos = (n - 1) * p
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return s[lo]
        frac = pos - lo
        return s[lo] + (s[hi] - s[lo]) * frac

    try:
        q1 = float(_pct(0.25))
        q3 = float(_pct(0.75))
    except (ValueError, TypeError, IndexError) as exc:
        log.debug("iqr quartile skipped: %s", exc)
        return None
    if not (math.isfinite(q1) and math.isfinite(q3)):
        return None
    iqr = q3 - q1
    if not math.isfinite(iqr) or iqr <= 0:
        return None
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    severe_lower = q1 - 3.0 * iqr
    severe_upper = q3 + 3.0 * iqr
    if not all(math.isfinite(v) for v in (lower, upper, severe_lower, severe_upper)):
        return None
    return (q1, q3, iqr, lower, upper, severe_lower, severe_upper)


# ── n.txt-9: controlled chart vocabulary ─────────────────────────────────
# Backend emits ONLY {type,x,y,series,title,interactive,data}. The frontend
# renders this spec and never executes LLM-supplied frontend code.
# Provenance stays in this backend payload (admin/debug) — user-facing pages
# decide what to display; AI Analytics hides it.
CHART_SPEC_VOCAB = ("line", "bar", "area", "scatter", "pie", "donut", "histogram", "heatmap")
_MAX_SPEC_ROWS = 500
# Raw-value pulls for distribution statistics (IQR/outliers/CV/skew) are
# bounded samples, never full-column dumps: the detectors aggregate in Python
# (binned histograms ≤12 bins, scalar statistics) and the honest sample size
# travels with each insight. Full-dataset aggregates stay SQL-side wherever
# the detector shape allows (monthly sums, top-N, grouped averages).
_MAX_RAW_VALUES = 2000


def _sanitize_spec_rows(rows):
    """Validate chart data before it leaves the backend: dicts only, JSON-safe
    scalars, capped length. Never raises — returns [] when unusable so the
    frontend can show an insufficient-data state."""
    if not isinstance(rows, (list, tuple)):
        return []
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        clean = {}
        for k, v in r.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = None
            elif isinstance(v, (str, int, float, bool)) or v is None:
                clean[k] = v
            else:
                clean[k] = str(v)
        out.append(clean)
        if len(out) >= _MAX_SPEC_ROWS:
            break
    return out


def _normalize_chart_spec(viz_type, viz_data, title=""):
    """Map internal detector viz types to the controlled vocabulary + infer
    x/y/series. Supported: line/bar/area/scatter/pie/donut/histogram/heatmap.
    Unknown types fall back to bar/line when the data shape fits, else the
    caller treats the spec as unsupported (frontend shows a graceful note)."""
    raw = (viz_type or "").lower()
    data = _sanitize_spec_rows(viz_data)
    vt = raw if raw in CHART_SPEC_VOCAB else ""
    if raw in ("leaderboard",):
        vt = "bar"
    elif raw in ("pie_share", "share"):
        vt = "pie"
    elif raw in ("box", "box_plot", "violin"):
        # No box-plot renderer in the controlled vocabulary: distribution
        # detail is preserved as a histogram (supported vocab).
        vt = "histogram"
    if not data:
        return {"type": vt or "bar", "x": None, "y": None, "series": [],
                "title": title or "", "interactive": True, "data": [],
                "unsupported": False, "insufficient": True}
    keys = list(data[0].keys())
    x = y = None
    series: list = []
    if vt in ("bar", "pie", "donut", "histogram") or not vt:
        if "name" in keys and "value" in keys:
            x, y, vt = "name", "value", vt or "bar"
        elif "period" in keys and "value" in keys:
            # Time-ordered bars render better as area/line unless caller said bar.
            x, y = "period", "value"
            vt = vt or "area"
        elif len(keys) >= 2:
            x, y = keys[0], keys[1]
            vt = vt or "bar"
        else:
            return {"type": "bar", "x": None, "y": None, "series": [],
                    "title": title or "", "interactive": True, "data": data,
                    "unsupported": False, "insufficient": True}
    elif vt in ("line", "area"):
        if "period" in keys and "value" in keys:
            x, y = "period", "value"
        elif len(keys) >= 2:
            x, y = keys[0], keys[1]
        # multi-series: extra numeric keys beyond x/y become toggleable
        # series so the frontend can render expected/band overlays. Bool
        # flags (e.g. anomaly) are excluded; any row carrying a real number
        # promotes the key (first-row-only checks would drop bands).
        series = [k for k in keys if k not in (x, y)
                and any(isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)
                        for r in data)]
        series = series[:8]
    elif vt == "scatter":
        if "table1" in keys and "table2" in keys:
            x, y = "table1", "table2"
        elif "x" in keys and "y" in keys:
            x, y = "x", "y"
        elif len(keys) >= 2:
            x, y = keys[0], keys[1]
    elif vt == "heatmap":
        if {"x", "y", "value"} <= set(keys):
            x, y, series = "x", "y", []
        elif len(keys) >= 3:
            x, y = keys[0], keys[1]
    if vt not in CHART_SPEC_VOCAB:
        return {"type": "bar", "x": x, "y": y, "series": series,
                "title": title or "", "interactive": True, "data": data,
                "unsupported": True, "insufficient": False}
    insufficient = (len(data) < 2 and vt in ("line", "area", "scatter")) or (len(data) < 1)
    return {"type": vt, "x": x, "y": y, "series": series,
            "title": title or "", "interactive": True, "data": data,
            "unsupported": False, "insufficient": insufficient}


# ── Generic per-insight contract (§§23-24,34) ─────────────────────────────
# All helpers are dataset-agnostic: no branching on table/column/metric
# names, no invented numbers. Everything derives from fields already
# present on the insight (or honest defaults when a detector provides no
# calibrated score).


def _confidence_level(conf) -> str:
    """Strong>=75, Moderate>=50, else Limited (from existing int only)."""
    try:
        c = float(conf)
    except (TypeError, ValueError):
        return "Limited"
    if c >= 75:
        return "Strong"
    if c >= 50:
        return "Moderate"
    return "Limited"


def _evidence_quality(sample_size) -> str:
    """Generic count thresholds only: Strong>=100, Moderate>=30, else Limited."""
    try:
        n = int(sample_size) if sample_size is not None else 0
    except (TypeError, ValueError):
        return "Limited"
    if n >= 100:
        return "Strong"
    if n >= 30:
        return "Moderate"
    return "Limited"


def _capability_flags(sample_size) -> tuple:
    """scenario_supported / monitoring_supported via core/evidence.py + n>=10."""
    try:
        n = int(sample_size) if sample_size is not None else 0
    except (TypeError, ValueError):
        n = 0
    scen = mon = False
    try:
        from core.evidence import assess_evidence as _assess
        scen = _assess(n, "scenario").get("status") == "supported"
        mon = _assess(n, "monitoring").get("status") == "supported"
    except (ValueError, KeyError, TypeError, AttributeError, ImportError) as exc:
        log.debug("capability assess skipped: %s", exc)
        scen = n >= 10
        mon = n >= 10
    return bool(scen), bool(mon)


def _histogram_bins(vals, nbins: int = 12):
    """Generic value histogram {name, value} bins from raw numbers only."""
    try:
        nums = [float(v) for v in (vals or []) if v is not None]
    except (ValueError, TypeError) as exc:
        log.debug("histogram input skipped: %s", exc)
        return []
    nums = [v for v in nums if math.isfinite(v)]
    if not nums:
        return []
    lo, hi = min(nums), max(nums)
    span = (hi - lo) or 1.0
    counts = [0] * nbins
    for v in nums:
        b = min(nbins - 1, max(0, int((v - lo) / span * nbins)))
        counts[b] += 1
    out = []
    for i in range(nbins):
        if not counts[i]:
            continue
        b_lo = lo + span * i / nbins
        b_hi = lo + span * (i + 1) / nbins
        out.append({"name": f"{_fmt(b_lo)}-{_fmt(b_hi)}", "value": counts[i]})
    return out


def _human_label(name) -> str:
    """Schema-agnostic human label for prose (never raw snake_case)."""
    try:
        from core.semantic import humanize as _hz
        return _hz(str(name or ""))
    except Exception:
        import re as _re
        s = _re.sub(r"[_-]+", " ", str(name or "")).strip()
        s = _re.sub(r"\s+", " ", s)
        return s[:1].upper() + s[1:] if s else ""


def _human_scale(v):
    """Human-scale numbers for prose: 42K / 1.8M, never 42000.0."""
    try:
        n = float(v)
    except (TypeError, ValueError):
        return str(v)
    a = abs(n)
    if a >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if a >= 1_000:
        return f"{n / 1_000:.1f}K"
    if a >= 100:
        return f"{int(round(n)):,}"
    if a >= 1:
        return f"{round(n, 1)}"
    return f"{round(n, 2)}"


def _scrub_line(s) -> str:
    """Remove raw schema traces from a prose line (identifiers/SQL/jargon)."""
    import re as _re
    t = str(s or "")
    t = _re.sub(r"[A-Za-z_][A-Za-z0-9_]*\.([A-Za-z_][A-Za-z0-9_]*)",
                lambda m: _human_label(m.group(1)), t)
    t = _re.sub(r"[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+",
                lambda m: _human_label(m.group(0)), t)
    t = _re.sub(r"\b(SELECT|FROM|WHERE|GROUP BY|ORDER BY|JOIN|LIMIT|SQL)\b",
                "the data", t)
    # Internal engine terms must never reach the user.
    t = _re.sub(r"\bp-?value\b", "reliability", t, flags=_re.IGNORECASE)
    t = _re.sub(r"\bz-?scores?\b", "unusual pattern", t, flags=_re.IGNORECASE)
    t = _re.sub(r"\bR\^2\b|\bR-squared\b", "pattern strength", t, flags=_re.IGNORECASE)
    t = _re.sub(r"\bOLS\b", "trend analysis", t)
    t = _re.sub(r"\bFK\b", "linked records", t)
    t = _re.sub(r"\bIQR\b", "usual range", t)
    t = _re.sub(r"\bCV\s*=\s*[\d.]+\b", "wide spread", t, flags=_re.IGNORECASE)
    t = _re.sub(r"\bSkew\s*=\s*[-\d.]+\b", "lopsided spread", t, flags=_re.IGNORECASE)
    return _re.sub(r"\s+", " ", t).strip()


def _top_of_viz(viz_data):
    """Top named entry from chart rows (real executed values only)."""
    try:
        rows = [r for r in (viz_data or []) if isinstance(r, dict)]
        cands = []
        for r in rows:
            name = r.get("name", r.get("period", r.get("x")))
            val = r.get("value", r.get("y"))
            try:
                f = float(val)
            except (TypeError, ValueError):
                continue
            if name is not None and math.isfinite(f):
                cands.append((str(name)[:40], f))
        if not cands:
            return None, None
        cands.sort(key=lambda t: t[1], reverse=True)
        return cands[0]
    except Exception:
        return None, None


def _build_takeaways(evidence, affected_period, observations, support_metrics):
    """2-4 conversational takeaways built ONLY from present numbers.

    Human-scale numbers (42K), up/down direction words, short sentences.
    Skips missing values — never invents. Falls back to the detector's own
    observation strings (scrubbed) when fewer than 2 numeric lines exist.
    Capped at 4.
    """
    takeaways: list = []
    ev = evidence or {}
    cp = ev.get("changePercent")
    baseline = ev.get("baseline")
    current = ev.get("currentValue")
    try:
        if cp is not None:
            cp_f = float(cp)
            if math.isfinite(cp_f):
                direction = "up" if cp_f > 0 else ("down" if cp_f < 0 else "flat")
                takeaways.append(_scrub_line(f"{direction.capitalize()} {abs(cp_f):.0f}% vs the usual level."))
    except (TypeError, ValueError):
        pass
    try:
        if baseline is not None and current is not None:
            b_f, c_f = float(baseline), float(current)
            if math.isfinite(b_f) and math.isfinite(c_f):
                if c_f > b_f:
                    takeaways.append(_scrub_line(f"Up from {_human_scale(b_f)} to {_human_scale(c_f)}."))
                elif c_f < b_f:
                    takeaways.append(_scrub_line(f"Down from {_human_scale(b_f)} to {_human_scale(c_f)}."))
                else:
                    takeaways.append(_scrub_line(f"Holding steady near {_human_scale(c_f)}."))
        elif baseline is not None:
            b_f = float(baseline)
            if math.isfinite(b_f):
                takeaways.append(_scrub_line(f"Usually around {_human_scale(b_f)}."))
        elif current is not None:
            c_f = float(current)
            if math.isfinite(c_f):
                takeaways.append(_scrub_line(f"Now at {_human_scale(c_f)}."))
    except (TypeError, ValueError):
        pass
    if affected_period and len(takeaways) < 3:
        takeaways.append(_scrub_line(f"Seen in {affected_period}."))
    # Fill to at least 2 with existing evidence strings (scrubbed, no new numbers).
    if len(takeaways) < 2:
        for s in list(observations or []) + list(support_metrics or []):
            if isinstance(s, str) and s.strip() and s not in takeaways:
                clean = _scrub_line(s)
                if clean and clean not in takeaways:
                    takeaways.append(clean)
                if len(takeaways) >= 4:
                    break
    return takeaways[:4]


def build_recommendation_detail(ins: dict) -> dict:
    """Evidence-backed recommendation: title + 1-2 steps + outcome in words.

    Derives ONLY from the insight's own computed evidence (change percent,
    baseline/current values, top chart entry, period) or returns an honest
    'not enough evidence' card. Never generic filler, never internal terms,
    fully schema-agnostic (humanized names only).
    """
    ins = ins if isinstance(ins, dict) else {}
    itype = str(ins.get("type", "") or "").lower()
    ev = ins.get("evidence") if isinstance(ins.get("evidence"), dict) else {}
    metric_raw = (ins.get("metric") or "").split(" (")[0].strip()
    metric = _human_label(metric_raw) or "this measure"
    metric_low = metric[:1].lower() + metric[1:] if metric else "this measure"
    period = str(ins.get("affectedPeriod") or "").strip()
    viz = (ins.get("visualization") or {}) if isinstance(ins.get("visualization"), dict) else {}
    top_name, top_val = _top_of_viz(viz.get("data"))

    def _num(v):
        try:
            f = float(v)
            return f if math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    cp = _num(ev.get("changePercent"))
    baseline = _num(ev.get("baseline"))
    current = _num(ev.get("currentValue"))
    if current is None:
        current = _num(ins.get("metricValue"))
    ss = ev.get("sampleSize")
    try:
        n = int(ss) if ss is not None else 0
    except (TypeError, ValueError):
        n = 0

    has_numbers = (
        cp is not None
        or (baseline is not None and current is not None)
        or (current is not None and (period or top_name))
        or top_val is not None
    )
    if not has_numbers and n < 10:
        return {
            "title": "Not enough evidence for a next step",
            "steps": [
                "Gather a little more history first — a few more periods or segments to compare.",
                f"Keep watching {metric_low} so the next change has something to measure against.",
            ],
            "outcome": "Once there is more to go on, ARIA will suggest a concrete next step.",
            "honest": True,
        }

    direction_up = (cp is not None and cp > 0) or (
        cp is None and baseline is not None and current is not None and current > baseline
    )
    direction_down = (cp is not None and cp < 0) or (
        cp is None and baseline is not None and current is not None and current < baseline
    )
    pct_s = f"{abs(cp):.0f}%" if cp is not None else ""
    cur_s = _human_scale(current) if current is not None else ""
    base_s = _human_scale(baseline) if baseline is not None else ""
    where = f" in {period}" if period else ""
    move_bits = ""
    if pct_s and cur_s and base_s:
        move_bits = f" ({pct_s} move from {base_s} to {cur_s})"
    elif pct_s and cur_s:
        move_bits = f" ({pct_s}, now at {cur_s})"
    elif cur_s and base_s:
        move_bits = f" (from {base_s} to {cur_s})"
    elif cur_s:
        move_bits = f" (now at {cur_s})"

    if "opportunit" in itype or "concentrat" in itype or "pareto" in itype or "segment" in itype:
        who = f"'{top_name}'" if top_name else f"your top group{where}"
        return {
            "title": f"Lean into {top_name or 'your top group'} — it carries the weight",
            "steps": [
                f"Give {who} extra attention this week — check stock, staffing and follow-up{move_bits}.",
                "Find what makes it work and repeat it in one weaker group.",
            ],
            "outcome": f"If the pattern holds, {metric_low} stays concentrated where it performs best.",
            "honest": False,
        }
    if "anomal" in itype or "outlier" in itype:
        return {
            "title": f"Check the unusual {metric_low}{where}",
            "steps": [
                f"Confirm the {cur_s or 'unusual'} reading{where} is real data, not a recording blip.",
                "Ask what changed around then — one answer usually explains it.",
            ],
            "outcome": "You will know whether to act on it or safely set it aside.",
            "honest": False,
        }
    if "correl" in itype or "relation" in itype:
        return {
            "title": "Test the link before acting on it",
            "steps": [
                f"Pick one small change affecting {metric_low} and compare matched periods.",
                "Only scale up what the comparison confirms.",
            ],
            "outcome": "You avoid spending on a link that only looked real in the chart.",
            "honest": False,
        }
    if "distribut" in itype:
        return {
            "title": f"Look at what drives the spread in {metric_low}",
            "steps": [
                "Pull the highest and lowest cases side by side and note what differs.",
                "Fix the one difference you can control first.",
            ],
            "outcome": "Less spread means a more predictable result next time.",
            "honest": False,
        }
    if direction_up or "growth" in itype or ("trend" in itype and not direction_down):
        return {
            "title": f"Keep the momentum in {metric_low}",
            "steps": [
                f"Put a bit more behind what worked{where}{move_bits} — do not change the mix yet.",
                "Write down what changed so you can repeat it next period.",
            ],
            "outcome": f"If the lift holds, {metric_low} stays up near {cur_s or 'its new level'}.",
            "honest": False,
        }
    if direction_down or "decline" in itype or "risk" in itype:
        return {
            "title": f"Steady the dip in {metric_low}",
            "steps": [
                f"Find what slipped{where}{move_bits} — talk to the team closest to it this week.",
                "Protect your strongest group first, then win back one weak spot.",
            ],
            "outcome": "You stop the slide early, before it becomes the new normal.",
            "honest": False,
        }
    # Fallback with real numbers when the type is generic.
    if has_numbers:
        return {
            "title": f"Follow up on {metric_low}{where}",
            "steps": [
                f"Review what moved{move_bits} and note the likely cause.",
                "Decide one small action to take before the next period.",
            ],
            "outcome": "A quick review now makes the next change easier to read.",
            "honest": False,
        }
    return {
        "title": "Not enough evidence for a next step",
        "steps": [
            "Gather a little more history first — a few more periods or segments to compare.",
            f"Keep watching {metric_low} so the next change has something to measure against.",
        ],
        "outcome": "Once there is more to go on, ARIA will suggest a concrete next step.",
        "honest": True,
    }


def build_explains(ins: dict) -> str:
    """Thorough, multi-paragraph explanation of an insight.

    Dynamically composed from the insight's own title, description, evidence
    numbers, observations, confidence, and type. Each paragraph serves a
    purpose: what happened, why it matters, what the evidence shows, and
    what to do next. No hardcoded business text — everything is derived
    from the data at hand. Returns '' when there is nothing meaningful
    to say (caller hides the section).
    """
    try:
        ins = ins if isinstance(ins, dict) else {}
        title = str(ins.get("title") or "").strip()
        desc = str(ins.get("description") or ins.get("summary") or "").strip()
        ev = ins.get("evidence") if isinstance(ins.get("evidence"), dict) else {}
        metric = _human_label(str(ins.get("metric") or "").split(" (")[0]) or "this measure"
        metric_low = metric[:1].lower() + metric[1:] if metric else "this measure"
        period = str(ins.get("affectedPeriod") or "").strip()
        itype = str(ins.get("type") or "").lower()
        conf = ins.get("confidence")
        try:
            conf = int(conf) if conf is not None else None
        except (TypeError, ValueError):
            conf = None

        def _num(v):
            try:
                f = float(v)
                return f if math.isfinite(f) else None
            except (TypeError, ValueError):
                return None

        cp = _num(ev.get("changePercent"))
        current = _num(ev.get("currentValue"))
        baseline = _num(ev.get("baseline"))
        ss = ev.get("sampleSize") or ins.get("sample_size")
        try:
            ss = int(ss) if ss is not None else None
        except (TypeError, ValueError):
            ss = None
        obs = ev.get("observations") or []
        sup = ev.get("supportingMetrics") or []
        conf_label = _confidence_level(conf) if conf is not None else ""
        eq_label = _evidence_quality(ss) if ss is not None else ""

        if not title and not desc:
            return ""

        paragraphs = []

        # --- Paragraph 1: The Finding ---
        if cp is not None and cp > 0:
            move = f"{metric} is up {abs(cp):.1f}%"
        elif cp is not None and cp < 0:
            move = f"{metric} is down {abs(cp):.1f}%"
        elif baseline is not None and current is not None:
            move = f"{metric} moved from {_human_scale(baseline)} to {_human_scale(current)}"
        elif current is not None:
            move = f"{metric} is at {_human_scale(current)}"
        else:
            move = _scrub_line(title) if title else ""

        if move:
            where = f" over {period}" if period else ""
            finding = f"{move}{where}."
            if title:
                title_clean = _scrub_line(title)
                if title_clean and title_clean.lower() != move.lower().rstrip("."):
                    finding = f"{move}{where}. {title_clean}."
            paragraphs.append(finding)
        elif title:
            paragraphs.append(_scrub_line(title))
        elif desc:
            paragraphs.append(_scrub_line(desc))

        # --- Paragraph 2: Why It Matters ---
        why_parts = []
        if cp is not None and baseline is not None and current is not None:
            direction = "growth" if cp > 0 else "decline"
            if abs(cp) >= 20:
                magnitude = "significant"
            elif abs(cp) >= 10:
                magnitude = "notable"
            elif abs(cp) >= 5:
                magnitude = "moderate"
            else:
                magnitude = "slight"
            why_parts.append(
                f"This {magnitude} {direction} ({_fmt_pct(cp)}) represents a shift "
                f"from {_human_scale(baseline)} to {_human_scale(current)}."
            )
            if abs(cp) >= 15 and ss and ss >= 30:
                why_parts.append(
                    f"With {ss:,} data points behind this movement, the pattern is "
                    f"unlikely to be random noise."
                )
            elif ss is not None and ss < 20:
                why_parts.append(
                    f"However, this is based on only {ss} observations, so the "
                    f"pattern may not hold at scale."
                )
        elif current is not None:
            why_parts.append(
                f"The current level of {_human_scale(current)} for {metric_low} "
                f"provides a baseline for understanding where things stand."
            )

        if "opportunit" in itype:
            if cp is not None and cp > 0:
                why_parts.append(
                    f"Because {metric_low} is trending upward, there may be a window "
                    f"to build on this momentum before the trend plateaus."
                )
            else:
                why_parts.append(
                    f"This represents a potential area to invest in or explore further, "
                    f"as the data suggests room for positive movement."
                )
        elif "risk" in itype or "decline" in itype:
            if cp is not None and cp < 0:
                why_parts.append(
                    f"The downward movement in {metric_low} signals something worth "
                    f"monitoring — left unaddressed, the trend could compound."
                )
            else:
                why_parts.append(
                    f"This pattern carries risk if it persists. Understanding the "
                    f"underlying drivers early gives you room to intervene."
                )
        elif "anomal" in itype:
            why_parts.append(
                f"An unusual pattern like this often points to an external factor "
                f"or a one-time event worth investigating."
            )

        if why_parts:
            paragraphs.append(" ".join(why_parts))

        # --- Paragraph 3: What the Evidence Shows ---
        evidence_parts = []
        if obs:
            top_obs = [_scrub_line(str(o)) for o in obs[:3] if str(o).strip()]
            if top_obs:
                evidence_parts.append(
                    f"The data shows: {'; '.join(top_obs)}."
                )
        if sup:
            top_sup = [_scrub_line(str(s)) for s in sup[:2] if str(s).strip()]
            if top_sup:
                evidence_parts.append(
                    f"Supporting this: {'; '.join(top_sup)}."
                )
        if conf_label and eq_label:
            evidence_parts.append(
                f"Confidence is {conf_label.lower()} with {eq_label.lower()} evidence."
            )
        elif conf_label:
            evidence_parts.append(f"Confidence: {conf_label.lower()}.")

        if evidence_parts:
            paragraphs.append(" ".join(evidence_parts))

        # --- Paragraph 4: What To Do ---
        action_parts = []
        if "opportunit" in itype:
            action_parts.append(
                f"Consider exploring what is driving this upward movement in "
                f"{metric_low} and whether those drivers can be amplified."
            )
            if cp is not None and cp > 10:
                action_parts.append(
                    f"A {abs(cp):.0f}% gain is material — even small investments to "
                    f"sustain it could compound into meaningful returns."
                )
        elif "risk" in itype or "decline" in itype:
            action_parts.append(
                f"It would be prudent to dig into what is causing {metric_low} to "
                f"weaken and whether the trend is accelerating or stabilizing."
            )
            if cp is not None and cp < -10:
                action_parts.append(
                    f"A {abs(cp):.0f}% decline is significant enough to warrant "
                    f"attention before the pattern becomes harder to reverse."
                )
        elif "anomal" in itype:
            action_parts.append(
                f"Investigate whether this anomaly correlates with a specific "
                f"event, segment, or time period to determine if action is needed."
            )
        else:
            action_parts.append(
                f"Keep monitoring {metric_low} to see whether this pattern "
                f"holds, strengthens, or fades over the next few periods."
            )

        if action_parts:
            paragraphs.append(" ".join(action_parts))

        if not paragraphs:
            return ""

        text = " ".join(paragraphs)
        text = _scrub_line(text)
        return text[:2000] if len(text) > 2000 else text
    except Exception:
        return ""


def _enrich_insight(ins: dict) -> dict:
    """Add generic contract fields to an already-shaped insight (idempotent).

    Adds: category, explanation, summary/description parity, confidence_level,
    evidence_quality, takeaways, recommendation, scenario_supported,
    monitoring_supported, associated_factors (only for real correlation
    evidence, else omitted). Ensures `visualization` exists via
    _normalize_chart_spec when only viz_type/viz_data are present.
    """
    if not isinstance(ins, dict):
        return ins
    itype = str(ins.get("type", "") or "")
    ins["category"] = itype.lower()
    desc = ins.get("description") or ins.get("summary") or ""
    if not ins.get("description") and ins.get("summary"):
        ins["description"] = ins.get("summary")
    if not ins.get("summary") and desc:
        ins["summary"] = desc
    ins["explanation"] = desc
    conf = ins.get("confidence")
    if not isinstance(conf, (int, float)) or not math.isfinite(float(conf)):
        conf = 50
        ins["confidence"] = conf
    else:
        conf = int(conf)
    ins["confidence_level"] = _confidence_level(conf)
    ev = ins.get("evidence") if isinstance(ins.get("evidence"), dict) else {}
    ss = ev.get("sampleSize")
    if ss is None:
        ss = ins.get("sample_size", ins.get("sampleSize"))
    ins["evidence_quality"] = _evidence_quality(ss)
    obs = ev.get("observations") or ins.get("observations") or []
    sup = ev.get("supportingMetrics") or ins.get("support_metrics") or []
    ins["takeaways"] = _build_takeaways(ev, ins.get("affectedPeriod"), obs, sup)
    # Conversational explainer (new prose field for the Insights page).
    try:
        _exp = build_explains(ins)
        if _exp:
            ins["explains"] = _exp
    except Exception:
        pass
    # Recommendation: evidence-backed detail (title + steps + outcome) is
    # canonical. The legacy string field carries a one-line rendering for
    # backward compatibility — never generic filler.
    try:
        _det = build_recommendation_detail(ins)
        if isinstance(_det, dict) and _det.get("title"):
            ins["recommendation_detail"] = _det
            _steps = _det.get("steps") or []
            _one = _det["title"].strip()
            if _steps:
                _one += " — " + str(_steps[0]).strip()
            if _det.get("outcome"):
                _one += " Expected: " + str(_det["outcome"]).strip()
            ins["recommendation"] = _scrub_line(_one)[:400]
    except Exception:
        pass
    if not ins.get("recommendation"):
        nxt = ins.get("recommendedNextAnalysis") or {}
        rec = (nxt.get("question") if isinstance(nxt, dict) else "") or ins.get("next_question") or ""
        if rec:
            ins["recommendation"] = _scrub_line(rec)[:400]
    scen, mon = _capability_flags(ss)
    ins["scenario_supported"] = scen
    ins["monitoring_supported"] = mon
    # associated_factors only when real correlation evidence exists.
    t_low = itype.lower()
    if "correl" in t_low:
        src = (ins.get("source") or {}) if isinstance(ins.get("source"), dict) else {}
        tables = src.get("tables") or ins.get("source_tables") or []
        factors = [t for t in tables if isinstance(t, str) and t]
        if factors:
            ins["associated_factors"] = factors[:8]
    # visualization parity: map viz_type/viz_data -> visualization.
    if not isinstance(ins.get("visualization"), dict):
        vt = ins.get("viz_type", "bar")
        vd = ins.get("viz_data", [])
        ins["visualization"] = _normalize_chart_spec(vt, vd, ins.get("title", ""))
    return ins


def _normalize_legacy_insight(raw: dict) -> dict:
    """Normalize a B-shape raw detector dict to the full A-shape contract.

    Maps summary->description (keeps summary), viz_type/viz_data->
    visualization via _normalize_chart_spec (bounded 500/2000 caps), then
    enriches with the generic per-insight fields. Pure mapping — no
    dataset-specific literals, no invented numbers.
    """
    if not isinstance(raw, dict):
        return raw
    if isinstance(raw.get("visualization"), dict) and raw.get("description"):
        return _enrich_insight(raw)
    now = datetime.now().isoformat()
    title = raw.get("title", "")
    summary = raw.get("summary", "")
    viz_type = raw.get("viz_type", "bar")
    viz_data = raw.get("viz_data", [])
    spec = _normalize_chart_spec(viz_type, viz_data, title)
    ev_raw = raw.get("evidence") if isinstance(raw.get("evidence"), dict) else {}
    sample_size = ev_raw.get("sampleSize", raw.get("sample_size"))
    table = raw.get("table", "")
    itype = raw.get("type", "")
    tags = [str(table)] if table else []
    if itype:
        tags.append(str(itype).lower())
    out = {
        "id": raw.get("id", f"ins_{now}"),
        "type": itype or "Trend",
        "severity": raw.get("severity", "low"),
        "title": title,
        "description": raw.get("description", summary),
        "summary": summary or raw.get("description", ""),
        "metric": raw.get("metric", ""),
        "metricValue": raw.get("metric_value", raw.get("metricValue")),
        "previousValue": raw.get("previous_value", raw.get("previousValue")),
        "magnitude": raw.get("magnitude", ""),
        "confidence": raw.get("confidence", 50),
        "detectedAt": raw.get("detectedAt", now),
        "affectedPeriod": raw.get("affectedPeriod", raw.get("affected_period", "")),
        "tags": tags,
        "evidence": {
            "observations": ev_raw.get("observations", raw.get("observations", [])),
            "supportingMetrics": ev_raw.get("supportingMetrics", raw.get("support_metrics", [])),
            "sampleSize": sample_size,
            "baseline": ev_raw.get("baseline", raw.get("baseline")),
            "currentValue": ev_raw.get("currentValue", raw.get("current_value")),
            "changePercent": ev_raw.get("changePercent", raw.get("change_pct", raw.get("changePercent"))),
        },
        "businessImpact": {
            "value": raw.get("impact_value", (raw.get("businessImpact") or {}).get("value") if isinstance(raw.get("businessImpact"), dict) else None),
            "direction": raw.get("impact_direction", "neutral"),
            "explanation": raw.get("impact_explanation", ""),
        },
        "recommendedNextAnalysis": {
            "question": raw.get("next_question", ""),
            "metric": raw.get("metric", ""),
            "dimensions": raw.get("next_dims", []),
        },
        "visualization": {
            "type": spec["type"],
            "x": spec["x"],
            "y": spec["y"],
            "series": spec["series"],
            "title": spec["title"],
            "interactive": spec["interactive"],
            "data": spec["data"],
            "insufficient": spec.get("insufficient", False),
            "unsupported": spec.get("unsupported", False),
        },
        "source": {
            "tables": raw.get("source_tables", ([table] if table else [])),
            "columns": raw.get("source_columns", []),
        },
    }
    # Preserve raw evidence extras (e.g. affectedValues) without overwriting.
    for k, v in ev_raw.items():
        if k not in out["evidence"]:
            out["evidence"][k] = v
    # Keep legacy viz keys for backward compatibility.
    out["viz_type"] = viz_type
    out["viz_data"] = _sanitize_spec_rows(viz_data)[:_MAX_SPEC_ROWS]
    return _enrich_insight(out)


# ── Insight Detection Engine ─────────────────────────────────────────────


def _translate_sql_for_sqlite(sql: str) -> str:
    """Translate the PG-isms used by InsightsEngine detectors to SQLite."""
    import re as _re

    sql = _re.sub(
        r"date_trunc\(\s*'month'\s*,\s*(.*?)\)",
        r"strftime('%Y-%m', \1)",
        sql,
        flags=_re.IGNORECASE,
    )
    return sql.replace("%s", "?")
