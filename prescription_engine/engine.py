"""Prescription Engine: evidence -> problem -> driver -> controllability ->
candidates -> scenarios -> rank. Self-contained statistics (numpy/scipy);
reuses only SchemaIntelligence for schema discovery. All data access is
bounded and read-only. The LLM never computes numbers — it may only polish
wording built deterministically from the evidence.
"""

import logging
import math
import re

import numpy as np

log = logging.getLogger("aria.prescription_engine")

try:
    from scipy import stats as _st
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False

MIN_N = 10
SAMPLE_LIMIT = 5000
MIN_ABS_R = 0.30

# --------------------------------------------------------------------------
# AUDIT (FYP top-notch): which lever shapes produce scenario actions today,
# and why the rest refuse. Decision recorded here where the gates live.
#
# SCENARIO actions require ALL of the following (thresholds UNCHANGED):
#   1. Numeric driver: same-table numeric column, cross-table numeric over a
#      declared FK edge, or the derived length_of_stay driver (2+ date cols).
#      Text dims NEVER qualify — classify_driver() returns "segmentation" for
#      non-numeric columns or segment-hint names (region/category/product/...),
#      and analyze() skips non-numeric same-table candidates outright, so a
#      field like ProductCategory can only ever seed investigate-type cards.
#   2. Correlation gates: |r| >= MIN_ABS_R (0.30) AND p < 0.05 (Pearson, with
#      a Spearman agreement backstop downstream). Weak correlations refuse
#      with no_prescription naming the tested columns — association without
#      strength is supporting evidence only, never an intervention.
#   3. Controllability gate: classify_driver() level must be controllable /
#      potentially_controllable. not_controllable / segmentation refuse (or
#      fall back to investigate). Technical/system + identifier/FK/date roles
#      (NON_LEVER_ROLES) refuse even earlier via semantic-role validation.
#   4. Justified direction: improvement_up() must establish whether higher or
#      lower metric values are better (recognized outcome semantics, else an
#      EXPLICIT recorded assumption with capped confidence). Unknown → refuse.
#   5. Full actionability verdict: assess_actionability() 8-point check —
#      correlation alone can never carry a lever.
#   6. Non-degenerate data: driver mean/variance must be non-zero; |r| >= 0.999
#      is rejected as a self-join/constant-column artifact.
# Consequence: retail-style CSVs (one metric + one text dim, no numeric
# lever) ALWAYS refuse scenarios — honest, but thin coverage for the FYP
# demo. The segment-focus + monitor actions below broaden coverage WITHOUT
# weakening honesty: each keeps its own evidence gates (share/gap/n and
# momentum/points) plus an explicit honest note, and engine-level validation
# drops anything that arrives with neither quantified impact nor note.
# --------------------------------------------------------------------------
MIN_REL_PCT_ALIAS = None  # (real floors live with quality_gate below)
SEGMENT_FOCUS_MIN_SHARE = 0.30   # dominant segment: top group holds >=30% of rows
SEGMENT_FOCUS_MIN_GAP = 0.20     # large-gap segment: top-vs-median mean gap >=20%
SEGMENT_FOCUS_MIN_N = 10         # total usable observations floor (== MIN_N)
SEGMENT_FOCUS_MIN_GROUP_N = 5    # per-group stability floor (matches investigate)
MONITOR_MIN_POINTS = 6           # momentum needs >=6 ordered points
MONITOR_MIN_MOMENTUM = 0.15      # |last-first|/mean >= 15% counts as strong
HONEST_IMPACT_NOTE = ("Impact could not be reliably estimated from the available "
                      "data — this card directs attention; it does not promise "
                      "a sized outcome.")

# --------------------------------------------------------------------------
# Structured error codes (§29)
# --------------------------------------------------------------------------
CODE_NO_CONNECTION = "NO_CONNECTION"
CODE_INVALID_METRIC = "INVALID_METRIC"
CODE_INVALID_DRIVER = "INVALID_DRIVER"
CODE_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
CODE_INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
CODE_SCENARIO_UNSUPPORTED = "SCENARIO_UNSUPPORTED"
CODE_ANALYSIS_FAILED = "ANALYSIS_FAILED"
CODE_MODEL_FAILED = "MODEL_FAILED"
CODE_LLM_FAILED = "LLM_FAILED"
CODE_TIMEOUT = "TIMEOUT"
CODE_INTERNAL_ERROR = "INTERNAL_ERROR"


def _fail(code, detail, **extra):
    out = {"ok": False, "success": False, "code": code,
           "detail": detail, "recoverable": True}
    out.update(extra)
    return out


# --------------------------------------------------------------------------
# Controllability (§8): name + type heuristics, never hardcoded schema
# --------------------------------------------------------------------------
_CONTROLLABLE_HINTS = ("price", "pricing", "discount", "promo", "spend",
                       "spending", "budget", "bid", "fee", "cost_per",
                       "inventory", "stock", "reorder", "staff", "hours")
_MAYBE_CONTROLLABLE = ("market", "advert", "campaign", "channel_spend",
                       "quantity_ordered", "supply", "stay", "nights", "duration",
                       "length_of_stay")
_SEGMENT_HINTS = ("region", "area", "city", "state", "country", "zone",
                  "territory", "category", "product", "brand", "sku",
                  "segment", "channel", "store", "department", "division",
                  "gender", "cohort", "tier", "payment", "ship")
_TIME_HINTS = ("date", "time", "day", "month", "year", "created", "updated",
               "timestamp", "period", "week", "quarter")
_LOWER_BETTER_HINTS = ("cost", "expense", "churn", "loss", "debt", "risk",
                       "complaint", "return_rate", "defect", "delay")


# --------------------------------------------------------------------------
# Semantic roles + actionability (hyp.txt Response §§5-6). Generalized name
# signals only — NEVER a hardcoded per-column check. Association is
# evidence of association: only a lever passing semantic-role +
# actionability validation may become a recommendation/scenario variable.
# Reuses core/semantic.py and core/relational.py where they exist.
# --------------------------------------------------------------------------

# Recognized business-outcome vocabulary: higher values indicate
# improvement. Generalized semantics (never lever→outcome pairs).
_HIGHER_BETTER_HINTS = ("revenue", "revenues", "profit", "profits", "margin",
                        "margins", "income", "sales", "conversion",
                        "conversions", "retention", "satisfaction", "growth",
                        "gain", "gains", "yield", "booking", "bookings",
                        "demand", "attendance", "enrollment", "signup",
                        "signups", "subscription", "subscriptions")

# Telemetry/system measurement vocabulary: analytic signal, never a lever.
_TECHNICAL_MEASURE_HINTS = (
    "bytes", "byte_size", "millisecond", "milliseconds", "millisec",
    "microsecond", "microseconds", "nanosecond", "nanoseconds",
    "latency", "ping", "ttl", "cpu", "gpu", "memory", "heap",
    "disk", "iops", "cache", "page_fault", "packet", "packets",
    "bandwidth", "throughput", "payload", "query_time", "query_duration",
    "execution_time", "exec_time", "response_time", "elapsed",
    "duration_ms", "duration_sec", "duration_seconds", "wait_time",
    "load_avg", "thread", "threads", "kernel", "uptime", "file_size",
    "filesize", "memory_mb", "cpu_time", "io_wait", "network_io",
)

# Counters / storage internals: analytic metadata, never a lever.
_SYSTEM_METADATA_HINTS = (
    "row_count", "rowcount", "record_count", "num_rows", "counter",
    "counters", "sequence", "seq_no", "seqno", "checksum", "encoding",
    "encoded", "partition_key",
)

# Recognized business-measure vocabulary (generic, schema-agnostic).
_BUSINESS_MEASURE_HINTS = (
    "price", "pricing", "cost", "amount", "revenue", "profit", "margin",
    "sales", "total", "payment", "spend", "spending", "budget", "discount",
    "quantity", "qty", "inventory", "stock", "wage", "salary", "fee",
    "rate", "pct", "percent", "ratio", "conversion", "retention",
    "order", "orders", "booking", "demand", "supply", "volume", "traffic",
    "freight", "fare", "tuition", "premium", "warranty",
)

# Roles that may (subject to the 8-point actionability check) become levers.
LEVER_ROLES = frozenset({"BUSINESS_MEASURE", "MEASURE"})
# Roles that can never be a recommendation outcome.
NON_OUTCOME_ROLES = frozenset({"TECHNICAL_MEASURE", "SYSTEM_METADATA",
                               "IDENTIFIER", "FOREIGN_KEY", "DATE_TIME",
                               "UNKNOWN"})
# Roles that can never be a lever/scenario variable.
NON_LEVER_ROLES = frozenset({"TECHNICAL_MEASURE", "SYSTEM_METADATA",
                             "IDENTIFIER", "FOREIGN_KEY", "DATE_TIME",
                             "DIMENSION", "CATEGORY", "TEXT", "UNKNOWN"})

# Canonical honest state when no legitimate business lever exists (§9).
NO_SUPPORTED_SCENARIO = "No supported actionable scenario"


def _name_tokens(name) -> list:
    """Lowercase alphanumeric tokens of a column name (snake/camel safe)."""
    parts = re.split(r"[^a-z0-9]+", str(name or "").lower())
    toks: list = []
    for p in parts:
        toks.extend(re.findall(r"[a-z]+|[0-9]+", p))
    return [t for t in toks if t]


def _has_hint(name, hints) -> bool:
    """Generalized vocabulary match: token equality for short hints,
    substring for compound/long hints (len>=8 or containing '_').

    Token equality keeps short signals precise (``ram`` never matches
    ``program``; ``counter`` never matches ``encounter``).
    """
    low = str(name or "").lower()
    toks = set(_name_tokens(low))
    for h in hints:
        h = str(h).lower()
        if not h:
            continue
        if h in toks:
            return True
        if len(h) >= 8 or "_" in h:
            if h in low:
                return True
    return False


def classify_semantic_role(name, col_category="", *, is_pk=False,
                           is_fk=False, is_join_key=False) -> str:
    """Semantic role of a field (hyp.txt §5) — generalized, data-aware.

    Returns one of: IDENTIFIER / FOREIGN_KEY / DATE_TIME / MEASURE /
    BUSINESS_MEASURE / TECHNICAL_MEASURE / DIMENSION / CATEGORY / TEXT /
    SYSTEM_METADATA / UNKNOWN. Reuses core/semantic.py and
    core/relational.py signals where available; never a per-column check.
    """
    low = str(name or "")
    cat = str(col_category or "").lower()
    try:
        from core.relational import is_ref_flavored as _is_ref
        _ref = bool(_is_ref(name))
    except Exception:
        _ref = False
    tail = low.lower().split("_")[-1] if low else ""
    id_like = (_ref or tail in ("id", "key", "code", "no", "num", "uuid",
                               "guid", "pk", "fk")
               or low.lower() in ("id", "key", "code"))
    if is_fk:
        return "FOREIGN_KEY"
    if is_pk or is_join_key and id_like:
        return "IDENTIFIER"
    if id_like:
        return "IDENTIFIER"
    if cat == "datetime" or any(h in low.lower() for h in _TIME_HINTS):
        return "DATE_TIME"
    if _has_hint(low, _TECHNICAL_MEASURE_HINTS):
        return "TECHNICAL_MEASURE" if cat in ("numeric", "", "unknown",
                                              "other") else "SYSTEM_METADATA"
    if _has_hint(low, _SYSTEM_METADATA_HINTS):
        return "SYSTEM_METADATA"
    if cat == "numeric":
        try:
            from core.semantic import classify_semantic_type as _cst
            _st = _cst(low, "")
        except Exception:
            _st = "measure"
        if _st == "identifier":
            return "IDENTIFIER"
        if _has_hint(low, _BUSINESS_MEASURE_HINTS):
            return "BUSINESS_MEASURE"
        return "MEASURE"
    if cat in ("text", "binary", "other", "", "unknown"):
        try:
            from core.semantic import classify_semantic_type as _cst2
            _st2 = _cst2(low, "")
        except Exception:
            _st2 = "text"
        if _st2 == "identifier":
            return "IDENTIFIER"
        if _st2 == "date":
            return "DATE_TIME"
        if _st2 == "category":
            return "CATEGORY"
        if _st2 == "geography":
            return "DIMENSION"
        return "TEXT"
    return "UNKNOWN"


def assess_actionability(lever_name, lever_role, control_level, *,
                         metric_role=None, direction_known=False,
                         has_evidence=False) -> dict:
    """8-point business-actionability check (hyp.txt §6).

    A variable becomes a recommendation/scenario lever ONLY if every
    applicable check passes. Correlation alone (has_evidence without the
    other gates) can never carry a lever.
    """
    failed, notes = [], []

    def _need(ok: bool, code: str, note: str):
        if not ok:
            failed.append(code)
            notes.append(note)

    _need(lever_role in LEVER_ROLES, "business_concept",
          f"'{lever_name}' is a {lever_role}, not a meaningful business concept.")
    _need(control_level in ("controllable", "potentially_controllable"),
          "controllable",
          f"'{lever_name}' is not controllable or influenceable ({control_level}).")
    _need(lever_role not in NON_LEVER_ROLES, "plausible_action",
          f"Changing '{lever_name}' is not a plausible business action.")
    if metric_role is not None:
        _need(metric_role not in NON_OUTCOME_ROLES, "outcome_metric",
              "No meaningful business outcome metric is attached to this driver.")
    _need(bool(has_evidence), "evidence",
          "No statistical evidence supports the driver→outcome relationship.")
    _need(bool(direction_known), "justified_direction",
          "The proposed direction of change is not justified.")
    _need(bool(has_evidence and direction_known
               and control_level in ("controllable", "potentially_controllable")
               and (metric_role is None or metric_role not in NON_OUTCOME_ROLES)),
          "not_correlation_only",
          "Correlation alone cannot justify an intervention.")
    # Check 8 (no invented assumption) holds structurally: direction is only
    # ever 'known' via recognized outcome semantics or an explicitly
    # recorded modeling assumption — never a silent generic rule.
    return {"actionable": not failed, "failed": failed, "notes": notes}


def _disp(raw):
    """User-facing 'Human (raw)' label for refusal/diagnostic strings.

    The humanized name is the primary label; the raw column name follows
    in parentheses so logs, tests and technical readers can still match
    the physical field. Findings prose uses humanize() alone instead.
    """
    from core.semantic import humanize as _h
    h = _h(raw)
    return f"{h} ({raw})" if h != str(raw) else str(raw)


def _join_phrase(join_used) -> str | None:
    """User-facing join provenance in business words.

    Generic over any declared edge ("linked X records"); the raw
    table.column path stays in the structured join/join_ref fields and
    server logs, never as the primary label.
    """
    if not join_used:
        return None
    try:
        from core.semantic import humanize as _h
        return f"linked {_h(join_used[0])} records"
    except Exception:
        return "linked records"


def classify_driver(name, col_category):
    """Return (level, reason) with level in controllable / potentially /
    segmentation / not_controllable."""
    low = (name or "").lower()
    tail = low.split("_")[-1] if low else ""
    if col_category == "datetime" or any(h in low for h in _TIME_HINTS):
        return "not_controllable", "Time cannot be changed — ARIA can only prepare for it."
    if tail in ("id", "key", "code", "no") or low in ("id", "key", "code"):
        return "not_controllable", "Identifiers cannot be intervened on."
    # Technical/system measurements and storage metadata describe the
    # machinery, not the business — association only, never a lever (§§4-6).
    # Generalized vocabulary (never per-column); checked BEFORE the
    # controllable hints so e.g. an inventory-flavored cache counter can
    # never slip through as actionable.
    if _has_hint(name, _TECHNICAL_MEASURE_HINTS) or _has_hint(name, _SYSTEM_METADATA_HINTS):
        return ("not_controllable",
                "Technical/system measurements cannot be intervened on — association only, never a driver.")
    if any(h in low for h in _CONTROLLABLE_HINTS):
        return "controllable", "This driver can be set directly as an intervention."
    if any(h in low for h in _MAYBE_CONTROLLABLE):
        if "stay" in low or "night" in low or "duration" in low:
            return ("potentially_controllable",
                    "Length of stay can be influenced (offers, minimum-stay rules) but not set directly.")
        return "potentially_controllable", "This driver may be adjustable; treat estimates cautiously."
    if col_category != "numeric" or any(h in low for h in _SEGMENT_HINTS):
        return "segmentation", "This field segments the business — target an intervention at a group, don't change the field itself."
    return "potentially_controllable", "No evidence this driver is directly settable; estimates are exploratory."


def improvement_up(metric):
    """Validated improvement direction (hyp.txt §12) — never a silent
    generic assumption.

    Returns (up_better_or_None, basis). Direction is established ONLY from
    recognized outcome semantics (higher-better / lower-better vocabulary)
    or, for business-plausible metrics without recognized semantics, an
    EXPLICITLY recorded modeling assumption (confidence is capped
    downstream when the fallback is used). Technical/system fields have
    unknown direction → None, which refuses the recommendation at the
    call site instead of inventing one.
    """
    role = classify_semantic_role(metric, "numeric")
    if role in NON_OUTCOME_ROLES:
        return None, ("direction cannot be established for this field — it is "
                      "not a business outcome, so no recommendation basis exists")
    low = (metric or "").lower()
    if any(h in low for h in _LOWER_BETTER_HINTS):
        return False, "lower values indicate improvement for this outcome (recognized cost/risk semantics)"
    if _has_hint(metric, _HIGHER_BETTER_HINTS):
        return True, "higher values indicate improvement for this outcome (recognized outcome semantics)"
    from core.semantic import humanize as _h
    return True, (f"no outcome semantics recognized for {_h(metric)} — treating higher values as "
                  "improvement as an explicit modeling assumption; confirm this matches your objective")


# --------------------------------------------------------------------------
# Bounded read-only data access (dialect-aware, no raw frontend SQL)
# --------------------------------------------------------------------------
def _exec(engine, sql, params=None):
    from sqlalchemy import text as _t
    with engine.connect() as conn:
        return conn.execute(_t(sql), params or {}).fetchall()


def _fetch_pairs(engine, dialect, table, col_a, col_b, filt=None, limit=SAMPLE_LIMIT):
    from core.schema_intel import _safe_name
    where = (f"{_safe_name(col_a, dialect)} IS NOT NULL AND "
             f"{_safe_name(col_b, dialect)} IS NOT NULL")
    params = {}
    if filt:
        where += f" AND {_safe_name(filt[0], dialect)} = :flt"
        params["flt"] = filt[1]
    # Deterministic sample: ORDER BY the sampled columns so identical
    # requests analyze identical rows (bare LIMIT is return-order dependent).
    return _exec(engine,
                 f"SELECT {_safe_name(col_a, dialect)}, {_safe_name(col_b, dialect)} "
                 f"FROM {_safe_name(table, dialect)} WHERE {where} "
                 f"ORDER BY {_safe_name(col_a, dialect)}, {_safe_name(col_b, dialect)} "
                 f"LIMIT {int(limit)}", params)


# --------------------------------------------------------------------------
# Statistics (deterministic; scipy when present, documented fallbacks)
# --------------------------------------------------------------------------
def _pearson_p(r, n):
    if n < 3 or abs(r) >= 1.0:
        return 0.0 if abs(r) >= 1.0 else 1.0
    if _SCIPY_OK:
        try:
            from scipy import stats as _s
            t = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
            return float(2 * (1 - _s.t.cdf(abs(t), n - 2)))
        except Exception:
            pass
    # Normal approximation fallback (documented, conservative).
    try:
        z = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(n - 3)
        return float(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))
    except Exception:
        return 1.0


def _spearman(x, y):
    if len(x) < 3:
        return 0.0, 1.0
    if _SCIPY_OK:
        rx = _st.rankdata(x, method='average')
        ry = _st.rankdata(y, method='average')
    else:
        # numpy-only fallback: average ranks with tie-breaking
        def _avg_rank(arr):
            order = arr.argsort()
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
            # average tied ranks
            sorted_arr = arr[order]
            i = 0
            while i < len(sorted_arr):
                j = i + 1
                while j < len(sorted_arr) and sorted_arr[j] == sorted_arr[i]:
                    j += 1
                if j > i + 1:
                    avg = np.arange(i + 1, j + 1, dtype=float).mean()
                    for k in range(i, j):
                        ranks[order[k]] = avg
                i = j
            return ranks
        rx = _avg_rank(np.asarray(x, dtype=float))
        ry = _avg_rank(np.asarray(y, dtype=float))
    r = float(np.corrcoef(rx, ry)[0, 1])
    return r, _pearson_p(r, len(x))


def fit_relationship(x, y):
    """OLS of y on x with slope SE, residual sigma, CIs. Returns None if unfit.

    Numerical failures (e.g. SVD non-convergence in polyfit on extreme
    scales) also return None — "cannot model", never an exception — so
    every caller degrades to its honest refusal path instead of a 500.
    """
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(x)
        if n < MIN_N or float(x.std()) == 0 or float(y.std()) == 0:
            return None
        r = float(np.corrcoef(x, y)[0, 1])
        p = _pearson_p(r, n)
        rs, ps = _spearman(x, y)
        slope, intercept = (float(v) for v in np.polyfit(x, y, 1))
    except Exception:
        return None
    try:
        pred = slope * x + intercept
        resid = y - pred
        sigma = float(resid.std(ddof=1)) if n > 2 else 0.0
        sxx = float(np.sum((x - x.mean()) ** 2))
        se_slope = sigma / math.sqrt(sxx) if sxx > 0 else 0.0
        z = 0.5 * math.log((1 + r) / (1 - r)) if abs(r) < 1 else 0.0
        se = 1.0 / math.sqrt(max(1, n - 3))
        ci = [round(math.tanh(z - 1.96 * se), 3), round(math.tanh(z + 1.96 * se), 3)]
    except Exception:
        return None
    return {"n": n, "r": round(r, 3), "p_value": p,
            "spearman_r": round(rs, 3), "spearman_p": ps,
            "slope": slope, "intercept": intercept,
            "se_slope": se_slope, "sigma": sigma, "ci": ci,
            "x_mean": float(x.mean()), "y_mean": float(y.mean()),
            "x_min": float(x.min()), "x_max": float(x.max())}


# --------------------------------------------------------------------------
# Scenario math (§11): pure functions of the fitted model — SIMULATED labels
# --------------------------------------------------------------------------
def scenario_for_pct(fit, pct):
    """Scenario KPIs for a driver change of pct/100. All values SIMULATED."""
    base = fit["intercept"] + fit["slope"] * fit["x_mean"]
    dx = fit["x_mean"] * (pct / 100.0)
    scen = base + fit["slope"] * dx
    delta = scen - base
    # Uncertainty: slope CI propagated (±1.96·se·|dx|) plus residual noise.
    half = 1.96 * fit["se_slope"] * abs(dx)
    lo = scen - half - fit["sigma"] * 0.1
    hi = scen + half + fit["sigma"] * 0.1
    new_x = fit["x_mean"] + dx
    in_range = fit["x_min"] <= new_x <= fit["x_max"]
    return {"driver_pct": pct, "driver_value": new_x,
            "baseline": base, "scenario": scen, "delta": delta,
            "rel_pct": (delta / abs(base) * 100) if base else 0.0,
            "range": [lo, hi], "in_historical_range": in_range}


def risk_level(fit, pct, strength):
    """LOW/MEDIUM/HIGH + reasons (§20)."""
    reasons = ["Historical data shows an association; it does not establish "
               "that changing this variable will directly cause the predicted result."]
    level = 1  # 0 low, 1 medium, 2 high
    scen = scenario_for_pct(fit, pct)
    if not scen["in_historical_range"]:
        level = max(level, 2)
        reasons.append("Scenario moves the driver outside its observed historical range.")
    if abs(fit["r"]) < 0.5:
        level = max(level, 1)
        reasons.append("Moderate relationship strength — estimates carry real uncertainty.")
    if fit["n"] < 50:
        level = max(level, 1)
        reasons.append(f"Modest sample ({fit['n']} observations).")
    if fit["p_value"] < 0.01 and fit["n"] >= 100 and scen["in_historical_range"] and abs(fit["r"]) >= 0.5:
        level = 0
    if abs(pct) > 15:
        level = max(level, 2)
        reasons.append("Large intervention size relative to observed variation.")
    return (["LOW", "MEDIUM", "HIGH"][level], reasons)


def _score(fit, pct, control_level):
    """Transparent 0–100 ranking (§14): components, never a mystery number."""
    comps = {
        "evidence_strength": round(abs(fit["r"]) * 30, 1),          # 0–30
        "expected_impact": round(min(20.0, abs(scenario_for_pct(fit, pct)["rel_pct"]) * 2), 1),  # 0–20
        "confidence": round((15 if fit["p_value"] < 0.01 else 8 if fit["p_value"] < 0.05 else 0)
                            + min(10.0, fit["n"] / 20), 1),          # 0–25
        "controllability": {"controllable": 15, "potentially_controllable": 9,
                            "segmentation": 5, "not_controllable": 0}[control_level],  # 0–15
        "data_quality": round(min(10.0, fit["n"] / 50), 1),          # 0–10
    }
    total = round(sum(comps.values()), 1)
    return total, comps


# --------------------------------------------------------------------------
# Canonical quality gate (§3): ONE function every decision surface uses.
# Thresholds reuse the module's own bars (MIN_N, MIN_ABS_R); the two extra
# effect floors are documented here, never invented per-call.
# --------------------------------------------------------------------------
MIN_REL_PCT = 5.0    # negligible-effect floor: |rel_pct| below this is noise
MIN_COHENS_D = 0.2   # Cohen's "small" boundary: below this is NOT_ACTIONABLE
CAUSATION_WARNING = ("Historical data shows an association; it does not establish "
                     "that changing this variable will directly cause the predicted result.")


def quality_gate(evidence: dict | None, *, require_baseline: bool = False) -> dict:
    """Reject/downgrade weak evidence. Returns a code + reason, never a Decision.

    Codes: OK | INSUFFICIENT_DATA | WEAK_SIGNAL | NOT_ACTIONABLE.
    `evidence` is a plain dict (engine prescription evidence + impact +
    confidence merged by the caller via gate_input_from_prescription).
    """
    ev = evidence if isinstance(evidence, dict) else {}

    def _num(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if f != f:  # NaN-safe
            return None
        return f

    metric = ev.get("metric")
    n = ev.get("n")
    try:
        n = int(n) if n is not None else None
    except (TypeError, ValueError):
        n = None
    p = _num(ev.get("p_value"))
    r = _num(ev.get("r") if ev.get("r") is not None else ev.get("effect_r"))
    rel = _num(ev.get("rel_pct") if ev.get("rel_pct") is not None
               else ev.get("effect_gap_pct"))
    d = _num(ev.get("cohens_d"))
    conf = str(ev.get("confidence") or "").strip().lower()
    baseline = ev.get("baseline")
    delta = ev.get("delta")
    control = str(ev.get("control_level") or ev.get("control") or "").lower()

    def _refuse(code, reason):
        try:
            from core.evidence import quality_gate_status as _qgs
            _status = _qgs(code)
        except Exception:
            _status = "insufficient_evidence"
        return {"pass": False, "code": code, "reason": reason,
                "status": _status,
                "evidence_quality": "Limited",
                "confidence": ev.get("confidence") if ev.get("confidence") else "Limited"}

    if not ev:
        return _refuse("INSUFFICIENT_DATA", "No evidence was computed for this driver.")
    if not metric:
        return _refuse("INSUFFICIENT_DATA", "No measurable target metric is attached to this evidence.")
    if ev.get("metric_reliable") is False:
        return _refuse("INSUFFICIENT_DATA", "The target metric failed the reliability check.")
    if n is None or n < MIN_N:
        return _refuse("INSUFFICIENT_DATA",
                       f"Only {n if n is not None else 0} usable observations — "
                       f"at least {MIN_N} are needed.")
    if p is None:
        return _refuse("INSUFFICIENT_DATA", "No significance test supports this driver yet.")
    if p >= 0.05:
        return _refuse("WEAK_SIGNAL", f"Not statistically significant (p={p:.3f} ≥ 0.05).")
    if r is not None and abs(r) < MIN_ABS_R:
        return _refuse("WEAK_SIGNAL",
                       f"Negligible association (|r|={abs(r):.2f} < {MIN_ABS_R:.2f}).")
    if rel is not None and abs(rel) < MIN_REL_PCT:
        return _refuse("NOT_ACTIONABLE",
                       f"Negligible effect ({rel:+.1f}% < ±{MIN_REL_PCT:g}% materiality floor).")
    if d is not None and abs(d) < MIN_COHENS_D:
        return _refuse("NOT_ACTIONABLE",
                       f"Tiny effect size (Cohen's d={d:.2f} < {MIN_COHENS_D:g}).")
    if conf in ("", "unknown", "none"):
        return _refuse("INSUFFICIENT_DATA",
                       "Confidence is unquantified — no validated decision can be made.")
    if ev.get("rank_consistent") is False:
        return _refuse("WEAK_SIGNAL",
                       "Unstable link — rank-based check disagrees with the linear fit.")
    _sp = _num(ev.get("spearman_r"))
    if r is not None and _sp is not None and abs(r - _sp) > 0.4:
        return _refuse("WEAK_SIGNAL",
                       f"Unstable link (Pearson {r:+.2f} vs Spearman {_sp:+.2f}).")
    if control in ("not_controllable", "segmentation"):
        return _refuse("NOT_ACTIONABLE",
                       "This field describes groups; it cannot be moved as an intervention.")
    if require_baseline:
        try:
            b = float(baseline) if baseline is not None else None
        except (TypeError, ValueError):
            b = None
        try:
            dl = float(delta) if delta is not None else None
        except (TypeError, ValueError):
            dl = None
        if b is None:
            return _refuse("INSUFFICIENT_DATA",
                           "No measured baseline — the impact cannot be quantified.")
        if dl is None and rel is None:
            return _refuse("INSUFFICIENT_DATA",
                           "Impact is unquantifiable — no modeled delta or relative change.")

    quality = "Moderate"
    try:
        strong = (p < 0.01 and n >= 100 and
                  ((r is not None and abs(r) >= 0.5) or (d is not None and abs(d) >= 0.8)))
    except TypeError:
        strong = False
    if strong:
        quality = "Strong"
    try:
        from core.evidence import quality_gate_status as _qgs2
        _ok_status = _qgs2("OK")
    except Exception:
        _ok_status = "supported"
    return {"pass": True, "code": "OK", "reason": None,
            "status": _ok_status,
            "evidence_quality": quality, "confidence": ev.get("confidence")}


def gate_input_from_prescription(rec: dict, *, require_baseline: bool = True) -> dict:
    """Merge one engine prescription into a quality_gate input dict."""
    rec = rec if isinstance(rec, dict) else {}
    ev = rec.get("evidence") if isinstance(rec.get("evidence"), dict) else {}
    imp = rec.get("impact") if isinstance(rec.get("impact"), dict) else {}
    return {
        "metric": rec.get("metric") or ev.get("metric"),
        "n": ev.get("n"),
        "p_value": ev.get("p_value"),
        "r": ev.get("effect_r", ev.get("r")),
        "effect_r": ev.get("effect_r", ev.get("r")),
        "spearman_r": ev.get("spearman_r"),
        "rank_consistent": ev.get("rank_consistent"),
        "rel_pct": imp.get("rel_pct"),
        "effect_gap_pct": ev.get("effect_gap_pct"),
        "cohens_d": ev.get("cohens_d"),
        "baseline": imp.get("baseline"),
        "delta": imp.get("delta"),
        "confidence": rec.get("confidence"),
        "control_level": rec.get("control_level"),
        "validation_method": ev.get("validation_method") or rec.get("validation_method"),
    }


def priority_score(rec: dict) -> tuple:
    """Transparent Priority (0–100): Impact × Confidence × Urgency ×
    Actionability × EvidenceQuality. Reuses this module's _score
    components plus impact rel% + p/n. Returns (total, breakdown)."""
    rec = rec if isinstance(rec, dict) else {}
    ev = rec.get("evidence") if isinstance(rec.get("evidence"), dict) else {}
    imp = rec.get("impact") if isinstance(rec.get("impact"), dict) else {}
    comps_in = rec.get("score_breakdown") if isinstance(rec.get("score_breakdown"), dict) else {}

    def _num(v, default=0.0):
        try:
            f = float(v)
            return f if f == f else default
        except (TypeError, ValueError):
            return default

    rel = imp.get("rel_pct")
    if rel is None:
        rel = ev.get("effect_gap_pct")
    impact_c = round(min(30.0, abs(_num(rel)) * 1.5), 1)                      # 0–30
    conf = str(rec.get("confidence") or "").lower()
    base = {"high": 20.0, "moderate": 12.0, "medium": 12.0, "low": 4.0}.get(conf, 0.0)
    p = _num(ev.get("p_value"), 1.0)
    bonus = 5.0 if p < 0.01 else (2.0 if p < 0.05 else 0.0)
    confidence_c = round(min(25.0, base + bonus), 1)                          # 0–25
    rp = str(rec.get("review_period") or "").lower()
    if "2 week" in rp:
        urgency_c = 10.0
    elif "4 week" in rp:
        urgency_c = 8.0
    elif "quarter" in rp or "month" in rp:
        urgency_c = 7.0
    else:
        urgency_c = 5.0                                                      # 0–10
    atype = str(rec.get("action_type") or "").lower()
    ctrl = _num(comps_in.get("controllability"), -1)
    if atype == "scenario":
        action_c = 20.0 if ctrl >= 15 else (14.0 if ctrl >= 9 else 10.0) if ctrl >= 0 else 12.0
    elif atype == "investigate":
        action_c = 10.0
    elif atype == "qualitative":
        action_c = 6.0
    else:
        action_c = 8.0                                                       # 0–20
    try:
        nn = int(ev.get("n")) if ev.get("n") is not None else 0
    except (TypeError, ValueError):
        nn = 0
    evidence_c = round(min(15.0, min(10.0, nn / 50.0)
                           + (5.0 if ev.get("rank_consistent") else 0.0)), 1)  # 0–15
    comps = {"impact": impact_c, "confidence": confidence_c, "urgency": urgency_c,
             "actionability": round(action_c, 1), "evidence_quality": evidence_c}
    return min(100.0, round(sum(comps.values()), 1)), comps


# ---------------------------------------------------------------------------
# Model-layer deduplication (hyp §§16-17 + §22): finding + unordered
# metric pair + scope + direction + analytical-basis identity — NEVER title
# string matching.
# "Test a 20% increase in Milliseconds" vs "Test a 45% increase in
# Milliseconds" vs "Test a 45% increase in Bytes (other table, same story)"
# share one key when the underlying finding is the same and collapse to the
# highest-scoring card. Scenario variants (+20/+30/+45%) belong INSIDE the
# surviving decision (`scenario_options`), not as cards. ONE finding → ONE
# decision context.
# §22: the pair is UNORDERED — "Milliseconds → Bytes" and
# "Bytes → Milliseconds" are the same bidirectional correlation evaluated
# in both directions and must collapse to one decision, never two cards.
# ---------------------------------------------------------------------------
def dedupe_key(rec: dict) -> tuple:
    """Duplicate identity for one prescription/decision card."""
    rec = rec if isinstance(rec, dict) else {}
    ev = rec.get("evidence") if isinstance(rec.get("evidence"), dict) else {}
    imp = rec.get("impact") if isinstance(rec.get("impact"), dict) else {}
    metric = rec.get("metric") or ev.get("metric")
    driver = rec.get("driver") or ev.get("driver")
    scope = rec.get("table") or ev.get("table")
    try:
        _pct = imp.get("driver_pct")
        _pct = float(_pct) if _pct is not None else 0.0
        direction = "up" if _pct > 0 else "down" if _pct < 0 else "flat"
    except (TypeError, ValueError):
        direction = "flat"
    r = ev.get("effect_r", ev.get("r"))
    try:
        _r = float(r) if r is not None else None
    except (TypeError, ValueError):
        _r = None
    gap = ev.get("effect_gap_pct")
    try:
        _gap = float(gap) if gap is not None else None
    except (TypeError, ValueError):
        _gap = None
    if _r is not None:
        basis = "pos" if _r > 0 else "neg" if _r < 0 else "none"
    elif _gap is not None:
        basis = "gap"
    else:
        basis = "none"
    finding = rec.get("action_type") or rec.get("scenario_label") or ""
    # Non-scenario cards are keyed by type+label so a segment-focus card and a
    # plain investigate card on the same dimension stay distinct findings
    # (ranked segment-focus > investigate). Scenario variants keep collapsing
    # (ONE finding → ONE decision; variants travel inside scenario_options).
    if str(finding or "").lower() != "scenario" and rec.get("scenario_label"):
        finding = f"{finding}|{rec.get('scenario_label')}"
    _m = str(metric or "").lower().strip()
    _d = str(driver or "").lower().strip()
    _pair = tuple(sorted([_m, _d]))
    return (_pair,
            str(scope or "").lower(), direction, basis,
            str(finding or "").lower())


def dedupe_prescriptions(recs: list) -> list:
    """Collapse model-layer duplicates, keeping the highest score each."""
    best, order = {}, []

    def _score_of(r):
        try:
            return float((r or {}).get("priority_score",
                                       (r or {}).get("score", 0)) or 0)
        except (TypeError, ValueError):
            return 0.0

    for r in (recs or []):
        k = dedupe_key(r if isinstance(r, dict) else {})
        if k not in best:
            best[k] = r
            order.append(k)
        elif _score_of(r) > _score_of(best[k]):
            best[k] = r
    return [best[k] for k in order]


# --------------------------------------------------------------------------
# Orchestrator (§4 pipeline)
# --------------------------------------------------------------------------
def _join_key_cols(intel, table):
    """Lowercased join-key columns touching `table` (declared edges + FKs).

    Generalized: derived from the schema mapping, never column literals.
    Join keys relate records; they are not business levers (§4).
    """
    keys = set()
    try:
        for e in ((intel.mapping or {}).get("relationship_edges") or []):
            s, sc = e.get("source_table"), e.get("source_column")
            t, tc = e.get("target_table"), e.get("target_column")
            if s == table and sc:
                keys.add(str(sc).lower())
            if t == table and tc:
                keys.add(str(tc).lower())
    except Exception:
        pass
    try:
        for c, _rt, _rc in intel.get_fks(table):
            keys.add(str(c).lower())
    except Exception:
        pass
    return keys


def _usable_measures(intel, table):
    """Numeric columns eligible as metric/lever candidates (§4).

    Excludes, by generalized rule (never per-column literals): PKs, FKs,
    join keys, identifier-like names, datetime-like names, technical
    measurements, system metadata/counters/encodings. A numeric column
    that merely CAN be increased/decreased is not automatically actionable
    — semantic-role + actionability gates decide downstream.
    """
    fks = {f[0].lower() for f in intel.get_fks(table)}
    pks = {p.lower() for p in intel.get_pk(table)}
    join_keys = _join_key_cols(intel, table)
    try:
        from core.semantic import is_year_like as _is_year_like
    except Exception:
        _is_year_like = lambda c: False  # noqa: E731
    out = []
    for c in intel.get_numeric_cols(table):
        low = c.lower()
        if low in fks or low in pks or low in join_keys:
            continue
        if low.endswith(("_id", "_key", "_no", "_code")) or low in ("id", "key", "code"):
            continue
        if _is_year_like(c):
            continue
        role = classify_semantic_role(
            c, "numeric", is_pk=(low in pks), is_fk=(low in fks),
            is_join_key=(low in join_keys))
        if role in NON_OUTCOME_ROLES or role in NON_LEVER_ROLES:
            continue
        out.append(c)
    return out


def _edge_neighbors(intel, table):
    """1-hop declared FK neighbors: (other_table, self_col, other_col).

    Only declared/inferred edges from the schema mapping — never guessed
    joins. Self-table edges and duplicate pairs are skipped.
    """
    out, seen = [], set()
    for e in (intel.mapping.get("relationship_edges") or []):
        s, sc = e.get("source_table"), e.get("source_column")
        t, tc = e.get("target_table"), e.get("target_column")
        if not (s and sc and t and tc):
            continue
        if s == table and t != table:
            key, row = (t, sc, tc), (t, sc, tc)
        elif t == table and s != table:
            key, row = (s, tc, sc), (s, tc, sc)
        else:
            continue
        if key not in seen:
            seen.add(key)
            out.append(row)
    for c, rt, rc in intel.get_fks(table):
        key = (rt, c, rc)
        if rt != table and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _stay_lengths(engine, dialect, table, c1, c2, metric, limit=SAMPLE_LIMIT):
    """Derived driver: days between two date columns vs the metric.

    Fully data-based (any table with 2+ dates + a numeric metric). Date
    parsing happens in Python so PG/MySQL/SQLite all behave identically.
    Returns (nights_list, metric_list) or (None, None) when unusable.
    """
    import pandas as pd
    from core.schema_intel import _safe_name
    try:
        rows = _exec(engine,
                     f"SELECT {_safe_name(c1, dialect)}, {_safe_name(c2, dialect)}, "
                     f"{_safe_name(metric, dialect)} FROM {_safe_name(table, dialect)} "
                     f"WHERE {_safe_name(c1, dialect)} IS NOT NULL "
                     f"AND {_safe_name(c2, dialect)} IS NOT NULL "
                     f"AND {_safe_name(metric, dialect)} IS NOT NULL "
                     f"LIMIT {int(limit)}")
    except Exception:
        return None, None
    if len(rows) < MIN_N:
        return None, None
    t1 = pd.Series(pd.to_datetime([r[0] for r in rows], errors="coerce"))
    t2 = pd.Series(pd.to_datetime([r[1] for r in rows], errors="coerce"))
    vals = pd.Series(pd.to_numeric([r[2] for r in rows], errors="coerce"))
    ok = t1.notna() & t2.notna() & vals.notna()
    if int(ok.sum()) < MIN_N:
        return None, None
    nights = ((t2[ok] - t1[ok]).dt.total_seconds() / 86400.0).to_numpy()
    ys = vals[ok].to_numpy(dtype=float)
    # Negative or absurd durations are data errors, not stays.
    mask = (nights >= 0) & (nights <= 365)
    if int(mask.sum()) < MIN_N:
        return None, None
    return nights[mask].tolist(), ys[mask].tolist()


def _kruskal_p(groups):
    """Kruskal-Wallis p across group arrays; None when scipy is absent."""
    if not _SCIPY_OK:
        return None
    try:
        _, p = _st.kruskal(*groups)
        return float(p)
    except Exception:
        return None


def _investigate_cands(engine, dialect, intel, table, metric, m_h,
                       review=None, review_basis=None):
    """Investigate-type prescriptions (§10/§22) from real segment gaps.

    For dimensions with 2–12 well-populated groups: Kruskal-Wallis + the
    top-vs-bottom gap. No simulated impact — the gap is OBSERVED evidence
    that names where to look next, never an invented intervention.
    """
    from core.semantic import humanize
    out = []
    for dim in intel.find_all_dimensions(table)[:3]:
        try:
            rows = _fetch_pairs(engine, dialect, table, dim, metric)
        except Exception:
            continue
        groups = {}
        for g, v in rows:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            groups.setdefault(str(g), []).append(fv)
        groups = {g: vs for g, vs in groups.items() if len(vs) >= 5}
        if not (2 <= len(groups) <= 12):
            continue
        arrs = [np.asarray(vs, dtype=float) for vs in groups.values()]
        p = _kruskal_p(arrs)
        if p is None or p >= 0.05:
            continue
        means = {g: sum(vs) / len(vs) for g, vs in groups.items()}
        overall = sum(means.values()) / len(means) if means else 0.0
        top = max(means, key=means.get)
        bottom = min(means, key=means.get)
        gap = ((means[top] - means[bottom]) / abs(overall)) if overall else 0.0
        if abs(gap) < 0.10:
            continue
        n = sum(len(vs) for vs in groups.values())
        d_h = humanize(dim)
        segs = sorted(((g, means[g], len(groups[g])) for g in groups),
                      key=lambda t: t[1], reverse=True)[:6]
        conf = ("high" if p < 0.01 and n >= 50 else "moderate")
        out.append({
            "id": f"rxinv_{table}_{metric}_{dim}",
            "rank": 0, "title": f"Investigate {d_h} performance for {m_h}: {top} vs {bottom}",
            "summary": (f"{m_h} differs across {d_h} ({top} averages {means[top]:,.1f} vs "
                        f"{bottom} at {means[bottom]:,.1f}, a {gap * 100:+.1f}% gap, "
                        f"seen across {n:,} rows). "
                        f"These groups are observed alongside the difference, "
                        f"so investigate what is linked before acting."),
            "action_type": "investigate", "scenario_label": "investigate",
            # hyp §13: a segment gap with no modeled intervention is an
            # Insight (a meaningful interpretation), never an Opportunity.
            "decision_type": "insight",
            "confidence": conf, "risk": "low",
            "risk_reasons": ["No intervention is modeled — this only directs investigation."],
            "score": round(min(60.0, abs(gap) * 100) + (15 if p < 0.01 else 8)
                           + min(10.0, n / 50), 1),
            "score_breakdown": {"observed_gap": round(min(30.0, abs(gap) * 100), 1),
                                "confidence": round(15 if p < 0.01 else 8, 1),
                                "data_quality": round(min(10.0, n / 50), 1)},
            "evidence": {"metric": metric, "driver": dim, "table": table,
                         "effect_gap_pct": round(gap * 100, 1), "p_value": p, "n": n,
                         "groups": [{"label": g, "mean": round(mn, 2), "n": cn}
                                    for g, mn, cn in segs],
                         "top": top, "bottom": bottom, "segment": None},
            "success_metric": {"metric": metric,
                               "direction": f"lift {bottom} toward {top} levels",
                               "target_pct": None,
                               "baseline": round(means[bottom], 2),
                               "note": "No modeled target yet — set one once investigation names the driver."},
            "review_period": review,
            "review_basis": review_basis,
            "impact": {"baseline": round(means[bottom], 2),
                       "scenario": round(means[top], 2),
                       "delta": round(means[top] - means[bottom], 2),
                       "rel_pct": round(gap * 100, 1),
                       "range": [round(means[bottom], 2), round(means[top], 2)],
                       "driver_pct": 0, "driver_value": None,
                       "labels": {"baseline": "OBSERVED", "scenario": "OBSERVED"}},
            "assumptions": ["Groups observed as-is; mix and other factors not separated."],
            "limitations": [
                f"Gap computed over {n} rows; smaller groups ({min(len(v) for v in groups.values())} rows) are noisier.",
                "A gap shows where to look — it does not say what to change.",
            ],
            "recommended_next_analysis": {
                "type": "hypothesis",
                "label": f"Test whether {m_h} truly differs across {d_h} and what drives it."},
        })
    out.sort(key=lambda c: c["score"], reverse=True)
    return out[:2]


def _segment_focus_cands(engine, dialect, intel, table, metric, m_h,
                         review=None, review_basis=None):
    """Segment-focus prescriptions: dominant / large-gap segments.

    Evidence gates (honest, descriptive — never a modeled intervention):
      * 2–12 groups, total n >= SEGMENT_FOCUS_MIN_N (10), top + median groups
        each n >= SEGMENT_FOCUS_MIN_GROUP_N (5) for stability;
      * FIRES when top-group row share >= 30% OR top-vs-median mean gap >= 20%.
    The Kruskal p-value rides along as INFORMATION in evidence (when scipy
    is present) but is NOT a firing gate: concentration (share of rows) is a
    descriptive Pareto-style fact, not a mean-difference claim. No impact
    numbers are invented — impact stays unquantified with an explicit
    HONEST_IMPACT_NOTE, which the engine-level validator requires.
    """
    from core.semantic import humanize
    out = []
    for dim in intel.find_all_dimensions(table)[:3]:
        try:
            rows = _fetch_pairs(engine, dialect, table, dim, metric)
        except Exception:
            continue
        groups = {}
        for g, v in rows:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            groups.setdefault(str(g), []).append(fv)
        groups = {g: vs for g, vs in groups.items()
                  if len(vs) >= SEGMENT_FOCUS_MIN_GROUP_N}
        if not (2 <= len(groups) <= 12):
            continue
        n = sum(len(vs) for vs in groups.values())
        if n < SEGMENT_FOCUS_MIN_N:
            continue
        means = {g: sum(vs) / len(vs) for g, vs in groups.items()}
        overall = sum(sum(vs) for vs in groups.values()) / n if n else 0.0
        ordered = sorted(groups, key=lambda g: means[g], reverse=True)
        mean_top, median = ordered[0], ordered[len(ordered) // 2]
        # Dominant segment = largest group BY ROW COUNT (Pareto concentration),
        # not top-by-mean: a high-share group with an average mean still
        # deserves review. The gap branch uses the top-by-mean group instead.
        biggest = max(groups, key=lambda g: len(groups[g]))
        share = len(groups[biggest]) / n if n else 0.0
        gap = ((means[mean_top] - means[median]) / abs(overall)) if overall else 0.0
        if not (share >= SEGMENT_FOCUS_MIN_SHARE
                or abs(gap) >= SEGMENT_FOCUS_MIN_GAP):
            continue
        top = biggest if share >= SEGMENT_FOCUS_MIN_SHARE else mean_top
        p = _kruskal_p([np.asarray(groups[g], dtype=float) for g in groups])
        share_pct = round(share * 100, 1)
        d_h = humanize(dim)
        segs = sorted(((g, means[g], len(groups[g])) for g in groups),
                      key=lambda t: t[1], reverse=True)[:6]
        conf = ("high" if p is not None and p < 0.01 and n >= 50 else "moderate")
        out.append({
            "id": f"rxseg_{table}_{metric}_{dim}",
            "rank": 0,
            "title": f"Review {top} — {share_pct:g}% of {m_h}",
            "summary": (f"{top} accounts for {share_pct:g}% of observed {m_h.lower()} rows "
                        f"(averaging {means[top]:,.1f} vs a midpoint segment at "
                        f"{means[median]:,.1f}, a {gap * 100:+.1f}% gap, "
                        f"seen across {n:,} rows). "
                        f"This is observed concentration, so review "
                        f"what is linked before acting."),
            "action_type": "investigate", "scenario_label": "segment-focus",
            "decision_type": "insight",
            "confidence": conf, "risk": "low",
            "risk_reasons": ["No intervention is modeled — this only directs attention."],
            "score": round(min(55.0, share * 100) + min(20.0, abs(gap) * 100)
                           + min(10.0, n / 50), 1),
            "score_breakdown": {"concentration": round(min(30.0, share * 100), 1),
                                "observed_gap": round(min(20.0, abs(gap) * 100), 1),
                                "data_quality": round(min(10.0, n / 50), 1)},
            "evidence": {"metric": metric, "driver": dim, "table": table,
                         "effect_gap_pct": round(gap * 100, 1),
                         "share_pct": share_pct, "p_value": p, "n": n,
                         "groups": [{"label": g, "mean": round(mn, 2), "n": cn}
                                    for g, mn, cn in segs],
                         "top": top, "median": median, "segment": None},
            "success_metric": {"metric": metric,
                               "direction": f"understand {top} concentration",
                               "target_pct": None,
                               "baseline": round(means[top], 2),
                               "note": "No modeled target yet — set one once review names the driver."},
            "review_period": review,
            "review_basis": review_basis,
            # Impact mirrors the OBSERVED top-vs-median gap (same precedent as
            # investigate cards) — measured group means, labeled OBSERVED, NOT
            # a modeled intervention effect. The honest note below carries the
            # unestimable-intervention disclaimer, so no impact is invented.
            "impact": {"baseline": round(means[median], 2),
                       "scenario": round(means[top], 2),
                       "delta": round(means[top] - means[median], 2),
                       "rel_pct": round(gap * 100, 1),
                       "range": [round(means[median], 2), round(means[top], 2)],
                       "driver_pct": 0, "driver_value": None,
                       "labels": {"baseline": "OBSERVED", "scenario": "OBSERVED"}},
            "impact_note": HONEST_IMPACT_NOTE,
            "assumptions": ["Groups observed as-is; mix and other factors not separated."],
            "limitations": [
                HONEST_IMPACT_NOTE,
                f"Concentration computed over {n} rows; smaller groups "
                f"({min(len(v) for v in groups.values())} rows) are noisier.",
                "A concentration shows where to look — it does not say what to change.",
            ],
            "recommended_next_analysis": {
                "type": "hypothesis",
                "label": f"Test what drives {m_h} concentration in {top} ({d_h})."},
        })
    out.sort(key=lambda c: c["score"], reverse=True)
    return out[:2]


def _monitor_cands(engine, dialect, intel, table, metric, m_h,
                   review=None, review_basis=None):
    """Monitor prescriptions: strong trend momentum, no causal claim.

    Evidence gates: >= MONITOR_MIN_POINTS (6) ordered points over a date-like
    column with |last - first| / |mean| >= MONITOR_MIN_MOMENTUM (15%).
    The card names metric + direction only ("Monitor Revenue — up 22% over 12
    periods") and carries HONEST_IMPACT_NOTE instead of a sized impact, which
    the engine-level validator requires. Never claims the movement will
    continue or names a cause.
    """
    from core.semantic import humanize
    dts = _date_like_cols(intel, table)[:1]
    if not dts:
        return []
    try:
        from core.schema_intel import _safe_name
        rows = _exec(engine,
                     f"SELECT {_safe_name(dts[0], dialect)}, {_safe_name(metric, dialect)} "
                     f"FROM {_safe_name(table, dialect)} "
                     f"WHERE {_safe_name(dts[0], dialect)} IS NOT NULL "
                     f"AND {_safe_name(metric, dialect)} IS NOT NULL "
                     f"LIMIT {int(SAMPLE_LIMIT)}")
    except Exception:
        return []
    import pandas as pd
    ts = pd.to_datetime([r[0] for r in rows], errors="coerce")
    vals = pd.Series(pd.to_numeric([r[1] for r in rows], errors="coerce"))
    ok = ts.notna() & vals.notna()
    if int(ok.sum()) < MONITOR_MIN_POINTS:
        return []
    frame = pd.DataFrame({"t": ts[ok], "v": vals[ok]}).sort_values("t")
    vs = frame["v"].to_numpy(dtype=float)
    if len(vs) < MONITOR_MIN_POINTS:
        return []
    first, last = float(vs[0]), float(vs[-1])
    mean = float(vs.mean()) if len(vs) else 0.0
    if not mean:
        return []
    momentum = (last - first) / abs(mean)
    if abs(momentum) < MONITOR_MIN_MOMENTUM:
        return []
    direction = "up" if momentum > 0 else "down"
    pct = abs(momentum) * 100
    n = len(vs)
    conf = "moderate" if n >= 12 else "low"
    _ = humanize  # (titles use m_h, passed in humanized)
    return [{
        "id": f"rxmon_{table}_{metric}",
        "rank": 0,
        "title": f"Monitor {m_h} — {direction} {pct:.0f}% over {n} periods",
        "summary": (f"{m_h} moved {direction} {pct:.1f}% from {first:,.1f} to {last:,.1f} "
                    f"across {n} ordered periods (mean {mean:,.1f}). This is an observed "
                    f"movement only — no cause is established and continuation is not "
                    f"promised, so keep watching before acting."),
        "action_type": "monitor", "scenario_label": "monitor",
        "decision_type": "insight",
        "confidence": conf, "risk": "low",
        "risk_reasons": ["No intervention is modeled — this only tracks momentum."],
        "score": round(min(40.0, pct) + min(10.0, n / 10), 1),
        "score_breakdown": {"momentum": round(min(30.0, pct), 1),
                            "data_quality": round(min(10.0, n / 10), 1)},
        "evidence": {"metric": metric, "table": table,
                     "n_points": n, "n": n,
                     "momentum_pct": round(momentum * 100, 1),
                     "direction": direction,
                     "first": round(first, 2), "last": round(last, 2),
                     "mean": round(mean, 2), "segment": None},
        "success_metric": {"metric": metric,
                           "direction": f"confirm whether the {direction}ward movement persists",
                           "target_pct": None,
                           "baseline": round(last, 2),
                           "note": "Watch-list only — set a target once a driver is established."},
        "review_period": review,
        "review_basis": review_basis,
        "impact": {"baseline": round(last, 2), "scenario": None, "delta": None,
                   "rel_pct": None, "range": [None, None],
                   "driver_pct": None, "driver_value": None,
                   "labels": {"baseline": "OBSERVED", "scenario": "NOT MODELED"}},
        "impact_note": HONEST_IMPACT_NOTE,
        "assumptions": ["Ordered by the data's own date column; sampling is as-stored."],
        "limitations": [
            HONEST_IMPACT_NOTE,
            f"Momentum computed over {n} ordered points — short histories are noisy.",
            "A movement shows what happened, not why, and not what happens next.",
        ],
        "recommended_next_analysis": {
            "type": "hypothesis",
            "label": f"Test what is associated with the {direction}ward movement in {m_h}."},
    }]


_HONEST_NOTE_MARKERS = ("could not be reliably estimated", "not modeled",
                         "no causal claim", "association only",
                         "does not promise", "not a promise")


def _monthly_buckets(engine, dialect, intel, table, metric):
    """Monthly (year, month) -> total for metric, dialect-free via pandas.

    Returns ([(y, m, total)], date_col) or ([], None). Buckets with zero
    rows are omitted (never fabricated); caller decides sufficiency.
    """
    dts = _date_like_cols(intel, table)[:1]
    if not dts:
        return [], None
    try:
        from core.schema_intel import _safe_name
        rows = _exec(engine,
                     f"SELECT {_safe_name(dts[0], dialect)}, {_safe_name(metric, dialect)} "
                     f"FROM {_safe_name(table, dialect)} "
                     f"WHERE {_safe_name(dts[0], dialect)} IS NOT NULL "
                     f"AND {_safe_name(metric, dialect)} IS NOT NULL "
                     f"LIMIT {int(SAMPLE_LIMIT)}")
    except Exception:
        return [], None
    import pandas as pd
    ts = pd.to_datetime([r[0] for r in rows], errors="coerce")
    vals = pd.Series(pd.to_numeric([r[1] for r in rows], errors="coerce"))
    ok = ts.notna() & vals.notna()
    if int(ok.sum()) < MIN_N:
        return [], dts[0]
    frame = pd.DataFrame({"t": ts[ok], "v": vals[ok]})
    frame["ym"] = frame["t"].dt.to_period("M")
    sums = frame.groupby("ym")["v"].sum().sort_index()
    return [(_p.year, _p.month, float(_v)) for _p, _v in sums.items()], dts[0]


def inaction_cost(engine, dialect, intel, table, metric, m_h=None):
    """Cost of doing nothing: project the recent monthly run-rate forward.

    Uses least-squares slope over up to the last 6 monthly buckets. Returns
    a dict with monthly_burn (positive = losing that much per month),
    direction, months_observed, and an honest note. If the trend is flat/up
    or history is short, monthly_burn is 0 with the reason stated — never
    a scary invented number.
    """
    from core.semantic import humanize as _hh
    m_h = m_h or _hh(metric)
    buckets, dcol = _monthly_buckets(engine, dialect, intel, table, metric)
    if len(buckets) < 3:
        return {"ok": True, "metric": metric, "table": table,
                "monthly_burn": 0.0, "direction": "unknown",
                "months_observed": len(buckets),
                "headline": f"Not enough monthly history to price inaction for {m_h.lower()}.",
                "note": "Needs at least 3 months of dated records."}
    import numpy as np
    recent = buckets[-6:]
    xs = np.arange(len(recent), dtype=float)
    ys = np.array([b[2] for b in recent], dtype=float)
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(recent) >= 2 else 0.0
    mean = float(ys.mean()) if len(ys) else 0.0
    burn = round(-slope, 2) if slope < 0 else 0.0
    direction = "down" if slope < 0 else ("up" if slope > 0 else "flat")
    rel = abs(slope) / abs(mean) * 100 if mean else 0.0
    if burn > 0:
        headline = (f"If nothing changes, {m_h.lower()} is drifting down "
                    f"about {burn:,.0f} per month on the recent run-rate.")
    else:
        headline = (f"No monthly bleed detected for {m_h.lower()} — "
                    f"recent run-rate is {direction}.")
    return {"ok": True, "metric": metric, "table": table,
            "monthly_burn": burn, "monthly_slope": round(slope, 2),
            "monthly_pct": round(rel, 1), "direction": direction,
            "months_observed": len(buckets),
            "window": {"from": f"{recent[0][0]}-{recent[0][1]:02d}",
                       "to": f"{recent[-1][0]}-{recent[-1][1]:02d}"},
            "headline": headline,
            "display": {"action_text": headline,
                        "outcome_text": (f"~{burn:,.0f}/month at stake" if burn > 0 else "No bleed detected"),
                        "confidence_label": "Moderate confidence" if len(buckets) >= 6 else "Early signal",
                        "why_summary": (f"Straight-line run-rate over the last {len(recent)} months. "
                                        "Trend holding steady is assumed — a shift in the business breaks this.")},
            "note": ("Straight-line projection of the recent monthly run-rate; "
                     "assumes the current mix holds. Not a forecast.")}


def detect_alerts(engine, dialect, intel, table, measures=None, max_alerts=5):
    """Alerts that matter: latest month vs prior month per measure.

    Threshold is data-driven: |change| must exceed max(2x stdev of past
    month-to-month deltas, 5%). Returns list of {metric, headline,
    change_pct, direction, severity, months_observed}. Empty list (not an
    error) when nothing moved unusually.
    """
    from core.semantic import humanize as _hh
    import numpy as np
    measures = measures or _usable_measures(intel, table)[:6]
    alerts = []
    for metric in measures:
        buckets, _ = _monthly_buckets(engine, dialect, intel, table, metric)
        if len(buckets) < 4:
            continue
        vals = np.array([b[2] for b in buckets], dtype=float)
        deltas = np.diff(vals)
        prev = vals[-2]
        if not prev:
            continue
        change = (vals[-1] - prev) / abs(prev) * 100
        scale = float(np.std(deltas[:-1])) if len(deltas) > 1 else 0.0
        thresh = max(2 * (scale / abs(prev) * 100 if prev else 0.0), 5.0)
        if abs(change) < thresh:
            continue
        direction = "down" if change < 0 else "up"
        m_h = _hh(metric)
        ym = buckets[-1]
        alerts.append({
            "metric": metric, "table": table,
            "headline": (f"{m_h} moved {direction} {abs(change):.0f}% in "
                         f"{ym[0]}-{ym[1]:02d} vs the prior month."),
            "change_pct": round(float(change), 1), "direction": direction,
            "severity": "high" if abs(change) >= 2 * thresh else "watch",
            "threshold_pct": round(float(thresh), 1),
            "months_observed": len(buckets),
            "display": {"action_text": f"Look at {m_h.lower()} this week.",
                        "outcome_text": f"{abs(change):.0f}% {direction} month-on-month",
                        "confidence_label": "Early signal",
                        "why_summary": (f" latest month broke the usual month-to-month wobble "
                                        f"(±{thresh:.0f}%). Worth a look, not proof of anything.")},
        })
    alerts.sort(key=lambda a: abs(a["change_pct"]), reverse=True)
    return alerts[:max_alerts]


def _validate_prescriptions(recs: list) -> tuple:
    """Engine-level validation: every emitted prescription must carry
    non-empty evidence AND (a quantified impact OR an explicit honest note).

    Returns (kept, dropped_count). Anything failing is dropped — never
    repaired with invented numbers — and the drop count rides in the
    response diagnostics so honesty stays observable.
    """
    kept, dropped = [], 0
    for r in (recs or []):
        if not isinstance(r, dict):
            dropped += 1
            continue
        ev = r.get("evidence")
        if not isinstance(ev, dict) or not ev:
            dropped += 1
            continue
        if not (ev.get("metric") or ev.get("document")):
            dropped += 1
            continue
        imp = r.get("impact") if isinstance(r.get("impact"), dict) else {}
        quantified = any(imp.get(k) is not None
                         for k in ("baseline", "scenario", "delta", "rel_pct"))
        hay = " ".join([
            str(r.get("impact_note") or ""),
            str((imp.get("note") if isinstance(imp, dict) else "") or ""),
            *[str(x) for x in (r.get("limitations") or [])],
            *[str(x) for x in (r.get("assumptions") or [])],
        ]).lower()
        honest = any(m in hay for m in _HONEST_NOTE_MARKERS)
        if not (quantified or honest):
            dropped += 1
            continue
        kept.append(r)
    return kept, dropped


def _rank_coverage(recs: list) -> list:
    """Rank: scenario > segment-focus > monitor > investigate; score within
    type. Segment-focus rides action_type 'investigate' with scenario_label
    'segment-focus', so it is keyed on the label, not the type."""
    def _tier(r):
        r = r if isinstance(r, dict) else {}
        if str(r.get("action_type") or "").lower() == "scenario":
            return 0
        if str(r.get("scenario_label") or "").lower() == "segment-focus":
            return 1
        if str(r.get("action_type") or "").lower() == "monitor":
            return 2
        return 3

    def _score(r):
        try:
            return float((r or {}).get("priority_score", (r or {}).get("score", 0)) or 0)
        except (TypeError, ValueError):
            return 0.0
    return sorted(recs or [], key=lambda r: (_tier(r), -_score(r)))


def _scenario_meta_for_fit(driver_used, fit, move):
    """Backend-declared supported range for What-If sliders (never hardcoded).

    The in-range pct edges come from THIS fit's observed driver range, capped
    by the ±50% refusal boundary (safety rails, not findings). The frontend
    slider must use scenario_meta.supported_pct — never a hardcoded ±30%.
    """
    try:
        mean = abs(float(fit["x_mean"])) or 0.0
    except (TypeError, ValueError, KeyError):
        return None
    if not mean:
        return None
    try:
        up_edge = (float(fit["x_max"]) - float(fit["x_mean"])) / mean * 100
        down_edge = (float(fit["x_mean"]) - float(fit["x_min"])) / mean * 100
    except (TypeError, ValueError, KeyError):
        return None
    up_edge = max(0.0, min(50.0, up_edge if up_edge == up_edge else 0.0))
    down_edge = max(0.0, min(50.0, down_edge if down_edge == down_edge else 0.0))
    return {"driver": driver_used,
            "supported_pct": {"min": -round(down_edge, 1),
                              "max": round(up_edge, 1)},
            "hard_limit_pct": 50,
            "observed": {"min": fit["x_min"], "max": fit["x_max"],
                         "mean": round(fit["x_mean"], 4)},
            "improve_direction": "increase" if move > 0 else "decrease"}


def _coverage_response(items, metric, table, m_h, priors, tested=None):
    """Completed response with NO scenario decision (no adjustable driver
    cleared the bar) but real segment-focus / monitor / investigate evidence.

    Items are deduped, validated (evidence + quantified-impact-or-honest-note)
    and capped at 5; the drop count rides in diagnostics. Refusal is preserved
    when nothing survives validation — the caller falls back to
    no_prescription.
    """
    pres = _rank_coverage(dedupe_prescriptions(items or []))
    pres, dropped = _validate_prescriptions(pres)
    pres = pres[:5]
    # NOTE: display/details/refusal are attached downstream by analyze()
    # via _finalize_output — nothing is attached here.
    if not pres:
        return {"ok": True, "success": True, "status": "no_prescription",
                "reason": (f"No adjustable driver for {_disp(metric)} cleared the evidence bar, "
                           f"and no segment or trend pattern met the attention gates either."),
                "alternatives": ["Investigate Driver", "Test Hypothesis", "Explore Data", "Ask ARIA"],
                "tested": [dict(t) for t in (tested or [])][:12],
                "metric": metric, "table": table,
                "diagnostics": {"dropped_invalid": dropped, "emitted": 0}}
    for i, c in enumerate(pres):
        c["rank"] = i + 1
        # First-class honest prescription marker (additive): this card
        # directs attention with plain wording and gated evidence — it is a
        # real prescription, not a refusal side-path.
        c["coverage"] = True
    top = pres[0]
    from core.semantic import humanize as _hh
    _top_ev = top.get("evidence", {}) if isinstance(top.get("evidence"), dict) else {}
    _top_label = str(top.get("scenario_label") or "").lower()
    if _top_label == "segment-focus":
        story = (f"No adjustable driver for {m_h} cleared the evidence bar, but "
                 f"{_top_ev.get('top')} holds {_top_ev.get('share_pct')}% of observed {m_h} rows. "
                 f"Review is warranted before any intervention.")
    elif str(top.get("action_type") or "").lower() == "monitor":
        story = (f"No adjustable driver for {m_h} cleared the evidence bar, but "
                 f"{m_h} shows strong momentum ({_top_ev.get('momentum_pct'):+.1f}% over "
                 f"{_top_ev.get('n_points')} periods). "
                 f"Monitoring is warranted before any intervention.")
    else:
        story = (f"No adjustable driver for {m_h} cleared the evidence bar, but "
                 f"{m_h} differs meaningfully across {_hh(_top_ev.get('driver', ''))} "
                 f"({_top_ev.get('effect_gap_pct', 0.0):+.1f}% gap). "
                 f"Investigation is warranted before any intervention.")
    _viz = _groups_bar_viz(top, m_h)
    return {"ok": True, "success": True, "status": "completed",
            "analysis_id": f"rxa_{table}_{metric}_coverage",
            "coverage": True,
            "prescription_class": "attention",
            "metric": metric, "driver": _top_ev.get("driver"),
            "table": table, "driver_table": table, "join": None, "join_ref": None,
            "control": {"level": "investigation",
                        "reason": "No intervention modeled — evidence directs attention."},
            "auto_note": None, "fit": None, "evidence_viz": _viz,
            "prescriptions": pres,
            "scenario_meta": None,
            "diagnostics": {"dropped_invalid": dropped,
                            "emitted": len(pres),
                            "tested": [dict(t) for t in (tested or [])][:12]},
            "signals": {"supporting": 1,
                        "priors_origin": (priors or {}).get("origin")},
            "story": story,
            "summary_plain": _plain_summary(m_h, _top_ev.get("driver") or m_h, None, top)}


def _coverage_or_refusal(engine, dialect, intel, table, metric, m_h, priors,
                         tested=None, reason="", evidence=None, review=None,
                         review_basis=None, driver=None):
    """Coverage fallback as a first-class honest prescription.

    Tries the gated investigate / segment-focus / monitor candidates first
    (each keeps its own evidence gates — share/gap/n and momentum/points —
    plus an explicit honest note; thresholds are untouched). When at least
    one candidate clears its gates, the coverage response below is returned
    as a completed, plain-worded attention prescription instead of a bare
    refusal. Only when nothing clears the gates does this fall back to the
    honest no_prescription refusal. Additive: API shapes are unchanged apart
    from the coverage flag carried by _coverage_response.
    """
    try:
        inv = _investigate_cands(engine, dialect, intel, table, metric, m_h,
                                 review=review, review_basis=review_basis)
    except Exception:
        inv = []
    try:
        seg = _segment_focus_cands(engine, dialect, intel, table, metric, m_h,
                                   review=review, review_basis=review_basis)
    except Exception:
        seg = []
    try:
        mon = _monitor_cands(engine, dialect, intel, table, metric, m_h,
                             review=review, review_basis=review_basis)
    except Exception:
        mon = []
    if seg or mon or inv:
        return _coverage_response(seg + mon + inv, metric, table,
                                  m_h, priors, tested=tested)
    out: dict = {"ok": True, "success": True, "status": "no_prescription",
                 "reason": reason,
                 "alternatives": ["Investigate Driver", "Test Hypothesis", "Explore Data", "Ask ARIA"],
                 "tested": [dict(t) for t in (tested or [])][:12],
                 "metric": metric, "table": table}
    if driver is not None:
        out["driver"] = driver
    if evidence is not None:
        out["evidence"] = evidence
    return out


def _groups_bar_viz(top, m_h):
    """Chart-ready bar spec from a prescription's group evidence (or None)."""
    from core.semantic import humanize as _hh2
    try:
        _groups = (top.get("evidence", {}) or {}).get("groups") or []
        _rows = []
        for g in _groups:
            if not isinstance(g, dict):
                continue
            if g.get("label") is None or g.get("mean") is None:
                continue
            try:
                _val = float(g.get("mean"))
            except (TypeError, ValueError):
                continue
            if _val != _val:  # NaN-safe
                continue
            _rows.append({"name": str(g.get("label")), "value": _val})
        if len(_rows) >= 2:
            _drv = (top.get("evidence", {}) or {}).get("driver", "")
            _title = f"{m_h} by {_hh2(_drv)}"
            try:
                from services.insights_service import _normalize_chart_spec as _norm
                _spec = _norm("bar", _rows[:200], _title)
                if isinstance(_spec, dict) and not _spec.get("unsupported") \
                        and not _spec.get("insufficient") and _spec.get("data"):
                    return {"type": "bar", "data": _spec["data"],
                            "title": _spec.get("title") or _title,
                            "x": _spec.get("x") or "name",
                            "y": _spec.get("y") or "value"}
                return {"type": "bar", "data": _rows[:200],
                        "title": _title, "x": "name", "y": "value"}
            except Exception:
                return {"type": "bar", "data": _rows[:200],
                        "title": _title, "x": "name", "y": "value"}
    except (TypeError, ValueError):
        pass
    return None


def _investigate_response(inv, metric, table, m_h, priors):
    """Completed response carrying only investigate prescriptions (no
    scenarios exist, but real segment evidence does).

    Thin wrapper over _coverage_response so investigate-only results share
    the dedupe → validate → cap-5 → diagnostics pipeline; the analysis_id
    keeps its legacy `_investigate` suffix.
    """
    out = _coverage_response(inv, metric, table, m_h, priors)
    if out.get("status") == "completed":
        out["analysis_id"] = f"rxa_{table}_{metric}_investigate"
    return out


def _date_like_cols(intel, table):
    """Datetime columns plus date-hinted text columns (CSV dates arrive as
    TEXT in SQLite — same fallback rule as the schema layer)."""
    cols = list(intel.get_datetime_cols(table) or [])
    for c in intel.get_text_cols(table) or []:
        low = c.lower()
        if c not in cols and any(h in low for h in ("date", "time", "day", "month",
                                                   "year", "created", "updated",
                                                   "timestamp", "period")):
            cols.append(c)
    return cols


def _review_period(engine, dialect, table, intel):
    """Review cadence from the data's own time span — never a guess.

    Returns (period_label, basis). No date column → one business cycle.
    """
    dts = (intel.get_datetime_cols(table) or [])[:1]
    if not dts:
        for c in intel.get_text_cols(table):
            low = c.lower()
            if any(h in low for h in ("date", "time", "day", "month", "year", "period")):
                dts = [c]
                break
    if dts:
        try:
            from core.schema_intel import _safe_name
            rows = _exec(engine,
                         f"SELECT MIN({_safe_name(dts[0], dialect)}), "
                         f"MAX({_safe_name(dts[0], dialect)}) "
                         f"FROM {_safe_name(table, dialect)}")
            if rows and rows[0][0] is not None and rows[0][1] is not None:
                import pandas as pd
                lo = pd.to_datetime(rows[0][0], errors="coerce")
                hi = pd.to_datetime(rows[0][1], errors="coerce")
                if pd.notna(lo) and pd.notna(hi):
                    span = max(0, (hi - lo).days)
                    if span <= 14:
                        return "2 weeks", f"data spans {span} days"
                    if span <= 62:
                        return "4 weeks", f"data spans {span} days"
                    if span <= 200:
                        return "one quarter", f"data spans {span} days"
                    return "6 months", f"data spans {span} days"
        except Exception:
            pass
    return "one full business cycle", "no usable time span in this data"


def _col_kind(intel, table, col):
    return intel._col_category.get(table, {}).get(col, "")


def _qi(name, dialect="postgresql"):
    """Quote one identifier part (table aliases qualify every column, so
    shared names like room_id never go ambiguous)."""
    if dialect == "mysql":
        return "`" + str(name).replace("`", "``") + "`"
    return '"' + str(name).replace('"', '""') + '"'


def _fetch_joined(engine, dialect, mtable, mcol, otable, ocol, self_col, other_col,
                  limit=SAMPLE_LIMIT):
    """Read-only joined driver/metric pairs across one declared edge.

    Table names may be schema-qualified (sales.orders on multi-schema
    databases): they are quoted per dot-segment ("sales"."orders") via
    the shared _safe_name helper. Column references stay bare and use
    the local _qi quoter.
    """
    try:
        from core.schema_intel import _safe_name as _qn
    except Exception:
        _qn = None

    def _qt(name):
        try:
            if _qn is not None:
                return _qn(name, dialect)
        except Exception:
            pass
        return _qi(name, dialect)

    mt, ot = _qt(mtable), _qt(otable)
    return _exec(engine,
                 f"SELECT o.{_qi(ocol, dialect)}, m.{_qi(mcol, dialect)} "
                 f"FROM {mt} m JOIN {ot} o ON "
                 f"m.{_qi(self_col, dialect)} = o.{_qi(other_col, dialect)} "
                 f"WHERE o.{_qi(ocol, dialect)} IS NOT NULL "
                 f"AND m.{_qi(mcol, dialect)} IS NOT NULL "
                 f"ORDER BY 1, 2 "
                 f"LIMIT {int(limit)}")


def _segment_extreme(engine, dialect, table, dim, metric):
    """Weakest group by mean(metric) — headroom target. Returns (value, mean)."""
    rows = _fetch_pairs(engine, dialect, table, dim, metric, limit=SAMPLE_LIMIT)
    groups = {}
    for g, v in rows:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        groups.setdefault(str(g), []).append(fv)
    groups = {g: vs for g, vs in groups.items() if len(vs) >= 5}
    if len(groups) < 2:
        return None
    means = {g: sum(vs) / len(vs) for g, vs in groups.items()}
    weak = min(means, key=means.get)
    return {"field": dim, "value": weak, "mean": means[weak],
            "overall": sum(means.values()) / len(means)}


def _dynamic_limitations(fit, scen, targeted, driver_h):
    """Limitations computed from THIS evidence — never boilerplate (§20).

    Plain-language only: no r/p/|r|/Spearman/slope numbers and no
    observed-range min-max here — those live in details.numbers/evidence
    (analyst expander). The single association-not-proof disclaimer lives
    in display.why_text, not repeated per sentence here.
    """
    lim = []
    try:
        n = int(fit.get("n", 0))
    except (TypeError, ValueError):
        n = 0
    if n < 30:
        lim.append(f"Small sample: only {n} usable observations — treat every estimate as rough.")
    elif n < 100:
        lim.append(f"Modest sample of {n} observations — direction is informative, exact sizes less so.")
    try:
        _r = abs(float(fit.get("r", 0) or 0))
    except (TypeError, ValueError):
        _r = 0.0
    if _r < 0.5:
        lim.append(f"Relationship strength is moderate — {driver_h} explains only part of the movement.")
    try:
        _rr = float(fit.get("r", 0) or 0)
        _rs = float(fit.get("spearman_r", 0) or 0)
        _rank_bad = abs(_rr - _rs) > 0.25 or (_rr > 0) != (_rs > 0)
    except (TypeError, ValueError):
        _rank_bad = False
    if _rank_bad:
        lim.append("A rank-based check disagrees — a few unusual rows may drive the result.")
    if not scen.get("in_historical_range"):
        lim.append("This scenario moves the driver outside what was observed — treat it as extrapolation.")
    if targeted:
        lim.append(f"Targeted at '{targeted['value']}' ({targeted['field']}) — segment estimates rest on fewer rows than the full table.")
    return lim


def _strength_word(r) -> str:
    """Plain strength word for user-visible text (Strong/Moderate).

    Thresholds reuse the engine's own bars: |r| >= 0.7 is Strong,
    anything clearing MIN_ABS_R is Moderate. No numbers are emitted.
    """
    try:
        a = abs(float(r))
    except (TypeError, ValueError):
        return "Moderate"
    return "Strong" if a >= 0.7 else "Moderate"


def _plain_summary(m_h, d_h, fit, rec, join_phrase=None):
    """Plain-English summary of the whole analysis — a normal person reads
    only this and understands what was tested, found, and recommended.

    Plain-language contract: no r=/p=/|r|/Spearman/slope numbers and no
    observed-range min-max — those live in details.numbers/evidence only.
    """
    if str(rec.get("action_type") or "").lower() == "monitor":
        ev = rec.get("evidence", {}) if isinstance(rec.get("evidence"), dict) else {}
        return (
            f"We tracked {m_h.lower()} across {ev.get('n_points', 0)} ordered periods and found "
            f"strong momentum ({ev.get('direction', '?')} {ev.get('momentum_pct', 0.0):+.1f}%). "
            f"Our recommendation: {rec['title']}. This is an observed movement only — "
            f"no cause is established, so keep watching before acting."
        )
    if rec.get("action_type") == "investigate":
        ev = rec.get("evidence", {})
        gs = ev.get("groups", [])
        top_mean = gs[0]["mean"] if gs else "?"
        bot_mean = gs[-1]["mean"] if gs else "?"
        return (
            f"We looked for an adjustable driver for {m_h.lower()} across "
            f"{ev.get('n', 0):,} rows but found no strong one. What we did find: "
            f"{ev.get('top')} averages {top_mean} vs "
            f"{ev.get('bottom')} at {bot_mean} "
            f"— a {ev.get('effect_gap_pct'):+.1f}% gap across {ev.get('driver')}. "
            f"Our recommendation: {rec['title']}. Start by finding what is "
            f"associated with the difference — a gap shows where to look, not what to change."
        )
    try:
        _slope = float((fit or {}).get("slope", 1) or 0)
    except (TypeError, ValueError):
        _slope = 1.0
    direction = "go up together" if _slope > 0 else "move in opposite directions"
    strength = _strength_word((fit or {}).get("r")).lower()
    r_imp = rec["impact"] if isinstance(rec.get("impact"), dict) else {}
    m, d = m_h.lower(), d_h.lower()
    try:
        _n = int((fit or {}).get("n", 0) or 0)
    except (TypeError, ValueError):
        _n = 0
    _conf_label = confidence_label_for(rec.get("confidence"))
    return (
        f"We tested whether {d} is linked with {m}, using {_n:,} rows of your data. "
        f"The two {direction} ({strength} link)"
        + (f", measured {join_phrase}" if join_phrase else "")
        + ". "
        f"Our recommendation: {rec['title']}: "
        f"{m} would go from about {r_imp.get('baseline', '?'):,.1f} to about {r_imp.get('scenario', '?'):,.1f} "
        f"({r_imp.get('rel_pct', 0.0):+.1f}%). "
        f"Confidence is {_conf_label}, risk is {rec.get('risk')} — past patterns show "
        f"an association, not proof the change would cause the result."
    )



def _analyze_core(engine, dialect, intel, table, metric, driver=None,
            dimensions=None, segment=None, question=None, priors=None):
    """Full prescription pipeline. Returns ranked prescriptions or a
    no_prescription refusal (§23) — never an invented action."""
    from core.semantic import humanize
    priors = priors or {}
    # Diagnostic trail for the whole run (populated during candidate
    # testing below; empty on early validation exits — still honest).
    tested: list = []

    # 1) Validate metric (§34/§35: schema-discovered, never assumed).
    # hyp.txt §10-11: the metric must be a genuine business outcome,
    # determined semantically (never hardcoded lever→outcome mappings). A
    # technical/system measurement is analytic signal — strong correlations
    # stay evidence, but no actionable recommendation is generated.
    measures = _usable_measures(intel, table)
    if metric not in measures:
        if metric in intel.get_numeric_cols(table):
            _mpks = {p.lower() for p in intel.get_pk(table)}
            _mfks = {f[0].lower() for f in intel.get_fks(table)}
            _mrole = classify_semantic_role(
                metric, "numeric", is_pk=(metric.lower() in _mpks),
                is_fk=(metric.lower() in _mfks),
                is_join_key=(metric.lower() in _join_key_cols(intel, table)))
            if _mrole in NON_OUTCOME_ROLES:
                return {"ok": True, "success": True, "status": "no_prescription",
                        "reason": (f"{_disp(metric)} is a {_mrole.lower().replace('_', ' ')} — useful for "
                                   f"analysis, but ARIA does not turn associations involving technical/system "
                                   f"measurements into actionable recommendations. Association is evidence of "
                                   f"association, not evidence of an actionable driver."),
                        "alternatives": ["Investigate Driver", "Test Hypothesis", "Explore Data", "Ask ARIA"],
                "tested": [dict(t) for t in tested][:12],
                        "metric": metric, "table": table}
        return _fail(CODE_INVALID_METRIC,
                     f"ARIA couldn't find a measurable '{metric}' in this data.",
                     available_metrics=[{"table": table, "column": c} for c in measures[:8]])
    dims = [d for d in (dimensions or [])
            if d in (intel.get_text_cols(table) + intel.get_datetime_cols(table)
                     + intel.get_numeric_cols(table))]
    # Review cadence from the data's own time span (one bounded query).
    review, review_basis = _review_period(engine, dialect, table, intel)

    # 2) Driver: explicit > prior > auto-strongest-evidence (§7).
    # Candidates are (column, join) with join=None for same-table or
    # (other_table, self_col, other_col) across ONE declared FK edge.
    # Guards fail loudly with the reason — never a silent wrong-pair analysis.
    if driver and driver == metric:
        return _fail(CODE_INVALID_DRIVER,
                     f"'{metric}' can't be both the target and the driver. Pick a different driver.")
    # hyp.txt §§4-6: an explicitly requested driver still needs semantic
    # actionability. A correlation with a technical/system field is supporting
    # evidence ONLY — it never becomes a "Test X% increase" lever, no matter
    # how strong the correlation is.
    if driver and driver != "length_of_stay":
        _dpks = {p.lower() for p in intel.get_pk(table)}
        _dfks = {f[0].lower() for f in intel.get_fks(table)}
        _dkeys = _join_key_cols(intel, table)
        _dkind = _col_kind(intel, table, driver)
        _drole = classify_semantic_role(
            driver, _dkind or "numeric", is_pk=(str(driver).lower() in _dpks),
            is_fk=(str(driver).lower() in _dfks),
            is_join_key=(str(driver).lower() in _dkeys))
        if _drole in NON_LEVER_ROLES:
            from core.semantic import humanize as _hg0
            _tested0 = ([{"driver": driver, "verdict": "skipped",
                          "note": f"{_disp(driver)} is a {_drole.lower().replace('_', ' ')} — categories and system fields cannot be adjusted as interventions."}]
                        + [dict(t) for t in tested])[:12]
            return _coverage_or_refusal(
                engine, dialect, intel, table, metric, _hg0(metric), priors,
                tested=_tested0,
                reason=(f"{_disp(driver)} is a {_drole.lower().replace('_', ' ')} — ARIA keeps the "
                        f"association with {_disp(metric)} as supporting evidence, but it cannot be "
                        f"moved as an intervention. Association is evidence of association, not evidence "
                        f"of an actionable driver."),
                review=review, review_basis=review_basis, driver=driver)
    neighbors = _edge_neighbors(intel, table)
    auto_note = None
    if driver == "length_of_stay":
        dts = _date_like_cols(intel, table)[:3]
        if len(dts) < 2:
            return _fail(CODE_INVALID_DRIVER,
                         "Stay length needs two date columns, which this table doesn't have.")
        nights, ys = _stay_lengths(engine, dialect, table, dts[0], dts[1], metric)
        if nights is None:
            return _fail(CODE_INSUFFICIENT_DATA,
                         "Not enough dated rows to measure stay length against "
                         f"{_disp(metric)}.")
        f = fit_relationship(nights, ys)
        if not (f and abs(f["r"]) >= MIN_ABS_R and f["p_value"] < 0.05):
            from core.semantic import humanize as _hg0s
            return _coverage_or_refusal(
                engine, dialect, intel, table, metric, _hg0s(metric), priors,
                tested=tested,
                reason=(f"ARIA measured stay length against {_disp(metric)} but the link "
                        f"is too weak to justify an intervention."),
                review=review, review_basis=review_basis)
        best, best_fit, best_xy, best_join = "length_of_stay", f, (nights, ys), None
        cands = []
    elif driver:
        table_cols = (intel.get_numeric_cols(table) + intel.get_text_cols(table)
                      + intel.get_datetime_cols(table))
        if driver in table_cols:
            cands = [(driver, None)]
        else:
            cands = []
            for ot, sc, oc in neighbors:
                if driver in _usable_measures(intel, ot):
                    cands = [(driver, (ot, sc, oc))]
                    break
            if not cands:
                elsewhere = [t for t in intel.tables if t != table and driver in (
                    intel.get_numeric_cols(t) + intel.get_text_cols(t)
                    + intel.get_datetime_cols(t))]
                if elsewhere:
                    return _fail(CODE_INVALID_DRIVER,
                                 f"'{driver}' lives in {elsewhere[0]}, which has no declared link to "
                                 f"{table}. ARIA only joins tables over declared relationships. "
                                 f"Usable drivers here: {', '.join(_usable_measures(intel, table)[:5]) or 'none'}.")
                return _fail(CODE_INVALID_DRIVER,
                             f"ARIA couldn't find a usable '{driver}' field in this data.")
    else:
        cands = []
        for c in intel.get_numeric_cols(table):
            if c != metric and c in measures:
                cands.append((c, None))
        # Cross-table levers over declared joins (e.g. rooms.price_per_night
        # for bookings.total_amt) — discovered, never assumed.
        for ot, sc, oc in neighbors:
            for c in _usable_measures(intel, ot)[:4]:
                cands.append((c, (ot, sc, oc)))
        for c in intel.find_all_dimensions(table)[:4]:
            if all(c != cc for cc, _ in cands):
                cands.append((c, None))
    if driver != "length_of_stay":
        best, best_fit, best_xy, best_join = None, None, None, None
    # Diagnostic trail: every candidate driver with its gate outcome.
    # Attached to refusals (and completions) so the UI can show what was
    # tested and why each item passed or failed — never a blank page.
    tested: list = []
    tried_stay = False
    if not driver:
        # Derived stay-length driver: days between the first two date columns
        # vs the metric (booking-style tables). Data-derived, never assumed.
        dts = _date_like_cols(intel, table)[:3]
        if len(dts) >= 2:
            nights, ys = _stay_lengths(engine, dialect, table, dts[0], dts[1], metric)
            tried_stay = nights is not None
            if nights:
                f = fit_relationship(nights, ys)
                if f and abs(f["r"]) >= MIN_ABS_R and f["p_value"] < 0.05:
                    best, best_fit, best_xy = "length_of_stay", f, (nights, ys)
    for c, join in cands:
        # Self-correlation guard: a driver with the same name as the metric
        # (even across a join, e.g. rooms.price vs bookings.price) correlates
        # ~1.0 with itself and would render as "X moves with X". Never a lever.
        if str(c).lower() == str(metric).lower():
            continue
        if join is None and _col_kind(intel, table, c) != "numeric":
            tested.append({"driver": c, "verdict": "skipped",
                           "note": "Text or category — it cannot be turned up or down, so it was not modeled."})
            continue
        try:
            if join is None:
                rows = _fetch_pairs(engine, dialect, table, c, metric)
            else:
                ot, sc, oc = join
                rows = _fetch_joined(engine, dialect, table, metric, ot, c, sc, oc)
            # TEXT-stored numerics (currency symbols, commas, percents)
            # parse via the shared text-numeric parser; unparsable pairs
            # are dropped instead of failing the whole driver.
            try:
                from core.text_numeric import try_parse_text_number as _tpn
            except Exception:
                _tpn = None
            if _tpn is not None:
                _pairs = [(_tpn(r[0]), _tpn(r[1])) for r in rows]
                _pairs = [(x, y) for x, y in _pairs
                          if x is not None and y is not None]
                if not _pairs:
                    raise ValueError("no parseable numeric pairs")
                xs = [p[0] for p in _pairs]
                ys = [p[1] for p in _pairs]
            else:
                xs = [float(r[0]) for r in rows]
                ys = [float(r[1]) for r in rows]
        except (TypeError, ValueError):
            tested.append({"driver": c, "verdict": "skipped",
                           "note": "Values could not be read as numbers."})
            continue
        except Exception:
            tested.append({"driver": c, "verdict": "skipped",
                           "note": "Data could not be loaded for this driver."})
            continue
        try:
            f = fit_relationship(xs, ys)
        except Exception:
            # Numerical failure (e.g. SVD non-convergence on extreme
            # scales) is a per-driver modeling miss, never a 500: the
            # driver is rejected and the remaining candidates still run.
            tested.append({"driver": c, "n": len(xs), "verdict": "rejected",
                           "note": "Could not be modeled reliably on the observed values."})
            continue
        # Perfect correlation is a self-join/constant-column artifact, not a
        # discovery (|r|>=0.999 cannot be a controllable lever).
        if f and abs(f["r"]) >= 0.999:
            tested.append({"driver": c, "n": f.get("n"), "verdict": "rejected",
                           "note": "Moves in lockstep with the metric — a self-match, not a lever."})
            continue
        if f and abs(f["r"]) >= MIN_ABS_R and f["p_value"] < 0.05:
            if best_fit is None or abs(f["r"]) > abs(best_fit["r"]):
                best, best_fit, best_xy, best_join = c, f, (xs, ys), join
        elif f:
            # Plain-language diagnostic: no r=/p=/|r| in the user-visible
            # note — raw stats stay in details.numbers/evidence only.
            _why = []
            if abs(f["r"]) < MIN_ABS_R:
                _why.append("link strength below the bar")
            if f["p_value"] >= 0.05:
                _why.append("not statistically solid")
            tested.append({"driver": c, "n": f.get("n"),
                           "verdict": "rejected",
                           "note": "; ".join(_why) + f" — measured on {f.get('n', 0)} rows."})
        else:
            tested.append({"driver": c, "verdict": "rejected",
                           "note": "Too few usable rows to test."})
    if best_fit is None:
        # No evidence-backed driver — refuse (§3/§23), don't invent.
        # Name exactly what was testable so the refusal is data-specific.
        try:
            fit_n_total = len(_fetch_pairs(engine, dialect, table, metric, metric))
        except Exception:
            fit_n_total = 0
        numeric_tested = sorted({(c if j is None else f"{c} ({j[0]})")
                                 for c, j in cands
                                 if (j is not None or _col_kind(intel, table, c) == "numeric")})
        if not numeric_tested:
            reason = (f"{_disp(metric)} is the only measurable column reachable from {table} — "
                      f"there are no other numbers in your data to test as drivers.")
        else:
            reason = (f"ARIA tested {', '.join(numeric_tested[:6])} against {_disp(metric)} "
                      f"({fit_n_total} usable rows) but found no link strong enough "
                      f"to justify an intervention.")
        from core.semantic import humanize as _hz
        inv = _investigate_cands(engine, dialect, intel, table, metric, _hz(metric),
                                 review=review, review_basis=review_basis)
        seg = _segment_focus_cands(engine, dialect, intel, table, metric, _hz(metric),
                                   review=review, review_basis=review_basis)
        mon = _monitor_cands(engine, dialect, intel, table, metric, _hz(metric),
                             review=review, review_basis=review_basis)
        if seg or mon or inv:
            return _coverage_response(seg + mon + inv, metric, table,
_hz(metric), priors, tested=tested)
        return {"ok": True, "success": True, "status": "no_prescription",
                "reason": reason,
                "alternatives": ["Investigate Driver", "Test Hypothesis", "Explore Data", "Ask ARIA"],
                "tested": [dict(t) for t in tested][:12],
                "metric": metric, "table": table}
    driver_used, fit = best, best_fit
    join_used = best_join
    join_label = (f"{table}.{join_used[1]} → {join_used[0]}.{join_used[2]}"
                  if join_used else None)
    driver_table = join_used[0] if join_used else table
    if not driver:
        auto_note = (f"ARIA selected {humanize(driver_used)} automatically as the strongest evidence-backed driver"
                     + (f" (measured {_join_phrase(join_used)})." if join_used else "."))
    # Controllability is judged on the driver wherever it lives.

    # 3) Controllability gate (§8) — typed by the DRIVER's own table,
    # which differs for cross-table drivers reached over a join. The derived
    # stay-length driver is numeric by construction.
    _driver_kind = ("numeric" if driver_used == "length_of_stay"
                    else _col_kind(intel, driver_table, driver_used))
    control_level, control_reason = classify_driver(driver_used, _driver_kind)
    if control_level in ("not_controllable", "segmentation"):
        from core.semantic import humanize as _hg
        inv0 = _investigate_cands(engine, dialect, intel, table, metric, _hg(metric),
                                  review=review, review_basis=review_basis)
        seg0 = _segment_focus_cands(engine, dialect, intel, table, metric, _hg(metric),
                                    review=review, review_basis=review_basis)
        mon0 = _monitor_cands(engine, dialect, intel, table, metric, _hg(metric),
                              review=review, review_basis=review_basis)
        if seg0 or mon0 or inv0:
            return _coverage_response(seg0 + mon0 + inv0, metric, table,
_hg(metric), priors, tested=tested)
        return {"ok": True, "success": True, "status": "no_prescription",
                "reason": (f"ARIA found a {control_level} pattern involving {_disp(driver_used)}, "
                           f"but the available data does not contain a controllable factor that can be "
                           f"reliably evaluated as an intervention. {control_reason}"),
                "alternatives": ["Investigate Driver", "Test Hypothesis", "Explore Data", "Ask ARIA"],
                "tested": [dict(t) for t in tested][:12],
                "metric": metric, "driver": driver_used, "table": table,
                "evidence": {"r": fit["r"], "p_value": fit["p_value"], "n": fit["n"]}}

    # 4) Improvement direction + candidates (§9, hyp.txt §12).
    # Directionality is validated, never generically assumed: unknown
    # direction → no recommendation. The metric gate above already refuses
    # technical/system outcomes; this is the structural backstop.
    up_better, up_assump = improvement_up(metric)
    if up_better is None:
        from core.semantic import humanize as _hg_dir
        return _coverage_or_refusal(
            engine, dialect, intel, table, metric, _hg_dir(metric), priors,
            tested=tested,
            reason=(f"ARIA cannot establish whether higher or lower values of {_disp(metric)} "
                    f"are desirable ({up_assump}). Without a justified direction, no actionable "
                    f"recommendation can be generated — the association stays supporting evidence only."),
            evidence={"r": fit["r"], "p_value": fit["p_value"], "n": fit["n"]},
            review=review, review_basis=review_basis, driver=driver_used)
    # Explicit fallback assumption (no recognized outcome semantics): the
    # recommendation is allowed for business-plausible metrics but carries
    # capped confidence — the assumption is recorded, never silent.
    _dir_fallback = up_assump.startswith("no outcome semantics")
    # Full actionability verdict (§6): correlation + controllability +
    # outcome + justified direction. Correlation alone never suffices.
    _metric_role = classify_semantic_role(metric, "numeric")
    _lever_role = classify_semantic_role(driver_used, _driver_kind or "numeric")
    _gate = assess_actionability(driver_used, _lever_role, control_level,
                                 metric_role=_metric_role,
                                 direction_known=(up_better is not None),
                                 has_evidence=True)
    if not _gate["actionable"]:
        from core.semantic import humanize as _hg2
        inv0b = _investigate_cands(engine, dialect, intel, table, metric, _hg2(metric),
                                   review=review, review_basis=review_basis)
        seg0b = _segment_focus_cands(engine, dialect, intel, table, metric, _hg2(metric),
                                     review=review, review_basis=review_basis)
        mon0b = _monitor_cands(engine, dialect, intel, table, metric, _hg2(metric),
                               review=review, review_basis=review_basis)
        if seg0b or mon0b or inv0b:
            return _coverage_response(seg0b + mon0b + inv0b, metric, table,
_hg2(metric), priors, tested=tested)
        return {"ok": True, "success": True, "status": "no_prescription",
                "reason": (f"{_disp(driver_used)} does not clear the actionability bar "
                           f"({'; '.join(_gate['notes']) or 'unmet conditions'}). The association with "
                           f"{_disp(metric)} stays supporting evidence only."),
                "alternatives": ["Investigate Driver", "Test Hypothesis", "Explore Data", "Ask ARIA"],
                "tested": [dict(t) for t in tested][:12],
                "metric": metric, "driver": driver_used, "table": table,
                "evidence": {"r": fit["r"], "p_value": fit["p_value"], "n": fit["n"]}}
    # Move the driver so the metric improves: sign = +1 means increase driver.
    move = 1 if ((fit["slope"] > 0) == up_better) else -1
    verb = "increase" if move > 0 else "reduction"
    # Preset sizes from the driver's OWN typical movement — never fixed
    # percentages. Sized as fractions of one standard deviation of the
    # observed driver values (0.25σ / 0.5σ / 1σ), clamped to the observed
    # in-range maximum so conservative+recommended stay interpolations
    # (green) and only stretch may extrapolate (amber). Absolute ceilings
    # mirror the ±50% refusal boundary; they are safety rails, not findings.
    _xs = np.asarray(best_xy[0], dtype=float) if best_xy else np.asarray([], dtype=float)
    _xmean = float(_xs.mean()) if len(_xs) else fit["x_mean"]
    _xstd = float(_xs.std()) if len(_xs) > 1 else 0.0
    if not _xmean or not _xstd:
        return _coverage_or_refusal(
            engine, dialect, intel, table, metric, humanize(metric), priors,
            tested=tested,
            reason=(f"Cannot compute scenario: driver '{humanize(driver_used)}' "
                    f"has zero mean or zero variance in the observed data."),
            evidence={"r": fit["r"], "p_value": fit["p_value"], "n": fit["n"]},
            review=review, review_basis=review_basis, driver=driver_used)
    _typical = abs(_xstd / _xmean * 100) if _xmean else 10.0
    _edge = ((fit["x_max"] - _xmean) / abs(_xmean) * 100) if move > 0 else ((_xmean - fit["x_min"]) / abs(_xmean) * 100)
    _edge = max(0.0, min(50.0, _edge if _edge == _edge else 0.0))  # NaN-safe
    def _size(k, floor, ceil):
        raw = _typical * k
        raw = min(raw, _edge) if _edge > 0 else raw
        return round(min(ceil, max(floor, raw)), 1)
    pcts = [_size(0.25, 1.0, 10.0) * move, _size(0.5, 2.0, 20.0) * move,
            _size(1.0, 5.0, 45.0) * move]
    # Dedupe (flat data can collapse sizes) while preserving order.
    seen, _uniq = set(), []
    for _p in pcts:
        if abs(_p) not in seen:
            seen.add(abs(_p))
            _uniq.append(_p)
    while len(_uniq) < 3:
        _grow = min(45.0, abs(_uniq[-1]) * 1.5 + 1.0)
        _uniq.append(round(_grow, 1) * (1 if move > 0 else -1))
    pcts = _uniq[:3]
    m_h, d_h = humanize(metric), humanize(driver_used)

    # Targeted variant (§10 segment analysis): weakest group headroom.
    targeted = None
    seg_dim = (segment or {}).get("field") or (dims[0] if dims else None)
    if seg_dim:
        try:
            targeted = _segment_extreme(engine, dialect, table, seg_dim, metric)
        except Exception:
            targeted = None

    # 5) Build + score candidates (§§11–14).
    cands_out = []
    for i, pct in enumerate(pcts):
        scen = scenario_for_pct(fit, pct)
        rlvl, rreasons = risk_level(fit, pct, abs(fit["r"]))
        score, comps = _score(fit, pct, control_level)
        # Labels from evidence order, not fixed sizes: smallest supported
        # change, strongest-evidence change, largest (highest-uncertainty).
        ordered_idx = sorted(range(len(pcts)), key=lambda k: abs(pcts[k]))
        label = ("conservative" if i == ordered_idx[0]
                 else "stretch" if i == ordered_idx[-1] else "recommended")
        # NOTE (hyp.txt §8): this title template is emitted ONLY here —
        # after the engine explicitly established controllability (§8 gate),
        # a semantic business outcome (§10 gate), a justified direction
        # (§12 gate), a measured baseline (scen["baseline"] from the fit),
        # and a modeled calculation (OLS slope via scenario_for_pct). A
        # bare correlation can never reach this line (gates above refuse
        # first with "No supported actionable scenario" / no_prescription).
        title = (f"Test a {abs(pct):g}% {verb} in {d_h}"
                 + (f" for {targeted['value']}" if targeted and label == "recommended" else ""))
        # Association is supporting evidence: plain words only (no r/p/
        # slope numbers — those live in details.numbers/evidence). The
        # direction basis is the validated one from improvement_up.
        why = (f"{m_h} is linked with {d_h} ({_strength_word(fit['r']).lower()} link, "
               f"seen across {fit['n']:,} rows); "
               f"moving {d_h} {'up' if move > 0 else 'down'} is modeled to move the metric "
               f"toward improvement. "
               f"Past patterns show an association, not proof the change would cause the result.")
        _conf = ("moderate" if (fit["r"] > 0) != (fit["spearman_r"] > 0)
                 or abs(fit["r"] - fit["spearman_r"]) > 0.4 else
                 "high" if fit["p_value"] < 0.01 and fit["n"] >= 100 else
                 "moderate" if fit["p_value"] < 0.05 else "low")
        if _dir_fallback and _conf == "high":
            _conf = "moderate"
        cands_out.append({
            "id": f"rx_{metric}_{driver_used}_{abs(pct)}",
            "rank": 0, "title": title, "summary": why,
            "action_type": "scenario", "scenario_label": label,
            # hyp §13 decision type: a validated scenario lever is an
            # evidence-backed area worth considering (opportunity), unless
            # the modeled risk is HIGH — then it is a risk worth
            # considering. The supported action text is the
            # Recommendation; what_to_consider is the Consideration.
            # control_level is stamped so downstream quality/priority gates
            # reuse the judged value instead of re-deriving it.
            "decision_type": "risk" if rlvl.lower() == "high" else "opportunity",
            "control_level": control_level,
            # Rank/linear conflict caps confidence: when Pearson and Spearman
            # disagree on direction or strength, the link is fragile.
            # An explicit fallback direction assumption also caps at moderate.
            "confidence": _conf,
            "risk": rlvl.lower(), "risk_reasons": rreasons,
            "score": score, "score_breakdown": comps,
            "evidence": {"metric": metric, "driver": driver_used, "table": table,
                         "driver_table": driver_table, "join": join_label,
                         "effect_r": fit["r"], "spearman_r": fit["spearman_r"],
                         "p_value": fit["p_value"], "n": fit["n"],
                         "slope": round(fit["slope"], 4),
                         "rank_consistent": bool((fit["r"] > 0) == (fit["spearman_r"] > 0)
                                                 and abs(fit["r"] - fit["spearman_r"]) <= 0.4),
                         "segment": targeted},
            "success_metric": {"metric": metric,
                               "direction": "increase" if up_better else "decrease",
                               "target_pct": round(scen["rel_pct"], 1),
                               "baseline": round(scen["baseline"], 2),
                               "note": "Modeled estimate from the recommended scenario."},
            "review_period": review,
            "review_basis": review_basis,
            "impact": {"baseline": round(scen["baseline"], 2),
                       "scenario": round(scen["scenario"], 2),
                       "delta": round(scen["delta"], 2),
                       "rel_pct": round(scen["rel_pct"], 2),
                       "range": [round(scen["range"][0], 2), round(scen["range"][1], 2)],
                       "driver_pct": pct,
                       "driver_value": round(scen["driver_value"], 3),
                       # Observed-range evidence (generic: from this fit's data).
                       # supported = inside observed driver range (green),
                       # limited = outside it (amber). Refusal stays ±50%.
                       "in_historical_range": scen["in_historical_range"],
                       "range_status": ("supported" if scen["in_historical_range"] else "limited"),
                       "observed": {"min": fit["x_min"], "max": fit["x_max"],
                                    "mean": round(fit["x_mean"], 4)},
                       "labels": {"baseline": "OBSERVED", "scenario": "SIMULATED"}},
            "assumptions": [up_assump,
                            "Linear response estimated from historical data;",
                            "All other factors held constant."],
            "limitations": _dynamic_limitations(fit, scen, targeted, d_h),
            "recommended_next_analysis": {
                "type": "hypothesis",
                "label": "Run a controlled hypothesis test comparing driver and metric response."},
        })
    cands_out.sort(key=lambda c: c["score"], reverse=True)
    for i, c in enumerate(cands_out):
        c["rank"] = i + 1
    # Recommended = best tradeoff, not the biggest number (§12): prefer the
    # highest-scoring candidate without HIGH risk. Labels are relative to
    # the evidence pick — never fixed percentages.
    rec = next((c for c in cands_out if c["risk"] in ("low", "medium")), cands_out[0])
    _rec_pct = abs(rec["impact"]["driver_pct"])
    for c in cands_out:
        ap = abs(c["impact"]["driver_pct"])
        c["scenario_label"] = ("recommended" if c["id"] == rec["id"]
                               else "conservative" if ap < _rec_pct else "stretch")
    # hyp §§16-17 ONE finding → ONE decision: the scenario variants computed
    # above (+conservative/+recommended/+stretch) are NOT Decision Center
    # cards. They travel INSIDE the single decision as `scenario_options`
    # for the detail What-If experience (downstream of the validated
    # decision, §32) — never as separate "Test 20% ..." / "Test 45% ..." cards.
    rec["scenario_options"] = [
        {"scenario_label": c["scenario_label"],
         "driver_pct": c["impact"]["driver_pct"],
         "driver_value": c["impact"]["driver_value"],
         "baseline": c["impact"]["baseline"],
         "scenario": c["impact"]["scenario"],
         "delta": c["impact"]["delta"],
         "rel_pct": c["impact"]["rel_pct"],
         "range": c["impact"]["range"],
         "risk": c["risk"],
         "in_historical_range": c["impact"]["in_historical_range"]}
        for c in cands_out[:3]
    ]
    # Broadened coverage: the scenario decision leads; segment-focus, monitor
    # and investigate follow-ons are different findings (different driver /
    # basis), so they may accompany the decision. Rank scenario >
    # segment-focus > monitor > investigate, dedupe model-layer duplicates,
    # validate (drop anything without evidence + quantified-impact-or-note),
    # cap 5. Scenario thresholds above are UNCHANGED — only the supporting
    # cast is new, and every new card keeps evidence gates + honesty notes.
    seg = _segment_focus_cands(engine, dialect, intel, table, metric, m_h,
                               review=review, review_basis=review_basis)
    mon = _monitor_cands(engine, dialect, intel, table, metric, m_h,
                         review=review, review_basis=review_basis)
    inv = _investigate_cands(engine, dialect, intel, table, metric, m_h,
                             review=review, review_basis=review_basis)
    pres = _rank_coverage(dedupe_prescriptions([rec] + seg + mon + inv))
    pres, _dropped = _validate_prescriptions(pres)
    # NOTE: display/details/refusal are attached downstream by analyze()
    # via _finalize_output — nothing is attached here.
    # The scenario decision must survive validation — if it was dropped, its
    # evidence was somehow unquantified, which cannot happen by construction;
    # fall back to the scenario alone rather than an empty completed set.
    if not any(str(p.get("action_type") or "").lower() == "scenario" for p in pres):
        pres = _rank_coverage(dedupe_prescriptions([rec]))
        _dropped = 0
    pres = pres[:5]
    for i, c in enumerate(pres):
        c["rank"] = i + 1

    story = (f"{m_h} is linked with {d_h} ({_strength_word(fit['r']).lower()} link, "
             f"seen across {fit['n']:,} observations"
             + (f", measured {_join_phrase(join_used)}" if join_used else "")
             + f"). A tested {verb} in {d_h} is the most supported move available. "
               f"Past patterns show an association, not proof the change would cause the result.")
    # Bounded evidence sample for the scatter chart (never the full dump).
    _sxs, _sys = best_xy or ([], [])
    _idx = sorted(set(int(i) for i in np.linspace(0, max(0, len(_sxs) - 1), min(len(_sxs), 200))))
    scatter = [{"x": round(float(_sxs[i]), 3), "y": round(float(_sys[i]), 3)} for i in _idx]
    out = {"ok": True, "success": True, "status": "completed",
           "analysis_id": f"rxa_{table}_{metric}_{driver_used}",
           "metric": metric, "driver": driver_used, "table": table,
           "driver_table": driver_table, "join": join_label,
           "join_ref": ({"table": join_used[0], "self_col": join_used[1],
                         "other_col": join_used[2]} if join_used else None),
           "control": {"level": control_level, "reason": control_reason},
           "auto_note": auto_note,
           "fit": {"r": fit["r"], "spearman_r": fit["spearman_r"],
                   "p_value": fit["p_value"], "n": fit["n"],
                   "slope": fit["slope"], "intercept": fit["intercept"]},
           "evidence_viz": {"type": "scatter", "x": driver_used, "y": metric,
                            "points": scatter,
                            "trend": {"slope": round(fit["slope"], 4),
                                      "intercept": round(fit["intercept"], 3)}},
            "prescriptions": pres,
            "scenario_meta": _scenario_meta_for_fit(driver_used, fit, move),
            "diagnostics": {"dropped_invalid": _dropped,
                            "emitted": len(pres),
                            "tested": [dict(t) for t in tested][:12]},
           "signals": {"supporting": 1 + (1 if (priors.get("evidence")) else 0),
                       "priors_origin": priors.get("origin")},
           "story": story,
            "summary_plain": _plain_summary(m_h, d_h, fit, rec, _join_phrase(join_used))}
    return out


def _run_scenario_core(engine, dialect, intel, table, metric, driver, changes, filt=None,
                 join=None):
    """Scenario engine (§27, hyp.txt §9): read-only recompute. changes=[{field, operation,
    value}], filt=(field, value) equals on a dimension. join=(other_table,
    self_col, other_col) across one declared edge for cross-table drivers.

    Never invents levers: a driver/metric that fails semantic-role +
    actionability validation returns SCENARIO_UNSUPPORTED with the honest
    "No supported actionable scenario" state instead of fabricating a
    numeric projection."""
    # hyp.txt §§4-6, §9: scenario variables must be legitimate business
    # levers with a real outcome — determined semantically, never assumed
    # from "numeric therefore movable".
    _spks = {p.lower() for p in intel.get_pk(table)}
    _sfks = {f[0].lower() for f in intel.get_fks(table)}
    _skeys = _join_key_cols(intel, table)
    if metric in intel.get_numeric_cols(table):
        _smrole = classify_semantic_role(
            metric, "numeric", is_pk=(metric.lower() in _spks),
            is_fk=(metric.lower() in _sfks),
            is_join_key=(metric.lower() in _skeys))
        if _smrole in NON_OUTCOME_ROLES:
            return _fail(CODE_SCENARIO_UNSUPPORTED,
                         f"{NO_SUPPORTED_SCENARIO}: '{metric}' is a "
                         f"{_smrole.lower().replace('_', ' ')}, not a business outcome ARIA can model.")
    derived_stay = (driver == "length_of_stay")
    if not derived_stay and driver in intel.get_numeric_cols(table):
        _sdrole = classify_semantic_role(
            driver, "numeric", is_pk=(str(driver).lower() in _spks),
            is_fk=(str(driver).lower() in _sfks),
            is_join_key=(str(driver).lower() in _skeys))
        if _sdrole in NON_LEVER_ROLES:
            return _fail(CODE_SCENARIO_UNSUPPORTED,
                         f"{NO_SUPPORTED_SCENARIO}: '{driver}' is a "
                         f"{_sdrole.lower().replace('_', ' ')}, not a controllable business driver.")
        _slevel, _swhy = classify_driver(driver, "numeric")
        if _slevel in ("not_controllable", "segmentation"):
            return _fail(CODE_SCENARIO_UNSUPPORTED,
                         f"{NO_SUPPORTED_SCENARIO}: '{driver}' cannot be moved as an intervention. {_swhy}")
    measures = _usable_measures(intel, table)
    if metric not in measures:
        return _fail(CODE_INVALID_METRIC, f"'{metric}' is not a measurable metric here.")
    if join is not None:
        ot, sc, oc = join
        if (ot, sc, oc) not in _edge_neighbors(intel, table):
            return _fail(CODE_SCENARIO_UNSUPPORTED, "Unknown join for this scenario.")
        if driver not in _usable_measures(intel, ot):
            return _fail(CODE_SCENARIO_UNSUPPORTED, f"Scenario modeling needs a numeric driver; '{driver}' isn't one.")
    elif not derived_stay and driver not in intel.get_numeric_cols(table):
        return _fail(CODE_SCENARIO_UNSUPPORTED, f"Scenario modeling needs a numeric driver; '{driver}' isn't one.")
    filt_t = None
    if filt:
        f, v = filt.get("field"), filt.get("value")
        if f not in intel.find_all_dimensions(table):
            return _fail(CODE_SCENARIO_UNSUPPORTED, f"Filter field '{f}' is not a usable segment here.")
        if join is not None:
            return _fail(CODE_SCENARIO_UNSUPPORTED,
                         "Segment filters with cross-table drivers aren't supported yet — run the overall scenario instead.")
        filt_t = (f, v)
    def _to_floats(rows):
        """Parse raw pairs, TEXT-numeric aware; drops unparsable pairs."""
        try:
            from core.text_numeric import try_parse_text_number as _tpn
        except Exception:
            _tpn = None
        if _tpn is None:
            return [float(r[0]) for r in rows], [float(r[1]) for r in rows]
        pairs = [(_tpn(r[0]), _tpn(r[1])) for r in rows]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if not pairs:
            raise ValueError("no parseable numeric pairs")
        return [p[0] for p in pairs], [p[1] for p in pairs]

    try:
        if join is not None:
            ot, sc, oc = join
            rows = _fetch_joined(engine, dialect, table, metric, ot, driver, sc, oc)
            xs, ys = _to_floats(rows)
        elif derived_stay:
            dts = _date_like_cols(intel, table)[:3]
            if len(dts) < 2:
                return _fail(CODE_SCENARIO_UNSUPPORTED, "Stay-length modeling needs two date columns.")
            xs, ys = _stay_lengths(engine, dialect, table, dts[0], dts[1], metric)
            if xs is None:
                return _fail(CODE_INSUFFICIENT_DATA, "Not enough dated rows to model stay length.")
        else:
            rows = _fetch_pairs(engine, dialect, table, driver, metric, filt=filt_t)
            xs, ys = _to_floats(rows)
    except (TypeError, ValueError):
        return _fail(CODE_MODEL_FAILED, "Scenario fields contain non-numeric values that can't be modeled.")
    try:
        fit = fit_relationship(xs, ys)
    except Exception:
        return _fail(CODE_MODEL_FAILED, "Scenario fields could not be modeled reliably.")
    if fit is None:
        return _fail(CODE_INSUFFICIENT_DATA,
                     f"Only {len(xs)} usable observations — at least {MIN_N} are needed to model a scenario.")
    results = []
    for ch in changes or []:
        field, op, val = ch.get("field"), ch.get("operation"), ch.get("value")
        if field != driver:
            return _fail(CODE_SCENARIO_UNSUPPORTED, "Scenarios can only change the modeled driver.")
        try:
            val = float(val)
        except (TypeError, ValueError):
            return _fail(CODE_SCENARIO_UNSUPPORTED, "Scenario value must be numeric.")
        if op == "percent":
            pct = val
        elif op == "set":
            pct = ((val - fit["x_mean"]) / abs(fit["x_mean"]) * 100) if fit["x_mean"] else 0.0
        else:
            return _fail(CODE_SCENARIO_UNSUPPORTED, "Operation must be 'percent' or 'set'.")
        if abs(pct) > 50:
            return _fail(CODE_SCENARIO_UNSUPPORTED,
                         "Interventions beyond ±50% leave the evidence behind — narrow the scenario.")
        scen = scenario_for_pct(fit, pct)
        rlvl, rreasons = risk_level(fit, pct, abs(fit["r"]))
        results.append({"change": ch, "baseline": round(scen["baseline"], 2),
                        "scenario": round(scen["scenario"], 2),
                        "delta": round(scen["delta"], 2),
                        "rel_pct": round(scen["rel_pct"], 2),
                        "range": [round(scen["range"][0], 2), round(scen["range"][1], 2)],
                        "driver_value": round(scen["driver_value"], 3),
                        "in_historical_range": scen["in_historical_range"],
                        "range_status": ("supported" if scen["in_historical_range"] else "limited"),
                        "observed": {"min": fit["x_min"], "max": fit["x_max"],
                                     "mean": round(fit["x_mean"], 4)},
                        "risk": rlvl.lower(), "labels": {"baseline": "OBSERVED", "scenario": "SIMULATED"}})
    return {"ok": True, "success": True, "metric": metric, "driver": driver,
            "table": table, "segment": filt_t, "n": fit["n"],
            "scenarios": results,
            "assumptions": ["Linear response from historical slope; all else held constant."],
            "warning": ("Simulated estimates are not guaranteed outcomes."
                        if results else None)}


# --------------------------------------------------------------------------
# Document mode (§6): text-only PDFs — qualitative, source-cited, no numbers
# --------------------------------------------------------------------------
_DOC_STOPWORDS = frozenset(
    "the a an and or of to in on for with is are was were be been by as at "
    "from that this these those it its into over under between will would can "
    "could should have has had not no yes if then than so such only also very "
    "more most other some any each per via vs etc what which who when where how "
    "about against during after before".split()
)

_PROBLEM_WORDS = ("declin", "drop", "fall", "risk", "delay", "issue", "concern",
                  "loss", "fail", "bottleneck", "backlog", "shortage", "complaint",
                  "breach", "incident", "downtime", "overrun")
_OPPORTUNITY_WORDS = ("growth", "opportunit", "improv", "gain", "expansion",
                      "launch", "record", "strong", "success", "saving")


def _doc_terms(text):
    import re as _re
    return [t for t in _re.findall(r"[a-z0-9]+", str(text or "").lower())
            if len(t) > 2 and t not in _DOC_STOPWORDS]


def _sanitize_label(text, max_len=90):
    """Neutralize untrusted document/question text for titles/labels.

    Strips HTML/XML tags, control characters and collapses whitespace so a
    malicious chunk section or filename cannot inject markup or directive
    prose into a recommendation title. Purely generic — no dataset names.
    """
    import re as _re
    s = str(text or "")
    s = _re.sub(r"<[^>]*>", " ", s)
    s = "".join(c if (c.isprintable() or c in ("…",)) else " " for c in s)
    s = _re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        return s[:max(0, max_len - 1)].rstrip() + "…"
    return s


def _doc_title(chunk):
    section = (chunk.get("section") or "").strip()
    if section:
        return _sanitize_label(section, 90) or "Untitled finding"
    text = " ".join(str(chunk.get("content") or "").split())
    head = text.split(". ")[0].strip()
    if len(head) > 90:
        return _sanitize_label(head, 90) or "Untitled finding"
    return _sanitize_label(head, 90) or "Untitled finding"


def _analyze_document_core(chunks, question, doc_name):
    """Qualitative prescriptions from document passages (§6).

    Retrieval-free inside: chunks are pre-scored by the caller or scored
    here by deterministic term overlap. Every claim is a verbatim-bounded
    quote with a page citation. No numeric modeling is ever claimed.
    """
    qterms = set(_doc_terms(question))
    if not qterms:
        return {"ok": True, "success": True, "status": "no_prescription",
                "reason": ("Tell ARIA what decision the document should inform "
                           "— e.g. what risks need action?"),
                "alternatives": ["Ask ARIA", "Explore Data"],
                "document": doc_name}
    scored = []
    for ch in chunks or []:
        terms = set(_doc_terms(ch.get("content")))
        overlap = qterms & terms
        if overlap:
            scored.append((len(overlap), ch))
    scored.sort(key=lambda t: t[0], reverse=True)
    top = [ch for _, ch in scored[:5]]
    if not top:
        _doc_display_empty = _sanitize_label(doc_name, 120) or "this document"
        return {"ok": True, "success": True, "status": "no_prescription",
                "reason": (f"No passage in {_doc_display_empty} speaks to this question. "
                           f"Try different wording or ask ARIA to summarize the document first."),
                "alternatives": ["Ask ARIA", "Explore Data"],
                "document": doc_name}
    recs = []
    _doc_display = _sanitize_label(doc_name, 120) or "this document"
    for ch in top[:3]:
        text = " ".join(str(ch.get("content") or "").split())
        low = text.lower()
        is_problem = any(w in low for w in _PROBLEM_WORDS)
        is_opp = any(w in low for w in _OPPORTUNITY_WORDS)
        page = ch.get("page_number")
        quote = text[:277] + "…" if len(text) > 280 else text
        if is_problem and not is_opp:
            title = f"Look into: {_doc_title(ch)}"
            action = "investigate"
        elif is_opp and not is_problem:
            title = f"Prepare for: {_doc_title(ch)}"
            action = "prepare"
        else:
            title = f"Consider: {_doc_title(ch)}"
            action = "review"
        recs.append({
            "id": f"rxdoc_{ch.get('id', len(recs))}",
            "rank": len(recs) + 1,
            "title": title,
            "summary": f"Passage from {_doc_display}" + (f" (page {page})" if page else "") + " reads: “" + quote + "”",
            "action_type": "qualitative", "scenario_label": "qualitative",
            # hyp §13: a cited passage is a Finding (an observed pattern),
            # never an Opportunity — there is no measured lever behind it.
            "decision_type": "finding",
            "confidence": "low", "risk": "medium",
            "risk_reasons": ["Text evidence only — strength cannot be quantified."],
            "score": round(10 + len(set(_doc_terms(text)) & qterms), 1),
            "score_breakdown": {"passage_relevance": len(set(_doc_terms(text)) & qterms)},
            "evidence": {"document": doc_name, "page": page,
                         "n": len(top), "segment": None},
            "citations": [{"page": page, "quote": quote,
                           "chunk_id": ch.get("id")}],
            "impact": {"baseline": None, "scenario": None, "delta": None,
                       "rel_pct": None, "range": [None, None],
                       "driver_pct": None, "driver_value": None,
                       "labels": {"baseline": "OBSERVED", "scenario": "NOT MODELED"}},
            "success_metric": {"metric": "decision readiness",
                               "direction": "confirm or rule out with numbers",
                               "target_pct": None, "baseline": None,
                               "note": "Qualitative — confirm with data before acting."},
            "review_period": "after stakeholder review",
            "review_basis": "text evidence has no time series to pace reviews",
            "assumptions": ["Passages interpreted at face value."],
            "limitations": ["Text evidence only — no numbers were modeled.",
                            f"Based on {len(top)} cited passage(s) from {_doc_display}; the rest of the document may add context.",
                            "Association in prose is not proof of cause or size."],
            "recommended_next_analysis": {"type": "copilot",
                                          "label": "Ask ARIA to dig into this passage further."},
        })
    story = (f"{_doc_display} speaks to your question in {len(top)} passage(s). "
              f"These are qualitative leads with citations — confirm them with numbers before acting.")
    return {"ok": True, "success": True, "status": "completed",
            "analysis_id": f"rxd_{abs(hash(doc_name)) % 10**8}",
            "document": doc_name, "metric": None, "driver": None, "table": None,
            "prescriptions": recs,
            "signals": {"supporting": 1, "priors_origin": "document"},
            "story": story,
            "summary_plain": (
                f"We read {_doc_display} for passages about your question and found "
                f"{len(top)} relevant one(s). The top lead is: {recs[0]['title']}. "
                f"Each card quotes its source passage — these are leads to verify, not measured effects.")}


# ---------------------------------------------------------------------------
# PHASE 10-11 Decision Center integration helpers (additive, no behavior change
# to analyze/run_scenario). Existing engine stays the single recommendation +
# scenario backend: these wrappers only validate + shape Decision payloads.
# ---------------------------------------------------------------------------
def recommendation_for_decision(decision: dict):
    """Supported recommendation text for a validated Decision, else None.

    Never returns the finding title. Never an LLM-only action: the text must
    already come from the engine-produced prescription; this wrapper only
    validates business_relevance + actionability + outcome + direction +
    evidence (via quality_gate + semantic roles) and returns None when
    unsupported.
    """
    d = decision if isinstance(decision, dict) else {}
    raw = d.get("recommendation")
    raw = raw.strip() if isinstance(raw, str) else None
    title = d.get("title") if isinstance(d.get("title"), str) else ""
    if raw and title and raw.strip().lower() == title.strip().lower():
        return None
    if not raw:
        return None
    ev = d.get("evidence")
    items = ev if isinstance(ev, list) else ([ev] if ev else [])
    stem = items[0] if items and isinstance(items[0], dict) else {}
    probe = gate_input_from_prescription(
        {"metric": d.get("metric"), "driver": d.get("driver"),
         "evidence": stem, "impact": d.get("impact") or {},
         "confidence": d.get("confidence"),
         "control_level": d.get("control_level"),
         "action_type": d.get("action_type")},
        require_baseline=(str(d.get("action_type") or "").lower() == "scenario"))
    gate = quality_gate(probe, require_baseline=(str(d.get("action_type") or "").lower() == "scenario"))
    if not gate.get("pass"):
        return None
    metric = d.get("metric") or ""
    driver = d.get("driver") or d.get("dimension") or ""
    try:
        if classify_semantic_role(metric, "numeric") in NON_OUTCOME_ROLES:
            return None
        if driver and classify_semantic_role(driver, "numeric") in NON_LEVER_ROLES:
            return None
    except Exception:
        return None
    if driver:
        try:
            level, _why = classify_driver(driver, "numeric")
        except Exception:
            return None
        if level in ("not_controllable", "segmentation"):
            return None
    try:
        up_better, _why = improvement_up(metric) if metric else (None, "no metric")
    except Exception:
        return None
    if str(d.get("action_type") or "").lower() == "scenario" and up_better is None:
        return None
    return raw


def scenario_supported_range(impact: dict | None):
    """Backend-derived supported range for What-If (never hardcoded ±%).

    Returns the observed {min, max, mean} frozen in the impact payload, or
    None when the backend never measured one.
    """
    imp = impact if isinstance(impact, dict) else {}
    obs = imp.get("observed")
    if not isinstance(obs, dict):
        return None
    try:
        lo = float(obs.get("min")) if obs.get("min") is not None else None
        hi = float(obs.get("max")) if obs.get("max") is not None else None
        mean = float(obs.get("mean")) if obs.get("mean") is not None else None
    except (TypeError, ValueError):
        return None
    if lo is None or hi is None:
        return None
    return {"min": lo, "max": hi, "mean": mean}


# ---------------------------------------------------------------------------
# Display-safe contract (DYNAMIC ONLY): every prescription carries display{}
# + details{} built from the live evidence — no table/column/database names
# in this logic, no hardcoded levers. All names flow through
# core.semantic humanize/display_name; all numbers through the money-aware
# rounders below (percent 0-1dp, money $K/M, counts integer, never >2dp).
# Additive: existing keys are never removed here — the routes layer strips
# internals and the frontend migrates to display/details.
# ---------------------------------------------------------------------------
try:
    from core.semantic import humanize as _sem_humanize
    from core.semantic import display_name as _sem_display
    from core.semantic import classify_semantic_type as _sem_type
except Exception:  # pragma: no cover - helper fallback, never blocks analysis
    def _sem_humanize(name):
        import re as _re
        return _re.sub(r"\s+", " ", str(name or "").replace("_", " ")).strip().title() \
            or str(name or "")

    def _sem_display(raw, fallback="this metric"):
        return _sem_humanize(raw) if str(raw or "").strip() else fallback

    def _sem_type(column_name, db_type=""):
        return "measure"


_MONEY_NAME_HINTS = (
    "revenue", "revenues", "profit", "profits", "margin", "margins",
    "income", "sales", "price", "pricing", "cost", "amount", "total",
    "payment", "spend", "spending", "budget", "discount", "wage",
    "salary", "fee", "fare", "freight", "tuition", "premium",
)

# Confidence labels reuse the engine's own confidence vocabulary — a pure
# mapping, no new numeric thresholds.
_CONFIDENCE_LABELS = {
    "high": "High",
    "moderate": "Moderate",
    "medium": "Moderate",
    "low": "Early",
    "limited": "Early",
    "unknown": "Early",
    "none": "Early",
    "": "Early",
}


def confidence_label_for(confidence) -> str:
    """High/Moderate/Early derived from the existing confidence value."""
    return _CONFIDENCE_LABELS.get(str(confidence or "").strip().lower(), "Early")


def _is_money_metric(name) -> bool:
    """Money-aware flag from the live name only (never a column literal)."""
    try:
        if _sem_type(str(name or ""), "") == "money":
            return True
    except Exception:
        pass
    return _has_hint(name, _MONEY_NAME_HINTS)


def _trim_num(v, max_dp=2) -> str:
    """Round to <=max_dp decimals, trimming trailing zeros (never >2dp)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:  # NaN-safe
        return str(v)
    neg = f < 0
    f = abs(f)
    for dp in range(max_dp, -1, -1):
        r = round(f, dp)
        s = f"{r:.{dp}f}" if dp else str(int(r))
        if dp == 0 or abs(f - r) < 0.5 * 10 ** -dp + 1e-12:
            if "." in s:
                s = s.rstrip("0").rstrip(".")
            return ("-" if neg else "") + s
    return ("-" if neg else "") + str(round(f, max_dp))


def _fmt_money_short(v) -> str:
    """$312K / $2.1M style — money-aware, never more than 1 decimal."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:
        return str(v)
    neg = "-" if f < 0 else ""
    a = abs(f)
    if a >= 1_000_000_000:
        return f"{neg}${_trim_num(a / 1_000_000_000, 1)}B"
    if a >= 1_000_000:
        return f"{neg}${_trim_num(a / 1_000_000, 1)}M"
    if a >= 1_000:
        return f"{neg}${_trim_num(a / 1_000, 1)}K"
    if a >= 100:
        return f"{neg}${int(round(a)):,}"
    return f"{neg}${_trim_num(a, 2)}"


def _fmt_pct_short(v) -> str:
    """+10% / -2.5% style — percent, 0-1dp with sign."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:
        return str(v)
    return f"{'+' if f >= 0 else ''}{_trim_num(f, 1)}%"


def _fmt_count_short(v) -> str:
    """Counts render as integers with thousands separators."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:
        return str(v)
    return f"{int(round(f)):,}"


def _human_amount(v, metric):
    """Money-aware humanized amount for an outcome number."""
    if v is None:
        return None
    if _is_money_metric(metric):
        return _fmt_money_short(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:
        return str(v)
    if abs(f) >= 1000:
        a = abs(f)
        neg = "-" if f < 0 else ""
        if a >= 1_000_000_000:
            return f"{neg}{_trim_num(a / 1_000_000_000, 1)}B"
        if a >= 1_000_000:
            return f"{neg}{_trim_num(a / 1_000_000, 1)}M"
        return f"{neg}{_trim_num(a / 1_000, 1)}K"
    if float(f).is_integer():
        return _fmt_count_short(f)
    return _trim_num(f, 1)


def _rec_names(rec: dict):
    """Dynamic humanized names only (metric/driver/top/bottom/direction)."""
    rec = rec if isinstance(rec, dict) else {}
    ev = rec.get("evidence") if isinstance(rec.get("evidence"), dict) else {}
    metric = rec.get("metric") or ev.get("metric")
    driver = rec.get("driver") or ev.get("driver")
    return {
        "metric": metric,
        "driver": driver,
        "M": _sem_display(metric),
        "D": _sem_display(driver, "this factor"),
        "top": ev.get("top"),
        "bottom": ev.get("bottom"),
        "direction": ev.get("direction"),
        "ev": ev,
    }


def _why_summary_for(rec: dict) -> str:
    """One plain sentence from the evidence: leader/gap/momentum wording.

    Qualitative only — never r/p/n or any statistic.
    """
    n = _rec_names(rec)
    atype = str((rec or {}).get("action_type") or "").lower()
    label = str((rec or {}).get("scenario_label") or "").lower()
    if atype == "scenario":
        return (f"{n['D']} is the strongest linked factor for {n['M']} "
                f"in your data, so testing a change there is the most supported move.")
    if label == "segment-focus":
        top = n["top"] or n["D"]
        return (f"{top} holds the largest share of {n['M']} rows, "
                f"so reviewing it first focuses effort where it counts.")
    if atype == "monitor":
        direction = str(n["direction"] or "").lower() or "shifting"
        return (f"{n['M']} is moving {direction}, "
                f"so keep watching it before deciding anything.")
    if atype == "investigate":
        top, bottom = n["top"], n["bottom"]
        if top and bottom:
            return (f"{top} stands out against {bottom} on {n['M']}, "
                    f"so that gap is where to look first.")
        return (f"{n['D']} shows the clearest gap for {n['M']}, "
                f"so start the investigation there.")
    if atype == "qualitative":
        return "This passage speaks to your question, so review it before acting."
    drv = n["driver"] or n["metric"]
    return (f"{_sem_display(drv, n['M'])} shows the clearest pattern "
            f"for {n['M']}, so start there.")


def _action_text_for(rec: dict) -> str:
    """Plain verb phrase for the card (dynamic names only)."""
    rec = rec if isinstance(rec, dict) else {}
    n = _rec_names(rec)
    atype = str(rec.get("action_type") or "").lower()
    label = str(rec.get("scenario_label") or "").lower()
    imp = rec.get("impact") if isinstance(rec.get("impact"), dict) else {}
    if atype == "scenario":
        pct = imp.get("driver_pct")
        try:
            pct = float(pct) if pct is not None else None
        except (TypeError, ValueError):
            pct = None
        seg = n["ev"].get("segment")
        seg_txt = ""
        if isinstance(seg, dict) and seg.get("value") is not None:
            seg_txt = f" for {seg.get('value')}"
        if pct is None:
            return f"Adjust {n['D']}{seg_txt} to shift {n['M']}"
        verb = "increase" if pct > 0 else "decrease" if pct < 0 else "adjust"
        return f"Try {abs(pct):g}% {verb} in {n['D']}{seg_txt}"
    if label == "segment-focus":
        return f"Review {n['top'] or n['D']} for {n['M']}"
    if atype == "monitor":
        return f"Keep watching {n['M']}"
    if atype == "investigate":
        if n["top"] and n["bottom"]:
            return f"Look into {n['top']} vs {n['bottom']} for {n['M']}"
        return f"Look into {n['D']} for {n['M']}"
    if atype == "qualitative":
        return "Review the cited passage"
    return f"Review {n['D']} for {n['M']}"


def _outcome_text_for(rec: dict) -> str:
    """Humanized, rounded outcome: $312K / +10% style, never >2 decimals."""
    rec = rec if isinstance(rec, dict) else {}
    n = _rec_names(rec)
    atype = str(rec.get("action_type") or "").lower()
    imp = rec.get("impact") if isinstance(rec.get("impact"), dict) else {}
    ev = n["ev"]
    metric = n["metric"]

    def _num(v):
        try:
            f = float(v)
            return f if f == f else None
        except (TypeError, ValueError):
            return None

    if atype == "monitor":
        mom = _num(ev.get("momentum_pct"))
        pts = ev.get("n_points", ev.get("n"))
        direction = str(ev.get("direction") or "").lower()
        if mom is not None:
            word = direction.capitalize() if direction else ("Up" if mom >= 0 else "Down")
            pts_txt = f" over {int(pts)} periods" if pts else ""
            return f"{word} {_trim_num(abs(mom), 0)}%{pts_txt}"
        return "Movement observed — watch before acting."
    base, scen = _num(imp.get("baseline")), _num(imp.get("scenario"))
    rel = _num(imp.get("rel_pct"))
    if base is not None and scen is not None:
        hb, hs = _human_amount(base, metric), _human_amount(scen, metric)
        if rel is not None:
            tag = "observed gap" if atype in ("investigate",) or str(
                rec.get("scenario_label") or "").lower() == "segment-focus" else "vs baseline"
            return f"{hb} → {hs} ({_fmt_pct_short(rel)} {tag})"
        return f"{hb} → {hs}"
    if rel is not None:
        return f"{_fmt_pct_short(rel)} vs baseline"
    if atype == "qualitative":
        return "No sized outcome — review only."
    note = str(rec.get("impact_note") or "")
    if note:
        return "No sized outcome — review only."
    return "No sized outcome — review only."


def _evidence_pass_for(rec: dict) -> bool:
    """Quality-gate verdict backing details.evidence_pass (never invented)."""
    try:
        atype = str((rec or {}).get("action_type") or "").lower()
        probe = gate_input_from_prescription(rec, require_baseline=(atype == "scenario"))
        gate = quality_gate(probe, require_baseline=(atype == "scenario"))
        return bool(gate.get("pass"))
    except Exception:
        return False


def _checks_for(rec: dict) -> list:
    """Named gates this card actually passed (computed from live evidence)."""
    rec = rec if isinstance(rec, dict) else {}
    ev = rec.get("evidence") if isinstance(rec.get("evidence"), dict) else {}
    imp = rec.get("impact") if isinstance(rec.get("impact"), dict) else {}
    checks = []

    def _num(v):
        try:
            f = float(v)
            return f if f == f else None
        except (TypeError, ValueError):
            return None

    nn = ev.get("n")
    try:
        nn = int(nn) if nn is not None else None
    except (TypeError, ValueError):
        nn = None
    if nn is not None and nn >= MIN_N:
        checks.append("sample_size")
    p = _num(ev.get("p_value"))
    if p is not None and p < 0.05:
        checks.append("significance")
    r = _num(ev.get("effect_r", ev.get("r")))
    if r is not None and abs(r) >= MIN_ABS_R:
        checks.append("strength")
    gap = _num(ev.get("effect_gap_pct"))
    share = _num(ev.get("share_pct"))
    mom = _num(ev.get("momentum_pct"))
    if ((gap is not None and abs(gap) >= MIN_REL_PCT)
            or (share is not None and share >= SEGMENT_FOCUS_MIN_SHARE * 100)
            or (mom is not None and abs(mom) >= MONITOR_MIN_MOMENTUM * 100)):
        checks.append("pattern_strength")
    ctrl = str(rec.get("control_level") or "").lower()
    if ctrl in ("controllable", "potentially_controllable"):
        checks.append("controllability")
    sm = rec.get("success_metric") if isinstance(rec.get("success_metric"), dict) else {}
    if sm.get("direction") or str(rec.get("action_type") or "").lower() != "scenario":
        checks.append("direction")
    if any(imp.get(k) is not None for k in ("baseline", "scenario", "delta", "rel_pct")):
        checks.append("impact_quantified")
    else:
        hay = " ".join([str(rec.get("impact_note") or "")]
                        + [str(x) for x in (rec.get("limitations") or [])]).lower()
        if any(m in hay for m in _HONEST_NOTE_MARKERS):
            checks.append("honest_note")
    return checks


_NUMERIC_EVIDENCE_KEYS = (
    "n", "n_points", "effect_r", "r", "spearman_r", "p_value", "slope",
    "effect_gap_pct", "share_pct", "momentum_pct", "first", "last", "mean",
)
_NUMERIC_IMPACT_KEYS = (
    "baseline", "scenario", "delta", "rel_pct", "driver_pct", "driver_value",
)
_LABEL_EVIDENCE_KEYS = ("top", "bottom", "median", "direction")


def _numbers_for(rec: dict) -> dict:
    """Existing raw evidence/impact numbers for the expander (flat, raw)."""
    rec = rec if isinstance(rec, dict) else {}
    ev = rec.get("evidence") if isinstance(rec.get("evidence"), dict) else {}
    imp = rec.get("impact") if isinstance(rec.get("impact"), dict) else {}
    out = {}
    for k in _NUMERIC_EVIDENCE_KEYS:
        v = ev.get(k)
        try:
            f = float(v) if v is not None else None
        except (TypeError, ValueError):
            f = None
        if f is not None and f == f:
            out[k] = f if k == "p_value" else (int(f) if k in ("n", "n_points") else f)
    for k in _NUMERIC_IMPACT_KEYS:
        v = imp.get(k)
        try:
            f = float(v) if v is not None else None
        except (TypeError, ValueError):
            f = None
        if f is not None and f == f:
            out[k] = f
    for k in _LABEL_EVIDENCE_KEYS:
        v = ev.get(k)
        if v is not None and str(v).strip() != "":
            out[k] = v
    seg = ev.get("segment")
    if isinstance(seg, dict) and seg.get("value") is not None:
        out["segment"] = f"{seg.get('field') or seg.get('column') or ''}={seg.get('value')}".strip("=")
    return out


_ASSOCIATION_DISCLAIMER = ("Past patterns show an association, not proof "
                             "that a change will cause the result.")

_CONFIDENCE_MEANINGS = {
    "High": "Strong support from past data — the pattern is stable.",
    "Moderate": "Reasonable support but real uncertainty remains.",
    "Early": "Early signal only — limited support, watch first.",
}


def _confidence_meaning_for(confidence) -> str:
    """Plain meaning for the single High/Moderate/Early vocabulary."""
    return _CONFIDENCE_MEANINGS.get(confidence_label_for(confidence), _CONFIDENCE_MEANINGS["Early"])


def _what_would_raise_for(confidence) -> str:
    """What would raise confidence — plain, actionable, generic."""
    label = confidence_label_for(confidence)
    if label == "High":
        return "Fresh data confirming the pattern holds would keep this High."
    if label == "Moderate":
        return "More rows and a repeat period with the same pattern would raise this to High."
    return "More rows, a stable repeat, and a tested driver would raise this above Early."


def _problem_text_for(rec: dict) -> str:
    """Plain-language Problem: what changed, by how much, segment/period.

    No r/p/Spearman/slope numbers and no observed-range min-max — stats
    live in details.numbers/evidence only.
    """
    rec = rec if isinstance(rec, dict) else {}
    n = _rec_names(rec)
    atype = str(rec.get("action_type") or "").lower()
    label = str(rec.get("scenario_label") or "").lower()
    imp = rec.get("impact") if isinstance(rec.get("impact"), dict) else {}
    ev = n["ev"]

    def _num(v):
        try:
            f = float(v)
            return f if f == f else None
        except (TypeError, ValueError):
            return None

    if atype == "scenario":
        rel, base, scen = _num(imp.get("rel_pct")), _num(imp.get("baseline")), _num(imp.get("scenario"))
        seg = ev.get("segment")
        seg_txt = ""
        if isinstance(seg, dict) and seg.get("value") is not None:
            seg_txt = f" for {seg.get('value')}"
        if rel is not None and base is not None and scen is not None:
            hb, hs = _human_amount(base, n["metric"]), _human_amount(scen, n["metric"])
            return (f"{n['M']} could shift {hs} vs {hb} now ({_fmt_pct_short(rel)}) "
                    f"if {n['D']} is tested{seg_txt}.")
        return f"{n['M']} is linked with {n['D']}{seg_txt} — testing a change there is the focus."
    if label == "segment-focus":
        top = n.get("top") or n["D"]
        share = ev.get("share_pct")
        try:
            _share = float(share) if share is not None else None
        except (TypeError, ValueError):
            _share = None
        if _share is not None:
            return f"{top} holds about {_trim_num(_share, 0)}% of {n['M']} rows — the concentration to review."
        return f"{top} holds the largest share of {n['M']} rows — the concentration to review."
    if atype == "monitor":
        mom = ev.get("momentum_pct")
        pts = ev.get("n_points", ev.get("n"))
        direction = str(ev.get("direction") or "").lower() or "shifting"
        try:
            _mom = float(mom) if mom is not None else None
        except (TypeError, ValueError):
            _mom = None
        if _mom is not None:
            pts_txt = f" over {int(pts)} periods" if pts else ""
            return f"{n['M']} moved {direction} about {_trim_num(abs(_mom), 0)}%{pts_txt}."
        return f"{n['M']} is moving {direction} — keep watching before deciding."
    if atype == "investigate":
        top, bottom = n.get("top"), n.get("bottom")
        gap = ev.get("effect_gap_pct")
        try:
            _gap = float(gap) if gap is not None else None
        except (TypeError, ValueError):
            _gap = None
        if top and bottom and _gap is not None:
            return f"{top} vs {bottom} differ by about {_trim_num(abs(_gap), 0)}% on {n['M']}."
        return f"{n['D']} shows the clearest gap for {n['M']}."
    if atype == "qualitative":
        return "A passage in your document speaks to the question — review it before acting."
    return f"{n['D']} shows the clearest pattern for {n['M']}."


def _why_text_for(rec: dict, tested=None, supporting=None) -> str:
    """Plain-language Why: 2-3 bullets + one disclaimer (Problem→Why→Action).

    Built from tested[] + segment-focus/monitor/investigate outputs passed
    as inputs (not sibling cards):
      1. strongest linked driver in words Strong/Moderate + row count,
      2. corroborating segment/time evidence,
      3. top rejected alternative.
    No r/p/slope numbers.
    """
    rec = rec if isinstance(rec, dict) else {}
    n = _rec_names(rec)
    ev = n["ev"]
    atype = str(rec.get("action_type") or "").lower()
    tested = [t for t in (tested or []) if isinstance(t, dict)]
    supporting = [s for s in (supporting or []) if isinstance(s, dict) and s is not rec]

    def _rows():
        for k in ("n", "n_points"):
            try:
                v = int(ev.get(k)) if ev.get(k) is not None else None
                if v:
                    return f"{v:,} rows"
            except (TypeError, ValueError):
                continue
        return "your available rows"

    bullets = []
    # Bullet 1: strongest linked driver in words + row count.
    if atype == "scenario":
        r = ev.get("effect_r", ev.get("r"))
        bullets.append(f"• {_strength_word(r)} link between {n['D']} and {n['M']} across {_rows()} — the strongest tested.")
    elif str(rec.get("scenario_label") or "").lower() == "segment-focus":
        top = n.get("top") or n["D"]
        bullets.append(f"• {top} holds the largest share of {n['M']} rows ({_rows()}) — focus lands where it counts.")
    elif atype == "monitor":
        direction = str(ev.get("direction") or "").lower() or "shift"
        bullets.append(f"• {n['M']} shows a clear {direction}ward movement across {_rows()} of ordered history.")
    elif atype == "investigate":
        top, bottom = n.get("top"), n.get("bottom")
        if top and bottom:
            bullets.append(f"• {top} stands out against {bottom} on {n['M']} across {_rows()} — the gap to probe.")
        else:
            bullets.append(f"• {n['D']} shows the clearest gap for {n['M']} across {_rows()}.")
    else:
        bullets.append(f"• Clearest pattern for {n['M']} involves {n['D']} across {_rows()}.")
    # Bullet 2: corroborating segment/time evidence from supporting inputs.
    corroborated = False
    for s in supporting:
        try:
            sev = s.get("evidence") if isinstance(s.get("evidence"), dict) else {}
            slabel = str(s.get("scenario_label") or "").lower()
            satype = str(s.get("action_type") or "").lower()
            if slabel == "segment-focus" and sev.get("top"):
                bullets.append(f"• Corroboration: {sev.get('top')} also concentrates {n['M']} rows in a segment split.")
                corroborated = True
                break
            if satype == "monitor" and sev.get("direction"):
                bullets.append(f"• Corroboration: {n['M']} is also moving {str(sev.get('direction')).lower()} over time.")
                corroborated = True
                break
            if satype == "investigate" and sev.get("top") and sev.get("bottom"):
                bullets.append(f"• Corroboration: {sev.get('top')} vs {sev.get('bottom')} differ on {n['M']} too.")
                corroborated = True
                break
        except Exception:
            continue
    if not corroborated:
        groups = ev.get("groups") or []
        if isinstance(groups, list) and len(groups) >= 2:
            try:
                bullets.append(f"• Corroboration: the split holds across {len(groups)} groups, not one outlier.")
                corroborated = True
            except Exception:
                pass
    if not corroborated:
        bullets.append("• Corroboration: no second split cleared the bar — this lead stands alone for now.")
    # Bullet 3: top rejected alternative from tested[].
    rej = next((t for t in tested if str(t.get("verdict") or "").lower() in ("rejected", "skipped")), None)
    if rej and rej.get("driver"):
        try:
            from core.semantic import humanize as _hh
            _dn = _hh(rej.get("driver"))
        except Exception:
            _dn = str(rej.get("driver"))
        bullets.append(f"• Ruled out: {_dn} did not clear the bar, so it is not recommended.")
    else:
        bullets.append("• Ruled out: other tested factors did not clear the bar.")
    bullets = bullets[:3]
    return "\n".join(bullets) + f"\n{_ASSOCIATION_DISCLAIMER}"


def _table_sampling(engine, dialect, table) -> dict:
    """Sampling provenance for LIMIT 5000 reads (additive details only)."""
    try:
        from core.schema_intel import _safe_name
        from sqlalchemy import text as _t
        with engine.connect() as conn:
            _n = conn.execute(_t(f"SELECT COUNT(*) FROM {_safe_name(table, dialect)}")).scalar()
        table_n = int(_n or 0)
    except Exception:
        return {}
    if table_n > SAMPLE_LIMIT:
        return {"sampled": True, "sample_n": int(SAMPLE_LIMIT), "table_n": table_n}
    return {"sampled": False, "sample_n": table_n, "table_n": table_n}


def validate_segment_value(engine, dialect, table, field, value, limit=100) -> dict:
    """Validate segment.value against SELECT DISTINCT LIMIT 100 + suggestions.

    Generic over any dimension (no hardcoded values). Returns
    {ok, distinct[], suggestions[]} — suggestions are case-insensitive
    substring matches, capped at 5.
    """
    try:
        from core.schema_intel import _safe_name
        rows = _exec(engine,
                     f"SELECT DISTINCT {_safe_name(field, dialect)} "
                     f"FROM {_safe_name(table, dialect)} "
                     f"WHERE {_safe_name(field, dialect)} IS NOT NULL "
                     f"LIMIT {int(limit)}")
        distinct = [r[0] for r in rows]
    except Exception:
        return {"ok": True, "distinct": [], "suggestions": []}
    try:
        if value in distinct or str(value) in [str(d) for d in distinct]:
            return {"ok": True, "distinct": [str(d) for d in distinct][:limit], "suggestions": []}
    except Exception:
        pass
    needle = str(value or "").lower()
    suggs = []
    for d in distinct:
        try:
            s = str(d)
            if needle and (needle in s.lower() or s.lower() in needle):
                suggs.append(s)
        except Exception:
            continue
        if len(suggs) >= 5:
            break
    if not suggs:
        suggs = [str(d) for d in distinct[:5]]
    return {"ok": False, "distinct": [str(d) for d in distinct][:limit], "suggestions": suggs}


_DIRECTION_WORDS_DOWN = ("down", "declin", "drop", "fall", "fell", "shrink", "slip", "weaken", "worsen", "bleed")
_DIRECTION_WORDS_UP = ("up", "increas", "grow", "growth", "rise", "risen", "rose", "improv", "rebound", "surge")


def parse_question_intent(question, intel, table, engine=None, dialect=None) -> dict:
    """Wire the dead `question` param to hypothesis ranking (generic).

    Parses metric + segment + direction words from the live schema values
    (no hardcoded category names): metric/driver names, dimension names,
    and DISTINCT dimension values (LIMIT 100 each, bounded). Returns
    {metric_hint, segment_field, segment_value, direction, ignored_reason}.
    Steering (not filtering): the caller boosts matching cards; a question
    with no schema match is documented as ignored, never silently applied.
    """
    q = str(question or "").strip()
    if not q:
        return {"metric_hint": None, "segment_field": None, "segment_value": None,
                "direction": None, "ignored_reason": "empty question — ignored"}
    low = q.lower()
    direction = None
    if any(w in low for w in _DIRECTION_WORDS_DOWN):
        direction = "down"
    elif any(w in low for w in _DIRECTION_WORDS_UP):
        direction = "up"
    metric_hint = None
    try:
        cands = list(intel.get_numeric_cols(table) or []) + list(_usable_measures(intel, table) or [])
        for c in cands:
            cl = str(c).lower()
            toks = [t for t in _name_tokens(cl) if len(t) > 2]
            if cl and cl in low:
                metric_hint = c
                break
            if any(t in low for t in toks):
                metric_hint = c
                break
    except Exception:
        metric_hint = None
    seg_field, seg_value = None, None
    try:
        dims = list(intel.find_all_dimensions(table) or [])[:4]
    except Exception:
        dims = []
    # Dimension NAME match first (e.g. "by category").
    for d in dims:
        dl = str(d).lower()
        toks = [t for t in _name_tokens(dl) if len(t) > 2]
        if dl and dl in low or any(t in low for t in toks):
            seg_field = d
            break
    # Dimension VALUE match from live DISTINCT values (generic, bounded).
    if engine is not None and dialect is not None:
        for d in dims:
            try:
                from core.schema_intel import _safe_name
                rows = _exec(engine,
                             f"SELECT DISTINCT {_safe_name(d, dialect)} "
                             f"FROM {_safe_name(table, dialect)} "
                             f"WHERE {_safe_name(d, dialect)} IS NOT NULL "
                             f"LIMIT 100")
                for r in rows:
                    try:
                        s = str(r[0])
                    except Exception:
                        continue
                    if len(s) < 2:
                        continue
                    if s.lower() in low:
                        seg_field, seg_value = d, s
                        break
                if seg_value:
                    break
            except Exception:
                continue
    if seg_field is None and seg_value is None and direction is None and metric_hint is None:
        return {"metric_hint": None, "segment_field": None, "segment_value": None,
                "direction": direction, "ignored_reason": "no schema match — ignored"}
    return {"metric_hint": metric_hint, "segment_field": seg_field, "segment_value": seg_value,
            "direction": direction, "ignored_reason": None}


def _apply_question_intent(pres, intent) -> list:
    """Steer ranking toward the asked segment (additive boost, never filter)."""
    try:
        intent = intent if isinstance(intent, dict) else {}
        seg_val = str(intent.get("segment_value") or "").lower()
        seg_field = str(intent.get("segment_field") or "").lower()
        direction = str(intent.get("direction") or "").lower()
        if not seg_val and not seg_field and not direction:
            return pres
        out = []
        for r in (pres or []):
            if not isinstance(r, dict):
                out.append(r)
                continue
            try:
                boost = 0.0
                ev = r.get("evidence") if isinstance(r.get("evidence"), dict) else {}
                hay = " ".join([str(ev.get("top") or ""), str(ev.get("bottom") or ""),
                                str(ev.get("median") or ""), str(ev.get("driver") or ""),
                                str((ev.get("segment") or {}).get("value") if isinstance(ev.get("segment"), dict) else ""),
                                str(r.get("title") or "")]).lower()
                if seg_val and seg_val in hay:
                    boost += 5.0
                elif seg_field and seg_field in hay:
                    boost += 2.0
                if direction and str(ev.get("direction") or "").lower() == direction:
                    boost += 1.0
                if boost:
                    r = dict(r)
                    try:
                        r["score"] = round(float(r.get("score", 0) or 0) + boost, 1)
                    except (TypeError, ValueError):
                        pass
                    _pr = dict(r.get("priority_score", {}) if isinstance(r.get("priority_score"), dict) else {})
                    out.append(r)
                else:
                    out.append(r)
            except Exception:
                out.append(r)
        # Re-sort within tiers so the asked segment surfaces first.
        try:
            return _rank_coverage(out)
        except Exception:
            return out
    except Exception:
        return pres


def _details_for(rec: dict, sampling=None) -> dict:
    """details:{evidence_pass, checks[], numbers{}, limitations[], +additive}.

    Additive extension: confidence_label/meaning/what_would_raise_it,
    score_breakdown/join_ref nesting, sampling provenance. Raw stats live
    here (analyst expander), never required in user-visible text.
    """
    rec = rec if isinstance(rec, dict) else {}
    lims = rec.get("limitations") if isinstance(rec.get("limitations"), list) else []
    det = {
        "evidence_pass": _evidence_pass_for(rec),
        "checks": _checks_for(rec),
        "numbers": _numbers_for(rec),
        "limitations": [str(x) for x in lims],
    }
    try:
        det["confidence_label"] = confidence_label_for(rec.get("confidence"))
        det["confidence_meaning"] = _confidence_meaning_for(rec.get("confidence"))
        det["what_would_raise_it"] = _what_would_raise_for(rec.get("confidence"))
    except Exception:
        pass
    try:
        if isinstance(rec.get("score_breakdown"), dict):
            det["score_breakdown"] = dict(rec.get("score_breakdown"))
    except Exception:
        pass
    try:
        _jr = rec.get("join_ref", rec.get("join"))
        if _jr is not None:
            det["join_ref"] = _jr
    except Exception:
        pass
    try:
        # Sampling provenance: always disclose when known (LIMIT 5000 reads),
        # not only when truncation kicks in. When sampled, note the ORDER BY
        # bias: the deterministic ordered sample favors the sort head, so the
        # tail of the distribution is under-represented.
        if isinstance(sampling, dict) and sampling.get("sample_n") is not None:
            det["sampled"] = bool(sampling.get("sampled"))
            try:
                det["sample_n"] = int(sampling.get("sample_n"))
            except (TypeError, ValueError):
                pass
            try:
                if sampling.get("table_n") is not None:
                    det["table_n"] = int(sampling.get("table_n"))
            except (TypeError, ValueError):
                pass
            if det.get("sampled") is True:
                det["sample_note"] = ("Analysis used a bounded ordered sample "
                                      "(first rows in sort order up to the row limit); "
                                      "tail values may be under-represented.")
    except Exception:
        pass
    return det


def build_display(rec: dict, tested=None, supporting=None) -> dict:
    """display:{title, action_text, outcome_text, confidence_label, why_summary, problem_text, why_text}.

    Additive: problem_text/why_text are new; the five legacy keys are kept.
    why_text is built from tested[] + segment/monitor/investigate inputs.
    """
    rec = rec if isinstance(rec, dict) else {}
    return {
        "title": str(rec.get("title") or "Recommendation"),
        "action_text": _action_text_for(rec),
        "outcome_text": _outcome_text_for(rec),
        "confidence_label": confidence_label_for(rec.get("confidence")),
        "why_summary": _why_summary_for(rec),
        "problem_text": _problem_text_for(rec),
        "why_text": _why_text_for(rec, tested=tested, supporting=supporting),
    }


def attach_display(rec: dict, tested=None, supporting=None, sampling=None) -> dict:
    """Attach the display-safe contract to one prescription (additive)."""
    if not isinstance(rec, dict):
        return rec
    try:
        rec["display"] = build_display(rec, tested=tested, supporting=supporting)
        rec["why_summary"] = rec["display"]["why_summary"]
        rec["confidence_label"] = rec["display"]["confidence_label"]
        rec["problem_text"] = rec["display"]["problem_text"]
        rec["why_text"] = rec["display"]["why_text"]
        rec["details"] = _details_for(rec, sampling=sampling)
        rec["evidence_pass"] = rec["details"]["evidence_pass"]
    except Exception:
        log.debug("attach_display failed", exc_info=True)
    return rec


# ---------------------------------------------------------------------------
# Computed refusal alternatives (DYNAMIC ONLY): numeric controllable drivers
# for the same metric (existing controllability classifier + a bounded
# n>=MIN_N quality-gate probe), else the table's top segment-focus dims.
# Never a static list — every entry is measured against the live schema.
# ---------------------------------------------------------------------------
def compute_refusal_alternatives(engine, dialect, intel, table, metric,
                                 exclude=None, limit=3) -> list:
    """[{metric, driver, label}] alternatives, max `limit` (default 3)."""
    alts, seen = [], set()
    try:
        tables = list(getattr(intel, "tables", []) or [])
    except Exception:
        tables = []
    if intel is None or not table or not metric or table not in tables:
        return alts
    try:
        measures = [c for c in _usable_measures(intel, table)
                    if c != metric and str(c).lower() != str(exclude or "").lower()]
    except Exception:
        measures = []
    for cand in measures[:8]:  # bounded probe budget — never a full scan
        if len(alts) >= limit:
            break
        try:
            kind = _col_kind(intel, table, cand) or "numeric"
            level, _ = classify_driver(cand, kind)
            if level not in ("controllable", "potentially_controllable"):
                continue
            if classify_semantic_role(cand, kind) in NON_LEVER_ROLES:
                continue
            rows = _fetch_pairs(engine, dialect, table, cand, metric, limit=500)
            n = len(rows or [])
            if n < MIN_N:  # quality-gate precheck: enough rows to test
                continue
            key = str(cand).lower()
            if key in seen:
                continue
            seen.add(key)
            alts.append({"metric": metric, "driver": cand,
                         "label": f"Try {_sem_humanize(cand)} for {_sem_humanize(metric)}"})
        except Exception:
            continue
    if len(alts) < limit:
        try:
            dims = list(intel.find_all_dimensions(table) or [])[:3]
        except Exception:
            dims = []
        for dim in dims:
            if len(alts) >= limit:
                break
            key = str(dim).lower()
            if key in seen or key == str(metric or "").lower():
                continue
            seen.add(key)
            alts.append({"metric": metric, "driver": dim,
                         "label": f"Compare {_sem_humanize(metric)} across {_sem_humanize(dim)}"})
    return alts[:limit]


def _infer_refusal_code(out: dict) -> str:
    """Reuse an existing refusal code (never invent one)."""
    out = out if isinstance(out, dict) else {}
    if out.get("code"):
        return str(out["code"])
    reason = f"{out.get('reason') or ''} {out.get('detail') or ''}".lower()
    if ("controllab" in reason or "intervention" in reason
            or "direction" in reason or "scenario" in reason):
        return CODE_SCENARIO_UNSUPPORTED
    if "stay length" in reason or "join" in reason or "link" in reason:
        return CODE_SCENARIO_UNSUPPORTED
    return CODE_INSUFFICIENT_DATA


def build_refusal(out: dict, *, engine=None, dialect=None, intel=None,
                  table=None, metric=None, driver=None) -> dict:
    """refusal:{headline, reason_code, alternatives[]} — all dynamic."""
    out = out if isinstance(out, dict) else {}
    m = metric or out.get("metric")
    d = driver or out.get("driver")
    headline = f"No supported action for {_sem_display(m)}"
    if d:
        headline += f" using {_sem_display(d, 'this factor')}"
    headline += "."
    try:
        alternatives = compute_refusal_alternatives(
            engine, dialect, intel, table or out.get("table"), m, exclude=d)
    except Exception:
        alternatives = []
    return {"headline": headline,
            "reason_code": _infer_refusal_code(out),
            "alternatives": alternatives[:3]}


def _finalize_output(out: dict, *, engine=None, dialect=None, intel=None,
                      table=None, metric=None, driver=None, question=None) -> dict:
    """Attach display/details to prescriptions + refusal to refusals.

    Additive only: existing keys are never removed or renamed here — the
    routes layer owns internal-key stripping. WHY is built from tested[] +
    segment-focus/monitor/investigate outputs passed as inputs; question
    intent (when given) steers ranking and is documented in diagnostics.
    """
    if not isinstance(out, dict):
        return out
    # Wire the `question` param (previously dead): parse metric/segment/
    # direction generically from schema values; steer ranking or document
    # the ignore. Additive: diagnostics.question_intent only.
    try:
        if question is not None and engine is not None and intel is not None and table:
            _intent = parse_question_intent(question, intel, table, engine=engine, dialect=dialect)
            try:
                _diag = out.get("diagnostics")
                if not isinstance(_diag, dict):
                    out["diagnostics"] = _diag = {}
                _diag["question_intent"] = _intent
            except Exception:
                pass
            try:
                _pres0 = out.get("prescriptions")
                if isinstance(_pres0, list) and _pres0 and _intent.get("ignored_reason") is None:
                    out["prescriptions"] = _apply_question_intent(_pres0, _intent)
            except Exception:
                pass
    except Exception:
        log.debug("question intent wiring failed", exc_info=True)
    # Sampling provenance (LIMIT 5000) — one bounded COUNT per analyze.
    # Always attached (sampled True/False) so callers can tell a full scan
    # from a truncated ordered sample.
    _sampling = {}
    try:
        if engine is not None and dialect is not None and table:
            _sampling = _table_sampling(engine, dialect, table) or {}
    except Exception:
        _sampling = {}
    pres = out.get("prescriptions")
    if isinstance(pres, list):
        # tested[] trail for WHY inputs (diagnostics.tested preferred).
        _tested = []
        try:
            _diag2 = out.get("diagnostics") if isinstance(out.get("diagnostics"), dict) else {}
            _tested = _diag2.get("tested", out.get("tested", [])) or []
        except Exception:
            _tested = []
        for r in pres:
            # Repair incomplete display contracts too, not just missing ones:
            # a partial/{} display from an older path must gain every key
            # the UI reads (title, action_text, outcome_text,
            # confidence_label, why_summary, problem_text, why_text).
            _needs_display = True
            try:
                _d = r.get("display") if isinstance(r, dict) else None
                _needs_display = not (
                    isinstance(_d, dict) and all(
                        k in _d for k in ("title", "action_text", "outcome_text",
                                         "confidence_label", "why_summary",
                                         "problem_text", "why_text")))
            except Exception:
                _needs_display = True
            if isinstance(r, dict) and _needs_display:
                try:
                    _supporting = [s for s in pres if isinstance(s, dict) and s is not r]
                except Exception:
                    _supporting = []
                attach_display(r, tested=_tested, supporting=_supporting, sampling=_sampling or None)
    scens = out.get("scenarios")
    if isinstance(scens, list) and out.get("ok"):
        for s in scens:
            if isinstance(s, dict) and "display" not in s:
                try:
                    s["display"] = _scenario_display(
                        s, metric or out.get("metric"), driver or out.get("driver"),
                        n=out.get("n"))
                except Exception:
                    log.debug("scenario display failed", exc_info=True)
    if out.get("status") == "no_prescription" or out.get("ok") is False:
        if "refusal" not in out:
            try:
                out["refusal"] = build_refusal(
                    out, engine=engine, dialect=dialect, intel=intel,
                    table=table, metric=metric, driver=driver)
            except Exception:
                log.debug("refusal build failed", exc_info=True)
    return out


def _scenario_display(item: dict, metric, driver, n=None) -> dict:
    """Display contract for one recomputed scenario (read-only result)."""
    item = item if isinstance(item, dict) else {}
    M, D = _sem_display(metric), _sem_display(driver, "this factor")
    ch = item.get("change") if isinstance(item.get("change"), dict) else {}
    try:
        pct = float(ch.get("value")) if ch.get("operation") == "percent" else None
    except (TypeError, ValueError):
        pct = None
    if pct is None:
        try:
            pct = float(item.get("driver_pct")) if item.get("driver_pct") is not None else 0.0
        except (TypeError, ValueError):
            pct = 0.0
    action_text = f"Apply {'+' if pct >= 0 else ''}{_trim_num(pct, 1)}% to {D}"

    def _num(v):
        try:
            f = float(v)
            return f if f == f else None
        except (TypeError, ValueError):
            return None

    base, scen, rel = _num(item.get("baseline")), _num(item.get("scenario")), _num(item.get("rel_pct"))
    if base is not None and scen is not None:
        outcome = f"{_human_amount(base, metric)} → {_human_amount(scen, metric)}"
        if rel is not None:
            outcome += f" ({_fmt_pct_short(rel)} vs baseline)"
    elif rel is not None:
        outcome = f"{_fmt_pct_short(rel)} vs baseline"
    else:
        outcome = "No sized outcome — review only."
    risk = str(item.get("risk") or "").lower()
    conf = {"low": "High", "medium": "Moderate", "high": "Early"}.get(risk, "Moderate")
    n_txt = f"{int(n):,} past observations" if n else "past data"
    return {"title": str(item.get("title") or f"Scenario: {D} {_trim_num(pct, 1)}%"),
            "action_text": action_text,
            "outcome_text": outcome,
            "confidence_label": conf,
            "why_summary": (f"Modeled from {n_txt} — a simulation from the observed link "
                            f"between {D} and {M}, not a promise.")}


def analyze(engine, dialect, intel, table, metric, driver=None,
            dimensions=None, segment=None, question=None, priors=None):
    """Full prescription pipeline + display-safe contract (additive wrapper)."""
    out = _analyze_core(engine, dialect, intel, table, metric, driver=driver,
                        dimensions=dimensions, segment=segment,
                        question=question, priors=priors)
    try:
        return _finalize_output(out, engine=engine, dialect=dialect,
                                intel=intel, table=table, metric=metric,
                                driver=driver, question=question)
    except Exception:
        log.debug("analyze display finalize failed", exc_info=True)
        return out


def run_scenario(engine, dialect, intel, table, metric, driver, changes, filt=None,
                 join=None):
    """Read-only scenario recompute + display-safe contract (additive wrapper)."""
    out = _run_scenario_core(engine, dialect, intel, table, metric, driver,
                             changes, filt=filt, join=join)
    try:
        return _finalize_output(out, engine=engine, dialect=dialect,
                                intel=intel, table=table, metric=metric,
                                driver=driver)
    except Exception:
        log.debug("run_scenario display finalize failed", exc_info=True)
        return out


def analyze_document(chunks, question, doc_name):
    """Qualitative document prescriptions + display-safe contract (wrapper)."""
    out = _analyze_document_core(chunks, question, doc_name)
    try:
        out = _finalize_output(out, metric=None, driver=None)
        if isinstance(out, dict) and out.get("status") == "no_prescription" \
                and isinstance(out.get("refusal"), dict):
            # Document refusals have no engine to probe: keep the computed
            # (empty) alternatives honest instead of static suggestions.
            out["refusal"]["reason_code"] = _infer_refusal_code(out)
        return out
    except Exception:
        log.debug("analyze_document display finalize failed", exc_info=True)
        return out
