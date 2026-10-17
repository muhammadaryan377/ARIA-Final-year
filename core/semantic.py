"""ARIA unified semantic layer helpers (t.txt Phase 1-2, sections 8-13).

Pure, dependency-free helpers shared by schema/goal/query routes:
- semantic type classification for arbitrary column names/types
- structured result envelope (result/evidence/confidence/provenance)
- identifier validation for user overrides

Deterministic only — no LLM calls here (truth layer, RL must not alter it).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_NUMERIC_HINTS = ("int", "float", "double", "decimal", "numeric", "number", "real")
_DATE_HINTS = ("date", "time", "timestamp", "datetime")
_BOOL_HINTS = ("bool",)
_ID_TOKENS = frozenset({"id", "uuid", "guid", "key", "code", "number", "no"})
_ID_SUFFIXES = ("_id", "_key", "_code", "_guid", "_uuid", "_no")


def _has_id_token(name: str) -> bool:
    """Whole-token / suffix identifier match (never bare substrings).

    "order_id" -> True; "valid"/"innovation"/"north"/"snow"/"grid" -> False
    ("id" inside "valid", "no" inside "innovation" must not trigger).
    """
    low = str(name or "").lower()
    # Split camelCase so MediaTypeId behaves like media_type_id.
    low = __import__("re").sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name or "")).lower()
    for tok in __import__("re").split(r"[^a-z0-9]+", low):
        if tok in _ID_TOKENS:
            return True
    return low.endswith(_ID_SUFFIXES)
_MONEY_HINTS = ("price", "cost", "amount", "revenue", "profit", "salary", "total", "payment")
_PCT_HINTS = ("pct", "percent", "rate", "ratio")
_GEO_HINTS = ("city", "country", "region", "state", "zip", "postal", "lat", "lon", "lng", "address")


def is_year_like(name: str) -> bool:
    """True for calendar-year columns (release_year, model_year, year).

    Years are labels on a timeline, not summable measures: SUM/AVG over
    them produces nonsense (total 17,746,905 "years"). Generic, name-only
    so every measure gate (dashboard, forecast eligibility, prescriptions,
    goal grounding) shares one rule. Year columns remain fully usable as
    dimensions and trend axes — only SUM/AVG candidacy is removed.
    """
    low = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name or "")).lower()
    low_nosep = re.sub(r"[^a-z0-9]+", "", low)
    if low in ("year", "years", "yr", "yrs"):
        return True
    if low.endswith(("_year", "_years")):
        return True
    # Separator-less variants (fiscalyear, releaseyear).
    if low_nosep.endswith("year") and len(low_nosep) > 4:
        return True
    return False


def is_valid_identifier(name: str, max_len: int = 64) -> bool:
    """Conservative identifier check for table/column names from user input."""
    return bool(name) and len(name) <= max_len and bool(IDENT_RE.match(name))


# Generic abbreviation expansions for user-facing labels only. These are
# common data-shorthand tokens (not business/schema words), so expansion
# stays generic: ``total_amt`` → ``Total Amount``, ``room_id`` → ``Room``.
# Unknown tokens pass through untouched; raw names remain available to
# callers for technical diagnostics.
_ABBR = {
    "amt": "Amount",
    "qty": "Quantity",
    "avg": "Average",
    "tot": "Total",
    "val": "Value",
    "num": "Number",
    "cnt": "Count",
    "pct": "Percent",
    "desc": "Description",
    "cat": "Category",
    "yr": "Year",
    "mon": "Month",
}


def humanize(name: str) -> str:
    """Turn a physical column/table name into natural language.

    Generic formatting only (no business dictionary): underscores become
    spaces, common shorthand tokens are expanded (``amt`` → ``Amount``),
    words are title-cased, trailing _id/_key/_code markers are dropped.
    ``order_date`` → ``Order Date``. Raw snake_case must never be shown as
    a primary user-facing label — use this helper; keep raw names for
    technical diagnostics only.
    """
    text = re.sub(r"[_-]+", " ", str(name or "")).strip()
    text = re.sub(r"\s+", " ", text)
    words = [w for w in text.split(" ") if w]
    drop_tail = {"id", "key", "code", "uuid", "guid", "pk", "fk"}
    while len(words) > 1 and words[-1].lower() in drop_tail:
        words.pop()
    words = [_ABBR.get(w.lower(), w) for w in words]
    return " ".join(w[:1].upper() + w[1:] for w in words) or str(name or "")


def display_name(raw: str | None, fallback: str = "this metric") -> str:
    """Primary user-facing label for a column/metric/driver name.

    Returns the humanized label, or the neutral ``fallback`` when there is
    nothing usable to show. The raw name is intentionally NOT returned —
    callers that need it for diagnostics must read the raw field itself.
    """
    if not raw or not str(raw).strip():
        return fallback
    return humanize(str(raw))


def kpi_label(agg: str, column: str) -> str:
    """Natural-language KPI label from aggregation + column (no table names)."""
    human = humanize(column)
    agg = (agg or "").upper()
    if agg == "AVG":
        return f"Average {human}"
    if agg == "DISTINCT":
        return f"{human} Count"
    if agg == "COUNT":
        return "Total Records"
    return f"Total {human}"


def classify_semantic_type(column_name: str, db_type: str = "") -> str:
    """Classify a column into a semantic type without assuming fixed schemas.

    Returns one of: identifier, measure, money, percentage, date, boolean,
    category, geography, text.
    """
    name = (column_name or "").lower()
    dtype = (db_type or "").lower()
    # Whole-word tokens (camelCase split first): every hint below matches
    # full tokens only — never bare substrings ("separate" is not a rate,
    # "salon" is not a lon, "candidate" is not a date).
    import re as _re
    toks = set(
        t for t in _re.split(
            r"[^a-z0-9]+",
            _re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(column_name or "")).lower(),
        )
        if t
    )
    if any(h in dtype for h in _BOOL_HINTS) or dtype in ("boolean", "bit"):
        return "boolean"
    if any(h in dtype for h in _DATE_HINTS):
        return "date"
    if any(h in dtype for h in _NUMERIC_HINTS):
        if _has_id_token(column_name):
            return "identifier"
        if not _has_dimension_veto(column_name):
            if toks & set(_MONEY_HINTS):
                return "money"
            if toks & set(_PCT_HINTS):
                return "percentage"
        return "measure"
    # Pass the original-case name so camelCase (MediaTypeId) splits correctly.
    if _has_id_token(column_name):
        return "identifier"
    if any(h in dtype for h in _DATE_HINTS) or toks & {"date", "time", "day", "month", "year", "created", "updated", "timestamp", "datetime", "at"}:
        if not _has_technical_signal(column_name):
            return "date"
    if toks & (set(_MONEY_HINTS) | set(_PCT_HINTS)) and not _has_dimension_veto(column_name):
        return "measure"
    if toks & set(_GEO_HINTS):
        return "geography"
    # Low-cardinality textual names behave as categories; default to text.
    return "text"


# ---------------------------------------------------------------------------
# hyp.txt PHASE 1 — canonical Decision Center field roles (additive).
#
# Required vocabulary (#11): IDENTIFIER, FOREIGN_KEY, DATE, DIMENSION,
# CATEGORY, MEASURE, BUSINESS_MEASURE, TECHNICAL_MEASURE, DERIVED_METRIC,
# TEXT, UNKNOWN. Existing classify_semantic_type() above is UNCHANGED;
# this is the Decision-eligibility layer built on generic semantic signals
# (never per-column / per-dataset name exclusions).
# ---------------------------------------------------------------------------

FIELD_ROLES = (
    "IDENTIFIER",
    "FOREIGN_KEY",
    "DATE",
    "DIMENSION",
    "CATEGORY",
    "MEASURE",
    "BUSINESS_MEASURE",
    "TECHNICAL_MEASURE",
    "DERIVED_METRIC",
    "TEXT",
    "UNKNOWN",
)

MEASURE_FAMILY = frozenset({
    "MEASURE", "BUSINESS_MEASURE", "TECHNICAL_MEASURE", "DERIVED_METRIC",
})

NON_OUTCOME_ROLES = frozenset({
    "TECHNICAL_MEASURE", "IDENTIFIER", "FOREIGN_KEY", "DATE", "UNKNOWN",
})

NON_LEVER_ROLES = frozenset({
    "TECHNICAL_MEASURE", "IDENTIFIER", "FOREIGN_KEY", "DATE",
    "DIMENSION", "CATEGORY", "TEXT", "UNKNOWN",
})

# Generic telemetry/system-measurement vocabulary (semantic category, not a
# column-name exclusion list): duration/size/compute/storage/network tokens.
# Matched on tokens/substrings generically so ANY dataset's telemetry
# (bytes, millisec, latency, cpu, ...) resolves to TECHNICAL_MEASURE
# without ever naming a concrete column.
_TECHNICAL_TOKENS = (
    "byte", "bytes", "millisec", "millisecond", "microsec", "nanosec",
    "latency", "ping", "ttl", "cpu", "gpu", "memory", "heap", "disk",
    "iops", "cache", "packet", "bandwidth", "throughput", "payload",
    "query_time", "query_duration", "execution_time", "exec_time",
    "response_time", "elapsed", "duration_ms", "duration_sec",
    "wait_time", "load_avg", "thread", "kernel", "uptime", "file_size",
    "filesize", "network_io", "io_wait",
)

# Generic derived-metric signals: ratios / per-unit / averaged constructs.
# These remain MEASURE-family (measure_used_as_dimension still fires when
# one is used as a segment), but actionability must be established
# separately (derived != actionable).
_DERIVED_HINTS = (
    "per_", "_per_", "per-night", "rate", "ratio", "pct", "percent",
    "average", "avg_", "_avg", "length_of_stay", "length-of-stay",
    "derived",
)

_BUSINESS_MEASURE_HINTS = (
    "price", "pricing", "cost", "amount", "revenue", "profit", "margin",
    "sales", "total", "payment", "spend", "spending", "budget", "discount",
    "quantity", "qty", "inventory", "stock", "wage", "salary", "fee",
    "fare", "freight", "tuition", "premium", "booking", "demand",
    "supply", "volume", "traffic", "conversion", "retention",
)

_REF_TAILS = ("id", "key", "code", "no", "num", "uuid", "guid", "pk", "fk")


def _field_tokens(name: str) -> list:
    import re as _re
    return [t for t in _re.split(r"[^a-z0-9]+", str(name or "").lower()) if t]


def _has_technical_signal(name: str) -> bool:
    low = str(name or "").lower()
    toks = set(_field_tokens(low))
    for h in _TECHNICAL_TOKENS:
        h = h.lower()
        if "_" in h or len(h) >= 8:
            if h in low:
                return True
        elif h in toks:
            return True
    return False


def _has_derived_signal(name: str) -> bool:
    # Whole-token match only: bare substrings misfire ("separate" has
    # "rate", "strategy" has "rate", "generate" has "rate", "ratio" in
    # "separation"). Multiword hints ("per night") still substring-match.
    toks = set(_field_tokens(name))
    low = str(name or "").lower()
    for h in _DERIVED_HINTS:
        h = h.lower()
        if "_" in h or " " in h or "-" in h:
            if h.replace("-", " ") in low.replace("_", " ").replace("-", " "):
                return True
        elif h in toks:
            return True
    return False


def _has_business_measure_signal(name: str) -> bool:
    low = str(name or "").lower()
    toks = set(_field_tokens(low))
    for h in _BUSINESS_MEASURE_HINTS:
        h = h.lower()
        if h in toks:
            return True
        if len(h) >= 8 or "_" in h:
            if h in low:
                return True
    return False


# Tokens that mark a column as a LABEL, not a number — even when a sibling
# token looks monetary ("payment_method" is a category, "payment_amount" is
# money). Token equality only, so "classified" does not trip on "class".
_DIMENSION_VETO_TOKENS = frozenset({
    "method", "mode", "type", "status", "name", "label", "category",
    "segment", "region", "group", "kind", "tag", "class", "code",
    "key", "id", "description", "title",
})


def _has_dimension_veto(name: str) -> bool:
    """True when a name token is intrinsically dimensional.

    A vetoed column is never a measure by NAME alone ("payment_method",
    "rate_type", "Ship Mode"). Real numeric dtypes still classify as
    measures through the dtype branches — the veto only demotes
    name-inferred TEXT/ambiguous columns to dimensions, never real numbers.
    """
    return bool(set(_field_tokens(name)) & _DIMENSION_VETO_TOKENS)


def _is_identifier_flavored(name: str) -> bool:
    low = str(name or "").lower().strip()
    if not low:
        return False
    if low in ("id", "key", "code", "uuid", "guid", "pk", "fk"):
        return True
    if low.endswith("_id"):
        return True
    # NOTE: bare endswith("id") was removed — it misclassified ordinary
    # words ("valid", "grid"). Separator-less forms ("storeid") are only
    # accepted for the unambiguous uuid/guid tails below.
    tail = low.split("_")[-1] if "_" in low else low
    if tail in _REF_TAILS:
        return True
    try:
        from schema_engine.lexical import is_ref_flavored as _is_ref
        return bool(_is_ref(name))
    except Exception:
        return False


def classify_field_role(column_name: str, db_type: str = "",
                         *, is_pk: bool = False, is_fk: bool = False,
                         distinct_count: int | None = None) -> str:
    """Canonical hyp.txt #11 role for Decision eligibility (deterministic).

    Order: FOREIGN_KEY (explicit metadata wins) > IDENTIFIER > DATE >
    TECHNICAL_MEASURE > DERIVED_METRIC > BUSINESS_MEASURE > MEASURE >
    DIMENSION/CATEGORY > TEXT > UNKNOWN.

    Guarantees for the regression suite:
    - ``Total Amt``/``total_amt`` -> BUSINESS_MEASURE (MEASURE family)
    - ``room_id`` with is_fk=True -> FOREIGN_KEY, else IDENTIFIER
    - ``Bytes``/``Milliseconds`` -> TECHNICAL_MEASURE (generic telemetry
      semantics, never a per-column exclusion)
    """
    name = str(column_name or "")
    low = name.lower()
    dtype = str(db_type or "").lower()
    if is_fk:
        return "FOREIGN_KEY"
    if is_pk or _is_identifier_flavored(name):
        return "IDENTIFIER"
    if any(h in dtype for h in _DATE_HINTS):
        return "DATE"
    # Whole-token date words only ("order_date", "created_at") — bare
    # substrings misfire on ordinary words ("candidate", "update",
    # "mandate"). CamelCase is split first (CreatedAt -> created_at).
    _dtoks = set(
        t for t in __import__("re").split(
            r"[^a-z0-9]+",
            __import__("re").sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name or "")).lower(),
        )
        if t
    )
    if (_dtoks & {"date", "time", "day", "month", "year", "created", "updated",
                  "timestamp", "datetime", "at"}
            or low.endswith(("_date", "_time", "_at", "_on", "_ts"))
            or low in ("date", "time", "datetime", "timestamp")) \
            and _is_identifier_flavored(name) is False:
        # 'time' alone is a date signal unless it is part of a technical
        # compound (response_time, execution_time -> technical wins below).
        if not _has_technical_signal(name):
            return "DATE"
    if _has_technical_signal(name):
        return "TECHNICAL_MEASURE"
    if _has_derived_signal(name):
        return "DERIVED_METRIC"
    if _has_business_measure_signal(name) and not _has_dimension_veto(name):
        return "BUSINESS_MEASURE"
    if any(h in dtype for h in _NUMERIC_HINTS):
        return "MEASURE"
    # Whole-token geography only ("city", "lon", "lat", "state" as full
    # tokens): substrings misfire ("velocity" has "city", "salon" has
    # "lon", "plate" has "lat", "estate" has "state").
    if set(_field_tokens(name)) & set(_GEO_HINTS):
        return "DIMENSION"
    if distinct_count is not None:
        try:
            if int(distinct_count) <= 50:
                return "CATEGORY"
        except (TypeError, ValueError):
            pass
    # Textual low-cardinality names already handled via DIMENSION above;
    # fall back on the legacy classifier for text/category nuance.
    try:
        legacy = classify_semantic_type(name, db_type)
    except Exception:
        legacy = "text"
    if legacy == "identifier":
        return "IDENTIFIER"
    if legacy == "date":
        return "DATE"
    if legacy == "category":
        return "CATEGORY"
    if legacy == "geography":
        return "DIMENSION"
    if legacy in ("measure", "money", "percentage"):
        return "MEASURE"
    if legacy == "text":
        return "TEXT"
    return "UNKNOWN"


def is_measure_role(role: str) -> bool:
    return str(role or "").upper() in MEASURE_FAMILY


def is_technical_role(role: str) -> bool:
    return str(role or "").upper() == "TECHNICAL_MEASURE"


def is_identifier_like(role: str) -> bool:
    return str(role or "").upper() in ("IDENTIFIER", "FOREIGN_KEY")


def is_dimension_like(role: str) -> bool:
    return str(role or "").upper() in ("DIMENSION", "CATEGORY")


def build_result_envelope(
    result: object = None,
    *,
    evidence: list | None = None,
    source: str = "unknown",
    confidence: str = "medium",
    assumptions: list | None = None,
    warnings: list | None = None,
    errors: list | None = None,
    methodology: str = "",
    provenance: dict | None = None,
) -> dict:
    """Build the structured agent-output contract (t.txt section 7).

    Additive helper — callers spread existing payloads and add this envelope
    so the frontend can render trustworthy results consistently.
    """
    return {
        "result": result,
        "evidence": evidence or [],
        "source": source,
        "confidence": confidence,
        "assumptions": assumptions or [],
        "warnings": warnings or [],
        "errors": errors or [],
        "methodology": methodology,
        "provenance": provenance or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
