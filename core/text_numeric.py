"""Text-stored numerics: python-side diagnostics + dialect-aware SQL coercion.

Some databases store numbers as TEXT (e.g. ``discounted_price`` holding
``'\\u20b91,299'``, ``discount_percentage`` holding ``'15%'``). The CSV ingest
path (``csv_handler.py``) already strips currency/percent artefacts; the
RELATIONAL path had no equivalent, so SUM/AVG over those columns failed.

This module is fully schema-agnostic (no table/column literals): it exposes

- :func:`is_text_numeric` / :func:`diagnose_text_numeric` -- python-side
  probe of a DISTINCT sample (currency symbols ``\\u20b9$\\u20ac\\u00a3\\u00a5``,
  commas, ``%`` and whitespace are stripped; ``>=80%`` of the non-null
  sample must parse as ``float``);
- :func:`coercion_sql` -- a SAFE ``CAST`` expression for a quoted column in
  each of the three supported dialects (unparsable values become ``NULL``,
  never invented numbers; percent values stay as-is numerically, i.e.
  ``15%`` -> ``15``, with the unit noted in the diagnostics metadata).

GROUP BY / WHERE on such columns must keep using the RAW column; only
SUM/AVG/MIN/MAX wrap the coercion expression.
"""

from __future__ import annotations

import re
import threading

# Currency symbols stripped both python-side and inside the SQL REPLACE
# chains. Generic set only -- never table/column specific.
CURRENCY_SYMBOLS = ("\u20b9", "$", "\u20ac", "\u00a3", "\u00a5")

# Characters removed by the nested REPLACE fallback chains (currency +
# thousand-separator comma + percent + blank space).
REPLACE_CHAIN_CHARS = ("\u20b9", "$", "\u20ac", "\u00a3", "\u00a5", ",", "%", " ")

COERCION_THRESHOLD = 0.8

_WS_RE = re.compile(r"\s+")

# Generic measure-like name fragments (substrings, so ``discounted`` matches
# via ``discount``). Business-concept vocabulary only -- no table/column
# literals, so this stays schema-agnostic.
_MEASURE_LIKE_FRAGMENTS = frozenset({
    "price", "pric", "cost", "amount", "total", "value", "sales", "sale",
    "revenue", "profit", "margin", "income", "turnover", "spend",
    "quantity", "qty", "unit", "volume", "count", "number", "num",
    "average", "avg", "mean", "sum", "rating", "rate", "score",
    "discount", "percent", "pct", "fee", "charge", "payment", "balance",
    "salary", "wage", "gmv",
})

# Regexp support per dialect. PostgreSQL/MySQL prefer REGEXP_REPLACE; when a
# deployment lacks it the nested REPLACE chain is used instead. The flag is
# resolved once per process (see ensure_dialect_support), never per query.
_REGEXP_SUPPORT: dict[str, bool] = {"postgresql": True, "mysql": True}
_REGEXP_PROBED: set[str] = set()

# In-memory probe cache for the process: (db_uri, table, column) -> record.
# Key space is unbounded (one entry per probed triple, forever), so the dict is
# capped LRU-style and guarded by a lock: probes run from request handlers and
# background threads alike. Touch-on-read keeps hit behavior identical for
# entries within the cap; only the least-recently-used entry is dropped when
# full (house style: api/autocomplete_routes.py _INDEX_CACHE).
_COERCED_CACHE: dict[tuple[str, str, str], dict] = {}
_COERCED_CACHE_MAX = 500
_COERCED_CACHE_LOCK = threading.Lock()


def clean_text_numeric_value(value) -> tuple[str | None, bool, bool]:
    """Strip a raw cell to its numeric-looking core.

    Returns ``(cleaned, percent_flag, currency_flag)`` where ``cleaned`` is
    ``None`` for null-ish input. Percent values are NOT divided (``15%`` ->
    ``'15'``); the caller notes the unit from ``percent_flag``.
    """
    if value is None:
        return None, False, False
    s = str(value).strip()
    if not s:
        return None, False, False
    percent_flag = "%" in s
    currency_flag = any(c in s for c in CURRENCY_SYMBOLS)
    t = s
    for c in CURRENCY_SYMBOLS:
        t = t.replace(c, "")
    t = t.replace(",", "").replace("%", "")
    t = _WS_RE.sub("", t)
    if not t:
        return None, percent_flag, currency_flag
    return t, percent_flag, currency_flag


def try_parse_text_number(value) -> float | None:
    """Parse one raw cell to float after cleaning; None when unparsable."""
    cleaned, _, _ = clean_text_numeric_value(value)
    if cleaned is None:
        return None
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def diagnose_text_numeric(values, threshold: float = COERCION_THRESHOLD) -> dict:
    """Coercion diagnostics for a DISTINCT sample from one text column.

    ``values`` is the raw sample (Nones preserved). Null-ish entries
    (``None`` / blank) are skipped; at least ``threshold`` (default 80%) of
    the remaining sample must parse as float. Records the percent and
    currency flags plus the observed ratio.
    """
    total = 0
    parsed = 0
    percent = False
    currency = False
    for v in values or []:
        cleaned, pct, cur = clean_text_numeric_value(v)
        if cleaned is None and (v is None or not str(v).strip()):
            continue
        total += 1
        if pct:
            percent = True
        if cur:
            currency = True
        if cleaned is not None:
            try:
                float(cleaned)
                parsed += 1
            except (ValueError, TypeError):
                pass
    ratio = (parsed / total) if total else 0.0
    return {
        "sample": total,
        "parsed": parsed,
        "ratio": ratio,
        "threshold": threshold,
        "coercible": bool(total and ratio >= threshold),
        "percent_unit": percent,
        "currency_unit": currency,
        "note": ("percent values stay as-is numerically (15% -> 15)")
        if percent else None,
    }


def is_text_numeric(values, threshold: float = COERCION_THRESHOLD) -> bool:
    """True when >= ``threshold`` of the non-null sample parses as float."""
    try:
        return bool(diagnose_text_numeric(values, threshold).get("coercible"))
    except Exception:
        return False


def is_measure_like_name(column_name) -> bool:
    """True when a column NAME looks like a measure (generic fragments only).

    Token equality first, then substring so ``discounted_price`` matches via
    ``discount``/``price`` and ``rating_count`` via ``rating``/``count``.
    """
    toks = [t for t in re.split(r"[^a-z0-9]+", str(column_name or "").lower()) if t]
    if not toks:
        return False
    tokset = set(toks)
    if tokset & _MEASURE_LIKE_FRAGMENTS:
        return True
    for tok in toks:
        for frag in _MEASURE_LIKE_FRAGMENTS:
            if frag in tok or tok in frag:
                return True
    return False


def set_regexp_support(dialect: str, supported: bool) -> None:
    """Override the REGEXP_REPLACE assumption for a dialect (tests/ops)."""
    _REGEXP_SUPPORT[str(dialect or "").lower()] = bool(supported)
    _REGEXP_PROBED.add(str(dialect or "").lower())


def ensure_dialect_support(engine, dialect: str) -> bool:
    """Probe once per process whether ``dialect`` supports REGEXP_REPLACE.

    Result is cached in-memory (never per query). Any failure falls back to
    the nested REPLACE chain. Returns the cached/observed flag.
    """
    d = str(dialect or "").lower()
    if d in _REGEXP_PROBED:
        return _REGEXP_SUPPORT.get(d, True)
    if d not in ("postgresql", "mysql"):
        _REGEXP_PROBED.add(d)
        return False
    try:
        from sqlalchemy import text as _text

        if d == "postgresql":
            probe = "SELECT regexp_replace('abc123', '[^0-9]', '', 'g')"
        else:
            probe = "SELECT REGEXP_REPLACE('abc123', '[^0-9]', '')"
        with engine.connect() as conn:
            conn.execute(_text(probe)).fetchone()
        _REGEXP_SUPPORT[d] = True
    except Exception:
        _REGEXP_SUPPORT[d] = False
    _REGEXP_PROBED.add(d)
    return _REGEXP_SUPPORT.get(d, True)


def _replace_chain(quoted_col: str) -> str:
    """Nested REPLACE chain stripping currency/commas/percent/spaces."""
    expr = quoted_col
    for ch in REPLACE_CHAIN_CHARS:
        expr = f"REPLACE({expr}, '{ch}', '')"
    return expr


def coercion_sql(dialect: str, quoted_col: str, use_regexp=None) -> str:
    """SAFE cast expression turning a text column reference numeric.

    ``quoted_col`` is the already-quoted column reference (bare or
    table-qualified). Unparsable values become ``NULL`` via the
    ``NULLIF(..., '')`` guard (the regexp variants additionally strip every
    non-numeric character, so residue cannot invent values). Percent values
    stay as-is numerically (``15%`` -> ``15``); the unit is noted in the
    diagnostics metadata, never divided here.
    """
    d = str(dialect or "").lower()
    col = quoted_col
    if use_regexp is None:
        use_regexp = _REGEXP_SUPPORT.get(d, True) if d in (
            "postgresql", "mysql") else False
    if d == "postgresql":
        if use_regexp:
            return (f"CAST(NULLIF(REGEXP_REPLACE({col}, '[^0-9.\\-]', '', 'g'), '')"
                    f" AS DOUBLE PRECISION)")
        return f"CAST(NULLIF({_replace_chain(col)}, '') AS DOUBLE PRECISION)"
    if d == "mysql":
        if use_regexp:
            return (f"CAST(NULLIF(REGEXP_REPLACE({col}, '[^0-9.\\-]', ''), '')"
                    f" AS DOUBLE)")
        return f"CAST(NULLIF({_replace_chain(col)}, '') AS DOUBLE)"
    # sqlite (and any other dialect): no REGEXP_REPLACE -- REPLACE chain.
    return f"CAST(NULLIF({_replace_chain(col)}, '') AS REAL)"


def cache_key(db_uri, table, column) -> tuple[str, str, str]:
    """In-memory probe-cache key per (db_uri, table, column)."""
    return (str(db_uri or ""), str(table or "").lower(),
            str(column or "").lower())


def get_coercion_cache(db_uri, table, column):
    """Return the cached probe record, or None when never probed."""
    key = cache_key(db_uri, table, column)
    with _COERCED_CACHE_LOCK:
        rec = _COERCED_CACHE.get(key)
        if rec is not None:
            # LRU touch: reinsert so eviction drops least-recently-used.
            _COERCED_CACHE.pop(key, None)
            _COERCED_CACHE[key] = rec
        return rec


def set_coercion_cache(db_uri, table, column, record: dict) -> dict:
    """Store a probe record (evicting the LRU entry at capacity) and return it."""
    key = cache_key(db_uri, table, column)
    with _COERCED_CACHE_LOCK:
        _COERCED_CACHE.pop(key, None)
        while len(_COERCED_CACHE) >= _COERCED_CACHE_MAX:
            _COERCED_CACHE.pop(next(iter(_COERCED_CACHE)))
        _COERCED_CACHE[key] = record
    return record


def clear_coercion_cache() -> None:
    """Drop all cached probe records (tests / ops reset)."""
    with _COERCED_CACHE_LOCK:
        _COERCED_CACHE.clear()
