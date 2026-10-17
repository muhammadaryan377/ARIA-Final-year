"""ARIA Intelligence Module — deterministic analytics service layer.

Implements the Explore / Hypotheses / Goals / What-If / Reports pipeline.
All calculations are deterministic — LLM may only explain/phrase, never
invent metrics, percentages, drivers, or recommendations.

Reuses: core.semantic, core.data_intelligence, hypothesis_engine,
prediction, prescription_engine, services.insights_service.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.config import SCHEMA_DIR, USERS_DIR
from core.semantic import (
    classify_field_role,
    display_name,
    humanize,
    is_dimension_like,
    is_measure_role,
    is_technical_role,
    MEASURE_FAMILY,
)
from services.intelligence_validation import to_date as _shared_to_date

log = logging.getLogger("aria.intelligence")

# Synthetic record-count metric used when a schema exposes zero numeric
# measures (generic COUNT(*) fallback — shape-based, never per-database).
COUNT_METRIC = "__count__"
COUNT_LABEL = "Record Count"
COUNT_FALLBACK_HINT = "no numeric measures found — showing record counts"

# ---------------------------------------------------------------------------
# Persistence paths (per-user isolation)
# ---------------------------------------------------------------------------

def _user_intel_dir(user_id: int) -> Path:
    d = USERS_DIR / f"user_{user_id}" / "intelligence"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _dataset_id(session: dict) -> str:
    """Stable fingerprint for the connected dataset."""
    db = session.get("db_name") or session.get("db") or session.get("db_uri") or ""
    src = session.get("source_type") or "unknown"
    return hashlib.sha256(f"{src}:{db}".encode()).hexdigest()[:16]


def _load_json(path: Path) -> dict | list:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, default=str, ensure_ascii=False), encoding="utf-8")


def _load_user_items(user_id: int, kind: str) -> list:
    d = _user_intel_dir(user_id)
    return _load_json(d / f"{kind}.json") if (d / f"{kind}.json").exists() else []


def _save_user_items(user_id: int, kind: str, items: list) -> None:
    d = _user_intel_dir(user_id)
    _save_json(d / f"{kind}.json", items)


# ---------------------------------------------------------------------------
# Schema / metadata helpers
# ---------------------------------------------------------------------------

NO_SCHEMA_REASON = "NO_SCHEMA"
NO_SCHEMA_HINT = "Connect a data source or run Extract Schema (POST /api/extract_schema)"


def _newest_fallback_mapping(search_dirs: list[Path]) -> Path | None:
    """Return newest schema_mapping_*_latest.json under search_dirs, or None."""
    best: Path | None = None
    best_mtime = -1.0
    for d in search_dirs:
        try:
            if not d or not d.is_dir():
                continue
            for cand in d.glob("schema_mapping_*_latest.json"):
                try:
                    if not cand.is_file():
                        continue
                    mt = cand.stat().st_mtime
                except Exception:
                    continue
                if mt > best_mtime:
                    best_mtime = mt
                    best = cand
        except Exception:
            continue
    return best


def _load_schema_mapping(session: dict) -> dict:
    """Load the current schema mapping JSON from session path.

    Fallback: if session["schema_path"] is missing/empty/unparseable, scan
    its dirname (or data/schemas/user_{id}/) for the newest
    schema_mapping_*_latest.json by mtime. Never raises; logs chosen file.
    """
    session = session or {}
    sp = session.get("schema_path")
    if sp:
        try:
            p = Path(str(sp))
            if p.exists() and p.is_file():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if data:
                        log.info("intelligence schema: using session schema_path=%s", p)
                        return data
                    log.warning("intelligence schema: session schema_path=%s parsed empty; trying fallback", p)
                except Exception as exc:
                    log.warning("intelligence schema: session schema_path=%s unparseable (%s); trying fallback", p, exc)
        except Exception as exc:
            log.warning("intelligence schema: bad schema_path=%r (%s); trying fallback", sp, exc)
    search_dirs: list[Path] = []
    try:
        if sp:
            parent = Path(str(sp)).parent
            search_dirs.append(parent)
    except Exception:
        pass
    for uid_key in ("user_id", "userId", "uid"):
        try:
            uid = session.get(uid_key)
            if uid is not None and str(uid).strip() not in ("", "None"):
                search_dirs.append(SCHEMA_DIR / f"user_{uid}")
        except Exception:
            continue
    # Infer user dir from schema_path string (e.g. ".../user_328/...").
    try:
        m = re.search(r"user_(\d+)", str(sp or ""))
        if m:
            search_dirs.append(SCHEMA_DIR / f"user_{m.group(1)}")
    except Exception:
        pass
    fallback = _newest_fallback_mapping(search_dirs)
    if fallback is not None:
        try:
            data = json.loads(fallback.read_text(encoding="utf-8"))
            if data:
                log.info("intelligence schema: using fallback mapping=%s", fallback)
                return data
            log.warning("intelligence schema: fallback mapping=%s parsed empty", fallback)
        except Exception as exc:
            log.warning("intelligence schema: fallback mapping=%s unparseable (%s)", fallback, exc)
    return {}


def _get_table_profiles(mapping: dict) -> dict:
    """Return {table_name: profile} from data_intelligence."""
    from core.data_intelligence import build_data_intelligence
    intel = build_data_intelligence(mapping)
    return intel.get("per_table") or {}


def _classify_columns(mapping: dict) -> dict:
    """Classify every column into semantic field roles.

    Returns {table: {col: role}}. Uses core.semantic.classify_field_role.
    """
    tables = mapping.get("tables") or {}
    result: Dict[str, Dict[str, str]] = {}
    for tname, tinfo in tables.items():
        cols = tinfo.get("columns", {})
        col_map: Dict[str, str] = {}
        pk = set(tinfo.get("primary_key", []) or tinfo.get("inferred_primary_key", []) or [])
        fk_cols = set()
        for fk in tinfo.get("foreign_keys", []) or []:
            if isinstance(fk, dict) and fk.get("column"):
                fk_cols.add(str(fk["column"]))
        if isinstance(cols, dict):
            for cname, cmeta in cols.items():
                cmeta = cmeta or {}
                col_map[str(cname)] = classify_field_role(
                    cname,
                    cmeta.get("data_type", ""),
                    is_pk=str(cname) in pk,
                    is_fk=str(cname) in fk_cols,
                )
        else:
            for col in cols or []:
                if isinstance(col, dict):
                    cn = str(col.get("column", ""))
                    col_map[cn] = classify_field_role(cn, col.get("data_type", ""),
                                                      is_pk=cn in pk, is_fk=cn in fk_cols)
        result[str(tname)] = col_map
    return result


# ---------------------------------------------------------------------------
# GET /api/intelligence/meta
# ---------------------------------------------------------------------------

def get_intelligence_meta(session: dict) -> dict:
    """Return available metrics, dimensions, date columns for the Explore tab.

    Filters: BUSINESS_MEASURE/MEASURE only for metrics; DIMENSION/CATEGORY
    only for breakdowns; DATE for time axis. Never includes FK, IDENTIFIER,
    or TECHNICAL_MEASURE in analysis options.
    """
    mapping = _load_schema_mapping(session)
    if not mapping:
        return {"ok": True, "metrics": [], "dimensions": [], "date_columns": [], "tables": [],
                "reason": NO_SCHEMA_REASON, "hint": NO_SCHEMA_HINT}

    roles = _classify_columns(mapping)
    metrics, dimensions, date_columns, tables = [], [], [], []
    _db_uri = session.get("db_uri")

    for tname, col_roles in roles.items():
        tables.append(tname)
        for cname, role in col_roles.items():
            label = display_name(cname, cname)
            entry = {"column": cname, "table": tname, "label": label, "role": role}
            if role in ("BUSINESS_MEASURE", "MEASURE", "DERIVED_METRIC"):
                metrics.append(entry)
            elif role in ("DIMENSION", "CATEGORY", "TEXT"):
                dimensions.append(entry)
            elif role == "DATE":
                date_columns.append(entry)
        # Role-poor tables (no text dimensions / no DATE role, e.g. a
        # TEXT date held in a composite key or an all-numeric table):
        # value-verified extras so Explore offers a time axis and real
        # breakdowns instead of an empty page. Bounded probes, skipped
        # entirely when roles already cover the table.
        try:
            _t_dims = [d for d in dimensions if d.get("table") == tname]
            _t_dates = [d for d in date_columns if d.get("table") == tname]
            if (not _t_dims or not _t_dates) and _db_uri:
                _extra = _probe_table_extras(
                    mapping, tname, col_roles or {}, _db_uri)
                if not _t_dims:
                    for c in (_extra.get("low_card_dims") or []):
                        if c in (col_roles or {}):
                            _e = {"column": c, "table": tname,
                                  "label": display_name(c, c),
                                  "role": "DIMENSION",
                                  "detected": "low_cardinality_values"}
                            dimensions.append(_e)
                            _t_dims.append(_e)
                if not _t_dates:
                    for c in (_extra.get("date_cols") or []):
                        if c in (col_roles or {}):
                            _e = {"column": c, "table": tname,
                                  "label": display_name(c, c),
                                  "role": "DATE",
                                  "detected": "date_values"}
                            date_columns.append(_e)
                            _t_dates.append(_e)
        except Exception:
            pass

    breakdowns = [{"column": d["column"], "table": d["table"], "label": d["label"],
                     "name": d["label"]} for d in dimensions]
    if not metrics and tables:
        # Generic COUNT(*) fallback (shape-based): schemas with zero numeric
        # measures (e.g. garbage-concatenated single-column tables) still get
        # a working Explore — record counts over any breakdown or overall.
        metrics.append({"column": COUNT_METRIC, "table": tables[0],
                        "label": COUNT_LABEL, "role": "COUNT"})
    time_grains = [{"name": "Monthly", "label": "Monthly"}, {"name": "Weekly", "label": "Weekly"},
                   {"name": "Quarterly", "label": "Quarterly"}, {"name": "Yearly", "label": "Yearly"}]
    comparisons = [{"name": "Previous Period", "label": "Previous Period"},
                   {"name": "Same Period Last Year", "label": "Same Period Last Year"},
                   {"name": "No Comparison", "label": "No Comparison"}]
    date_ranges = [{"name": "Last 12 months", "label": "Last 12 months"},
                   {"name": "Last 6 months", "label": "Last 6 months"},
                   {"name": "Last 3 months", "label": "Last 3 months"},
                   {"name": "All data", "label": "All data"}]
    filter_cols = [{"column": d["column"], "table": d["table"], "label": d["label"],
                    "name": d["label"]} for d in dimensions]

    out = {
        "ok": True,
        "metrics": metrics,
        "measures": metrics,
        "dimensions": dimensions,
        "breakdowns": breakdowns,
        "date_columns": date_columns,
        "tables": tables,
        "time_grains": [t["name"] for t in time_grains],
        "timeGrains": [t["name"] for t in time_grains],
        "comparisons": [c["name"] for c in comparisons],
        "date_ranges": [d["name"] for d in date_ranges],
        "dateRanges": [d["name"] for d in date_ranges],
        "filterable_columns": filter_cols,
        "filterColumns": filter_cols,
        "change_types": ["Increase", "Decrease"],
        "changeTypes": ["Increase", "Decrease"],
        "apply_targets": ["All data", "Top segment", "Filtered set"],
        "applyTargets": ["All data", "Top segment", "Filtered set"],
    }
    if metrics and metrics[0].get("column") == COUNT_METRIC:
        out["hint"] = COUNT_FALLBACK_HINT
        out["count_fallback"] = True
    return out


# ---------------------------------------------------------------------------
# POST /api/intelligence/explore
# ---------------------------------------------------------------------------

def _human_total(v: float) -> str:
    """Human-scale total: 1.2M, 34.5K — never raw long decimals."""
    try:
        a = abs(float(v))
    except Exception:
        return "a modest total"
    if a >= 1_000_000_000:
        return f"about {float(v) / 1_000_000_000:.1f} billion"
    if a >= 1_000_000:
        return f"about {float(v) / 1_000_000:.1f} million"
    if a >= 10_000:
        return f"about {float(v) / 1_000:.1f} thousand"
    if a >= 100:
        return f"around {int(round(float(v))):,}"
    if a >= 1:
        return f"around {round(float(v), 1)}"
    return f"around {round(float(v), 2)}"


_EVENT_DATE_HINTS = ("order_date", "orderdate", "invoice_date", "sale_date",
                      "sales_date", "transaction_date", "created", "created_at",
                      "purchase_date", "order_time")
_BIRTH_HIRE_HINTS = ("birth", "dob", "birthday", "hire", "death", "anniversary")


def _is_birth_hire_col(name: str) -> bool:
    s = str(name or "").lower()
    return any(h in s for h in _BIRTH_HIRE_HINTS)


def _best_date_col(mapping: dict, roles: dict, primary_table: str | None,
                   exclude_birth_hire: bool = True):
    """Pick the best event DATE column: (column, home_table) or (None, None).

    Prefers transaction-like names (order_date, created, ...) over generic
    ones, never birth/hire dates (unless nothing else exists), and prefers
    the primary table, then 1-hop, then 2-hop neighbours.
    """
    cands = []  # (col, home)
    for _t, _cr in (roles or {}).items():
        for _c, _r in ((_cr or {}).items()):
            if _r == "DATE":
                cands.append((_c, _t))
    if not cands:
        return None, None
    if exclude_birth_hire:
        _nonbh = [(c, t) for c, t in cands if not _is_birth_hire_col(c)]
        if _nonbh:
            cands = _nonbh

    def _hops(home: str) -> int:
        if not primary_table or home == primary_table:
            return 0
        try:
            _p = _find_join_path(mapping, roles, primary_table, home)
        except Exception:
            return 99
        if _p is None:
            return 99
        return len(_p)

    def _name_rank(col: str) -> int:
        s = str(col or "").lower()
        for _i, _h in enumerate(_EVENT_DATE_HINTS):
            if _h in s:
                return _i
        if s in ("date", "day", "month", "time", "timestamp", "datetime"):
            return len(_EVENT_DATE_HINTS)
        return len(_EVENT_DATE_HINTS) + 1

    cands.sort(key=lambda ct: (_hops(ct[1]), _name_rank(ct[0]), str(ct[0])))
    return cands[0]


def _resolve_column(roles: dict, name: str | None) -> str | None:
    """Resolve a UI label or column name to the real column (case-insensitive)."""
    if not name:
        return None
    s = str(name).strip()
    if not s:
        return None
    # exact column match first
    for tname, col_roles in (roles or {}).items():
        if s in (col_roles or {}):
            return s
    low = s.lower()
    for tname, col_roles in (roles or {}).items():
        for cname in (col_roles or {}):
            if str(cname).lower() == low:
                return cname
            try:
                if display_name(cname, cname).lower() == low:
                    return cname
            except Exception:
                pass
            if humanize(cname).lower() == low:
                return cname
    return s


def _probe_text_cols(proles: dict, mapping: dict, table: str) -> list:
    """TEXT-dtype, non-measure columns eligible for the stored-date probe
    (date-like names first, then the rest alphabetically). Shape-based:
    values decide, not names — covers oddly-named TEXT date storage."""
    try:
        from core.semantic import is_measure_role as _ism
    except Exception:
        _ism = lambda _r: False
    out = []
    for c, r in (proles or {}).items():
        try:
            if _ism(r):
                continue
        except Exception:
            pass
        if not _is_text_dtype(_col_data_type(mapping, table, c)):
            continue
        s = str(c).lower()
        if _is_birth_hire_col(s):
            continue
        _named = bool(re.search(r"date|time|day|month|year|\bdt\b", s))
        out.append((0 if _named else 1, str(c), c))
    out.sort()
    return [c for _, _, c in out]


def _home_table(roles: dict, column: str | None) -> str | None:
    """Home table owning ``column`` (exact match, first in mapping order)."""
    if not column:
        return None
    for tname, col_roles in (roles or {}).items():
        if str(column) in (col_roles or {}):
            return tname
    return None


def _dialect_of(db_uri: str | None) -> str:
    u = str(db_uri or "")
    if "mysql" in u:
        return "mysql"
    if "postgresql" in u or "psycopg" in u:
        return "postgresql"
    return "sqlite"


def _qident(dialect: str, name: str) -> str:
    if dialect == "mysql":
        return "`" + str(name).replace("`", "``") + "`"
    return '"' + str(name).replace('"', '""') + '"'


def _probe_table_extras(mapping: dict, table: str, col_roles: dict,
                        db_uri: str | None) -> dict:
    """Value-verified date/dimension columns for role-poor tables.

    Some real schemas expose no DATE role (e.g. a DD-MM-YYYY TEXT date
    classified IDENTIFIER as part of a composite key) and no text
    dimensions (all-numeric tables). Meta/hypotheses built from roles alone
    then offer no time axis, no breakdowns and no hypotheses — the
    Intelligence page looks broken although Explore's own value probe
    handles the same columns fine.

    Returns ``{"date_cols": [...], "low_card_dims": [...]}`` (mapping
    order, fully schema-agnostic — values decide, never names):
    - ``date_cols``: TEXT non-measure columns whose DISTINCT values parse
      as dates (same >=3 parsed / >=50% / years 1970..2100 rule as
      run_explore's stored-date probe).
    - ``low_card_dims``: IDENTIFIER columns with 2..50 distinct values
      (genuine segments like a store number; high-cardinality keys such
      as product ids are excluded by the count).
    Bounded (one DISTINCT-LIMIT-200 per TEXT candidate, one COUNT
    DISTINCT per IDENTIFIER candidate) and only meaningful when the
    caller lacks roles — callers skip it otherwise. Never raises: []
    on any failure (including no db_uri).
    """
    out = {"date_cols": [], "low_card_dims": []}
    try:
        if not db_uri or not table or not col_roles:
            return out
        dialect = _dialect_of(db_uri)
        engine = _pooled_engine(str(db_uri))
        schema = _table_schema(mapping, table, dialect=dialect, engine=engine)
        qt = _qtbl(table, schema, dialect)
        from sqlalchemy import text as _stext
        with engine.connect() as conn:
            for c in _probe_text_cols(col_roles, mapping, table):
                try:
                    rows = conn.execute(
                        _stext(f"SELECT DISTINCT {_qident(dialect, c)} AS _d "
                               f"FROM {qt} LIMIT 200")).fetchall()
                    vals = [r[0] for r in rows]
                    if not vals:
                        continue
                    parsed = [_shared_to_date(v) for v in vals]
                    parsed = [d for d in parsed if d is not None]
                    if (len(parsed) >= 3
                            and len(parsed) >= 0.5 * len(vals)
                            and min(d.year for d in parsed) >= 1970
                            and max(d.year for d in parsed) <= 2100):
                        out["date_cols"].append(c)
                except Exception:
                    continue
            for c, r in (col_roles or {}).items():
                if r != "IDENTIFIER":
                    continue
                if len(out["low_card_dims"]) >= 8:
                    break
                try:
                    n = conn.execute(
                        _stext(f"SELECT COUNT(DISTINCT {_qident(dialect, c)}) "
                               f"FROM {qt}")).fetchone()
                    n = int(n[0] or 0) if n else 0
                    if 2 <= n <= 50:
                        out["low_card_dims"].append(c)
                except Exception:
                    continue
    except Exception:
        pass
    return out


def _col_data_type(mapping: dict, table: str | None, column: str | None) -> str:
    """Storage data-type of ``column`` in ``table`` per the schema mapping.

    Shape-based (never per-database): drives the numeric-safe metric
    expression (TEXT prices like ``₹1,099`` need a cast, not a crash) and
    storage-type-aware date bucketing (native DATE -> SQL buckets, anything
    else -> Python bucketing with the shared tolerant parser).
    """
    try:
        if not table or not column:
            return ""
        tinfo = ((mapping or {}).get("tables") or {}).get(str(table)) or {}
        cols = tinfo.get("columns", {})
        if isinstance(cols, dict):
            return str(((cols.get(str(column))) or {}).get("data_type") or "")
        for c in cols or []:
            if isinstance(c, dict) and str(c.get("column")) == str(column):
                return str(c.get("data_type") or "")
    except Exception:
        pass
    return ""


def _is_text_dtype(dtype: str) -> bool:
    dt = str(dtype or "").lower()
    return any(k in dt for k in ("char", "text", "clob", "string"))


def _is_native_date_dtype(dtype: str) -> bool:
    dt = str(dtype or "").lower()
    return any(k in dt for k in ("date", "time", "stamp"))


def _numeric_metric_sql(ref: str, dtype: str, dialect: str) -> str:
    """Numeric-safe SQL expression for a SUM over ``ref``.

    TEXT-stored measures (currency symbols, commas, ``%`` — e.g. amazon
    ``discounted_price`` = ``₹1,099``) become a dialect-appropriate numeric
    cast yielding NULL for unparseable values, instead of raising
    ``function sum(text) does not exist`` (a *type* error, never a join
    error). Cleaning mirrors ``_sqlite_numstrip`` in every dialect: a stray
    leading dot from prefixes like ``Rs.`` drops only before 3+ digits, and
    non-numeric shapes become NULL rather than raising.
    """
    if not _is_text_dtype(dtype):
        return ref
    if dialect == "postgresql":
        _clean = (f"regexp_replace(regexp_replace({ref}, '[^0-9.\\-]', '', 'g'), "
                  f"'^\\.([0-9]{{3,}})$', '\\1')")
        return (f"CAST(NULLIF(CASE WHEN {_clean} ~ '^-?[0-9]*\\.?[0-9]+$' "
                f"THEN {_clean} END, '') AS DOUBLE PRECISION)")
    if dialect == "mysql":
        _clean = (f"REGEXP_REPLACE(REGEXP_REPLACE({ref}, '[^0-9.\\-]', ''), "
                  f"'^\\\\.([0-9]{{3,}})$', '$1')")
        return (f"CAST(NULLIF(CASE WHEN {_clean} REGEXP '^-?[0-9]*\\.?[0-9]+$' "
                f"THEN {_clean} END, '') AS DOUBLE)")
    # sqlite has no regexp_replace — aria_numstrip (registered on the engine
    # via validation.ensure_sqlite_helpers) strips to the numeric core.
    return f"CAST(NULLIF(aria_numstrip({ref}), '') AS REAL)"


def _nan_safe_sql(ref: str, dtype: str, dialect: str) -> str:
    """Map float NaN values to NULL so one NaN never poisons a SUM.

    Postgres propagates NaN through SUM (blinkit ``item_weight``); filtering
    ``= 'NaN'`` (postgres NaN equality) keeps the aggregate honest. Other
    dialects / non-float storage pass through (Python-side ``_fnum``
    sanitizes whatever remains).
    """
    dt = str(dtype or "").lower()
    if dialect == "postgresql" and any(
            k in dt for k in ("float", "double", "real", "numeric", "decimal")):
        return f"(CASE WHEN {ref} = 'NaN' THEN NULL ELSE {ref} END)"
    return ref


class _NoJoinPath(Exception):
    """Honest cross-table failure — surfaced as NO_JOIN_PATH, never masked."""

    def __init__(self, payload: dict):
        super().__init__(payload.get("detail", ""))
        self.payload = payload


def _is_undefined_column(exc: Exception) -> bool:
    """True for missing-column DB errors across drivers.

    Covers postgres UndefinedColumn, sqlite ``no such column``, mysql
    ``unknown column``, generic ``invalid column`` spellings, MSSQL
    ``invalid column name`` and Oracle ``invalid identifier`` (ORA-00904).
    Anchored on the *column* subject so sibling errors with similar text —
    e.g. postgres ``function sum(text) does not exist`` (UndefinedFunction,
    a *type* error needing a numeric cast, not a join error) or ``relation
    ... does not exist`` — never map to NO_JOIN_PATH.
    """
    s = f"{type(exc).__name__}: {exc}".lower()
    if ("undefinedfunction" in s or "undefined function" in s
            or "relation" in s and "does not exist" in s and "column" not in s):
        return False
    return ("undefinedcolumn" in s or "undefined column" in s
            or "no such column" in s or "unknown column" in s
            or "invalid column" in s or "invalid identifier" in s
            or ("column" in s and "does not exist" in s))


def _no_join_path_result(roles: dict, primary_table: str,
                         target_table: str, column: str) -> dict:
    """Honest NO_JOIN_PATH payload with same-table dimension alternatives."""
    alternatives = [c for c, r in (roles.get(primary_table) or {}).items()
                    if r in ("DIMENSION", "CATEGORY", "TEXT")][:3]
    if alternatives:
        sugg = ", ".join(display_name(c, c) for c in alternatives)
        tail = (f" Try breaking down by {sugg} instead — "
                f"those live in '{primary_table}'.")
    else:
        tail = f" Try a dimension from '{primary_table}'."
    return {
        "ok": False,
        "code": "NO_JOIN_PATH",
        "detail": (f"'{column}' lives in '{target_table}' which has no "
                   f"relationship to '{primary_table}'.{tail}"),
    }


def _table_schema(mapping: dict, table: str, conn=None, dialect: str = "sqlite",
                  cache: dict | None = None, engine=None) -> str | None:
    """Resolve the DB schema for ``table``.

    (a) per-table ``tables[table]["schema"]`` if present; (b) top-level
    ``mapping["schema"]`` or ``mapping["db_schema"]``; (c) else probe
    ``information_schema.tables`` for the schema (results cached in
    ``cache`` dict for the call). Never raises — returns None when unknown
    (sqlite / public handled by _qtbl as unqualified).
    """
    mapping = mapping or {}
    try:
        tinfo = (mapping.get("tables") or {}).get(str(table)) or {}
        for _k in ("schema", "table_schema"):
            s = tinfo.get(_k)
            if s is not None and str(s).strip() and str(s).strip() != "*":
                return str(s).strip()
        # Qualified key without an entry field: the leading segment is the
        # schema (sales.employee -> sales). Generic, no names.
        _segs = [p for p in str(table or "").split(".") if p]
        if len(_segs) > 1 and str(_segs[0]).strip() != "*":
            return str(_segs[0]).strip()
    except Exception:
        pass
    try:
        for _k in ("schema", "db_schema"):
            s = mapping.get(_k)
            if s is not None and str(s).strip() and str(s).strip() != "*":
                return str(s).strip()
    except Exception:
        pass
    if dialect == "sqlite":
        return None
    if cache is not None and str(table) in cache:
        return cache.get(str(table))
    schema = None
    src = conn if conn is not None else engine
    if src is not None:
        try:
            from sqlalchemy import text as _stext
            _q = _stext(
                "SELECT table_schema FROM information_schema.tables "
                "WHERE LOWER(table_name) = LOWER(:t) "
                "AND table_schema NOT IN ('pg_catalog', 'information_schema') "
                "LIMIT 5")
            if hasattr(src, "execute"):
                _rows = src.execute(_q, {"t": str(table)}).fetchall()
            elif hasattr(src, "connect"):
                with src.connect() as _c:
                    _rows = _c.execute(_q, {"t": str(table)}).fetchall()
            else:
                _rows = []
            if _rows:
                schema = str(_rows[0][0]).strip() or None
        except Exception:
            schema = None
    if cache is not None:
        cache[str(table)] = schema
    return schema


def _qtbl(table: str, schema: str | None = None, dialect: str = "sqlite") -> str:
    """Schema-qualified quoted table ref: ``"schema"."table"`` (postgres),
    ```schema`.`table``` (mysql), plain quoted ``"table"`` for sqlite or
    when schema is public/none. Dotted table keys (sales.employee from
    multi-schema mappings) split into segments — never one dotted name.
    """
    try:
        _segs = [s for s in str(table or "").split(".") if s]
        if len(_segs) > 1 and not (schema and str(schema).strip()):
            table, schema = _segs[-1], _segs[0]
        elif len(_segs) > 1:
            table = _segs[-1]
    except Exception:
        pass
    if dialect == "mysql":
        def _q(n) -> str:
            return "`" + str(n).replace("`", "``") + "`"
        if schema and str(schema).strip() and str(schema).strip().lower() != "public":
            return f"{_q(str(schema).strip())}.{_q(table)}"
        return _q(table)

    def _q(n) -> str:
        return '"' + str(n).replace('"', '""') + '"'
    if dialect == "sqlite":
        return _q(table)
    if schema and str(schema).strip() and str(schema).strip().lower() != "public":
        return f"{_q(str(schema).strip())}.{_q(table)}"
    return _q(table)


def _iter_rel_edges(mapping: dict) -> list:
    """Normalized relationship edges as (left_table, left_col, right_table,
    right_col), declared edges first, deduplicated.

    Merges declared_relationships (table_name/column_name shape),
    per-table foreign_keys (both the flat and the nested-references shapes),
    inferred_relationships, and relationship_edges (left_/source_ spellings).
    Mirrors the goal_agent join-planning contract where each edge means
    ``left_table.left_col = right_table.right_col``.
    """
    mapping = mapping or {}
    seen: set = set()
    declared: list = []
    inferred: list = []

    def _add(a, ac, b, bc, is_declared: bool):
        if not a or not ac or not b or not bc:
            return
        key = (str(a), str(ac), str(b), str(bc))
        rkey = (str(b), str(bc), str(a), str(ac))
        if key in seen or rkey in seen:
            return
        seen.add(key)
        (declared if is_declared else inferred).append(key)

    for r in (mapping.get("declared_relationships") or []):
        if isinstance(r, dict):
            _add(r.get("table_name", r.get("table")),
                 r.get("column_name", r.get("column")),
                 r.get("references_table"), r.get("references_column"), True)
    for tname, tinfo in ((mapping.get("tables") or {}).items()):
        for fk in ((tinfo or {}).get("foreign_keys") or []):
            if not isinstance(fk, dict):
                continue
            ref = fk.get("references") or {}
            _add(tname, fk.get("column"),
                 fk.get("referenced_table", ref.get("table")),
                 fk.get("referenced_column", ref.get("column")), True)
    for r in (mapping.get("inferred_relationships") or []):
        if isinstance(r, dict):
            _add(r.get("table", r.get("table_name")),
                 r.get("column", r.get("column_name")),
                 r.get("references_table"), r.get("references_column"), False)
    for e in (mapping.get("relationship_edges") or []):
        if not isinstance(e, dict):
            continue
        a = e.get("left_table", e.get("source_table"))
        ac = e.get("left_column", e.get("source_column"))
        b = e.get("right_table", e.get("target_table"))
        bc = e.get("right_column", e.get("target_column"))
        typ = str(e.get("type", e.get("relationship_type", ""))).lower()
        _add(a, ac, b, bc, typ != "inferred")
    return declared + inferred


def _find_join_path(mapping: dict, roles: dict, from_table: str,
                    need_columns) -> list | None:
    """Join path from ``from_table`` to the home table(s) of ``need_columns``.

    ``need_columns`` may be a column name, a table name, or a list of those.
    Uses declared relationships first, then inferred ones (each edge:
    ``left_table.left_col = right_table.right_col``), mirroring the
    goal_agent ``_plan_joins``/``_bfs_join_path`` pattern. Supports paths of
    up to 4 hops with a per-path cycle guard (declared edges preferred by
    adjacency order).

    Returns a list of ``(table, on_left, on_right)`` hops in join order, where
    ``table`` is the newly joined table and ``on_left``/``on_right`` are
    ``(table, column)`` endpoint tuples for the ``ON`` clause; ``[]`` when
    everything already lives in ``from_table``; ``None`` when any target is
    unreachable within 4 hops.
    """
    mapping = mapping or {}
    items = [need_columns] if isinstance(need_columns, str) else list(need_columns or [])
    tables = mapping.get("tables") or {}
    targets: list[str] = []
    for it in items:
        if it is None:
            continue
        s = str(it)
        t = s if s in tables else _home_table(roles, s)
        if t and t != from_table and t not in targets:
            targets.append(t)
    if not targets:
        return []
    edges = _iter_rel_edges(mapping)
    adj: Dict[str, list] = {}
    emaps: dict = {}
    for a, ac, b, bc in edges:
        adj.setdefault(a, []).append((b, ac, bc))
        adj.setdefault(b, []).append((a, bc, ac))
        emaps.setdefault((a, b), (ac, bc))
        emaps.setdefault((b, a), (bc, ac))

    def _bfs(sources: set, target: str):
        queue = [[s] for s in sources]
        seen = set(sources)
        while queue:
            path = queue.pop(0)
            if len(path) - 1 >= 4:
                continue
            tail = path[-1]
            for nbr, _ucol, _vcol in adj.get(tail, []):
                if nbr in path:  # per-path cycle guard
                    continue
                if nbr in seen:
                    continue
                new_path = path + [nbr]
                if nbr == target:
                    hops = []
                    for x, y in zip(new_path, new_path[1:]):
                        xc, yc = emaps[(x, y)]
                        hops.append((y, (x, xc), (y, yc)))
                    return hops
                seen.add(nbr)
                queue.append(new_path)
        return None

    connected = {from_table}
    known = {from_table}
    hops_all: list = []
    for t in targets:
        if t in connected:
            continue
        found = _bfs(connected, t)
        if found is None:
            return None
        for h in found:
            if h[0] not in known:
                known.add(h[0])
                hops_all.append(h)
            connected.add(h[0])
            connected.add(h[1][0])
            connected.add(h[2][0])
    return hops_all


# Maximum breakdown groups pulled into Python per explore query. Driven by
# COUNT DISTINCT, never table/column names: high-cardinality dimensions
# stream only their top-N groups; the scope block tells the UI/validator.
_EXPLORE_TOP_CATS = 50

# Pooled engines per db_uri (bounded, thread-safe): creating a pool per
# request wastes handshakes (esp. postgres). sqlite file DBs share one
# pooled file connection set; credentials stay in the key only in memory.
import threading

_ENGINE_POOL: dict = {}
_ENGINE_LOCK = threading.Lock()


def _pooled_engine(db_uri: str):
    """Return a cached SQLAlchemy engine for db_uri (pool_size=5)."""
    key = str(db_uri or "")
    with _ENGINE_LOCK:
        eng = _ENGINE_POOL.get(key)
        if eng is None:
            from sqlalchemy import create_engine as _ce
            eng = _ce(key or "sqlite://", pool_pre_ping=True,
                      pool_size=5, max_overflow=10, pool_recycle=1800)
            if len(_ENGINE_POOL) >= 32:
                try:
                    _old_k, _old_e = next(iter(_ENGINE_POOL.items()))
                    _ENGINE_POOL.pop(_old_k, None)
                    _old_e.dispose()
                except Exception:
                    pass
            _ENGINE_POOL[key] = eng
        return eng


def run_explore(session: dict, request: dict) -> dict:
    """Server-side aggregation for the Explore tab.

    Returns chart contract + breakdown table + observations.
    All numbers from DB, never fabricated. All user-facing sentences are
    human-worded (human scale, no raw long decimals, no SQL).
    Accepts both legacy keys (time/compare) and frontend keys
    (time_grain/comparison/secondary_breakdown/date_range string).
    """
    roles_pre = _classify_columns(_load_schema_mapping(session)) if request.get("metric") else {}
    metric_col = _resolve_column(roles_pre, request.get("metric")) or request.get("metric")
    breakdown_col = _resolve_column(roles_pre, request.get("breakdown")) or request.get("breakdown")
    # Frontend aliases
    time_col_raw = request.get("time") or request.get("time_grain") or request.get("timeGrain")
    time_col = _resolve_column(roles_pre, time_col_raw) or time_col_raw
    compare = request.get("compare") or request.get("comparison") or "previous_period"
    date_range = request.get("date_range") or request.get("dateRange") or {}
    filters = request.get("filters") or []
    # secondary breakdown accepted but folded into note (single-level agg)
    secondary_raw = request.get("secondary_breakdown") or request.get("secondaryBreakdown")
    secondary = _resolve_column(roles_pre, secondary_raw) or secondary_raw
    # normalize date_range string -> dict (no filtering, honest label)
    date_label = "All data"
    if isinstance(date_range, str):
        date_label = date_range
        date_range = {}
    elif isinstance(date_range, dict) and date_range:
        date_label = date_range.get("label") or date_range.get("name") or "Selected range"

    if not metric_col:
        return {"ok": False, "detail": "Metric is required.", "code": "INVALID_REQUEST"}

    mapping = _load_schema_mapping(session)
    if not mapping:
        return {"ok": False, "detail": "No schema mapping found.", "code": "NO_DATA"}

    roles = _classify_columns(mapping)
    # Grain labels (Monthly/Weekly/...) are not columns: map to the best
    # event DATE column (transaction dates like order_date, never
    # birth/hire dates).
    try:
        _all_cols = {c for cr in roles.values() for c in (cr or {})}
        if time_col and time_col not in _all_cols:
            _date_col, _date_home0 = _best_date_col(
                mapping, roles, None, exclude_birth_hire=True)
            # keep grain choice for display but use real date column for SQL
            request["_grain_label"] = time_col
            time_col = _date_col
    except Exception:
        pass
    tables = mapping.get("tables") or {}
    primary_table = None
    metric_role = None
    _is_count = (metric_col == COUNT_METRIC)
    if _is_count:
        # Generic COUNT(*) fallback: any table works; prefer the breakdown's
        # home table so counts group naturally, else the first table.
        _bhome0 = _home_table(roles, breakdown_col) if breakdown_col else None
        primary_table = _bhome0 or (next(iter(tables)) if tables else None)
        metric_role = "COUNT"
        if not primary_table:
            return {"ok": False, "detail": "No tables found for record counts.", "code": "NO_DATA"}
    else:
        for tname, col_roles in roles.items():
            if metric_col in col_roles:
                primary_table = tname
                metric_role = col_roles[metric_col]
                break
    if not primary_table:
        return {"ok": False, "detail": f"Column '{metric_col}' not found.", "code": "COLUMN_NOT_FOUND"}

    # Validate: metric must be measure-family (COUNT fallback exempt)
    if metric_role and not _is_count and not is_measure_role(metric_role):
        return {"ok": False, "detail": f"'{metric_col}' is not a metric (role={metric_role}).", "code": "INVALID_METRIC"}

    # Validate: breakdown must be dimension-like (resolved in its HOME table —
    # the breakdown may live in another table reached via a declared/inferred
    # relationship; order matters: check technical/FK/measure in sequence).
    b_home = _home_table(roles, breakdown_col) if breakdown_col else None
    if breakdown_col:
        if not b_home:
            return {"ok": False, "detail": f"Column '{breakdown_col}' not found.", "code": "COLUMN_NOT_FOUND"}
        b_role = (roles.get(b_home) or {}).get(breakdown_col) if b_home else None
        if b_role and b_role == "TECHNICAL_MEASURE":
            return {"ok": False, "detail": f"Breakdown '{breakdown_col}' is a technical field.", "code": "TECHNICAL_ONLY"}
        if b_role and b_role == "FOREIGN_KEY":
            return {"ok": False, "detail": f"Breakdown '{breakdown_col}' is a foreign key — cannot use as business dimension.", "code": "FK_NOT_DRIVER"}
        if b_role and is_measure_role(b_role):
            return {"ok": False, "detail": f"Breakdown '{breakdown_col}' is a measure — cannot use as dimension.", "code": "MEASURE_USED_AS_DIMENSION"}
        if b_home and b_home != primary_table:
            # Cross-table breakdown: fail honest (NO_JOIN_PATH) when no
            # relationship connects the two tables — never SQL-crash.
            if _find_join_path(mapping, roles, primary_table, b_home) is None:
                return _no_join_path_result(roles, primary_table, b_home, breakdown_col)

    # Build SQL
    db_uri = session.get("db_uri")
    dialect = "sqlite"
    if db_uri:
        if "mysql" in db_uri:
            dialect = "mysql"
        elif "postgresql" in db_uri or "psycopg" in db_uri:
            dialect = "postgresql"

    try:
        from sqlalchemy import create_engine, text as sql_text
        engine = _pooled_engine(db_uri)
        if dialect == "sqlite":
            try:
                from services.intelligence_validation import (
                    ensure_sqlite_helpers as _ensure_lite)
                _ensure_lite(engine)
            except Exception:
                pass
        with engine.connect() as conn:
            def _exec(_sql, _p=None):
                """Execute with honest cross-table errors: an UndefinedColumn
                (or sqlite/mysql equivalent) becomes NO_JOIN_PATH, never
                ANALYSIS_FAILED; anything else re-raises."""
                try:
                    return conn.execute(sql_text(_sql), _p or {})
                except _NoJoinPath:
                    raise
                except Exception as _e:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    log.warning("explore sql failed: %s sql=%.300s error=%s",
                                type(_e).__name__, _sql, _e)
                    if _is_undefined_column(_e):
                        raise _NoJoinPath({
                            "ok": False, "code": "NO_JOIN_PATH",
                            "detail": (
                                f"Could not resolve a join for "
                                f"'{breakdown_col or metric_col}' from "
                                f"'{primary_table}'. Try a dimension from "
                                f"'{primary_table}'."),
                        })
                    raise
            # Build aggregation query
            metric_label = (COUNT_LABEL if _is_count
                            else display_name(metric_col, metric_col))
            time_label = display_name(time_col, time_col) if time_col else None
            breakdown_label = display_name(breakdown_col, breakdown_col) if breakdown_col else None

            # PNG-grade aggregation: dialect-aware time bucketing with a real
            # previous-period comparison. Grain labels (Monthly/Weekly/
            # Quarterly/Yearly) map onto the first DATE column of the primary
            # table; Monthly is the default.
            MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

            def _qi(name: str) -> str:
                if dialect == "mysql":
                    return "`" + str(name).replace("`", "``") + "`"
                return '"' + str(name).replace('"', '""') + '"'

            def _to_date(v):
                # Single shared tolerant parser (services.intelligence_validation):
                # ISO, MM/DD/YYYY, MM-DD-YYYY, DD-Mon-YYYY, month names,
                # epoch s/ms, YYYYMMDD. Service and validator share it so
                # bucketing can never disagree with validation.
                try:
                    return _shared_to_date(v)
                except Exception:
                    return None

            def _fnum(v):
                """NaN/Inf-safe float: unparseable or non-finite -> 0.0.

                Shape-based guard — a single NaN (e.g. postgres float column
                with NaN values, where SUM propagates NaN) must never poison
                a whole breakdown or total.
                """
                try:
                    f = float(v)
                except Exception:
                    return 0.0
                if math.isnan(f) or math.isinf(f):
                    return 0.0
                return f

            def _r2(v):
                try:
                    f = float(v)
                    if math.isnan(f) or math.isinf(f):
                        return 0.0
                    return round(f, 2)
                except Exception:
                    return 0.0

            def _pct(cur, prev):
                try:
                    if prev is None or float(prev) == 0:
                        return None
                    return round((float(cur) - float(prev)) / abs(float(prev)) * 100, 2)
                except Exception:
                    return None

            _grain_raw = str(request.get("_grain_label") or time_col_raw or "Monthly")
            _gl = _grain_raw.lower()
            if "week" in _gl:
                grain = "weekly"
            elif "quarter" in _gl:
                grain = "quarterly"
            elif "year" in _gl:
                grain = "yearly"
            else:
                grain = "monthly"
            grain_label = {"weekly": "Weekly", "quarterly": "Quarterly",
                           "yearly": "Yearly"}.get(grain, "Monthly")

            _cmp = str(compare or "previous_period").strip().lower().replace(" ", "_").replace("-", "_")
            do_compare = not (_cmp in ("no_comparison", "no", "none")
                              or (_cmp.startswith("no_") and "compar" in _cmp))

            _proles = roles.get(primary_table) or {}
            if time_col and _proles.get(time_col) == "DATE":
                date_col = time_col
            else:
                # Prefer a real event DATE — never a birth/hire date when an
                # alternative exists (validation fails closed on those).
                date_col = next(
                    (c for c, r in _proles.items()
                     if r == "DATE" and not _is_birth_hire_col(c)), None)
                if date_col is None:
                    date_col = next((c for c, r in _proles.items() if r == "DATE"), None)
            if date_col and _home_table(roles, date_col) not in (None, primary_table):
                # An explicitly requested DATE column may live in another
                # table — keep it only when a join path exists.
                _dh0 = _home_table(roles, date_col)
                if _find_join_path(mapping, roles, primary_table, _dh0) is None:
                    date_col = None
            if not date_col and (not time_col or _home_table(roles, time_col) != primary_table):
                # No time requested, or the requested/mapped DATE column lives
                # in another table: look for a DATE column in a table
                # reachable within 2 hops so monthly bucketing still works
                # cross-table (e.g. orders.order_date for an
                # order_details metric).
                # Prefer the requested/mapped time column's own home table
                # when it holds an event date and is reachable; otherwise
                # take the best reachable event DATE (never birth/hire).
                _th = _home_table(roles, time_col) if time_col else None
                if (_th and _th != primary_table
                        and not _is_birth_hire_col(time_col)
                        and _find_join_path(mapping, roles, primary_table, _th) is not None):
                    date_col = time_col
                if not date_col:
                    _bc, _bh = _best_date_col(mapping, roles, primary_table)
                    if _bc and (_bh == primary_table or _find_join_path(
                            mapping, roles, primary_table, _bh) is not None):
                        date_col = _bc
                if not date_col:
                    # Fallback: TEXT columns whose stored values actually parse
                    # as dates (e.g. walmart "date" TEXT holding MM-DD-YYYY,
                    # classified IDENTIFIER by the name heuristic). Shape-based:
                    # probe DISTINCT values, require >=50% parseable with a
                    # sane range. Primary table only — cross-table dates are
                    # covered by the DATE-role path.
                    for _pc in _probe_text_cols(
                            roles.get(primary_table) or {}, mapping, primary_table):
                        try:
                            _pqt = _qtbl(
                                primary_table,
                                _table_schema(mapping, primary_table, conn,
                                              dialect, {}), dialect)
                            _pr = _exec(
                                f"SELECT DISTINCT {_qi(_pc)} AS _d FROM "
                                f"{_pqt} LIMIT 200")
                            _pk = list(_pr.keys())
                            _pvals = [dict(zip(_pk, _r)).get("_d")
                                      for _r in _pr.fetchall()]
                            if not _pvals:
                                continue
                            _pp = [_to_date(v) for v in _pvals]
                            _pp = [d for d in _pp if d is not None]
                            if (len(_pp) >= 3
                                    and len(_pp) >= 0.5 * len(_pvals)
                                    and min(d.year for d in _pp) >= 1970
                                    and max(d.year for d in _pp) <= 2100):
                                date_col = _pc
                                break
                        except _NoJoinPath:
                            raise
                        except Exception:
                            continue

            # ---- cross-table join planning ---------------------------------
            # Breakdown / date / filter columns may live outside the primary
            # table. Resolve each home table, merge the 1–2 hop paths, and
            # expose every table under an alias (t0 = primary, t1, ...).
            # ALL column references below are alias-qualified so cross-table
            # combinations can never produce an UndefinedColumn crash.
            _need: list = []
            if breakdown_col:
                _bh = _home_table(roles, breakdown_col)
                if _bh and _bh != primary_table:
                    _need.append(_bh)
            _date_home = _home_table(roles, date_col) if date_col else None
            if _date_home and _date_home != primary_table:
                _need.append(_date_home)
            _filter_specs: list = []  # (param_key, column, home_table or None)
            for _fi, _flt in enumerate(filters or []):
                if not isinstance(_flt, dict):
                    continue
                _fcol = _resolve_column(roles, _flt.get("column")) or _flt.get("column")
                if not _fcol:
                    continue
                _fh = _home_table(roles, _fcol)
                if not _fh:
                    continue
                if _fh != primary_table and _fh not in _need:
                    _need.append(_fh)
                _filter_specs.append((f"flt{_fi}", _fcol, _fh, _flt.get("value")))
            _need = list(dict.fromkeys(_need))
            _join_hops: list = _find_join_path(mapping, roles, primary_table, _need) if _need else []
            if _join_hops is None:
                # Degrade gracefully per target: the breakdown was already
                # proven reachable above, so only date/filters drop out here.
                _join_hops = []
                _drop_tables: set = set()
                for _t in _need:
                    _p = _find_join_path(mapping, roles, primary_table, _t)
                    if _p is None:
                        _drop_tables.add(_t)
                    else:
                        _known = {h[0] for h in _join_hops} | {primary_table}
                        _join_hops.extend([h for h in _p if h[0] not in _known])
                if _date_home in _drop_tables:
                    date_col, _date_home = None, None
                _filter_specs = [s for s in _filter_specs if s[2] not in _drop_tables]
                if breakdown_col and _home_table(roles, breakdown_col) in _drop_tables:
                    _bh2 = _home_table(roles, breakdown_col)
                    return _no_join_path_result(roles, primary_table, _bh2, breakdown_col)

            _alias: Dict[str, str] = {primary_table: "t0"}
            for _ht, _hl, _hr in _join_hops:
                for _t in (_hl[0], _hr[0], _ht):
                    if _t not in _alias:
                        _alias[_t] = f"t{len(_alias)}"

            def _ref(_t: str, _c: str) -> str:
                return f"{_alias[_t]}.{_qi(_c)}"

            _schema_cache: dict = {}

            def _qtbl_for(_t: str) -> str:
                return _qtbl(_t, _table_schema(mapping, _t, conn, dialect, _schema_cache), dialect)

            _join_sql = " ".join(
                f"INNER JOIN {_qtbl_for(_ht)} AS {_alias[_ht]} "
                f"ON {_ref(_hl[0], _hl[1])} = {_ref(_hr[0], _hr[1])}"
                for _ht, _hl, _hr in _join_hops)
            _qt = f"{_qtbl_for(primary_table)} AS t0" + (f" {_join_sql}" if _join_sql else "")
            # Storage-type-aware metric expression (generic, shape-based):
            # TEXT-stored measures get a numeric cast (never SUM(text));
            # float NaN values are NULLed so one NaN never poisons a SUM.
            _metric_dtype = ("" if _is_count
                             else _col_data_type(mapping, primary_table, metric_col))
            _qm_raw = f"t0.{_qi(metric_col)}" if not _is_count else "t0.*"
            _qm = ("*" if _is_count
                   else _nan_safe_sql(
                       _numeric_metric_sql(_qm_raw, _metric_dtype, dialect),
                       _metric_dtype, dialect))
            _agg = "COUNT(*)" if _is_count else f"SUM({_qm})"
            _qb = _ref(_home_table(roles, breakdown_col), breakdown_col) if breakdown_col else None
            _qd = _ref(_date_home, date_col) if date_col else None
            # Native DATE storage -> SQL bucket functions; anything else
            # (TEXT MM-DD-YYYY, epoch ints, ...) -> Python bucketing with the
            # shared tolerant parser (SQL string comparison across spellings
            # is meaningless).
            _date_storage = ("native" if _is_native_date_dtype(
                _col_data_type(mapping, _date_home or primary_table, date_col)
                if date_col else "") else "text")
            if _date_storage == "text" and dialect == "sqlite" and date_col:
                # ISO-shaped TEXT ('YYYY-MM-DD...') is strftime-buckettable
                # in SQL — no need for million-row Python bucketing. Probe a
                # small DISTINCT sample; >=80% ISO prefix => native path.
                # Generic (storage-shape check, never table/column names).
                try:
                    import re as _re
                    _iso_probe = _exec(
                        f"SELECT DISTINCT {_qd} AS _d FROM {_qt} LIMIT 2000").fetchall()
                    _iso_hit = _iso_tot = 0
                    for _pr in _iso_probe:
                        _v = str(_pr[0]) if _pr and _pr[0] is not None else ""
                        if not _v.strip():
                            continue
                        _iso_tot += 1
                        if _re.match(r"^\d{4}-\d{2}-\d{2}", _v.strip()):
                            _iso_hit += 1
                    if _iso_tot >= 50 and _iso_hit / _iso_tot >= 0.8:
                        _date_storage = "native"
                except _NoJoinPath:
                    raise
                except Exception:
                    pass

            # Equality filters on real columns (home table may be joined).
            # Unjoinable filter columns are skipped — never crash.
            _filter_clauses = []
            _params: dict = {}
            for _fk, _fcol, _fh, _fval in _filter_specs:
                if _fh not in _alias:
                    continue
                _filter_clauses.append(f"{_ref(_fh, _fcol)} = :{_fk}")
                _params[_fk] = _fval

            def _where(extra=()):
                _all = list(_filter_clauses) + [e for e in (extra or []) if e]
                return (" WHERE " + " AND ".join(_all)) if _all else ""

            # Resolve current/previous windows, then fetch grouped cells.
            # Default: latest calendar year vs the same months prior year.
            # Explicit date_range dict with start/end: that window vs the
            # prior equal-length window immediately preceding it.
            from datetime import date as _date, timedelta as _td

            _explicit_start = _explicit_end = None
            if isinstance(date_range, dict):
                if date_range.get("start"):
                    _explicit_start = _to_date(date_range["start"])
                if date_range.get("end"):
                    _explicit_end = _to_date(date_range["end"])

            _has_time = bool(date_col)
            _cells: dict = {}        # (category, "cur"|"prev", bucket_idx) -> float
            _cat_only: list = []     # no-DATE path: [(category, value)]
            _snap_scope_total: int | None = None
            _pref_total_groups: int | None = None
            _cur_seq: list = []
            _prev_seq: list = []
            _labels: list = []
            _cur_start = _cur_end = _prev_start = _prev_end = None
            _cur_vlabel = _prev_vlabel = ""
            _explicit_window = False
            _raw_count = 0
            _cat_seen: dict = {}

            def _catkey(cat):
                c = str(cat) if cat not in (None, "") else "Unknown"
                if c not in _cat_seen:
                    _cat_seen[c] = True
                return c

            def _add_cell(cat, window, idx, val):
                v = _fnum(val)
                key = (_catkey(cat), window, idx)
                _cells[key] = _cells.get(key, 0.0) + v

            def _fetch_snapshot():
                out = []
                _sel = f"{_agg} AS _val"
                _grp = ""
                _snap_extra = []
                _snap_params = {}
                _snap_total = None
                if breakdown_col:
                    _sel = f"{_qb} AS _cat, " + _sel
                    _grp = f" GROUP BY {_qb}"
                    # Same top-N bounding as the time path: high-cardinality
                    # breakdowns must not stream every group into Python.
                    try:
                        _snap_total = int(_exec(
                            f"SELECT COUNT(DISTINCT {_qb}) FROM {_qt}{_where()}",
                            _params).scalar() or 0)
                        if _snap_total > _EXPLORE_TOP_CATS:
                            _top_rows = _exec(
                                f"SELECT {_qb} AS _cat, {_agg} AS _val FROM {_qt}"
                                f"{_where()} GROUP BY {_qb} "
                                f"ORDER BY _val DESC LIMIT {_EXPLORE_TOP_CATS}",
                                _params).fetchall()
                            _tc = [_catkey(_r[0]) for _r in _top_rows]
                            if _tc:
                                _plist = ", ".join(f":_sc{i}" for i in range(len(_tc)))
                                _snap_extra = [f"{_qb} IN ({_plist})"]
                                _snap_params = {f"_sc{i}": _c for i, _c in enumerate(_tc)}
                    except _NoJoinPath:
                        raise
                    except Exception:
                        log.warning("Explore snapshot top-N prefetch failed; unbounded fallback",
                                    exc_info=True)
                _sql = f"SELECT {_sel} FROM {_qt}{_where(_snap_extra)}{_grp}"
                if breakdown_col:
                    _sql += " ORDER BY _val DESC"
                _sp = dict(_params)
                _sp.update(_snap_params)
                _res = _exec(_sql, _sp)
                _cols = list(_res.keys())
                for _row in _res.fetchall():
                    _d = dict(zip(_cols, _row))
                    _c = _catkey(_d.get("_cat")) if breakdown_col else "Overall"
                    out.append((_c, _fnum(_d.get("_val"))))
                return out, _snap_total

            def _fetch_count(extra=(), extra_params=None):
                try:
                    _p = dict(_params)
                    if extra_params:
                        _p.update(extra_params)
                    return int(_exec(
                        f"SELECT COUNT(*) FROM {_qt}{_where(extra)}", _p).scalar() or 0)
                except _NoJoinPath:
                    raise
                except Exception:
                    return 0

            _snap_scope_total: int | None = None
            if not _has_time:
                _cat_only, _snap_scope_total = _fetch_snapshot()
                _raw_count = _fetch_count()
            else:
                _native = (_date_storage == "native")
                try:
                    if _native:
                        _mm = _exec(
                            f"SELECT MIN({_qd}), MAX({_qd}) FROM {_qt}").fetchone()
                    else:
                        # Lexicographic MIN/MAX over TEXT/odd storage is
                        # meaningless (e.g. MM-DD-YYYY) — resolve the true
                        # range from DISTINCT values via the shared parser.
                        _mm = None
                        try:
                            _dr = _exec(
                                f"SELECT DISTINCT {_qd} AS _d FROM {_qt} "
                                f"LIMIT 10000")
                            _dk = list(_dr.keys())
                            _parsed = [_to_date(dict(zip(_dk, _r)).get("_d"))
                                       for _r in _dr.fetchall()]
                            _parsed = [d for d in _parsed if d is not None]
                            if _parsed:
                                _mm = (min(_parsed), max(_parsed))
                        except _NoJoinPath:
                            raise
                        except Exception:
                            _mm = None
                except _NoJoinPath:
                    raise
                except Exception:
                    _mm = None
                _mind = _to_date(_mm[0]) if _mm else None
                _maxd = _to_date(_mm[1]) if _mm else None
                if _mind is None or _maxd is None:
                    # DATE column exists but values are not parseable —
                    # honest category snapshot instead of fabricated buckets.
                    _has_time = False
                    _cat_only, _snap_scope_total = _fetch_snapshot()
                    _raw_count = _fetch_count()
                elif (_mind.year < 1970 or _maxd.year > 2100
                        or _mind > _maxd
                        or (_maxd.year < 1971 and not (
                            _explicit_start and _explicit_end
                            and _explicit_start <= _explicit_end))):
                    # Garbage/degenerate date range (e.g. epoch-0-ish doubles
                    # whose default latest-year-vs-prior-year windows would
                    # start before 1970) — windows would be absurd, so
                    # snapshot honestly instead of failing validation.
                    _has_time = False
                    _cat_only, _snap_scope_total = _fetch_snapshot()
                    _raw_count = _fetch_count()
                else:
                    if _explicit_start and _explicit_end and _explicit_start <= _explicit_end:
                        _cur_start, _cur_end = _explicit_start, _explicit_end
                        _days = (_cur_end - _cur_start).days + 1
                        _prev_end = _cur_start - _td(days=1)
                        _prev_start = _prev_end - _td(days=_days - 1)
                        _explicit_window = True
                    else:
                        _cy = _maxd.year
                        _cur_start, _cur_end = _date(_cy, 1, 1), _date(_cy, 12, 31)
                        _prev_start, _prev_end = _date(_cy - 1, 1, 1), _date(_cy - 1, 12, 31)
                    if _explicit_window:
                        _cur_vlabel = f"{_cur_start.isoformat()} to {_cur_end.isoformat()}"
                        _prev_vlabel = f"{_prev_start.isoformat()} to {_prev_end.isoformat()}"
                    else:
                        _cur_vlabel = str(_cur_start.year)
                        _prev_vlabel = str(_prev_start.year)

                    def _month_seq(a, b):
                        out = []
                        y, m = a.year, a.month
                        while (y, m) <= (b.year, b.month):
                            out.append((y, m))
                            m += 1
                            if m > 12:
                                m, y = 1, y + 1
                        return out

                    def _quarter_seq(a, b):
                        out = []
                        y, q = a.year, (a.month - 1) // 3 + 1
                        ey, eq = b.year, (b.month - 1) // 3 + 1
                        while (y, q) <= (ey, eq):
                            out.append((y, q))
                            q += 1
                            if q > 4:
                                q, y = 1, y + 1
                        return out

                    def _week_seq(a, b):
                        out = []
                        _d = b - _td(days=b.weekday())
                        while _d + _td(days=6) >= a and len(out) < 52:
                            out.append(_d)
                            _d -= _td(weeks=1)
                        return sorted(out)

                    if grain == "monthly":
                        if _explicit_window:
                            _cseq = _month_seq(_cur_start, _cur_end)
                            _pseq = _month_seq(_prev_start, _prev_end)
                            if len(_cseq) > 12:
                                _cseq = _cseq[-12:]
                            if len(_pseq) > len(_cseq):
                                _pseq = _pseq[-len(_cseq):]
                            _multi = len({y for y, _ in _cseq}) > 1
                            _labels = [f"{MONTH_LABELS[m - 1]} '{str(y)[2:]}" if _multi
                                       else MONTH_LABELS[m - 1] for y, m in _cseq]
                        else:
                            _cseq = [(_cur_start.year, m) for m in range(1, 13)]
                            _pseq = [(_prev_start.year, m) for m in range(1, 13)]
                            _labels = list(MONTH_LABELS)
                        _cur_seq, _prev_seq = _cseq, _pseq
                    elif grain == "quarterly":
                        if _explicit_window:
                            _cseq = _quarter_seq(_cur_start, _cur_end)[-12:]
                            _pseq = _quarter_seq(_prev_start, _prev_end)
                            if len(_pseq) > len(_cseq):
                                _pseq = _pseq[-len(_cseq):]
                            _labels = [f"Q{q} {y}" for y, q in _cseq]
                        else:
                            _cseq = [(_cur_start.year, q) for q in range(1, 5)]
                            _pseq = [(_prev_start.year, q) for q in range(1, 5)]
                            _labels = [f"Q{q}" for _, q in _cseq]
                        _cur_seq, _prev_seq = _cseq, _pseq
                    elif grain == "yearly":
                        if _explicit_window:
                            _y0, _y1 = _cur_start.year, _cur_end.year
                            if _y1 - _y0 > 11:
                                _y0 = _y1 - 11
                        else:
                            _y1 = _cur_end.year
                            _y0 = _y1 - 7
                        _cur_seq = list(range(_y0, _y1 + 1))
                        _prev_seq = [y - 1 for y in _cur_seq]
                        _labels = [str(y) for y in _cur_seq]
                        _cur_start, _cur_end = _date(_y0, 1, 1), _date(_y1, 12, 31)
                        _prev_start, _prev_end = _date(_y0 - 1, 1, 1), _date(_y0 - 1, 12, 31)
                    else:  # weekly: last up-to-12 weeks of each window
                        _cw = _week_seq(_cur_start, _cur_end)[-12:]
                        _pw = _week_seq(_prev_start, _prev_end)
                        if len(_pw) > len(_cw):
                            _pw = _pw[-len(_cw):]
                        _cur_seq, _prev_seq = _cw, _pw
                        _labels = [f"{MONTH_LABELS[d.month - 1]} {d.day}" for d in _cw]

                    def _key_of(dt):
                        if grain == "monthly":
                            return (dt.year, dt.month)
                        if grain == "quarterly":
                            return (dt.year, (dt.month - 1) // 3 + 1)
                        if grain == "yearly":
                            return dt.year
                        return dt - _td(days=dt.weekday())

                    def _fetch_raw_cells():
                        # Python bucketing path, parsed with the shared
                        # tolerant parser. Native storage: grouped raw dates
                        # with the SQL span filter (existing behaviour).
                        # Non-native storage (TEXT/odd dates): row-level fetch
                        # with NO SQL date filter (string comparison across
                        # spellings is meaningless); windows are applied in
                        # Python below so counts and buckets stay honest.
                        if _native:
                            _sel = f"{_qd} AS _d, {_agg} AS _val"
                            _grp = f" GROUP BY {_qd}"
                            if breakdown_col:
                                _sel = f"{_qd} AS _d, {_qb} AS _cat, {_agg} AS _val"
                                _grp = f" GROUP BY {_qd}, {_qb}"
                            _sql = (f"SELECT {_sel} FROM {_qt}"
                                    f"{_where([f'{_qd} >= :_ps', f'{_qd} <= :_ce'] + _top_extra)}")
                            _p = dict(_params)
                            _p["_ps"] = _prev_start.isoformat()
                            _p["_ce"] = _cur_end.isoformat()
                            _p.update(_top_params)
                        else:
                            _vsel = "1 AS _val" if _is_count else f"{_qm} AS _val"
                            _sel = f"{_qd} AS _d, {_vsel}"
                            if breakdown_col:
                                _sel = f"{_qd} AS _d, {_qb} AS _cat, {_vsel}"
                            _sql = f"SELECT {_sel} FROM {_qt}{_where()}"
                            _p = dict(_params)
                        _res = _exec(_sql, _p)
                        _cols = list(_res.keys())
                        for _row in _res.fetchall():
                            _d = dict(zip(_cols, _row))
                            _dt = _to_date(_d.get("_d"))
                            if _dt is None:
                                continue
                            if not _native and not (
                                    _prev_start <= _dt <= _cur_end):
                                continue
                            _c = _catkey(_d.get("_cat")) if breakdown_col else "Overall"
                            yield (_c, _dt, _fnum(_d.get("_val")))

                    _span = [f"{_qd} >= :_ps", f"{_qd} <= :_ce"]
                    _p2 = dict(_params)
                    _p2["_ps"] = _prev_start.isoformat()
                    _p2["_ce"] = _cur_end.isoformat()
                    # Two-phase top-categories: high-cardinality breakdowns
                    # (e.g. 500k+ distinct products) would otherwise pull
                    # millions of (category x bucket) rows into Python.
                    # Fetch the top 50 groups + total group count first, then
                    # restrict every downstream query to them. Generic:
                    # driven by COUNT DISTINCT, never table/column names.
                    _TOP_CATS_LIMIT = _EXPLORE_TOP_CATS
                    _top_cats: list | None = None
                    _total_groups: int | None = None
                    _top_extra: list = []
                    _top_params: dict = {}
                    _span_cats = list(_span)
                    _p2c = dict(_p2)
                    if breakdown_col:
                        try:
                            # Span filter is only meaningful on native DATE
                            # storage; on TEXT/odd dates it would silently
                            # empty the prefetch, so use full-table scope.
                            _wspan = _span if _native else []
                            _wp = _p2 if _native else dict(_params)
                            _pref_total_groups = int(_exec(
                                f"SELECT COUNT(DISTINCT {_qb}) FROM {_qt}"
                                f"{_where(_wspan)}", _wp).scalar() or 0)
                            _top_rows = _exec(
                                f"SELECT {_qb} AS _cat, {_agg} AS _val FROM {_qt}"
                                f"{_where(_wspan)} GROUP BY {_qb} "
                                f"ORDER BY _val DESC LIMIT {_TOP_CATS_LIMIT}",
                                _wp).fetchall()
                            _top_cats = [_catkey(_r[0]) for _r in _top_rows]
                            if _pref_total_groups > _TOP_CATS_LIMIT and _top_cats:
                                _top_params = {f"_tc{i}": _c for i, _c in enumerate(_top_cats)}
                                _tc_list = ", ".join(f":_tc{i}" for i in range(len(_top_cats)))
                                _top_extra = [f"{_qb} IN ({_tc_list})"]
                                _span_cats = list(_span) + _top_extra
                                _p2c = dict(_p2)
                                _p2c.update(_top_params)
                            else:
                                _top_cats = None
                        except _NoJoinPath:
                            raise
                        except Exception:
                            log.warning("Explore top-categories prefetch failed; unbounded fallback",
                                        exc_info=True)
                            _top_cats = None
                            _pref_total_groups = None
                            _span_cats = list(_span)
                            _p2c = dict(_p2)
                    if _native:
                        _raw_count = _fetch_count(_span, {"_ps": _p2["_ps"], "_ce": _p2["_ce"]})
                    else:
                        # Row count inside the span, parsed in Python (TEXT
                        # dates cannot be range-filtered in SQL).
                        _raw_count = 0
                        try:
                            for _c, _dt, _v in _fetch_raw_cells():
                                _raw_count += 1
                        except _NoJoinPath:
                            raise
                        except Exception:
                            _raw_count = 0
                        # Rewind: _fetch_raw_cells is a generator over fresh
                        # SQL each call, so the bucketing pass below re-runs
                        # it independently.
                    if _raw_count == 0:
                        _trend_title = (f"{metric_label} by {breakdown_label}"
                                        if breakdown_col and breakdown_label else metric_label)
                        return {
                            "ok": False,
                            "detail": "No rows matched this selection in your data.",
                            "code": "NO_DATA",
                            "reason": "NO_DATA",
                            "metric": metric_label,
                            "breakdown": breakdown_label,
                            "trend": {"title": _trend_title, "subtitle": date_label, "points": []},
                            "series": [], "points": [],
                            "breakdown_data": [], "breakdown_rows": [], "rows": [], "table": [],
                            "time_series": [],
                            "chart": {"type": "bar", "title": _trend_title, "data": [], "series": []},
                            "title": _trend_title, "subtitle": date_label,
                            "summary": "No rows matched this selection in your data.",
                            "observations": [],
                            "observation": {"summary": "", "bullets": [],
                                            "basis_note": "No rows matched this selection."},
                            "aria_observation": {"summary": "", "bullets": [],
                                                 "basis_note": "No rows matched this selection."},
                            "basis_note": "No rows matched this selection.",
                            "total": 0, "total_display": _human_total(0),
                            "previous_total": None, "change": None, "change_pct": None,
                            "row_count": 0, "analysis_period": date_label,
                        }

                    _hits: list = []  # (category, bucket_key, value)
                    _cat_sel = f"{_qb} AS _cat, " if breakdown_col else ""
                    _cat_grp = f"{_qb}, " if breakdown_col else ""
                    _sql_ok = True
                    # Full-scope bucket shape (no category): reused below so
                    # the trend line always covers ALL data even when the
                    # breakdown is top-N restricted. Captured per grain.
                    _full_bsql: str | None = None
                    _full_bkey: str | None = None
                    try:
                        if not _native:
                            # Non-native DATE storage (TEXT MM-DD-YYYY, epoch
                            # ints, ...): SQL bucket functions / string
                            # comparison are meaningless — Python bucketing
                            # below handles it via the shared parser.
                            raise ValueError("non-native date storage")
                        if grain == "monthly":
                            # DATE_TRUNC (postgres) / DATE_FORMAT (mysql) /
                            # strftime (sqlite) month buckets.
                            if dialect == "postgresql":
                                _bexpr = f"DATE_TRUNC('month', CAST({_qd} AS TIMESTAMP))"
                            elif dialect == "mysql":
                                _bexpr = f"DATE_FORMAT({_qd}, '%Y-%m-01')"
                            else:
                                _bexpr = f"strftime('%Y-%m-01', {_qd})"
                            _sql = (f"SELECT {_cat_sel}{_bexpr} AS _bucket, {_agg} AS _val "
                                    f"FROM {_qt}{_where(_span_cats)} GROUP BY {_cat_grp}_bucket")
                            _full_bsql = (f"SELECT {_bexpr} AS _bucket, {_agg} AS _val "
                                          f"FROM {_qt}{_where(_span)} GROUP BY _bucket")
                            _full_bkey = "bucket"
                            _res = _exec(_sql, _p2c)
                            _cols = list(_res.keys())
                            for _row in _res.fetchall():
                                _d = dict(zip(_cols, _row))
                                _b = _d.get("_bucket")
                                try:
                                    if hasattr(_b, "year"):
                                        _k = (_b.year, _b.month)
                                    else:
                                        _s = str(_b)[:7]
                                        _k = (int(_s[:4]), int(_s[5:7]))
                                except Exception:
                                    continue
                                _c = _catkey(_d.get("_cat")) if breakdown_col else "Overall"
                                _hits.append((_c, _k, _d.get("_val")))
                        elif grain == "quarterly":
                            if dialect == "postgresql":
                                _yexpr = "CAST(EXTRACT(YEAR FROM CAST(%s AS TIMESTAMP)) AS INTEGER)" % _qd
                                _qexpr = "CAST(EXTRACT(QUARTER FROM CAST(%s AS TIMESTAMP)) AS INTEGER)" % _qd
                            elif dialect == "mysql":
                                _yexpr = f"YEAR({_qd})"
                                _qexpr = f"QUARTER({_qd})"
                            else:
                                _yexpr = f"CAST(strftime('%Y', {_qd}) AS INTEGER)"
                                _qexpr = (f"CAST((CAST(strftime('%m', {_qd}) AS INTEGER)+2)/3 "
                                          f"AS INTEGER)")
                            _sql = (f"SELECT {_cat_sel}{_yexpr} AS _yr, {_qexpr} AS _qq, "
                                    f"{_agg} AS _val FROM {_qt}{_where(_span_cats)} "
                                    f"GROUP BY {_cat_grp}_yr, _qq")
                            _full_bsql = (f"SELECT {_yexpr} AS _yr, {_qexpr} AS _qq, "
                                          f"{_agg} AS _val FROM {_qt}{_where(_span)} "
                                          f"GROUP BY _yr, _qq")
                            _full_bkey = "yrqq"
                            _res = _exec(_sql, _p2c)
                            _cols = list(_res.keys())
                            for _row in _res.fetchall():
                                _d = dict(zip(_cols, _row))
                                try:
                                    if _d.get("_yr") is None or _d.get("_qq") is None:
                                        continue
                                    _k = (int(_d.get("_yr")), int(_d.get("_qq")))
                                except Exception:
                                    continue
                                _c = _catkey(_d.get("_cat")) if breakdown_col else "Overall"
                                _hits.append((_c, _k, _d.get("_val")))
                        elif grain == "yearly":
                            if dialect == "postgresql":
                                _yexpr = "CAST(EXTRACT(YEAR FROM CAST(%s AS TIMESTAMP)) AS INTEGER)" % _qd
                            elif dialect == "mysql":
                                _yexpr = f"YEAR({_qd})"
                            else:
                                _yexpr = f"CAST(strftime('%Y', {_qd}) AS INTEGER)"
                            _sql = (f"SELECT {_cat_sel}{_yexpr} AS _yr, {_agg} AS _val "
                                    f"FROM {_qt}{_where(_span_cats)} GROUP BY {_cat_grp}_yr")
                            _full_bsql = (f"SELECT {_yexpr} AS _yr, {_agg} AS _val "
                                          f"FROM {_qt}{_where(_span)} GROUP BY _yr")
                            _full_bkey = "yr"
                            _res = _exec(_sql, _p2c)
                            _cols = list(_res.keys())
                            for _row in _res.fetchall():
                                _d = dict(zip(_cols, _row))
                                try:
                                    if _d.get("_yr") is None:
                                        continue
                                    _k = int(_d.get("_yr"))
                                except Exception:
                                    continue
                                _c = _catkey(_d.get("_cat")) if breakdown_col else "Overall"
                                _hits.append((_c, _k, _d.get("_val")))
                        else:  # weekly Monday buckets
                            if dialect == "postgresql":
                                _wexpr = f"CAST(DATE_TRUNC('week', CAST({_qd} AS TIMESTAMP)) AS DATE)"
                            elif dialect == "mysql":
                                _wexpr = f"DATE(DATE_SUB({_qd}, INTERVAL WEEKDAY({_qd}) DAY))"
                            else:
                                _wexpr = f"date({_qd}, 'weekday 0', '-6 days')"
                            _sql = (f"SELECT {_cat_sel}{_wexpr} AS _bucket, {_agg} AS _val "
                                    f"FROM {_qt}{_where(_span_cats)} GROUP BY {_cat_grp}_bucket")
                            _full_bsql = (f"SELECT {_wexpr} AS _bucket, {_agg} AS _val "
                                          f"FROM {_qt}{_where(_span)} GROUP BY _bucket")
                            _full_bkey = "weekbucket"
                            _res = _exec(_sql, _p2c)
                            _cols = list(_res.keys())
                            for _row in _res.fetchall():
                                _d = dict(zip(_cols, _row))
                                _dt = _to_date(_d.get("_bucket"))
                                if _dt is None:
                                    continue
                                _k = _dt - _td(days=_dt.weekday())
                                _c = _catkey(_d.get("_cat")) if breakdown_col else "Overall"
                                _hits.append((_c, _k, _d.get("_val")))
                    except _NoJoinPath:
                        raise
                    except Exception:
                        if _native:
                            log.warning("Explore time-bucket query failed; using Python bucketing",
                                        exc_info=True)
                        _sql_ok = False
                    if (not _sql_ok or not _hits) and _raw_count > 0:
                        # Dialect bucket functions could not parse the stored
                        # date values — bucketize the grouped raw dates in
                        # Python instead of returning fabricated empties.
                        _hits = []
                        for _c, _dt, _v in _fetch_raw_cells():
                            _hits.append((_c, _key_of(_dt), _v))
                    _cur_idx = {k: i for i, k in enumerate(_cur_seq)}
                    _prev_idx = {k: i for i, k in enumerate(_prev_seq)}
                    for _c, _k, _v in _hits:
                        if _k in _cur_idx:
                            _add_cell(_c, "cur", _cur_idx[_k], _v)
                        elif _k in _prev_idx:
                            _add_cell(_c, "prev", _prev_idx[_k], _v)
                    # Full-scope trend: when the breakdown is top-N
                    # restricted, per-bucket totals across ALL groups come
                    # from one extra bucket-only query (dozens of rows).
                    # The trend line then describes all data, not the top 50.
                    _full_cur: list | None = None
                    _full_prev: list | None = None
                    if (_top_extra and breakdown_col and _sql_ok and _full_bsql
                            and _full_bkey and _native):
                        try:
                            _fres = _exec(_full_bsql, _p2)
                            _fcols = list(_fres.keys())
                            _fb: dict = {}
                            for _frow in _fres.fetchall():
                                _fd = dict(zip(_fcols, _frow))
                                try:
                                    if _full_bkey == "weekbucket":
                                        _fdtw = _to_date(_fd.get("_bucket"))
                                        if _fdtw is None:
                                            continue
                                        _fk = _fdtw - _td(days=_fdtw.weekday())
                                    elif _full_bkey == "bucket":
                                        _bb = _fd.get("_bucket")
                                        if hasattr(_bb, "year"):
                                            _fk = (_bb.year, _bb.month)
                                        else:
                                            _bs = str(_bb)[:10]
                                            _fdt = _to_date(_bs)
                                            if _fdt is None:
                                                continue
                                            _fk = (_fdt.year, _fdt.month)
                                    elif _full_bkey == "yrqq":
                                        if _fd.get("_yr") is None or _fd.get("_qq") is None:
                                            continue
                                        _fk = (int(_fd.get("_yr")), int(_fd.get("_qq")))
                                    elif _full_bkey == "yr":
                                        if _fd.get("_yr") is None:
                                            continue
                                        _fk = int(_fd.get("_yr"))
                                    else:
                                        continue
                                except Exception:
                                    continue
                                _fb[_fk] = _fb.get(_fk, 0.0) + _fnum(_fd.get("_val"))
                            _full_cur = [0.0] * len(_cur_seq)
                            _full_prev = [0.0] * len(_prev_seq)
                            for _fk, _fv in _fb.items():
                                if _fk in _cur_idx:
                                    _full_cur[_cur_idx[_fk]] += _fv
                                elif _fk in _prev_idx:
                                    _full_prev[_prev_idx[_fk]] += _fv
                        except _NoJoinPath:
                            raise
                        except Exception:
                            log.warning("Explore full-scope trend fetch failed; restricted trend kept",
                                        exc_info=True)
                            _full_cur = _full_prev = None

            # Assemble the frontend contract from real aggregates only.
            _trend_title = (f"{metric_label} by {breakdown_label}"
                            if breakdown_col and breakdown_label else metric_label)
            total_current = 0.0
            # Snapshot (no time axis) has no previous window by construction —
            # report None (never 0.0), or the validator's fail-closed
            # snapshot rule fires.
            total_previous: float | None = (0.0 if do_compare and _has_time
                                            else None)
            breakdown_data: list = []
            time_series: list = []
            trend_points: list = []

            if not _has_time:
                for _c, _v in _cat_only:
                    total_current += _v
                total_current = _r2(total_current)
                if (_snap_scope_total is not None
                        and _snap_scope_total > len(_cat_only)):
                    # Top-N restriction active (high-cardinality breakdown):
                    # _cat_only covers the top groups only, so its sum is a
                    # partial total the validator's total_check fails closed
                    # on. Totals must describe ALL groups (mirrors the time
                    # path's dedicated full-span SUMs) — one cheap exact
                    # query; rows stay top-N with scope.truncated set.
                    try:
                        _tfull = _exec(
                            f"SELECT {_agg} AS _val FROM {_qt}{_where()}",
                            _params).fetchone()
                        total_current = _r2(_fnum((_tfull or [None])[0]))
                    except _NoJoinPath:
                        raise
                    except Exception:
                        pass
                _cat_only = sorted(_cat_only, key=lambda kv: kv[1], reverse=True)
                breakdown_data = [{"category": c, "current": _r2(v), "previous": None,
                                   "change": None, "change_pct": None, "trend": []}
                                  for c, v in _cat_only[:30]]
                trend_points = [{"label": c, "current": _r2(v), "previous": None}
                                for c, v in _cat_only[:30]]
                _period_label = date_label
            else:
                _n = len(_cur_seq)
                _entries = []  # [category, cur_arr, prev_arr, cur_sum, prev_sum]
                for _c in _cat_seen:
                    _ca = [0.0] * _n
                    _pa = [0.0] * _n
                    for ((_cc, _w, _i), _v) in _cells.items():
                        if _cc != _c or not (0 <= _i < _n):
                            continue
                        if _w == "cur":
                            _ca[_i] += _v
                        elif _w == "prev":
                            _pa[_i] += _v
                    if any(_ca) or any(_pa):
                        _entries.append([_c, _ca, _pa])
                if not breakdown_col and not _entries:
                    _entries.append(["Overall", [0.0] * _n, [0.0] * _n])
                for _e in _entries:
                    _e.append(_r2(sum(_e[1])))
                    _e.append(_r2(sum(_e[2])) if do_compare else None)
                _entries.sort(key=lambda e: e[3], reverse=True)
                # Full-scope totals + trend when the top-N restriction is
                # active: entries/buckets above cover only the top groups,
                # but totals and the trend line must describe ALL data.
                # Dedicated full-span SUMs are ~0.3s each — cheap and exact.
                _restricted = bool(_top_extra)
                if _restricted:
                    try:
                        _tcur = _exec(
                            f"SELECT {_agg} AS _val FROM {_qt}"
                            f"{_where([f'{_qd} >= :_cs', f'{_qd} <= :_ce'])}",
                            {**_params, "_cs": _cur_start.isoformat(),
                             "_ce": _cur_end.isoformat()}).fetchone()
                        total_current = _r2(_fnum((_tcur or [None])[0]))
                    except Exception:
                        total_current = _r2(sum(_e[3] for _e in _entries))
                    if do_compare:
                        try:
                            _tprev = _exec(
                                f"SELECT {_agg} AS _val FROM {_qt}"
                                f"{_where([f'{_qd} >= :_ps', f'{_qd} <= :_pe'])}",
                                {**_params, "_ps": _prev_start.isoformat(),
                                 "_pe": _prev_end.isoformat()}).fetchone()
                            total_previous = _r2(_fnum((_tprev or [None])[0]))
                        except Exception:
                            total_previous = _r2(sum(_e[4] for _e in _entries if _e[4] is not None))
                    else:
                        total_current = _r2(sum(_e[3] for _e in _entries))
                else:
                    total_current = _r2(sum(_e[3] for _e in _entries))
                    if do_compare:
                        total_previous = _r2(sum(_e[4] for _e in _entries if _e[4] is not None))
                for _i, _lab in enumerate(_labels):
                    if _full_cur is not None and _i < len(_full_cur):
                        _cc = _r2(_full_cur[_i])
                        _pp = (_r2(_full_prev[_i]) if _full_prev is not None and _i < len(_full_prev) and do_compare else None)
                    else:
                        _cc = _r2(sum(_e[1][_i] for _e in _entries))
                        _pp = _r2(sum(_e[2][_i] for _e in _entries)) if do_compare else None
                    trend_points.append({"label": _lab, "current": _cc, "previous": _pp})
                    time_series.append({"period": _lab, "value": _cc, "previous": _pp})
                if breakdown_col:
                    for _e in _entries[:30]:
                        _ch = _r2(_e[3] - _e[4]) if _e[4] is not None else None
                        breakdown_data.append({
                            "category": _e[0], "current": _e[3], "previous": _e[4],
                            "change": _ch, "change_pct": _pct(_e[3], _e[4]),
                            "trend": [_r2(v) for v in _e[1]],
                        })
                _period_label = (f"{_cur_vlabel} vs {_prev_vlabel}"
                                 if do_compare else _cur_vlabel)

            if _has_time and breakdown_col:
                _row_entries = [(e[0], e[3], e[4], e[1]) for e in _entries[:50]]
            elif _has_time:
                _row_entries = [(e[0], e[3], e[4], e[1]) for e in _entries[:1]]
            else:
                _row_entries = [(c, _r2(v), None, []) for c, v in _cat_only[:50]]
            rows_out = []
            for _c, _cs, _ps, _spark in _row_entries:
                _ch = _r2(_cs - _ps) if _ps is not None else None
                _cp = _pct(_cs, _ps)
                rows_out.append({
                    "category": _c, "label": _c, "name": _c,
                    "current": _cs, "current_value": _cs, "value": _cs,
                    "current_display": _human_total(_cs or 0),
                    "previous": _ps, "previous_value": _ps,
                    "change": _ch, "delta": _ch,
                    "change_pct": _cp, "pct_change": _cp,
                    "trend": [_r2(v) for v in (_spark or [])],
                })

            # Explicit truncation scope (generic, shape-based): totals and
            # previous_totals always cover ALL groups, while rows_out may be
            # capped (top-30 breakdown_data / top-50 rows). The validator uses
            # this scope to pick containment rules when truncated and keeps
            # the strict equality rules otherwise.
            _total_groups = ((len(_entries) if _pref_total_groups is None
                              else max(len(_entries), _pref_total_groups))
                             if _has_time
                             else (_snap_scope_total if _snap_scope_total is not None
                                   else len(_cat_only)))
            _scope = {"returned": len(rows_out),
                      "total_groups": _total_groups,
                      "truncated": bool(_total_groups > len(rows_out))}

            # Human-worded ARIA observation with real % growth + bullets.
            observations: list[str] = []
            bullets: list[str] = []
            if _has_time:
                _pct_all = _pct(total_current, total_previous)
                if total_previous is not None and _pct_all is not None:
                    _dir = "up" if _pct_all >= 0 else "down"
                    observations.append(
                        f"{metric_label} totaled {_human_total(total_current)} in {_cur_vlabel} — "
                        f"{_dir} {abs(_pct_all):.1f}% vs {_prev_vlabel} "
                        f"({_human_total(total_previous)}).")
                elif total_current:
                    observations.append(
                        f"{metric_label} totals {_human_total(total_current)} across "
                        f"{_raw_count} records in {_period_label}.")
                if breakdown_data:
                    _top = breakdown_data[0]
                    _share = (_top["current"] / total_current * 100) if total_current else 0
                    if _share >= 50:
                        _lvl = "more than half"
                    elif _share >= 30:
                        _lvl = "about a third"
                    elif _share >= 15:
                        _lvl = "a notable share"
                    else:
                        _lvl = "the largest share"
                    observations.append(
                        f"{_top['category']} leads {metric_label.lower()} with {_lvl} of the total.")
                    _tpct = _top.get("change_pct")
                    if _tpct is not None:
                        _td = "up" if _tpct >= 0 else "down"
                        bullets.append(
                            f"{_top['category']} leads at {_human_total(_top['current'])} "
                            f"({_share:.0f}% of total), {_td} {abs(_tpct):.1f}% vs {_prev_vlabel}.")
                    else:
                        bullets.append(
                            f"Top segment is {_top['category']} — roughly {_share:.0f} "
                            f"percent of the total.")
                    _gainers = sorted(
                        [d for d in breakdown_data if (d.get("change_pct") or 0) > 0],
                        key=lambda d: d["change_pct"], reverse=True)
                    _losers = sorted(
                        [d for d in breakdown_data if (d.get("change_pct") or 0) < 0],
                        key=lambda d: d["change_pct"])
                    if _gainers and _gainers[0]["category"] != _top["category"]:
                        _g = _gainers[0]
                        bullets.append(
                            f"{_g['category']} grew fastest ({_g['change_pct']:.1f}%), "
                            f"outpacing the field.")
                    if _losers:
                        _l = _losers[0]
                        bullets.append(
                            f"{_l['category']} softened {_l['change_pct']:.1f}% — worth checking "
                            f"whether that dip is expected.")
                if len(trend_points) >= 6:
                    _half = len(trend_points) // 2
                    _h1 = sum((p.get("current") or 0) for p in trend_points[:_half])
                    _h2 = sum((p.get("current") or 0) for p in trend_points[_half:])
                    if _h2 > _h1 * 1.02:
                        bullets.append("Momentum picked up in the second half of the window.")
                    elif _h1 > _h2 * 1.02:
                        bullets.append("Momentum softened in the second half of the window.")
                    else:
                        bullets.append("Performance held broadly steady across the window.")
                basis = (
                    f"Based on {_raw_count} records from {primary_table} "
                    f"({_cur_start.isoformat()} to {_cur_end.isoformat()} vs "
                    f"{_prev_start.isoformat()} to {_prev_end.isoformat()}; {date_label}). "
                    "Category and monthly figures are summed from your data; "
                    "percentages compare the two windows.")
            else:
                _period_label = date_label
                if total_current:
                    observations.append(
                        f"{metric_label} totals {_human_total(total_current)} across "
                        f"{_raw_count} records.")
                if breakdown_data:
                    _top = breakdown_data[0]
                    _share = (_top["current"] / total_current * 100) if total_current else 0
                    observations.append(
                        f"{_top['category']} leads {metric_label.lower()} with "
                        f"roughly {_share:.0f} percent of the total.")
                    bullets.append(
                        f"Top segment is {_top['category']} — roughly {_share:.0f} "
                        f"percent of the total.")
                basis = (
                    f"No date column was found in {primary_table}, so this is a category "
                    f"snapshot without a previous-period comparison. Based on {_raw_count} "
                    f"records. Figures below are rounded to a human scale.")
            if secondary:
                bullets.append(
                    f"Secondary view by {display_name(str(secondary), str(secondary))} is noted "
                    f"— refine filters to drill deeper.")
            summary = " ".join(observations) if observations else (
                f"{metric_label} was examined across {len(rows_out)} groups. "
                f"No dominant pattern stands out.")

            _change_all = (_r2(total_current - total_previous)
                           if total_previous is not None else None)
            _groups = len(_cells) if _has_time else len(_cat_only)
            chart_data = {
                "type": "bar" if breakdown_col else "line",
                "title": _trend_title,
                "data": trend_points,
                "series": [
                    {"name": "Current Period",
                     "data": [{"x": p["label"], "y": p["current"]} for p in trend_points]},
                    {"name": "Previous Period",
                     "data": [{"x": p["label"], "y": p.get("previous")} for p in trend_points]},
                ],
            }

            _result = {
                "ok": True,
                "metric": metric_label,
                "metric_column": metric_col,
                "breakdown": breakdown_data,
                "breakdown_label": breakdown_label,
                "breakdown_column": breakdown_col,
                "breakdown_data": breakdown_data,
                "breakdown_rows": rows_out,
                "breakdown_title": f"{breakdown_label} Breakdown" if breakdown_label else "Breakdown",
                "rows": rows_out,
                "table": rows_out,
                "time_series": time_series,
                "time_grain": grain_label,
                "grain": grain_label,
                "date_column": date_col,
                "comparison": compare,
                "total": total_current,
                "total_display": _human_total(total_current),
                "previous_total": total_previous,
                "previous_display": (_human_total(total_previous)
                                     if total_previous is not None else None),
                "change": _change_all,
                "change_pct": _pct(total_current, total_previous),
                "trend": {"title": _trend_title,
                          "subtitle": f"{_period_label} · {_groups} groups",
                          "points": trend_points},
                "series": trend_points,
                "points": trend_points,
                "chart": chart_data,
                "title": _trend_title,
                "subtitle": f"{_period_label} · {_groups} groups",
                "summary": summary,
                "observations": observations,
                "observation": {"summary": summary, "bullets": bullets, "basis_note": basis},
                "aria_observation": {"summary": summary, "bullets": bullets, "basis_note": basis},
                "basis_note": basis,
                "row_count": _groups,
                "analysis_period": _period_label,
                "breakdown_scope": _scope,
            }
            # ---- independent validation (fresh SQL, separate connection) --
            # Fail-closed: any failed check flips ok -> False with
            # code VALIDATION_FAILED (numbers kept for debugging, never
            # shown as verified). Validator never raises.
            try:
                from services.intelligence_validation import (
                    validate_explore as _validate_explore)
                if _has_time and _cur_start is not None and _qd:
                    _wc_sql = _where([f"{_qd} >= :_vc_s",
                                      f"{_qd} <= :_vc_e"])
                    _wp_sql = _where([f"{_qd} >= :_vp_s",
                                      f"{_qd} <= :_vp_e"])
                    _v_params = dict(_params)
                    _v_params["_vc_s"] = _cur_start.isoformat()
                    _v_params["_vc_e"] = _cur_end.isoformat()
                    _v_params["_vp_s"] = _prev_start.isoformat()
                    _v_params["_vp_e"] = _prev_end.isoformat()
                else:
                    _wc_sql = _where()
                    _wp_sql = _where()
                    _v_params = dict(_params)
                _vr = _validate_explore(
                    engine, dialect, _qm, _qt, _wc_sql, _wp_sql,
                    _v_params, _result,
                    date_col=date_col, qd=_qd,
                    cur_start=_cur_start, cur_end=_cur_end,
                    prev_start=_prev_start, prev_end=_prev_end,
                    do_compare=do_compare,
                    agg=("COUNT" if _is_count else "SUM"),
                    where_base_sql=_where(),
                    date_storage=(_date_storage if _has_time else "native"))
                if _vr.get("passed"):
                    _result["validation"] = {
                        "passed": True,
                        "checks": [c.get("name") for c in _vr.get("checks", [])
                                   if c.get("passed")],
                    }
                else:
                    _failed = [c.get("name") for c in _vr.get("checks", [])
                               if not c.get("passed")]
                    _details = {c.get("name"): c.get("detail")
                                for c in _vr.get("checks", [])}
                    log.error("intelligence validation failed: %s",
                              "; ".join(f"{k}={v}" for k, v in _details.items()))
                    _result["validation"] = {
                        "passed": False,
                        "failed": _failed,
                        "checks": [c.get("name") for c in _vr.get("checks", [])
                                   if c.get("passed")],
                    }
                    _result["ok"] = False
                    _result["code"] = "VALIDATION_FAILED"
                    _result["detail"] = (
                        "Validation failed for: " + ", ".join(_failed) +
                        ". Numbers withheld as unverified.")
            except Exception as _vexc:
                log.error("intelligence validator error: %s", _vexc)
                _result["validation"] = {"passed": False, "failed": ["validator_error"],
                                         "checks": []}
                _result["ok"] = False
                _result["code"] = "VALIDATION_FAILED"
                _result["detail"] = (
                    f"Validation error (validator_error): {_vexc}. "
                    "Numbers withheld as unverified.")
            return _result
    except _NoJoinPath as njp:
        return njp.payload
    except Exception as exc:
        log.exception("Explore failed")
        return {"ok": False, "detail": "ARIA couldn't complete this analysis from your data. Try a different metric or range.", "code": "ANALYSIS_FAILED"}


# ---------------------------------------------------------------------------
# POST /api/intelligence/hypotheses
# ---------------------------------------------------------------------------

def _hypothesis_items(session: dict) -> list:
    """Build full hypothesis cards with human-worded summaries (no raw dumps)."""
    mapping = _load_schema_mapping(session)
    if not mapping:
        return []
    roles = _classify_columns(mapping)
    items: list[dict] = []
    _h_db_uri = session.get("db_uri") if isinstance(session, dict) else None
    for tname, col_roles in roles.items():
        measures = [c for c, r in col_roles.items() if r in ("BUSINESS_MEASURE", "MEASURE", "DERIVED_METRIC")]
        dims = [c for c, r in col_roles.items() if r in ("DIMENSION", "CATEGORY", "TEXT")]
        dates = [c for c, r in col_roles.items() if r == "DATE"]
        if (not dims or not dates) and _h_db_uri:
            # Same role-poor gap as meta (e.g. walmart: TEXT date + store
            # held as IDENTIFIERs): value-verified extras so trend/segment
            # hypotheses exist instead of an empty Hypotheses tab.
            try:
                _he = _probe_table_extras(
                    mapping, tname, col_roles or {}, _h_db_uri)
                if not dims:
                    dims = [c for c in (_he.get("low_card_dims") or [])
                            if c in (col_roles or {})][:3]
                if not dates:
                    dates = [c for c in (_he.get("date_cols") or [])
                             if c in (col_roles or {})][:1]
            except Exception:
                pass
        for m in measures[:5]:
            ml = display_name(m, m)
            for d in dims[:3]:
                dl = display_name(d, d)
                hid = hashlib.sha256(f"{tname}:{m}:{d}".encode()).hexdigest()[:12]
                items.append({
                    "id": hid, "type": "segment", "metric": m, "dimension": d,
                    "table": tname, "metric_label": ml, "dimension_label": dl,
                    "title": f"Why does {ml.lower()} vary by {dl.lower()}?",
                    "question": f"Why might {ml} have changed?",
                    "period": "All data",
                    "verdict": "Ready to investigate",
                    "drivers": [{"driver": dl, "type": "Segment", "association": "Segment difference", "evidence_strength": "Moderate"}],
                    "summary": f"Performance in {ml.lower()} differs across {dl.lower()} groups — ARIA can test which segments stand apart.",
                    "description": f"Segment comparison of {ml} across {dl} in {display_name(tname, tname)}.",
                    "evidence_strength": "Moderate", "strength": "Moderate", "status": "Ready to investigate",
                    "observed_pattern": {
                        "headline": f"{ml} is unevenly distributed across {dl.lower()}",
                        "detail": f"Some {dl.lower()} groups contribute far more {ml.lower()} than others. Open this hypothesis to see the breakdown.",
                    },
                    "key_takeaways": [
                        f"{ml} concentration differs meaningfully across {dl.lower()}.",
                        "Focus follow-up on the top and bottom segments first.",
                    ],
                    "takeaways": [f"{ml} concentration differs across {dl.lower()}."],
                    "associated_factors": [{"factor": dl, "name": dl, "type": "Segment", "kind": "Segment",
                                            "association": f"Groups differ in {ml.lower()}", "relationship": "Segment difference",
                                            "evidence_strength": "Moderate", "strength": "Moderate"}],
                    "factors": [{"factor": dl, "type": "Segment", "association": "Segment difference", "evidence_strength": "Moderate"}],
                    "evidence": [f"Grouped totals of {ml.lower()} by {dl.lower()} show a clear spread between leaders and laggards."],
                    "evidence_points": [f"Grouped totals of {ml.lower()} by {dl.lower()} show spread."],
                    "related": [], "next_steps": [f"Open Explore with {ml} broken down by {dl} to see the leaders.", "Test a scenario on the top segment."],
                    "context": {"metric": ml, "breakdown": dl, "dimension": dl, "table": display_name(tname, tname), "period": "All data"},
                })
        for m in measures[:3]:
            ml = display_name(m, m)
            for dt in dates[:1]:
                hid = hashlib.sha256(f"{tname}:{m}:{dt}:trend".encode()).hexdigest()[:12]
                items.append({
                    "id": hid, "type": "trend", "metric": m, "time": dt,
                    "table": tname, "metric_label": ml,
                    "title": f"Is {ml.lower()} trending up or down over time?",
                    "question": f"Why might {ml} have changed?",
                    "period": "All data",
                    "verdict": "Ready to investigate",
                    "drivers": [],
                    "summary": f"ARIA can trace {ml.lower()} over time to describe momentum in plain language.",
                    "description": f"Trend of {ml} over {display_name(dt, dt)}.",
                    "evidence_strength": "Moderate", "strength": "Moderate", "status": "Ready to investigate",
                    "observed_pattern": {"headline": f"{ml} moves over time", "detail": f"Run Explore on {ml} to see whether momentum is building or softening."},
                    "key_takeaways": [f"Direction of {ml.lower()} over time is testable from your history."],
                    "takeaways": [f"Direction of {ml.lower()} is testable."],
                    "associated_factors": [],
                    "evidence": [f"Time-ordered totals of {ml.lower()} reveal the current direction."],
                    "related": [], "next_steps": [f"Open Explore with {ml} over time."],
                    "context": {"metric": ml, "breakdown": "Time", "table": display_name(tname, tname), "period": "All data"},
                })
    return items


def generate_hypotheses(session: dict, request: dict) -> dict:
    """Generate hypothesis candidates — supports search/sort/page (server-side)."""
    request = request or {}
    items = _hypothesis_items(session)
    if not items and not _load_schema_mapping(session):
        return {"ok": False, "detail": "No data connected.", "code": "NO_DATA",
                "candidates": [], "items": [], "data": [],
                "total_candidates": 0, "total": 0, "count": 0,
                "reason": NO_SCHEMA_REASON, "hint": NO_SCHEMA_HINT}
    search = str(request.get("search") or "").strip().lower()
    if search:
        items = [h for h in items if search in f"{h.get('title','')} {h.get('summary','')} {h.get('metric_label','')}".lower()]
    sort = str(request.get("sort") or "recent").lower()
    if sort == "az":
        items = sorted(items, key=lambda h: str(h.get("title", "")))
    elif sort in ("strongest", "strong"):
        order = {"Strong": 0, "Moderate": 1}
        items = sorted(items, key=lambda h: order.get(str(h.get("evidence_strength", "Moderate")), 2))
    try:
        page = max(1, int(request.get("page") or 1))
    except Exception:
        page = 1
    try:
        page_size = max(1, min(50, int(request.get("page_size") or request.get("pageSize") or 8)))
    except Exception:
        page_size = 8
    total = len(items)
    start = (page - 1) * page_size
    sliced = items[start:start + page_size]
    return {
        "ok": True,
        "candidates": items,
        "total_candidates": total,
        "items": sliced,
        "data": sliced,
        "total": total,
        "count": total,
        "page": page,
        "page_size": page_size,
    }


def get_hypothesis_detail(session: dict, hypothesis_id: str) -> dict | None:
    """Single hypothesis with full human-worded detail; enriches with live spread.

    Computes a real observed pattern from the database (top/bottom segment
    totals with share + mini-series for segment hypotheses; first-half vs
    second-half totals with direction for trend hypotheses). All numbers are
    rounded to a human scale — never fabricated.
    """
    items = _hypothesis_items(session)
    found = next((h for h in items if str(h.get("id")) == str(hypothesis_id)), None)
    if not found:
        return None
    found = {**found}
    # Contract aliases (panel 3): hypotheses[] {id,title,metric,period,verdict,evidence,drivers[]}
    found.setdefault("period", (found.get("context") or {}).get("period") or "All data")
    found.setdefault("verdict", found.get("status") or "Ready to investigate")
    if not found.get("drivers"):
        found["drivers"] = list(found.get("associated_factors") or found.get("factors") or [])
    # Related analyses: other hypotheses over the same metric (real relation).
    try:
        same = [h for h in items
                if str(h.get("id")) != str(found.get("id"))
                and str(h.get("metric") or "") == str(found.get("metric") or "")]
        found["related"] = [{"id": h.get("id"), "title": h.get("title"),
                             "question": h.get("question")} for h in same[:3]]
    except Exception:
        found.setdefault("related", [])
    # Try to enrich segment hypotheses with a live top/bottom sentence
    try:
        if found.get("type") == "segment":
            db_uri = session.get("db_uri")
            if db_uri:
                from sqlalchemy import create_engine, text as sql_text
                engine = _pooled_engine(db_uri)
                with engine.connect() as conn:
                    t, m, d = found["table"], found["metric"], found["dimension"]
                    qm = "`" if db_uri and "mysql" in db_uri else '"'
                    # TEXT-stored measures (currency strings) need the shared
                    # numeric coercion — raw SUM(text) raises and the card
                    # silently loses its enrichment.
                    _hd = _dialect_of(db_uri)
                    try:
                        _hmap = _load_schema_mapping(session)
                        _hdt = _col_data_type(_hmap, t, m)
                        _hexpr = _nan_safe_sql(
                            _numeric_metric_sql(f"{qm}{m}{qm}", _hdt, _hd),
                            _hdt, _hd)
                        if _hd == "sqlite":
                            try:
                                from services.intelligence_validation import (
                                    ensure_sqlite_helpers as _esh)
                                _esh(engine)
                            except Exception:
                                pass
                    except Exception:
                        _hexpr = f"{qm}{m}{qm}"
                    sql = f"SELECT {qm}{d}{qm} AS c, SUM({_hexpr}) AS v FROM {qm}{t}{qm} GROUP BY {qm}{d}{qm} ORDER BY v DESC LIMIT 8"
                    _r = conn.execute(sql_text(sql))
                    _c = list(_r.keys())
                    rows = _r.fetchall()
                    if rows and len(rows) >= 2:
                        pairs = [(str(dict(zip(_c, r)).get("c") or "Unknown"),
                                  float(dict(zip(_c, r)).get("v") or 0)) for r in rows]
                        total = sum(v for _, v in pairs) or 1.0
                        top_c, top_v = pairs[0]
                        bot_c, bot_v = pairs[-1]
                        share = top_v / total * 100 if total else 0
                        gap = (top_v - bot_v) / top_v * 100 if top_v else 0
                        ml = found.get("metric_label") or display_name(str(m), str(m))
                        dl = found.get("dimension_label") or display_name(str(d), str(d))
                        found["key_takeaways"] = [
                            f"{top_c} leads while {bot_c} trails — start there.",
                            *found.get("key_takeaways", [])[:2],
                        ]
                        series = [{"label": c, "value": v} for c, v in pairs]
                        found["observed_pattern"] = {
                            "headline": f"{ml} is unevenly distributed across {dl.lower()}",
                            "detail": (f"{top_c} holds {share:.0f} percent of {ml.lower()} "
                                       f"({_human_total(top_v)} of {_human_total(total)}); "
                                       f"{bot_c} trails at {_human_total(bot_v)} — a {gap:.0f} percent gap top to bottom."),
                            "change_pct": round(gap, 1),
                            "direction": "spread",
                            "before": bot_v,
                            "after": top_v,
                            "before_display": _human_total(bot_v),
                            "after_display": _human_total(top_v),
                            "series": series,
                            "points": series,
                        }
                        facs = []
                        for c, v in pairs[:5]:
                            sh = v / total * 100 if total else 0
                            facs.append({
                                "factor": f"{dl}: {c}",
                                "name": f"{dl}: {c}",
                                "type": "Dimension",
                                "kind": "Dimension",
                                "association": f"{'Higher' if v >= total / len(pairs) else 'Lower'} total ({sh:.0f}% share)",
                                "relationship": "Segment difference",
                                "evidence_strength": "Strong" if sh >= 25 else "Moderate",
                                "strength": "Strong" if sh >= 25 else "Moderate",
                                "explore": {"metric": m, "breakdown": d},
                                "whatif": {"variable": m},
                            })
                        found["associated_factors"] = facs
                        found["factors"] = facs
                        found["drivers"] = facs
                        found["evidence"] = [
                            f"Grouped totals of {ml.lower()} by {dl.lower()} across {len(pairs)} groups show a {gap:.0f} percent gap between {top_c} and {bot_c}.",
                            *found.get("evidence", [])[:2],
                        ]
        elif found.get("type") == "trend":
            db_uri = session.get("db_uri")
            # Resolve the real DATE column for this table.
            _hmap2 = _load_schema_mapping(session)
            roles = _classify_columns(_hmap2)
            tcols = roles.get(found.get("table") or "", {}) or {}
            dt = found.get("time") or next((c for c, r in tcols.items() if r == "DATE"), None)
            if db_uri and dt:
                from sqlalchemy import create_engine, text as sql_text
                engine = _pooled_engine(db_uri)
                with engine.connect() as conn:
                    t, m = found["table"], found["metric"]
                    qm = "`" if db_uri and "mysql" in db_uri else '"'
                    _hd2 = _dialect_of(db_uri)
                    try:
                        _mdt = _col_data_type(_hmap2, t, m)
                        _mexpr = _nan_safe_sql(
                            _numeric_metric_sql(f"{qm}{m}{qm}", _mdt, _hd2),
                            _mdt, _hd2)
                        if _hd2 == "sqlite":
                            try:
                                from services.intelligence_validation import (
                                    ensure_sqlite_helpers as _esh2)
                                _esh2(engine)
                            except Exception:
                                pass
                    except Exception:
                        _mdt, _mexpr = "", f"{qm}{m}{qm}"
                    _native_dt = _is_native_date_dtype(
                        _col_data_type(_hmap2, t, dt))
                    pts: list = []
                    if _native_dt and not _is_text_dtype(_mdt):
                        sql = f"SELECT {qm}{dt}{qm} AS p, SUM({_mexpr}) AS v FROM {qm}{t}{qm} GROUP BY {qm}{dt}{qm} ORDER BY {qm}{dt}{qm} ASC LIMIT 24"
                        _r = conn.execute(sql_text(sql))
                        _c = list(_r.keys())
                        rows = _r.fetchall()
                        if rows:
                            pts = [(str(dict(zip(_c, r)).get("p") or ""),
                                    float(dict(zip(_c, r)).get("v") or 0)) for r in rows]
                    else:
                        # TEXT/odd-stored dates (or TEXT measures): SQL
                        # ORDER BY over stored spellings is meaningless
                        # (DD-MM-YYYY scrambles), so bucket in SQL but sort
                        # by the shared tolerant parser in Python. Earliest
                        # 24 buckets mirrors the native LIMIT 24 ASC window.
                        _sql2 = f"SELECT {qm}{dt}{qm} AS p, SUM({_mexpr}) AS v FROM {qm}{t}{qm} GROUP BY {qm}{dt}{qm}"
                        _r = conn.execute(sql_text(_sql2))
                        _c = list(_r.keys())
                        _pairs = []
                        for _rw in _r.fetchall():
                            _dd = dict(zip(_c, _rw))
                            _pd = _shared_to_date(_dd.get("p"))
                            if _pd is None:
                                continue
                            try:
                                _vv = float(_dd.get("v") or 0)
                            except Exception:
                                continue
                            _pairs.append((_pd, str(_dd.get("p") or ""), _vv))
                        _pairs.sort(key=lambda x: x[0])
                        pts = [(lab, v) for _, lab, v in _pairs[:24]]
                    if pts and len(pts) >= 2:
                        half = max(1, len(pts) // 2)
                        before = sum(v for _, v in pts[:half]) or 0
                        after = sum(v for _, v in pts[half:]) or 0
                        pct = ((after - before) / before * 100) if before else 0
                        direction = "declined" if pct < 0 else "increased" if pct > 0 else "held steady"
                        ml = found.get("metric_label") or display_name(str(m), str(m))
                        series = [{"label": p, "value": v} for p, v in pts]
                        found["observed_pattern"] = {
                            "headline": f"{ml} {direction} by {abs(pct):.1f}%",
                            "detail": f"{_human_total(before)} → {_human_total(after)} across the window.",
                            "change_pct": round(pct, 1),
                            "direction": direction,
                            "before": before,
                            "after": after,
                            "before_display": _human_total(before),
                            "after_display": _human_total(after),
                            "series": series,
                            "points": series,
                        }
                        found["key_takeaways"] = [
                            f"{ml} {direction} by about {abs(pct):.0f} percent ({_human_total(before)} → {_human_total(after)}).",
                            *found.get("key_takeaways", [])[:2],
                        ]
    except Exception:
        pass
    return found


# ---------------------------------------------------------------------------
# POST /api/intelligence/what-if/run
# ---------------------------------------------------------------------------

def run_whatif(session: dict, request: dict) -> dict:
    """What-If scenario: deterministic % adjustment with backend range.

    Returns adjusted projection + assumptions. Never fabricates ROI.
    """
    variable = request.get("variable") or request.get("metric") or request.get("column")
    raw_ct = str(request.get("change_type") or request.get("changeType") or "increase")
    change_type = "increase" if "decr" not in raw_ct.lower() and "reduc" not in raw_ct.lower() and raw_ct.lower() not in ("decrease", "down", "-") else ("decrease" if "incr" not in raw_ct.lower() else raw_ct.lower())
    if raw_ct.lower() in ("increase", "decrease"):
        change_type = raw_ct.lower()
    elif "increase" in raw_ct.lower():
        change_type = "increase"
    elif "decrease" in raw_ct.lower():
        change_type = "decrease"
    try:
        change_value = float(request.get("change_value", request.get("changeValue", 10)))
    except Exception:
        change_value = 10
    apply_to = request.get("apply_to") or request.get("applyTo") or request.get("scope") or "All data"

    if not variable:
        return {"ok": False, "detail": "Variable is required.", "code": "INVALID_REQUEST"}

    mapping = _load_schema_mapping(session)
    if not mapping:
        return {"ok": False, "detail": "No data connected.", "code": "NO_DATA"}

    # Get current value from the data
    db_uri = session.get("db_uri")
    if not db_uri:
        return {"ok": False, "detail": "No database connection.", "code": "NO_DATA"}

    try:
        from sqlalchemy import create_engine, text as sql_text
        engine = _pooled_engine(db_uri)

        # Find which table has this column (accept label or column)
        roles = _classify_columns(mapping)
        resolved_var = _resolve_column(roles, variable) or variable
        variable = resolved_var
        table = None
        for tname, col_roles in roles.items():
            if variable in col_roles:
                table = tname
                break
        if not table:
            return {"ok": False, "detail": f"Variable '{variable}' not found.", "code": "COLUMN_NOT_FOUND"}

        # Numeric-safe aggregation (mirrors run_explore): TEXT-stored
        # measures (currency symbols, commas, prefixes) get a
        # dialect-appropriate cast instead of SUM(text); float NaN maps to
        # NULL so one NaN never poisons the total. Shape-based, never
        # per-database.
        _wdialect = ("mysql" if "mysql" in (db_uri or "")
                     else "postgresql" if ("postgresql" in (db_uri or "")
                                           or "psycopg" in (db_uri or ""))
                     else "sqlite")
        if _wdialect == "sqlite":
            try:
                from services.intelligence_validation import (
                    ensure_sqlite_helpers as _ensure_wlite)
                _ensure_wlite(engine)
            except Exception:
                pass

        def _wsum(col: str, tbl: str) -> str:
            """SUM(...) over ``tbl.col``, numeric-safe for TEXT storage."""
            _q = '"' + str(col).replace('"', '""') + '"'
            _qt = '"' + str(tbl).replace('"', '""') + '"'
            if _wdialect == "mysql":
                _q = '`' + str(col).replace('`', '``') + '`'
                _qt = '`' + str(tbl).replace('`', '``') + '`'
            _dtype = _col_data_type(mapping, tbl, col)
            _expr = _nan_safe_sql(
                _numeric_metric_sql(f"{_qt}.{_q}", _dtype, _wdialect),
                _dtype, _wdialect)
            return f"SUM({_expr})", _qt

        with engine.connect() as conn:
            if variable == COUNT_METRIC:
                _qtbl_only = ('`' + str(table).replace('`', '``') + '`'
                              if _wdialect == "mysql"
                              else '"' + str(table).replace('"', '""') + '"')
                _res2 = conn.execute(
                    sql_text(f"SELECT COUNT(*) as _count FROM {_qtbl_only}"))
                _cols2 = list(_res2.keys())
                row = _res2.fetchone()
                row_count = int(dict(zip(_cols2, row)).get("_count") or 0) if row else 0
                current_total = float(row_count)
            else:
                _wexpr, _wqt = _wsum(variable, table)
                sql = f"SELECT {_wexpr} as _total, COUNT(*) as _count FROM {_wqt}"
                _res2 = conn.execute(sql_text(sql))
                _cols2 = list(_res2.keys())
                row = _res2.fetchone()
                if row:
                    d = dict(zip(_cols2, row))
                    try:
                        current_total = float(d.get("_total") or 0)
                    except (TypeError, ValueError):
                        current_total = 0.0
                    if math.isnan(current_total) or math.isinf(current_total):
                        current_total = 0.0
                    row_count = int(d.get("_count") or 0)
                else:
                    current_total = 0
                    row_count = 0

            # Panel-5 contract: revenue / orders / aov + monthly baseline-vs-scenario,
            # all computed from real totals (scenario = total*(1+pct) for
            # revenue/aov, orders flat) — never mocked.
            factor = change_value / 100.0
            sign = 1 if change_type == "increase" else -1
            scope_txt = apply_to if isinstance(apply_to, str) and apply_to else "All data"
            # Detect sibling measures in the same table for revenue/orders context.
            rev_col, date_col = None, None
            try:
                t_roles = roles.get(table) or {}
                for cname, role in t_roles.items():
                    if role in ("BUSINESS_MEASURE", "MEASURE", "DERIVED_METRIC") and re.search(
                            r"revenue|sales|total|amount|payment|profit|income", str(cname), re.I):
                        rev_col = cname
                        break
                for cname, role in t_roles.items():
                    if role == "DATE":
                        date_col = cname
                        break
            except Exception:
                pass
            revenue_total = current_total
            if rev_col and rev_col != variable:
                try:
                    with engine.connect() as _c2:
                        _rexpr, _rqt = _wsum(rev_col, table)
                        _r2 = _c2.execute(
                            sql_text(f"SELECT {_rexpr} as _rt FROM {_rqt}"))
                        _k2 = list(_r2.keys())
                        _row2 = _r2.fetchone()
                        if _row2:
                            try:
                                revenue_total = float(dict(zip(_k2, _row2)).get("_rt") or 0)
                            except (TypeError, ValueError):
                                revenue_total = current_total
                            if math.isnan(revenue_total) or math.isinf(revenue_total):
                                revenue_total = current_total
                except Exception:
                    revenue_total = current_total
            orders_total = float(row_count or 0)
            aov_total = (revenue_total / orders_total) if orders_total else 0.0
            projected_revenue = revenue_total * (1 + sign * factor)
            projected_aov = aov_total * (1 + sign * factor)
            projected_orders = orders_total  # orders flat by design
            # Monthly baseline series (first 6 periods) for the comparison bar.
            monthly: list[dict] = []
            try:
                if date_col:
                    with engine.connect() as _c3:
                        _qm = '`' if _wdialect == "mysql" else '"'
                        _mcol = rev_col or variable
                        if _mcol == COUNT_METRIC:
                            _mexpr = "COUNT(*)"
                        else:
                            # Table-qualified expr is valid in this
                            # single-table FROM (no alias), so keep it as-is.
                            _mexpr, _ = _wsum(_mcol, table)
                        _msql = (f"SELECT {_qm}{date_col}{_qm} AS p, {_mexpr} AS v "
                                 f"FROM {_qm}{table}{_qm} GROUP BY {_qm}{date_col}{_qm} "
                                 f"ORDER BY {_qm}{date_col}{_qm} ASC LIMIT 6")
                        _r3 = _c3.execute(sql_text(_msql))
                        _k3 = list(_r3.keys())
                        for _rr in _r3.fetchall():
                            _dd = dict(zip(_k3, _rr))
                            _lbl = str(_dd.get("p") or "")[:7] or "Period"
                            _bv = float(_dd.get("v") or 0)
                            monthly.append({"label": _lbl, "baseline": _bv,
                                            "scenario": _bv * (1 + sign * factor)})
            except Exception:
                monthly = []
            if not monthly:
                monthly = [{"label": "All data", "baseline": revenue_total,
                            "scenario": projected_revenue}]

        # Calculate projected impact (variable-level, kept for compatibility)
        if change_type == "increase":
            projected = current_total * (1 + factor)
        else:
            projected = current_total * (1 - factor)
        delta = projected - current_total

        direction = "growing" if change_type == "increase" else "reducing"
        vlabel = display_name(variable, variable)
        if delta >= 0:
            impact_word = "lift"
        else:
            impact_word = "reduction"
        rev_delta = projected_revenue - revenue_total
        rev_pct = (sign * factor * 100)
        summary = (
            f"{direction.capitalize()} {vlabel.lower()} by about {abs(change_value):g} percent "
            f"would move the total from {_human_total(current_total)} to {_human_total(projected)} — "
            f"a {impact_word} of {_human_total(abs(delta))} "
            f"(applied to {scope_txt})."
        )
        aov_note = f"+{abs(change_value):g}%" if sign > 0 else f"-{abs(change_value):g}%"
        _wresult = {
            "ok": True,
            "variable": variable,
            "variable_label": vlabel,
            "current_value": current_total,
            "current_display": _human_total(current_total),
            "change_type": change_type,
            "change_value": change_value,
            "projected_value": projected,
            "projected_display": _human_total(projected),
            "delta": delta,
            "delta_display": _human_total(abs(delta)),
            "delta_pct": round(factor * 100, 1) if change_type == "increase" else round(-factor * 100, 1),
            "row_count": row_count,
            "apply_to": apply_to,
            "summary": summary,
            "scenario_summary": summary,
            # Panel-5 contract: revenue / orders / aov + monthly baseline-vs-scenario
            "revenue": {"current": revenue_total, "projected": projected_revenue,
                        "delta": rev_delta, "delta_pct": round(rev_pct, 1),
                        "display": _human_total(projected_revenue),
                        "current_display": _human_total(revenue_total)},
            "orders": {"current": orders_total, "projected": projected_orders,
                       "delta": 0, "delta_pct": 0,
                       "display": _human_total(projected_orders),
                       "current_display": _human_total(orders_total)},
            "aov": {"current": aov_total, "projected": projected_aov,
                    "delta": projected_aov - aov_total,
                    "delta_pct": round(rev_pct, 1),
                    "display": _human_total(projected_aov),
                    "current_display": _human_total(aov_total)},
            "monthly": monthly,
            "projected_impact": [
                {"label": "Revenue", "metric": "Revenue", "value": projected_revenue, "display": _human_total(projected_revenue), "change_note": f"{rev_pct:+.1f}% vs baseline", "delta": _human_total(abs(rev_delta))},
                {"label": "Total Orders", "metric": "Total Orders", "value": projected_orders, "display": _human_total(projected_orders), "change_note": "No change", "delta": _human_total(0)},
                {"label": "Avg Order Value", "metric": "Avg Order Value", "value": projected_aov, "display": _human_total(projected_aov), "change_note": aov_note, "delta": _human_total(abs(projected_aov - aov_total))},
            ],
            "impacts": [
                {"label": "Current total", "value": current_total, "display": _human_total(current_total)},
                {"label": "Projected total", "value": projected, "display": _human_total(projected)},
            ],
            "kpis": [
                {"label": "Current total", "value": current_total, "display": _human_total(current_total)},
                {"label": "Projected total", "value": projected, "display": _human_total(projected)},
            ],
            "revenue_comparison": monthly,
            "comparison_series": monthly,
            "comparison": monthly,
            "comparison_title": f"Revenue Comparison — baseline vs scenario",
            "assumptions": [
                f"Starting from {_human_total(revenue_total)} revenue across about {row_count} records.",
                f"Models a {change_value:g} percent {change_type} applied to {scope_txt}.",
                "Orders held flat; revenue and average order value scale with the change.",
                "Assumes the change scales evenly — real-world effects are usually less tidy.",
            ],
            "supported_range": {"min": -50, "max": 100, "step": 1},
            "chart": {
                "type": "comparison",
                "title": f"Impact of {change_value:g}% {change_type} on {vlabel}",
                "data": monthly,
                "series": [
                    {"name": "Baseline", "value": revenue_total},
                    {"name": "Scenario", "value": projected_revenue},
                ],
            },
        }
        # ---- what-if validation (warn-only: estimate, keep result) -------
        try:
            from services.intelligence_validation import (
                validate_whatif as _validate_whatif)
            _wv = _validate_whatif(_wresult)
            _w_failed = [c.get("name") for c in _wv.get("checks", [])
                         if not c.get("passed")]
            _wresult["validation"] = {
                "passed": bool(_wv.get("passed")),
                "checks": [c.get("name") for c in _wv.get("checks", [])
                           if c.get("passed")],
                "failed": _w_failed,
            }
            if _w_failed:
                log.warning("what-if validation warnings: %s", _w_failed)
        except Exception as _wvexc:
            log.warning("what-if validator error: %s", _wvexc)
            _wresult["validation"] = {"passed": False, "checks": [],
                                      "failed": ["validator_error"]}
        return _wresult
    except Exception as exc:
        log.exception("What-If failed")
        return {"ok": False, "detail": "ARIA couldn't model this scenario from your data. Try a smaller change.", "code": "SCENARIO_FAILED"}


# ---------------------------------------------------------------------------
# POST /api/intelligence/what-if/meta (slider range)
# ---------------------------------------------------------------------------

def get_whatif_meta(session: dict) -> dict:
    """Return available variables and their supported ranges (frontend contract)."""
    mapping = _load_schema_mapping(session)
    if not mapping:
        return {"ok": True, "variables": [], "change_types": ["Increase by", "Decrease by"], "changeTypes": ["Increase by", "Decrease by"],
                "apply_targets": ["All data"], "applyTargets": ["All data"], "ranges": {}, "range": None,
                "reason": NO_SCHEMA_REASON, "hint": NO_SCHEMA_HINT}
    roles = _classify_columns(mapping)
    variables = []
    ranges: dict[str, dict] = {}
    for tname, col_roles in roles.items():
        for cname, role in col_roles.items():
            if role in ("BUSINESS_MEASURE", "MEASURE", "DERIVED_METRIC"):
                label = display_name(cname, cname)
                rng = {"min": -50, "max": 100, "step": 1}
                variables.append({
                    "column": cname, "name": label, "label": label,
                    "table": tname, "role": role, "supported_range": rng, "range": rng,
                })
                ranges[cname] = rng
                ranges[label] = rng
    cts = ["Increase by", "Decrease by"]
    targets = ["All data", "Top segment", "Specific category", "Specific region"]
    return {"ok": True, "variables": variables,
            "change_types": cts, "changeTypes": cts,
            "apply_targets": targets, "applyTargets": targets, "scopes": targets,
            "ranges": ranges, "range": {"min": -50, "max": 100, "step": 1}}


# ---------------------------------------------------------------------------
# Goals CRUD
# ---------------------------------------------------------------------------

def list_goals(user_id: int, dataset_id: str, status_filter: str = "active") -> list:
    """List goals filtered by dataset and status."""
    items = _load_user_items(user_id, "goals")
    filtered = [g for g in items if g.get("dataset_id") == dataset_id]
    if status_filter and status_filter != "all":
        if status_filter == "active":
            filtered = [g for g in filtered if g.get("status") not in ("completed", "dismissed")]
        else:
            filtered = [g for g in filtered if g.get("status") == status_filter]
    return filtered


def _num_or_none(v):
    if v is None or v == "":
        return None
    try:
        s = str(v).replace(",", "").strip()
        # accept human scale like 1.2M / 350K
        m = re.match(r"^([+-]?[\d.]+)\s*([kmb])?$", s, re.I)
        if m:
            n = float(m.group(1))
            suf = (m.group(2) or "").lower()
            if suf == "k":
                n *= 1_000
            elif suf == "m":
                n *= 1_000_000
            elif suf == "b":
                n *= 1_000_000_000
            return n
        return float(s)
    except Exception:
        return None


def create_goal(user_id: int, dataset_id: str, data: dict) -> dict:
    """Create a new goal with backend-computed status (accepts target/current aliases)."""
    target = data.get("target_value", data.get("target", data.get("targetValue")))
    current = data.get("current_value", data.get("current", data.get("currentValue")))
    goal = {
        "id": uuid.uuid4().hex[:12],
        "user_id": user_id,
        "dataset_id": dataset_id,
        "title": data.get("title") or data.get("name") or "Untitled goal",
        "metric": data.get("metric"),
        "target_value": _num_or_none(target),
        "target": _num_or_none(target),
        "current_value": _num_or_none(current),
        "current": _num_or_none(current),
        "due_date": data.get("due_date") or data.get("dueDate") or data.get("due"),
        "due": data.get("due_date") or data.get("dueDate"),
        "status": data.get("status") or "active",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    # Compute backend status
    _compute_goal_status(goal)
    items = _load_user_items(user_id, "goals")
    items.append(goal)
    _save_user_items(user_id, "goals", items)
    return goal


def update_goal(user_id: int, goal_id: str, data: dict) -> dict | None:
    """Update goal status/fields. Returns updated goal or None."""
    items = _load_user_items(user_id, "goals")
    alias = {"target": "target_value", "targetValue": "target_value",
             "current": "current_value", "currentValue": "current_value",
             "due": "due_date", "dueDate": "due_date", "name": "title"}
    for g in items:
        gid = str(g.get("id") or g.get("goal_id") or "")
        if gid == str(goal_id) and g.get("user_id") == user_id:
            for k, v in (data or {}).items():
                kk = alias.get(k, k)
                if kk in ("title", "metric", "target_value", "current_value", "due_date", "status"):
                    if v is not None:
                        if kk in ("target_value", "current_value"):
                            g[kk] = _num_or_none(v)
                            g["target" if kk == "target_value" else "current"] = _num_or_none(v)
                        else:
                            g[kk] = v
            g["updated_at"] = datetime.now(timezone.utc).isoformat()
            _compute_goal_status(g)
            _save_user_items(user_id, "goals", items)
            return g
    return None


def delete_goal(user_id: int, goal_id: str) -> bool:
    items = _load_user_items(user_id, "goals")
    new_items = [g for g in items if not (str(g.get("id") or g.get("goal_id") or "") == str(goal_id) and g.get("user_id") == user_id)]
    if len(new_items) < len(items):
        _save_user_items(user_id, "goals", new_items)
        return True
    return False


def refresh_goal(user_id: int, dataset_id: str, goal_id: str,
                 session: dict) -> dict:
    """Recompute one goal's current value from live data (auto-progress).

    Reuses the run_explore-equivalent aggregation for the goal's metric
    (overall, no breakdown) on the user's current dataset: the live total
    becomes the goal's current, progress/current-target status is recomputed
    via _compute_goal_status, and the stored goal is updated.

    Honest failure (ok False, goal unchanged) when the goal has no metric,
    the goal is not found, or the aggregation cannot run.
    """
    items = _load_user_items(user_id, "goals")
    idx = next((i for i, g in enumerate(items)
                if str(g.get("id") or g.get("goal_id") or "") == str(goal_id)
                and g.get("user_id") == user_id
                and g.get("dataset_id") == dataset_id), None)
    if idx is None:
        # Fall back to a dataset-agnostic lookup so refresh still finds
        # goals created before dataset fingerprinting stabilized.
        idx = next((i for i, g in enumerate(items)
                    if str(g.get("id") or g.get("goal_id") or "") == str(goal_id)
                    and g.get("user_id") == user_id), None)
        if idx is None:
            return {"ok": False, "detail": "Goal not found.", "code": "NOT_FOUND"}
    goal = items[idx]
    metric = goal.get("metric")
    if not metric or not str(metric).strip():
        return {"ok": False, "detail": "This goal has no metric to refresh — set a metric first.",
                "code": "METRIC_MISSING", "item": goal, "data": goal}
    try:
        agg = run_explore(session or {}, {"metric": str(metric).strip()})
    except Exception as exc:
        log.warning("goal refresh aggregation crashed: %s", exc)
        return {"ok": False, "detail": "ARIA couldn't re-read this metric from your data. Try again in a moment.",
                "code": "ANALYSIS_FAILED", "item": goal, "data": goal}
    if not agg.get("ok"):
        return {"ok": False,
                "detail": agg.get("detail") or "ARIA couldn't re-read this metric from your data.",
                "code": agg.get("code") or "ANALYSIS_FAILED",
                "item": goal, "data": goal}
    total = agg.get("total")
    try:
        live = float(total) if total is not None else None
    except (TypeError, ValueError):
        live = None
    if live is None or (isinstance(live, float) and (math.isnan(live) or math.isinf(live))):
        return {"ok": False, "detail": "The live total for this metric is not numeric — goal unchanged.",
                "code": "NON_NUMERIC_TOTAL", "item": goal, "data": goal}
    goal["current_value"] = round(live, 2)
    goal["current"] = round(live, 2)
    goal["updated_at"] = datetime.now(timezone.utc).isoformat()
    goal["last_refresh"] = goal["updated_at"]
    goal["last_refresh_total"] = round(live, 2)
    _compute_goal_status(goal)
    items[idx] = goal
    _save_user_items(user_id, "goals", items)
    out = {"ok": True, "item": goal, "data": goal}
    return out


def _compute_goal_status(goal: dict) -> None:
    """Compute progress % and status label from current/target (human-worded)."""
    try:
        cur = goal.get("current_value", goal.get("current"))
        tgt = goal.get("target_value", goal.get("target"))
        current = float(cur or 0)
        target = float(tgt or 0)
    except (TypeError, ValueError):
        return
    goal["current_value"] = current if goal.get("current_value") is not None or goal.get("current") is not None else goal.get("current_value")
    goal["current"] = current if goal.get("current") is not None or goal.get("current_value") is not None else goal.get("current")
    goal["target_value"] = target if goal.get("target_value") is not None or goal.get("target") is not None else goal.get("target_value")
    goal["target"] = target if goal.get("target") is not None or goal.get("target_value") is not None else goal.get("target")
    if target <= 0:
        goal["status"] = goal.get("status") or "active"
        return
    pct = current / target
    prog = round(min(pct * 100, 100), 1)
    goal["progress_pct"] = prog
    goal["progress"] = prog
    goal["percent_complete"] = prog
    # Status-pill thresholds (shared with the frontend statusTone mapping:
    # rose Behind / amber Near / emerald On track + Achieved).
    if pct >= 1.0:
        goal["status_label"] = "Achieved"
    elif pct >= 0.85:
        goal["status_label"] = "Near target"
    elif pct >= 0.5:
        goal["status_label"] = "On track"
    else:
        goal["status_label"] = "Behind target"
    if not goal.get("status"):
        goal["status"] = "active"
    goal["delta"] = round(current - target, 2)
    goal["difference"] = round(current - target, 2)
    goal["delta_pct"] = round((current - target) / target * 100, 1) if target else 0
    goal["delta_percent"] = goal["delta_pct"]
    goal["pct_off"] = goal["delta_pct"]
    # Human sentence for cards
    if pct >= 1.0:
        goal["summary"] = f"Reached {_human_total(current)} against a {_human_total(target)} target — nicely done."
    elif pct >= 0.85:
        goal["summary"] = f"At {_human_total(current)} of {_human_total(target)} — within touching distance."
    else:
        goal["summary"] = f"At {_human_total(current)} of {_human_total(target)} — needs attention to close the gap."


# ---------------------------------------------------------------------------
# Analyses (saved explore results)
# ---------------------------------------------------------------------------

def _sanitize_saved_text(value, max_len=200):
    """Neutralize saved titles echoed in list endpoints (tags/controls/cap)."""
    s = "" if value is None else str(value)
    s = re.sub(r"<[^>]*>", " ", s)
    s = "".join(c if c.isprintable() else " " for c in s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        s = s[:max(0, max_len - 1)].rstrip() + "…"
    return s


def save_analysis(user_id: int, dataset_id: str, data: dict) -> dict:
    data = data or {}
    metric = data.get("metric")
    breakdown = data.get("breakdown")
    ml = display_name(str(metric), "Analysis") if metric else "Analysis"
    bl = display_name(str(breakdown), "") if breakdown else ""
    title = _sanitize_saved_text(
        data.get("title") or data.get("name") or (f"{ml} by {bl}" if bl else ml), 200)
    now = datetime.now(timezone.utc).isoformat()
    analysis = {
        "id": uuid.uuid4().hex[:12],
        "user_id": user_id,
        "dataset_id": dataset_id,
        "title": title,
        "name": title,
        "metric": metric,
        "breakdown": breakdown,
        "time_grain": data.get("time_grain") or data.get("time") or data.get("timeGrain"),
        "result": data.get("result"),
        "result_summary": data.get("result_summary") or data.get("summary") or "",
        "saved": True,
        "collection": data.get("collection"),
        "status": "new",
        "created_at": now,
        "updated_at": now,
    }
    items = _load_user_items(user_id, "analyses")
    items.append(analysis)
    _save_user_items(user_id, "analyses", items)
    return analysis


def list_analyses(user_id: int, dataset_id: str, params: dict | None = None) -> list:
    items = _load_user_items(user_id, "analyses")
    items = [a for a in items if a.get("dataset_id") == dataset_id]
    params = params or {}
    sort = str(params.get("sort") or "recent").lower()
    if sort == "recent":
        items = sorted(items, key=lambda a: str(a.get("created_at", "")), reverse=True)
    try:
        page = max(1, int(params.get("page") or 1))
    except Exception:
        page = 1
    try:
        page_size = int(params.get("page_size") or params.get("pageSize") or 50)
    except Exception:
        page_size = 50
    page_size = max(1, min(100, page_size))
    start = (page - 1) * page_size
    return items[start:start + page_size]


def save_scenario(user_id: int, dataset_id: str, data: dict) -> dict:
    data = data or {}
    now = datetime.now(timezone.utc).isoformat()
    scen = {
        "id": uuid.uuid4().hex[:12],
        "user_id": user_id,
        "dataset_id": dataset_id,
        "title": _sanitize_saved_text(data.get("title") or data.get("name") or f"{display_name(str(data.get('variable', 'Scenario')), 'Scenario')} scenario", 200),
        "name": _sanitize_saved_text(data.get("name") or data.get("title"), 200) or None,
        "variable": data.get("variable"),
        "change_value": data.get("change_value"),
        "change_type": data.get("change_type"),
        "result": data.get("result") or data,
        "created_at": now,
    }
    items = _load_user_items(user_id, "scenarios")
    items.append(scen)
    _save_user_items(user_id, "scenarios", items)
    return scen


def list_scenarios(user_id: int, dataset_id: str, params: dict | None = None) -> list:
    items = _load_user_items(user_id, "scenarios")
    items = [a for a in items if a.get("dataset_id") == dataset_id]
    items = sorted(items, key=lambda a: str(a.get("created_at", "")), reverse=True)
    params = params or {}
    try:
        page_size = int(params.get("page_size") or params.get("pageSize") or 6)
    except Exception:
        page_size = 6
    return items[:max(1, min(50, page_size))]


def list_report_templates(user_id: int) -> list:
    items = _load_user_items(user_id, "report_templates")
    if not items:
        defaults = [
            {"id": "professional", "name": "Professional", "description": "Clean and modern",
             "sections": ["Executive Summary", "Key Metrics", "Analysis & Charts", "Insights", "Recommendations"]},
            {"id": "executive", "name": "Executive", "description": "Focused on key insights",
             "sections": ["Executive Summary", "Key Metrics", "Insights", "Hypotheses", "Recommendations"]},
            {"id": "detailed", "name": "Detailed", "description": "Comprehensive analysis",
             "sections": ["Executive Summary", "Key Metrics", "Analysis & Charts", "Insights", "Hypotheses", "Forecast", "Goals", "What-If Scenarios", "Recommendations"]},
        ]
        _save_user_items(user_id, "report_templates", defaults)
        return defaults
    return items


def save_report_template(user_id: int, data: dict) -> dict:
    data = data or {}
    tpl = {
        "id": uuid.uuid4().hex[:8],
        "name": data.get("name") or "Untitled template",
        "description": data.get("description") or "Saved from the Reports tab.",
        "sections": data.get("sections") or [],
        "template": data.get("template"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    items = _load_user_items(user_id, "report_templates")
    # keep defaults + custom
    if not items:
        items = list_report_templates(user_id)
    items.append(tpl)
    _save_user_items(user_id, "report_templates", items)
    return tpl


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def generate_report(user_id: int, dataset_id: str, data: dict) -> dict:
    """Generate structured report from analysis data.

    Accepts optional draft hypothesis items (hypothesis-to-report working
    draft): ``items`` / ``hypotheses`` / ``hypothesis_ids`` — each either an
    id string or an {id, title} dict. Every entry is resolved against the
    live hypothesis set so the report embeds real titles, summaries and
    evidence strength; unknown ids are kept by title only and flagged.
    When the sections list is empty, draft items are appended as a custom
    "Hypothesis Spotlight" section assembled from real hypothesis data.
    """
    data = data or {}
    report = {
        "id": uuid.uuid4().hex[:12],
        "user_id": user_id,
        "dataset_id": dataset_id,
        "name": data.get("name", "ARIA Report"),
        "description": data.get("description", ""),
        "date_range": data.get("date_range"),
        "sections": data.get("sections", []),
        "template": data.get("template", "professional"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "generated",
    }

    # Assemble report content from the latest verified outputs:
    # latest analysis + hypotheses + goals (never fabricated).
    analyses = list_analyses(user_id, dataset_id)
    latest = analyses[0] if analyses else {}
    try:
        hyps = _hypothesis_items({"user_id": user_id}) or []
    except Exception:
        hyps = []
    goals = list_goals(user_id, dataset_id, "all")
    try:
        scenarios = list_scenarios(user_id, dataset_id, {"page_size": 10}) or []
    except Exception:
        scenarios = []

    # ---- working-draft items (hypothesis-to-report) ---------------------
    raw_items = data.get("items") or data.get("hypotheses") or data.get("hypothesis_ids") or []
    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    draft_items: list[dict] = []
    for entry in list(raw_items or []):
        hid = entry.get("id") if isinstance(entry, dict) else entry
        htitle = (entry.get("title") if isinstance(entry, dict) else None) or ""
        live = next((h for h in hyps if str(h.get("id")) == str(hid)), None) if hid else None
        if live is None and htitle:
            live = next((h for h in hyps
                         if str(h.get("title") or "").strip().lower() == str(htitle).strip().lower()), None)
        if live is not None:
            draft_items.append({
                "id": live.get("id"),
                "title": live.get("title") or live.get("question") or htitle,
                "summary": live.get("summary") or "",
                "metric": live.get("metric_label") or live.get("metric") or "",
                "evidence_strength": live.get("evidence_strength") or live.get("strength") or "",
                "status": live.get("status") or "",
            })
        elif hid or htitle:
            draft_items.append({"id": hid, "title": htitle or str(hid),
                                "summary": "", "metric": "",
                                "evidence_strength": "", "status": "unresolved",
                                "note": "No longer matches a live hypothesis — kept by title."})
    sections = list(report.get("sections") or [])
    if draft_items:
        spotlight = "; ".join(
            f"{d.get('title')}"
            f"{(' — ' + str(d.get('summary'))) if d.get('summary') else ''}"
            f"{(' (evidence: ' + str(d.get('evidence_strength')) + ')') if d.get('evidence_strength') else ''}"
            for d in draft_items)
        custom = {"name": "Hypothesis Spotlight",
                  "text": f"Working-draft hypotheses ({len(draft_items)}): {spotlight}"}
        report["draft_items"] = draft_items
        report["draft_section"] = custom
        if not any((s.get("name") if isinstance(s, dict) else s) == "Hypothesis Spotlight"
                   for s in sections):
            sections = [*sections, custom]
            report["sections"] = sections

    report["content"] = {
        "analyses_included": len(analyses),
        "generated_at": report["generated_at"],
        "latest_analysis": {
            "title": latest.get("title") or latest.get("name") or "",
            "metric": latest.get("metric") or "",
            "summary": latest.get("result_summary") or latest.get("summary") or "",
        } if latest else {},
        "hypotheses": [{"id": h.get("id"), "title": h.get("title")} for h in hyps[:5]],
        "hypotheses_included": len(hyps),
        "draft_items": draft_items,
        "draft_included": len(draft_items),
        "goals": [{"title": g.get("title") or g.get("name"),
                   "metric": g.get("metric") or "",
                   "current": g.get("current_value", g.get("current")),
                   "target": g.get("target_value", g.get("target")),
                   "progress": g.get("progress_pct", g.get("progress")),
                   "progress_pct": g.get("progress_pct", g.get("progress")),
                   "status": g.get("status_label") or g.get("status")} for g in goals[:10]],
        "goals_included": len(goals),
        "scenarios": [{"id": s.get("id"), "title": s.get("title") or s.get("name"),
                       "variable": s.get("variable"),
                       "change_value": s.get("change_value"),
                       "change_type": s.get("change_type")} for s in scenarios[:10]],
        "scenarios_included": len(scenarios),
    }

    items = _load_user_items(user_id, "reports")
    items.append(report)
    _save_user_items(user_id, "reports", items)
    return report


def list_reports(user_id: int, dataset_id: str) -> list:
    items = _load_user_items(user_id, "reports")
    return [r for r in items if r.get("dataset_id") == dataset_id]


# ---------------------------------------------------------------------------
# Coverage audit
# ---------------------------------------------------------------------------

def get_coverage(session: dict) -> dict:
    """Return what ARIA examined in the last analysis run (both shapes)."""
    mapping = _load_schema_mapping(session)
    if not mapping:
        return {"ok": True, "examined": 0, "accepted": 0, "rejected": 0, "reasons": {},
                "tables": [], "columns": [], "total_cells": 0,
                "reason": NO_SCHEMA_REASON, "hint": NO_SCHEMA_HINT}
    roles = _classify_columns(mapping)
    examined = 0
    accepted = 0
    rejected = 0
    reasons = {}
    tables = list(roles.keys())
    columns: list[dict] = []
    for tname, col_roles in roles.items():
        measures = [c for c, r in col_roles.items() if r in MEASURE_FAMILY]
        dims = [c for c, r in col_roles.items() if r in ("DIMENSION", "CATEGORY", "TEXT")]
        for c, r in col_roles.items():
            columns.append({"table": tname, "column": c, "label": display_name(c, c), "role": r})
        for m in measures:
            for d in dims:
                examined += 1
                if is_measure_role(col_roles.get(d, "")):
                    rejected += 1
                    reasons["measure_used_as_dimension"] = reasons.get("measure_used_as_dimension", 0) + 1
                else:
                    accepted += 1
    return {
        "ok": True,
        "examined": examined,
        "accepted": accepted,
        "rejected": rejected,
        "reasons": reasons,
        "tables": tables,
        "columns": columns,
        "total_cells": examined,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_num(v) -> str:
    if v is None:
        return "N/A"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(v) >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"{v / 1_000:.1f}K"
    if abs(v) < 0.01 and v != 0:
        return f"{v:.4f}"
    return f"{v:,.2f}"


def delete_analysis(user_id: int, analysis_id: str) -> bool:
    items = _load_user_items(user_id, "analyses")
    new_items = [a for a in items if str(a.get("id") or "") != str(analysis_id)]
    if len(new_items) < len(items):
        _save_user_items(user_id, "analyses", new_items)
        return True
    return False


def delete_scenario(user_id: int, scenario_id: str) -> bool:
    items = _load_user_items(user_id, "scenarios")
    new_items = [a for a in items if str(a.get("id") or "") != str(scenario_id)]
    if len(new_items) < len(items):
        _save_user_items(user_id, "scenarios", new_items)
        return True
    return False


def delete_report(user_id: int, report_id: str) -> bool:
    items = _load_user_items(user_id, "reports")
    new_items = [a for a in items if str(a.get("id") or "") != str(report_id)]
    if len(new_items) < len(items):
        _save_user_items(user_id, "reports", new_items)
        return True
    return False


def get_report(user_id: int, dataset_id: str, report_id: str) -> dict | None:
    items = _load_user_items(user_id, "reports")
    for r in items:
        if str(r.get("id") or "") == str(report_id) and r.get("dataset_id") == dataset_id:
            return r
    return None
