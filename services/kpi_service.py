"""KPI/dashboard business logic (services layer — canonical home).

Moved verbatim from api/kpi_routes.py: KPI calc, chart recommendation,
time-series fetching, retail discovery. The router re-exports every name
imported from here, so existing imports keep working. No FastAPI imports in
this module (SQLAlchemy use stays inside functions as before).
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import types
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text

from core.schema_intel import _safe_name
from core.semantic import humanize, kpi_label

from core.errors import DB_ERRORS, PURE_ERRORS, get_request_id, log_and_sanitize

logger = logging.getLogger("aria.kpi.service")

__all__ = ["_SERIES_FALLBACK_MAX", "_CHART_BY_INTENT", "DASHBOARD_MIN_SCORE", "_AMOUNT_HINTS", "_ENTITY_HINTS", "_DATE_HINTS", "_STORE_HINTS", "_HOLIDAY_HINTS", "_SALES_HINTS", "_DATE_PARSE_FORMATS", "_MONTH_NAMES", "_fmt", "_is_currency_col", "_safe", "_validate_schema_ident", "_bound_limit", "_recommend_chart", "_profile_numeric", "_rollback", "_variation", "_score_kpi", "_time_grain_expr", "_parse_full_date", "_parse_date_to_month", "_pretty_month", "_pretty_date", "_discover_special_columns", "_fetch_time_series", "_holiday_lift", "_retail_candidates_for_table", "_measure_expr", "_coerce_num"]


def _measure_expr(intel, tname: str, col: str, dialect: str) -> str:
    """Quoted ref for SUM/AVG over a measure, TEXT-storage safe.

    Fully schema-agnostic (no table/column literals): when the declared
    dtype is text-ish (e.g. a relational table holding ``'\\u20b91,299'`` or
    ``'15%'`` in TEXT columns), returns the shared
    :func:`core.text_numeric.coercion_sql` expression so aggregates yield
    numbers instead of raising ``function sum(text) does not exist``.
    Unparsable values become NULL (never invented). Native numerics and
    unknown dtypes keep the plain quoted column. Never raises.
    """
    from core.schema_intel import _safe_name
    try:
        q = _safe_name(col, dialect)
    except Exception:
        return str(col)
    try:
        dtype = intel.dtype_of(tname, col) if intel is not None else ""
    except Exception:
        dtype = ""
    dt = str(dtype or "").lower()
    if any(k in dt for k in ("char", "text", "varchar", "string", "clob", "uuid")):
        try:
            from core.text_numeric import coercion_sql as _cs
            return _cs(dialect, q)
        except Exception:
            return q
    return q


def _coerce_num(value):
    """Parse one raw cell to float, TEXT-numeric aware (currency/percent/commas).

    Returns None when unparsable. Schema-agnostic: reuses
    :func:`core.text_numeric.try_parse_text_number`, falling back to a
    comma-stripped float for plain numerics.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            import math as _math
            f = float(value)
            return f if _math.isfinite(f) else None
        except (TypeError, ValueError):
            return None
    try:
        from core.text_numeric import try_parse_text_number as _tpn
        parsed = _tpn(value)
        if parsed is not None:
            import math
            return parsed if math.isfinite(parsed) else None
    except Exception:
        pass
    try:
        import math
        parsed = float(str(value).replace(",", "").strip())
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _fmt(v, is_currency=False):
    if v is None:
        return "N/A"
    if is_currency:
        if abs(v) >= 1_000_000:
            return f"${v/1_000_000:,.1f}M"
        if abs(v) >= 1_000:
            return f"${v/1_000:,.0f}K"
        return f"${v:,.0f}"
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:,.1f}M"
    if abs(v) >= 1_000:
        return f"{v/1_000:,.0f}K"
    return f"{v:,.0f}"


def _is_currency_col(col_name: str) -> bool:
    from core.schema_intel import AMOUNT_HINTS
    cl = col_name.lower()
    return any(k in cl for k in AMOUNT_HINTS)


def _safe(val):
    """SQL-escape a string value."""
    if val is None:
        return "NULL"
    return str(val).replace("'", "''")


def _validate_schema_ident(intel, tname: str, col: str | None = None) -> None:
    """Validate dynamic identifiers against the known schema mapping.

    Table/column names can never be bound params, so the control is
    membership in the extracted schema plus the existing ``_safe_name``
    quoting at use sites. Unknown names raise instead of reaching SQL.

    Multi-schema mappings key tables as ``schema.table`` (e.g. AdventureWorks
    ``sales.salesorderheader``): each dot-separated segment is validated as
    an identifier (max 3 segments), then the full key must be a member of the
    extracted schema. Single-segment names follow the exact old path.
    """
    from core.semantic import is_valid_identifier

    def _ok_dotted(value: str) -> bool:
        try:
            segs = [s for s in str(value or "").split(".") if s]
        except Exception:
            return False
        if not segs or len(segs) > 3:
            return False
        # Reject empty segments ("a..b", ".a", "a.") and any segment that
        # is not a plain identifier (injection payloads still raise).
        if len(".".join(segs)) != len(str(value or "")):
            return False
        return all(is_valid_identifier(s, max_len=128) for s in segs)

    if not _ok_dotted(str(tname or "")):
        raise ValueError(f"Invalid table name: {tname!r}")
    known_tables = set(getattr(intel, "tables", {}).keys()) if intel is not None else set()
    if known_tables and str(tname) not in known_tables:
        raise ValueError(f"Unknown table: {tname!r}")
    if col is not None:
        if not _ok_dotted(str(col or "")):
            raise ValueError(f"Invalid column name: {col!r}")
        try:
            known_cols = set(intel.get_col_names(tname)) if intel is not None else set()
        except PURE_ERRORS as exc:
            logger.debug("schema ident columns skipped: %s", exc)
            known_cols = set()
        if known_cols and str(col) not in known_cols:
            raise ValueError(f"Unknown column: {tname}.{col!r}")


def _bound_limit(n, default: int = 30, maximum: int = 500) -> int:
    """Clamp a dynamic LIMIT to a validated int (never raw user input)."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return default
    return max(1, min(n, maximum))


# Bounded raw-pair pull for the non-ISO-date series fallback (spec §3):
# month bucketing needs raw rows, but never an unbounded dump.
_SERIES_FALLBACK_MAX = 20000


# ── Chart Selection Intelligence ────────────────────────────────────────
# Section 45 of spec: choose chart based on analytical intent, cardinality,
# measure type, time dimension, etc. Deterministic — no LLM needed.

_CHART_BY_INTENT = {
    "time_series": "line",
    "category_comparison": "bar",
    "distribution": "histogram",
    "correlation": "scatter",
    "composition": "stacked_bar",
    "hierarchy": "treemap",
    "geographic": "map",
    "flow": "sankey",
    "financial": "waterfall",
}


def _recommend_chart(columns: list, data: list, col_types: dict = None) -> dict:
    """Recommend best chart type based on data characteristics.

    Returns {chart_type, reason, compatible_types}.
    Never overrides analytical correctness — only selects presentation.
    """
    if not columns or not data:
        return {"chart_type": "table", "reason": "No data", "compatible_types": ["table"]}

    col_types = col_types or {}
    num_cols = [c for c in columns if col_types.get(c) == "numeric" or
                col_types.get(c.lower()) == "numeric" or
                (data and isinstance(data[0].get(c), (int, float)))]
    cat_cols = [c for c in columns if c not in num_cols and
                (col_types.get(c, col_types.get(c.lower(), "")) in ("text", "category", "")
                 or (data and data[0].get(c) is not None))]
    time_cols = [c for c in columns if any(k in c.lower() for k in
                 ("date", "time", "year", "month", "day", "created", "updated"))]

    # Time series detection
    if time_cols and num_cols:
        return {
            "chart_type": "line",
            "reason": f"Time dimension ({time_cols[0]}) with numeric measure",
            "compatible_types": ["line", "area", "bar"],
        }

    # Single category + single numeric → bar
    if len(cat_cols) == 1 and len(num_cols) == 1:
        cardinality = len(set(str(row.get(cat_cols[0])) for row in data[:100]))
        if cardinality <= 20:
            return {
                "chart_type": "bar",
                "reason": f"Categorical comparison ({cardinality} categories)",
                "compatible_types": ["bar", "horizontal_bar", "pie"],
            }
        else:
            return {
                "chart_type": "bar",
                "reason": f"High cardinality category ({cardinality} values)",
                "compatible_types": ["bar", "horizontal_bar"],
            }

    # Two numerics → scatter
    if len(num_cols) >= 2 and len(cat_cols) <= 2:
        return {
            "chart_type": "scatter",
            "reason": "Two numeric measures for correlation",
            "compatible_types": ["scatter", "bubble", "line"],
        }

    # Multiple categories + numeric → stacked or grouped bar
    if len(cat_cols) >= 2 and len(num_cols) >= 1:
        return {
            "chart_type": "stacked_bar",
            "reason": "Multi-dimensional comparison",
            "compatible_types": ["stacked_bar", "grouped_bar", "heatmap"],
        }

    # Single numeric only → histogram/distribution
    if len(num_cols) == 1 and len(cat_cols) == 0:
        return {
            "chart_type": "histogram",
            "reason": "Distribution of single numeric",
            "compatible_types": ["histogram", "box", "density"],
        }

    # Default: table for complex cases
    return {
        "chart_type": "table",
        "reason": "Complex data structure",
        "compatible_types": ["table", "bar"],
    }


# ── Schema loading ───────────────────────────────────────────────────────

# ── Dynamic dashboard composition (spec: profile → score → dedupe → select) ──
# Relevance > completeness > quantity. Threshold configurable via env (§17).

DASHBOARD_MIN_SCORE = float(os.getenv("DASHBOARD_MIN_SCORE", "0.35"))

_AMOUNT_HINTS = ("amount", "revenue", "sales", "profit", "price", "cost", "total",
                 "value", "payment", "expense", "income", "salary", "turnover", "billing")
_ENTITY_HINTS = ("customer", "user", "client", "member", "employee", "product",
                 "order", "account", "patient", "student", "ticket")
_DATE_HINTS = ("date", "time", "day", "month", "year", "created", "updated",
               "timestamp", "period")


def _profile_numeric(conn, tname: str, col: str, dialect: str, intel=None) -> dict | None:
    """One-query profile of a numeric column (spec §1).

    TEXT-stored measures aggregate through :func:`_measure_expr` so a
    relational TEXT column holding currency strings profiles real numbers
    instead of raising ``function sum(text) does not exist``.
    """
    if intel is not None:
        try:
            _validate_schema_ident(intel, tname, col)
        except ValueError:
            return None
    try:
        _mref = _measure_expr(intel, tname, col, dialect)
        row = conn.execute(text(
            f"SELECT COUNT(*) AS n, COUNT({_safe_name(col, dialect)}) AS nn, "
            f"COUNT(DISTINCT {_safe_name(col, dialect)}) AS d, "
            f"SUM({_mref}) AS s, AVG({_mref}) AS a, "
            f"MIN({_mref}) AS mi, MAX({_mref}) AS ma "
            f"FROM {_safe_name(tname, dialect)}"
        )).fetchone()
        if not row or not row[0]:
            return None
        n = float(row[0])
        return {"n": n, "nn": float(row[1] or 0), "distinct": float(row[2] or 0),
                "sum": row[3], "avg": row[4], "min": row[5], "max": row[6]}
    except DB_ERRORS as exc:
        logger.debug("numeric profile skipped for %s.%s: %s", tname, col, exc)
        _rollback(conn)
        return None


def _rollback(conn) -> None:
    """Reset a poisoned transaction so later queries on the same connection work.

    PostgreSQL aborts the whole transaction on one failed statement (e.g. SUM
    over a text column); without an explicit rollback every subsequent query
    on that connection fails too — which once blanked the entire dashboard.
    """
    try:
        conn.rollback()
    except DB_ERRORS as exc:
        logger.debug("rollback skipped: %s", exc)


def _variation(mean, lo, hi) -> float:
    try:
        mean, lo, hi = float(mean), float(lo), float(hi)
    except (TypeError, ValueError):
        return 0.0
    if mean == 0 or hi == lo:
        return 0.0
    return max(0.0, min(1.0, (hi - lo) / abs(mean)))


def _score_kpi(kind: str, col: str, prof: dict, has_date: bool, is_currency: bool) -> tuple[float, str]:
    """Spec §2 relevance score + human-readable reason (§18)."""
    coverage = (prof["nn"] / prof["n"]) if prof["n"] else 0.0
    null_pct = 1.0 - coverage
    var = _variation(prof["avg"], prof["min"], prof["max"])
    low = col.lower()
    semantic = 0.10
    why = [f"covers {coverage:.0%} of records"]
    if kind == "outcome":
        semantic = 0.30 if any(h in low for h in _AMOUNT_HINTS) else 0.20
        why.append("additive business measure" if any(h in low for h in _AMOUNT_HINTS) else "summable measure")
    elif kind == "entity":
        semantic = 0.25
        why.append("unique-entity count")
    elif kind == "efficiency":
        semantic = 0.20
        why.append("per-unit efficiency view")
    elif kind == "volume":
        semantic = 0.15
        why.append("dataset volume")
    business = 0.15 if is_currency else (0.10 if kind in ("outcome", "entity") else 0.05)
    trend = 0.10 if has_date else 0.0
    if has_date:
        why.append("trend comparable over time")
    score = (semantic + coverage * 0.25 + var * 0.20 + business + trend
             - null_pct * 0.50)
    if var == 0:
        why.append("no variation detected")
        score -= 0.15
    return round(score, 3), "; ".join(why)


def _time_grain_expr(col: str, dialect: str, span_days: float) -> tuple[str, str]:
    """Pick time grain from span (§6): daily / weekly / monthly / quarterly."""
    c = _safe_name(col, dialect)
    if dialect == "sqlite":
        if span_days < 62:
            return f"strftime('%Y-%m-%d', {c})", "day"
        if span_days < 200:
            return f"strftime('%Y-W%W', {c})", "week"
        if span_days < 1200:
            return f"strftime('%Y-%m', {c})", "month"
        return f"strftime('%Y', {c}) || '-Q' || ((CAST(strftime('%m', {c}) AS INTEGER) + 2) / 3)", "quarter"
    if dialect == "mysql":
        if span_days < 62:
            return f"DATE_FORMAT({c}, '%Y-%m-%d')", "day"
        if span_days < 200:
            return f"DATE_FORMAT({c}, '%Y-u%v')", "week"
        if span_days < 1200:
            return f"DATE_FORMAT({c}, '%Y-%m')", "month"
        return f"CONCAT(YEAR({c}), '-Q', QUARTER({c}))", "quarter"
    if span_days < 62:
        return f"TO_CHAR({c}, 'YYYY-MM-DD')", "day"
    if span_days < 200:
        return f"TO_CHAR({c}, 'IYYY-\"W\"IW')", "week"
    if span_days < 1200:
        return f"TO_CHAR({c}, 'YYYY-MM')", "month"
    return f"TO_CHAR({c}, 'YYYY') || '-Q' || TO_CHAR({c}, 'Q')", "quarter"


# ── Retail-aware KPI discovery (single-table safe) ─────────────────────────
# Generic: no hardcoded table names. Discovers the sales-like measure, the
# date column, a low-cardinality store-like entity column and a holiday-flag
# column by name hints + SQL-measured cardinality, then computes every value
# with real executed SQL (COUNT/SUM/AVG/MIN/MAX). Values always match the DB.
# Scoring still goes through _score_kpi (unchanged) so relevance semantics
# are preserved; these candidates are only *guaranteed consideration* so a
# 1-table source (e.g. Walmart: 6435 rows) never yields an empty dashboard.

_STORE_HINTS = ("store", "shop", "branch", "outlet", "location", "site",
                "warehouse", "depot")
_HOLIDAY_HINTS = ("holiday", "is_holiday", "holiday_flag", "festival",
                  "event_flag", "promo_flag", "is_promo")
_SALES_HINTS = ("weekly_sales", "sales", "revenue", "turnover", "amount",
                "total", "income", "profit")

# Flexible parse order matters: day-first before month-first so DMY sources
# (e.g. Walmart "05-02-2010" = 5 Feb 2010, proven by "26-10-2012") bucket
# correctly, while ISO values still parse via the fast path below.
# Month-name forms are unambiguous in any position (catalog exports like
# "June 7, 2019" or "7 June 2019").
_DATE_PARSE_FORMATS = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d",
    "%d-%m-%Y %H:%M:%S", "%d-%m-%Y", "%d/%m/%Y",
    "%m-%d-%Y", "%m/%d/%Y", "%d-%m-%y", "%m-%d-%y",
    "%Y-%m", "%Y%m%d",
    "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y",
    "%d %B %Y", "%d %b %Y", "%d %B, %Y", "%d %b, %Y",
    "%Y.%m.%d", "%d.%m.%Y", "%m.%d.%Y",
)

_MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _parse_full_date(raw) -> datetime | None:
    """Parse a DB date value flexibly. Returns datetime or None."""
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw
    s = str(raw).strip()
    if not s:
        return None
    try:  # ISO fast path: YYYY-MM-DD[ HH:MM:SS]
        return datetime.fromisoformat(s[:19])
    except (ValueError, TypeError) as exc:
        logger.debug("iso date parse skipped for %r: %s", s, exc)
    for fmt in _DATE_PARSE_FORMATS:
        try:
            return datetime.strptime(s[:19], fmt)
        except (ValueError, TypeError) as exc:
            logger.debug("date parse skipped for format %s: %s", fmt, exc)
            continue
    return None


def _parse_date_to_month(raw) -> str | None:
    """Bucket a DB date value to 'YYYY-MM'. Returns None when unparseable."""
    dt = _parse_full_date(raw)
    if dt is None:
        return None
    return f"{dt.year:04d}-{dt.month:02d}"


def _pretty_month(period: str) -> str:
    """'2010-02' -> 'Feb 2010'. Passes through anything unparseable."""
    m = re.match(r"^(\d{4})-(\d{2})$", str(period or "").strip())
    if m:
        try:
            return f"{_MONTH_NAMES[int(m.group(2)) - 1]} {m.group(1)}"
        except (ValueError, TypeError, IndexError) as exc:
            logger.debug("pretty month skipped for %r: %s", period, exc)
    return str(period)


def _pretty_date(raw) -> str:
    """'05-02-2010' -> '5 Feb 2010'. Passes through anything unparseable."""
    dt = _parse_full_date(raw)
    if dt is None:
        return str(raw)
    return f"{dt.day} {_MONTH_NAMES[dt.month - 1]} {dt.year}"


def _discover_special_columns(intel, tname: str) -> dict:
    """Discover (measure, date, store, holiday) columns for one table.

    Pure metadata + name hints — no SQL, no hardcoded table names.
    """
    try:
        col_names = [str(c) for c in intel.get_col_names(tname)]
    except PURE_ERRORS as exc:
        logger.debug("special-column discovery skipped for %s: %s", tname, exc)
        col_names = []
    try:
        numerics = set(intel.get_numeric_cols(tname))
    except PURE_ERRORS as exc:
        logger.debug("numeric discovery skipped for %s: %s", tname, exc)
        numerics = set()

    measure = None
    try:
        measure = intel.find_best_measure(tname)
    except PURE_ERRORS as exc:
        logger.debug("measure discovery skipped for %s: %s", tname, exc)
        measure = None
    if not measure or measure not in numerics:
        for c in col_names:
            if c in numerics and any(h in c.lower() for h in _SALES_HINTS):
                measure = c
                break
        if not measure and numerics:
            for c in col_names:
                if c in numerics and not c.lower().endswith("_id") and c.lower() != "id":
                    measure = c
                    break

    try:
        date_col = intel.find_best_datetime(tname)
    except PURE_ERRORS as exc:
        logger.debug("date discovery skipped for %s: %s", tname, exc)
        date_col = None

    store_col = None
    try:
        if hasattr(intel, "find_store_column"):
            store_col = intel.find_store_column(tname)
    except PURE_ERRORS as exc:
        logger.debug("store discovery skipped for %s: %s", tname, exc)
        store_col = None
    if not store_col:
        for c in col_names:
            if any(h in c.lower() for h in _STORE_HINTS):
                store_col = c
                break

    holiday_col = None
    try:
        if hasattr(intel, "find_holiday_column"):
            holiday_col = intel.find_holiday_column(tname)
    except PURE_ERRORS as exc:
        logger.debug("holiday discovery skipped for %s: %s", tname, exc)
        holiday_col = None
    if not holiday_col:
        for c in col_names:
            if any(h in c.lower() for h in _HOLIDAY_HINTS):
                holiday_col = c
                break

    return {"measure": measure, "date_col": date_col,
            "store_col": store_col, "holiday_col": holiday_col}


def _fetch_time_series(conn, tname: str, dt_col: str, measure: str,
                       dialect: str, limit: int = 30, intel=None) -> tuple[str | None, list]:
    """Grain-aware time series from real executed SQL.

    Tries SQL-side grouping first (exact, dialect-aware). When the date
    column is non-ISO text (e.g. Walmart 'DD-MM-YYYY'), SQL date functions
    return NULL — falls back to fetching raw (date, measure) pairs with real
    SQL and     bucketing to months in Python. Either way every number comes
    from the database. Returns (grain, [{period, value}]).

    Identifiers are validated against the known schema when ``intel`` is
    supplied; LIMIT values are clamped ints (bound-params-where-possible —
    identifiers themselves can only be validated + quoted).
    """
    if intel is not None:
        try:
            _validate_schema_ident(intel, tname, dt_col)
            _validate_schema_ident(intel, tname, measure)
        except ValueError:
            return None, []
    limit = _bound_limit(limit)
    try:
        _mref = _measure_expr(intel, tname, measure, dialect)
        srow = conn.execute(text(
            f"SELECT MIN({_safe_name(dt_col, dialect)}), "
            f"MAX({_safe_name(dt_col, dialect)}), "
            f"SUM({_mref}), "
            f"AVG({_mref}), "
            f"MIN({_mref}), "
            f"MAX({_mref}) "
            f"FROM {_safe_name(tname, dialect)} "
            f"WHERE {_safe_name(dt_col, dialect)} IS NOT NULL"
        )).fetchone()
    except DB_ERRORS as exc:
        logger.debug("time-series profile skipped for %s.%s: %s", tname, measure, exc)
        _rollback(conn)
        return None, []
    if not srow or srow[0] is None:
        return None, []

    dmin, dmax = _parse_full_date(srow[0]), _parse_full_date(srow[1])
    if dmin is not None and dmax is not None:
        span_days = max(0.0, (dmax - dmin).total_seconds() / 86400)
    else:
        span_days = 365.0
    time_expr, grain = _time_grain_expr(dt_col, dialect, span_days)
    limit_n = _bound_limit(60 if grain == "day" else limit, maximum=500)

    try:
        result = conn.execute(
            text(
                f"SELECT period, val FROM (SELECT {time_expr} AS period, "
                f"SUM({_mref}) AS val "
                f"FROM {_safe_name(tname, dialect)} "
                f"WHERE {_safe_name(dt_col, dialect)} IS NOT NULL "
                f"GROUP BY period ORDER BY period DESC LIMIT :limit_n) "
                f"ORDER BY period"
            ),
            {"limit_n": limit_n},
        )
        rows = result.fetchall()
        chart_data = [{"period": str(r[0]), "value": float(r[1])}
                      for r in rows if r[0] is not None and r[1] is not None]
        if len(chart_data) >= 2:
            return grain, chart_data[-limit:]
    except DB_ERRORS as exc:
        logger.debug("time-series grouping skipped for %s: %s", tname, exc)
        _rollback(conn)

    # Fallback for non-ISO date text: real SQL fetches raw pairs, Python
    # only buckets them to months (no values invented). Bounded: at most
    # _SERIES_FALLBACK_MAX raw pairs feed the month buckets, and the chart
    # itself carries at most `limit` (≤500) aggregated points.
    try:
        result = conn.execute(
            text(
                f"SELECT {_safe_name(dt_col, dialect)} AS d, "
                f"{_mref} AS v "
                f"FROM {_safe_name(tname, dialect)} "
                f"WHERE {_safe_name(dt_col, dialect)} IS NOT NULL "
                f"LIMIT :fallback_limit"
            ),
            {"fallback_limit": _SERIES_FALLBACK_MAX},
        )
        buckets: dict[str, float] = {}
        for d, v in result.fetchall():
            m = _parse_date_to_month(d)
            if not m or v is None:
                continue
            try:
                buckets[m] = buckets.get(m, 0.0) + float(v)
            except (TypeError, ValueError):
                continue
        if len(buckets) >= 2:
            chart_data = [{"period": k, "value": buckets[k]}
                          for k in sorted(buckets)][-limit:]
            return "month", chart_data
    except DB_ERRORS as exc:
        logger.debug("time-series fallback skipped for %s: %s", tname, exc)
        _rollback(conn)
    return None, []


def _fetch_count_series(conn, tname: str, dt_col: str, dialect: str,
                        limit: int = 30, intel=None) -> tuple[str | None, list]:
    """Row-count trend per month from real data (no summable measure needed).

    Text-heavy tables (catalogs, logs) often have a real date axis but no
    SUM-able number; counting rows per month still yields a truthful volume
    trend ("Titles added over time"). Dialect-agnostic: fetches raw date
    values with bounded real SQL and buckets months in Python — the same
    contract as the _fetch_time_series text fallback. Never invents values.
    """
    if intel is not None:
        try:
            _validate_schema_ident(intel, tname, dt_col)
        except ValueError:
            return None, []
    limit = _bound_limit(limit)
    try:
        result = conn.execute(
            text(
                f"SELECT {_safe_name(dt_col, dialect)} AS d "
                f"FROM {_safe_name(tname, dialect)} "
                f"WHERE {_safe_name(dt_col, dialect)} IS NOT NULL "
                f"LIMIT :fallback_limit"
            ),
            {"fallback_limit": _SERIES_FALLBACK_MAX},
        )
        buckets: dict[str, int] = {}
        for (d,) in result.fetchall():
            m = _parse_date_to_month(d)
            if not m:
                continue
            buckets[m] = buckets.get(m, 0) + 1
        if len(buckets) >= 2:
            chart_data = [{"period": k, "value": float(buckets[k])}
                          for k in sorted(buckets)][-limit:]
            return "month", chart_data
    except DB_ERRORS as exc:
        logger.debug("count-series skipped for %s: %s", tname, exc)
        _rollback(conn)
    return None, []


def _holiday_lift(conn, tname: str, measure: str, holiday_col: str,
                   dialect: str, intel=None) -> tuple[float | None, float | None,
                                          float | None, int]:
    """Holiday vs non-holiday average lift (%) from real executed SQL.

    Returns (lift_pct, holiday_avg, non_holiday_avg, holiday_n).
    lift_pct is None when the flag column has no usable 0/1 split.
    """
    if intel is not None:
        try:
            _validate_schema_ident(intel, tname, measure)
            _validate_schema_ident(intel, tname, holiday_col)
        except ValueError:
            return None, None, None, 0
    mc = _measure_expr(intel, tname, measure, dialect)
    hc = _safe_name(holiday_col, dialect)
    tc = _safe_name(tname, dialect)
    predicates = [
        (f"CAST({hc} AS INTEGER) = 1", f"(CAST({hc} AS INTEGER) = 0)"),
        (f"UPPER(TRIM(CAST({hc} AS TEXT))) IN ('1','TRUE','T','YES','Y')",
         f"UPPER(TRIM(CAST({hc} AS TEXT))) IN ('0','FALSE','F','NO','N')"),
    ]
    for hol_pred, non_pred in predicates:
        try:
            row = conn.execute(text(
                f"SELECT AVG(CASE WHEN {hol_pred} THEN {mc} END), "
                f"AVG(CASE WHEN {non_pred} THEN {mc} END), "
                f"COUNT(CASE WHEN {hol_pred} THEN 1 END), "
                f"COUNT(CASE WHEN {non_pred} THEN 1 END) "
                f"FROM {tc}"
            )).fetchone()
            if not row or row[0] is None or row[1] is None:
                continue
            hol_avg, non_avg = float(row[0]), float(row[1])
            hol_n, non_n = int(row[2] or 0), int(row[3] or 0)
            if hol_n < 1 or non_n < 1 or non_avg == 0:
                continue
            lift = ((hol_avg - non_avg) / abs(non_avg)) * 100
            return round(lift, 1), hol_avg, non_avg, hol_n
        except DB_ERRORS as exc:
            logger.debug("holiday lift skipped for %s: %s", tname, exc)
            _rollback(conn)
            continue
    return None, None, None, 0


def _retail_candidates_for_table(conn, intel, tname: str,
                                 dialect: str) -> list[dict]:
    """Scored retail KPI candidates for one table, all from real SQL.

    Surfaces: total + average of the sales measure, distinct store count,
    date-range span, and holiday lift. Each candidate carries score/reason
    via the unchanged _score_kpi so the diversity pass treats them like any
    other KPI — they are guaranteed *consideration*, never mocked values.
    Numeric SUM candidates also carry spark/trend series from real SQL.
    """
    cands: list[dict] = []
    try:
        _validate_schema_ident(intel, tname)
    except ValueError:
        return []
    disc = _discover_special_columns(intel, tname)
    measure, date_col = disc["measure"], disc["date_col"]
    store_col, holiday_col = disc["store_col"], disc["holiday_col"]
    tdesc = intel.tables[tname].get("description", tname)

    try:
        nrow = conn.execute(text(
            f"SELECT COUNT(*) FROM {_safe_name(tname, dialect)}")).fetchone()
        n = int(nrow[0]) if nrow and nrow[0] else 0
    except DB_ERRORS as exc:
        logger.debug("retail row-count skipped for %s: %s", tname, exc)
        _rollback(conn)
        return []
    if not n:
        return []

    if measure:
        try:
            numerics = set(intel.get_numeric_cols(tname))
        except PURE_ERRORS as exc:
            logger.debug("retail numerics skipped for %s: %s", tname, exc)
            numerics = {measure}
        if measure in numerics:
            prof = _profile_numeric(conn, tname, measure, dialect, intel=intel)
            if prof and prof["nn"]:
                is_curr = _is_currency_col(measure)
                spark: list = []
                trend: list = []
                if date_col:
                    _grain, ts = _fetch_time_series(conn, tname, date_col,
                                                    measure, dialect, intel=intel)
                    if ts:
                        trend = ts
                        spark = [p["value"] for p in ts[-12:]]
                if prof["sum"] is not None:
                    score, reason = _score_kpi("outcome", measure, prof,
                                               bool(date_col), is_curr)
                    cands.append({
                        "kind": "outcome", "table": tname, "column": measure,
                        "agg": "SUM", "value": float(prof["sum"]),
                        "score": score, "reason": reason,
                        "is_currency": is_curr, "date_col": date_col,
                        "tdesc": tdesc, "spark": spark, "trend": trend,
                    })
                if prof["avg"] is not None:
                    score, reason = _score_kpi("efficiency", measure, prof,
                                               bool(date_col), is_curr)
                    cands.append({
                        "kind": "efficiency", "table": tname, "column": measure,
                        "agg": "AVG", "value": float(prof["avg"]),
                        "score": score, "reason": reason,
                        "is_currency": is_curr, "date_col": date_col,
                        "tdesc": tdesc, "spark": spark, "trend": trend,
                    })
                if holiday_col:
                    lift, hol_avg, non_avg, hol_n = _holiday_lift(
                        conn, tname, measure, holiday_col, dialect, intel=intel)
                    if lift is not None:
                        score, reason = _score_kpi("efficiency", measure, prof,
                                                   bool(date_col), is_curr)
                        cands.append({
                            "kind": "efficiency", "table": tname,
                            "column": f"{measure}__holiday_lift",
                            "agg": "LIFT", "value": float(lift),
                            "score": score, "reason": reason,
                            "is_currency": False, "date_col": date_col,
                            "tdesc": tdesc,
                            "label_override": "Holiday Lift",
                            "format_override": "percent",
                            "period_override": (
                                f"Holiday avg {_fmt(hol_avg, is_curr)} vs "
                                f"{_fmt(non_avg, is_curr)} "
                                f"({hol_n} holiday periods)"),
                            "change_override": float(lift),
                            "spark": [non_avg, hol_avg] if non_avg is not None else [],
                        })

    if store_col:
        try:
            erow = conn.execute(text(
                f"SELECT COUNT(DISTINCT {_safe_name(store_col, dialect)}) "
                f"FROM {_safe_name(tname, dialect)}")).fetchone()
            ecount = int(erow[0]) if erow and erow[0] else 0
            # Single-table safe: any entity count well below the row count
            # is a real breakdown dimension (Walmart: 45 stores / 6435 rows).
            if 2 <= ecount <= max(10, n) and ecount < 0.98 * n:
                eprof = {"n": float(n), "nn": float(n),
                         "distinct": float(ecount), "sum": None,
                         "avg": None, "min": 0, "max": 1}
                score, reason = _score_kpi("entity", store_col, eprof,
                                           bool(date_col), False)
                cands.append({
                    "kind": "entity", "table": tname, "column": store_col,
                    "agg": "DISTINCT", "value": float(ecount),
                    "score": score, "reason": reason,
                    "is_currency": False, "date_col": date_col,
                    "tdesc": tdesc,
                })
        except DB_ERRORS as exc:
            logger.debug("retail entity count skipped for %s: %s", tname, exc)
            _rollback(conn)

    if date_col:
        try:
            drow = conn.execute(text(
                f"SELECT MIN({_safe_name(date_col, dialect)}), "
                f"MAX({_safe_name(date_col, dialect)}), "
                f"COUNT({_safe_name(date_col, dialect)}), "
                f"COUNT(DISTINCT {_safe_name(date_col, dialect)}) "
                f"FROM {_safe_name(tname, dialect)}")).fetchone()
            if drow and drow[0] is not None and drow[1] is not None:
                nn_dates = float(drow[2] or 0)
                n_distinct = int(drow[3] or 0)
                # Score via _score_kpi like every other KPI; the distinct
                # date count is the honest spread signal for a range.
                vprof = {"n": float(n), "nn": nn_dates,
                         "distinct": float(n_distinct), "sum": None,
                         "avg": (float(n_distinct) / 2.0) if n_distinct else None,
                         "min": 0, "max": n_distinct}
                score, reason = _score_kpi("volume", date_col, vprof,
                                           False, False)
                cands.append({
                    "kind": "volume", "table": tname, "column": date_col,
                    "agg": "RANGE", "value": float(n_distinct or 0),
                    "score": score, "reason": reason,
                    "is_currency": False, "date_col": date_col,
                    "tdesc": tdesc,
                    "label_override": "Date Range",
                    "value_override": (
                        f"{_pretty_date(drow[0])} → {_pretty_date(drow[1])}"),
                    "format_override": "text",
                    "period_override": f"{n:,} records",
                })
        except DB_ERRORS as exc:
            logger.debug("retail date-range skipped for %s: %s", tname, exc)
            _rollback(conn)

    return cands
