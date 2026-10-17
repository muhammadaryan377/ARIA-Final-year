"""Predictive analysis and prescription routes.

Provides standalone endpoints for forecasting and generating
actionable recommendations from the connected database.
"""

import json
import logging
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Any

from core.config import get_session
from core.data_intelligence import hint_hit as _hint_hit
from core.deps import require_data, require_writable, get_current_user, reject_document_sql
from core.secret_box import decrypt as _decrypt_secret

log = logging.getLogger("aria.predictive_routes")
router = APIRouter()


class ForecastFilter(BaseModel):
    """Structured row filter: applied as SQL WHERE + in-memory post-filter.

    Additive-only: optional ``filters`` on ForecastRequest. Generic —
    column validated against the resolved table's schema, op from a fixed
    allow-list, value bound as a query parameter (never interpolated).
    """

    column: str
    op: str = "="
    value: Any = None


class ForecastRequest(BaseModel):
    question: str = ""
    column: str | None = None
    time_dim: str | None = None
    # None = auto-suggest from frequency + history length (dynamic default);
    # any integer overrides the suggestion.
    horizon: int | None = None
    # Structured filters (additive-only): validated against schema and
    # applied as WHERE in both the GoalAgent path (question intent +
    # in-memory enforcement) and the deterministic fallback SQL.
    filters: list[ForecastFilter] | None = None


class PrescriptionRequest(BaseModel):
    insight_id: str = ""


class ScenarioRequest(BaseModel):
    column: str | None = None
    pct_change: float = 10.0
    periods: int = 3


_DATE_NAME_HINTS = ("date", "time", "day", "month", "year", "created",
                    "updated", "timestamp", "period", "week", "quarter")

# Allow-listed structured-filter ops (generic, no table/column literals).
# Maps normalized op -> SQL operator. "IN" expects a list/tuple value.
_FILTER_OP_SQL: dict[str, str] = {
    "=": "=",
    "==": "=",
    "!=": "!=",
    "<>": "<>",
    ">": ">",
    ">=": ">=",
    "<": "<",
    "<=": "<=",
    "like": "LIKE",
    "in": "IN",
}

# Single deterministic-fallback row cap (perf: bounded GROUP BY grain query).
_DETERMINISTIC_LIMIT = 2000


def _normalize_filter_op(op):
    """Normalize a filter op to its SQL operator, or None if disallowed."""
    try:
        key = str(op or "").strip().lower()
    except Exception:
        return None
    if key in ("=", "=="):
        return "="
    return _FILTER_OP_SQL.get(key) or _FILTER_OP_SQL.get(str(op or "").strip())


def _validate_forecast_filters(filters, tables, table):
    """Validate structured filters against the resolved table's schema.

    Exact-only column resolution (no substring match): case-insensitive
    exact match against real columns. Returns (applied, errors) where
    applied = [{column, op, value}] with resolved column names. Generic.
    """
    if not filters:
        return [], []
    try:
        valid_cols = {n for n, _d in _table_columns(tables, table)}
    except Exception:
        valid_cols = set()
    low = {c.lower(): c for c in valid_cols}
    applied: list[dict] = []
    errors: list[str] = []
    for f in filters:
        try:
            raw_col = str((f.column if isinstance(f, dict) else getattr(f, "column", "")) or "").strip()
            raw_op = (f["op"] if isinstance(f, dict) and "op" in f else getattr(f, "op", "="))
            raw_val = (f["value"] if isinstance(f, dict) and "value" in f else getattr(f, "value", None))
        except Exception:
            errors.append("Invalid filter entry; expected {column, op, value}.")
            continue
        if not raw_col:
            errors.append("Filter is missing 'column'.")
            continue
        resolved = low.get(raw_col.lower())
        # Exact-only: raw_col must equal a real column (case-insensitive).
        if resolved is None and raw_col not in valid_cols:
            errors.append(f"Filter column '{raw_col}' is not a column of '{table}'.")
            continue
        resolved = resolved or raw_col
        sql_op = _normalize_filter_op(raw_op)
        if sql_op is None:
            errors.append(
                f"Filter op '{raw_op}' is not supported. "
                f"Use one of: =, !=, <>, >, >=, <, <=, LIKE, IN."
            )
            continue
        if sql_op == "IN" and not isinstance(raw_val, (list, tuple)):
            errors.append(f"Filter op 'IN' on '{resolved}' needs a list value.")
            continue
        applied.append({"column": resolved, "op": sql_op, "value": raw_val})
    return applied, errors


def _filter_where_clause(applied, dialect):
    """Build (WHERE fragment, params) for validated filters. Param-bound."""
    from core.schema_intel import _safe_name
    if not applied:
        return "", {}
    parts: list[str] = []
    params: dict = {}
    for i, f in enumerate(applied):
        try:
            q = _safe_name(f["column"], dialect)
        except Exception:
            q = str(f["column"])
        pname = f"flt_{i}"
        if f["op"] == "IN":
            vals = list(f["value"] or [])
            if not vals:
                parts.append("1 = 0")
                continue
            names = []
            for j, v in enumerate(vals):
                pn = f"{pname}_{j}"
                names.append(f":{pn}")
                params[pn] = v
            parts.append(f"{q} IN ({', '.join(names)})")
        else:
            parts.append(f"{q} {f['op']} :{pname}")
            params[pname] = f["value"]
    return (" AND ".join(parts), params)


def _apply_filters_in_memory(rows, applied):
    """Enforce validated filters on in-memory GoalAgent result rows."""
    if not applied or not rows:
        return rows
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        keep = True
        for f in applied:
            col, op, val = f["column"], f["op"], f["value"]
            # Exact-only column lookup (case-insensitive exact fallback).
            if col in r:
                rv = r[col]
            else:
                low = {str(k).lower(): k for k in r.keys()}
                rv = r.get(low.get(str(col).lower(), col), None)
            try:
                if op == "=":
                    ok = str(rv) == str(val) if isinstance(val, bool) is False and isinstance(rv, bool) is False else rv == val
                    # Numeric leniency: "10" == 10 only when both parse.
                    if not ok:
                        try:
                            ok = float(rv) == float(val)
                        except Exception:
                            ok = False
                elif op in ("!=", "<>"):
                    try:
                        ok = not (float(rv) == float(val))
                    except Exception:
                        ok = str(rv) != str(val)
                elif op in (">", ">=", "<", "<="):
                    try:
                        l, rr = float(rv), float(val)
                        ok = (l > rr) if op == ">" else (l >= rr) if op == ">=" else (l < rr) if op == "<" else (l <= rr)
                    except Exception:
                        ok = (str(rv) > str(val)) if op == ">" else (str(rv) >= str(val)) if op == ">=" else (str(rv) < str(val)) if op == "<" else (str(rv) <= str(val))
                elif op == "LIKE":
                    import re as _re
                    pat = "^" + _re.escape(str(val)).replace("%", ".*").replace("_", ".") + "$"
                    ok = _re.match(pat, str(rv), _re.IGNORECASE) is not None
                elif op == "IN":
                    try:
                        ok = rv in (val or []) or str(rv) in [str(v) for v in (val or [])]
                    except Exception:
                        ok = False
                else:
                    ok = True
            except Exception:
                ok = True
            if not ok:
                keep = False
                break
        if keep:
            out.append(r)
    return out


def _describe_filters_for_question(applied):
    """Natural-language filter intent appended to the GoalAgent question so its
    generated SQL WHERE reflects the structured filters (generic)."""
    if not applied:
        return ""
    bits = []
    for f in applied:
        try:
            if f["op"] == "IN":
                bits.append(f"{f['column']} IN ({', '.join(str(v) for v in (f['value'] or []))})")
            else:
                bits.append(f"{f['column']} {f['op']} {f['value']}")
        except Exception:
            continue
    return (" Only include rows where " + " AND ".join(bits) + ".") if bits else ""


def _horizon_info(requested, forecast):
    """Top-level horizon disclosure {requested, used, capped, note}."""
    try:
        used = forecast.get("horizon") if isinstance(forecast, dict) else None
    except Exception:
        used = None
    try:
        capped = bool((forecast or {}).get("horizon_capped")) if isinstance(forecast, dict) else False
    except Exception:
        capped = False
    try:
        note = (forecast or {}).get("horizon_note") if isinstance(forecast, dict) else None
    except Exception:
        note = None
    if used is None:
        try:
            used = int(requested) if requested is not None else 6
        except Exception:
            used = 6
    if capped and not note:
        note = f"Requested horizon {requested} exceeds available history; capped to {used}."
    return {"requested": requested, "used": used, "capped": capped, "note": note}


def _table_columns(tables, tname):
    tinfo = tables.get(tname, {})
    cols = tinfo.get("columns", []) if isinstance(tinfo, dict) else []
    if isinstance(cols, dict):
        return [(k, str((v or {}).get("data_type", ""))) for k, v in cols.items()]
    out = []
    for c in cols:
        if isinstance(c, dict):
            out.append((c.get("column") or c.get("name") or "", str(c.get("data_type", "") or "")))
        else:
            out.append((str(c), ""))
    return [(n, d) for n, d in out if n]


def _coerced_ref(tables, tname, col, dialect):
    """Quoted ref for SUM/AVG over a measure, TEXT-storage safe.

    Fully schema-agnostic (no table/column literals): TEXT-dtype columns
    aggregate through :func:`core.text_numeric.coercion_sql` so currency
    strings never raise ``function sum(text) does not exist``. Unparsable
    values become NULL. Native numerics keep the plain column. Never raises.
    """
    from core.schema_intel import _safe_name
    try:
        q = _safe_name(col, dialect)
    except Exception:
        return str(col)
    try:
        dtype = next((d for n, d in _table_columns(tables, tname) if n == col), "")
    except Exception:
        dtype = ""
    if any(k in str(dtype or "").lower()
           for k in ("char", "text", "varchar", "string", "clob", "uuid")):
        try:
            from core.text_numeric import coercion_sql as _cs
            return _cs(dialect, q)
        except Exception:
            return q
    return q


# Eligibility cache: keyed by (user_id, db_name, schema mtime) so reconnects
# and re-renders never rescan wide databases. Bounded, short TTL, and only
# successful results are cached — errors always recompute. Generic pattern
# (mirrors the dashboard config cache).
_ELIG_CACHE: dict = {}
_ELIG_CACHE_TTL = 180.0
_ELIG_CACHE_MAX = 200


def _apply_stmt_timeout(conn, dialect):
    """Best-effort per-connection statement timeout (ms, env-tunable).

    Guards the eligibility scan against pathological full-table aggregations
    on very large databases. Never raises; dialects without support skip.
    """
    try:
        import os as _os
        _ms = int(_os.getenv("ARIA_ELIG_STMT_TIMEOUT_MS", "8000"))
    except Exception:
        _ms = 8000
    if _ms <= 0:
        return
    try:
        from sqlalchemy import text as _t
        if str(dialect or "").lower() in ("postgresql", "postgres", "pgsql"):
            conn.execute(_t(f"SET statement_timeout TO '{int(_ms)}ms'"))
        elif str(dialect or "").lower() == "sqlite":
            conn.execute(_t(f"PRAGMA busy_timeout={int(_ms)}"))
    except Exception:
        pass


def _validate_time_dim(engine_or_conn, dialect, table, column, dtype):
    """Check a candidate time column against REAL data.

    Returns dict with kind/distinct/null_pct/parse_ratio/span or None when
    the column cannot serve as a forecast axis. Never invents dates.

    Accepts an Engine (opens a short-lived connection) or a live
    Connection (reused by the eligibility pass to avoid per-candidate
    reconnects on wide multi-table databases).
    """
    from sqlalchemy import text as _t
    from core.schema_intel import _safe_name

    def _run(conn):
        row = conn.execute(_t(
            f"SELECT COUNT(*) n, COUNT({_safe_name(column, dialect)}) nn, "
            f"COUNT(DISTINCT {_safe_name(column, dialect)}) nd "
            f"FROM {_safe_name(table, dialect)}"
        )).fetchone()
        n, nn, nd = int(row[0] or 0), int(row[1] or 0), int(row[2] or 0)
        if n == 0 or nd < 2:
            return None, None
        vals = [r[0] for r in conn.execute(_t(
            f"SELECT DISTINCT {_safe_name(column, dialect)} "
            f"FROM {_safe_name(table, dialect)} "
            f"WHERE {_safe_name(column, dialect)} IS NOT NULL LIMIT 200"
        )).fetchall()]
        return (n, nn, nd), vals

    try:
        if hasattr(engine_or_conn, "connect"):
            with engine_or_conn.connect() as conn:
                _apply_stmt_timeout(conn, dialect)
                counts, vals = _run(conn)
        else:
            counts, vals = _run(engine_or_conn)
        if counts is None:
            return None
        n, nn, nd = counts
    except Exception as exc:
        log.warning("Eligibility sampling failed for %s.%s: %s", table, column, exc)
        return None
    low_dtype = (dtype or "").lower()
    real_date = any(k in low_dtype for k in ("date", "time", "timestamp"))
    try:
        import pandas as _pd
        parsed = _pd.to_datetime([str(v) for v in vals], errors="coerce")
        ratio = float(parsed.notna().sum()) / max(1, len(vals))
    except Exception:
        ratio = 1.0 if real_date else 0.0
    if not real_date and ratio < 0.8:
        return None
    span = None
    try:
        import pandas as _pd2
        ts = _pd2.to_datetime([str(v) for v in vals], errors="coerce").dropna()
        if len(ts):
            span = [str(ts.min()), str(ts.max())]
    except Exception:
        span = None
    return {
        "column": column,
        "kind": "date" if real_date else "text",
        "distinct": nd,
        "null_pct": round((1 - nn / n) * 100, 1) if n else 100.0,
        "parse_ratio": round(ratio, 2),
        "span": span,
    }


def _attach_forecast_verdict(payload):
    """Best-effort Verification Agent hookup for forecasts (never blocks).

    Attaches ``payload["verdict"]``. On any failure attaches an honest
    not-verified verdict and still returns the data untouched.
    """
    try:
        from verification.agent import verify_forecast
        verdict = verify_forecast(payload if isinstance(payload, dict) else {}, {})
    except Exception:
        verdict = {"verified": False, "checks": [],
                   "warnings": ["verification unavailable"]}
    try:
        payload["verdict"] = verdict
    except Exception:
        pass
    return payload


def _evidence_state_for_forecast(forecast, table_rows, table):
    """Additive canonical state for forecast payloads (never overrides)."""
    try:
        from core.evidence import assess_evidence as _assess
        n_obs = (forecast or {}).get("n_obs")
        rows = table_rows if table_rows is not None else n_obs
        return _assess(rows if rows is not None else 0, "forecast",
                       history_points=n_obs, dataset_name=table)
    except Exception:
        return None


def _scenario_evidence_state(row_count, table, history_points=None):
    """Additive canonical state for scenario payloads."""
    try:
        from core.evidence import assess_evidence as _assess
        return _assess(row_count, "scenario", history_points=history_points,
                       dataset_name=table)
    except Exception:
        return None


@router.get("/api/predictive/eligibility")
def forecast_eligibility(request: Request, user: dict = Depends(get_current_user)):
    """Inspect the connected source and report what can actually be forecast.

    Dynamic across PostgreSQL/MySQL/CSV: discovers temporal columns from real
    data (dtype + value parsing, never invented), lists measurable metrics
    per table (numeric, non-key, non-FK via SchemaIntelligence), counts
    observations, and recommends a valid default target + time dimension.
    """
    from api.kpi_routes import _build_engine as _kb
    from core.schema_intel import SchemaIntelligence as _SI

    session = get_session(user["user_id"])
    reject_document_sql(session)
    schema_path = session.get("schema_path")
    if not schema_path or not Path(str(schema_path)).exists():
        return {"ok": True, "eligible": False, "connected": False,
                "tables": [], "recommended": None,
                "reasons": ["No data source is connected yet. Connect PostgreSQL, MySQL, CSV or PDF first."]}
    try:
        mapping = json.loads(Path(str(schema_path)).read_text(encoding="utf-8"))
    except Exception:
        return {"ok": True, "eligible": False, "connected": True,
                "tables": [], "recommended": None,
                "reasons": ["The schema description could not be read. Reconnect or rescan the schema, then try again."]}
    tables = mapping.get("tables", {})
    if not tables:
        return {"ok": True, "eligible": False, "connected": True,
                "tables": [], "recommended": None,
                "reasons": ["The connected source has no tables to analyze."]}
    intel = _SI(mapping)
    engine, dialect, _dbt = _kb(session)
    if engine is None:
        return {"ok": True, "eligible": False, "connected": True,
                "tables": [], "recommended": None,
                "reasons": ["This source type cannot be queried for forecasting yet."]}
    from sqlalchemy import text as _t
    from core.schema_intel import _safe_name

    # Short-TTL cache so remounts, StrictMode double-fetches and DB-switch
    # re-renders never rescan wide databases. Keyed by schema mtime so
    # reconnects/uploads invalidate automatically. Errors are never cached.
    try:
        _schema_mtime = Path(str(schema_path)).stat().st_mtime
    except Exception:
        _schema_mtime = 0.0
    _elig_key = (user.get("user_id"), session.get("db_name", ""), _schema_mtime)
    _cached = _ELIG_CACHE.get(_elig_key)
    if _cached and (time.time() - _cached[0]) < _ELIG_CACHE_TTL:
        return {"ok": True, **_cached[1], "cached": True}
    if len(_ELIG_CACHE) > _ELIG_CACHE_MAX:
        _ELIG_CACHE.clear()

    # Metadata-only plan (no SQL): measures + time candidates per table.
    # Date-dtype columns first, then date-hinted text; capped so wide
    # databases validate the most promising axes instead of everything.
    _plan = []
    for tname in list(tables)[:15]:
        cols = _table_columns(tables, tname)
        measures = [m for m in (intel.find_all_measures(tname) or [])]
        # Time candidates: real date dtypes first, then date-hinted text cols.
        # Token-based hints (never substrings): "holiday" must not match "day".
        date_cols = [n for n, d in cols
                     if any(k in d.lower() for k in ("date", "time", "timestamp"))]
        text_hints = [n for n, d in cols
                      if n not in date_cols and _hint_hit(n, _DATE_NAME_HINTS)]
        _plan.append((tname, cols, measures, (date_cols + text_hints)[:3]))

    def _scan_one(tname, cols, measures, cands):
        """Validate one table on its own connection (thread-safe)."""
        t_reasons: list[str] = []
        time_dims = []
        n_rows = 0
        try:
            with engine.connect() as conn:
                _apply_stmt_timeout(conn, dialect)
                for cand in cands:
                    dtype = next((d for n, d in cols if n == cand), "")
                    checked = _validate_time_dim(conn, dialect, tname, cand, dtype)
                    if checked:
                        if checked["null_pct"] > 50:
                            t_reasons.append(
                                f"{tname}.{cand} looks like a date but is empty too often "
                                f"({checked['null_pct']}% missing), so it was skipped.")
                            continue
                        time_dims.append(checked)
                try:
                    n_rows = int(conn.execute(_t(
                        f"SELECT COUNT(*) FROM {_safe_name(tname, dialect)}")).fetchone()[0] or 0)
                except Exception:
                    n_rows = 0
        except Exception:
            pass
        return ({"table": tname, "measures": measures,
                 "time_dims": time_dims, "row_count": n_rows}, t_reasons)

    def _prio(item):
        _tname, _cols, _measures, _cands = item
        if _measures and _cands:
            return 0
        if _measures or _cands:
            return 1
        return 2

    # Forecast-ready tables scan first; results reassembled in mapping order
    # so dropdowns and the recommendation pick stay deterministic.
    _order = sorted(range(len(_plan)), key=lambda i: (_prio(_plan[i]), i))
    _entries: dict[int, dict] = {}
    _t_reasons: dict[int, list] = {}
    if str(dialect or "").lower() in ("postgresql", "postgres", "pgsql", "mysql", "mariadb"):
        import concurrent.futures as _cf
        with _cf.ThreadPoolExecutor(max_workers=4) as _ex:
            for _idx, _res in zip(_order, _ex.map(lambda _i: _scan_one(*_plan[_i]), _order)):
                _entries[_idx], _t_reasons[_idx] = _res
    else:
        for _idx in _order:
            _entries[_idx], _t_reasons[_idx] = _scan_one(*_plan[_idx])
    out_tables = []
    reasons: list[str] = []
    for _i in range(len(_plan)):
        out_tables.append(_entries[_i])
        reasons.extend(_t_reasons[_i])
    if not any(t["measures"] for t in out_tables):
        reasons.append("No measurable numeric metric was found (only ids, keys, codes or text). "
                       "Forecasting needs a number that changes over time, such as an amount, quantity or count.")
    if not any(t["time_dims"] for t in out_tables):
        reasons.append("No usable time dimension was found. ARIA looked for date/datetime columns "
                       "and date-like text (month, year, created …) with real parseable values, "
                       "but nothing had enough valid distinct dates.")
    recommended = None
    for t in out_tables:
        if t["measures"] and t["time_dims"]:
            best_td = max(t["time_dims"], key=lambda d: (d["kind"] == "date", d["distinct"]))
            if best_td["distinct"] >= 5:
                recommended = {
                    "table": t["table"],
                    "measure": t["measures"][0],
                    "time_dim": best_td["column"],
                    "history_points": best_td["distinct"],
                }
                break
    if recommended is None:
        for t in out_tables:
            if t["measures"] and t["time_dims"]:
                td = t["time_dims"][0]
                recommended = {
                    "table": t["table"],
                    "measure": t["measures"][0],
                    "time_dim": td["column"],
                    "history_points": td["distinct"],
                }
                reasons.append(
                    f"Only {td['distinct']} distinct time values were found — a reliable forecast "
                    f"needs at least 5. Results will be marked accordingly.")
                break
    try:
        engine.dispose()
    except Exception:
        pass
    # Per-target prediction validation (core/data_intelligence.py): cheap,
    # pure and generic — attached for transparency alongside the live
    # measure/time-dim discovery above. Never overrides it.
    try:
        from core.data_intelligence import (
            build_data_intelligence as _bdi,
            detect_capabilities as _dcaps,
        )
        _caps = _dcaps(_bdi(mapping))
        prediction_targets = _caps.get("prediction_targets", [])
    except Exception:
        prediction_targets = []
    # Additive canonical state: global rows<10 dominates (REUSED
    # GLOBAL_MIN_ROWS=10); per-table forecast minimum (min_points=5 history
    # periods) stays in the per-table time_dims/distinct checks above.
    try:
        from core.evidence import assess_evidence as _assess_el
        _max_rows = max([int(t.get("row_count") or 0) for t in out_tables] or [0])
        _ev_el = _assess_el(_max_rows, "forecast",
                           history_points=(recommended or {}).get("history_points"),
                           dataset_name=(recommended or {}).get("table"))
    except Exception:
        _ev_el = None
    _elig = {
        "ok": True,
        "eligible": recommended is not None,
        "connected": True,
        "tables": out_tables,
        "recommended": recommended,
        "reasons": reasons,
        "min_points": 5,
        "prediction_targets": prediction_targets,
    }
    if _ev_el is not None:
        _elig["evidence_state"] = _ev_el
        if _ev_el.get("status") != "supported":
            _elig["eligible"] = False
            if _ev_el.get("reason") and _ev_el["reason"] not in _elig["reasons"]:
                _elig["reasons"] = [_ev_el["reason"]] + list(_elig["reasons"])
    try:
        _ELIG_CACHE[_elig_key] = (time.time(), {k: v for k, v in _elig.items() if k != "ok"})
    except Exception:
        pass
    return _elig


@router.post("/api/predictive/scenario")
def run_scenario(req: ScenarioRequest, request: Request, user: dict = Depends(require_writable)):
    """Deterministic what-if analysis on real aggregates (spec §16).

    Applies a user-supplied percentage change to the trailing average of a
    measure. Pure arithmetic — labelled an estimate, never a forecast.
    """
    session = get_session(user["user_id"])
    reject_document_sql(session)
    schema_path = session.get("schema_path")
    if not schema_path or not Path(str(schema_path)).exists():
        raise HTTPException(status_code=400, detail="No database connected.")
    mapping = json.loads(Path(str(schema_path)).read_text(encoding="utf-8"))
    tables = mapping.get("tables", {})
    if not tables:
        raise HTTPException(status_code=400, detail="No tables found in schema.")

    from api.kpi_routes import _build_engine
    from core.schema_intel import SchemaIntelligence, _safe_name
    from sqlalchemy import text as _text

    intel = SchemaIntelligence(mapping)
    target_table, target_col = None, None

    def _holds_numbers(eng, dialect, tname, col, sample=20):
        """Runtime check: does this column actually hold numbers?

        TEXT-stored numerics (currency symbols, commas, percents) count via
        the shared text-numeric parser so TEXT columns of real numbers are
        never mistaken for "no numeric measure".
        """
        try:
            with eng.connect() as _c:
                _rows = _c.execute(_text(
                    f"SELECT {_safe_name(col, dialect)} "
                    f"FROM {_safe_name(tname, dialect)} LIMIT {int(sample)}")).fetchall()
            _vals = [r[0] for r in _rows]
            if not _vals:
                return False
            try:
                from core.text_numeric import try_parse_text_number as _tpn
            except Exception:
                _tpn = None
            _good = 0
            for _v in _vals:
                if isinstance(_v, bool):
                    continue
                try:
                    if _tpn is not None:
                        if _tpn(_v) is not None:
                            _good += 1
                    else:
                        float(str(_v).replace(",", ""))
                        _good += 1
                except (TypeError, ValueError):
                    continue
            return _good >= max(1, len(_vals) // 2)
        except Exception:
            try:
                _c.rollback()
            except Exception:
                pass
            return False

    _eng_probe, _dial_probe, _ = _build_engine(session)

    def _verified_measure(tname):
        try:
            m = intel.find_best_measure(tname)
            if m and m in (intel.get_numeric_cols(tname) or []):
                if _eng_probe is not None and not _holds_numbers(_eng_probe, _dial_probe, tname, m):
                    return None
                return m
        except Exception:
            pass
        return None

    if req.column:
        want = req.column.split(" (")[0].strip().lower()
        for tname, tinfo in tables.items():
            cols = tinfo.get("columns", []) if isinstance(tinfo, dict) else []
            names = [(c.get("column") if isinstance(c, dict) else c) for c in cols]
            for n in names:
                if not (n and str(n).lower() == want):
                    continue
                try:
                    numeric = n in (intel.get_numeric_cols(tname) or [])
                except Exception:
                    numeric = False
                if not numeric:
                    continue
                if _eng_probe is not None and not _holds_numbers(_eng_probe, _dial_probe, tname, n):
                    continue
                target_table, target_col = tname, n
                break
            if target_table:
                break
            if target_table:
                break
    if not target_col:
        for tname in tables:
            m = _verified_measure(tname)
            if m:
                target_table, target_col = tname, m
                break
    if not target_col:
        raise HTTPException(status_code=400, detail="No numeric measure available for scenario analysis.")

    pct = max(-95.0, min(95.0, float(req.pct_change or 0.0)))
    n_periods = max(1, min(12, int(req.periods or 3)))

    engine, dialect, _dbt = _build_engine(session)
    if engine is None:
        raise HTTPException(status_code=400, detail="No queryable database for this source.")
    dt_col = intel.find_best_datetime(target_table)
    # TEXT time columns (e.g. CSV months stored as '2024-01') are not real
    # dates: date-format functions return NULL/raise on them. Group by the
    # raw value instead — fully generic, works on every dialect.
    dt_is_real_date = False
    if dt_col:
        try:
            _tinfo = tables.get(target_table, {})
            _cols = _tinfo.get("columns", []) if isinstance(_tinfo, dict) else []
            if isinstance(_cols, dict):
                _dtype = str((_cols.get(dt_col) or {}).get("data_type", "") or "").lower()
            else:
                _dtype = ""
                for _c in _cols:
                    _n = (_c.get("column") if isinstance(_c, dict) else _c)
                    if _n and str(_n) == dt_col:
                        _dtype = str((_c.get("data_type", "") if isinstance(_c, dict) else "") or "").lower()
                        break
            dt_is_real_date = any(k in _dtype for k in ("date", "time", "timestamp"))
        except Exception:
            dt_is_real_date = False
    try:
        _sref = _coerced_ref(mapping.get("tables", {}), target_table, target_col, dialect)
        with engine.connect() as conn:
            if dt_col and dt_is_real_date:
                if dialect == "sqlite":
                    gexpr = f"strftime('%Y-%m', {_safe_name(dt_col, dialect)})"
                elif dialect == "mysql":
                    gexpr = f"DATE_FORMAT({_safe_name(dt_col, dialect)}, '%Y-%m')"
                else:
                    gexpr = f"TO_CHAR({_safe_name(dt_col, dialect)}, 'YYYY-MM')"
                rows = conn.execute(_text(
                    f"SELECT {gexpr} AS p, SUM({_sref}) AS v "
                    f"FROM {_safe_name(target_table, dialect)} "
                    f"WHERE {_safe_name(dt_col, dialect)} IS NOT NULL "
                    f"GROUP BY p ORDER BY p DESC LIMIT {n_periods}"
                )).fetchall()
                series = [{"period": str(r[0]), "actual": float(r[1])}
                          for r in rows if r[0] is not None and r[1] is not None]
                series.reverse()
            elif dt_col:
                rows = conn.execute(_text(
                    f"SELECT {_safe_name(dt_col, dialect)} AS p, SUM({_sref}) AS v "
                    f"FROM {_safe_name(target_table, dialect)} "
                    f"WHERE {_safe_name(dt_col, dialect)} IS NOT NULL "
                    f"GROUP BY {_safe_name(dt_col, dialect)} "
                    f"ORDER BY {_safe_name(dt_col, dialect)} DESC LIMIT {n_periods}"
                )).fetchall()
                series = [{"period": str(r[0]), "actual": float(r[1])}
                          for r in rows if r[0] is not None and r[1] is not None]
                series.reverse()
            else:
                total_row = conn.execute(_text(
                    f"SELECT SUM({_sref}) "
                    f"FROM {_safe_name(target_table, dialect)}")).fetchone()
                total = float(total_row[0]) if total_row and total_row[0] is not None else 0.0
                series = [{"period": "all data", "actual": total}]
    except HTTPException:
        raise
    except Exception as exc:
        from core.errors import sanitize_error as _san
        raise HTTPException(status_code=500, detail=f"Scenario data retrieval failed: {_san(exc)[:200]}")
    if not series:
        raise HTTPException(status_code=400, detail="Not enough data for scenario analysis.")
    # Centralized global gate: rows < 10 -> insufficient scenario state.
    # Consume the mapping row count first (csv_handler contract), else the
    # live COUNT(*). REUSES SCENARIO_MIN_OBS == 10 (prescription MIN_N).
    _sc_rows, _sc_known = None, False
    try:
        _ti_sc = (mapping.get("tables") or {}).get(target_table) or {}
        if isinstance(_ti_sc.get("row_count"), int):
            _sc_rows, _sc_known = max(0, int(_ti_sc["row_count"])), True
    except Exception:
        pass
    if not _sc_known:
        try:
            with engine.connect() as _cc:
                _sc_rows = int(_cc.execute(_text(
                    f"SELECT COUNT(*) FROM {_safe_name(target_table, dialect)}")).fetchone()[0] or 0)
            _sc_known = True
        except Exception:
            pass
    if _sc_known:
        _ev_sc = _scenario_evidence_state(_sc_rows, target_table, len(series))
        if _ev_sc is not None and _ev_sc.get("status") != "supported":
            raise HTTPException(status_code=400, detail=_ev_sc.get("reason") or
                                "Not enough data for scenario analysis.")
    baseline = sum(s["actual"] for s in series) / len(series)
    factor = 1.0 + pct / 100.0
    for s in series:
        s["scenario"] = round(s["actual"] * factor, 2)
    scenario_value = round(baseline * factor, 2)
    from insight_agent import sanitize_json
    _ev_sc_ok = None
    try:
        _ev_sc_ok = _scenario_evidence_state(
            _sc_rows if _sc_known else len(series), target_table, len(series))
    except Exception:
        pass
    _sc_payload = {
        "ok": True,
        "estimate": True,
        "table": target_table,
        "column": target_col,
        "pct_change": pct,
        "periods_used": len(series),
        "baseline": round(baseline, 2),
        "scenario_value": scenario_value,
        "delta": round(scenario_value - baseline, 2),
        "series": series,
        "assumptions": [
            f"Uniform {pct:+.1f}% change applied to {target_col} across the last {len(series)} period(s).",
            "No seasonality, trend, capacity, or cost effects modelled.",
            "Baseline is the trailing average, not a forecast.",
        ],
        "limitations": "Arithmetic estimate only. Combine with the validated forecast for a fuller picture.",
    }
    if _ev_sc_ok is not None:
        _sc_payload["evidence_state"] = _ev_sc_ok
    return sanitize_json(_sc_payload)


@router.post("/api/predictive/forecast")
def generate_forecast(req: ForecastRequest, request: Request, user: dict = Depends(require_writable)):
    """Generate a real forecast from the connected database."""
    import traceback as _tb
    session = get_session(user["user_id"])
    reject_document_sql(session)
    schema_path = session.get("schema_path")
    if not schema_path or not Path(str(schema_path)).exists():
        raise HTTPException(status_code=400, detail="No database connected.")

    db_config = dict(session.get("db") or {})
    if not db_config:
        raise HTTPException(status_code=400, detail="No database configuration found.")
    if db_config.get("password"):
        try:
            db_config["password"] = _decrypt_secret(db_config["password"])
        except Exception:
            pass

    db_uri = session.get("db_uri", "")
    db_type = session.get("db_type", "postgresql")
    dialect = "postgresql" if "postgres" in db_type else "mysql" if "mysql" in db_type else "sqlite"
    provider = session.get("provider")

    # Step 0: Validate the requested column against real measures. Keys,
    # FKs and id-like columns (e.g. reports_to) are not forecastable —
    # asking the GoalAgent to "forecast" one yields garbage like COUNT(*).
    # Exact-only resolution (no substring match): anything that is not a
    # genuine measure falls back to auto-select with substituted=True.
    from core.schema_intel import SchemaIntelligence as _SI0
    from insight_agent import sanitize_json as _sj_shared
    _mapping0 = json.loads(Path(str(schema_path)).read_text(encoding="utf-8"))
    _intel0 = _SI0(_mapping0)
    _requested_measure_raw = ((req.column or "").split(" (")[0]).strip() or None
    _want0 = _requested_measure_raw or ""
    # Perf: single shared engine for all direct SQL in this request
    # (row-count gate + deterministic fallback share it; GoalAgent keeps
    # its own internal engine). Disposed in finally at each return site.
    from api.kpi_routes import _build_engine as _kb_shared
    from core.schema_intel import _safe_name as _sn_shared
    _shared_engine, _shared_dialect = None, dialect
    try:
        _shared_engine, _shared_dialect_tmp, _ = _kb_shared(session)
        if _shared_engine is not None and _shared_dialect_tmp:
            _shared_dialect = _shared_dialect_tmp
    except Exception:
        _shared_engine = None

    def _dispose_shared():
        try:
            if _shared_engine is not None:
                _shared_engine.dispose()
        except Exception:
            pass

    # Exact-only measure lookup across tables (case-insensitive exact;
    # never substring): collect every (table, measure) whose measure
    # exactly equals the request.
    _measure_candidates: list[tuple] = []
    if _want0:
        for _tn in _mapping0.get("tables", {}):
            try:
                _ms = _intel0.find_all_measures(_tn) or []
            except Exception:
                _ms = []
            for _m in _ms:
                try:
                    if str(_m).lower() == _want0.lower():
                        _measure_candidates.append((_tn, _m))
                except Exception:
                    continue
    _measure_tables: dict[str, str] = {}  # lower column -> table (first hit, compat)
    for _tn in _mapping0.get("tables", {}):
        try:
            for _m in (_intel0.find_all_measures(_tn) or []):
                _measure_tables.setdefault(str(_m).lower(), _tn)
        except Exception:
            continue
    if len(_measure_candidates) > 1:
        # Ambiguous request: same measure name in several tables — return
        # candidates for disambiguation instead of silently picking one.
        _ambig = [{"table": t, "measure": m} for t, m in _measure_candidates]
        try:
            _dispose_shared()
        except Exception:
            pass
        return _sj_shared({
            "ok": False,
            "question": req.question or f"Show {_want0} over time",
            "reason": f"'{_want0}' exists in several tables; specify which one to forecast.",
            "missing_field": "measure",
            "suggestion": "Re-run with the table-qualified measure, e.g. "
                          f"'{_measure_candidates[0][1]} ({_measure_candidates[0][0]})'.",
            "target": _want0,
            "candidates": _ambig,
            "target_candidates": _ambig,
            "selection": {"measure": None, "table": None, "time_dim": None,
                          "requested_measure": _requested_measure_raw,
                          "requested_time_dim": ((req.time_dim or "").split(" (")[0]).strip() or None,
                          "substituted": False,
                          "substitution_reason": "Ambiguous measure: candidates returned."},
            "info": {"rows_analyzed": 0, "fallback_used": False, "goal_message": None,
                     "requested_measure": _requested_measure_raw, "resolved_measure": None,
                     "requested_time_dim": ((req.time_dim or "").split(" (")[0]).strip() or None,
                     "resolved_time_dim": None, "substituted": False,
                     "substitution_reason": "Ambiguous measure: candidates returned.",
                     "filters_applied": []},
            "horizon_info": {"requested": req.horizon, "used": int(req.horizon or 6),
                             "capped": False, "note": None},
            "reliability": "insufficient_data",
            "n_obs": 0,
        })
    if _want0 and _measure_candidates:
        valid_measure, valid_table = _measure_candidates[0][1], _measure_candidates[0][0]
    else:
        valid_measure, valid_table = None, None
        for _tn in _mapping0.get("tables", {}):
            try:
                _m = _intel0.find_best_measure(_tn)
            except Exception:
                _m = None
            if _m:
                valid_measure, valid_table = _m, _tn
                break
    if valid_measure is None:
        try:
            _dispose_shared()
        except Exception:
            pass
        raise HTTPException(
            status_code=400,
            detail="No forecastable number found in this data. Forecasting needs a numeric measure (not ids, keys or codes).",
        )
    # Requested-vs-resolved echo for the measure (fixes silent swaps).
    if _requested_measure_raw and _requested_measure_raw.lower() == str(valid_measure).lower():
        _measure_substituted, _measure_reason = False, None
    elif _requested_measure_raw:
        _measure_substituted = True
        _measure_reason = (
            f"Requested column '{_requested_measure_raw}' is not an exact forecastable "
            f"measure; using '{valid_measure}' from '{valid_table}'."
        )
    else:
        _measure_substituted, _measure_reason = False, None
    # Structured filters: validate against the resolved table's schema now
    # so both the GoalAgent path and the deterministic SQL share them.
    try:
        _filter_applied, _filter_errors = _validate_forecast_filters(
            list(req.filters or []) if req.filters else [],
            _mapping0.get("tables", {}), valid_table)
    except Exception:
        _filter_applied, _filter_errors = [], ["Could not validate filters."]
    if _filter_errors:
        try:
            _dispose_shared()
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="; ".join(_filter_errors)[:500])

    # Centralized global gate (core/evidence.py): rows < 10 -> refusal for
    # ALL substantive conclusions, BEFORE any GoalAgent/LLM work. Consume the
    # row counts the mapping exposes; fall back to a live COUNT(*) only when
    # the mapping has no counts (reusing the single shared engine).
    # Per-capability forecast minimum (min_points=5 history periods) is
    # REUSED downstream in PredictionAgent.
    _fc_rows, _fc_known = None, False
    try:
        _tinfo0 = (_mapping0.get("tables") or {}).get(valid_table) or {}
        if isinstance(_tinfo0.get("row_count"), int):
            _fc_rows, _fc_known = max(0, int(_tinfo0["row_count"])), True
    except Exception:
        pass
    if not _fc_known and _shared_engine is not None:
        try:
            from sqlalchemy import text as _t0
            with _shared_engine.connect() as _c0:
                _fc_rows = int(_c0.execute(_t0(
                    f"SELECT COUNT(*) FROM {_sn_shared(valid_table, _shared_dialect)}")).fetchone()[0] or 0)
            _fc_known = True
        except Exception:
            pass
    if _fc_known:
        try:
            from core.evidence import assess_evidence as _assess_fc
            _ev_fc = _assess_fc(_fc_rows, "forecast", dataset_name=valid_table)
        except Exception:
            _ev_fc = None
        if _ev_fc is not None and _ev_fc.get("status") != "supported":
            from insight_agent import sanitize_json as _sj_fc
            try:
                return _sj_fc({
                    "ok": False,
                    "question": req.question or f"Show {valid_measure} over time",
                    "reason": _ev_fc.get("reason"),
                    "missing_field": "history",
                    "suggestion": "Connect at least 10 usable records, then try again.",
                    "target": valid_measure,
                    "selection": {"measure": valid_measure, "table": valid_table,
                                  "time_dim": None,
                                  "requested_measure": _requested_measure_raw,
                                  "requested_time_dim": ((req.time_dim or "").split(" (")[0]).strip() or None,
                                  "substituted": _measure_substituted,
                                  "substitution_reason": _measure_reason},
                    "info": {"rows_analyzed": _fc_rows, "fallback_used": False,
                             "goal_message": None,
                             "requested_measure": _requested_measure_raw,
                             "resolved_measure": valid_measure,
                             "requested_time_dim": ((req.time_dim or "").split(" (")[0]).strip() or None,
                             "resolved_time_dim": None,
                             "substituted": _measure_substituted,
                             "substitution_reason": _measure_reason,
                             "filters_applied": _filter_applied},
                    "evidence_state": _ev_fc,
                    "reliability": "insufficient_data",
                    "n_obs": _fc_rows,
                    "horizon_info": {"requested": req.horizon, "used": int(req.horizon or 6),
                                     "capped": False, "note": None},
                })
            finally:
                _dispose_shared()

    # Validate the requested time dimension against the table's real
    # temporal candidates (exact-only, no substring); fall back to the
    # best-detected one with substituted=True + reason (fixes silent swaps).
    _tcols = _table_columns(_mapping0.get("tables", {}), valid_table)
    _time_cands = [n for n, d in _tcols
                   if any(k in d.lower() for k in ("date", "time", "timestamp"))
                   or _hint_hit(n, _DATE_NAME_HINTS)]
    _requested_time_raw = ((req.time_dim or "").split(" (")[0]).strip() or None
    _want_t = _requested_time_raw or ""
    valid_time = next((n for n in _time_cands if n.lower() == _want_t.lower()),
                      None) if _want_t else None
    if valid_time is None:
        valid_time = _intel0.find_best_datetime(valid_table)
    if _requested_time_raw and valid_time and _requested_time_raw.lower() == str(valid_time).lower():
        _time_substituted, _time_reason = False, None
    elif _requested_time_raw and valid_time:
        _time_substituted = True
        _time_reason = (
            f"Requested time dimension '{_requested_time_raw}' is not an exact temporal "
            f"column of '{valid_table}'; using '{valid_time}'."
        )
    elif _requested_time_raw and not valid_time:
        _time_substituted, _time_reason = True, (
            f"Requested time dimension '{_requested_time_raw}' is not usable; "
            "auto-detecting the time axis from the data."
        )
    else:
        _time_substituted, _time_reason = False, None
    _substituted = bool(_measure_substituted or _time_substituted)
    _reasons = [r for r in [_measure_reason, _time_reason] if r]
    _substitution_reason = "; ".join(_reasons) if _reasons else None

    # Step 1: Run GoalAgent to get time-series data. When the caller asked
    # for a non-measure column, ask about the validated measure instead —
    # the raw request would only produce garbage SQL. Exact-only: the raw
    # question is kept only when the request exactly names the resolved
    # measure; otherwise the validated question is used (substitution is
    # disclosed via selection/info). Structured filters are appended as
    # natural-language intent so the generated SQL WHERE reflects them,
    # and are additionally enforced in-memory + in deterministic SQL.
    _filter_suffix = _describe_filters_for_question(_filter_applied)
    if req.question and _measure_candidates:
        question = req.question + _filter_suffix
    else:
        question = f"Show {valid_measure} by {valid_time} over time" if valid_time \
            else f"Show {valid_measure} over time"
        question = question + _filter_suffix
    # Perf: pass dicts in-memory; GoalAgent still needs an output_path for
    # its file contract, so use the OS temp dir (never artifacts/ files).
    import tempfile as _tf
    try:
        _tmp_goal = Path(_tf.gettempdir()) / f"aria_forecast_{user['user_id']}_{int(time.time() * 1000)}.json"
    except Exception:
        _tmp_goal = Path(str(schema_path)).parent / f"forecast_input_{int(time.time())}.json"
    output_path = _tmp_goal
    goal_error = None
    goal_result = None
    try:
        from goal_agent.agent import GoalAgent
        from goal_agent.schema import load_user_overrides
        from core.config import resolve_provider_key

        api_key = resolve_provider_key(provider, session.get("api_keys", {}))
        agent = GoalAgent(
            schema_json_path=str(schema_path),
            db_uri=db_uri,
            provider=provider,
            api_key=api_key,
            dialect=dialect,
            overrides=load_user_overrides(user["user_id"], session.get("db_name", "")),
            schemas=session.get("schemas"),
        )
        goal_result = agent.process_goal(question, output_path=str(output_path))
    except Exception as e:
        # No 500: fall through to the deterministic path, then return an
        # honest invalid forecast with the reason if nothing works.
        log.error("GoalAgent failed: %s\n%s", e, _tb.format_exc())
        goal_error = str(e)[:300]

    # Step 2: Run PredictionAgent (dicts in-memory; single shared engine;
    # single deterministic-fallback attempt; GROUP BY grain + LIMIT 2000).
    processed_data: dict = {}
    try:
        from prediction.agent import PredictionAgent

        # In-memory first: GoalAgent's return value when it is a dict;
        # legacy file-contract fallback (temp path only) otherwise.
        processed_data: dict = {}
        try:
            if isinstance(goal_result, dict):
                processed_data = goal_result
            elif isinstance(goal_result, (str, Path)) and Path(str(goal_result)).exists():
                processed_data = json.loads(Path(str(goal_result)).read_text(encoding="utf-8"))
            elif output_path.exists():
                processed_data = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception:
            processed_data = processed_data or {}
        finally:
            try:
                if output_path.exists():
                    output_path.unlink()
            except Exception:
                pass
        # Enforce structured filters on the GoalAgent rows in-memory so the
        # GoalAgent path honors the same WHERE semantics as the fallback SQL.
        try:
            if _filter_applied and isinstance(processed_data.get("data"), list):
                processed_data["data"] = _apply_filters_in_memory(
                    processed_data.get("data") or [], _filter_applied)
        except Exception:
            pass

        # Deterministic fallback: builds the time series directly from the
        # schema — SELECT grain(time), SUM(measure) GROUP BY grain ORDER BY
        # grain — using the validated measure/table + filter WHERE. Fully
        # generic, no hardcoded tables or columns. Reuses the single shared
        # engine/connection; bounded with LIMIT 2000.
        _fallback_attempted = False

        def _deterministic_series():
            try:
                from core.schema_intel import _safe_name as _sn
                from sqlalchemy import text as _t

                if _shared_engine is None:
                    return None
                _dial = _shared_dialect or dialect
                _dt = valid_time or _intel0.find_best_datetime(valid_table)
                if not _dt:
                    return None
                # Real-date detection (generic, dtype-driven like scenario).
                _dtype = ""
                try:
                    for _n, _d in _table_columns(_mapping0.get("tables", {}), valid_table):
                        if _n == _dt:
                            _dtype = str(_d or "").lower()
                            break
                except Exception:
                    _dtype = ""
                _is_date = any(k in _dtype for k in ("date", "time", "timestamp"))
                _fref = _coerced_ref(_mapping0.get("tables", {}), valid_table,
                                     valid_measure, _dial)
                _flt_where, _flt_params = _filter_where_clause(_filter_applied, _dial)
                _extra = f" AND {_flt_where}" if _flt_where else ""
                try:
                    with _shared_engine.connect() as _conn:
                        try:
                            _apply_stmt_timeout(_conn, _dial)
                        except Exception:
                            pass
                        if _is_date:
                            if _dial == "sqlite":
                                _gexpr = f"strftime('%Y-%m', {_sn(_dt, _dial)})"
                            elif _dial == "mysql":
                                _gexpr = f"DATE_FORMAT({_sn(_dt, _dial)}, '%Y-%m')"
                            else:
                                _gexpr = f"TO_CHAR({_sn(_dt, _dial)}, 'YYYY-MM')"
                            _rows = _conn.execute(_t(
                                f"SELECT {_gexpr} AS t, SUM({_fref}) AS v "
                                f"FROM {_sn(valid_table, _dial)} "
                                f"WHERE {_sn(_dt, _dial)} IS NOT NULL{_extra} "
                                f"GROUP BY {_gexpr} "
                                f"ORDER BY {_gexpr} LIMIT {_DETERMINISTIC_LIMIT}"
                            ), _flt_params).fetchall()
                        else:
                            _rows = _conn.execute(_t(
                                f"SELECT {_sn(_dt, _dial)} AS t, SUM({_fref}) AS v "
                                f"FROM {_sn(valid_table, _dial)} "
                                f"WHERE {_sn(_dt, _dial)} IS NOT NULL{_extra} "
                                f"GROUP BY {_sn(_dt, _dial)} "
                                f"ORDER BY {_sn(_dt, _dial)} LIMIT {_DETERMINISTIC_LIMIT}"
                            ), _flt_params).fetchall()
                except Exception as _fe:
                    log.warning("Forecast deterministic fallback failed: %s", _fe)
                    return None
                _series = [{_dt: r[0], valid_measure: float(r[1])}
                           for r in _rows if r[0] is not None and r[1] is not None]
                if len(_series) >= 2:
                    return {
                        "user_goal": question,
                        "data": _series,
                        "sql_used": processed_data.get("sql_used") or processed_data.get("sql"),
                        "fallback": "deterministic time aggregation",
                    }
            except Exception as _fe:
                log.warning("Forecast deterministic fallback failed: %s", _fe)
            return None

        def _try_fallback_once():
            """Single fallback attempt across both call sites."""
            nonlocal processed_data, _fallback_attempted
            if _fallback_attempted:
                return False
            _fallback_attempted = True
            try:
                _fb = _deterministic_series()
            except Exception:
                _fb = None
            finally:
                pass
            if _fb:
                processed_data = _fb
                return True
            return False

        # Exact-only target resolution (no substring match): exact
        # case-sensitive, then case-insensitive exact. No match ->
        # auto-select (numeric_col=None). Multiple case-insensitive hits ->
        # ambiguous: return None + record candidates for disambiguation.
        _target_candidates: list = []

        def _resolve_target(data):
            result_cols = list((data.get("data") or [{}])[0].keys()) \
                if data.get("data") else []
            if valid_measure in result_cols:
                return valid_measure
            _hits = [c for c in result_cols
                     if str(c).lower() == str(valid_measure).lower()]
            if len(_hits) == 1:
                return _hits[0]
            if len(_hits) > 1:
                try:
                    _target_candidates.extend(_hits)
                except Exception:
                    pass
                return None
            return None

        pred_agent = PredictionAgent(
            min_points=5,
            valid_threshold=70.0,
        )
        goal_sql = processed_data.get("sql_used") or processed_data.get("sql")
        if not (processed_data.get("data") or []):
            _try_fallback_once()
        # Pass the validated time dimension through only when it is really a
        # column of the result rows (exact-only) — otherwise the agent
        # auto-detects the time axis from the data (fully dynamic).
        def _resolve_time(data):
            result_cols = list((data.get("data") or [{}])[0].keys()) \
                if data.get("data") else []
            if valid_time in result_cols:
                return valid_time
            if valid_time:
                _hits = [c for c in result_cols
                         if str(c).lower() == str(valid_time).lower()]
                if len(_hits) == 1:
                    return _hits[0]
            return None

        forecast = pred_agent.forecast(
            processed_data,
            numeric_col=_resolve_target(processed_data),
            datetime_col=_resolve_time(processed_data),
            horizon=req.horizon,
        )
        if not forecast.get("valid") and not (forecast.get("points") or []) \
                and not processed_data.get("fallback"):
            if _try_fallback_once():
                forecast = pred_agent.forecast(
                    processed_data,
                    numeric_col=_resolve_target(processed_data),
                    datetime_col=_resolve_time(processed_data),
                    horizon=req.horizon,
                )
    except Exception as e:
        from core.errors import sanitize_error as _san3
        log.error("PredictionAgent failed: %s", _san3(e))
        forecast = {
            "ok": False,
            "valid": False,
            "method": "unavailable",
            "model": "none",
            "model_selected": None,
            "candidates": [],
            "drivers": [],
            "details": {},
            "preparation": {},
            "accuracy": {"pct": None, "summary": "Forecast unavailable."},
            "reason": f"Forecast unavailable right now: {_san3(e)[:200]}",
            "missing_field": "internal",
            "suggestion": "Ask ARIA to analyze this dataset instead.",
            "message": None,
            "recommendation": None,
            "n_obs": 0,
            "reliability": "insufficient_data",
            "forecast_stage": "none",
            "points": [],
            "history": [],
            "historical": [],
            "forecast": [],
            "confidence_interval": [],
            "timestamps": [],
            "frequency": None,
            "target": valid_measure,
            "horizon": int(req.horizon or 6),
            "horizon_suggested": None,
            "horizon_capped": False,
            "horizon_note": None,
            "metrics": {"mae": None, "rmse": None, "mape": None},
            "selection_statement": None,
            "uncertainty_note": None,
            "interval_method": None,
        }

    if not forecast.get("valid") and not (forecast.get("points") or []) \
            and not forecast.get("reason") and goal_error:
        forecast["reason"] = (
            "The data could not be prepared for forecasting right now. "
            "Try again shortly or pick a different number.")
    # Data contract (n.txt item 3): the `forecast` object itself carries the
    # contract keys (historical/forecast/frequency/target/model/horizon/
    # metrics). They are mirrored at the top level for convenience — except
    # `forecast`, which keeps its legacy meaning (the full forecast object);
    # the future series is aliased as `forecast_series` up here.
    from insight_agent import sanitize_json
    # Genuinely unforecastable data: structured graceful JSON (ok:false with
    # reason + missing field + suggestion) — never a fabricated forecast,
    # never a stack trace.
    if forecast.get("ok") is False and not (forecast.get("points") or []):
        try:
            from core.evidence import assess_evidence as _assess_uf
            _ev_uf = _assess_uf(forecast.get("n_obs", 0), "forecast",
                               history_points=forecast.get("n_obs", 0),
                               dataset_name=valid_table)
        except Exception:
            _ev_uf = None
        _uf_selection = {
            "measure": valid_measure,
            "table": valid_table,
            "time_dim": valid_time,
            "requested_measure": _requested_measure_raw,
            "requested_time_dim": _requested_time_raw,
            "substituted": _substituted,
            "substitution_reason": _substitution_reason,
        }
        _uf_info = {
            "rows_analyzed": len((processed_data.get("data") or [])),
            "fallback_used": bool(processed_data.get("fallback")),
            "goal_message": processed_data.get("message"),
            "requested_measure": _requested_measure_raw,
            "resolved_measure": valid_measure,
            "requested_time_dim": _requested_time_raw,
            "resolved_time_dim": valid_time,
            "substituted": _substituted,
            "substitution_reason": _substitution_reason,
            "filters_applied": _filter_applied,
        }
        _uf_payload = {
            "ok": False,
            "question": question,
            "reason": forecast.get("reason") or (
                goal_error or "This dataset cannot be forecast."),
            "missing_field": forecast.get("missing_field"),
            "suggestion": forecast.get("suggestion")
            or "Ask ARIA to analyze this dataset instead.",
            "target": forecast.get("target") or valid_measure,
            "selection": _uf_selection,
            "info": _uf_info,
            "preparation": forecast.get("preparation") or {},
            "reliability": forecast.get("reliability"),
            "n_obs": forecast.get("n_obs", 0),
            "horizon_info": _horizon_info(req.horizon, forecast),
        }
        # Exact-only ambiguity in result columns: surface candidates so the
        # caller can disambiguate (additive; model candidates stay empty here).
        try:
            if _target_candidates:
                _uf_payload["target_candidates"] = list(_target_candidates)
                if not _uf_payload.get("candidates"):
                    _uf_payload["candidates"] = list(_target_candidates)
        except Exception:
            pass
        if _ev_uf is not None:
            _uf_payload["evidence_state"] = _ev_uf
        try:
            _attach_forecast_verdict(_uf_payload)
        except Exception:
            pass
        try:
            return sanitize_json(_uf_payload)
        finally:
            _dispose_shared()
    _future_series = forecast.get("forecast") or []
    if isinstance(_future_series, dict):
        _future_series = []
    contract = {
        "historical": forecast.get("historical") or [],
        "forecast_series": _future_series,
        "confidence_interval": forecast.get("confidence_interval") or [],
        "timestamps": forecast.get("timestamps") or [],
        "frequency": forecast.get("frequency"),
        "target": forecast.get("target") or valid_measure,
        "model": forecast.get("model") or forecast.get("method") or "none",
        "model_selected": forecast.get("model_selected")
        or forecast.get("model") or forecast.get("method") or "none",
        "candidates": forecast.get("candidates") or [],
        "drivers": forecast.get("drivers") or [],
        "accuracy": forecast.get("accuracy"),
        "details": forecast.get("details") or {},
        "preparation": forecast.get("preparation") or {},
        "validation": forecast.get("validation") or {},
        "reliability": forecast.get("reliability"),
        "forecast_stage": forecast.get("forecast_stage"),
        "message": forecast.get("message"),
        "recommendation": forecast.get("recommendation"),
        "n_obs": forecast.get("n_obs", 0),
        "horizon": forecast.get("horizon") or int(req.horizon or 6),
        "horizon_suggested": forecast.get("horizon_suggested"),
        "horizon_capped": forecast.get("horizon_capped") or False,
        "horizon_note": forecast.get("horizon_note"),
        "horizon_info": _horizon_info(req.horizon, forecast),
        "metrics": forecast.get("metrics") or {"mae": None, "rmse": None, "mape": None},
        "selection_statement": forecast.get("selection_statement"),
        "uncertainty_note": forecast.get("uncertainty_note"),
        "interval_method": forecast.get("interval_method")
        or ("pointwise" if (forecast.get("points") or []) else None),
    }
    try:
        _ok_payload = {
            "ok": True,
            "question": question,
            "forecast": forecast,
            "selection": {
                "measure": valid_measure,
                "table": valid_table,
                "time_dim": valid_time,
                "requested_measure": _requested_measure_raw,
                "requested_time_dim": _requested_time_raw,
                "substituted": _substituted,
                "substitution_reason": _substitution_reason,
            },
            "info": {
                "rows_analyzed": len((processed_data.get("data") or [])),
                "fallback_used": bool(processed_data.get("fallback")),
                "goal_message": processed_data.get("message"),
                "requested_measure": _requested_measure_raw,
                "resolved_measure": valid_measure,
                "requested_time_dim": _requested_time_raw,
                "resolved_time_dim": valid_time,
                "substituted": _substituted,
                "substitution_reason": _substitution_reason,
                "filters_applied": _filter_applied,
            },
            "evidence_state": _evidence_state_for_forecast(
                forecast, _fc_rows if _fc_known else None, valid_table),
            "horizon_info": _horizon_info(req.horizon, forecast),
            **contract,
        }
        try:
            _attach_forecast_verdict(_ok_payload)
        except Exception:
            pass
        return sanitize_json(_ok_payload)
    finally:
        _dispose_shared()


@router.post("/api/prescription/generate")
def generate_prescription(req: PrescriptionRequest, request: Request, user: dict = Depends(require_writable)):
    """DEPRECATED — kept for old clients only. Use /api/prescriptions/* or Decision Center instead.

    Legacy duplicate of the prescription path: runs the old GoalAgent ->
    InsightAgent -> PredictionAgent -> PrescriptionAgent chain. New code must
    use the shared prescription_engine (analyze/run_scenario/priority_score/
    quality_gate) via /api/prescriptions/* or the Decision Center
    (/api/decision-center/*), which never calls PrescriptionAgent. DO NOT
    delete this endpoint (old clients still call it)."""
    import traceback as _tb
    session = get_session(user["user_id"])
    reject_document_sql(session)
    schema_path = session.get("schema_path")
    if not schema_path or not Path(str(schema_path)).exists():
        raise HTTPException(status_code=400, detail="No database connected.")

    db_config = dict(session.get("db") or {})
    if not db_config:
        raise HTTPException(status_code=400, detail="No database configuration found.")
    if db_config.get("password"):
        try:
            db_config["password"] = _decrypt_secret(db_config["password"])
        except Exception:
            pass

    db_uri = session.get("db_uri", "")
    db_type = session.get("db_type", "postgresql")
    dialect = "postgresql" if "postgres" in db_type else "mysql" if "mysql" in db_type else "sqlite"
    provider = session.get("provider")

    # Step 1: Run GoalAgent
    try:
        from goal_agent.agent import GoalAgent
        from goal_agent.schema import load_user_overrides
        from core.config import resolve_provider_key

        output_dir = Path("artifacts/processed_data") / f"user_{user['user_id']}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"prescription_input_{int(time.time())}.json"

        api_key = resolve_provider_key(provider, session.get("api_keys", {}))
        agent = GoalAgent(
            schema_json_path=str(schema_path),
            db_uri=db_uri,
            provider=provider,
            api_key=api_key,
            dialect=dialect,
            overrides=load_user_overrides(user["user_id"], session.get("db_name", "")),
            schemas=session.get("schemas"),
        )
        goal_result = agent.process_goal(
            "Analyze the data and identify key metrics, trends, and anomalies",
            output_path=str(output_path),
        )
    except Exception as e:
        from core.errors import sanitize_error as _san2
        raise HTTPException(status_code=500, detail=f"Data preparation failed: {_san2(e)}")

    processed_data = {}
    if output_path.exists():
        processed_data = json.loads(output_path.read_text(encoding="utf-8"))

    # Step 2: Run InsightAgent for KPIs, trends, anomalies
    kpis = []
    trends = []
    anomalies = []
    hypotheses = []
    try:
        # Canonical insight path: insight_agent.agent.InsightAgent only.
        # Advanced/Industry variants (insight_agent/advanced.py,
        # insight_agent/industry.py, repo-root insight_agent_*.py) are
        # intentionally NOT in the live path and stay unreachable but intact.
        from insight_agent.agent import InsightAgent

        insight_agent = InsightAgent(provider=provider)
        insights = insight_agent.analyze(
            str(output_path),  # file path, not dict
            output_path=str(output_dir / f"prescription_insights_{int(time.time())}.json"),
        )
        kpis = insights.get("kpis", [])
        trends = insights.get("trends", [])
        anomalies = insights.get("anomalies", [])
        hypotheses = insights.get("hypotheses", [])
    except Exception as e:
        log.warning("InsightAgent failed: %s", e)

    # Step 3: Run PredictionAgent for forecast
    predictions = None
    try:
        from prediction.agent import PredictionAgent

        pred_agent = PredictionAgent(horizon=6)
        predictions = pred_agent.forecast(processed_data)
    except Exception as e:
        log.warning("PredictionAgent failed: %s", e)

    # Step 4: Run PrescriptionAgent
    try:
        from prescription_agent import PrescriptionAgent

        rx_agent = PrescriptionAgent(provider=provider)
        prescriptions = rx_agent.prescribe(
            processed_data, kpis, trends, anomalies, hypotheses,
            predictions=predictions,
        )
    except Exception as e:
        log.warning("PrescriptionAgent failed: %s", e)
        prescriptions = {
            "actions": [
                "Review the key metrics identified in the analysis",
                "Investigate any anomalies or unusual patterns detected",
                "Monitor the trends highlighted in the data",
            ],
            "source": "template",
            "impact_note": "Impact could not be reliably estimated from the available data.",
        }

    return {
        "ok": True,
        "prescriptions": prescriptions,
        "context": {
            "kpis_count": len(kpis),
            "trends_count": len(trends),
            "anomalies_count": len(anomalies),
            "has_forecast": predictions is not None and predictions.get("valid", False),
        },
    }
