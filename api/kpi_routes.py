"""Dashboard KPI routes — fully dynamic, database-agnostic.

Generates KPIs and charts from ANY connected database using schema metadata.
Zero hardcoded table/column names. Uses SchemaIntelligence for all decisions.
"""

import json
import logging
import sqlite3
import os
import re
import types
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text

from core.config import get_session, SCHEMA_DIR
from core.deps import require_data, get_current_user, reject_document_sql
from core.errors import DB_ERRORS, PURE_ERRORS, get_request_id, log_and_sanitize
from core.secret_box import decrypt as _decrypt_secret
from core.semantic import humanize, kpi_label
from core.schema_intel import SchemaIntelligence, _safe_name, _run_sql

router = APIRouter()

# Short-TTL KPI cache per dataset (schema-mtime key, 300s, cap 100).
# KPI detection runs full-table aggregates; repeat dashboard views must
# not recompute. Errors/empties are never cached.
_KPI_CACHE: dict = {}
_KPI_CACHE_TTL = 300
_KPI_CACHE_MAX = 100
from services.kpi_service import (_fmt, _is_currency_col, _safe, _validate_schema_ident, _bound_limit, _recommend_chart, _profile_numeric, _rollback, _variation, _score_kpi, _time_grain_expr, _parse_full_date, _parse_date_to_month, _pretty_month, _pretty_date, _discover_special_columns, _fetch_time_series, _fetch_count_series, _holiday_lift, _retail_candidates_for_table, _measure_expr, _coerce_num, _SERIES_FALLBACK_MAX, _CHART_BY_INTENT, DASHBOARD_MIN_SCORE, _AMOUNT_HINTS, _ENTITY_HINTS, _DATE_HINTS, _STORE_HINTS, _HOLIDAY_HINTS, _SALES_HINTS, _DATE_PARSE_FORMATS, _MONTH_NAMES)  # noqa: F401,E402  # thin-wrapper re-export


# ── Helpers ──────────────────────────────────────────────────────────────

def _sanitize_json_floats(value):
    """Recursively replace non-finite floats (NaN/Infinity) with None.

    Statistical code paths (z-scores, 0/0 shares, log(0)) can produce
    non-finite floats on unseen data shapes; Starlette raises
    "Out of range float values are not JSON compliant" and the endpoint
    500s. Sanitizing at the response boundary keeps every dashboard
    endpoint total. Tuples become lists (JSON has no tuples).
    """
    try:
        import math as _math
    except Exception:
        return value
    try:
        if isinstance(value, float):
            return value if _math.isfinite(value) else None
        if isinstance(value, dict):
            return {k: _sanitize_json_floats(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_sanitize_json_floats(v) for v in value]
    except Exception:
        return None
    return value


def _attach_verdict(config_dict, session, cached=False, cache_age_s=0.0):
    """Best-effort Verification Agent hookup (never blocks the dashboard).

    Attaches ``config_dict["verdict"]`` plus a ``_meta`` freshness envelope
    (``cached``/``cache_age_s``/``ttl_s``) that the agent reads generically.
    On any failure attaches ``{verified: False}`` and still returns data.
    """
    try:
        import os as _os
        from verification.agent import verify_dashboard
        try:
            _ttl = float(_os.getenv("ARIA_DASH_CACHE_TTL", "120"))
        except (TypeError, ValueError):
            _ttl = 120.0
        try:
            _meta = dict(config_dict.get("_meta") or {})
        except Exception:
            _meta = {}
        _meta.update({"cached": bool(cached),
                      "cache_age_s": float(cache_age_s or 0.0),
                      "ttl_s": _ttl})
        try:
            config_dict["_meta"] = _meta
        except Exception:
            pass
        verdict = verify_dashboard(config_dict, dict(session or {}))
    except Exception:
        verdict = {"verified": False, "checks": [],
                   "warnings": ["verification unavailable"]}
    try:
        config_dict["verdict"] = verdict
    except Exception:
        pass
    return config_dict


def _build_engine(session):
    """Create a SQLAlchemy engine for the session's DB."""
    db_config = dict(session.get("db") or {})
    db_type = session.get("db_type", "postgresql")

    if not db_config or session.get("source_type") != "relational":
        sqlite_path = session.get("db_uri") or session.get("csv_db_path")
        if sqlite_path:
            if sqlite_path.startswith("sqlite:///"):
                uri = sqlite_path
            elif sqlite_path.startswith("sqlite://"):
                uri = sqlite_path
            else:
                uri = f"sqlite:///{sqlite_path}"
            return create_engine(uri), "sqlite", "sqlite"
        return None, "unknown", "unknown"

    if db_config.get("password"):
        try:
            db_config["password"] = _decrypt_secret(db_config["password"])
        except (ValueError, KeyError, TypeError) as exc:
            logging.getLogger("aria.kpi").debug("kpi secret decrypt skipped: %s", exc)

    schema = db_config.get("db_schema", "public")
    # Multi-schema databases: resolve via every mapped schema so bare
    # table names work on all engines (session["schemas"] is authoritative;
    # db_schema stays the single-schema fallback). Generic, no names.
    try:
        _schemas = [str(s).strip() for s in (session.get("schemas") or [])
                    if str(s or "").strip()] or [schema]
    except Exception:
        _schemas = [schema]
    # Ensure the discovered/actual schema is included even when session
    # schemas were not updated (legacy sessions with stale "public").
    # db_schema may be comma-separated (e.g. "sales,production,humanresources")
    # from _discover_table_schema — split into individual names.
    if schema:
        for _s in schema.split(","):
            _s = _s.strip()
            if _s and _s not in _schemas:
                _schemas.insert(0, _s)
    # Required connection fields via .get(): incomplete mappings return
    # (None, ...) so callers show "not connected" instead of 500 KeyError.
    _user = db_config.get("user")
    _pw = db_config.get("password")
    _host = db_config.get("host")
    _db = db_config.get("db")
    if not all([_user, _pw, _host, _db]):
        return None, db_type, db_type
    if db_type == "mysql":
        uri = (f"mysql+pymysql://{quote_plus(str(_user))}:{quote_plus(str(_pw))}"
               f"@{_host}:{db_config.get('port', 3306)}/{_db}")
        return create_engine(uri, pool_pre_ping=True,
                             connect_args={"connect_timeout": 10}), db_type, db_type
    else:
        from urllib.parse import quote as _uq
        _sp = ",".join(_schemas + (["public"] if "public" not in _schemas else []))
        uri = (f"postgresql://{quote_plus(str(_user))}:{quote_plus(str(_pw))}"
               f"@{_host}:{db_config.get('port', 5432)}/{_db}"
               f"?options=-c%20search_path%3D{_uq(_sp, safe=',')}")

    return create_engine(uri, pool_pre_ping=True,
                         connect_args={"connect_timeout": 10}), db_type, db_type


def _load_schema(session, user_id=None):
    """Load the schema mapping and build SchemaIntelligence."""
    schema_path = session.get("schema_path")
    # Try user-specific DB schema file first (set by connect_routes)
    if schema_path and Path(str(schema_path)).exists():
        mapping = json.loads(Path(str(schema_path)).read_text(encoding="utf-8"))
        return SchemaIntelligence(mapping)
    # NOTE: no global SCHEMA_DIR/schema_mapping_latest.json fallback here.
    # A shared default would leak one user's tables into another user's
    # session — per-user paths only (session schema_path, then user dir).
    from core.config import SCHEMA_DIR
    # Try user-specific file name using DB name (connect_routes naming convention)
    db_name = session.get("db_name", "")
    if db_name:
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", db_name)[:64] or "db"
        user_dir = f"user_{user_id}"
        alt_path = SCHEMA_DIR / user_dir / f"schema_mapping_{safe_name}_latest.json"
        if alt_path.exists():
            mapping = json.loads(alt_path.read_text(encoding="utf-8"))
            return SchemaIntelligence(mapping)
        # Also try the user's directory without DB name suffix
        user_default = SCHEMA_DIR / user_dir / "schema_mapping_latest.json"
        if user_default.exists():
            mapping = json.loads(user_default.read_text(encoding="utf-8"))
            return SchemaIntelligence(mapping)
    # Last resort: try to trigger auto-extraction from the session DB info.
    # Canonical stack is schema_engine/; schema_agent/ is legacy and used here
    # only as a backward-compat fallback (new code must NOT import it).
    if session.get("db") and session.get("db_type") in ("postgresql", "mysql"):
        try:
            import warnings as _warnings
            import schema_agent  # noqa: DEPRECATED legacy fallback only
            _warnings.warn(
                "kpi_routes legacy schema_agent fallback in use; prefer schema_engine.",
                DeprecationWarning, stacklevel=2)
            config = dict(session.get("db", {}))
            if "password" in config and config.get("password"):
                try:
                    config["password"] = _decrypt_secret(config["password"])
                except Exception:
                    pass
            conn = schema_agent.get_connection(types.SimpleNamespace(**config))
            try:
                db_type = session.get("db_type", "postgresql")
                schema_filter = config.get("db_schema", "public") if db_type == "postgresql" else config.get("db", "")
                mapping = schema_agent.build_schema_mapping(conn, schema_filter, db_type=db_type)
                conn.close()
                if mapping.get("tables"):
                    # Save to user-specific path for future requests
                    user_dir_path = SCHEMA_DIR / f"user_{user_id}"
                    user_dir_path.mkdir(parents=True, exist_ok=True)
                    safe_db = re.sub(r"[^a-zA-Z0-9_-]+", "_", db_name)[:64] or "db"
                    user_path = user_dir_path / f"schema_mapping_{safe_db}_latest.json"
                    user_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
                    session["schema_path"] = user_path
                    from core.config import save_session
                    # We can't save here easily without user_id — skip persistence
                    return SchemaIntelligence(mapping)
            except Exception:
                pass
        except Exception:
            pass
    # Final fallback: read schema directly from SQLite DB file (CSV uploads, broken sessions)
    sqlite_path_str = session.get("db_uri") or session.get("csv_db_path")
    if sqlite_path_str and isinstance(sqlite_path_str, (str, Path)):
        sqlite_path_str = str(sqlite_path_str)
        if sqlite_path_str.startswith("sqlite:///"):
            sqlite_file = sqlite_path_str[10:]
        elif sqlite_path_str.startswith("sqlite://"):
            sqlite_file = sqlite_path_str[9:]
        else:
            sqlite_file = sqlite_path_str
        try:
            sqlite_path_obj = Path(sqlite_file)
            if sqlite_path_obj.exists():
                import sqlite3
                conn_sqlite = sqlite3.connect(str(sqlite_path_obj))
                cursor_sqlite = conn_sqlite.cursor()
                cursor_sqlite.execute("SELECT name FROM sqlite_master WHERE type='table';")
                table_rows = cursor_sqlite.fetchall()
                conn_sqlite.close()
                tables_mapping = {}
                for (tname,) in table_rows:
                    # Get column info from PRAGMA
                    try:
                        conn2 = sqlite3.connect(str(sqlite_path_obj))
                        cur2 = conn2.cursor()
                        cur2.execute(f'PRAGMA table_info("{tname}")')
                        col_rows = cur2.fetchall()
                        conn2.close()
                        columns = []
                        inferred_pk = []
                        for row in col_rows:
                            cid, cname, ctype, notnull, dflt, pk = row
                            columns.append({
                                "column": cname,
                                "data_type": ctype or "TEXT",
                                "nullable": not (notnull == 1),
                            })
                            if pk == 1:
                                inferred_pk.append(cname)
                        tables_mapping[tname] = {
                            "columns": columns,
                            "primary_key": inferred_pk,
                            "inferred_primary_key": inferred_pk,
                            "description": tname,
                        }
                    except Exception:
                        pass
                if tables_mapping:
                    mapping = {
                        "database": session.get("db_name") or Path(sqlite_file).stem,
                        "tables": tables_mapping,
                        "declared_relationships": [],
                        "inferred_relationships": [],
                    }
                    return SchemaIntelligence(mapping)
        except Exception:
            pass
    return None


# ── Generic KPI endpoint (works for ANY database) ────────────────────────

@router.get("/api/dashboard/kpis")
def get_dashboard_kpis(request: Request, user: dict = Depends(get_current_user)):
    """Return real KPI metrics from any connected database."""
    session = get_session(user["user_id"])
    reject_document_sql(session)
    intel = _load_schema(session, user["user_id"])
    if not intel or not intel.tables:
        return {"ok": True, "kpis": [], "charts": [], "summary": {}}

    # Short-TTL cache per dataset (schema mtime key): KPI detection runs
    # full-table aggregates, so repeat views must not recompute. Errors and
    # empty results are never cached.
    try:
        import os as _os, time as _time
        _sp = session.get("schema_path")
        _ss = str(_sp) if _sp is not None else ""
        _stag = f"{_ss}:{_os.path.getmtime(_ss)}" if _ss and _os.path.exists(_ss) else _ss
        _ckey = (user["user_id"], _stag)
        _hit = _KPI_CACHE.get(_ckey)
        if _hit is not None and _time.time() - _hit[0] < _KPI_CACHE_TTL:
            return _hit[1]
    except Exception:
        _ckey = None

    engine, dialect, db_type = _build_engine(session)
    if engine is None:
        return {"ok": True, "kpis": [], "charts": [], "summary": {
            "source_type": session.get("source_type", "file"),
            "message": "KPIs are not available for this source yet",
        }}

    kpis = []
    charts = []

    try:
        with engine.connect() as conn:
            # 1. Total records across all tables
            total = 0
            for tname in list(intel.tables.keys())[:14]:
                try:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {_safe_name(tname, dialect)}"))
                    row = result.fetchone()
                    if row:
                        total += row[0]
                except DB_ERRORS as exc:
                    logging.getLogger("aria.kpi").debug("kpi total-records skipped for %s: %s", tname, exc)
                    continue
            if total > 0:
                kpis.append({
                    "label": "Total Records",
                    "value": _fmt(total),
                    "rawValue": total,
                    "change": None,
                    "prefix": "",
                    "period": "All data",
                })

            # 2. Per-table measures: find the best measure per table
            tables_with_measures = []
            for tname in list(intel.tables.keys())[:10]:
                if intel.tables[tname].get("empty"):
                    continue
                measure = intel.find_best_measure(tname)
                if not measure:
                    continue
                tables_with_measures.append((tname, measure))

            for tname, measure in tables_with_measures[:5]:
                is_curr = _is_currency_col(measure)
                try:
                    result = conn.execute(text(
                        f"SELECT SUM({_measure_expr(intel, tname, measure, dialect)}) FROM {_safe_name(tname, dialect)}"
                    ))
                    row = result.fetchone()
                    if row and row[0] is not None:  # noqa: SIM108 (typed guard below)
                        val = float(row[0])
                        tdesc = intel.tables[tname].get("description", tname)
                        kpis.append({
                            "label": f"{measure} ({tdesc[:20]})",
                            "value": _fmt(val, is_curr),
                            "rawValue": val,
                            "change": None,
                            "prefix": "$" if is_curr else "",
                            "period": "Total",
                        })
                except Exception:
                    continue

            # 3. Per-table averages
            for tname, measure in tables_with_measures[:3]:
                is_curr = _is_currency_col(measure)
                try:
                    result = conn.execute(text(
                        f"SELECT AVG({_measure_expr(intel, tname, measure, dialect)}) FROM {_safe_name(tname, dialect)}"
                    ))
                    row = result.fetchone()
                    if row and row[0] is not None:
                        val = float(row[0])
                        if len(kpis) < 8:
                            kpis.append({
                                "label": f"Avg {measure}",
                                "value": _fmt(val, is_curr),
                                "rawValue": val,
                                "change": None,
                                "prefix": "$" if is_curr else "",
                                "period": "Average",
                            })
                except Exception:
                    continue

            # 3b. Retail-aware KPIs from real execution (single-table safe).
            # Surfaces store count, date range and holiday lift, and attaches
            # real spark series to the sales total/average. Never mocked:
            # every number comes from executed SQL on the actual table.
            try:
                _have_labels = {str(k.get("label", "")).lower() for k in kpis}
                for tname in list(intel.tables.keys())[:3]:
                    if intel.tables[tname].get("empty"):
                        continue
                    for cand in _retail_candidates_for_table(conn, intel, tname, dialect):
                        agg, col = cand["agg"], cand["column"]
                        if agg in ("SUM", "AVG") and "__" not in col:
                            # Enrich the matching generic KPI with a spark
                            # instead of duplicating it.
                            for k in kpis:
                                if (col.lower() in str(k.get("label", "")).lower()
                                        and k.get("period") == ("Total" if agg == "SUM" else "Average")
                                        and not k.get("spark") and cand.get("spark")):
                                    k["spark"] = cand["spark"]
                                    if cand.get("trend"):
                                        k["trend"] = cand["trend"]
                            continue
                        if agg == "DISTINCT":
                            label = kpi_label("DISTINCT", col)
                        elif agg == "RANGE":
                            label = str(cand.get("label_override") or "Date Range")
                        elif agg == "LIFT":
                            label = str(cand.get("label_override") or "Holiday Lift")
                        else:
                            continue
                        if label.lower() in _have_labels:
                            continue
                        if agg == "RANGE":
                            kpis.append({
                                "label": label,
                                "value": cand.get("value_override") or _fmt(cand["value"]),
                                "rawValue": cand["value"],
                                "change": None,
                                "prefix": "",
                                "period": cand.get("period_override") or "All data",
                                "format": "text",
                            })
                        elif agg == "LIFT":
                            kpis.append({
                                "label": label,
                                "value": f"{cand['value']:+.1f}%",
                                "rawValue": cand["value"],
                                "change": cand.get("change_override", cand["value"]),
                                "prefix": "",
                                "period": cand.get("period_override") or "Holiday vs non-holiday",
                                "format": "percent",
                                "spark": cand.get("spark") or [],
                            })
                        else:
                            kpis.append({
                                "label": label,
                                "value": _fmt(cand["value"]),
                                "rawValue": cand["value"],
                                "change": None,
                                "prefix": "",
                                "period": "All data",
                                "spark": cand.get("spark") or [],
                            })
                        _have_labels.add(label.lower())
                        if len(kpis) >= 8:
                            break
                    if len(kpis) >= 8:
                        break
            except Exception:
                pass

            # 4. Time-series chart: find a table with datetime + measure
            for tname in list(intel.tables.keys())[:14]:
                if intel.tables[tname].get("empty"):
                    continue
                dt_col = intel.find_best_datetime(tname)
                measure = intel.find_best_measure(tname)
                if not dt_col or not measure:
                    continue

                try:
                    # Grain-aware series from real SQL; falls back to raw
                    # pairs bucketed in Python for non-ISO date text.
                    _grain, chart_data = _fetch_time_series(
                        conn, tname, dt_col, measure, dialect, limit=30, intel=intel)
                    if chart_data:
                        tdesc = intel.tables[tname].get("description", tname)
                        is_curr = _is_currency_col(measure)
                        charts.append({
                            "title": f"{measure} over time ({tdesc[:15]})",
                            "type": "area",
                            "data": chart_data,
                            "dataKey": "value",
                            "xKey": "period",
                            "format": "currency" if is_curr else "number",
                        })
                        if len(charts) >= 3:
                            break
                except Exception:
                    continue

            # 5. Categorical chart: find a table with a good dimension + measure
            if len(charts) < 3:
                for tname in list(intel.tables.keys())[:14]:
                    if intel.tables[tname].get("empty"):
                        continue
                    dim = intel.find_best_dimension(tname)
                    measure = intel.find_best_measure(tname)
                    if not dim or not measure:
                        continue
                    # Skip if we already have a chart for this table
                    if any(tname in str(c.get("title", "")) for c in charts):
                        continue

                    try:
                        result = conn.execute(text(
                            f"SELECT {_safe_name(dim, dialect)} AS dim_val, "
                            f"SUM({_measure_expr(intel, tname, measure, dialect)}) AS val "
                            f"FROM {_safe_name(tname, dialect)} "
                            f"WHERE {_safe_name(dim, dialect)} IS NOT NULL "
                            f"GROUP BY dim_val ORDER BY val DESC LIMIT 8"
                        ))
                        rows = result.fetchall()
                        if rows:
                            chart_data = [{"name": str(r[0]), "value": float(r[1])} for r in rows if r[1] is not None]
                            if chart_data:
                                tdesc = intel.tables[tname].get("description", tname)
                                is_curr = _is_currency_col(measure)
                                charts.append({
                                    "title": f"{measure} by {dim} ({tdesc[:15]})",
                                    "type": "bar",
                                    "data": chart_data,
                                    "dataKey": "value",
                                    "xKey": "name",
                                    "format": "currency" if is_curr else "number",
                                })
                                if len(charts) >= 3:
                                    break
                    except Exception:
                        continue

    except DB_ERRORS as exc:
        logging.getLogger("aria.kpi").exception(
            "dashboard kpis query phase failed request_id=%s: %s",
            get_request_id(request), exc)

    # Last-resort fallback: a connected 1-table source must never return [].
    # Per-table row counts from real SQL so KPIs are always present on data.
    if not kpis:
        try:
            if engine is not None:
                with engine.connect() as _conn:
                    for tname in list(intel.tables.keys())[:5]:
                        try:
                            _r = _conn.execute(text(
                                f"SELECT COUNT(*) FROM {_safe_name(tname, dialect)}")).fetchone()
                            if _r and _r[0]:
                                _tdesc = intel.tables[tname].get("description", tname)
                                kpis.append({
                                    "label": f"{_tdesc} Records",
                                    "value": _fmt(int(_r[0])),
                                    "rawValue": int(_r[0]),
                                    "change": None,
                                    "prefix": "",
                                    "period": "All data",
                                })
                                if len(kpis) >= 5:
                                    break
                        except DB_ERRORS as exc:
                            logging.getLogger("aria.kpi").debug(
                                "kpi fallback row-count skipped for %s: %s", tname, exc)
                            try:
                                _conn.rollback()
                            except DB_ERRORS as rb_exc:
                                logging.getLogger("aria.kpi").debug(
                                    "kpi fallback rollback skipped: %s", rb_exc)
                            continue
        except Exception:
            pass

    # Summary
    db_name = session.get("db_name", "")
    summary = {
        "source_type": "relational",
        "database": db_name,
        "tables_used": len(intel.tables),
        "message": None,
    }

    # Generate smart recommendations based on schema. All user-facing
    # wording goes through humanize so raw snake_case / schema-qualified
    # names never leak into suggested questions (generic, no literals).
    def _qdesc(name):
        try:
            return humanize(str(name or "").split(".")[-1])
        except Exception:
            return humanize(str(name or ""))
    recommendations = []
    for tname in list(intel.tables.keys())[:5]:
        tdesc = intel.tables[tname].get("description", tname)
        measure = intel.find_best_measure(tname)
        dim = intel.find_best_dimension(tname)
        dt = intel.find_best_datetime(tname)
        if measure and dim:
            recommendations.append({"q": f"What are the top {_qdesc(dim)} by {_qdesc(measure)}?"})
        if measure and dt:
            recommendations.append({"q": f"How has {_qdesc(measure)} changed over time?"})
        if measure and dim and len(recommendations) < 6:
            recommendations.append({"q": f"Show {_qdesc(measure)} breakdown by {_qdesc(dim)}"})

    out = {"ok": True, "kpis": kpis[:8], "charts": charts[:3], "summary": summary, "recommendations": recommendations[:6]}
    # Same JSON-safety boundary as get_dashboard_config (see helper above).
    out = _sanitize_json_floats(out)
    try:
        if _ckey is not None and out.get("kpis"):
            import time as _t2
            if len(_KPI_CACHE) >= _KPI_CACHE_MAX:
                _KPI_CACHE.pop(next(iter(_KPI_CACHE)))
            _KPI_CACHE[_ckey] = (_t2.time(), out)
    except Exception:
        pass
    return out


# ── Real KPI endpoint (same route, enhanced logic) ───────────────────────

@router.get("/api/dashboard/real_kpis")
def get_real_kpis(request: Request, user: dict = Depends(get_current_user)):
    """Alias for /api/dashboard/kpis — uses the same dynamic engine."""
    return get_dashboard_kpis(request, user)


# ── Dashboard summary (wraps KPIs + recommendations) ─────────────────────

@router.get("/api/dashboard/summary")
def get_dashboard_summary(request: Request, user: dict = Depends(get_current_user)):
    """Return KPI summary with smart recommendations."""
    session = get_session(user["user_id"])
    reject_document_sql(session)
    intel = _load_schema(session, user["user_id"])
    if not intel or not intel.tables:
        return {"ok": True, "kpis": [], "charts": [], "recommendations": []}

    kpi_result = get_dashboard_kpis(request, user)
    kpis = kpi_result.get("kpis", [])
    charts = kpi_result.get("charts", [])

    # Generate recommendations based on what tables exist. Wording is
    # humanized (see above): raw identifiers never reach the user.
    def _qdesc2(name):
        try:
            return humanize(str(name or "").split(".")[-1])
        except Exception:
            return humanize(str(name or ""))
    recommendations = []
    fact_tables = [t for t in intel.tables if intel.is_fact_table(t)]
    dim_tables = [t for t in intel.tables if intel.is_dimension_table(t)]

    for tname in fact_tables[:3]:
        tdesc = _qdesc2(intel.tables[tname].get("description", tname))
        measure = intel.find_best_measure(tname)
        dim = intel.find_best_dimension(tname)
        dt = intel.find_best_datetime(tname)
        if measure and dim:
            recommendations.append(f"What are the top {_qdesc2(dim)} by {_qdesc2(measure)} in {tdesc}?")
        if measure and dt:
            recommendations.append(f"How has {_qdesc2(measure)} changed over time in {tdesc}?")

    for tname in dim_tables[:2]:
        tdesc = _qdesc2(intel.tables[tname].get("description", tname))
        recommendations.append(f"Show all records in {tdesc}")

    return {
        "ok": True,
        "kpis": kpis,
        "charts": charts,
        "recommendations": recommendations[:6],
    }


# ── Dashboard section summaries (one per analytical page) ────────────────

@router.get("/api/dashboard/summaries")
def get_dashboard_summaries(request: Request, user: dict = Depends(get_current_user)):
    """Return lightweight summaries for each Dashboard section (AI Analytics,
    Insights, Predictive, Prescription, Hypothesis). Each summary is generated
    from the actual connected data using schema intelligence + quick queries."""
    session = get_session(user["user_id"])
    reject_document_sql(session)
    intel = _load_schema(session, user["user_id"])
    if not intel or not intel.tables:
        return {"ok": True, "sections": []}

    engine, dialect, db_type = _build_engine(session)
    if engine is None:
        return {"ok": True, "sections": []}

    sections = []

    try:
        with engine.connect() as conn:
            # --- AI Analytics summary: top metric + trend ---
            for tname in list(intel.tables.keys()):
                measure = intel.find_best_measure(tname)
                if not measure:
                    continue
                tdesc = intel.tables[tname].get("description", tname)
                is_curr = _is_currency_col(measure)

                # Get total
                try:
                    result = conn.execute(text(
                        f"SELECT SUM({_measure_expr(intel, tname, measure, dialect)}) FROM {_safe_name(tname, dialect)}"
                    ))
                    row = result.fetchone()
                    total = float(row[0]) if row and row[0] is not None else None
                except Exception:
                    total = None

                if total is not None:
                    sections.append({
                        "type": "analytics",
                        "title": f"{measure} Overview",
                        "metric": _fmt(total, is_curr),
                        "description": f"Total {measure} across {tdesc}: {_fmt(total, is_curr)}. Use AI Analytics to explore trends, breakdowns, and detailed analysis.",
                        "query": f"What is the total {measure}?",
                    })
                    break

            # --- Insights summary: top anomaly or trend ---
            for tname in list(intel.tables.keys()):
                measure = intel.find_best_measure(tname)
                dt = intel.find_best_datetime(tname)
                if not measure or not dt:
                    continue
                tdesc = intel.tables[tname].get("description", tname)
                try:
                    if dialect == "sqlite":
                        time_expr = f"strftime('%Y-%m', {_safe_name(dt, dialect)})"
                    elif dialect == "mysql":
                        time_expr = f"DATE_FORMAT({_safe_name(dt, dialect)}, '%Y-%m')"
                    else:
                        time_expr = f"TO_CHAR({_safe_name(dt, dialect)}, 'YYYY-MM')"
                    result = conn.execute(text(
                        f"SELECT {time_expr} AS period, SUM({_measure_expr(intel, tname, measure, dialect)}) AS val "
                        f"FROM {_safe_name(tname, dialect)} "
                        f"WHERE {_safe_name(dt, dialect)} IS NOT NULL "
                        f"GROUP BY period ORDER BY period DESC LIMIT 6"
                    ))
                    rows = result.fetchall()
                    if not rows:
                        # TEXT-stored dates the bucket function cannot parse
                        # (NULL periods): fall back to raw-value grouping so
                        # month-shaped TEXT ('2024-01') still trends.
                        result = conn.execute(text(
                            f"SELECT {_safe_name(dt, dialect)} AS period, SUM({_measure_expr(intel, tname, measure, dialect)}) AS val "
                            f"FROM {_safe_name(tname, dialect)} "
                            f"WHERE {_safe_name(dt, dialect)} IS NOT NULL "
                            f"GROUP BY period ORDER BY period DESC LIMIT 6"
                        ))
                        rows = result.fetchall()
                    if len(rows) >= 2:
                        vals = [(str(r[0]), float(r[1])) for r in rows if r[1] is not None]
                        if len(vals) >= 2:
                            curr, prev = vals[0], vals[1]
                            if prev != 0:
                                change = ((curr[1] - prev[1]) / abs(prev[1])) * 100
                                direction = "increased" if change > 0 else "decreased"
                                sections.append({
                                    "type": "insights",
                                    "title": f"{measure} Trend Detected",
                                    "metric": f"{change:+.1f}%",
                                    "description": f"{measure} in {tdesc} {direction} by {abs(change):.1f}% from {prev[0]} to {curr[0]}. Insights page has detailed analysis of all patterns.",
                                    "query": f"What are the trends in {measure}?",
                                })
                                break
                except Exception:
                    continue

            # --- Predictive summary: time-series availability ---
            for tname in list(intel.tables.keys()):
                measure = intel.find_best_measure(tname)
                dt = intel.find_best_datetime(tname)
                if not measure or not dt:
                    continue
                tdesc = intel.tables[tname].get("description", tname)
                try:
                    if dialect == "sqlite":
                        months_expr = f"strftime('%Y-%m', {_safe_name(dt, dialect)})"
                    elif dialect == "mysql":
                        months_expr = f"DATE_FORMAT({_safe_name(dt, dialect)}, '%Y-%m')"
                    else:
                        months_expr = f"TO_CHAR({_safe_name(dt, dialect)}, 'YYYY-MM')"
                    result = conn.execute(text(
                        f"SELECT COUNT(DISTINCT {months_expr}) "
                        f"FROM {_safe_name(tname, dialect)} "
                        f"WHERE {_safe_name(dt, dialect)} IS NOT NULL"
                    ))
                    row = result.fetchone()
                    months = int(row[0]) if row and row[0] else 0
                    if months >= 3:
                        sections.append({
                            "type": "predictive",
                            "title": f"{measure} Forecast Available",
                            "metric": f"{months} months of data",
                            "description": f"Connected data has {months} months of {measure} history in {tdesc}. Predictive Analysis can generate forecasts with confidence intervals.",
                            "query": f"Forecast {measure} over time",
                        })
                        break
                except Exception:
                    continue

            # --- Prescription summary ---
            for tname in list(intel.tables.keys()):
                measure = intel.find_best_measure(tname)
                dim = intel.find_best_dimension(tname)
                if not measure or not dim:
                    continue
                tdesc = intel.tables[tname].get("description", tname)
                try:
                    result = conn.execute(text(
                        f"SELECT {_safe_name(dim, dialect)} AS d, SUM({_measure_expr(intel, tname, measure, dialect)}) AS v "
                        f"FROM {_safe_name(tname, dialect)} "
                        f"WHERE {_safe_name(dim, dialect)} IS NOT NULL "
                        f"GROUP BY d ORDER BY v DESC LIMIT 5"
                    ))
                    rows = result.fetchall()
                    if len(rows) >= 2:
                        top = rows[0]
                        sections.append({
                            "type": "prescription",
                            "title": f"Top {dim}: {top[0]}",
                            "metric": _fmt(float(top[1]), _is_currency_col(measure)),
                            "description": f"The leading {dim} in {tdesc} is '{top[0]}'. Prescription page can generate evidence-based recommendations.",
                            "query": f"What actions should I take based on {measure}?",
                        })
                        break
                except Exception:
                    continue

            # --- Hypothesis summary: numeric pairs available ---
            has_numeric_pairs = False
            for tname in list(intel.tables.keys()):
                numerics = intel.get_numeric_cols(tname)
                non_id = [c for c in numerics if not c.lower().endswith("_id") and c.lower() != "id"]
                if len(non_id) >= 2:
                    tdesc = intel.tables[tname].get("description", tname)
                    sections.append({
                        "type": "hypothesis",
                        "title": f"Statistical Testing Available",
                        "metric": f"{len(non_id)} numeric measures",
                        "description": f"{tdesc} has {len(non_id)} numeric measures suitable for correlation and hypothesis testing.",
                        "query": "Is there a correlation between numeric measures?",
                    })
                    has_numeric_pairs = True
                    break

    except Exception:
        pass

    return {"ok": True, "sections": sections[:5]}


# ── Unified dashboard config (backend-driven, frontend renders) ───────────

@router.get("/api/dashboard/config")
def get_dashboard_config(request: Request, user: dict = Depends(get_current_user)):
    """Return a structured dashboard configuration generated from the connected data.

    The frontend renders whatever this endpoint returns — zero business logic
    on the client side. This is the single source of truth for the dashboard.
    """
    import time as _time

    session = get_session(user["user_id"])
    intel = _load_schema(session, user["user_id"])
    if not intel or not intel.tables:
        _empty_config = {
                "summary": {"title": "No Data Connected", "description": "Connect a database or upload a file to start analyzing.", "tableCount": 0, "totalRows": 0},
                "metrics": [],
                "visualizations": [],
                "sections": [],
                "insights": [],
                "anomalies": [],
                "quality": {"score": 0, "issues": []},
                "recommendations": [],
            }
        try:
            _attach_verdict(_empty_config, session)
        except Exception:
            try:
                _empty_config["verdict"] = {"verified": False, "checks": [], "warnings": ["verification unavailable"]}
            except Exception:
                pass
        return {
            "ok": True,
            "config": _empty_config,
        }

    # Short-TTL cache (§40): keyed by schema file mtime so reconnects/uploads
    # invalidate automatically. Bounded size, no stale-data risk beyond TTL.
    global _DASH_CONFIG_CACHE
    try:
        _DASH_CONFIG_CACHE
    except NameError:
        _DASH_CONFIG_CACHE = {}
    try:
        _schema_mtime = Path(str(session.get("schema_path", ""))).stat().st_mtime
    except Exception:
        _schema_mtime = 0.0
    _cache_key = (user["user_id"], session.get("db_name", ""), _schema_mtime)
    _cached = _DASH_CONFIG_CACHE.get(_cache_key)
    if _cached and _time.monotonic() - _cached[0] < float(os.getenv("ARIA_DASH_CACHE_TTL", "120")):
        try:
            _cached_age = _time.monotonic() - _cached[0]
            _cached_config = dict(_cached[1])
            _attach_verdict(_cached_config, session, cached=True, cache_age_s=_cached_age)
        except Exception:
            try:
                _cached_config = dict(_cached[1])
                _cached_config["verdict"] = {"verified": False, "checks": [], "warnings": ["verification unavailable"]}
            except Exception:
                return {"ok": True, "config": _cached[1], "cached": True}
        return {"ok": True, "config": _cached_config, "cached": True}
    if len(_DASH_CONFIG_CACHE) > 200:
        _DASH_CONFIG_CACHE.clear()

    engine, dialect, db_type = _build_engine(session)
    db_name = session.get("db_name", session.get("source_type", "data"))
    tables_list = list(intel.tables.keys())
    # Prioritize analysis-ready tables (metadata only, no SQL): tables with
    # real measures and datetime columns first so the [:N] discovery slices
    # below reach fact tables on wide multi-schema databases instead of
    # stopping at lookup/dimension tables. Generic — no table names.
    def _dash_rank(tname):
        try:
            _has_m = bool(intel.find_all_measures(tname))
        except Exception:
            _has_m = False
        try:
            _has_d = bool(intel.get_datetime_cols(tname))
        except Exception:
            _has_d = False
        if _has_m and _has_d:
            return (0, str(tname))
        if _has_m or _has_d:
            return (1, str(tname))
        return (2, str(tname))
    try:
        tables_ranked = sorted(tables_list, key=_dash_rank)
    except Exception:
        tables_ranked = list(tables_list)

    metrics = []
    charts = []
    sections = []
    recommendations = []
    total_rows = 0
    quality_issues = []

    if engine is not None:
        try:
            with engine.connect() as conn:
                # ── Total records ──
                for tname in tables_list[:14]:
                    try:
                        result = conn.execute(text(f"SELECT COUNT(*) FROM {_safe_name(tname, dialect)}"))
                        row = result.fetchone()
                        if row:
                            total_rows += row[0]
                    except Exception:
                        _rollback(conn)
                        continue

                if total_rows > 0:
                    metrics.append({
                        "id": "total_records",
                        "label": "Total Records",
                        "value": _fmt(total_rows),
                        "rawValue": total_rows,
                        "format": "number",
                        "period": "All data",
                        "icon": "database",
                        "color": "#2F6FAD",
                    })

                # ── Scored KPI candidates (spec §2-4, §13-15, §17) ──
                # Profile → score → diversity → threshold. Shape unchanged.
                kpi_candidates: list[dict] = []
                tables_with_measures = []
                for tname in tables_ranked[:10]:
                    if intel.tables[tname].get("empty"):
                        continue
                    _m = intel.find_best_measure(tname)
                    # Weak mappings can label text columns as measures; only
                    # profile real numerics so one bad column cannot poison
                    # the shared connection for every later query.
                    if _m and _m not in intel.get_numeric_cols(tname):
                        continue
                    if _m:
                        tables_with_measures.append((tname, _m))
                for tname, measure in tables_with_measures[:5]:
                    if intel.tables[tname].get("empty"):
                        continue
                    prof = _profile_numeric(conn, tname, measure, dialect, intel=intel)
                    if not prof or not prof["nn"]:
                        continue
                    is_curr = _is_currency_col(measure)
                    dt_col = intel.find_best_datetime(tname)
                    tdesc = intel.tables[tname].get("description", tname)
                    val = float(prof["sum"]) if prof["sum"] is not None else None
                    avg = float(prof["avg"]) if prof["avg"] is not None else None

                    if val is not None:
                        score, reason = _score_kpi("outcome", measure, prof, bool(dt_col), is_curr)
                        kpi_candidates.append({
                            "kind": "outcome", "table": tname, "column": measure,
                            "agg": "SUM", "value": val, "score": score, "reason": reason,
                            "is_currency": is_curr, "date_col": dt_col, "tdesc": tdesc,
                        })
                    if avg is not None:
                        score, reason = _score_kpi("efficiency", measure, prof, bool(dt_col), is_curr)
                        kpi_candidates.append({
                            "kind": "efficiency", "table": tname, "column": measure,
                            "agg": "AVG", "value": avg, "score": score, "reason": reason,
                            "is_currency": is_curr, "date_col": dt_col, "tdesc": tdesc,
                        })
                    # Entity count: distinct values of the best dimension (§3).
                    dim = intel.find_best_dimension(tname)
                    if dim:
                        try:
                            erow = conn.execute(text(
                                f"SELECT COUNT(DISTINCT {_safe_name(dim, dialect)}) "
                                f"FROM {_safe_name(tname, dialect)}"
                            )).fetchone()
                            if erow and erow[0]:
                                ecount = int(erow[0])
                                nrow = conn.execute(text(
                                    f"SELECT COUNT(*) FROM {_safe_name(tname, dialect)}")).fetchone()
                                total_n = float(nrow[0]) if nrow and nrow[0] else 0.0
                                # Redundancy (§15): a unique-per-row column restates
                                # Total Records — skip it.
                                if 2 <= ecount <= max(10, total_n) and ecount < 0.98 * (total_n or 1.0):
                                    eprof = {"n": total_n or 1.0, "nn": total_n or 1.0,
                                             "distinct": float(ecount), "sum": None,
                                             "avg": None, "min": 0, "max": 1}
                                    score, reason = _score_kpi("entity", dim, eprof, bool(dt_col), False)
                                    kpi_candidates.append({
                                        "kind": "entity", "table": tname, "column": dim,
                                        "agg": "DISTINCT", "value": float(ecount),
                                        "score": score, "reason": reason,
                                        "is_currency": False, "date_col": dt_col, "tdesc": tdesc,
                                    })
                        except Exception:
                            _rollback(conn)
                            pass

                # Entity counts stand alone (no measure needed): distinct
                # countries, ratings, categories are first-class KPIs on
                # text-heavy tables that have no summable number. Deduped
                # by (column, agg) in the diversity pass below.
                try:
                    for tname in tables_ranked[:5]:
                        if intel.tables[tname].get("empty"):
                            continue
                        dim = intel.find_best_dimension(tname)
                        if not dim:
                            continue
                        try:
                            erow = conn.execute(text(
                                f"SELECT COUNT(DISTINCT {_safe_name(dim, dialect)}) "
                                f"FROM {_safe_name(tname, dialect)}"
                            )).fetchone()
                            if not (erow and erow[0]):
                                continue
                            ecount = int(erow[0])
                            nrow = conn.execute(text(
                                f"SELECT COUNT(*) FROM {_safe_name(tname, dialect)}")).fetchone()
                            total_n = float(nrow[0]) if nrow and nrow[0] else 0.0
                            if not (2 <= ecount <= max(10, total_n)
                                    and ecount < 0.98 * (total_n or 1.0)):
                                continue
                            _dt0 = intel.find_best_datetime(tname)
                            _td0 = intel.tables[tname].get("description", tname)
                            eprof = {"n": total_n or 1.0, "nn": total_n or 1.0,
                                     "distinct": float(ecount), "sum": None,
                                     "avg": None, "min": 0, "max": 1}
                            score, reason = _score_kpi("entity", dim, eprof, bool(_dt0), False)
                            kpi_candidates.append({
                                "kind": "entity", "table": tname, "column": dim,
                                "agg": "DISTINCT", "value": float(ecount),
                                "score": score, "reason": reason,
                                "is_currency": False, "date_col": _dt0, "tdesc": _td0,
                            })
                        except Exception:
                            _rollback(conn)
                            continue
                except Exception:
                    pass

                # Retail-aware candidates from real execution (single-table
                # safe): sales total/avg (with spark), store count, date
                # range, holiday lift. Scored via the same _score_kpi and
                # prepended so the diversity pass below considers them
                # first — dupes by (column, agg) are admitted once.
                _retail_cands: list[dict] = []
                try:
                    for tname in tables_ranked[:3]:
                        if intel.tables[tname].get("empty"):
                            continue
                        _retail_cands.extend(
                            _retail_candidates_for_table(conn, intel, tname, dialect))
                except Exception:
                    pass
                kpi_candidates = _retail_cands + kpi_candidates

                # Diversity pass: best of each kind first, then fill by score.
                # Redundancy: same (column, agg) admitted once; ≤2 KPIs per table.
                picked: list[dict] = []
                seen_col_agg: set[tuple] = set()
                per_table: dict[str, int] = {}

                def _accept(c: dict) -> bool:
                    key = (c["column"].lower(), c["agg"])
                    if key in seen_col_agg:
                        return False
                    if per_table.get(c["table"], 0) >= 3:
                        return False
                    if c["score"] < DASHBOARD_MIN_SCORE:
                        return False
                    seen_col_agg.add(key)
                    per_table[c["table"]] = per_table.get(c["table"], 0) + 1
                    picked.append(c)
                    return True

                for kind in ("outcome", "entity", "efficiency", "volume"):
                    best = next((c for c in sorted(kpi_candidates, key=lambda x: -x["score"])
                                 if c["kind"] == kind), None)
                    if best and len(picked) < 5:
                        _accept(best)
                for c in sorted(kpi_candidates, key=lambda x: -x["score"]):
                    if len(picked) >= 5:
                        break
                    _accept(c)

                for c in picked:
                    is_curr = c["is_currency"]
                    change = c.get("change_override")
                    period = c.get("period_override") or (
                        "Total" if c["agg"] == "SUM"
                        else ("Average" if c["agg"] == "AVG" else "All data"))
                    # Period comparison (§4): current vs previous grain period.
                    if c["agg"] == "SUM" and c["date_col"]:
                        try:
                            if dialect == "sqlite":
                                gexpr = f"strftime('%Y-%m', {_safe_name(c['date_col'], dialect)})"
                            elif dialect == "mysql":
                                gexpr = f"DATE_FORMAT({_safe_name(c['date_col'], dialect)}, '%Y-%m')"
                            else:
                                gexpr = f"TO_CHAR({_safe_name(c['date_col'], dialect)}, 'YYYY-MM')"
                            prow = conn.execute(text(
                                f"SELECT {gexpr} AS p, SUM({_measure_expr(intel, c['table'], c['column'], dialect)}) AS v "
                                f"FROM {_safe_name(c['table'], dialect)} "
                                f"WHERE {_safe_name(c['date_col'], dialect)} IS NOT NULL "
                                f"GROUP BY p ORDER BY p DESC LIMIT 2"
                            )).fetchall()
                            vals = [float(r[1]) for r in prow if r[1] is not None]
                            if len(vals) >= 2 and vals[1] != 0:
                                change = round(((vals[0] - vals[1]) / abs(vals[1])) * 100, 1)
                                period = "vs prev period"
                        except Exception:
                            _rollback(conn)
                            pass
                        # Non-ISO date text makes SQL grouping return NULL —
                        # fall back to the real-SQL trend series instead.
                        if change is None and len(c.get("trend") or []) >= 2:
                            try:
                                tvals = [float(p["value"]) for p in c["trend"][-2:]]
                                if tvals[0] != 0:
                                    change = round(((tvals[1] - tvals[0]) / abs(tvals[0])) * 100, 1)
                                    period = "vs prev period"
                            except Exception:
                                pass
                    label = c.get("label_override") or kpi_label(c["agg"], c["column"])
                    if c["agg"] == "RANGE":
                        disp_value = c.get("value_override") or _fmt(c["value"], is_curr)
                    elif c["agg"] == "LIFT":
                        disp_value = f"{c['value']:+.1f}%"
                    else:
                        disp_value = _fmt(c["value"], is_curr)
                    metrics.append({
                        "id": f"{c['agg'].lower()}_{c['table']}_{c['column']}",
                        "label": label,
                        "value": disp_value,
                        "rawValue": c["value"],
                        "format": c.get("format_override") or ("currency" if is_curr else "number"),
                        "period": period,
                        "icon": "trending-up" if c["agg"] == "SUM" else "bar-chart",
                        "color": "#059669" if is_curr else "#2F6FAD",
                        "score": c["score"],
                        "reason": f"Selected ({c['kind']}, score {c['score']}): {c['reason']}.",
                    })
                    if change is not None:
                        metrics[-1]["change"] = change
                    if c.get("spark"):
                        metrics[-1]["spark"] = c["spark"]
                    if c.get("trend"):
                        metrics[-1]["trend"] = c["trend"]

                # Guarantee: retail specials (store count, date range,
                # holiday lift) that passed the score threshold are surfaced
                # even when the per-table diversity cap crowded them out.
                try:
                    _picked_keys = {(p["column"].lower(), p["agg"]) for p in picked}
                    for c in _retail_cands:
                        if len(metrics) >= 6:
                            break
                        if c["agg"] not in ("DISTINCT", "RANGE", "LIFT"):
                            continue
                        if (c["column"].lower(), c["agg"]) in _picked_keys:
                            continue
                        if c["score"] < DASHBOARD_MIN_SCORE:
                            continue
                        _picked_keys.add((c["column"].lower(), c["agg"]))
                        if c["agg"] == "RANGE":
                            _mv = c.get("value_override") or _fmt(c["value"])
                            _mf = "text"
                        elif c["agg"] == "LIFT":
                            _mv = f"{c['value']:+.1f}%"
                            _mf = "percent"
                        else:
                            _mv = _fmt(c["value"])
                            _mf = "number"
                        _m = {
                            "id": f"{c['agg'].lower()}_{c['table']}_{c['column']}",
                            "label": c.get("label_override") or kpi_label(c["agg"], c["column"]),
                            "value": _mv,
                            "rawValue": c["value"],
                            "format": _mf,
                            "period": c.get("period_override") or "All data",
                            "icon": "bar-chart",
                            "color": "#2F6FAD",
                            "score": c["score"],
                            "reason": f"Selected ({c['kind']}, score {c['score']}): {c['reason']}.",
                        }
                        if c.get("change_override") is not None:
                            _m["change"] = c["change_override"]
                        if c.get("spark"):
                            _m["spark"] = c["spark"]
                        metrics.append(_m)
                except Exception:
                    pass

                # Last-resort fallback: never return zero metrics on data.
                if not metrics and total_rows > 0:
                    metrics.append({
                        "id": "total_records",
                        "label": "Total Records",
                        "value": _fmt(total_rows),
                        "rawValue": total_rows,
                        "format": "number",
                        "period": "All data",
                        "icon": "database",
                        "color": "#2F6FAD",
                    })

                # ── Scored chart candidates (spec §5-12, §15, §17) ──
                # Grain-aware time series + Top-N categoricals, deduped, capped at 4.
                chart_candidates: list[dict] = []
                used_measure_kind: set[tuple] = set()

                def _chart_score(periods: int, variation: float, groups: int = 0) -> float:
                    base = min(1.0, periods / 8) * 0.5 + min(1.0, variation) * 0.3
                    if groups:
                        # A valid Top-N breakdown already carries comparison value.
                        base += 0.25 + min(1.0, groups / 6) * 0.2
                    return round(base, 3)

                for tname in tables_list[:14]:
                    if intel.tables[tname].get("empty"):
                        continue
                    dt_col = intel.find_best_datetime(tname)
                    measure = intel.find_best_measure(tname)
                    if not dt_col:
                        continue
                    if measure and measure in intel.get_numeric_cols(tname):
                        try:
                            # Real-SQL profile for variation scoring.
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
                            if not srow or srow[0] is None:
                                continue
                            # Grain-aware series from real SQL; handles non-ISO
                            # date text via a raw-pair Python bucketing fallback.
                            grain, chart_data = _fetch_time_series(
                                conn, tname, dt_col, measure, dialect, limit=30, intel=intel)
                            if len(chart_data) >= 2:
                                var = _variation(srow[3], srow[4], srow[5])
                                chart_candidates.append({
                                    "kind": "time", "table": tname, "column": measure,
                                    "score": _chart_score(len(chart_data), var),
                                    "grain": grain or "month", "data": chart_data,
                                    "reason": (f"{len(chart_data)} {grain or 'month'} periods with "
                                               f"{'strong' if var > 0.5 else 'moderate' if var > 0.15 else 'low'} variation."),
                                })
                        except Exception:
                            _rollback(conn)
                            continue
                    else:
                        # No summable measure (catalogs, logs, text-heavy
                        # tables): a row-count volume trend is still a
                        # truthful time chart ("Records added over time").
                        try:
                            grain, chart_data = _fetch_count_series(
                                conn, tname, dt_col, dialect, limit=30, intel=intel)
                            if len(chart_data) >= 2:
                                _vals = [c["value"] for c in chart_data]
                                _avg = sum(_vals) / len(_vals)
                                var = _variation(_avg, min(_vals), max(_vals))
                                chart_candidates.append({
                                    "kind": "time", "table": tname, "column": "Records",
                                    "score": _chart_score(len(chart_data), var),
                                    "grain": grain or "month", "data": chart_data,
                                    "reason": (f"{len(chart_data)} {grain or 'month'} periods of record volume."),
                                })
                        except Exception:
                            _rollback(conn)
                            continue

                if True:
                    for tname in tables_list[:14]:
                        if intel.tables[tname].get("empty"):
                            continue
                        measure = intel.find_best_measure(tname)
                        if not measure:
                            continue
                        if measure not in intel.get_numeric_cols(tname):
                            continue
                        # Candidate dimensions: best first, then other text cols.
                        # Skip near-unique ones (§13) instead of charting IDs.
                        dim_candidates = []
                        _bd = intel.find_best_dimension(tname)
                        if _bd:
                            dim_candidates.append(_bd)
                        for _c in intel.get_text_cols(tname):
                            if _c not in dim_candidates:
                                dim_candidates.append(_c)
                        # Rank candidate dims by cardinality first (cheap): prefer
                        # compact breakdowns (Region, Category) over near-unique
                        # columns (IDs, dates). One data query for the winner.
                        # Up to 10 text columns are probed so genuinely useful
                        # breakdowns later in column order (ratings, genres)
                        # are reached on wide text-heavy tables — the probe is
                        # one cheap COUNT per column, winners capped at 2.
                        ranked_dims: list[tuple] = []
                        for dim in dim_candidates[:10]:
                            try:
                                crow = conn.execute(text(
                                    f"SELECT COUNT(*) AS n, COUNT(DISTINCT {_safe_name(dim, dialect)}) AS d "
                                    f"FROM {_safe_name(tname, dialect)}"
                                )).fetchone()
                                if not crow or not crow[0]:
                                    continue
                                ratio = float(crow[1] or 0) / float(crow[0])
                                if ratio > 0.5 or (crow[1] or 0) < 2:
                                    continue
                                ranked_dims.append((crow[1], dim))
                            except Exception:
                                _rollback(conn)
                                continue
                        ranked_dims.sort(key=lambda x: x[0])
                        for _, dim in ranked_dims[:2]:
                            try:
                                result = conn.execute(text(
                                    f"SELECT {_safe_name(dim, dialect)} AS dim_val, "
                                    f"SUM({_measure_expr(intel, tname, measure, dialect)}) AS val "
                                    f"FROM {_safe_name(tname, dialect)} "
                                    f"WHERE {_safe_name(dim, dialect)} IS NOT NULL "
                                    f"GROUP BY dim_val ORDER BY val DESC LIMIT 8"
                                ))
                                rows = result.fetchall()
                                chart_data = [{"name": str(r[0]), "value": float(r[1])}
                                              for r in rows if r[1] is not None]
                                # §7-8: 2..12 groups; skip single-slice or micro-fragmented.
                                if 2 <= len(chart_data) <= 12:
                                    total_v = sum(c["value"] for c in chart_data) or 1.0
                                    top_share = max(c["value"] for c in chart_data) / total_v
                                    if top_share < 0.98:
                                        vals = [c["value"] for c in chart_data]
                                        var = _variation(sum(vals) / len(vals), min(vals), max(vals))
                                        chart_candidates.append({
                                            "kind": "category", "table": tname, "column": measure,
                                            "dim": dim,
                                            "score": _chart_score(0, var, len(chart_data)),
                                            "data": chart_data,
                                            "reason": (f"Top-{len(chart_data)} {dim} comparison with "
                                                       f"{'clear' if var > 0.4 else 'moderate'} spread."),
                                        })
                            except Exception:
                                _rollback(conn)
                                continue

                # Scatter chart: for tables with 2+ numeric columns, compare two measures.
                if len(charts) < 2:
                    for tname in tables_list[:6]:
                        if intel.tables[tname].get("empty"):
                            continue
                        num_cols = intel.get_numeric_cols(tname)
                        if len(num_cols) < 2:
                            continue
                        m1 = num_cols[0]
                        m2 = num_cols[1]
                        try:
                            result = conn.execute(text(
                                f"SELECT {_measure_expr(intel, tname, m1, dialect)} AS x, "
                                f"{_measure_expr(intel, tname, m2, dialect)} AS y "
                                f"FROM {_safe_name(tname, dialect)} "
                                f"WHERE {_safe_name(m1, dialect)} IS NOT NULL "
                                f"AND {_safe_name(m2, dialect)} IS NOT NULL "
                                f"LIMIT 200"
                            ))
                            rows = result.fetchall()
                            if len(rows) >= 10:
                                scatter_data = [{"x": float(r[0]), "y": float(r[1])}
                                                for r in rows if r[0] is not None and r[1] is not None]
                                if scatter_data:
                                    chart_candidates.append({
                                        "kind": "scatter", "table": tname,
                                        "column": m1, "dim": m2,
                                        "score": 0.5,
                                        "data": scatter_data,
                                        "reason": f"Relationship between {m1} and {m2}.",
                                    })
                        except Exception:
                            _rollback(conn)
                            continue

                # Diversity: one time + one category per measure first, then fill.
                chart_candidates.sort(key=lambda c: -c["score"])
                for c in chart_candidates:
                    if len(charts) >= 4:
                        break
                    if c["score"] < DASHBOARD_MIN_SCORE:
                        continue
                    key = (c["column"].lower(), c["kind"])
                    if key in used_measure_kind:
                        continue
                    used_measure_kind.add(key)
                    tdesc = intel.tables[c["table"]].get("description", c["table"])
                    is_curr = _is_currency_col(c["column"])
                    if c["kind"] == "time":
                        charts.append({
                            "id": f"ts_{c['table']}_{c['column']}",
                            "title": f"{humanize(c['column'])} Over Time",
                            "subtitle": f"{len(c['data'])} {c.get('grain', 'period')} periods",
                            "type": "area",
                            "data": c["data"],
                            "dataKey": "value",
                            "xKey": "period",
                            "format": "currency" if is_curr else "number",
                            "score": c["score"],
                            "reason": c["reason"],
                        })
                    elif c["kind"] == "scatter":
                        charts.append({
                            "id": f"scatter_{c['table']}_{c['column']}",
                            "title": f"{humanize(c['column'])} vs {humanize(c['dim'])}",
                            "subtitle": f"{len(c['data'])} data points",
                            "type": "scatter",
                            "data": c["data"],
                            "xKey": "x",
                            "yKey": "y",
                            "format": "number",
                            "score": c["score"],
                            "reason": c["reason"],
                        })
                    else:
                        charts.append({
                            "id": f"cat_{c['table']}_{c['column']}",
                            "title": f"{humanize(c['column'])} by {humanize(c['dim'])}",
                            "subtitle": f"Top {len(c['data'])}",
                            "type": "bar",
                            "data": c["data"],
                            "dataKey": "value",
                            "xKey": "name",
                            "format": "currency" if is_curr else "number",
                            "score": c["score"],
                            "reason": c["reason"],
                        })

                # ── Section summaries (analytics, insights, predictive, etc.) ──
                # Analytics
                for tname in tables_list:
                    measure = intel.find_best_measure(tname)
                    if not measure:
                        continue
                    if measure not in intel.get_numeric_cols(tname):
                        continue
                    tdesc = intel.tables[tname].get("description", tname)
                    is_curr = _is_currency_col(measure)
                    try:
                        result = conn.execute(text(
                            f"SELECT SUM({_measure_expr(intel, tname, measure, dialect)}) FROM {_safe_name(tname, dialect)}"
                        ))
                        row = result.fetchone()
                        total = float(row[0]) if row and row[0] is not None else None
                    except Exception:
                        _rollback(conn)
                        total = None
                    if total is not None:
                        sections.append({
                            "type": "analytics",
                            "title": f"{humanize(measure)} Overview",
                            "metric": _fmt(total, is_curr),
                            "description": f"Total {humanize(measure)}: {_fmt(total, is_curr)}. Use AI Analytics to explore trends, breakdowns, and detailed analysis.",
                            "query": f"What is the total {humanize(measure).lower()}?",
                        })
                        break

                # Insights
                for tname in tables_list:
                    measure = intel.find_best_measure(tname)
                    dt = intel.find_best_datetime(tname)
                    if not measure or not dt:
                        continue
                    if measure not in intel.get_numeric_cols(tname):
                        continue
                    tdesc = intel.tables[tname].get("description", tname)
                    try:
                        if dialect == "sqlite":
                            time_expr = f"strftime('%Y-%m', {_safe_name(dt, dialect)})"
                        elif dialect == "mysql":
                            time_expr = f"DATE_FORMAT({_safe_name(dt, dialect)}, '%Y-%m')"
                        else:
                            time_expr = f"TO_CHAR({_safe_name(dt, dialect)}, 'YYYY-MM')"
                        result = conn.execute(text(
                            f"SELECT {time_expr} AS period, SUM({_measure_expr(intel, tname, measure, dialect)}) AS val "
                            f"FROM {_safe_name(tname, dialect)} "
                            f"WHERE {_safe_name(dt, dialect)} IS NOT NULL "
                            f"GROUP BY period ORDER BY period DESC LIMIT 6"
                        ))
                        rows = result.fetchall()
                        if len(rows) >= 2:
                            vals = [(str(r[0]), float(r[1])) for r in rows if r[1] is not None]
                            if len(vals) >= 2:
                                curr, prev = vals[0], vals[1]
                                if prev != 0:
                                    change = ((curr[1] - prev[1]) / abs(prev[1])) * 100
                                    direction = "increased" if change > 0 else "decreased"
                                    sections.append({
                                        "type": "insights",
                                        "title": f"{humanize(measure)} Trend Detected",
                                        "metric": f"{change:+.1f}%",
                                        "description": f"{humanize(measure)} {direction} by {abs(change):.1f}% from {prev[0]} to {curr[0]}. Insights page has detailed analysis of all patterns.",
                                        "query": f"What are the trends in {humanize(measure).lower()}?",
                                    })
                            break
                    except Exception:
                        _rollback(conn)
                        continue

                # Predictive
                for tname in tables_list:
                    measure = intel.find_best_measure(tname)
                    dt = intel.find_best_datetime(tname)
                    if not measure or not dt:
                        continue
                    if measure not in intel.get_numeric_cols(tname):
                        continue
                    tdesc = intel.tables[tname].get("description", tname)
                    try:
                        if dialect == "sqlite":
                            _mexpr = f"strftime('%Y-%m', {_safe_name(dt, dialect)})"
                        elif dialect == "mysql":
                            _mexpr = f"DATE_FORMAT({_safe_name(dt, dialect)}, '%Y-%m')"
                        else:
                            _mexpr = f"TO_CHAR({_safe_name(dt, dialect)}, 'YYYY-MM')"
                        result = conn.execute(text(
                            f"SELECT COUNT(DISTINCT {_mexpr}) "
                            f"FROM {_safe_name(tname, dialect)} "
                            f"WHERE {_safe_name(dt, dialect)} IS NOT NULL"
                        ))
                        row = result.fetchone()
                        months = int(row[0]) if row and row[0] else 0
                        if months >= 3:
                            sections.append({
                                "type": "predictive",
                                "title": f"{humanize(measure)} Forecast Available",
                                "metric": f"{months} months of data",
                                "description": f"Connected data has {months} months of {humanize(measure).lower()} history. Predictive Analysis can generate forecasts with confidence intervals.",
                                "query": f"Forecast {humanize(measure).lower()} over time",
                            })
                            break
                    except Exception:
                        _rollback(conn)
                        continue

                # Prescription
                for tname in tables_list:
                    measure = intel.find_best_measure(tname)
                    dim = intel.find_best_dimension(tname)
                    if not measure or not dim:
                        continue
                    if measure not in intel.get_numeric_cols(tname):
                        continue
                    tdesc = intel.tables[tname].get("description", tname)
                    try:
                        result = conn.execute(text(
                            f"SELECT {_safe_name(dim, dialect)} AS d, SUM({_measure_expr(intel, tname, measure, dialect)}) AS v "
                            f"FROM {_safe_name(tname, dialect)} "
                            f"WHERE {_safe_name(dim, dialect)} IS NOT NULL "
                            f"GROUP BY d ORDER BY v DESC LIMIT 5"
                        ))
                        rows = result.fetchall()
                        if len(rows) >= 2:
                            top = rows[0]
                            sections.append({
                                "type": "prescription",
                                "title": f"Top {humanize(dim)}: {top[0]}",
                                "metric": _fmt(float(top[1]), _is_currency_col(measure)),
                                "description": f"The leading {humanize(dim).lower()} is '{top[0]}'. Prescription page can generate evidence-based recommendations.",
                                "query": f"What actions should I take based on {humanize(measure).lower()}?",
                            })
                            break
                    except Exception:
                        _rollback(conn)
                        continue

                # Hypothesis
                for tname in tables_list:
                    numerics = intel.get_numeric_cols(tname)
                    non_id = [c for c in numerics if not c.lower().endswith("_id") and c.lower() != "id"]
                    if len(non_id) >= 2:
                        tdesc = intel.tables[tname].get("description", tname)
                        sections.append({
                            "type": "hypothesis",
                            "title": "Statistical Testing Available",
                            "metric": f"{len(non_id)} numeric measures",
                            "description": f"{tdesc} has {len(non_id)} numeric measures suitable for correlation and hypothesis testing.",
                            "query": "Is there a correlation between numeric measures?",
                        })
                        break

                # ── Recommendations ──
                for tname in tables_list[:5]:
                    measure = intel.find_best_measure(tname)
                    dim = intel.find_best_dimension(tname)
                    dt = intel.find_best_datetime(tname)
                    if measure and dim:
                        recommendations.append({"id": f"rec_{tname}_dim", "question": f"What are the top {humanize(dim).lower()} by {humanize(measure).lower()}?", "type": "dimension"})
                    if measure and dt:
                        recommendations.append({"id": f"rec_{tname}_trend", "question": f"How has {humanize(measure).lower()} changed over time?", "type": "trend"})
                    if measure and dim and len(recommendations) < 6:
                        recommendations.append({"id": f"rec_{tname}_breakdown", "question": f"Show {humanize(measure).lower()} breakdown by {humanize(dim).lower()}", "type": "breakdown"})

                # ── Data quality ──
                quality_score = 100
                for tname in tables_list[:5]:
                    measure = intel.find_best_measure(tname)
                    if not measure:
                        continue
                    try:
                        result = conn.execute(text(
                            f"SELECT COUNT(*) FROM {_safe_name(tname, dialect)} WHERE {_safe_name(measure, dialect)} IS NULL"
                        ))
                        row = result.fetchone()
                        if row and row[0] > 0:
                            total_q = conn.execute(text(f"SELECT COUNT(*) FROM {_safe_name(tname, dialect)}")).fetchone()
                            if total_q and total_q[0] > 0:
                                pct = (row[0] / total_q[0]) * 100
                                if pct > 0:
                                    quality_issues.append({
                                        "field": f"{tname}.{measure}",
                                        "issue": f"{pct:.1f}% null values",
                                        "severity": "warning" if pct < 10 else "critical",
                                        "affectedRecords": row[0],
                                    })
                                    quality_score -= min(10, int(pct))
                    except Exception:
                        _rollback(conn)
                        continue

        except Exception:
            pass

    # ── Auto insights + anomalies (no prior ask needed) ──
    # Runs the same detectors as /api/insights so the dashboard is alive
    # immediately after connect. Best-effort: failures leave empty lists.
    auto_insights: list[dict] = []
    auto_anomalies: list[dict] = []
    try:
        from api.insights_routes import InsightsEngine, _SQLiteCompatConnection

        _mapping = None
        try:
            _sp = Path(str(session.get("schema_path", "")))
            if _sp.exists():
                _mapping = json.loads(_sp.read_text(encoding="utf-8"))
        except Exception:
            _mapping = None
        _iconn = None
        if _mapping and _mapping.get("tables"):
            try:
                if dialect == "sqlite":
                    import sqlite3 as _sqlite3

                    _db_uri = session.get("db_uri", "") or ""
                    _path = _db_uri[len("sqlite:///"):] if _db_uri.startswith("sqlite:///") else _db_uri
                    _iconn = _SQLiteCompatConnection(_sqlite3.connect(_path))
                elif engine is not None:
                    _iconn = engine.raw_connection()
                if _iconn is not None:
                    _auto = InsightsEngine(_iconn, _mapping).generate_all_insights(max_insights=8)
                    for _ins in _auto:
                        if str(_ins.get("type", "")).lower() == "anomaly":
                            if len(auto_anomalies) < 3:
                                auto_anomalies.append(_ins)
                        elif len(auto_insights) < 4:
                            auto_insights.append(_ins)
            finally:
                try:
                    if _iconn is not None:
                        _iconn.close()
                except Exception:
                    pass
    except Exception:
        pass

    config = {
        "summary": {
            "title": f"Data Analysis: {db_name}",
            "description": f"ARIA analyzed {len(tables_list)} tables with {_fmt(total_rows)} total records.",
            "tableCount": len(tables_list),
            "totalRows": total_rows,
            "database": db_name,
        },
        "metrics": metrics[:6],
        "visualizations": charts[:4],
        "sections": sections[:5],
        "insights": auto_insights,
        "anomalies": auto_anomalies,
        "quality": {"score": max(0, quality_score), "issues": quality_issues[:10]},
        "recommendations": recommendations[:6],
    }
    # JSON-safety boundary (generic): statistical detectors can emit
    # NaN/Infinity (0/0, log(0)) on unseen data shapes, which crashes JSON
    # serialization into a 500 ("couldn't load this analysis"). Sanitize
    # recursively so the endpoint never fails on payload encoding.
    config = _sanitize_json_floats(config)
    # Verification Agent hookup (best-effort, never blocks the dashboard).
    try:
        _attach_verdict(config, session)
    except Exception:
        try:
            config["verdict"] = {"verified": False, "checks": [], "warnings": ["verification unavailable"]}
        except Exception:
            pass
    try:
        _DASH_CONFIG_CACHE[_cache_key] = (_time.monotonic(), config)
    except Exception:
        pass

    return {"ok": True, "config": config}


@router.get("/api/chart/recommend")
def recommend_chart(request: Request, columns: str = "", user: dict = Depends(get_current_user)):
    """Recommend chart type based on column characteristics.

    Query params: columns (comma-separated column names).
    Returns recommended chart type and compatible alternatives.
    """
    session = get_session(user["user_id"])
    reject_document_sql(session)
    cols = [c.strip() for c in columns.split(",") if c.strip()]

    # Column types from the schema mapping (always available — no dependence
    # on a prior analysis having written processed rows).
    col_types: dict[str, str] = {}
    try:
        _intel = _load_schema(session, user["user_id"])
        if _intel is not None:
            _num = set()
            for _t in _intel.tables:
                _num.update(_intel.get_numeric_cols(_t))
                for _c in _intel.get_datetime_cols(_t):
                    col_types[_c] = "datetime"
                    col_types[_c.lower()] = "datetime"
            for _c in _num:
                col_types.setdefault(_c, "numeric")
                col_types.setdefault(_c.lower(), "numeric")
            for _c in cols:
                col_types.setdefault(_c, "text" if _c not in col_types else col_types[_c])
    except Exception:
        pass

    # Get data sample if available (refines cardinality/type guesses)
    data_sample = []
    processed_path = session.get("processed_path")
    if processed_path:
        try:
            pp = Path(processed_path)
            if pp.exists():
                pdata = json.loads(pp.read_text(encoding="utf-8"))
                data_sample = (pdata.get("data") or [])[:5]
        except Exception:
            pass

    result = _recommend_chart(cols, data_sample, col_types or None)

    # Apply user preference override if available
    from api.feedback_routes import _load_prefs
    prefs = _load_prefs(user["user_id"])
    chart_history = prefs.get("chart_type_history", {})
    context_key = "_".join(sorted(cols[:3])) if cols else "general"
    if context_key in chart_history:
        counts = chart_history[context_key]
        preferred = max(counts, key=counts.get) if counts else None
        if preferred and preferred in result.get("compatible_types", []):
            result["preferred_type"] = preferred
            result["user_override"] = True

    return {"ok": True, "recommendation": result}
