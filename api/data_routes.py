"""Data section routes: preview, profiling, quality, connection health, user overrides."""

import json
import re
import time
import types
from pathlib import Path
from urllib.parse import quote_plus

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, field_validator
from sqlalchemy import create_engine, text

import schema_agent
from core.config import SCHEMA_DIR, get_session, save_session
from core.errors import sanitize_error
from core.deps import get_current_user, require_data, require_writable, reject_document_sql
from core.secret_box import decrypt as _decrypt_secret

router = APIRouter()


def _get_db_config(session):
    """Extract decrypted DB config from session."""
    db_config = session.get("db")
    if not db_config:
        return None, None, None
    db_type = session.get("db_type", "postgresql")
    if db_type in ("sqlite", "document"):
        return db_config, db_type, None
    config = dict(db_config)
    if "password" in config and config["password"] is not None:
        try:
            config["password"] = _decrypt_secret(config["password"])
        except Exception:
            pass
    driver = "mysql" if db_type == "mysql" else "postgresql"
    dialect = "mysql" if driver == "mysql" else "postgresql"
    if driver == "mysql":
        engine_url = f"mysql+pymysql://{quote_plus(str(config.get('user', '')))}:{quote_plus(str(config.get('password', '')))}@{config.get('host', 'localhost')}:{config.get('port', 3306)}/{config.get('db', '')}"
    else:
        engine_url = f"postgresql+psycopg2://{quote_plus(str(config.get('user', '')))}:{quote_plus(str(config.get('password', '')))}@{config.get('host', 'localhost')}:{config.get('port', 5432)}/{config.get('db', '')}"
    return config, db_type, engine_url


def _get_schema_mapping(session, user_id):
    """Load the current schema mapping from disk."""
    path = session.get("schema_path")
    if path and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _get_user_overrides_path(user_id, db_name):
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(db_name or "db"))[:64] or "db"
    return SCHEMA_DIR / f"user_{user_id}" / f"overrides_{safe}.json"


def _load_user_overrides(user_id, db_name):
    path = _get_user_overrides_path(user_id, db_name)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"pk": {}, "fk": {}, "rejected": []}


def _save_user_overrides(user_id, db_name, overrides):
    path = _get_user_overrides_path(user_id, db_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(overrides, indent=2), encoding="utf-8")


def _quote(name, dialect="postgresql"):
    # Dotted names (schema.table from multi-schema mappings) quote each
    # part separately: "sales"."employee", never "sales.employee".
    try:
        segs = [s for s in str(name or "").split(".") if s]
        if len(segs) > 1:
            if dialect == "mysql":
                return ".".join(f"`{s.replace('`', '``')}`" for s in segs)
            return ".".join(
                f'"{s.replace(chr(34), chr(34) + chr(34))}"' for s in segs)
    except Exception:
        pass
    if dialect == "mysql":
        return f"`{name.replace('`', '``')}`"
    return f'"{name.replace(chr(34), chr(34)+chr(34))}"'


def _resolve_table_param(table, tables):
    """Resolve a preview/table parameter against mapping keys.

    Accepts the qualified key (sales.employee) or the bare segment
    (employee) when it is unambiguous; raises HTTPException 400 otherwise.
    Generic — no schema or table names.
    """
    keys = list((tables or {}).keys())
    if table in keys:
        return table
    try:
        bare = str(table or "").split(".")[-1].lower()
        hits = [k for k in keys if str(k).split(".")[-1].lower() == bare]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise HTTPException(
                status_code=400,
                detail=(f"Table '{table}' exists in several schemas "
                        f"({', '.join(hits)}). Pick one explicitly."))
    except HTTPException:
        raise
    except Exception:
        pass
    raise HTTPException(status_code=400, detail=f"Table '{table}' not found in schema.")


# =============================================================
# CONNECTION HEALTH
# =============================================================

@router.get("/api/data/connection")
def get_connection_health(request: Request, user: dict = Depends(get_current_user)):
    """Return connection health status and database metadata."""
    session = get_session(user["user_id"])
    source_type = session.get("source_type", "none")
    db_name = session.get("db_name", "")
    db_type = session.get("db_type", "")
    provider_name = session.get("provider_name", "")

    if not session.get("db") and source_type == "relational":
        return {"ok": True, "status": "disconnected", "source_type": source_type}

    config, dtype, engine_url = _get_db_config(session)
    if engine_url:
        try:
            engine = create_engine(engine_url, pool_pre_ping=True, connect_args={"connect_timeout": 5})
            with engine.connect() as conn:
                if dtype == "mysql":
                    result = conn.execute(text("SELECT 1"))
                else:
                    result = conn.execute(text("SELECT 1"))
            status = "connected"
        except Exception as e:
            status = f"error: {sanitize_error(e, 200)}"
    elif source_type in ("semi-structured", "document"):
        status = "connected"
    else:
        status = "unknown"

    mapping = _get_schema_mapping(session, user["user_id"])
    table_count = len(mapping.get("tables", {})) if mapping else 0
    declared_fk = len(mapping.get("declared_relationships", [])) if mapping else 0
    inferred_fk = len(mapping.get("inferred_relationships", [])) if mapping else 0
    total_columns = sum(len(t.get("columns", [])) for t in mapping.get("tables", {}).values()) if mapping else 0
    tables_with_pk = mapping.get("summary", {}).get("tables_with_pk", []) if mapping else []
    tables_without_pk = mapping.get("summary", {}).get("tables_without_pk", []) if mapping else []
    generated_at = mapping.get("generated_at", "") if mapping else ""

    host_info = ""
    if config and isinstance(config, dict):
        host_info = f"{config.get('host', '')}:{config.get('port', '')}"

    return {
        "ok": True,
        "status": status,
        "source_type": source_type,
        "db_type": db_type,
        "database": db_name,
        "host": host_info,
        "schema": config.get("db_schema", "") if config and isinstance(config, dict) else "",
        "table_count": table_count,
        "column_count": total_columns,
        "declared_relationships": declared_fk,
        "inferred_relationships": inferred_fk,
        "tables_with_pk": tables_with_pk,
        "tables_without_pk": tables_without_pk,
        "last_schema_scan": generated_at,
        "provider": provider_name,
    }


@router.post("/api/data/refresh")
def refresh_connection(request: Request, user: dict = Depends(require_writable)):
    """Refresh the database connection and re-validate."""
    session = get_session(user["user_id"])
    require_data(request)
    config, dtype, engine_url = _get_db_config(session)
    if not engine_url:
        if session.get("source_type") in ("semi-structured", "document"):
            return {"ok": True, "status": "connected", "message": "File source is active."}
        return {"ok": False, "detail": "No database connection to refresh."}

    try:
        engine = create_engine(engine_url, pool_pre_ping=True, connect_args={"connect_timeout": 10})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True, "status": "connected", "message": "Connection validated successfully."}
    except Exception as e:
        return {"ok": False, "detail": f"Connection refresh failed: {sanitize_error(e)}"}


# =============================================================
# TABLE SUMMARY (lightweight)
# =============================================================

@router.get("/api/data/summary")
def get_data_summary(request: Request, user: dict = Depends(get_current_user)):
    """Return a lightweight summary of all tables with REAL row counts from the database."""
    session = get_session(user["user_id"])
    mapping = _get_schema_mapping(session, user["user_id"])
    if not mapping:
        return {"ok": True, "tables": [], "total_tables": 0}

    overrides = _load_user_overrides(user["user_id"], session.get("db_name", ""))

    # Get real row counts from the database
    row_counts: dict = {}
    config, db_type, engine_url = _get_db_config(session)
    if engine_url:
        try:
            engine = create_engine(engine_url, pool_pre_ping=True, connect_args={"connect_timeout": 5})
            with engine.connect() as conn:
                for name in mapping.get("tables", {}):
                    try:
                        result = conn.execute(text(f"SELECT COUNT(*) FROM {_quote(name, db_type)}"))
                        row_counts[name] = result.scalar() or 0
                    except Exception:
                        row_counts[name] = 0
            engine.dispose()
        except Exception:
            pass

    tables = []
    for name, tbl in mapping.get("tables", {}).items():
        cols = tbl.get("columns", [])
        pk = overrides["pk"].get(name, tbl.get("primary_key", tbl.get("inferred_primary_key", [])))
        tables.append({
            "name": name,
            "column_count": len(cols),
            "row_count": row_counts.get(name, tbl.get("row_count", 0)),
            "primary_key": pk or [],
            "has_pk": bool(pk),
            "description": tbl.get("description", ""),
            "empty": tbl.get("empty", False),
        })

    summary = mapping.get("summary", {})
    return {
        "ok": True,
        "tables": tables,
        "total_tables": summary.get("total_db_tables", len(tables)),
        "tables_with_pk": summary.get("tables_with_pk", []),
        "tables_without_pk": summary.get("tables_without_pk", []),
    }


# =============================================================
# DATA PREVIEW (paginated rows)
# =============================================================

MAX_PREVIEW_LIMIT = 1000
MAX_PREVIEW_OFFSET = 1000000
MAX_PREVIEW_SEARCH = 200


class PreviewRequest(BaseModel):
    table: str
    limit: int = 50
    offset: int = 0
    search: str = ""
    sort_column: str = ""
    sort_direction: str = "ASC"
    # Optional explicit search scope; when omitted the first 50 text
    # columns are searched (was a silent first-20 cut).
    search_columns: list[str] | None = None

    @field_validator("limit")
    @classmethod
    def _bounded_limit(cls, v):
        try:
            n = int(v or 0)
        except (TypeError, ValueError):
            raise ValueError("limit must be an integer")
        if n < 1:
            return 1
        return min(n, MAX_PREVIEW_LIMIT)

    @field_validator("offset")
    @classmethod
    def _non_negative_offset(cls, v):
        try:
            n = int(v or 0)
        except (TypeError, ValueError):
            raise ValueError("offset must be an integer")
        if n < 0:
            return 0
        return min(n, MAX_PREVIEW_OFFSET)

    @field_validator("search")
    @classmethod
    def _bounded_search(cls, v):
        s = str(v or "")
        if len(s) > MAX_PREVIEW_SEARCH:
            return s[:MAX_PREVIEW_SEARCH]
        return s

    @field_validator("sort_column")
    @classmethod
    def _safe_sort_column(cls, v):
        from core.semantic import is_valid_identifier
        s = str(v or "")
        if s and not is_valid_identifier(s):
            raise ValueError(f"Invalid sort_column name: {s!r}")
        return s

    @field_validator("sort_direction")
    @classmethod
    def _safe_sort_direction(cls, v):
        s = str(v or "ASC").upper()
        return "DESC" if s == "DESC" else "ASC"


@router.post("/api/data/preview")
def preview_table(req: PreviewRequest, request: Request, user: dict = Depends(get_current_user)):
    """Return paginated rows from a table in the connected database."""
    session = get_session(user["user_id"])
    reject_document_sql(session)
    mapping = _get_schema_mapping(session, user["user_id"])
    if not mapping:
        raise HTTPException(status_code=400, detail="No schema extracted. Run schema extraction first.")

    _tkey = _resolve_table_param(req.table, mapping.get("tables", {}))

    table_info = mapping["tables"][_tkey]
    columns = [c["column"] for c in table_info.get("columns", [])]
    dialect = session.get("dialect", "postgresql")
    # Explicit search scope wins (validated against the mapping so unknown
    # names cannot inject SQL); otherwise the first 50 columns. The old
    # silent first-20 cut could never match wide-table columns.
    if req.search_columns:
        _known = set(columns)
        _sel = [c for c in req.search_columns if c in _known][:50]
        search_cols = _sel or columns[:50]
    else:
        search_cols = columns[:50]

    config, db_type, engine_url = _get_db_config(session)
    if not engine_url:
        if db_type == "sqlite" and config:
            # CSV uploads store the real SQLite URI in session["db_uri"];
            # session["db"][".csv"] is the raw CSV file, NOT a database.
            engine_url = session.get("db_uri") or None
            if not engine_url and isinstance(config, dict):
                for ext in (".db", ".sqlite", ".sqlite3"):
                    if ext in config:
                        engine_url = f"sqlite:///{config[ext]}"
                        break
            if not engine_url:
                return {"ok": True, "rows": [], "columns": columns, "total": 0}
        else:
            return {"ok": True, "rows": [], "columns": columns, "total": 0}

    try:
        engine = create_engine(engine_url, pool_pre_ping=True)
        q_t = _quote(_tkey, dialect)
        # ILIKE is PostgreSQL-only; use portable LOWER(...) LIKE on others.
        like_op = "ILIKE" if dialect == "postgresql" else "LIKE"

        def _search_cond(columns):
            # CAST target is dialect-specific: TEXT (pg/sqlite), CHAR (mysql).
            _cast = "CHAR" if dialect == "mysql" else "TEXT"
            if like_op == "ILIKE":
                return " OR ".join(
                    f"CAST({_quote(c, dialect)} AS {_cast}) ILIKE :search" for c in search_cols
                )
            return " OR ".join(
                f"LOWER(CAST({_quote(c, dialect)} AS {_cast})) LIKE LOWER(:search)"
                for c in search_cols
            )

        # Count total
        count_sql = f"SELECT COUNT(*) FROM {q_t}"
        params = {}
        if req.search and columns:
            count_sql += f" WHERE {_search_cond(columns)}"
            params["search"] = f"%{req.search}%"

        with engine.connect() as conn:
            total = conn.execute(text(count_sql), params).scalar() or 0

        # Fetch rows
        col_list = ", ".join(_quote(c, dialect) for c in columns)
        query = f"SELECT {col_list} FROM {q_t}"

        if req.search and columns:
            query += f" WHERE {_search_cond(columns)}"
            params["search"] = f"%{req.search}%"

        if req.sort_column and req.sort_column in columns:
            direction = "DESC" if req.sort_direction.upper() == "DESC" else "ASC"
            query += f" ORDER BY {_quote(req.sort_column, dialect)} {direction}"
        else:
            query += f" ORDER BY 1"

        safe_limit = min(max(1, int(req.limit)), MAX_PREVIEW_LIMIT)
        safe_offset = min(max(0, int(req.offset)), MAX_PREVIEW_OFFSET)
        # Bound params for pagination values (identifiers stay quoted via
        # _quote after schema-membership + sort_column validation above).
        query += " LIMIT :limit OFFSET :offset"
        params["limit"] = safe_limit
        params["offset"] = safe_offset

        with engine.connect() as conn:
            rows = conn.execute(text(query), params).fetchall()
            result_rows = [dict(row._mapping) for row in rows]

        # CSV formula-injection signal (additive only, no mutation): flag when
        # any preview cell starts with =, +, -, @ so the UI can warn before
        # export/open. Imported lazily with a local fallback to avoid cycles.
        try:
            from api.connect_routes import _scan_formula_risk as _scan_fr
            _formula_risk = bool(_scan_fr(result_rows))
        except Exception:
            _formula_risk = any(
                isinstance(r, dict) and any(
                    str(v).lstrip()[:1] in ("=", "+", "-", "@")
                    for v in r.values() if v is not None
                )
                for r in (result_rows or [])
            )

        return {
            "ok": True,
            "rows": result_rows,
            "columns": columns,
            "total": total,
            "limit": safe_limit,
            "offset": safe_offset,
            "formula_risk": _formula_risk,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview failed: {sanitize_error(e)}")


# =============================================================
# COLUMN PROFILING
# =============================================================

@router.get("/api/data/columns")
def get_column_profile(request: Request, table: str = "", user: dict = Depends(get_current_user)):
    """Return detailed column metadata and profiling for a table."""
    session = get_session(user["user_id"])
    reject_document_sql(session)
    mapping = _get_schema_mapping(session, user["user_id"])
    if not mapping:
        raise HTTPException(status_code=400, detail="No schema extracted.")

    if not table:
        raise HTTPException(status_code=400, detail="Table '' not found.")
    table = _resolve_table_param(table, mapping.get("tables", {}))

    table_info = mapping["tables"][table]
    cols = table_info.get("columns", [])
    pk = table_info.get("primary_key", table_info.get("inferred_primary_key", []))
    null_stats = table_info.get("null_stats", {})
    overrides = _load_user_overrides(user["user_id"], session.get("db_name", ""))
    user_pk = overrides["pk"].get(table, pk)

    # Get FK info
    declared_fks = []
    for rel in mapping.get("declared_relationships", []):
        if rel.get("table_name") == table:
            declared_fks.append({
                "column": rel["column_name"],
                "references_table": rel["references_table"],
                "references_column": rel["references_column"],
                "source": "database",
                "status": "active",
            })

    inferred_fks = []
    for rel in mapping.get("inferred_relationships", []):
        if rel.get("table") == table:
            rel_id = f"{table}.{rel['column']}->{rel['references_table']}.{rel['references_column']}"
            status = "active"
            if rel_id in overrides.get("rejected", []):
                status = "rejected"
            elif rel_id in overrides.get("fk", {}):
                status = "confirmed"
            inferred_fks.append({
                "column": rel["column"],
                "references_table": rel["references_table"],
                "references_column": rel["references_column"],
                "confidence": rel.get("confidence", ""),
                "confidence_score": rel.get("confidence_score", 0),
                "confidence_band": rel.get("confidence_band", ""),
                "relationship_type": rel.get("relationship_type", ""),
                "source": "aria",
                "status": status,
                "evidence": rel.get("evidence", []),
            })

    column_details = []
    for col in cols:
        col_name = col["column"]
        ns = null_stats.get(col_name, {})
        is_pk = col_name in (user_pk or [])
        is_fk = any(f["column"] == col_name for f in declared_fks + inferred_fks)
        is_unique = col_name in table_info.get("unique_keys", [])

        column_details.append({
            "name": col_name,
            "data_type": col.get("data_type", "unknown"),
            "nullable": col.get("nullable", True),
            "is_primary_key": is_pk,
            "is_foreign_key": is_fk,
            "is_unique": is_unique,
            "null_count": ns.get("nulls", 0),
            "null_percentage": ns.get("null_pct", 0),
            "total_rows": ns.get("total_rows", 0),
            "sample_values": col.get("sample_values", []),
        })

    # Run column profiling if connected to a real DB
    config, db_type, engine_url = _get_db_config(session)
    if engine_url:
        try:
            engine = create_engine(engine_url, pool_pre_ping=True)
            dialect = session.get("dialect", "postgresql")
            q_t = _quote(table, dialect)
            with engine.connect() as conn:
                for cd in column_details:
                    cn = _quote(cd["name"], dialect)
                    try:
                        # Distinct count
                        r = conn.execute(text(f"SELECT COUNT(DISTINCT {cn}) FROM {q_t}")).scalar()
                        cd["distinct_count"] = r or 0

                        # Min/Max for numeric-ish types
                        dt = cd["data_type"].upper()
                        if any(t in dt for t in ["INT", "NUMERIC", "DECIMAL", "FLOAT", "DOUBLE", "REAL", "MONEY"]):
                            mn = conn.execute(text(f"SELECT MIN({cn}) FROM {q_t}")).scalar()
                            mx = conn.execute(text(f"SELECT MAX({cn}) FROM {q_t}")).scalar()
                            avg = conn.execute(text(f"SELECT AVG({cn}) FROM {q_t}")).scalar()
                            cd["min"] = str(mn) if mn is not None else None
                            cd["max"] = str(mx) if mx is not None else None
                            cd["avg"] = str(round(float(avg), 2)) if avg is not None else None
                        elif any(t in dt for t in ["DATE", "TIME", "TIMESTAMP"]):
                            mn = conn.execute(text(f"SELECT MIN({cn}) FROM {q_t}")).scalar()
                            mx = conn.execute(text(f"SELECT MAX({cn}) FROM {q_t}")).scalar()
                            cd["min"] = str(mn) if mn is not None else None
                            cd["max"] = str(mx) if mx is not None else None
                    except Exception:
                        pass
        except Exception:
            pass

    return {
        "ok": True,
        "table": table,
        "columns": column_details,
        "primary_key": user_pk or pk or [],
        "declared_fks": declared_fks,
        "inferred_fks": inferred_fks,
        "row_count": table_info.get("row_count"),
    }


# =============================================================
# DATA QUALITY
# =============================================================

@router.get("/api/data/quality")
def get_data_quality(request: Request, user: dict = Depends(get_current_user)):
    """Return data quality analysis for the connected database."""
    session = get_session(user["user_id"])
    reject_document_sql(session)
    mapping = _get_schema_mapping(session, user["user_id"])
    if not mapping:
        return {"ok": True, "quality": {"overall": 0, "issues": [], "metrics": {}}}

    tables = mapping.get("tables", {})
    issues = []
    total_tables = len(tables)

    if total_tables == 0:
        return {"ok": True, "quality": {"overall": 100, "issues": [], "metrics": {}}}

    # Analyze each table
    total_null_pct = 0
    total_tables_with_pk = 0
    total_empty = 0
    total_duplicate_suspect = 0

    for tname, tinfo in tables.items():
        cols = tinfo.get("columns", [])
        null_stats = tinfo.get("null_stats", {})
        pk = tinfo.get("primary_key", tinfo.get("inferred_primary_key", []))

        if tinfo.get("empty", False):
            total_empty += 1
            issues.append({
                "table": tname,
                "column": None,
                "check": "empty_table",
                "status": "error",
                "severity": "high",
                "message": f"Table '{tname}' is empty (0 rows).",
                "affected_rows": 0,
                "percentage": 100,
            })
            continue

        if pk:
            total_tables_with_pk += 1

        for col_info in cols:
            cname = col_info["column"]
            ns = null_stats.get(cname, {})
            null_pct = ns.get("null_pct", 0)
            total_null_pct += null_pct

            # Null check
            if null_pct > 50:
                issues.append({
                    "table": tname,
                    "column": cname,
                    "check": "null_check",
                    "status": "error",
                    "severity": "high",
                    "message": f"{tname}.{cname} is {null_pct:.0f}% NULL.",
                    "affected_rows": ns.get("nulls", 0),
                    "percentage": null_pct,
                })
            elif null_pct > 20:
                issues.append({
                    "table": tname,
                    "column": cname,
                    "check": "null_check",
                    "status": "warning",
                    "severity": "medium",
                    "message": f"{tname}.{cname} has {null_pct:.0f}% NULL values.",
                    "affected_rows": ns.get("nulls", 0),
                    "percentage": null_pct,
                })

        # Missing PK
        if not pk:
            issues.append({
                "table": tname,
                "column": None,
                "check": "missing_pk",
                "status": "warning",
                "severity": "medium",
                "message": f"Table '{tname}' has no primary key detected.",
                "affected_rows": 0,
                "percentage": 0,
            })

    # Calculate metrics
    avg_null_pct = total_null_pct / max(1, sum(len(t.get("columns", [])) for t in tables.values()))
    pk_coverage = (total_tables_with_pk / max(1, total_tables)) * 100
    completeness = max(0, 100 - avg_null_pct)
    empty_ratio = (total_empty / max(1, total_tables)) * 100
    overall = max(0, round((completeness * 0.4 + pk_coverage * 0.3 + max(0, 100 - empty_ratio * 2) * 0.3)))

    # Referential integrity check
    declared = mapping.get("declared_relationships", [])
    if declared:
        config, db_type, engine_url = _get_db_config(session)
        if engine_url:
            try:
                engine = create_engine(engine_url, pool_pre_ping=True, connect_args={"connect_timeout": 10})
                dialect = session.get("dialect", "postgresql")
                with engine.connect() as conn:
                    for rel in declared[:10]:
                        try:
                            src_table = _quote(rel["table_name"], dialect)
                            src_col = _quote(rel["column_name"], dialect)
                            tgt_table = _quote(rel["references_table"], dialect)
                            tgt_col = _quote(rel["references_column"], dialect)
                            sql = f"""
                                SELECT COUNT(*) FROM {src_table} s
                                WHERE s.{src_col} IS NOT NULL
                                AND NOT EXISTS (
                                    SELECT 1 FROM {tgt_table} t WHERE t.{tgt_col} = s.{src_col}
                                )
                            """
                            orphans = conn.execute(text(sql)).scalar() or 0
                            if orphans > 0:
                                issues.append({
                                    "table": rel["table_name"],
                                    "column": rel["column_name"],
                                    "check": "referential_integrity",
                                    "status": "error",
                                    "severity": "high",
                                    "message": f"{rel['table_name']}.{rel['column_name']} has {orphans} orphaned records referencing {rel['references_table']}.{rel['references_column']}.",
                                    "affected_rows": orphans,
                                    "percentage": 0,
                                    "details": {
                                        "references_table": rel["references_table"],
                                        "references_column": rel["references_column"],
                                    },
                                })
                        except Exception:
                            pass
            except Exception:
                pass

    return {
        "ok": True,
        "quality": {
            "overall": overall,
            "metrics": {
                "completeness": round(completeness, 1),
                "pk_coverage": round(pk_coverage, 1),
                "empty_tables": total_empty,
                "total_tables": total_tables,
                "avg_null_percentage": round(avg_null_pct, 1),
            },
            "issues": sorted(issues, key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.get("severity", "low"), 3)),
            "last_scan": mapping.get("generated_at", ""),
        },
    }


# =============================================================
# USER OVERRIDES (PK/FK/relationship editing)
# =============================================================

class OverrideRequest(BaseModel):
    action: str  # "set_pk" | "remove_pk" | "set_fk" | "remove_fk" | "confirm_relationship" | "reject_relationship"
    table: str
    column: str = ""
    references_table: str = ""
    references_column: str = ""


@router.post("/api/data/schema/override")
def apply_override(req: OverrideRequest, request: Request, user: dict = Depends(require_writable)):
    """Apply a user override to the schema (PK/FK/relationship changes)."""
    from core.semantic import is_valid_identifier

    session = get_session(user["user_id"])
    mapping = _get_schema_mapping(session, user["user_id"])
    if not mapping:
        raise HTTPException(status_code=400, detail="No schema extracted.")

    # Identifier hygiene (t.txt section 58): never trust raw table/column strings.
    # Multi-schema keys are schema-qualified (sales.employee): validate each
    # dot-separated segment, then _resolve_table_param enforces membership.
    def _ident_ok(value: str) -> bool:
        try:
            segs = [s for s in str(value or "").split(".") if s]
        except Exception:
            return False
        if not segs or len(segs) > 3:
            return False
        if len(".".join(segs)) != len(str(value or "")):
            return False
        return all(is_valid_identifier(s) for s in segs)

    for label, value in (
        ("table", req.table),
        ("column", req.column),
        ("references_table", req.references_table),
        ("references_column", req.references_column),
    ):
        if value and not _ident_ok(value):
            raise HTTPException(status_code=400, detail=f"Invalid {label} name: {value!r}")
    tables = mapping.get("tables", {})
    if isinstance(tables, dict) and tables:
        try:
            req.table = _resolve_table_param(req.table, tables)
        except HTTPException as _he:
            raise HTTPException(status_code=400, detail=f"Unknown table: {req.table!r}") from _he
        if req.table not in tables:
            raise HTTPException(status_code=400, detail=f"Unknown table: {req.table!r}")
        cols = tables[req.table].get("columns", []) if isinstance(tables[req.table], dict) else []
        if req.column and cols:
            if isinstance(cols, dict):
                col_names = set(cols.keys())
            else:
                col_names = {
                    (c.get("column") or c.get("name")) if isinstance(c, dict) else c
                    for c in cols
                }
            if req.column not in col_names:
                raise HTTPException(status_code=400, detail=f"Unknown column: {req.table}.{req.column}")

    db_name = session.get("db_name", "")
    overrides = _load_user_overrides(user["user_id"], db_name)

    if req.action == "set_pk":
        # Validate: check uniqueness
        config, db_type, engine_url = _get_db_config(session)
        if engine_url:
            try:
                engine = create_engine(engine_url, pool_pre_ping=True, connect_args={"connect_timeout": 10})
                dialect = session.get("dialect", "postgresql")
                with engine.connect() as conn:
                    q_t = _quote(req.table, dialect)
                    q_c = _quote(req.column, dialect)
                    total = conn.execute(text(f"SELECT COUNT(*) FROM {q_t}")).scalar() or 0
                    distinct = conn.execute(text(f"SELECT COUNT(DISTINCT {q_c}) FROM {q_t}")).scalar() or 0
                    nulls = conn.execute(text(f"SELECT COUNT(*) FROM {q_t} WHERE {q_c} IS NULL")).scalar() or 0
                    if distinct < total:
                        return {"ok": False, "detail": f"Column '{req.column}' has {total - distinct} duplicate values and may not be suitable as a primary key.", "warning": True}
                    if nulls > 0:
                        return {"ok": False, "detail": f"Column '{req.column}' contains {nulls} NULL values.", "warning": True}
            except Exception:
                pass
        overrides["pk"][req.table] = [req.column]
        _save_user_overrides(user["user_id"], db_name, overrides)
        # Also update the mapping
        if req.table in mapping.get("tables", {}):
            mapping["tables"][req.table]["primary_key"] = [req.column]
            path = session.get("schema_path")
            if path:
                path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        return {"ok": True, "message": f"Primary key set to {req.table}.{req.column}"}

    elif req.action == "remove_pk":
        if req.table in overrides.get("pk", {}):
            del overrides["pk"][req.table]
        _save_user_overrides(user["user_id"], db_name, overrides)
        if req.table in mapping.get("tables", {}):
            mapping["tables"][req.table]["primary_key"] = []
            path = session.get("schema_path")
            if path:
                path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        return {"ok": True, "message": f"Primary key removed from {req.table}"}

    elif req.action == "set_fk":
        rel_id = f"{req.table}.{req.column}->{req.references_table}.{req.references_column}"
        overrides["fk"][rel_id] = {
            "table": req.table, "column": req.column,
            "references_table": req.references_table, "references_column": req.references_column,
        }
        # Remove from rejected if present
        overrides["rejected"] = [r for r in overrides.get("rejected", []) if r != rel_id]
        _save_user_overrides(user["user_id"], db_name, overrides)
        return {"ok": True, "message": f"Foreign key set: {rel_id}"}

    elif req.action == "remove_fk":
        rel_id = f"{req.table}.{req.column}->{req.references_table}.{req.references_column}"
        overrides["fk"] = {k: v for k, v in overrides.get("fk", {}).items() if k != rel_id}
        _save_user_overrides(user["user_id"], db_name, overrides)
        return {"ok": True, "message": f"Foreign key removed: {rel_id}"}

    elif req.action == "confirm_relationship":
        rel_id = f"{req.table}.{req.column}->{req.references_table}.{req.references_column}"
        overrides["fk"][rel_id] = {"confirmed": True}
        overrides["rejected"] = [r for r in overrides.get("rejected", []) if r != rel_id]
        _save_user_overrides(user["user_id"], db_name, overrides)
        return {"ok": True, "message": f"Relationship confirmed: {rel_id}"}

    elif req.action == "reject_relationship":
        rel_id = f"{req.table}.{req.column}->{req.references_table}.{req.references_column}"
        if rel_id not in overrides.get("rejected", []):
            overrides.setdefault("rejected", []).append(rel_id)
        overrides["fk"] = {k: v for k, v in overrides.get("fk", {}).items() if k != rel_id}
        _save_user_overrides(user["user_id"], db_name, overrides)
        return {"ok": True, "message": f"Relationship rejected: {rel_id}"}

    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}")


@router.get("/api/data/overrides")
def get_overrides(request: Request, user: dict = Depends(get_current_user)):
    """Return all user overrides for the current database."""
    session = get_session(user["user_id"])
    db_name = session.get("db_name", "")
    overrides = _load_user_overrides(user["user_id"], db_name)
    return {"ok": True, "overrides": overrides}
