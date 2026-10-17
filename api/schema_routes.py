"""Schema Agent routes: schema extraction and retrieval."""

import json
import re
import types
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

import schema_agent
from core.config import SCHEMA_DIR, get_session, save_session
from core.deps import get_current_user, require_data, require_writable

router = APIRouter()


def _safe_db(name):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(name))[:64] or "db"


def _extract_schema_impl(request: Request, user: dict, db_type: str = "postgresql") -> dict:
    """Run the Schema Agent against the connected source for `db_type`.

    For file sources (CSV / PDF) the mapping was already built at connect time,
    so this returns the saved mapping counts.
    """
    session = get_session(user["user_id"])
    require_data(request)

    if session["source_type"] != "relational":
        _sp = session.get("schema_path")
        if _sp is None:
            from fastapi import HTTPException as _HTTPException
            raise _HTTPException(
                status_code=400,
                detail="This source is a text document — use Document Chat "
                       "(Copilot) instead of schema analytics.",
            )
        mapping = json.loads(Path(_sp).read_text(encoding="utf-8"))
        return {
            "ok": True,
            "provider": session["provider_name"],
            "snapshot_id": mapping.get("snapshot_id"),
            "tables": len(mapping.get("tables", {})),
            "declared_relationships": len(mapping.get("declared_relationships", [])),
            "inferred_relationships": len(mapping.get("inferred_relationships", [])),
            "source": session.get("file_type") or "file",
        }

    session_db_type = session.get("db_type") or "postgresql"
    if db_type != session_db_type:
        raise HTTPException(
            status_code=400,
            detail=f"Schema extraction endpoint /api/extract_schema/{db_type} requires a {db_type} "
                   f"connection, but the session is connected to {session_db_type}. "
                   f"Re-connect via /api/connect/{db_type} first.",
        )

    config = dict(session["db"])
    if "password" in config and config["password"] is not None:
        from core.secret_box import decrypt as _decrypt_secret
        config["password"] = _decrypt_secret(config["password"])
    schema_filter = config["db"] if db_type == "mysql" else config["db_schema"]
    try:
        conn = schema_agent.get_connection(types.SimpleNamespace(**config))
        try:
            # Dynamic schema resolution (generic, no database names):
            # - single-schema DBs: keep the configured schema when it has
            #   tables; otherwise fall back to populated schemas.
            # - multi-schema DBs: map ALL populated schemas with
            #   schema-qualified table names so goals can span schemas.
            #   When the requested schema is empty, default to ALL populated
            #   schemas (never to nothing).
            multi_schema = False
            if db_type == "postgresql":
                try:
                    mode = schema_agent.detect_schema_mode(conn, requested=schema_filter)
                    multi_schema = mode["mode"] == "multi"
                    if multi_schema:
                        schema_filter = "*"
                    elif not schema_agent.schema_has_tables(conn, schema_filter):
                        try:
                            _all = schema_agent.resolve_schemas_for_requested(
                                conn, schema_filter)
                        except Exception:
                            _all = []
                        if len(_all or []) > 1:
                            multi_schema = True
                            schema_filter = "*"
                        else:
                            alt = schema_agent.find_schema_with_tables(conn)
                            if alt and alt != schema_filter:
                                schema_filter = alt
                except Exception:
                    pass
            # Item 5: seed drift detection from the previous run of this DB.
            previous_mapping = None
            try:
                prev_latest = SCHEMA_DIR / f"user_{user['user_id']}" / (
                    f"schema_mapping_{_safe_db(session['db_name'])}_latest.json"
                )
                if prev_latest.exists():
                    previous_mapping = json.loads(prev_latest.read_text(encoding="utf-8"))
            except Exception:
                previous_mapping = None
            if multi_schema:
                mapping = schema_agent.build_schema_mapping_all(
                    conn, db_type=db_type, llm=session.get("provider"),
                    previous_mapping=previous_mapping,
                )
            else:
                mapping = schema_agent.build_schema_mapping(
                    conn, schema_filter, db_type=db_type, llm=session.get("provider"),
                    previous_mapping=previous_mapping,
                )
            if not mapping.get("tables"):
                hint = ""
                if db_type == "postgresql":
                    try:
                        alt = schema_agent.find_schema_with_tables(conn)
                        if alt and alt != schema_filter:
                            hint = f" (tables were found in schema '{alt}' instead)"
                    except Exception:
                        pass
                raise HTTPException(
                    status_code=400,
                    detail=f"No tables found in {db_type} schema '{schema_filter}'. "
                           f"The schema is empty or does not exist{hint}; nothing was mapped.",
                )
            mapping["database"] = session["db_name"]
        finally:
            conn.close()
    except SystemExit as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        from core.errors import sanitize_error as _san
        raise HTTPException(status_code=500, detail=f"Schema extraction failed: {_san(exc)}")

    user_schema_dir = SCHEMA_DIR / f"user_{user['user_id']}"
    output_path, snapshot_path, latest_path, snapshot_id = schema_agent.get_output_paths(
        str(user_schema_dir / "schema_mapping.json"),
        db_name=session["db_name"],
        snapshot_id=mapping.get("snapshot_id"),
    )
    mapping["snapshot_id"] = snapshot_id
    for path in (output_path, snapshot_path, latest_path):
        path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    session["schema_path"] = latest_path
    session["schema_status"] = "ok" if mapping.get("tables") else "empty"
    save_session(user["user_id"], session)

    summary = mapping.get("summary", {})

    # Build a short natural-language summary of the database.
    db_name = session["db_name"]
    n_used = summary.get("tables_used", len(mapping["tables"]))
    n_total = summary.get("total_db_tables", len(mapping["tables"]))
    dropped = summary.get("tables_dropped_empty", [])
    n_declared = len(mapping["declared_relationships"])
    n_inferred = len(mapping["inferred_relationships"])
    descs = [
        mapping["tables"][t].get("description", "")
        for t in mapping["tables"]
        if mapping["tables"][t].get("description")
    ]

    parts = [f'"{db_name}" contains {n_used} tables']
    if dropped:
        parts[0] += f' ({len(dropped)} empty table{"s" if len(dropped) > 1 else ""} dropped: {", ".join(dropped)})'
    if descs:
        parts.append("covering " + "; ".join(descs[:5]))
    if n_declared:
        fk_s = "s" if n_declared > 1 else ""
        parts.append(f"{n_declared} declared foreign key{fk_s}")
    if n_inferred:
        fk_s = "s" if n_inferred > 1 else ""
        parts.append(f"{n_inferred} inferred relationship{fk_s}")
    no_pk = summary.get("tables_without_pk", [])
    if no_pk:
        parts.append(f"no primary key on {', '.join(no_pk)}")

    short_summary = ". ".join(parts) + "."

    return {
        "ok": True,
        "provider": session["provider_name"],
        "snapshot_id": mapping["snapshot_id"],
        "database": session["db_name"],
        "short_summary": short_summary,
        "tables_used": summary.get("tables_used", len(mapping["tables"])),
        "total_db_tables": summary.get("total_db_tables", len(mapping["tables"])),
        "tables_dropped_empty": summary.get("tables_dropped_empty", []),
        "declared_relationships": n_declared,
        "declared_fk_list": [
            {
                "from": f"{r['table_name']}.{r['column_name']}",
                "to": f"{r['references_table']}.{r['references_column']}",
            }
            for r in mapping["declared_relationships"]
        ],
        "inferred_relationships": n_inferred,
        "inferred_fk_list": [
            {
                "from": f"{r['table']}.{r['column']}",
                "to": f"{r['references_table']}.{r['references_column']}",
                "confidence": r.get("confidence", "heuristic"),
                "confidence_score": r.get("confidence_score"),
                "confidence_band": r.get("confidence_band"),
                "review_status": r.get("review_status"),
                "relationship_state": r.get("relationship_state"),
                "relationship_type": r.get("relationship_type"),
                "ambiguous": r.get("ambiguous"),
                "self_referencing": r.get("self_referencing"),
                "cardinality": r.get("cardinality"),
                "evidence": r.get("evidence", []),
                "evidence_detail": r.get("evidence_detail"),
                "note": r.get("note", ""),
            }
            for r in mapping["inferred_relationships"]
        ],
        "tables_with_pk": summary.get("tables_with_pk", []),
        "tables_without_pk": no_pk,
        "llm_reasoning": mapping.get("reasoning"),
        "drift": mapping.get("drift"),
    }


@router.post("/api/extract_schema")
def extract_schema(request: Request, user: dict = Depends(require_writable)):
    """Run the Schema Agent against the connected source (db type from session)."""
    session = get_session(user["user_id"])
    return _extract_schema_impl(request, user, session.get("db_type") or "postgresql")


@router.post("/api/extract_schema/postgresql")
def extract_schema_postgresql(request: Request, user: dict = Depends(require_writable)):
    """Run the Schema Agent for a PostgreSQL source (explicit route)."""
    return _extract_schema_impl(request, user, "postgresql")


@router.post("/api/extract_schema/mysql")
def extract_schema_mysql(request: Request, user: dict = Depends(require_writable)):
    """Run the Schema Agent for a MySQL source (explicit route)."""
    return _extract_schema_impl(request, user, "mysql")


@router.get("/api/schema")
def get_schema(request: Request, user: dict = Depends(get_current_user)):
    """Return the current schema mapping (tables, columns, relationships)."""
    session = get_session(user["user_id"])
    require_data(request)
    path = session.get("schema_path")
    if not path:
        raise HTTPException(status_code=404, detail="No schema path configured. Connect a database first.")
    path = Path(str(path))
    if not path.exists():
        # Auto-extract schema when reconnecting — the DB was never disconnected,
        # so the schema file just needs to be rebuilt from the live connection.
        try:
            _extract_schema_impl(request, user, session.get("db_type") or "postgresql")
        except Exception:
            # If extraction fails, return graceful message instead of blocking user.
            return {"ok": False, "detail": "Schema not found and could not be rebuilt automatically. The database connection is active, but schema extraction may require a manual retry or a different database. Try reconnecting or selecting a source with declared relationships."}
    # Re-read session schema_path after extraction (file path may have changed).
    updated_path = Path(str(session.get("schema_path", path)))
    if not updated_path.exists():
        return {"ok": False, "detail": "Schema file still missing after extraction attempt. The database is connected but may be empty or use a different schema name."}
    return {"ok": True, "schema": json.loads(updated_path.read_text(encoding="utf-8"))}
