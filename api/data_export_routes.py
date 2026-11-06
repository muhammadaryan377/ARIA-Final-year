"""Authenticated, read-only HTML report and bounded CSV export for Data."""

import csv
import html
import io
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from sqlalchemy import create_engine, text

from api.data_routes import (
    _get_db_config,
    _get_schema_mapping,
    _quote,
    _resolve_table_param,
    get_data_quality,
    get_data_summary,
)
from core.config import get_session
from core.deps import get_current_user, reject_document_sql
from core.errors import sanitize_error

router = APIRouter()
MAX_CSV_ROWS = 5000


def _escape(value) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _safe_csv_cell(value):
    """Prevent spreadsheet formula evaluation for untrusted text cells."""
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        value = value.hex()
    if not isinstance(value, str):
        return value
    # Leading whitespace and control characters can hide a CSV formula.
    if value.lstrip(" \t\r\n").startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _build_html_report(session: dict, mapping: dict, summary: dict, quality: dict) -> str:
    """Build a self-contained, escaped document with no external resources."""
    now = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")
    source_name = _escape(session.get("db_name") or "Connected data source")
    source_type = _escape(session.get("source_type") or "Unknown")
    tables = summary.get("tables") or []
    info = quality.get("quality") or {}
    metrics = info.get("metrics") or {}
    issues = info.get("issues") or []
    score = info.get("overall")
    total = summary.get("total_tables", len(tables))

    table_rows = []
    for table in tables:
        name = str(table.get("name") or "")
        table_rows.append(
            "<tr><td>" + _escape(name) + "</td><td>" + _escape(table.get("row_count", "N/A"))
            + "</td><td>" + _escape(table.get("column_count", 0))
            + "</td><td>" + ("Yes" if table.get("has_pk") else "No") + "</td></tr>"
        )

    column_sections = []
    for table_name, table_info in (mapping.get("tables") or {}).items():
        cells = []
        for col in (table_info.get("columns") or []):
            cells.append("<tr><td>" + _escape(col.get("column", ""))
                         + "</td><td>" + _escape(col.get("data_type", col.get("type", "")))
                         + "</td></tr>")
        column_sections.append(
            "<section><h3>" + _escape(table_name) + "</h3>"
            + "<table><thead><tr><th>Column</th><th>Data type</th></tr></thead><tbody>"
            + ("".join(cells) or "<tr><td colspan='2'>No columns detected</td></tr>")
            + "</tbody></table></section>"
        )

    issue_rows = []
    for issue in issues[:100]:
        issue_rows.append(
            "<tr><td>" + _escape(issue.get("severity", ""))
            + "</td><td>" + _escape(issue.get("table", ""))
            + "</td><td>" + _escape(issue.get("message", "")) + "</td></tr>"
        )
    omitted = f"<p>Showing first 100 of {len(issues)} issues.</p>" if len(issues) > 100 else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ARIA Data Report</title>
<style>
:root {{ color-scheme:light; }} body {{ background:#f8f9f6;color:#1e2b27;font:14px/1.6 system-ui,sans-serif;margin:0; }}
main {{ max-width:950px;margin:35px auto;padding:30px;background:#fff;border:1px solid #e5e9e3;border-radius:16px; }}
h1 {{ margin:0;font-size:28px; }} h2 {{ margin-top:32px;color:#285743;border-bottom:1px solid #e4e9e4;padding-bottom:7px; }}
h3 {{ margin:20px 0 7px;font-size:15px; }} .muted {{ color:#61736a; }}
.stats {{ display:flex;gap:12px;flex-wrap:wrap; }} .stat {{ flex:1;min-width:140px;padding:14px;border:1px solid #e4e9e4;border-radius:10px;background:#f8faf7; }}
.stat b {{ display:block;font-size:23px;color:#285743; }} table {{ width:100%;border-collapse:collapse;margin:12px 0; }}
th,td {{ padding:9px;text-align:left;border-bottom:1px solid #e5e9e3;overflow-wrap:anywhere; }} th {{ background:#eef4ed;color:#33493b; }}
footer {{ margin-top:35px;color:#61736a;font-size:12px; }} @media print {{ body {{ background:white; }} main {{ border:0;margin:0;padding:0; }} }}
</style></head><body><main>
<h1>ARIA Data Report</h1><p class="muted">Generated {now} · {source_name} · {source_type}</p>
<h2>Overview</h2><div class="stats">
<div class="stat">Tables<b>{_escape(total)}</b></div>
<div class="stat">Data quality score<b>{_escape(score) if score is not None else 'N/A'}/100</b></div>
<div class="stat">Completeness<b>{_escape(metrics.get('completeness', 'N/A'))}%</b></div>
<div class="stat">Quality issues<b>{len(issues)}</b></div></div>
<h2>Table Summary</h2><table><thead><tr><th>Table</th><th>Rows</th><th>Columns</th><th>Primary key</th></tr></thead>
<tbody>{''.join(table_rows) or '<tr><td colspan="4">No tables found</td></tr>'}</tbody></table>
<h2>Data Quality Findings</h2><table><thead><tr><th>Severity</th><th>Table</th><th>Finding</th></tr></thead>
<tbody>{''.join(issue_rows) or '<tr><td colspan="3">No issues detected</td></tr>'}</tbody></table>{omitted}
<h2>Schema / Column Definitions</h2>{''.join(column_sections) or '<p>No schema columns available.</p>'}
<footer>Generated from your current authenticated ARIA data connection. This report contains metadata and quality findings, not raw records.</footer>
</main></body></html>"""


def _export_csv(session: dict, mapping: dict, table: str) -> Response:
    table_key = _resolve_table_param(table, mapping.get("tables", {}))
    cols = [item["column"] for item in mapping["tables"][table_key].get("columns", [])]
    if not cols:
        raise HTTPException(status_code=400, detail="This table has no exportable columns.")

    config, db_type, engine_url = _get_db_config(session)
    if not engine_url and db_type == "sqlite":
        engine_url = session.get("db_uri")
        if not engine_url and isinstance(config, dict):
            for ext in (".db", ".sqlite", ".sqlite3"):
                if ext in config:
                    engine_url = f"sqlite:///{config[ext]}"
                    break
    if not engine_url:
        raise HTTPException(status_code=400, detail="No tabular database is connected.")

    dialect = session.get("dialect") or db_type or "postgresql"
    quote_cols = ", ".join(_quote(col, dialect) for col in cols)
    statement = text(
        f"SELECT {quote_cols} FROM {_quote(table_key, dialect)} ORDER BY 1 LIMIT :row_limit"
    )
    engine = None
    try:
        engine = create_engine(engine_url, pool_pre_ping=True)
        with engine.connect() as conn:
            rows = conn.execute(statement, {"row_limit": MAX_CSV_ROWS + 1}).fetchall()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"CSV export failed: {sanitize_error(exc)}") from exc
    finally:
        if engine is not None:
            engine.dispose()

    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(cols)
    for row in rows[:MAX_CSV_ROWS]:
        writer.writerow([_safe_csv_cell(value) for value in row])

    # Fixed ASCII filename prevents header injection from untrusted table names.
    filename = f"aria-table-{datetime.now(timezone.utc):%Y%m%d}.csv"
    return Response(
        content="\ufeff" + output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "X-ARIA-Export-Row-Limit": str(MAX_CSV_ROWS),
            "X-ARIA-Export-Truncated": str(len(rows) > MAX_CSV_ROWS).lower(),
        },
    )


@router.get("/api/data/export")
def export_my_data(
    request: Request,
    format: Literal["html", "csv"] = Query("html"),
    table: str | None = Query(default=None, max_length=255),
    user: dict = Depends(get_current_user),
):
    """Export current user's data summary (HTML) or table records (CSV)."""
    session = get_session(user["user_id"])
    reject_document_sql(session)
    mapping = _get_schema_mapping(session, user["user_id"])
    if not mapping or not mapping.get("tables"):
        raise HTTPException(status_code=400, detail="Connect a tabular data source and extract its schema first.")

    if format == "csv":
        if not table:
            raise HTTPException(status_code=400, detail="Select a table to export as CSV.")
        return _export_csv(session, mapping, table)

    summary = get_data_summary(request, user)
    quality = get_data_quality(request, user)
    report = _build_html_report(session, mapping, summary, quality)
    return HTMLResponse(
        content=report,
        headers={
            "Content-Disposition": f'attachment; filename="aria-data-report-{datetime.now(timezone.utc):%Y%m%d}.html"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'",
        },
    )
