"""ARIA Reporting Agent (Phase 3).

Builds a styled, self-contained HTML report from live payloads only.
No numbers are invented: every finding, table cell, chart point and SQL
string comes from the caller-supplied payload.

PDF path: pdfplumber is read-only, so no server-side PDF rendering.
The backend serves printable HTML (GET /api/intelligence/reports/{id}/html);
the frontend offers Download HTML + print-to-PDF (window.print).
"""

from __future__ import annotations

import hashlib
import html as _html
import json
from datetime import datetime, timezone
from typing import Any


MAX_TABLE_ROWS = 25
MAX_CHART_POINTS = 12


def fingerprint_data(payload: dict) -> str:
    """Stable fingerprint of the report inputs (proves what was rendered)."""
    try:
        canonical = json.dumps(payload or {}, sort_keys=True, default=str, ensure_ascii=False)
    except Exception:
        canonical = str(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _esc(value: Any) -> str:
    if value is None:
        return ""
    return _html.escape(str(value), quote=True)


def _num(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        f = float(value)
        if f != f or f in (float("inf"), float("-inf")):  # NaN / inf guard
            return None
        return f
    except (TypeError, ValueError):
        return None


def _fmt_num(value: Any) -> str:
    f = _num(value)
    if f is None:
        return _esc(value) if value not in (None, "") else "—"
    if abs(f) >= 1_000_000:
        return f"{f / 1_000_000:.1f}M"
    if abs(f) >= 1_000:
        return f"{f / 1_000:.1f}K"
    return f"{f:,.2f}"


def _svg_bar_chart(title: str, points: list[dict]) -> str:
    """Embedded SVG horizontal bar chart (no JS, no external deps)."""
    clean: list[tuple[str, float]] = []
    for p in (points or [])[:MAX_CHART_POINTS]:
        if not isinstance(p, dict):
            continue
        label = str(p.get("label") or p.get("name") or p.get("category") or "")[:40]
        v = _num(p.get("value", p.get("v", p.get("total"))))
        if not label or v is None:
            continue
        clean.append((label, v))
    if not clean:
        return ""
    vmax = max(v for _, v in clean) or 1.0
    vmin = min(0.0, min(v for _, v in clean))
    span = (vmax - vmin) or 1.0
    rows: list[str] = []
    y = 0
    bar_h = 18
    gap = 26
    width = 560
    label_w = 150
    for label, v in clean:
        w = max(2.0, ((v - vmin) / span) * (width - label_w - 90.0))
        fill = "#2f6fad" if v >= 0 else "#c2410c"
        rows.append(
            f'<text x="0" y="{y + 13}" font-size="11" fill="#334155">{_esc(label)}</text>'
            f'<rect x="{label_w}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="4" fill="{fill}" opacity="0.85"/>'
            f'<text x="{label_w + w + 6:.1f}" y="{y + 13}" font-size="11" fill="#0f2440">{_esc(_fmt_num(v))}</text>'
        )
        y += bar_h + gap - 12
    height = y + 6
    return (
        f'<figure class="chart"><figcaption>{_esc(title)}</figcaption>'
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="{_esc(title)} bar chart">{"".join(rows)}</svg></figure>'
    )


def _table_html(title: str, columns: list, rows: list) -> str:
    cols = [str(c) for c in (columns or [])][:8]
    if not cols:
        return ""
    body_rows: list[str] = []
    truncated = len(rows or []) > MAX_TABLE_ROWS
    for r in (rows or [])[:MAX_TABLE_ROWS]:
        if isinstance(r, dict):
            cells = [_esc(r.get(c, "")) for c in cols]
        elif isinstance(r, (list, tuple)):
            cells = [_esc(v) for v in list(r)[: len(cols)]]
        else:
            cells = [_esc(r)] + [""] * (len(cols) - 1)
        body_rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    if not body_rows:
        return ""
    head = "".join(f"<th scope=\"col\">{_esc(c)}</th>" for c in cols)
    note = (
        f"<p class=\"muted\">Showing first {MAX_TABLE_ROWS} of {len(rows)} rows.</p>"
        if truncated
        else ""
    )
    return (
        f'<section class="block"><h3>{_esc(title)}</h3>'
        f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body_rows)}</tbody></table></div>{note}</section>"
    )


def build_report(kind: str, payload: dict) -> dict:
    """Render a styled HTML report from a live payload.

    kind: report family label (e.g. "dashboard", "analytics", "intelligence").
    payload: plain dict with any of: title, description, date_range,
      template, dataset_label, findings [{title, summary}], tables
      [{title, columns, rows}], charts [{title, points [{label, value}]}],
      sql_queries [str], sections [str].
    """
    payload = payload or {}
    kind_label = str(kind or payload.get("kind") or "report").strip() or "report"
    title = str(payload.get("title") or "ARIA Report").strip() or "ARIA Report"
    description = str(payload.get("description") or "").strip()
    date_range = str(payload.get("date_range") or "").strip()
    template = str(payload.get("template") or "").strip()
    dataset_label = str(payload.get("dataset_label") or payload.get("dataset") or "").strip()

    findings = [f for f in (payload.get("findings") or []) if isinstance(f, dict)]
    tables = [t for t in (payload.get("tables") or []) if isinstance(t, dict)]
    charts = [c for c in (payload.get("charts") or []) if isinstance(c, dict)]
    sql_queries = [str(s) for s in (payload.get("sql_queries") or []) if str(s).strip()]
    sections = [str(s) for s in (payload.get("sections") or []) if str(s).strip()]

    generated_at = datetime.now(timezone.utc).isoformat()
    fingerprint = fingerprint_data(payload)

    # ---- findings ----
    if findings:
        finding_items = "".join(
            f"<li><strong>{_esc(f.get('title') or 'Finding')}</strong>"
            + (f" — {_esc(f.get('summary') or f.get('detail') or '')}" if (f.get("summary") or f.get("detail")) else "")
            + (f" <span class=\"val\">({_esc(f.get('value_display') or f.get('value') or f.get('metric') or '')})</span>"
               if (f.get("value_display") or f.get("value") or f.get("metric")) else "")
            + "</li>"
            for f in findings[:50]
        )
        findings_html = f"<section class=\"block\"><h2>Findings</h2><ul class=\"findings\">{finding_items}</ul></section>"
    else:
        findings_html = (
            "<section class=\"block\"><h2>Findings</h2>"
            "<p class=\"muted\">No findings were supplied with this report — "
            "nothing was invented to fill this section.</p></section>"
        )

    # ---- charts (embedded SVG) ----
    chart_html = "".join(
        _svg_bar_chart(
            str(c.get("title") or "Chart"),
            c.get("points") or c.get("data") or [],
        )
        for c in charts[:6]
    )
    charts_section = (
        f"<section class=\"block\"><h2>Charts</h2>{chart_html}</section>" if chart_html else ""
    )

    # ---- tables ----
    tables_html = "".join(
        _table_html(
            str(t.get("title") or "Data"),
            t.get("columns") or t.get("headers") or [],
            t.get("rows") or t.get("data") or [],
        )
        for t in tables[:6]
    )

    # ---- evidence SQL appendix ----
    if sql_queries:
        sql_blocks = "".join(f"<pre>{_esc(q)}</pre>" for q in sql_queries[:20])
        sql_html = f"<section class=\"block\"><h2>Evidence SQL</h2>{sql_blocks}</section>"
    else:
        sql_html = ""

    sections_html = (
        "<section class=\"block\"><h2>Sections</h2><ul>"
        + "".join(f"<li>{_esc(s)}</li>" for s in sections[:30])
        + "</ul></section>"
        if sections
        else ""
    )

    desc_html = f"<p class=\"desc\">{_esc(description)}</p>" if description else ""
    meta_bits = []
    if date_range:
        meta_bits.append(f"<span>Date range: {_esc(date_range)}</span>")
    if template:
        meta_bits.append(f"<span>Template: {_esc(template)}</span>")
    if dataset_label:
        meta_bits.append(f"<span>Dataset: {_esc(dataset_label)}</span>")
    meta_html = f"<p class=\"meta\">{' · '.join(meta_bits)}</p>" if meta_bits else ""

    body_html = (
        f"<header><p class=\"brand\">ARIA</p><h1>{_esc(title)}</h1>"
        f"<p class=\"kind\">{_esc(kind_label.title())} report</p>{desc_html}{meta_html}</header>"
        f"{sections_html}{findings_html}{charts_section}{tables_html}{sql_html}"
        f"<footer><p>Generated at {_esc(generated_at)} · Data fingerprint {_esc(fingerprint)}</p>"
        f"<p class=\"muted\">All numbers come from live payloads — nothing was invented. "
        f"To keep a PDF: use your browser's Print → Save as PDF.</p></footer>"
    )

    page = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{_esc(title)}</title>"
        "<style>"
        "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:760px;margin:32px auto;padding:0 20px;color:#0f2440;line-height:1.55}"
        ".brand{color:#1d4ed8;font-weight:800;letter-spacing:3px;margin:0}"
        "h1{margin:4px 0 2px;font-size:26px}.kind{color:#64748b;margin:0 0 8px}"
        ".desc{color:#334155}.meta{color:#64748b;font-size:13px}"
        ".block{margin-top:22px;border-top:1px solid #e2e8f0;padding-top:14px}"
        "h2{font-size:16px;margin:0 0 8px}h3{font-size:14px;margin:0 0 8px}"
        ".findings{padding-left:18px}.val{color:#475569}"
        ".chart figcaption{font-weight:600;font-size:13px;margin-bottom:6px}"
        ".table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:12px}"
        "th,td{border:1px solid #e2e8f0;padding:6px 8px;text-align:left}"
        "th{background:#f1f5f9}pre{background:#0f172a;color:#e2e8f0;padding:10px;border-radius:8px;overflow-x:auto;font-size:11.5px}"
        "footer{margin-top:26px;border-top:1px solid #e2e8f0;padding-top:12px;font-size:12px;color:#475569}"
        ".muted{color:#64748b;font-size:12px}"
        "@media print{body{margin:0;max-width:none}header{page-break-after:avoid}.block{page-break-inside:avoid}}"
        "</style></head><body>"
        + body_html
        + "</body></html>"
    )

    return {
        "title": title,
        "kind": kind_label,
        "html": page,
        "fingerprint": fingerprint,
        "generated_at": generated_at,
        "finding_count": len(findings),
        "table_count": len([t for t in tables if t.get("rows") or t.get("data")]),
        "chart_count": len(charts),
        "sql_count": len(sql_queries),
    }


def intel_report_to_payload(stored: dict) -> dict:
    """Convert a stored Intelligence report into a build_report payload.

    Only uses values already present in the stored report content —
    never invents numbers.
    """
    stored = stored or {}
    content = stored.get("content") or {}

    findings: list[dict] = []
    latest = content.get("latest_analysis") or {}
    if latest.get("title") or latest.get("summary"):
        findings.append({
            "title": latest.get("title") or "Latest analysis",
            "summary": latest.get("summary") or "",
            "metric": latest.get("metric") or "",
        })
    for d in (content.get("draft_items") or stored.get("draft_items") or [])[:20]:
        if isinstance(d, dict) and (d.get("title") or d.get("summary")):
            findings.append({
                "title": d.get("title") or "Draft item",
                "summary": d.get("summary") or "",
                "metric": d.get("metric") or "",
            })
    for h in (content.get("hypotheses") or [])[:10]:
        if isinstance(h, dict) and h.get("title"):
            findings.append({"title": h.get("title"), "summary": ""})
    for g in (content.get("goals") or [])[:10]:
        if isinstance(g, dict) and (g.get("title") or g.get("metric")):
            cur = g.get("current", g.get("current_value", ""))
            tgt = g.get("target", g.get("target_value", ""))
            val = f"{cur} / {tgt}" if (cur != "" or tgt != "") else ""
            findings.append({
                "title": g.get("title") or "Goal",
                "summary": str(g.get("status") or ""),
                "value_display": str(val),
                "metric": str(g.get("metric") or ""),
            })
    for s in (content.get("scenarios") or [])[:10]:
        if isinstance(s, dict) and (s.get("title") or s.get("variable")):
            findings.append({
                "title": s.get("title") or str(s.get("variable") or "Scenario"),
                "summary": f"{s.get('change_value', '')} {s.get('change_type', '')}".strip(),
            })

    tables: list[dict] = []
    goals = content.get("goals") or []
    if goals:
        tables.append({
            "title": "Goals",
            "columns": ["Title", "Metric", "Current", "Target", "Status"],
            "rows": [
                [
                    g.get("title") or "",
                    g.get("metric") or "",
                    g.get("current", g.get("current_value", "")),
                    g.get("target", g.get("target_value", "")),
                    g.get("status") or "",
                ]
                for g in goals[:MAX_TABLE_ROWS]
                if isinstance(g, dict)
            ],
        })

    sql_queries: list[str] = []
    for key in ("sections",):
        for entry in (stored.get(key) or []):
            if isinstance(entry, dict):
                for sk in ("sql", "query", "evidence_sql"):
                    if entry.get(sk):
                        sql_queries.append(str(entry[sk]))

    return {
        "title": stored.get("name") or "ARIA Report",
        "description": stored.get("description") or "",
        "date_range": stored.get("date_range") or "",
        "template": stored.get("template") or "",
        "sections": [
            s.get("name") if isinstance(s, dict) else str(s)
            for s in (stored.get("sections") or [])
        ],
        "findings": findings,
        "tables": tables,
        "charts": [],
        "sql_queries": sql_queries,
    }


# ---------------------------------------------------------------------------
# Phase 4: scheduled digests (additive-only).
# ---------------------------------------------------------------------------

def digest_to_definition(digest: dict) -> dict:
    """Report definition for a digest schedule (persisted for re-run).

    Plain pass-through of the schedule's own name/sections/template plus a
    digest marker — no numbers are invented here.
    """
    digest = digest or {}
    return {
        "name": digest.get("name") or "Digest",
        "description": digest.get("description") or "",
        "date_range": None,
        "sections": list(digest.get("sections") or []),
        "template": digest.get("template") or "professional",
        "digest_id": digest.get("id"),
        "digest_name": digest.get("name"),
        "digest_frequency": digest.get("frequency"),
        "scheduled": True,
    }


def build_digest_report(digest: dict, stored_report: dict) -> dict:
    """Render a digest run as styled HTML, reusing the report engine.

    Only uses values already present in the stored report / schedule —
    never invents numbers. Returns the same shape as build_report().
    """
    digest = digest or {}
    stored = stored_report or {}
    payload = intel_report_to_payload(stored)
    label = str(digest.get("frequency") or stored.get("digest_frequency") or "scheduled")
    digest_name = str(digest.get("name") or stored.get("digest_name") or payload.get("title") or "Digest")
    payload["title"] = digest_name
    payload["description"] = str(
        payload.get("description") or stored.get("description") or ""
    ).strip()
    sections = list(payload.get("sections") or [])
    marker = f"Scheduled digest ({label})"
    if marker not in sections:
        sections = [*sections, marker]
    payload["sections"] = sections
    return build_report("digest", payload)
