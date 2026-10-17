"""Prescription API: evidence-gated action workspace endpoints.

GET  /api/prescriptions/context   dynamic metrics/drivers/dimensions/dates
POST /api/prescriptions/analyze   ranked prescriptions or honest refusal
POST /api/prescriptions/scenario  read-only scenario recomputation
POST /api/prescriptions/save      persist a structured prescription
GET  /api/prescriptions/saved     list saved prescriptions
DELETE /api/prescriptions/saved/{sid}  remove one
"""

import logging
import time
from datetime import datetime

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from core.config import get_session, save_session
from core.deps import get_current_user, require_writable
from api.hypothesis_routes import _engine_for_session

router = APIRouter(tags=["prescriptions"])

log = logging.getLogger("aria.prescription")


# ---------------------------------------------------------------------------
# Leak audit: internal keys never leave the server. Cards keep the
# display-safe allowlist (id/title/summary/action/confidence/
# confidence_label/risk/display/why_summary/details/impact_note/
# limitations/assumptions/evidence_pass) plus the operational fields the
# What-If playground needs (driver/metric/table/rank/score/impact/evidence/
# scenario_options/success_metric/review_*). Everything stripped is logged
# server-side via log.debug rather than sent.
# ---------------------------------------------------------------------------
_RX_INTERNAL_KEYS = frozenset({
    "scenario_label", "dedupe_key", "risk_reasons", "score_breakdown",
    "join_ref", "sql", "query", "debug", "traceback", "trace",
})
_TOP_INTERNAL_KEYS = frozenset({"fit", "sql", "query", "debug", "traceback", "trace",
                                "score_breakdown", "join_ref"})


def _strip_prescription_response(out):
    """Strip internal keys from an analyze/scenario payload (in place).

    Returns (cleaned, stripped_names). Variant labels inside
    scenario_options are renamed scenario_label -> label so the What-If
    detail keeps its options without leaking the internal key name.
    """
    stripped: list = []
    if not isinstance(out, dict):
        return out, stripped
    for k in list(out.keys()):
        if k in _TOP_INTERNAL_KEYS or "sql" in str(k).lower():
            stripped.append(str(k))
            out.pop(k, None)
    pres = out.get("prescriptions")
    if isinstance(pres, list):
        for r in pres:
            if not isinstance(r, dict):
                continue
            for k in list(r.keys()):
                if k in _RX_INTERNAL_KEYS or "sql" in str(k).lower():
                    stripped.append(f"prescription.{k}")
                    r.pop(k, None)
            opts = r.get("scenario_options")
            if isinstance(opts, list):
                for opt in opts:
                    if isinstance(opt, dict) and "scenario_label" in opt:
                        opt["label"] = opt.pop("scenario_label")
                        stripped.append("prescription.scenario_options.scenario_label->label")
    scens = out.get("scenarios")
    if isinstance(scens, list):
        for s in scens:
            if not isinstance(s, dict):
                continue
            for k in list(s.keys()):
                if k in _RX_INTERNAL_KEYS or "sql" in str(k).lower():
                    stripped.append(f"scenario.{k}")
                    s.pop(k, None)
    if stripped:
        log.debug("prescription response stripped internal keys: %s",
                  sorted(set(stripped)))
    return out, stripped


class SegmentRef(BaseModel):
    field: str | None = None
    value: str | int | float | None = None


class AnalyzeContext(BaseModel):
    origin: str | None = None
    insight_id: str | None = None
    metric: str | None = None
    driver: str | None = None
    effect: float | None = None
    confidence: str | None = None


class AnalyzeRequest(BaseModel):
    context: AnalyzeContext | None = None
    metric: str | None = None
    driver: str | None = None
    dimensions: list[str] | None = None
    segment: SegmentRef | None = None
    table: str | None = None
    question: str | None = None


class ScenarioChange(BaseModel):
    field: str
    operation: str
    value: float


class ScenarioFilter(BaseModel):
    field: str
    operator: str = "equals"
    value: str | int | float


class ScenarioJoin(BaseModel):
    table: str
    self_col: str
    other_col: str


class ScenarioRequest(BaseModel):
    table: str
    metric: str
    driver: str
    changes: list[ScenarioChange] = []
    filters: list[ScenarioFilter] | None = None
    join: ScenarioJoin | None = None


class SaveRequest(BaseModel):
    prescription: dict = {}
    question: str | None = None


def _attach_prescription_verdict(out):
    """Best-effort Verification Agent hookup for prescriptions (never blocks).

    Attaches ``out["verdict"]``. On any failure attaches an honest
    not-verified verdict and still returns the data untouched.
    """
    try:
        from verification.agent import verify_prescription
        verdict = verify_prescription(out if isinstance(out, dict) else {}, {})
    except Exception:
        verdict = {"verified": False, "checks": [],
                   "warnings": ["verification unavailable"]}
    try:
        out["verdict"] = verdict
    except Exception:
        pass
    return out


def _setup(user):
    """Shared loader: mapping + tables + engine for any supported source."""
    from core.schema_intel import SchemaIntelligence
    session = get_session(user["user_id"])
    schema_path = session.get("schema_path")
    if not schema_path:
        return None, {"ok": False, "success": False, "code": "NO_CONNECTION",
                      "detail": "No data source is connected yet.", "recoverable": True}
    try:
        from pathlib import Path as _P
        import json as _j
        mapping = _j.loads(_P(str(schema_path)).read_text(encoding="utf-8"))
    except Exception:
        return None, {"ok": False, "success": False, "code": "NO_CONNECTION",
                      "detail": "The schema description could not be read. Reconnect first.",
                      "recoverable": True}
    tables = list(mapping.get("tables", {}).keys())
    if not tables:
        return None, {"ok": False, "success": False, "code": "NO_CONNECTION",
                      "detail": "The connected source has no tables to analyze.",
                      "recoverable": True}
    engine, dialect = _engine_for_session(session)
    if engine is None:
        return None, {"ok": False, "success": False, "code": "NO_CONNECTION",
                      "detail": dialect, "recoverable": True}
    return (mapping, tables, engine, dialect, SchemaIntelligence(mapping), session), None


def _doc_setup(user):
    """Text-only document source: reloadable chunk texts from canonical RAG.

    Reads from the ``insight_rag`` manifest + chunks (PGVector-backed) instead
    of the legacy FAISS SQLite database.  Returns ``((chunks, doc_name, session),
    None)`` or ``(None, err)``.

    Per astra.txt Phase 2: this is *document context*, not analytical proof.
    """
    import json as _j
    from insight_rag.storage import RAGMetadataStore
    session = get_session(user["user_id"])
    if session.get("source_type") != "document" or not session.get("document_id"):
        return None, None
    doc_id = session["document_id"]
    meta = RAGMetadataStore(user["user_id"])
    record = meta.get_document(doc_id)
    if not record:
        return None, {"ok": False, "success": False, "code": "NO_CONNECTION",
                      "detail": "This document has no readable text to base recommendations on.",
                      "recoverable": True}
    doc_name = record.get("filename") or doc_id
    try:
        raw_chunks = meta.load_chunks(doc_id)
    except Exception:
        return None, {"ok": False, "success": False, "code": "NO_CONNECTION",
                      "detail": "This document has no readable text to base recommendations on.",
                      "recoverable": True}
    chunks = [
        {"id": str(c.metadata.get("chunk_id", "")),
         "content": c.page_content,
         "page_number": c.metadata.get("page"),
         "section": c.metadata.get("section")}
        for c in raw_chunks
        if (c.page_content or "").strip()
    ]
    if not chunks:
        return None, {"ok": False, "success": False, "code": "NO_CONNECTION",
                      "detail": "This document has no readable text to base recommendations on.",
                      "recoverable": True}
    return (chunks, doc_name, session), None


@router.get("/api/prescriptions/context")
def prescription_context(user: dict = Depends(get_current_user)):
    """Dynamically discovered prescription context — never hardcoded columns."""
    from hypothesis_engine import data_context
    from insight_agent import sanitize_json
    setup, err = _setup(user)
    if err:
        doc, doc_err = _doc_setup(user)
        if doc is None:
            return err if doc_err is None else doc_err
        chunks, doc_name, session = doc
        return sanitize_json({
            "ok": True, "success": True,
            "connection": {"status": "connected", "source_type": "document",
                           "db_name": doc_name},
            "document": {"name": doc_name, "chunks": len(chunks), "tables": 0},
            "metrics": [], "drivers": [], "dimensions": [],
            "date_columns": [], "tables": [], "time_range": {},
            "opportunities": [],
            "limitations": ["Text-only document — qualitative recommendations with "
                            "citations only; no numeric modeling is possible."],
            "latest": {"saved_insights": len(session.get("saved_insights", [])),
                       "saved_prescriptions": len(session.get("saved_prescriptions", []))},
        })
    mapping, tables, engine, dialect, intel, session = setup
    try:
        ctx = data_context(engine, dialect, intel, tables)
        # Derived stay-length driver: offered only when the table really has
        # 2+ date columns — still 100% data-based, never assumed.
        by_table: dict[str, list] = {}
        for d in ctx.get("date_columns", []):
            by_table.setdefault(d["table"], []).append(d["column"])
        for t, cols in by_table.items():
            if len(cols) >= 2:
                ctx["drivers"].append({"table": t, "column": "length_of_stay",
                                       "kind": "derived",
                                       "note": f"days between {cols[0]} and {cols[1]}"})
        enrich = _enrich_context(engine, dialect, intel, tables, session)
        return sanitize_json({"ok": True, "success": True,
                              "connection": {"status": "connected",
                                             "source_type": session.get("source_type"),
                                             "db_name": session.get("db_name")},
                              **ctx, **enrich})
    finally:
        try:
            engine.dispose()
        except Exception:
            pass


def _enrich_context(engine, dialect, intel, tables, session):
    """Opportunities, per-table limitations, and latest saved evidence (§1).

    Opportunity candidates are schema-discovered only (no data scanned):
    metric × controllable-ish drivers within the same table. Foreign-key
    relationships never auto-generate drivers (hyp §4) and derived fields
    never auto-generate drivers (hyp §5) — actionability for those is
    established separately inside the prescription engine behind statistical
    evidence gates. Actionable reasons name row counts and missing pieces.
    """
    from prescription_engine.engine import _usable_measures, classify_driver
    from prescription_engine.engine import (classify_semantic_role as _role,
                                            NON_LEVER_ROLES as _NON_LEVER,
                                            NON_OUTCOME_ROLES as _NON_OUTCOME)
    from sqlalchemy import text as _t
    opportunities, limitations = [], []

    def _count(table):
        from core.schema_intel import _safe_name
        try:
            with engine.connect() as conn:
                return int(conn.execute(
                    _t(f"SELECT COUNT(*) FROM {_safe_name(table, dialect)}")).scalar() or 0)
        except Exception:
            return -1

    # Rank-first scan: tiny lookup tables must not eat the 3-table modeling
    # budget (e.g. northwind's categories/demographics hid orders entirely).
    # Up to 12 tables are cheaply COUNTed; the 3 most promising (most rows,
    # most measures) get the full opportunity treatment.
    ranked = []
    for t in tables[:12]:
        measures = _usable_measures(intel, t)
        if not measures:
            limitations.append(f"{t}: no measurable numeric columns to improve.")
            continue
        n = _count(t)
        if n >= 0 and n < 10:
            limitations.append(f"{t}: only {n} rows — below the 10-observation minimum for modeling.")
            continue
        ranked.append((n, t, measures))
    # Uncountable tables (-1) sort last; then most rows, then most measures.
    ranked.sort(key=lambda x: (x[0] < 0, -(x[0] if x[0] >= 0 else 0), -len(x[2])))
    for _n, t, measures in ranked[:3]:
        for m in measures[:2]:
            cands = []
            for c in measures:
                if c != m:
                    cands.append((c, None, "same table"))
            # hyp §4: FK relationships are join metadata, never auto-drivers —
            # no cross-table candidates are seeded here. An explicit driver
            # request over a declared join is still supported inside the
            # prescription engine, behind full statistical evidence gates.
            # hyp §5: derived fields (e.g. stay length from dates) are never
            # auto-drivers either — the engine measures them with evidence
            # gates when they are actually analyzed.
            if not cands:
                limitations.append(f"{t}.{m}: no other numbers reachable to test as drivers.")
                continue
            for c, ot, how in cands[:3]:
                # hyp.txt §§4/10-11: technical/system measurements stay
                # analytic — they never become auto-opportunities, however
                # "measurable" they look. Generalized role check, never
                # per-column literals.
                try:
                    _m_role = _role(m, "numeric")
                except Exception:
                    _m_role = "MEASURE"
                try:
                    _c_role = _role(c, "numeric")
                except Exception:
                    _c_role = "MEASURE"
                if _m_role in _NON_OUTCOME or _c_role in _NON_LEVER:
                    _bad = c if _c_role in _NON_LEVER else m
                    _kind = str(_c_role if _c_role in _NON_LEVER else _m_role).lower().replace("_", " ")
                    limitations.append(f"{t}.{_bad}: a {_kind} — analytic signal only, not an actionable opportunity.")
                    continue
                level, _ = classify_driver(c, "numeric")
                opportunities.append({"table": t, "metric": m, "driver": c,
                                      "driver_table": ot or t, "control": level,
                                      "note": how})
    saved_rx = session.get("saved_prescriptions", [])
    latest = {"saved_insights": len(session.get("saved_insights", [])),
              "saved_prescriptions": len(saved_rx)}
    if saved_rx:
        latest["last_prescription"] = {k: saved_rx[0].get(k)
                                       for k in ("id", "title", "created_at")}
    return {"opportunities": opportunities[:9], "table_limitations": limitations[:6],
            "latest": latest}


@router.post("/api/prescriptions/analyze")
def prescription_analyze(req: AnalyzeRequest, request: Request,
                         user: dict = Depends(require_writable)):
    """Ranked, evidence-backed prescriptions — or an honest refusal."""
    from prescription_engine import analyze as _analyze, analyze_document as _analyze_doc
    from insight_agent import sanitize_json
    setup, err = _setup(user)
    if err:
        doc, doc_err = _doc_setup(user)
        if doc is None:
            return err if doc_err is None else doc_err
        chunks, doc_name, _ = doc
        out = _analyze_doc(chunks, (req.question or ""), doc_name)
        try:
            _attach_prescription_verdict(out)
        except Exception:
            pass
        return sanitize_json(out)
    mapping, tables, engine, dialect, intel, session = setup
    try:
        metric = req.metric or (req.context.metric if req.context else None)
        driver = req.driver or (req.context.driver if req.context else None)
        priors = {}
        if req.context:
            priors = {"origin": req.context.origin, "insight_id": req.context.insight_id,
                      "effect": req.context.effect, "confidence": req.context.confidence}
        if not metric:
            return {"ok": False, "success": False, "code": "INVALID_METRIC",
                    "detail": "Choose a target metric first — ARIA won't guess what to improve.",
                    "recoverable": True}

        def _canon_requested(value, candidates):
            """Resolve a UI-supplied name to its canonical schema spelling.

            Exact match first, then case-insensitive, then humanized-form
            match ("Gross Revenue" -> "gross_revenue"). Generic — no names.
            Returns None when nothing matches.
            """
            s = str(value or "").strip()
            if not s:
                return None
            if s in candidates:
                return s
            low = s.lower()
            for c in candidates:
                if str(c).lower() == low:
                    return c
            norm = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in low).strip("_")
            for c in candidates:
                if str(c).lower() == norm:
                    return c
            return None

        try:
            _all_numeric: list[str] = []
            for _t in tables:
                try:
                    _all_numeric.extend(intel.get_numeric_cols(_t) or [])
                except Exception:
                    pass
            _canon_metric = _canon_requested(metric, _all_numeric)
            if _canon_metric:
                metric = _canon_metric
        except Exception:
            pass
        # Locate the metric's table (schema-discovered, never assumed).
        table = req.table
        if table not in tables:
            table = None
            for t in tables:
                if metric in [c for c in intel.get_numeric_cols(t)]:
                    table = t
                    break
        if not table:
            return {"ok": False, "success": False, "code": "INVALID_METRIC",
                    "detail": f"ARIA couldn't find a measurable '{metric}' in this data.",
                    "recoverable": True}
        if driver:
            try:
                _drv_cands: list[str] = []
                try:
                    _drv_cands.extend(intel.get_col_names(table) or [])
                except Exception:
                    pass
                for _t in tables:
                    try:
                        _drv_cands.extend(intel.get_col_names(_t) or [])
                    except Exception:
                        pass
                _canon_drv = _canon_requested(driver, _drv_cands)
                if _canon_drv:
                    driver = _canon_drv
            except Exception:
                pass
        seg = None
        if req.segment and req.segment.field:
            seg = {"field": req.segment.field, "value": req.segment.value}
            # Integrity gate: validate segment.value against live DISTINCT
            # values (LIMIT 100) with suggestions — never a silent wrong cut.
            try:
                from prescription_engine.engine import validate_segment_value as _vseg
                _dims = set()
                try:
                    _dims = set(intel.find_all_dimensions(table) or [])
                except Exception:
                    _dims = set()
                if seg["field"] not in _dims:
                    return {"ok": False, "success": False, "code": "INVALID_SEGMENT",
                            "detail": f"'{seg['field']}' is not a usable segment here.",
                            "recoverable": True}
                if seg.get("value") is not None:
                    _chk = _vseg(engine, dialect, table, seg["field"], seg["value"], limit=100)
                    if isinstance(_chk, dict) and not _chk.get("ok", True):
                        return {"ok": False, "success": False, "code": "INVALID_SEGMENT",
                                "detail": (f"'{seg['value']}' was not found in '{seg['field']}'. "
                                           f"Try one of: {', '.join(_chk.get('suggestions', [])[:5]) or 'no values'}."),
                                "suggestions": _chk.get("suggestions", [])[:5],
                                "recoverable": True}
            except Exception:
                log.debug("segment validation failed closed-open", exc_info=True)
        out = _analyze(engine, dialect, intel, table, metric, driver=driver,
                       dimensions=req.dimensions, segment=seg,
                       question=req.question, priors=priors)
        # NOTE: refusal alternatives are computed inside the engine
        # (compute_refusal_alternatives with quality-gate probes) and
        # attached by _finalize_output — never a static list here.
        if out.get("status") == "completed":
            polished = _polish_story(session, out)
            if polished:
                out["story"] = polished
        out, _ = _strip_prescription_response(out)
        try:
            _attach_prescription_verdict(out)
        except Exception:
            pass
        return sanitize_json(out)
    except Exception:
        import logging as _logging
        _logging.getLogger("aria.prescription").exception("Analyze failed")
        return {"ok": False, "success": False, "code": "ANALYSIS_FAILED",
                "detail": "Prescription analysis failed unexpectedly. Your data was not changed.",
                "recoverable": True}
    finally:
        try:
            engine.dispose()
        except Exception:
            pass


_POLISH_CAUSALITY_VERBS = ("cause", "causes", "caused", "causing", "prove",
                             "proves", "proved", "proof", "guarantee", "guarantees",
                             "will increase", "will decrease", "will improve",
                             "definitely", "certainly")


def _polish_numbers(text):
    """Numbers appearing in a story string (ints/floats/%-values)."""
    import re as _re
    try:
        return _re.findall(r"-?\d+(?:\.\d+)?%?", str(text or ""))
    except Exception:
        return []


def _polish_story(session, out):
    """LLM rewording of the story only; None on any failure (§24).

    Skipped for the local provider: a cold/slow Ollama can stall the
    response ~30s for zero analytical gain — the deterministic story,
    built from real numbers, is the product.

    Evidence gate FIRST: prescriptions resting on < 10 observations keep
    their deterministic honest wording — the LLM never compensates.

    Integrity gate: the polished text is REJECTED (fallback to the
    deterministic story) when numbers differ from the original or when
    causality verbs appear — the LLM may only polish wording, never
    compute numbers or claim proof.
    """
    try:
        try:
            _fit = (out.get("fit") or {}) if isinstance(out, dict) else {}
            _n = _fit.get("n")
            if _n is None:
                _ev0 = ((out.get("prescriptions") or [{}])[0].get("evidence")
                        if isinstance(out.get("prescriptions"), list) else {}) or {}
                _n = _ev0.get("n")
            if _n is not None and int(_n) < 10:
                return None
        except (TypeError, ValueError):
            pass
        if (session.get("provider_name") or "local") == "local":
            return None
        prov = session.get("provider")
        if prov is None:
            return None
        original = str(out.get("story") or "")
        prompt = (
            "Rewrite this business finding in one plain sentence for a manager. "
            "Keep every number exactly as given. Do not calculate, invent, or "
            "claim causality. Do not use the words cause/prove/guarantee/will.\n"
            "Finding: " + original
        )
        text = prov.chat("story", messages=[{"role": "user", "content": prompt}],
                         temperature=0.3, num_predict=120, timeout=30)
        text = (text or "").strip()
        if not (20 <= len(text) <= 400):
            return None
        # Reject when numbers differ from the deterministic story.
        if _polish_numbers(text) != _polish_numbers(original):
            log.debug("polish story rejected: numbers differ")
            return None
        # Reject when causality verbs appear.
        _low = text.lower()
        if any(v in _low for v in _POLISH_CAUSALITY_VERBS):
            log.debug("polish story rejected: causality verb")
            return None
        return text
    except Exception:
        pass
    return None


@router.post("/api/prescriptions/scenario")
def prescription_scenario(req: ScenarioRequest, request: Request,
                          user: dict = Depends(require_writable)):
    """Read-only scenario recomputation — the source database is never touched."""
    from prescription_engine import run_scenario as _scenario
    from insight_agent import sanitize_json
    setup, err = _setup(user)
    if err:
        doc, doc_err = _doc_setup(user)
        if doc is None:
            return err if doc_err is None else doc_err
        return {"ok": False, "success": False, "code": "SCENARIO_UNSUPPORTED",
                "detail": "Scenarios need modeled numbers; this source is a text document. "
                          "Use the cited passages above instead.",
                "recoverable": True}
    mapping, tables, engine, dialect, intel, session = setup
    try:
        if req.table not in tables:
            return {"ok": False, "success": False, "code": "SCENARIO_UNSUPPORTED",
                    "detail": "Unknown table for this scenario.", "recoverable": True}
        filt = None
        if req.filters:
            f0 = req.filters[0]
            if (f0.operator or "equals") != "equals":
                return {"ok": False, "success": False, "code": "SCENARIO_UNSUPPORTED",
                        "detail": "Only 'equals' segment filters are supported.",
                        "recoverable": True}
            filt = (f0.field, f0.value)
        join = None
        if req.join:
            # Validate the join against DECLARED edges only (§34) — the
            # frontend may never inject arbitrary join SQL.
            from prescription_engine.engine import _edge_neighbors
            allowed = _edge_neighbors(intel, req.table)
            if (req.join.table, req.join.self_col, req.join.other_col) not in allowed:
                return {"ok": False, "success": False, "code": "SCENARIO_UNSUPPORTED",
                        "detail": "Unknown join for this scenario.", "recoverable": True}
            join = (req.join.table, req.join.self_col, req.join.other_col)
        out = _scenario(engine, dialect, intel, req.table, req.metric, req.driver,
                        [c.model_dump() for c in req.changes], filt=filt, join=join)
        out, _ = _strip_prescription_response(out)
        try:
            _attach_prescription_verdict(out)
        except Exception:
            pass
        return sanitize_json(out)
    except Exception:
        import logging as _logging
        _logging.getLogger("aria.prescription").exception("Scenario failed")
        return {"ok": False, "success": False, "code": "ANALYSIS_FAILED",
                "detail": "Scenario failed unexpectedly. Your data was not changed.",
                "recoverable": True}
    finally:
        try:
            engine.dispose()
        except Exception:
            pass


def _sanitize_saved_text(value, max_len=500):
    """Neutralize saved question/note text: strip tags + controls, cap length.

    Saved items are echoed back in list endpoints and rendered as titles, so
    untrusted markup or directive prose must not survive the round-trip.
    Generic helper — no dataset names.
    """
    import re as _re
    s = "" if value is None else str(value)
    s = _re.sub(r"<[^>]*>", " ", s)
    s = "".join(c if c.isprintable() else " " for c in s)
    s = _re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        s = s[:max(0, max_len - 1)].rstrip() + "…"
    return s


@router.post("/api/prescriptions/save")
def prescription_save(req: SaveRequest, request: Request,
                      user: dict = Depends(require_writable)):
    """Persist the structured prescription — never prose alone (§28)."""
    session = get_session(user["user_id"])
    saved = session.get("saved_prescriptions", [])
    entry = {"id": f"rxsv_{int(time.time() * 1000)}",
             "created_at": datetime.now().isoformat(),
             "question": _sanitize_saved_text(req.question, 500) or None,
             **(req.prescription or {})}
    saved = [entry] + saved[:49]
    session["saved_prescriptions"] = saved
    save_session(user["user_id"], session)
    return {"ok": True, "success": True, "saved": saved}


@router.get("/api/prescriptions/saved")
def prescription_saved(user: dict = Depends(require_writable)):
    session = get_session(user["user_id"])
    return {"ok": True, "success": True,
            "saved": session.get("saved_prescriptions", [])}


@router.delete("/api/prescriptions/saved/{sid}")
def prescription_unsave(sid: str, user: dict = Depends(require_writable)):
    session = get_session(user["user_id"])
    saved = [s for s in session.get("saved_prescriptions", []) if s.get("id") != sid]
    session["saved_prescriptions"] = saved
    save_session(user["user_id"], session)
    return {"ok": True, "success": True, "saved": saved}


class SavedUpdateRequest(BaseModel):
    owner: str | None = None
    priority: str | None = None
    due: str | None = None
    review_outcome: str | None = None
    review_note: str | None = None
    review_at: str | None = None


_RX_SAVED_EDITABLE = ("owner", "priority", "due", "review_outcome",
                      "review_note", "review_at")


@router.patch("/api/prescriptions/saved/{sid}")
def prescription_saved_update(sid: str, req: SavedUpdateRequest,
                              user: dict = Depends(require_writable)):
    """Update business fields of a saved prescription (owner, priority,
    due date, review outcome). Additive — stored entry otherwise untouched."""
    session = get_session(user["user_id"])
    saved = session.get("saved_prescriptions", [])
    found = False
    patch = {k: v for k, v in req.model_dump().items()
             if k in _RX_SAVED_EDITABLE and v is not None}
    for _tk in ("owner", "review_note"):
        if isinstance(patch.get(_tk), str):
            patch[_tk] = _sanitize_saved_text(patch[_tk], 500) or None
    if patch.get("priority") and str(patch["priority"]).lower() not in (
            "low", "medium", "high", "critical"):
        return {"ok": False, "success": False, "code": "INVALID_REQUEST",
                "detail": "Priority must be low, medium, high or critical."}
    if patch.get("review_outcome") and str(patch["review_outcome"]).lower() not in (
            "achieved", "missed", "inconclusive"):
        return {"ok": False, "success": False, "code": "INVALID_REQUEST",
                "detail": "Review outcome must be achieved, missed or inconclusive."}
    for s in saved:
        if isinstance(s, dict) and s.get("id") == sid:
            s.update(patch)
            s["updated_at"] = datetime.now().isoformat()
            found = True
            break
    if not found:
        return {"ok": False, "success": False, "code": "NOT_FOUND",
                "detail": "Saved prescription not found."}
    session["saved_prescriptions"] = saved
    save_session(user["user_id"], session)
    return {"ok": True, "success": True, "saved": saved}


class InactionRequest(BaseModel):
    metric: str | None = None
    table: str | None = None


@router.post("/api/prescriptions/inaction")
def prescription_inaction(req: InactionRequest, request: Request,
                          user: dict = Depends(require_writable)):
    """Cost of doing nothing: monthly run-rate projection for one metric."""
    from prescription_engine.engine import inaction_cost, _usable_measures
    setup, err = _setup(user)
    if err:
        return err
    mapping, tables, engine, dialect, intel, session = setup
    try:
        metric, table = req.metric, req.table
        if table not in tables:
            table = next((t for t in tables if metric in intel.get_numeric_cols(t)), None) if metric else None
            if table is None:
                table = next((t for t in tables if _usable_measures(intel, t)), tables[0] if tables else None)
        if not metric and table:
            ms = _usable_measures(intel, table)
            metric = ms[0] if ms else None
        if not metric or not table:
            return {"ok": False, "success": False, "code": "INVALID_METRIC",
                    "detail": "Pick a metric with dated history first."}
        out = inaction_cost(engine, dialect, intel, table, metric)
        return {"ok": True, "success": True, **out}
    except Exception:
        import logging as _logging
        _logging.getLogger("aria.prescription").exception("Inaction failed")
        return {"ok": False, "success": False, "code": "ANALYSIS_FAILED",
                "detail": "Could not price inaction from this data."}
    finally:
        try:
            engine.dispose()
        except Exception:
            pass


@router.get("/api/prescriptions/levers")
def prescription_levers(request: Request, metric: str | None = None,
                        table: str | None = None,
                        user: dict = Depends(require_writable)):
    """Lever badges: for each numeric column, testable / needs-data /
    not-a-lever with the reason. Cheap bounded probes only (counts and
    distincts) — never a full correlation run. Fully data-driven."""
    from prescription_engine.engine import (
        _usable_measures, classify_driver, classify_semantic_role,
        NON_LEVER_ROLES, MIN_N)
    from sqlalchemy import text as _t
    setup, err = _setup(user)
    if err:
        return err
    mapping, tables, engine, dialect, intel, session = setup
    try:
        if table not in tables:
            table = next((t for t in tables if metric in intel.get_numeric_cols(t)), None) if metric else None
            if table is None:
                table = next((t for t in tables if _usable_measures(intel, t)), tables[0] if tables else None)
        if not table:
            return {"ok": False, "success": False, "code": "INVALID_METRIC",
                    "detail": "Pick a metric with numeric data first."}
        if not metric:
            ms = _usable_measures(intel, table)
            metric = ms[0] if ms else None
        levers = []
        try:
            from core.schema_intel import _safe_name as _qn
            _cols = intel.get_numeric_cols(table)
            _counts = {}
            with engine.connect() as _conn:
                for _c in _cols[:12]:
                    try:
                        _r = _conn.execute(_t(
                            f"SELECT COUNT({_qn(_c)}) AS n, COUNT(DISTINCT {_qn(_c)}) AS d "
                            f"FROM {_qn(table)}")).fetchone()
                        _counts[_c] = (int(_r[0] or 0), int(_r[1] or 0))
                    except Exception:
                        continue
        except Exception:
            _counts = {}
        for _c in intel.get_numeric_cols(table)[:12]:
            if _c == metric:
                continue
            try:
                _level, _ = classify_driver(_c, "numeric")
            except Exception:
                _level = "unknown"
            _role = classify_semantic_role(_c, "numeric")
            _n, _d = _counts.get(_c, (0, 0))
            if _role in NON_LEVER_ROLES or _level in ("not_controllable", "segmentation"):
                levers.append({"column": _c, "status": "not-a-lever",
                               "reason": "Text, category or system field — it cannot be turned up or down."})
            elif _n < MIN_N:
                levers.append({"column": _c, "status": "needs-data",
                               "reason": f"Only {_n} usable rows — at least {MIN_N} are needed to test it."})
            elif _d <= 1:
                levers.append({"column": _c, "status": "needs-data",
                               "reason": "No variation in the observed values — nothing to model."})
            else:
                levers.append({"column": _c, "status": "testable",
                               "reason": f"{_n} rows, {_d} distinct values — ready to test as a lever."})
        # Text/dimension columns are never levers — list them so the picker
        # explains WHY instead of showing nothing.
        try:
            _dims = list(intel.find_all_dimensions(table) or [])[:8]
        except Exception:
            _dims = []
        for _c in _dims:
            if _c == metric or any(l["column"] == _c for l in levers):
                continue
            levers.append({"column": _c, "status": "not-a-lever",
                           "reason": "Category — useful for breaking numbers down, but it cannot be adjusted as an intervention."})
        return {"ok": True, "success": True, "metric": metric, "table": table, "levers": levers}
    finally:
        try:
            engine.dispose()
        except Exception:
            pass


@router.get("/api/prescriptions/presets")
def prescription_presets(request: Request, user: dict = Depends(require_writable)):
    """One-click starter questions built from the live schema: for each
    usable measure, 'What should I do about declining {metric}?' Never
    static — derived from the connected mapping."""
    from prescription_engine.engine import _usable_measures
    setup, err = _setup(user)
    if err:
        return err
    mapping, tables, engine, dialect, intel, session = setup
    try:
        from core.semantic import humanize as _hz
        presets = []
        for t in tables[:6]:
            for m in _usable_measures(intel, t)[:3]:
                presets.append({
                    "question": f"What should I do about {_hz(m).lower()}?",
                    "metric": m, "table": t,
                })
                if len(presets) >= 6:
                    break
            if len(presets) >= 6:
                break
        return {"ok": True, "success": True, "presets": presets}
    finally:
        try:
            engine.dispose()
        except Exception:
            pass


@router.get("/api/prescriptions/alerts")
def prescription_alerts(request: Request, user: dict = Depends(require_writable)):
    """Alerts that matter: unusual latest-month moves across top measures."""
    from prescription_engine.engine import detect_alerts
    setup, err = _setup(user)
    if err:
        return err
    mapping, tables, engine, dialect, intel, session = setup
    try:
        alerts = []
        for t in tables[:4]:
            try:
                alerts.extend(detect_alerts(engine, dialect, intel, t))
            except Exception:
                continue
        alerts.sort(key=lambda a: abs(a.get("change_pct") or 0), reverse=True)
        return {"ok": True, "success": True, "alerts": alerts[:5]}
    except Exception:
        import logging as _logging
        _logging.getLogger("aria.prescription").exception("Alerts failed")
        return {"ok": True, "success": True, "alerts": []}
    finally:
        try:
            engine.dispose()
        except Exception:
            pass
