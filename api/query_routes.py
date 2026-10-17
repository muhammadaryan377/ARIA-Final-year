"""Goal Agent + Insight Agent routes: suggestions, ask, and insight analysis."""

import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

import auth
from core.config import get_session, save_session, resolve_provider_key
from core.deps import require_data, require_writable
from core.validation import validate_goal_text
from goal_agent import GoalAgent
from goal_agent.schema import load_user_overrides
from goal_agent.proactive_intelligence import analyze_result_intelligence
from core.analytical_result import analyze_result
from insight_agent import InsightAgent, sanitize_json

log = logging.getLogger(__name__)

router = APIRouter()


class GoalRequest(BaseModel):
    goal: str
    previous_goal: str | None = None
    previous_result: dict | None = None
    conversation_id: str | None = None
    # Context-aware analytics context (all optional; resolved dynamically).
    metric: str | None = None
    time_range: dict | str | None = None
    filters: dict | list | None = None
    grouping: str | list | None = None
    dataset: str | None = None
    previous_context: dict | None = None


def _schema_terms_for(schema_path) -> set:
    """Token vocabulary of real table/column names for schema grounding."""
    try:
        import re as _re

        if not schema_path or not Path(str(schema_path)).exists():
            return set()
        data = json.loads(Path(str(schema_path)).read_text(encoding="utf-8"))
        terms: set = set()
        for tname, tinfo in (data.get("tables", {}) or {}).items():
            terms.add(str(tname).lower())
            for part in _re.split(r"[^a-z0-9]+", str(tname).lower()):
                if part:
                    terms.add(part)
            cols = (tinfo or {}).get("columns", {})
            names = cols.keys() if isinstance(cols, dict) else [
                (c.get("column") if isinstance(c, dict) else str(c)) for c in (cols or [])]
            for cname in names or []:
                terms.add(str(cname).lower())
                for part in _re.split(r"[^a-z0-9]+", str(cname).lower()):
                    if part:
                        terms.add(part)
        return terms
    except Exception:
        return set()


def _build_why(*, goal, tables, columns, row_count, sql, explanation,
               confidence, warnings) -> dict:
    """Structured Why block (§21): stored at analysis time, shown on demand."""
    from core.analytical_result import natural_label as _nl
    limitations = []
    if not row_count:
        limitations.append("The query returned no rows, so no conclusion can be drawn.")
    if warnings:
        limitations.append(f"{len(warnings)} data-quality warning(s) apply to this result.")
    limitations.append("Aggregates describe association in this dataset, not external causation.")
    return {
        "what_happened": (explanation or "Analysis completed over the connected data.").strip()[:800],
        "why_it_matters": (
            f"Answers '{goal}' from {row_count} row(s) across "
            f"{', '.join(_nl(t) for t in tables[:3]) or 'the connected source'}."
        ),
        "evidence": {
            "tables": tables,
            "columns": columns[:10],
            "row_count": row_count,
            "sql": sql,
        },
        "contributing_factors": [_nl(c) for c in (columns[1:4] if len(columns) > 1 else [])],
        "comparisons": [],
        "confidence": confidence,
        "limitations": limitations,
    }


def _build_story(*, goal, tables, columns, result_data, why_block) -> str:
    """Coherent data story (§23): finding → evidence → contributors → next step."""
    from core.analytical_result import natural_label as _nl
    n = len(result_data or [])
    first = (result_data or [{}])[0] if result_data else {}
    if isinstance(first, dict) and first:
        lead_col = next(iter(first.keys()))
        lead_val = first[lead_col]
        finding = f"Top result: {_nl(lead_col)} is {lead_val} across {n} row(s)."
    elif n:
        finding = f"Query returned {n} row(s) from {', '.join(_nl(t) for t in tables[:3]) or 'the source'}."
    else:
        finding = "No matching rows were found for this question."
    evidence = f"Evidence: {why_block['evidence']['row_count']} row(s), covering {', '.join(_nl(c) for c in columns[:5]) or '—'}."
    factors = ""
    if why_block["contributing_factors"]:
        factors = f" Segmented by {', '.join(why_block['contributing_factors'])}."
    return (
        f"{finding} {evidence}{factors} "
        f"Confidence: {why_block['confidence']}. "
        f"Limitation: {why_block['limitations'][0]}"
    )


def _chart_spec_for_result(columns, result_data) -> dict:
    """Chart spec generated from the actual result shape (§19) via the registry."""
    try:
        from insight_agent.chart_catalog import compatible_chart_options
    except Exception:
        return {"chart_type": "table", "reason": "chart catalog unavailable"}
    rows = result_data or []
    first = rows[0] if rows and isinstance(rows[0], dict) else {}
    num, cat, dt = 0, 0, 0
    for c in columns or []:
        v = first.get(c)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            num += 1
        elif isinstance(v, str) and any(k in c.lower() for k in
                                        ("date", "time", "month", "year", "day", "period")):
            dt += 1
        elif v is not None:
            cat += 1
    card = len(set(str((r or {}).get(columns[0])) for r in rows[:200] if isinstance(r, dict))) if columns else 0
    options = compatible_chart_options(num, cat, dt, card)
    if not options:
        return {"chart_type": "table", "reason": "no compatible chart for this shape"}
    top = options[0]
    ctype = top["chart_type"]
    # Validation gate 3: x/y must point at REAL columns in the data, and the
    # chart family must fit the shape — else the frontend renders a wrong
    # chart. Deterministic downgrade, never a faulty viz.
    cols = list(columns or [])
    x_col, y_col = None, None
    try:
        num_cols = [c for c in cols if isinstance(first.get(c), (int, float))
                    and not isinstance(first.get(c), bool)]
        cat_cols = [c for c in cols if c not in num_cols and first.get(c) is not None]
        dt_cols = [c for c in cols if any(k in c.lower() for k in
                   ("date", "time", "month", "year", "day", "period"))]
        if ctype in ("line", "area") and dt_cols and num_cols:
            x_col, y_col = dt_cols[0], num_cols[0]
        elif ctype in ("bar", "column", "pie", "doughnut") and cat_cols and num_cols:
            x_col, y_col = cat_cols[0], num_cols[0]
            if ctype in ("pie", "doughnut"):
                card_n = len(set(str((r or {}).get(x_col)) for r in rows[:200]))
                if card_n < 2 or card_n > 8:
                    ctype = "bar"  # pie with 1 or 9+ slices misleads
        elif num_cols:
            x_col, y_col = (cat_cols + dt_cols + num_cols[:1])[0], num_cols[0]
        else:
            ctype = "table"
    except Exception:
        ctype = "table"
        x_col, y_col = None, None
    if not rows or not cols:
        ctype = "table"
    return {"chart_type": ctype, "x": x_col, "y": y_col, "label": top["label"],
            "reason": top["analyst_use"], "compatible": [o["chart_type"] for o in options[1:4]]}


def _summarize_result(goal, columns, result_data) -> str:
    """Deterministic answer sentence from real rows (fallback when the agent
    returns no message, so the Answer section is never empty on success)."""
    from core.analytical_result import natural_label as _nl
    rows = result_data or []
    if not rows:
        return "No matching rows were found for this question."
    first = rows[0] if isinstance(rows[0], dict) else {}
    if first:
        keys = list(first.keys())
        # Name the LEADERS, not just the count: for multi-row results, name
        # the top 3 entries (e.g. artists, regions) with their values so the
        # answer reads naturally and matches the chart. Generic — works for
        # any columns on any unseen schema. Raw table/column identifiers are
        # humanized; table names never surface.
        if len(rows) > 1 and len(keys) >= 2:
            name_col = next((k for k in keys
                             if not isinstance(first.get(k), (int, float))
                             or isinstance(first.get(k), bool)), keys[0])
            val_col = next((k for k in keys
                            if k != name_col and isinstance(first.get(k), (int, float))
                            and not isinstance(first.get(k), bool)), None)
            if val_col:
                leaders = ", ".join(
                    f"{r.get(name_col)} ({r.get(val_col)})" for r in rows[:3])
                return (f"Top {len(rows)} {_nl(name_col)}s by {_nl(val_col)}: "
                        f"{leaders}.")
        lead_col = next(iter(first.keys()))
        lead_val = first[lead_col]
        extra = ""
        if len(rows) > 1:
            extra = f" ({len(rows)} rows, led by {lead_val})"
        return f"Top result: {_nl(lead_col)} is {lead_val}{extra}."
    return f"Query returned {len(rows)} row(s)."


_DOC_INTENT_WORDS = ("target", "goal", "plan", "report", "forecast says",
                      "document", "mentioned", "according to", "versus", "vs ",
                      "compare with", "management", "says", "stated")


def _document_evidence(user_id, goal_text, limit: int = 3) -> list[dict]:
    """Cross-source document context (§6): search the user's documents when
    the question hints at external information.

    Returns [] when no documents or no doc-intent signal (cheap: skipped
    entirely).  Uses the canonical ``insight_rag`` pipeline — NOT the legacy
    FAISS ``document_processor`` — so document context and analytical data
    evidence are always served from the same canonical RAG engine.

    Per astra.txt Phase 2: RAG provides *context*, not analytical proof.
    The Decision Center distinguishes ``data_evidence`` (from analytics) from
    ``document_context`` (from this function).
    """
    try:
        from insight_rag.storage import RAGMetadataStore, get_user_vector_store

        if not any(w in goal_text.lower() for w in _DOC_INTENT_WORDS):
            return []

        meta = RAGMetadataStore(str(user_id))
        documents = meta.list_documents()
        if not documents:
            return []

        store = get_user_vector_store(str(user_id))
        results = store.search(goal_text, k=limit)

        out: list[dict] = []
        for doc, score in results:
            m = doc.metadata if hasattr(doc, "metadata") else {}
            out.append({
                "document_id": m.get("document_id", ""),
                "content": (doc.page_content[:500] if hasattr(doc, "page_content") else ""),
                "score": round(float(score), 6),
                "page_number": m.get("page", m.get("page_number")),
                "filename": m.get("filename"),
                "content_type": m.get("content_type", "text"),
            })
        return out
    except Exception:
        return []


def _attach_ask_verdict(resp, session):
    """Best-effort Verification Agent hookup for ask answers (never blocks).

    Attaches ``resp["verdict"]``. On any failure attaches an honest
    not-verified verdict and still returns the data untouched.
    """
    try:
        from verification.agent import verify_ask
        verdict = verify_ask(resp if isinstance(resp, dict) else {},
                             dict(session or {}))
    except Exception:
        verdict = {"verified": False, "checks": [],
                   "warnings": ["verification unavailable"]}
    try:
        resp["verdict"] = verdict
    except Exception:
        pass
    return resp


def _check_consistency(chart_spec, result_data, why_block) -> dict:
    """§24: chart + result + explanation must describe the same outcome."""
    issues = []
    rows = result_data or []
    ctype = (chart_spec or {}).get("chart_type", "table")
    if not rows and ctype != "table":
        issues.append("empty result paired with a non-table chart; downgraded to table")
        chart_spec = dict(chart_spec or {}, chart_type="table")
    if why_block.get("evidence", {}).get("row_count", 0) != len(rows):
        issues.append("row-count drift between evidence and returned data")
    return {"consistent": not issues, "issues": issues, "chart_spec": chart_spec}


def _suggestion_column_holds_data(session: dict, table: str, column: str,
                                    sample: int = 20, numeric: bool = False) -> bool:
    """Best-effort live check that a candidate column holds usable values.

    Mirrors api.copilot_routes._column_holds_data (kept local to avoid a
    router import cycle). Never raises: on any failure return True so a
    transient DB hiccup never blanks the suggestion list.
    """
    try:
        from api.copilot_routes import _column_holds_data as _shared
        return bool(_shared(session or {}, table, column, sample=sample, numeric=numeric))
    except Exception:
        return True


def _suggestion_measure_ok(session: dict, table: str, column: str) -> bool:
    return _suggestion_column_holds_data(session, table, column, numeric=True)


def _suggestion_datetime_ok(session: dict, table: str, column: str) -> bool:
    return _suggestion_column_holds_data(session, table, column, numeric=False)


# Plain-ask blocklist: outlier / correlation / forecast / what-if need
# drivers or history and live on their own tabs — never as plain suggestions.
_BLOCKED_SUGGESTION_RES = None


def _is_blocked_suggestion(text: str) -> bool:
    import re as _re
    global _BLOCKED_SUGGESTION_RES
    if _BLOCKED_SUGGESTION_RES is None:
        _BLOCKED_SUGGESTION_RES = [
            _re.compile(p, _re.IGNORECASE) for p in (
                r"\boutlier\w*\b", r"\banomal\w*\b", r"\bunusual\b",
                r"\bcorrelat\w*\b", r"\brelationship between\b",
                r"\bforecast\w*\b", r"\bpredict\w*\b",
                r"\bwhat if\b", r"\bwhat-if\b",
                r"\bscenario\b", r"\bdata quality\b",
            )
        ]
    s = str(text or "")
    return any(p.search(s) for p in _BLOCKED_SUGGESTION_RES)


@router.post("/api/suggestions")
def get_suggestions(request: Request, payload: dict | None = None, user: dict = Depends(require_writable)):
    """LLM-driven + template-driven suggested search goals. Works with any
    connected database because it uses dynamic schema intelligence rather
    than hardcoded table/column references."""
    session = get_session(user["user_id"])
    provider = require_data(request)
    try:
        limit = int((payload or {}).get("limit", 8))
    except Exception:
        limit = 8
    # Cap to the 4-8 window.
    limit = max(4, min(8, limit))
    try:
        # Read the schema mapping directly (no slow agent initialization) so
        # suggestions work instantly regardless of database size.
        schema_path = Path(str(session.get("schema_path", "")))
        schema_data = {}
        if schema_path.exists():
            with open(schema_path, "r", encoding="utf-8") as f:
                schema_data = json.load(f)
        tables = list(schema_data.get("tables", {}).keys()) if schema_data else []
        if not tables:
            # No schema context: only plain-ask answerable generics. Trend /
            # outlier / correlation / forecast / what-if need verified
            # columns, drivers or history — never here.
            suggestions = [
                "How many total records are in the data?",
                "Which categories or segments appear most frequently?",
                "What are the top items by value or quantity?",
                "What is the average of the primary measure?",
                "How is the main metric distributed across categories?",
                "What is the total of the main numeric measure?",
            ][:limit]
        else:
            # Build suggestions from actual table/column names — fully dynamic.
            from core.schema_intel import SchemaIntelligence, _is_id_like
            from core.analytical_result import natural_label as _nl, _plural as _pl
            intel = SchemaIntelligence(schema_data)
            suggestions = []

            # Find best measure and dimension per table (natural English:
            # labels are humanized and table names never surface in prose).
            # A trend is only suggested when a real datetime column exists
            # AND it holds data; a SUM is only suggested when the measure is
            # verified numeric-with-data, else COUNT/records. ID-like
            # dimensions are never suggested.
            for tname in list(intel.tables)[:5]:
                measure = intel.find_best_measure(tname)
                if measure and not _suggestion_measure_ok(session, tname, measure):
                    measure = None
                dim = intel.find_best_dimension(tname)
                if dim and _is_id_like(str(dim)):
                    dim = None
                dt = intel.find_best_datetime(tname)
                if dt and not _suggestion_datetime_ok(session, tname, dt):
                    dt = None
                if measure:
                    _ml = _nl(measure)
                    if _ml.strip().lower() in ("total", "sum", "count", "value", "amount"):
                        suggestions.append("What is the overall total value?")
                    else:
                        suggestions.append(f"What is the total {_ml}?")
                else:
                    suggestions.append("How many records are in the data?")
                if dim:
                    suggestions.append(f"Which {_nl(dim)} has the most records?")
                if measure and dt:
                    suggestions.append(f"How has {_nl(measure)} changed over time?")
                if measure and dim:
                    suggestions.append(f"What are the top {_pl(_nl(dim))} by {_nl(measure)}?")

            # Pad with answerable patterns only if we need more (never
            # outlier / correlation / forecast / what-if / data-quality).
            generic = [
                "How many records are in the data?",
                "Which categories appear most frequently?",
                "What are the top items by value?",
            ]
            for g in generic:
                if len(suggestions) < limit and g not in suggestions:
                    suggestions.append(g)

            suggestions = suggestions[:limit]
    except Exception:
        suggestions = [
            "How many total records are in the data?",
            "Which categories or segments appear most frequently?",
            "What are the top items by value or quantity?",
            "What is the average of the primary measure?",
            "How is the main metric distributed across categories?",
            "What is the total of the main numeric measure?",
        ][:limit]

    # Defensive filter + dedupe (keeps business wording, drops any blocked
    # pattern that slipped in from older templates).
    suggestions = [s for s in suggestions if not _is_blocked_suggestion(s)]
    suggestions = list(dict.fromkeys(suggestions))[:limit]
    # Guarantee the 4-8 window when the schema yielded too few: pad with
    # honest answerable generics rather than returning 1-2 items.
    if len(suggestions) < 4:
        for _pad in ("How many records are in the data?",
                     "Which categories appear most frequently?",
                     "What are the top items by value?"):
            if _pad not in suggestions:
                suggestions.append(_pad)
            if len(suggestions) >= 4:
                break

    # ── Preference-aware ranking ───────────────────────────────────────
    # Boost suggestions similar to previously clicked ones.
    try:
        from api.feedback_routes import _load_prefs
        prefs = _load_prefs(user["user_id"])
        clicks = prefs.get("suggestion_clicks", {})
        if clicks:
            # Score each suggestion by similarity to clicked suggestions
            scored = []
            for s in suggestions:
                score = 0
                s_lower = s.lower()
                for clicked_text, count in clicks.items():
                    clicked_lower = clicked_text.lower()
                    # Simple word overlap scoring
                    s_words = set(s_lower.split())
                    c_words = set(clicked_lower.split())
                    overlap = len(s_words & c_words)
                    if overlap > 0:
                        score += overlap * count
                scored.append((score, s))
            # Stable sort: higher score first, preserve original order for ties
            scored.sort(key=lambda x: -x[0])
            suggestions = [s for _, s in scored]
    except Exception as exc:
        log.warning("Preference-aware suggestion ranking failed: %s", exc, exc_info=True)

    return {"ok": True, "provider": session.get("provider_name", provider), "suggestions": suggestions}


@router.post("/api/ask")
def ask(request: Request, req: GoalRequest, user: dict = Depends(require_writable)):
    """Run the Goal Agent: goal -> SQL -> processed_data.json.

    Uses the connected database's actual schema (fully dynamic — no hardcoded
    table/column names) and falls back to deterministic SQL templates when the
    LLM is slow or unavailable, so any analytical question about the data is
    handled rather than timing out.
    """
    session = get_session(user["user_id"])
    provider = require_data(request)
    # Fail fast: never query the wrong DB. A relational session without a
    # rebuilt db_uri (undecryptable/missing credentials) must reconnect.
    if session.get("source_type") == "relational" and not session.get("db_uri"):
        raise HTTPException(
            status_code=400,
            detail="No database connection in session — reconnect first",
        )
    # P3 §31: structured pipeline log (internal only, never returned).
    import time as _t0
    import uuid as _uuid
    _req_id = f"q_{_uuid.uuid4().hex[:12]}"
    _start = _t0.time()

    # ── Conversational memory ──────────────────────────────────────────
    conv_id = req.conversation_id or session.get("conversation_id")
    if not conv_id:
        conv_id = f"conv_{user['user_id']}_{int(datetime.now().timestamp())}"
    session["conversation_id"] = conv_id

    # Retrieve conversation history from session
    conv_history = session.get("conversation_history", [])

    # DEFECT 1: pure conversational inputs (greetings, thanks, bye, help,
    # 'who are you', empty) must NEVER inherit multi-turn history (which
    # turned 'hi' into a data answer) and must NEVER hit validate_goal_text
    # (which 400s single-word inputs). Detect on the CURRENT question only,
    # schema-agnostic, before any history augmentation or validation.
    def _is_conversational_text(_t):
        import re as _re2
        try:
            _cur = str(_t or "")
            _m = _re2.search(r"current question\s*:\s*(.+)$", _cur,
                             _re2.IGNORECASE | _re2.DOTALL)
            if _m and _m.group(1).strip():
                _cur = _m.group(1).strip()
            _low = _cur.strip().lower()
            if not _low or not _re2.search(r"[a-z]", _low):
                return True
            _norm = _re2.sub(r"[^a-z ]+", " ", _low)
            _norm = _re2.sub(r"\s+", " ", _norm).strip()
            if not _norm:
                return True
            _words = _norm.split()
            if len(_words) > 8:
                return False
            _hints = ("total", "sum", "count", "average", "avg", "revenue",
                      "sales", "profit", "quantity", "amount", "value",
                      "price", "cost", "trend", "monthly", "growth",
                      "decline", "decrease", "increase", "change", "drop",
                      "fall", "top", "compare", "distribution", "breakdown",
                      "share", "month", "year", "over time", "how many",
                      "number of")
            _stripped = _re2.sub(
                r"^(hi+|hello|helo|heloo|hey|yo|howdy|greetings|good\s+"
                r"(morning|afternoon|evening))[\s,!\.]*", "", _norm).strip()
            if _stripped and _stripped != _norm and any(
                    h in f" {_stripped} " for h in _hints):
                return False
            if any(h in f" {_norm} " for h in _hints) and _norm not in (
                    "hi", "hello", "hey", "thanks", "thank you",
                    "thankyou", "thx"):
                return False
            if _norm in ("hi", "hii", "hiii", "hello", "helo", "heloo",
                         "hey", "yo", "howdy", "greetings", "hi there",
                         "hello there", "hey there", "hi aria",
                         "hello aria", "hey aria", "thanks", "thank you",
                         "thankyou", "thx", "ty", "thanks a lot",
                         "thank you so much", "thank you very much",
                         "many thanks", "bye", "goodbye", "good bye",
                         "see you", "see you soon", "cya", "farewell",
                         "help", "help me", "help please", "i need help",
                         "what can you do", "what can u do",
                         "what do you do", "how does this work",
                         "how do i use this", "how do i use", "how to use",
                         "guide", "guide me", "show help", "ok", "okay",
                         "k", "got it", "gotit", "cool", "nice", "great",
                         "perfect", "sure", "yes", "yeah", "yep", "fine",
                         "sounds good", "?", "hi?", "hello?", "hey?"):
                return True
            if _re2.search(r"\b(who are you|what are you|your name|about you|"
                           r"who is aria|what is aria|introduce yourself)\b",
                           _norm) and len(_words) <= 8:
                return True
            if _re2.fullmatch(
                    r"(good\s+morning|good\s+afternoon|good\s+evening)"
                    r"(\s+aria)?", _norm):
                return True
            return False
        except Exception:
            return False

    _is_convo = _is_conversational_text(req.goal)

    # Build goal with multi-turn context (conversational turns skip history
    # so 'hi' never inherits previous tables).
    full_goal = req.goal
    if _is_convo:
        full_goal = str(req.goal or "")
    elif req.previous_goal and req.previous_result:
        prev_data = req.previous_result.get("data", [])
        prev_cols = req.previous_result.get("columns", [])
        prev_tables = req.previous_result.get("data_selection", {}).get("tables", [])
        full_goal = (
            f"Previous question: {req.previous_goal}\n"
            f"Previous result columns: {', '.join(prev_cols[:10])}\n"
            f"Previous tables used: {', '.join(prev_tables[:5])}\n"
            f"Current question: {req.goal}"
        )
    elif conv_history:
        # Use stored conversation history for context
        recent = conv_history[-3:]  # last 3 turns
        context_parts = []
        for turn in recent:
            context_parts.append(f"Q: {turn.get('goal', '')}")
            if turn.get("tables"):
                context_parts.append(f"Tables: {', '.join(turn['tables'][:3])}")
        full_goal = "Conversation context:\n" + "\n".join(context_parts) + f"\nCurrent question: {req.goal}"

    if _is_convo:
        goal = str(full_goal or "").strip()
    else:
        goal = validate_goal_text(full_goal)
    # Explicit-intent-wins (§12): a fresh COUNT/ranking-of-records question
    # must not inherit measures from previous turns ("most readings" means
    # COUNT, even after discussing averages). Only when no measure word is
    # present ("most valuable" stays a revenue ranking).
    import re as _re

    _gl = req.goal.lower()
    _measure_words = ("amount", "value", "price", "revenue", "profit", "total",
                      "average", "avg", "cost", "temperature", "temp", "sales",
                      "money", "score", "rate", "percent")
    _count_like = _re.search(
        r"\b(how many|number of|count\b|most\s+\w+|fewest\s+\w+|least\s+\w+)", _gl)
    if _count_like and not any(w in _gl for w in _measure_words):
        goal = (goal + "\nNOTE: the current question explicitly asks for a count "
                "of records. Use COUNT(*) and do not reuse measures or averages "
                "from previous questions.")
    # Fail fast with an actionable error when schema extraction never
    # completed, instead of leaking an internal path from the agent.
    _schema_path = session.get("schema_path")
    if not _schema_path or not Path(str(_schema_path)).exists():
        status_note = session.get("schema_status", "unknown")
        raise HTTPException(
            status_code=400,
            detail=("Schema extraction did not complete for this connection "
                    f"(status: {status_note}). Reconnect or use Rescan Schema on the Data page, "
                    "then ask again."),
        )
    agent = None
    try:
        api_key = resolve_provider_key(provider, session.get("api_keys", {}))
        agent = GoalAgent(
            schema_json_path=str(session["schema_path"]),
            db_uri=session["db_uri"],
            provider=provider,
            api_key=api_key,
            dialect=session.get("dialect"),
            overrides=load_user_overrides(user["user_id"], session.get("db_name", "")),
            schemas=session.get("schemas"),
        )
        # P3 §31 context for the internal diagnostic trace (never exposed).
        try:
            agent._active_request_id = _req_id
            agent._active_session_id = str(user.get("user_id", ""))
            agent._active_dataset_id = str(session.get("db_name", "") or Path(str(session.get("schema_path", ""))).stem)
            agent._provider_name = session.get("provider_name", provider)
            # Multi-turn isolation: the raw current question tells the agent
            # exactly which line of full_goal is current, so history keywords
            # (e.g. a previous turn's 'average') never poison this turn's IR.
            agent._active_current_question = str(req.goal or "").strip()
        except Exception:
            pass
        # Use a bounded execution so slow LLM calls never hang the endpoint.
        agent.process_goal(goal, output_path=str(session["processed_path"]))
    except Exception as exc:
        # Graceful failure: write a fallback result instead of crashing the endpoint.
        # The frontend expects a processed file; we provide a minimal contract.
        # P3 §50: sanitize internal details before shaping the user message.
        from core.errors import sanitize_error as _san_q
        _safe_exc = _san_q(exc)
        fallback_path = session.get("processed_path", Path("/tmp/fallback_processed.json"))
        if isinstance(fallback_path, str):
            fallback_path = Path(fallback_path)
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        fallback_output = {
            "user_goal": goal,
            "kpi_alignment": {},
            "join_path": [],
            "sql_used": None,
            "row_count": 0,
            "data": [],
            "message": f"Analysis could not complete: {_safe_exc}. No result was produced. Check the source and provider settings, then retry.",
            "preprocessing": {"notes": ["Analysis failed before a usable result was produced."]},
            "missing_values_handled": "preserved (NULLs kept as-is)",
            "timestamp": datetime.now().isoformat(),
            "status": "query_failed",
            "suggested_questions": [
                "How many records are in the data?",
                "What is the total of the main measure?",
                "Which categories appear most frequently?",
                "How has the main metric changed over time?",
            ],
        }
        try:
            with fallback_path.open("w", encoding="utf-8") as f:
                json.dump(fallback_output, f, indent=2, default=str)
        except Exception:
            pass
        # Continue with the graceful result instead of raising a 500.
    finally:
        if agent and getattr(agent, "engine", None):
            try:
                agent.engine.dispose()
            except Exception:
                pass

    if not session.get("processed_path") or not Path(str(session.get("processed_path", ""))).exists():
        # If the agent produced nothing, return the graceful message directly.
        return {
            "ok": True,
            "provider": session.get("provider_name", provider),
            "status": "query_failed",
            "message": "The analytical engine could not complete this request. The database schema is loaded dynamically. Try a simpler analytical question or verify the database has numeric/datetime columns for analysis.",
            "data": [],
            "suggested_questions": [
                "How many total records?",
                "What is the sum of key values?",
                "Which items rank highest?",
                "Is there a time-based trend?",
            ],
        }
    processed = json.loads(session["processed_path"].read_text(encoding="utf-8"))
    processed = sanitize_json(processed)
    # DEFECT 1 passthrough: status=conversation flows untouched — message +
    # suggestions, empty data, no SQL, no fallback, no analytical rewrite.
    if str(processed.get("status") or "") == "conversation":
        try:
            _conv_sugg = list(processed.get("suggested_questions", []) or [])[:8]
        except Exception:
            _conv_sugg = []
        try:
            conv_history.append({
                "goal": req.goal,
                "tables": [],
                "columns": [],
                "sql": None,
                "row_count": 0,
                "timestamp": datetime.now().isoformat(),
            })
            session["conversation_history"] = conv_history[-10:]
            save_session(user["user_id"], session)
        except Exception:
            pass
        return {
            "ok": True,
            "provider": session.get("provider_name", provider),
            "status": "conversation",
            "message": processed.get("message") or processed.get("question") or "",
            "question": processed.get("question") or processed.get("message") or "",
            "data": [],
            "columns": [],
            "row_count": 0,
            "sql_used": None,
            "suggested_questions": _conv_sugg,
            "suggestedQuestions": _conv_sugg,
            "follow_up_questions": _conv_sugg,
            "conversation_id": conv_id,
            "turn_number": len(conv_history),
            "warnings": [],
        }
    columns = list(processed.get("data", [{}])[0].keys()) if processed.get("data") else []

    # ── Deterministic fallback: never return empty when the schema can answer.
    # Vague/business questions ("electronics sales down", "compare segments")
    # used to end as query_failed with no graph. If the agent left no rows,
    # run a simple top-breakdown so the UI always has numbers + chart + 360.
    # The substitution is always labeled partial (never success) with an
    # honest message, quoted dot-aware for multi-schema tables, and worded
    # in plain business language. Generic — no table names.
    try:
        _needs_fb = (not processed.get("data")) or (processed.get("status") == "query_failed")
        if _needs_fb:
            from core.schema_intel import SchemaIntelligence as _SI
            from core.schema_intel import _safe_name as _qn_fb
            try:
                from core.semantic import humanize as _hum_fb
            except Exception:
                _hum_fb = None

            def _plain_fb(v):
                try:
                    s = str(v or "").split(".")[-1]
                    return _hum_fb(s).lower() if _hum_fb else s
                except Exception:
                    return str(v or "")

            _sp = Path(str(session.get("schema_path", "")))
            if _sp.exists():
                _sdata = json.loads(_sp.read_text(encoding="utf-8"))
                _intel_fb = _SI(_sdata)
                _tables = list((_sdata.get("tables", {}) or {}).keys())
                _fb_done = False
                for _t in _tables[:5]:
                    if _fb_done:
                        break
                    try:
                        _m = _intel_fb.find_best_measure(_t)
                        _d = _intel_fb.find_best_dimension(_t)
                        if not _m:
                            continue
                        from api.kpi_routes import _build_engine as _be_fb
                        from sqlalchemy import text as _txt_fb
                        _eng_fb, _dial_fb, _dbt_fb = _be_fb(session)
                        if _eng_fb is None:
                            break
                        try:
                            with _eng_fb.connect() as _conn:
                                if _d:
                                    _sql_fb = (f"SELECT {_qn_fb(_d, _dial_fb)} AS seg, "
                                               f"SUM({_qn_fb(_m, _dial_fb)}) AS val "
                                               f"FROM {_qn_fb(_t, _dial_fb)} "
                                               f"GROUP BY {_qn_fb(_d, _dial_fb)} "
                                               f"ORDER BY val DESC LIMIT 10")
                                else:
                                    _sql_fb = (f"SELECT SUM({_qn_fb(_m, _dial_fb)}) AS val "
                                               f"FROM {_qn_fb(_t, _dial_fb)}")
                                _rows_fb = [dict(r._mapping) for r in _conn.execute(_txt_fb(_sql_fb))]
                            if _rows_fb:
                                processed["data"] = _rows_fb
                                processed["row_count"] = len(_rows_fb)
                                columns = list(_rows_fb[0].keys())
                                processed["sql_used"] = _sql_fb
                                processed["status"] = "partial"
                                processed["message"] = (
                                    f"Your exact question could not be parsed, so here is the top "
                                    f"breakdown of {_plain_fb(_m)}"
                                    f"{' by ' + _plain_fb(_d) if _d else ''}. "
                                    f"Try rephrasing with a measure and a grouping."
                                )
                                _fb_done = True
                        finally:
                            try:
                                _eng_fb.dispose()
                            except Exception:
                                pass
                    except Exception:
                        continue
    except Exception:
        pass

    # ── Analytical validation / repair / answer synthesis (spec §5-§11) ──
    analysis = analyze_result(
        ir=processed.get("goal_ir") or {},
        columns=columns,
        rows=processed.get("data", []) or [],
        goal=req.goal,
        sql=processed.get("sql_used", ""),
        row_count=processed.get("row_count", 0),
    )
    result_data = analysis["data"]
    columns = analysis["columns"]
    if analysis.get("repaired"):
        processed["data"] = result_data
        processed["row_count"] = len(result_data)
        processed["columns"] = columns
    # Result-validation gate (n.txt item 2): never send a broken frame blindly
    # to the UI. Invalid/empty results keep status honest and fall back to a
    # table viz; guards/auth above (require_writable) are untouched.
    # P3 §31 structured log (internal only — never added to the response).
    import logging as _logging
    _qlog = _logging.getLogger("aria.query_pipeline")
    _valid = bool(analysis.get("valid")) and bool(processed.get("sql_used"))
    _shape = {"rows": len(result_data or []), "columns": list(columns or [])}
    _tables = processed.get("data_selection", {}).get("tables", []) or processed.get("join_path", [])
    _lat = round((_t0.time() - _start) * 1000)
    _qlog.info(
        "pipeline request_id=%s session_id=%s dataset_id=%s question=%r "
        "intent=%s tables=%s columns=%s sql=%.200s validation=%s "
        "exec_status=%s result_shape=%s context=%s model=%s latency_ms=%s final_status=%s",
        _req_id, str(user.get("user_id", "")),
        str(session.get("db_name", "") or ""),
        req.goal[:160],
        (processed.get("goal_ir") or {}).get("intent"),
        _tables, list(columns or [])[:10],
        str(processed.get("sql_used") or "")[:200],
        {"valid": analysis.get("valid"), "repaired": analysis.get("repaired")},
        "ok" if _valid else "failed", _shape,
        {"chart": (analysis.get("chart_spec") or {}).get("chart_type")},
        session.get("provider_name", provider), _lat,
        processed.get("status", "success"),
    )
    if not _valid and processed.get("status") == "success":
        processed["status"] = "query_failed"
    if not result_data and (analysis.get("chart_spec") or {}).get("chart_type") != "table":
        analysis["chart_spec"] = dict(analysis.get("chart_spec") or {},
                                     chart_type="table",
                                     reason="empty result paired with a non-table chart; downgraded to table")

    # Run proactive intelligence analysis on the validated result data
    proactive = analyze_result_intelligence(
        data=result_data,
        goal=req.goal,
        message=analysis.get("answer") or processed.get("message", ""),
    )

    # ── Provenance ─────────────────────────────────────────────────────
    tables_used = processed.get("data_selection", {}).get("tables", [])
    join_path = processed.get("join_path", [])
    # Normalize join-path entries: 4-tuples, edge-dicts, or opaque strings.
    norm_joins = []
    for j in join_path or []:
        try:
            if isinstance(j, dict) and {"from", "to"} <= set(j):
                norm_joins.append({
                    "from": j.get("from"), "from_col": j.get("from_col"),
                    "to": j.get("to"), "to_col": j.get("to_col")})
            elif isinstance(j, (list, tuple)) and len(j) >= 4:
                norm_joins.append({"from": j[0], "from_col": j[1], "to": j[2], "to_col": j[3]})
        except Exception:
            continue
    # For aggregate queries, row_count is the number of result rows, not total records
    sql_row_count = processed.get("row_count", 0)
    provenance = {
        "source": session.get("db_name", "unknown"),
        "source_type": session.get("source_type", "unknown"),
        "dialect": session.get("dialect", "unknown"),
        "tables": tables_used,
        "columns": columns[:10],
        "sql": processed.get("sql_used", ""),
        "join_path": norm_joins,
        "row_count": sql_row_count,
        "data_preview": result_data[:3] if result_data else [],
        # Structured envelope (t.txt section 7/57 — additive keys only).
        "methodology": processed.get("methodology", "SQL aggregation over connected source; LLM used for intent/explanation only."),
        "confidence": proactive.get("confidence", "high"),
        "assumptions": processed.get("assumptions", []),
        "warnings": processed.get("warnings", []),
        "timestamp": datetime.now().isoformat(),
    }

    # ── Enhanced follow-up suggestions ─────────────────────────────────
    # Humanized labels only: raw column/table identifiers never surface in
    # user-facing suggestion prose. Every suggestion must be answerable as a
    # plain ask: verified measures/datetimes only, never ID-like segments,
    # never outlier / correlation / forecast / what-if (those need
    # drivers/history and live on their own tabs).
    from core.analytical_result import natural_label as _nl
    base_suggestions = [s for s in (processed.get("suggested_questions", []) or [])
                        if s and not _is_blocked_suggestion(str(s))]
    proactive_suggestions = []
    if result_data and columns:
        first_row = result_data[0] if result_data else {}
        num_cols = []
        cat_cols = []
        for c in columns:
            # Raw SQL expressions (SUM(x), COUNT(*)) and keys are data
            # plumbing, never suggestion prose. ID-like columns (any
            # separator: order_id, "Order ID") are never suggested as
            # measures or breakdowns.
            if "(" in str(c) or "*" in str(c):
                continue
            try:
                from core.schema_intel import _is_id_like as _id_like
                if _id_like(str(c)):
                    continue
            except Exception:
                if str(c).lower() == "id" or str(c).lower().endswith("_id"):
                    continue
            v = first_row.get(c) if isinstance(first_row, dict) else None
            if isinstance(v, (int, float)):
                num_cols.append(c)
            elif v:
                cat_cols.append(c)
        if num_cols:
            proactive_suggestions.append(f"What is the average {_nl(num_cols[0])}?")
            proactive_suggestions.append(f"How is {_nl(num_cols[0])} distributed?")
        if cat_cols and num_cols:
            proactive_suggestions.append(f"Show {_nl(num_cols[0])} by {_nl(cat_cols[0])}")
            proactive_suggestions.append(f"Which {_nl(cat_cols[0])} has the highest {_nl(num_cols[0])}?")
    # Schema-derived fallback suggestions (spec §21): built from the actual
    # connected schema, never a static generic list.
    general_suggestions = []
    try:
        from core.schema_intel import SchemaIntelligence, _is_id_like as _id_like2

        _schema_path = Path(str(session.get("schema_path", "")))
        if _schema_path.exists():
            with open(_schema_path, "r", encoding="utf-8") as _fh:
                _schema_data = json.load(_fh)
            _intel = SchemaIntelligence(_schema_data)
            for _t in list(_intel.tables.keys())[:3]:
                _m = _intel.find_best_measure(_t)
                if _m and not _suggestion_measure_ok(session, _t, _m):
                    _m = None
                _d = _intel.find_best_dimension(_t)
                if _d and _id_like2(str(_d)):
                    _d = None
                _dt = _intel.find_best_datetime(_t)
                if _dt and not _suggestion_datetime_ok(session, _t, _dt):
                    _dt = None
                if _m and _d:
                    general_suggestions.append(f"Show {_nl(_m)} breakdown by {_nl(_d)}")
                    general_suggestions.append(f"Which {_nl(_d)} has the highest {_nl(_m)}?")
                elif _m:
                    _ml2 = _nl(_m)
                    if _ml2.strip().lower() in ("total", "sum", "count", "value", "amount"):
                        general_suggestions.append("What is the overall total value?")
                    else:
                        general_suggestions.append(f"What is the total {_ml2}?")
                else:
                    general_suggestions.append("How many records are in the data?")
                if _m and _dt:
                    general_suggestions.append(f"How has {_nl(_m)} changed over time?")
                if len(general_suggestions) >= 6:
                    break
    except Exception as exc:
        log.warning("Schema-driven follow-up suggestions failed: %s", exc, exc_info=True)
    if not general_suggestions:
        # No schema context at all: say so honestly instead of inventing specifics.
        general_suggestions = ["What data is currently connected?"]
    all_suggestions = list(dict.fromkeys(
        [s for s in (base_suggestions + proactive_suggestions + general_suggestions)
         if s and not _is_blocked_suggestion(str(s))]))[:8]

    # ── Preference-aware follow-up ranking ──────────────────────────────
    try:
        from api.feedback_routes import _load_prefs
        prefs = _load_prefs(user["user_id"])
        clicks = prefs.get("suggestion_clicks", {})
        if clicks:
            scored = []
            for s in all_suggestions:
                score = 0
                s_lower = s.lower()
                for clicked_text, count in clicks.items():
                    clicked_lower = clicked_text.lower()
                    s_words = set(s_lower.split())
                    c_words = set(clicked_lower.split())
                    overlap = len(s_words & c_words)
                    if overlap > 0:
                        score += overlap * count
                scored.append((score, s))
            scored.sort(key=lambda x: -x[0])
            all_suggestions = [s for _, s in scored]
    except Exception as exc:
        log.warning("Preference-aware follow-up ranking failed: %s", exc, exc_info=True)

    # ── Store turn in conversation history ─────────────────────────────
    turn_record = {
        "goal": req.goal,
        "tables": tables_used,
        "columns": columns[:5],
        "sql": processed.get("sql_used", ""),
        "row_count": processed.get("row_count", 0),
        "timestamp": datetime.now().isoformat(),
    }
    conv_history.append(turn_record)
    # Keep last 10 turns
    session["conversation_history"] = conv_history[-10:]
    save_session(user["user_id"], session)

    session["pending_history_id"] = auth.add_history(
        user["user_id"],
        goal=goal,
        sql_used=processed.get("sql_used"),
        row_count=processed.get("row_count"),
        columns=columns,
        source_type=session.get("source_type"),
        db_name=session.get("db_name"),
        dialect=session.get("dialect"),
        processed=processed,
    )
    # ── Why / story / chart-spec envelope (§20-24, intent-aware) ─────────
    why_block = analysis["why"]
    story_text = analysis["story"]
    chart_spec = analysis["chart_spec"]
    consistency = analysis["consistency"]
    answer_message = analysis["answer"] or processed.get("message") or _summarize_result(
        req.goal, columns, result_data)
    # Cross-source: document evidence fused alongside the structured answer.
    # Per astra.txt Phase 2: RAG provides *context*, not proof.  The
    # response carries ``document_context`` (from uploaded PDFs) separately
    # from ``data_evidence`` (from SQL analytics) so the Decision Center
    # can label each channel appropriately.
    doc_evidence = _document_evidence(user["user_id"], req.goal)
    if doc_evidence:
        provenance["document_context"] = [
            {"document_id": d["document_id"], "page": d["page_number"],
             "excerpt": d["content"][:200],
             "content_type": d.get("content_type", "text"),
             "filename": d.get("filename")} for d in doc_evidence
        ]
        # Backward-compat: keep the legacy key for any frontend code that
        # still reads ``document_evidence`` directly.
        provenance["document_evidence"] = provenance["document_context"]
    # ── Explicit display metadata (t.txt-5): server-defined metric,
    # aggregation, dimensions, chart recommendation and KPI values. The
    # frontend renders these instead of recomputing them from raw rows.
    _ainfo = analysis.get("analysis", {}) or {}
    _cspec = analysis.get("chart_spec", {}) or {}
    _kpis = {}
    for _c in columns:
        try:
            _vals = [float(r.get(_c)) for r in result_data
                     if r.get(_c) is not None and str(r.get(_c)).strip() != ""
                     and float(r.get(_c)) == float(r.get(_c))]
        except (TypeError, ValueError):
            continue
        if len(_vals) >= 1 and len(_vals) >= max(1, len(result_data) // 2):
            _kpis[_c] = {
                "sum": round(sum(_vals), 2),
                "avg": round(sum(_vals) / len(_vals), 2),
                "min": round(min(_vals), 2),
                "max": round(max(_vals), 2),
                "count": len(_vals),
            }
    display = {
        "metric": _ainfo.get("measure_label"),
        "measure_column": _ainfo.get("measure_column"),
        "aggregation": _ainfo.get("aggregation"),
        "dimensions": _ainfo.get("dimension_columns", []),
        "chart": {"type": _cspec.get("chart_type"), "x": _cspec.get("x"),
                  "y": _cspec.get("y"), "label": _cspec.get("label"),
                  "reason": _cspec.get("reason")},
        "kpis": _kpis,
    }
    # ── Stable analytics contract (AI-Analytics 1-8, 17-20, additive) ──
    # Frontend renders whatever is present; nothing here is hardcoded to a
    # domain or phrase — intent comes from the classifier + schema grounding,
    # values come from executed SQL, errors are sanitized.
    try:
        from goal_agent.analytics_intent import (
            classify_analytics as _classify,
            resolve_comparison as _resolve_cmp,
            map_visualization as _map_viz,
            compute_possible_drivers as _drivers,
            build_followups as _followups,
            decide_status as _decide,
            graceful_message as _grace_msg,
        )
        from core.errors import sanitize_error as _san
        _schema_terms = _schema_terms_for(session.get("schema_path"))
        _cls = _classify(req.goal, _schema_terms)
        _primary = _cls.get("primary", "descriptive_summary")
        _capabilities = _cls.get("capabilities", [_primary])
        _cmp = _resolve_cmp(req.goal, req.time_range,
                            req.previous_context
                            or ({"question": req.previous_goal}
                                if req.previous_goal else None))
        _metric_col = (_ainfo.get("measure_column")
                       or display.get("measure_column")
                       or req.metric)
        _metric_label = (_ainfo.get("measure_label")
                         or display.get("metric")
                         or _metric_col)
        _dims = list(_ainfo.get("dimension_columns", [])
                     or display.get("dimensions", []) or [])
        if req.grouping:
            _g = [req.grouping] if isinstance(req.grouping, str) else list(req.grouping)
            for _x in _g:
                if _x and _x not in _dims:
                    _dims.append(_x)
        _tables_used = list(tables_used or []) or list(
            (processed.get("data_selection", {}) or {}).get("tables", []) or [])
        _main_table = _tables_used[0] if _tables_used else None
        # Possible drivers from REAL aggregates (best-effort, never fatal).
        _drivers_list: list = []
        try:
            if _main_table and _metric_col and result_data:
                from sqlalchemy import create_engine as _mk
                _eng = _mk(session["db_uri"])
                try:
                    _drivers_list = _drivers(
                        _eng, session.get("dialect", "postgresql"),
                        _main_table, _metric_col, _dims,
                        period_label=_cmp.get("granularity", ""))
                finally:
                    try:
                        _eng.dispose()
                    except Exception:
                        pass
        except Exception:
            _drivers_list = []
        _viz_type = _map_viz(_primary, chart_spec)
        _evidence = [{
            "tables": _tables_used,
            "columns": columns[:10],
            "row_count": len(result_data or []),
            "values": (result_data or [])[:5],
            "sql": processed.get("sql_used", ""),
        }]
        _findings: list[str] = []
        _ans = (answer_message or "").strip()
        if _ans:
            _findings.append(_ans[:500])
        _findings.append(_cmp.get("description", ""))
        for _pi in (proactive.get("proactive_insights", []) or [])[:3]:
            _t = _pi if isinstance(_pi, str) else str(
                (_pi or {}).get("insight") or (_pi or {}).get("text") or _pi)[:300]
            if _t and _t not in _findings:
                _findings.append(_t)
        _limitations = list((why_block or {}).get("limitations", []) or [])
        for _iss in (analysis.get("issues", []) or []):
            if _iss and _iss not in _limitations:
                _limitations.append(str(_iss))
        _recommendations: list[dict] = []
        for _pi in (proactive.get("proactive_insights", []) or [])[:4]:
            if isinstance(_pi, dict) and (_pi.get("action") or _pi.get("recommendation")):
                _recommendations.append({
                    "text": str(_pi.get("action") or _pi.get("recommendation"))[:400],
                    "evidence": _pi.get("evidence") or _tables_used,
                    "confidence": _pi.get("confidence") or proactive.get("confidence", "medium"),
                })
        _hypothesis = None
        if _primary in ("explanation", "root_cause", "hypothesis",
                        "change_detection", "anomaly") or _drivers_list:
            try:
                from goal_agent.analytics_intent import build_hypothesis as _build_hyp
                _top_period = ""
                try:
                    _top_period = str((_drivers_list[0] or {}).get("period") or "")
                except Exception:
                    _top_period = ""
                _hypothesis = _build_hyp(
                    _drivers_list,
                    metric_label=_metric_label,
                    metric_col=_metric_col,
                    period_label=_top_period or _cmp.get("granularity", ""),
                    observation_count=len(result_data or []),
                    dimensions=_dims,
                )
            except Exception:
                if _drivers_list:
                    _top = _drivers_list[0]
                    _hypothesis = {
                        "text": (f"Observed association: {_top.get('value')} "
                                 f"({_top.get('dimension')}) accounts for about "
                                 f"{_top.get('contribution_pct')}% of "
                                 f"{_metric_label} in "
                                 f"{_top.get('period')}. This is a statistical "
                                 f"association in the data, not a proven cause."),
                        "confidence": _top.get("confidence", "low"),
                        "drivers": [d.get("driver") for d in _drivers_list[:3]],
                    }
                else:
                    _hypothesis = {
                        "text": (f"No strong association found for {_metric_label}: "
                                 "no single segment dominates in the returned "
                                 "data. Statistical association only, not proven cause."),
                        "confidence": "low",
                        "drivers": [],
                    }
        _follow_dims = []
        try:
            from core.schema_intel import _is_id_like as _env_id_like
            for d in _dims:
                if "(" in str(d) or "*" in str(d):
                    continue
                if _env_id_like(str(d)):
                    continue
                _follow_dims.append(d)
        except Exception:
            _follow_dims = [d for d in _dims
                            if "(" not in str(d) and "*" not in str(d)][:4]
        _follow_dims = _follow_dims[:4]
        # Trend follow-ups only when a real datetime column with data exists.
        _has_dt = False
        try:
            from core.schema_intel import SchemaIntelligence as _SI_env
            _sp_env = Path(str(session.get("schema_path", "")))
            if _sp_env.exists():
                _sd_env = json.loads(_sp_env.read_text(encoding="utf-8"))
                _intel_env = _SI_env(_sd_env)
                for _et in (_tables_used or list(_intel_env.tables.keys())[:1]):
                    _cand_dt = _intel_env.find_best_datetime(_et)
                    if _cand_dt and _suggestion_datetime_ok(session, _et, _cand_dt):
                        _has_dt = True
                        break
        except Exception:
            _has_dt = False
        _follow = _followups(req.goal, _primary, _metric_label,
                             _follow_dims,
                             _cmp.get("granularity", ""), _drivers_list,
                             has_datetime=_has_dt)
        # Merge result-derived follow-ups first, keep backend suggestions.
        # Blocked patterns (outlier/correlation/forecast/what-if) never
        # surface as plain ask questions; dedupe + cap 4-8.
        _merged_follow = list(dict.fromkeys(
            [s for s in (list(_follow) + list(all_suggestions or []))
             if s and not _is_blocked_suggestion(str(s))]))[:8]
        _clar = bool(processed.get("status") == "needs_clarification"
                     or "clarification" in str(processed.get("status", "")))
        _rstatus = _decide(valid=bool(analysis.get("valid", True)),
                           rows=result_data or [],
                           repaired=bool(analysis.get("repaired", False)),
                           clarification=_clar,
                           issues=analysis.get("issues", []) or [])
        if processed.get("status") not in ("success", "needs_clarification"):
            if processed.get("status") == "partial" and result_data:
                # Honest substitution (fallback answered a nearby question):
                # keep partial so the UI shows the rephrase hint, never a
                # failure banner and never a fake success.
                _rstatus = "partial"
            else:
                _rstatus = "backend-failure" if _rstatus == "success" else _rstatus
        _analytics_envelope = {
            "interpreted_intent": {"primary": _primary,
                                   "capabilities": _capabilities,
                                   "confidence": _cls.get("confidence", "low")},
            "metric": _metric_label,
            "metric_column": _metric_col,
            "dimensions": _dims,
            "filters": req.filters or (processed.get("data_selection", {}) or {}).get("filters", []),
            "time_range": req.time_range or {},
            "comparison_period": _cmp,
            "answer": _ans,
            "findings": _findings[:8],
            "evidence": _evidence,
            "visualization": {"type": _viz_type,
                              "data": (result_data or [])[:50],
                              "x": (_cspec.get("x")),
                              "y": (_cspec.get("y"))},
            "confidence": proactive.get("confidence", "high"),
            "limitations": _limitations,
            "possible_drivers": _drivers_list,
            "hypothesis": _hypothesis,
            "recommendations": _recommendations,
            "follow_up_questions": _merged_follow,
            "result_status": _rstatus,
            "result_message": _grace_msg(_rstatus),
        }
    except Exception as _env_exc:
        from core.errors import sanitize_error as _san2
        _analytics_envelope = {
            "interpreted_intent": {"primary": "descriptive_summary",
                                   "capabilities": ["descriptive_summary"],
                                   "confidence": "low"},
            "metric": None, "metric_column": None, "dimensions": [], "filters": req.filters or [],
            "time_range": req.time_range or {},
            "comparison_period": {"granularity": "month",
                                  "description": "Comparison unavailable."},
            "answer": answer_message, "findings": [answer_message] if answer_message else [],
            "evidence": [], "visualization": {"type": "bar", "data": []},
            "confidence": "low", "limitations": [_san2(_env_exc)],
            "possible_drivers": [], "hypothesis": None, "recommendations": [],
            "follow_up_questions": list(all_suggestions or [])[:8],
            "result_status": "partial" if result_data else "backend-failure",
            "result_message": ("Extended analytics metadata was unavailable. Review the partial results."
                               if result_data else "Analysis did not produce a usable result. Please retry."),
        }
    # ── Business 360 (non-technical 360° view: what + why + what-to-do) ──
    # Best-effort, never fatal. Powers the AI Analytics inline card.
    _b360 = None
    try:
        from services.business_360 import build_business360 as _build360
        _rx_for_360 = None
        try:
            _tables_360 = list(tables_used or []) or list(
                (processed.get("data_selection", {}) or {}).get("tables", []) or [])
            _t360 = _tables_360[0] if _tables_360 else None
            if _t360 and _metric_col:
                from sqlalchemy import create_engine as _mk360
                from core.schema_intel import SchemaIntelligence as _SI360
                _sp360 = Path(str(session.get("schema_path", "")))
                if _sp360.exists():
                    _sd360 = json.loads(_sp360.read_text(encoding="utf-8"))
                    _eng360 = _mk360(session["db_uri"])
                    try:
                        from prescription_engine.engine import analyze as _rx_analyze
                        _rx_for_360 = _rx_analyze(
                            _eng360, session.get("dialect", "postgresql"),
                            _SI360(_sd360), _t360, _metric_col,
                            question=req.goal)
                    finally:
                        try:
                            _eng360.dispose()
                        except Exception:
                            pass
        except Exception:
            _rx_for_360 = None
        _b360 = _build360(
            question=req.goal,
            result_data=result_data,
            columns=columns,
            metric_label=_analytics_envelope.get("metric") if "_analytics_envelope" in dir() else None,
            metric_col=_metric_col if "_metric_col" in dir() else None,
            dims=_dims if "_dims" in dir() else [],
            drivers=_drivers_list if "_drivers_list" in dir() else [],
            hypothesis=_hypothesis if "_hypothesis" in dir() else None,
            prescription=_rx_for_360,
            chart_spec=chart_spec,
        )
    except Exception:
        _b360 = None
    _resp = {
        "ok": True,
        "provider": session["provider_name"],
        "business360": _b360,
        "status": processed.get("status", "success"),
        "question": processed.get("question"),
        "user_goal": processed.get("user_goal"),
        "kpi_alignment": processed.get("kpi_alignment"),
        "join_path": processed.get("join_path"),
        "sql_used": processed.get("sql_used"),
        "row_count": processed.get("row_count"),
        "columns": columns,
        "message": answer_message,
        "preprocessing": processed.get("preprocessing"),
        "data": processed.get("data", [])[:50],
        "goal": processed.get("goal"),
        "data_selection": processed.get("data_selection"),
        "analysis_plan": processed.get("analysis_plan"),
        "execution": processed.get("execution"),
        "warnings": processed.get("warnings"),
        "suggested_questions": _analytics_envelope.get("follow_up_questions", all_suggestions),
        "suggestedQuestions": _analytics_envelope.get("follow_up_questions", all_suggestions),
        "proactive_insights": proactive.get("proactive_insights", []),
        "anomalies": proactive.get("anomalies", []),
        "data_quality_warnings": proactive.get("data_quality_warnings", []),
        "explanation": proactive.get("explanation"),
        "confidence": proactive.get("confidence", "high"),
        "why": why_block,
        "story": story_text,
        "chart_spec": chart_spec,
        "display": display,
        "consistency": consistency,
        "analysis": analysis.get("analysis"),
        "result_valid": analysis.get("valid", True),
        "repaired": analysis.get("repaired", False),
        "document_context": doc_evidence,
        "history_id": session["pending_history_id"],
        "conversation_id": conv_id,
        "turn_number": len(conv_history),
        "provenance": provenance,
        **_analytics_envelope,
    }
    # Verification verdict, best-effort: data is never blocked on failure.
    try:
        _attach_ask_verdict(_resp, session)
    except Exception:
        pass
    return _resp


@router.post("/api/insight")
def insight(request: Request, user: dict = Depends(require_writable)):
    """Run the Insight Agent on the last processed data.

    Legacy path kept for existing callers (Dashboard/Predictive summaries).
    The canonical Insights page uses GET /api/insights (InsightsEngine-shaped
    cards) and never renders this payload as cards.
    """
    session = get_session(user["user_id"])
    provider = require_data(request)
    if not session.get("processed_path") or not Path(str(session.get("processed_path", ""))).exists():
        raise HTTPException(status_code=400, detail="No processed data yet. Call POST /api/ask first.")

    try:
        agent = InsightAgent(provider=provider)
        agent.analyze(str(session["processed_path"]), str(session["insights_path"]))
        insights = json.loads(session["insights_path"].read_text(encoding="utf-8"))
    except Exception as exc:
        from core.errors import sanitize_error as _san_i
        processed = json.loads(session["processed_path"].read_text(encoding="utf-8")) if session["processed_path"].exists() else {}
        insights = {
            "generated_at": datetime.now().isoformat(),
            "user_goal": processed.get("user_goal", "Unknown goal"),
            "summary": {"rows": processed.get("row_count", 0), "columns": []},
            "kpis": [],
            "trends": [],
            "anomalies": [],
            "hypotheses": [],
            "dashboard": [],
            "business_story": "The insight agent could not complete this analysis, but the dataset was still received and the UI is ready for retry.",
            "warning": _san_i(exc),
        }
        session["insights_path"].write_text(json.dumps(sanitize_json(insights), indent=2), encoding="utf-8")

    insights = sanitize_json(insights)

    if session["pending_history_id"]:
        auth.set_history_insights(session["pending_history_id"], user["user_id"], insights)
        session["pending_history_id"] = None
        save_session(user["user_id"], session)

    return {
        "ok": True,
        "provider": session["provider_name"],
        "insights": insights,
    }
