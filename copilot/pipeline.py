"""ARIA Copilot RAG pipeline (provider-independent).

    User question
      -> intent / question understanding (deterministic + conversation memory)
      -> query router (t.txt: intent determines the tool plan, never SQL-only)
      -> retrieve relevant context (schema index + hybrid ranking)
      -> select + execute tools (SQL and/or analysis/forecast/hypothesis/document)
      -> validate results
      -> analytical reasoning over the retrieved data
      -> natural-language business response (+ chart when useful)

Tool plans by intent:
    schema         -> schema index only (no SQL, no LLM needed)
    document       -> document guidance (no SQL; document RAG lives in insight_rag)
    general        -> LLM general knowledge (no SQL, no schema needed)
    data           -> SQL -> validate -> execute -> narrate (classic lookup path)
    forecast       -> SQL history -> trend analysis -> directional outlook + honesty
    hypothesis     -> SQL evidence -> relationship analysis -> possible explanations
    anomaly        -> SQL evidence -> distribution/change scan -> unusual activity
    recommendation -> SQL evidence -> analysis -> business recommendation + evidence
    health         -> SQL evidence -> multi-finding summary -> what to watch
    mixed          -> SQL evidence -> analysis -> hypothesis + recommendation + evidence

The LLM is passed in as an object exposing `.complete(role, prompt, ...)` /
`.chat(role, messages, ...)` — exactly the interface ARIA's existing
`llm_provider.LLMProvider` implements for BOTH Ollama and Groq. This module
never branches on the provider, and never touches the database directly.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from .conversation import ConversationMemory, dataset_fingerprint
from .intent import (
    ANOMALY,
    COMBINED,
    DATA,
    DOCUMENT,
    FORECAST,
    GENERAL,
    HEALTH,
    HYPOTHESIS,
    MIXED,
    RECOMMENDATION,
    SCHEMA,
    classify,
)
from .retriever import SchemaRetriever, RetrievedContext
from .grounded_context import (
    GroundedCtxView,
    build_grounded_db_context,
    decide_knowledge_source,
)
from .answer_grounding import check_answer_grounding
from .schema_index import build_schema_index
from .sql_engine import SqlEngine, clean_sql

log = logging.getLogger(__name__)

_MEASURE_HINTS = ("amount", "total", "revenue", "sales", "price", "cost",
                  "profit", "value", "quantity", "qty", "count", "sum")


@dataclass
class CopilotAnswer:
    """Result of one Copilot turn — the only shape the API/UI needs."""

    ok: bool
    answer: str
    sql: str | None = None
    rows: list[dict] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    chart: dict | None = None
    error: str | None = None
    tables_used: list[str] = field(default_factory=list)
    retrieved: dict | None = None
    attempts: int = 1
    provider: str | None = None
    intent: str | None = None
    grounding: dict | None = None

    def to_public(self, include_rows: int = 50) -> dict:
        """The user-facing payload. Never leaks schema dumps or internals."""
        out = {
            "ok": self.ok,
            "answer": self.answer,
            "sql": self.sql,
            "columns": self.columns,
            "row_count": len(self.rows),
            "data": self.rows[:include_rows],
            "chart": self.chart,
            "tables_used": self.tables_used,
        }
        if self.error:
            out["error"] = self.error
        if self.intent:
            out["intent"] = self.intent
        if self.grounding:
            out["grounding"] = self.grounding
        return out


def _numeric(v) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        s = v.strip().replace(",", "").replace("$", "").replace("%", "")
        try:
            float(s)
            return True
        except ValueError:
            return False
    return False


def _datetime_name(name: str) -> bool:
    return bool(re.search(r"(date|time|month|year|period|day|week|quarter)",
                          str(name or ""), re.IGNORECASE))


def infer_chart(columns: list[str], rows: list[dict], question: str) -> dict | None:
    """Infer a chart-ready spec from the RETURNED data + question intent.

    Fully dynamic: no fixed chart mappings, no fixed column names. Ranking
    questions -> bar/horizontal bar; time questions -> line/area; two measures
    -> scatter; share questions -> pie when few categories.
    """
    if not rows or len(columns) < 2:
        return None
    sample = rows[0]
    numeric_cols = [c for c in columns if _numeric(sample.get(c))]
    date_cols = [c for c in columns if _datetime_name(c)]
    cat_cols = [c for c in columns if c not in numeric_cols and c not in date_cols]
    q = (question or "").lower()

    if not numeric_cols:
        return None

    wants_share = any(w in q for w in ("share", "percentage", "percent", "proportion", "pie", "donut"))
    wants_trend = any(w in q for w in ("trend", "over time", "monthly", "yearly", "by month", "by year", "growth"))
    wants_rank = any(w in q for w in ("top", "bottom", "best", "worst", "highest", "lowest", "most", "least", "rank"))

    x = (date_cols or cat_cols or [columns[0]])[0]
    y = numeric_cols[0]

    if date_cols and not wants_rank:
        chart_type = "area" if wants_trend else "line"
        return {"type": chart_type, "x": date_cols[0], "y": y,
                "reason": "time series detected in the returned data"}
    if len(numeric_cols) >= 2 and not cat_cols and not date_cols:
        return {"type": "scatter", "x": numeric_cols[0], "y": numeric_cols[1],
                "reason": "two measures returned — relationship view"}
    if wants_share and cat_cols and 1 < len(rows) <= 8:
        return {"type": "pie", "x": x, "y": y,
                "reason": "share question over few categories"}
    if wants_rank or len(rows) > 8:
        return {"type": "bar_horizontal", "x": x, "y": y,
                "reason": "ranked comparison"}
    return {"type": "bar", "x": x, "y": y, "reason": "category comparison"}


def _summarize_rows(columns: list[str], rows: list[dict]) -> str:
    """Deterministic fallback answer from the returned rows (no LLM needed).

    Business-friendly by construction: human-scale numbers, no raw dumps.
    """
    if not rows:
        return "No rows matched that question against the connected data."
    if len(columns) == 1:
        return f"The result is {_biz_num(rows[0].get(columns[0]))}."
    numeric_cols = [c for c in columns if _numeric(rows[0].get(c))]
    label_cols = [c for c in columns if c not in numeric_cols]
    if label_cols and numeric_cols:
        label, measure = label_cols[0], numeric_cols[-1]
        top = rows[0]
        return (f"{top.get(label)} has the highest {measure} at {_biz_num(top.get(measure))} "
                f"(across {len(rows)} result rows).")
    return f"Found {len(rows)} matching rows in the connected data."


# What each app page is for (page ids are app constants, never data).
# Used only to answer "what is this page showing?" from UI context.
_PAGE_BLURBS: dict[str, tuple[str, str]] = {
    "dashboard": ("Dashboard", "your business overview — live KPIs, charts, anomalies and recommendations"),
    "analytics": ("AI Analytics", "ask anything and get charts, tables and key insights"),
    "insights": ("Insights", "the anomalies, opportunities and risks ARIA found in your data"),
    "predictive": ("Forecast", "future projections with likely ranges and what-if scenarios"),
    "intelligence": ("Intelligence", "deep dives — explore, hypotheses, prescriptions and reports"),
    "data": ("Data", "where you connect sources and inspect your schema"),
    "documents": ("Documents", "where you chat with your uploaded documents"),
    "history": ("History", "your past analyses"),
    "settings": ("Settings", "providers, models and preferences"),
    "copilot": ("Copilot", "full conversational analysis over your data"),
}


class CopilotPipeline:
    """One Copilot, interchangeable LLM backends."""

    def __init__(self, llm, schema_mapping: dict, db_uri: str,
                 dialect: str = "postgresql", user_id="anon",
                 embedder=None, provider_name: str | None = None,
                 document_id: str | None = None, linked: dict | None = None):
        self.llm = llm
        self.dialect = dialect or "postgresql"
        self.db_uri = db_uri
        self.provider_name = provider_name
        self.user_id = user_id
        # Multi-source (§7): an uploaded document and/or a stashed database
        # connection that coexist with the primary source.
        self.document_id = document_id
        self.linked = linked or {}
        self.memory = ConversationMemory(user_id)
        # Dataset scope for conversation memory (t.txt §13/26): every turn
        # is tagged with this fingerprint, so a dataset switch invalidates
        # stale schema/tables/SQL instead of reusing them. The fingerprint
        # covers column names+types per table, the edge set, document ids
        # and row-count buckets — so even a same-named source with different
        # columns isolates cleanly. Secret-free (names/types/buckets only,
        # never the URI, credentials or values) and provider-independent.
        self.dataset_fp = dataset_fingerprint(
            mapping=schema_mapping or {}, document_id=document_id,
            linked=linked or {},
        )
        self.index = build_schema_index(schema_mapping or {}, dialect=self.dialect)
        self.retriever = SchemaRetriever(self.index, embedder=embedder)
        try:
            _schemas = list((schema_mapping or {}).get("schemas") or [])
        except Exception:
            _schemas = []
        self.sql_engine = SqlEngine(db_uri, dialect=self.dialect, schemas=_schemas)
        self.mapping = schema_mapping or {}
        self._embedder = embedder

    # -- main entry ------------------------------------------------------

    def _context_snapshot(self, context: dict | None) -> dict:
        """Collect visible page/chart fields by shape (generic, never data names).

        Looks top-level plus one nested level (``chart``/``context``) since
        drill links nest the snapshot. Missing fields stay None, never
        invented. Never raises.
        """
        snap: dict = {"page": None, "metric": None, "chart": None,
                      "label": None, "value": None, "segment": None,
                      "period": None, "source": None}
        try:
            if not isinstance(context, dict):
                return snap
            snap["page"] = context.get("page")
            node = context
            for _ in range(2):
                if not isinstance(node, dict):
                    break
                if any(k in node for k in ("metric", "label", "value", "chart",
                                          "segment", "dimension", "period", "source")):
                    break
                nxt = node.get("chart") or node.get("context")
                if not isinstance(nxt, dict):
                    break
                node = nxt
            if isinstance(node, dict):
                snap["metric"] = node.get("metric")
                snap["chart"] = node.get("chart")
                snap["label"] = node.get("label")
                snap["value"] = node.get("value")
                snap["segment"] = node.get("segment", node.get("dimension"))
                snap["period"] = node.get("period")
                snap["source"] = node.get("source")
            # Fall back to top-level for fields the nested node lacked.
            for k in ("metric", "chart", "label", "value", "segment",
                      "period", "source"):
                if snap.get(k) in (None, "") and isinstance(context, dict):
                    if k == "segment":
                        v = context.get("segment", context.get("dimension"))
                    else:
                        v = context.get(k)
                    if v not in (None, ""):
                        snap[k] = v
        except Exception:
            pass
        return snap

    @staticmethod
    def _snapshot_has_substance(snap: dict) -> bool:
        """True when the snapshot carries something answerable (generic)."""
        try:
            if str(snap.get("page") or "") in _PAGE_BLURBS:
                return True
            for k in ("metric", "chart", "label", "value", "segment",
                      "period", "source"):
                v = snap.get(k)
                if v is not None and str(v).strip() != "":
                    return True
        except Exception:
            pass
        return False

    def _describe_snapshot(self, snap: dict) -> str:
        """One generic paragraph from the visible snapshot (no fixed data)."""
        parts: list[str] = []
        try:
            page = str(snap.get("page") or "")
            blurb = _PAGE_BLURBS.get(page, (None, None))
            metric = snap.get("metric")
            chart = snap.get("chart")
            label = snap.get("label")
            value = snap.get("value")
            segment = snap.get("segment")
            period = snap.get("period")
            if chart or metric or label is not None or value is not None or segment:
                head = []
                if chart or metric:
                    head.append(f"This {chart or 'chart'} shows {metric or 'the highlighted measure'}")
                else:
                    head.append("These results show the highlighted values")
                bits = "".join(head)
                if label not in (None, ""):
                    bits += f" — {label}"
                    if value is not None:
                        try:
                            bits += (f" at {float(value):,.2f}".rstrip("0").rstrip(".")
                                     if isinstance(value, (int, float)) or str(value).replace('.', '', 1).lstrip('-').isdigit()
                                     else f" at {value}")
                        except Exception:
                            bits += f" at {value}"
                elif value is not None:
                    try:
                        bits += (f" at {float(value):,.2f}".rstrip("0").rstrip(".")
                                 if isinstance(value, (int, float)) or str(value).replace('.', '', 1).lstrip('-').isdigit()
                                 else f" at {value}")
                    except Exception:
                        bits += f" at {value}"
                if segment not in (None, ""):
                    bits += f" for {segment}"
                if period not in (None, ""):
                    bits += f" ({period})"
                parts.append(bits + ".")
                if blurb[0] is not None:
                    parts.append(f"You're seeing this on {blurb[0]} — {blurb[1]}.")
                parts.append("Ask me to break it down, compare segments, or explain why it matters.")
                return " ".join(parts)
            if blurb[0] is not None:
                db = (self.mapping or {}).get("database") or "your connected data"
                tables = (self.mapping or {}).get("tables", {})
                n = len(tables) if isinstance(tables, dict) else 0
                text = (f"You're on {blurb[0]} — {blurb[1]}. "
                        f"Right now it's showing {db}"
                        + (f" ({n} tables)." if n else "."))
                return text
        except Exception:
            pass
        return ""

    def _answer_page_context(self, question: str,
                               context: dict | None) -> str | None:
        """Answer deictic page/chart questions from visible context.

        Handles "what is this chart/page showing?" plus vague references
        to visible results ("explain these results", "what does this
        mean", "why is this happening") when page/chart context is
        present. Generic: page ids are app constants, chart fields are
        read by shape (metric/chart/label/value), never by database
        names. Returns None when the question is not deictic or the
        context lacks substance, letting normal flow proceed.
        Never raises — callers fall through on any failure.
        """
        try:
            if not isinstance(context, dict) or not context:
                return None
            ql = str(question or "").lower()
            # Only explicit page/chart deixis ("what is this chart showing?",
            # "tell me about this page"). Bare "explain this drop in sales"
            # stays a data question — never hijack analytical wording.
            is_explicit = bool(re.search(
                r"\bthis (page|screen|chart|graph|visual|metric)\b|"
                r"what am i looking at", ql))
            # Vague references to the visible results. Only honored when
            # the UI sent real page/chart substance (checked below), so
            # bare analytical questions without context keep normal flow.
            is_vague = bool(re.search(
                r"explain\s+(these|this|the)\s+(results?|data|numbers?|figures?)|"
                r"\bthese\s+results?\b|\bthis\s+data\b|"
                r"what\s+does\s+(this|these|it)\s+mean|"
                r"what\s+do\s+(these|this|the\s+results?)\s+(mean|show)|"
                r"why\s+is\s+this\s+happening|"
                r"what('s| is)\s+(going\s+on|happening)(\s+here)?|"
                r"summar(y|ise|ize)\s+(this|these|the\s+results?)|"
                r"what\s+does\s+it\s+show|tell\s+me\s+about\s+(these|this|the)\s+(results?|data)",
                ql))
            if not is_explicit and not is_vague:
                return None
            snap = self._context_snapshot(context)
            if not self._snapshot_has_substance(snap):
                return None
            # Chart snapshot may sit top-level (drill links) or nested.
            node = context
            for _ in range(2):
                if not isinstance(node, dict):
                    break
                if any(k in node for k in ("metric", "label", "value", "chart")):
                    break
                nxt = node.get("chart") or node.get("context")
                if not isinstance(nxt, dict):
                    break
                node = nxt
            metric = node.get("metric") if isinstance(node, dict) else None
            chart = node.get("chart") if isinstance(node, dict) else None
            label = node.get("label") if isinstance(node, dict) else None
            value = node.get("value") if isinstance(node, dict) else None
            wants_chart = bool(re.search(
                r"\b(chart|graph|visual|metric|results?|data|mean|happening|going on)\b", ql))
            if wants_chart and (metric or chart or label is not None
                                or value is not None or snap.get("segment")):
                text = self._describe_snapshot(snap)
                if text:
                    return text
                bits = []
                if chart or metric:
                    bits.append(f"This {chart or 'chart'} shows {metric or 'the highlighted measure'}")
                else:
                    bits.append("This chart")
                if label not in (None, ""):
                    bits.append(f" — {label}")
                    if value is not None:
                        try:
                            bits.append(f" at {float(value):,.2f}".rstrip("0").rstrip(".")
                                        if isinstance(value, (int, float)) or str(value).replace('.', '', 1).lstrip('-').isdigit()
                                        else f" at {value}")
                        except Exception:
                            bits.append(f" at {value}")
                text = "".join(bits) + "."
                tables = (self.mapping or {}).get("tables", {})
                if isinstance(tables, dict) and tables:
                    text += " Ask me to break it down, compare it, or explain why it matters."
                return text
            page = context.get("page") if isinstance(context, dict) else None
            blurb = _PAGE_BLURBS.get(str(page or ""), (None, None))
            if blurb[0] is None:
                text = self._describe_snapshot(snap)
                return text or None
            # Vague ask with both page + chart substance: lead with what is
            # visible, not just the page description.
            if is_vague:
                text = self._describe_snapshot(snap)
                if text:
                    return text
            db = (self.mapping or {}).get("database") or "your connected data"
            tables = (self.mapping or {}).get("tables", {})
            n = len(tables) if isinstance(tables, dict) else 0
            text = (f"You're on {blurb[0]} — {blurb[1]}. "
                    f"Right now it's showing {db}"
                    + (f" ({n} tables)." if n else "."))
            return text
        except Exception:
            return None

    def ask(self, question: str, context: dict | None = None) -> CopilotAnswer:
        if not question or not question.strip():
            return CopilotAnswer(ok=False, answer="Please ask a question.",
                                 provider=self.provider_name)
        # Page/chart context (sent by the Mini ARIA panel and chart drill
        # links): deictic questions ("what is this chart showing?") are
        # answered straight from the visible context — no SQL needed, and
        # never an error when the context carries the substance.
        try:
            _page_answer = self._answer_page_context(question, context)
        except Exception:
            _page_answer = None
        if _page_answer is not None:
            try:
                _pg = context.get("page") if isinstance(context, dict) else None
            except Exception:
                _pg = None
            self._remember(question, {"page": _pg, "intent": "page-context"},
                           _page_answer, None, "page-context")
            return CopilotAnswer(ok=True, answer=_page_answer,
                                 provider=self.provider_name,
                                 intent="page-context")
        resolved = self.memory.resolve(question, dataset=self.dataset_fp)
        route = classify(resolved)

        # Schema-less intents run without a relational mapping (t.txt §14):
        # document questions need only the document store, general questions
        # need nothing but the LLM.
        if route.intent in (DOCUMENT, COMBINED, GENERAL):
            if route.intent == COMBINED:
                return self._answer_combined(question, resolved, route.intent)
            if route.intent == DOCUMENT:
                return self._answer_document(question, route.intent)
            return self._answer_general(question, route.intent)

        if not self.index.tables:
            return CopilotAnswer(
                ok=False,
                answer=("No database schema is available yet. Connect a PostgreSQL "
                        "or MySQL source (or upload a CSV/PDF) and try again."),
                provider=self.provider_name,
            )

        if route.intent == SCHEMA:
            return self._answer_schema(question, route.intent)

        if route.intent == DATA:
            return self._answer_data(question, resolved, route.intent)

        # Analytical intents: SQL evidence -> deterministic analysis ->
        # grounded business reasoning (hypothesis / forecast / recommendation).
        return self._answer_analytical(question, resolved, route.intent)

    # -- intent branches -------------------------------------------------

    def _remember(self, question: str, context: dict, answer: str,
                   sql: str | None, intent: str) -> None:
        try:
            self.memory.append(question=question, context=context,
                               answer=answer, sql=sql,
                               dataset=self.dataset_fp)
        except TypeError:
            # Older ConversationMemory.append without extra kwargs.
            self.memory.append(question=question, context=context,
                               answer=answer)

    def _answer_schema(self, question: str, intent: str) -> CopilotAnswer:
        """Answer from the schema index only — never generates SQL."""
        tables = self.index.all_tables()
        bits = []
        db = (self.mapping or {}).get("database") or "connected data"
        bits.append(f"Your {db} has {len(tables)} table{'s' if len(tables) != 1 else ''}: "
                    + ", ".join(sorted(tables)) + ".")
        for t in sorted(tables)[:6]:
            cols = self._table_columns(t)
            if cols:
                bits.append(f"{t}: {', '.join(cols[:8])}"
                            + ("…" if len(cols) > 8 else "") + ".")
        answer = " ".join(bits)
        ctx = {"tables": sorted(tables), "intent": intent}
        self._remember(question, ctx, answer, None, intent)
        return CopilotAnswer(ok=True, answer=answer, tables_used=sorted(tables),
                             retrieved=ctx, provider=self.provider_name,
                             intent=intent)

    def _table_columns(self, table: str) -> list[str]:
        info = (self.mapping or {}).get("tables", {}).get(table, {})
        cols = info.get("columns", [])
        if isinstance(cols, list):
            return [c.get("column") if isinstance(c, dict) else str(c) for c in cols]
        if isinstance(cols, dict):
            return list(cols.keys())
        return []

    # -- combined database + document answers (t.txt §7) -------------------

    def _linked_tools(self):
        """Build (index, retriever, engine, mapping) for the linked database.

        Returns None when no linked database is available. Built lazily so
        single-source sessions pay nothing.
        """
        uri = (self.linked or {}).get("db_uri")
        mapping = (self.linked or {}).get("mapping") or {}
        if not uri or not (mapping.get("tables")):
            return None
        try:
            dialect = (self.linked or {}).get("dialect") or "postgresql"
            index = build_schema_index(mapping, dialect=dialect)
            if not index.tables:
                return None
            return {
                "index": index,
                "retriever": SchemaRetriever(index, embedder=self._embedder),
                "engine": SqlEngine(uri, dialect=dialect),
                "mapping": mapping,
                "db_name": (self.linked or {}).get("db_name") or mapping.get("database") or "database",
            }
        except Exception as exc:
            log.info("Copilot linked-DB tools unavailable (%s).", exc)
            return None

    def _primary_is_document(self) -> bool:
        return str(self.db_uri or "").startswith("document://")

    def _answer_combined(self, question: str, resolved: str, intent: str) -> CopilotAnswer:
        """Database evidence + document evidence, attributed per source.

        Downgrades gracefully: database-only when no document is connected,
        document-only when no database side exists.
        """
        linked = self._linked_tools()
        # Database side: linked DB wins when primary is a document, else primary.
        db_tools = None
        if self._primary_is_document():
            db_tools = linked
        elif self.index.tables:
            db_tools = {"index": self.index, "retriever": self.retriever,
                        "engine": self.sql_engine, "mapping": self.mapping,
                        "db_name": (self.mapping or {}).get("database") or "database"}
            if linked is None:
                pass
        doc_id = self.document_id
        if not doc_id and self._primary_is_document():
            cand = str(self.db_uri).split("document://", 1)[1].split("/")[0]
            if re.fullmatch(r"[A-Za-z0-9_-]{8,80}", cand or ""):
                doc_id = cand

        if db_tools is None and not doc_id and not self._doc_candidates():
            return self._answer_data(question, resolved, DATA)

        sql_text, analysis, rows, result = None, None, [], None
        db_tables: list = []
        if db_tools is not None:
            ctx = db_tools["retriever"].retrieve(resolved)
            if not ctx.is_empty():
                db_tables = list(ctx.tables)
                result = db_tools["engine"].generate_and_run(
                    resolved, ctx, self.llm,
                    tables=(db_tools.get("mapping") or {}).get("tables"),
                    model_role="sql",
                )
                if result.ok:
                    rows = _sanitize_rows(result.rows)
                    analysis = _analyze_rows(result.columns, rows, intent)
                    sql_text = result.sql

        doc_answer = self._retrieve_document_text(question, doc_id) if (doc_id or self._doc_candidates()) else None

        if sql_text is None and doc_answer is None:
            # Neither side produced evidence: fall back to the normal path.
            return self._answer_data(question, resolved, DATA)

        narrative = self._narrate_combined(question, rows, analysis, doc_answer,
                                           db_tools, result)
        chart = infer_chart(result.columns, rows, question) if result and result.ok else None
        ctx_state = {"tables": db_tables, "intent": intent, "combined": True}
        self._remember(question, ctx_state, narrative, sql_text, intent)
        return CopilotAnswer(
            ok=True, answer=narrative, sql=sql_text,
            rows=rows, columns=(result.columns if result else []),
            chart=chart, tables_used=db_tables, retrieved=ctx_state,
            attempts=(result.attempts if result else 1),
            provider=self.provider_name, intent=intent,
        )

    def _doc_candidates(self) -> list[str]:
        """Document ids available beyond the session's pinned one."""
        try:
            from insight_rag.storage import RAGMetadataStore
            docs = RAGMetadataStore(self.user_id).list_documents()
            return [str(d.get("document_id")) for d in docs if d.get("document_id")][:3]
        except Exception as exc:
            log.warning("Failed to list recent documents: %s", exc)
            return []

    def _retrieve_document_text(self, question: str, doc_id: str | None):
        """Shared extractive retrieval; returns (answer, sources) or None."""
        out = self._retrieve_document(question, doc_id)
        return out

    def _narrate_combined(self, question, rows, analysis, doc_answer,
                          db_tools, result) -> str:
        """Synthesize database + document evidence with attribution."""
        db_bits = ""
        if analysis:
            db_bits = "; ".join(analysis.get("findings", [])[:5])
        doc_bits = ""
        if doc_answer:
            doc_bits = doc_answer[0][:1200]
        db_name = (db_tools or {}).get("db_name", "database")
        if db_bits and doc_bits:
            det = (f"From the database ({db_name}): {db_bits}. "
                   f"From the document: {doc_bits} "
                   f"These are two separate pieces of evidence — the document "
                   f"suggests a possible contributing factor, it does not prove cause.")
        elif db_bits:
            det = (f"From the database ({db_name}): {db_bits}. "
                   f"No supporting passage was found in the connected document.")
        else:
            det = (f"From the document: {doc_bits} "
                   f"The database did not return supporting evidence for this question.")
        try:
            prompt = (
                "You are ARIA Copilot, a business analyst explaining data in plain business English. "
                "Combine the two labeled evidence blocks below into 3-4 sentences: what the database shows, "
                "what the document says, and whether they point the same way. Attribute every claim "
                "(say 'the data shows' vs 'the report suggests'). "
                "Round numbers to human scale. Never mention SQL, tables, models, or embeddings. "
                "Never claim causation — use 'may have contributed' language.\n\n"
                f"QUESTION: {question}\n"
                f"DATABASE EVIDENCE: {db_bits or '(none)'}\n"
                f"DOCUMENT EVIDENCE: {doc_bits or '(none)'}\n"
            )
            text = self.llm.complete("story", prompt, temperature=0.2,
                                     num_predict=200)
            text = strip_text(text)
            if text and len(text) >= 3:
                return text
        except Exception as exc:
            log.info("Copilot combined narration LLM failed (%s).", exc)
        return det

    def _answer_document(self, question: str, intent: str) -> CopilotAnswer:
        """Document questions via lexical chunk retrieval — never SQL (t.txt §33).

        Reads the user's uploaded-document store (metadata + text chunks) and
        answers extractively with source labels. Falls back to honest guidance
        when no document or no relevant passage exists.
        """
        doc_id = self.document_id
        if not doc_id:
            uri = str(self.db_uri or "")
            if uri.startswith("document://"):
                cand = uri.split("document://", 1)[1].strip().split("/")[0]
                if re.fullmatch(r"[A-Za-z0-9_-]{8,80}", cand or ""):
                    doc_id = cand
        retrieved = self._retrieve_document(question, doc_id)
        if retrieved is None:
            if doc_id:
                answer = ("I couldn't find a passage in that document that answers "
                          "this reliably. Try asking about a topic, figure, or term "
                          "from the document, or open it in Document Chat for deeper search.")
            else:
                answer = ("I don't see a document connected to this session, so I can't "
                          "read document content. Upload a PDF on the Data page and ask "
                          "in Document Chat, or ask me about the connected tables instead.")
            ctx = {"tables": [], "intent": intent}
            self._remember(question, ctx, answer, None, intent)
            return CopilotAnswer(ok=False, answer=answer,
                                 provider=self.provider_name, intent=intent)
        answer, sources, tables = retrieved
        ctx = {"tables": tables, "intent": intent,
               "sources": [s.get("label") for s in sources]}
        self._remember(question, ctx, answer, None, intent)
        return CopilotAnswer(ok=True, answer=answer, tables_used=tables,
                             retrieved=ctx, provider=self.provider_name,
                             intent=intent)

    def _retrieve_document(self, question: str,
                           doc_id: str | None) -> tuple[str, list[dict], list[str]] | None:
        """Lexical document Q&A over stored chunks. Never raises; None = no answer."""
        try:
            from insight_rag.storage import RAGMetadataStore
        except Exception as exc:
            log.info("Copilot document store unavailable (%s).", exc)
            return None
        try:
            user_id = getattr(self.memory, "user_id", "anon")
            store = RAGMetadataStore(user_id)
            docs = store.list_documents() if not doc_id else [
                d for d in [store.get_document(doc_id)] if d]
            if not docs:
                return None
            if doc_id:
                targets = [doc_id]
            else:
                targets = [str(d.get("document_id")) for d in docs[:3]
                           if d.get("document_id")]
            scored: list[tuple[int, str, dict, str]] = []
            for tid in targets:
                try:
                    chunks = store.load_chunks(tid)
                except Exception as exc:
                    log.warning("Failed to load chunks for %s: %s", tid, exc)
                    continue
                meta = store.get_document(tid) or {}
                fname = str(meta.get("filename") or meta.get("original_filename")
                            or tid)
                for ch in chunks:
                    text = getattr(ch, "page_content", "") or ""
                    score = _lexical_score(question, text)
                    if score > 0:
                        scored.append((score, text, getattr(ch, "metadata", {}) or {},
                                       fname))
            if not scored:
                return None
            scored.sort(key=lambda t: t[0], reverse=True)
            top = scored[:3]
            parts, sources = [], []
            for score, text, meta, fname in top:
                page = meta.get("page")
                label = f"{fname}" + (f" p.{page}" if page is not None else "")
                snippet = " ".join(text.split())[:400]
                parts.append(f"From {label}: {snippet}")
                sources.append({"label": label, "page": page})
            lead = ("Here's what your document says on that:"
                    if len(top) == 1 or top[0][0] >= top[1][0] * 2
                    else "The closest passages I found:")
            answer = (lead + " " + " ".join(parts)
                      + " Quoted from the document text, not paraphrased beyond trimming.")
            return answer, sources, []
        except Exception as exc:
            log.info("Copilot document retrieval failed (%s).", exc)
            return None

    def _answer_general(self, question: str, intent: str) -> CopilotAnswer:
        """Definitional/business-concept questions: general LLM knowledge, no SQL."""
        try:
            prompt = (
                "You are ARIA, a business analyst. Give a concise (2-3 sentence) "
                "plain-English explanation of the business concept below. No SQL, "
                "no statistics, no mention of tables or models.\n\n"
                f"QUESTION: {question.strip()}"
            )
            text = self.llm.complete("story", prompt, temperature=0.2,
                                     num_predict=180)
            text = strip_text(text)
            if text and len(text) >= 3:
                self._remember(question, {"tables": [], "intent": intent},
                               text, None, intent)
                return CopilotAnswer(ok=True, answer=text,
                                     provider=self.provider_name, intent=intent)
        except Exception as exc:
            log.info("Copilot general-answer LLM failed (%s).", exc)
        answer = ("That's a general business question I can explain, but I couldn't "
                  "reach the language model just now. Please try again in a moment.")
        return CopilotAnswer(ok=False, answer=answer,
                             provider=self.provider_name, intent=intent)

    def _grounded_view(self, question: str, ctx):
        """Wrap retrieved context in the bounded grounded view (n.txt §5).

        Never raises; falls back to the raw retriever context so SQL
        generation always has a usable prompt and never the whole DB.
        """
        try:
            grounded = build_grounded_db_context(
                question, ctx, self.mapping,
                db_uri=self.db_uri, dialect=self.dialect,
            )
            return GroundedCtxView(ctx, grounded), grounded
        except Exception as exc:
            log.info("Grounded context unavailable (%s); raw context.", exc)
            return ctx, None

    def _join_lines_for(self, tables) -> list[str]:
        """Validated join edges touching the retrieved tables, prompt-ready.

        Generic (mapping-driven, no table names): used to steer the repair
        when SQL runs but returns zero rows on a question that should have
        data — usually a wrong join the DB accepted (e.g. order_id on a
        product key yields an empty set, not an error).
        """
        lines: list[str] = []
        try:
            wanted = {str(t).lower() for t in (tables or [])}
            _all = []
            for e in ((self.mapping or {}).get("relationship_edges") or []):
                s, sc = e.get("source_table"), e.get("source_column")
                t, tc = e.get("target_table"), e.get("target_column")
                if not s or not t:
                    continue
                _all.append((str(s), str(sc), str(t), str(tc)))
            # 1-hop expansion: the bridge table joining the wanted tables is
            # often NOT itself retrieved (orderdetail bridges header/product),
            # so include edges touching wanted tables' neighbors too.
            _neighbors = set(wanted)
            for s, _sc, t, _tc in _all:
                if s.lower() in wanted:
                    _neighbors.add(t.lower())
                if t.lower() in wanted:
                    _neighbors.add(s.lower())
            def _rank(e):
                s, _sc, t, _tc = e
                _both = (s.lower() in wanted and t.lower() in wanted)
                _one = (s.lower() in wanted or t.lower() in wanted)
                _nbor = (s.lower() in _neighbors or t.lower() in _neighbors)
                return (0 if _both else (1 if _one else (2 if _nbor else 3)))
            # Composite FKs share one table pair across edges: merge into a
            # single ANDed line so the model emits every condition (a
            # partial join fans out and silently inflates aggregates).
            _pairs: dict = {}
            for s, sc, t, tc in sorted(_all, key=_rank):
                if _rank((s, sc, t, tc)) > 2:
                    continue
                _pk = (str(s).lower(), str(t).lower())
                _rk = (str(t).lower(), str(s).lower())
                _key = _pk if _pk in _pairs or _rk not in _pairs else _rk
                _pairs.setdefault(_key, []).append((s, sc, t, tc))
                if len(_pairs) >= 60:
                    break
            seen = set()
            for (_ps, _pt), _conds in _pairs.items():
                _parts = []
                for s, sc, t, tc in _conds:
                    key = (str(s).lower(), str(sc).lower(), str(t).lower(), str(tc).lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    _parts.append(f"{s}.{sc} = {t}.{tc}")
                if _parts:
                    lines.append(" AND ".join(_parts))
            # Completeness tail: ranked lines above cover the question's
            # neighborhood; the bridge BFS + join check need EVERYTHING.
            # Appended last so relevance ranking is preserved.
            for s, sc, t, tc in _all:
                try:
                    key = (str(s).lower(), str(sc).lower(), str(t).lower(), str(tc).lower())
                except Exception:
                    continue
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"{s}.{sc} = {t}.{tc}")
        except Exception:
            pass
        return lines

    def _repair_empty_result(self, resolved, view, ctx, result):
        """One bounded re-generation when SQL runs but returns zero rows.

        An empty set on an aggregate/ranking question almost always means a
        wrong join (not wrong filters): steer the retry at the validated
        relationships instead of letting the model repeat the same join.
        Returns the new SqlResult (or the original when no retry applies).
        """
        try:
            if not result or not result.ok or (result.rows or []):
                return result
            rels = self._join_lines_for(getattr(ctx, "tables", None) or [])
            if not rels:
                return result
            from .sql_engine import build_sql_prompt
            prompt = build_sql_prompt(
                resolved, view, self.dialect,
                feedback=("The previous SQL ran successfully but returned ZERO "
                          "rows. The JOIN is almost certainly wrong (joining "
                          "unrelated keys yields an empty set, not an error). "
                          "Join ONLY through these validated relationships, "
                          "adding the bridge table when the two sides live in "
                          "different tables:\n" + "\n".join(f"- {r}" for r in rels)))
            raw = self.llm.complete("sql", prompt, temperature=0.2, num_predict=300)
            from .sql_engine import clean_sql
            sql = clean_sql(raw)
            retry = self.sql_engine.execute(sql, self.index.tables)
            retry.attempts = (result.attempts or 1) + 1
            if retry.ok and (retry.rows or []):
                log.info("Copilot empty-result repair succeeded with validated join.")
                return retry
        except Exception as exc:
            log.info("Copilot empty-result repair skipped (%s).", exc)
        return result

    @staticmethod
    def _retrieved_columns(ctx, mapping: dict) -> list[str]:
        cols: list[str] = []
        try:
            for t in (ctx.tables or []):
                for bucket in ("measures", "dimensions", "dates", "keys"):
                    cols.extend((getattr(ctx, bucket, None) or {}).get(t, []) or [])
        except Exception as exc:
            log.warning("Failed to extract retrieved columns: %s", exc, exc_info=True)
        return cols

    def _verify_narrative(self, question: str, narrative: str, ctx,
                          result, rows: list[dict], intent: str,
                          fallback: str) -> tuple[str, dict]:
        """Answer-grounding check (n.txt §6): SQL authoritative, fail-closed.

        Returns (final_narrative, grounding_payload). On numeric mismatch the
        deterministic fallback (built from executed rows) replaces the LLM
        text so invented numbers never reach the UI.
        """
        try:
            check = check_answer_grounding(
                question=question, answer=narrative,
                retrieved_tables=list(ctx.tables or []),
                retrieved_columns=self._retrieved_columns(ctx, self.mapping),
                sql=result.sql if result else None,
                rows=rows, columns=(result.columns if result else []),
                intent=intent,
            )
        except Exception as exc:
            log.info("Grounding verification error (%s).", exc)
            check = {"verdict": "needs_review", "reason": "check_error",
                     "checks": {}, "quality": {}}
        if check.get("verdict") == "ungrounded":
            log.info("Copilot grounding failed (%s); deterministic fallback.",
                     check.get("reason"))
            try:
                final = check_answer_grounding(
                    question=question, answer=fallback,
                    retrieved_tables=list(ctx.tables or []),
                    retrieved_columns=self._retrieved_columns(ctx, self.mapping),
                    sql=result.sql if result else None,
                    rows=rows, columns=(result.columns if result else []),
                    intent=intent,
                )
            except Exception as exc:
                log.info("Second grounding check failed (%s); using first check.", exc)
                final = check
            payload = {**final, "repaired": True,
                       "original_verdict": check.get("verdict"),
                       "original_reason": check.get("reason")}
            return fallback, payload
        return narrative, {**check, "repaired": False}

    def _answer_data(self, question: str, resolved: str, intent: str) -> CopilotAnswer:
        """Classic lookup path: retrieve -> SQL -> validate -> execute -> narrate."""
        followup_ctx = (self.memory.last(dataset=self.dataset_fp) or {}).get("context") if resolved != question else None

        ctx = self.retriever.retrieve(resolved, followup_context=followup_ctx)
        if ctx.is_empty():
            return CopilotAnswer(
                ok=False,
                answer=("I couldn't find tables in the connected schema that relate "
                        "to that question. Try naming a subject, measure or time range."),
                provider=self.provider_name,
                intent=intent,
            )

        view, grounded = self._grounded_view(resolved, ctx)
        result = self.sql_engine.generate_and_run(
            resolved, view, self.llm, tables=self.index.tables, model_role="sql",
            relationships=self._join_lines_for(getattr(ctx, "tables", None) or []),
        )
        result = self._repair_empty_result(resolved, view, ctx, result)

        if not result.ok and resolved != question:
            # Follow-up repair (t.txt §13/26): re-run the previous turn's
            # validated SQL and analyze it through the new intent.
            result = self._rerun_previous_sql() or result
            if result.ok:
                ctx = self._previous_context(ctx)

        if not result.ok:
            answer = self._explain_failure(result)
            return CopilotAnswer(
                ok=False, answer=answer, sql=result.sql, error=result.error,
                tables_used=ctx.tables, retrieved=ctx.to_state(),
                attempts=result.attempts, provider=self.provider_name,
                intent=intent,
            )

        rows = _sanitize_rows(result.rows)
        narrative = self._narrate(question, result, rows)
        chart = infer_chart(result.columns, rows, question)
        narrative, grounding = self._verify_narrative(
            question, narrative, ctx, result, rows, intent,
            fallback=_summarize_rows(result.columns, rows),
        )

        self._remember(question, ctx.to_state(), narrative, result.sql, intent)

        return CopilotAnswer(
            ok=True,
            answer=narrative,
            sql=result.sql,
            rows=rows,
            columns=result.columns,
            chart=chart,
            tables_used=ctx.tables,
            retrieved=ctx.to_state(),
            attempts=result.attempts,
            provider=self.provider_name,
            intent=intent,
            grounding=grounding,
        )

    def _answer_analytical(self, question: str, resolved: str, intent: str) -> CopilotAnswer:
        """Multi-step path: SQL evidence -> analysis -> grounded reasoning."""
        followup_ctx = (self.memory.last(dataset=self.dataset_fp) or {}).get("context") if resolved != question else None

        ctx = self.retriever.retrieve(resolved, followup_context=followup_ctx)
        if ctx.is_empty():
            return CopilotAnswer(
                ok=False,
                answer=("I couldn't find tables in the connected schema that relate "
                        "to that question. Try naming a subject, measure or time range."),
                provider=self.provider_name,
                intent=intent,
            )

        view, grounded = self._grounded_view(resolved, ctx)
        result = self.sql_engine.generate_and_run(
            resolved, view, self.llm, tables=self.index.tables, model_role="sql",
        )
        result = self._repair_empty_result(resolved, view, ctx, result)

        if not result.ok and resolved != question:
            result = self._rerun_previous_sql() or result
            if result.ok:
                ctx = self._previous_context(ctx)

        if not result.ok and intent == FORECAST:
            # Forecast history is fully determined (measure × date): build it
            # directly instead of failing on LLM generation flakiness.
            det = self._deterministic_history(ctx)
            if det is not None and det.ok and det.rows:
                result = det

        if not result.ok:
            answer = self._explain_failure(result)
            return CopilotAnswer(
                ok=False, answer=answer, sql=result.sql, error=result.error,
                tables_used=ctx.tables, retrieved=ctx.to_state(),
                attempts=result.attempts, provider=self.provider_name,
                intent=intent,
            )

        rows = _sanitize_rows(result.rows)
        analysis = _analyze_rows(result.columns, rows, intent)
        if intent == FORECAST:
            narrative = self._narrate_forecast(question, rows, analysis)
            if narrative is None:
                narrative = self._narrate_analytical(question, result, rows,
                                                     analysis, intent)
        else:
            narrative = self._narrate_analytical(question, result, rows, analysis, intent)
        chart = infer_chart(result.columns, rows, question)
        narrative, grounding = self._verify_narrative(
            question, narrative, ctx, result, rows, intent,
            fallback=_fallback_analytical(question, analysis, intent),
        )

        self._remember(question, ctx.to_state(), narrative, result.sql, intent)

        return CopilotAnswer(
            ok=True,
            answer=narrative,
            sql=result.sql,
            rows=rows,
            columns=result.columns,
            chart=chart,
            tables_used=ctx.tables,
            retrieved=ctx.to_state(),
            attempts=result.attempts,
            provider=self.provider_name,
            intent=intent,
            grounding=grounding,
        )

    # -- follow-up repair --------------------------------------------------

    def _rerun_previous_sql(self):
        """Re-execute the previous turn's validated SQL (read-only).

        Returns the SqlResult, or None when there is no previous SQL.
        Only same-dataset turns are eligible: after a dataset switch
        there is no previous SQL to re-run. Never raises; validation +
        execution are both bounded.
        """
        try:
            prev = self.memory.last(dataset=self.dataset_fp) or {}
            sql = prev.get("sql")
            if not sql:
                return None
            return self.sql_engine.execute(sql, self.index.tables)
        except Exception as exc:
            log.info("Copilot previous-SQL re-run failed (%s).", exc)
            return None

    def _previous_context(self, ctx):
        """Prefer the previous turn's retrieved context for continuity.

        Only same-dataset turns are eligible, so a follow-up after a
        dataset switch never inherits stale tables.
        """
        try:
            prev = self.memory.last(dataset=self.dataset_fp) or {}
            saved = (prev.get("context") or {})
            tables = saved.get("tables") or []
            if tables:
                ctx.tables = list(tables)
            return ctx
        except Exception as exc:
            log.warning("Failed to restore previous context: %s", exc)
            return ctx

    # -- deterministic forecast history (no LLM needed) --------------------

    @staticmethod
    def _history_query(ctx, dialect: str) -> str | None:
        """Build a monthly SUM aggregation over the retrieved context.

        Used when LLM SQL generation fails for forecast questions: the needed
        history query is fully determined (one measure × one date), so there
        is no reason to fail the whole forecast on generation flakiness.
        """
        def _q(name: str) -> str:
            safe = str(name or "").replace("`", "``").replace('"', '""')
            return f"`{safe}`" if dialect == "mysql" else f'"{safe}"'

        def _id_like(name: str) -> bool:
            low = str(name or "").lower().strip()
            return low.endswith("_id") or low.endswith("_key") or low in ("id", "key", "code")

        for table in (ctx.tables or []):
            measures = [m for m in (ctx.measures.get(table) or []) if not _id_like(m)]
            dates = list(ctx.dates.get(table) or [])
            if not measures or not dates:
                continue
            m, d = measures[0], dates[0]
            if dialect == "mysql":
                month = f"DATE_FORMAT({_q(d)}, '%Y-%m')"
            elif dialect == "sqlite":
                month = f"strftime('%Y-%m', {_q(d)})"
            else:
                month = f"date_trunc('month', {_q(d)})"
            return (
                f"SELECT {month} AS month, SUM({_q(m)}) AS total "
                f"FROM {_q(table)} GROUP BY 1 ORDER BY 1 LIMIT 120"
            )
        return None

    def _deterministic_history(self, ctx):
        """Execute a deterministic monthly history query for forecasts.

        Returns the SqlResult, or None when no measure×date pair exists.
        Never raises.
        """
        try:
            sql = self._history_query(ctx, self.dialect)
            if not sql:
                return None
            return self.sql_engine.execute(sql, self.index.tables)
        except Exception as exc:
            log.info("Copilot deterministic history failed (%s).", exc)
            return None

    # -- response generation (provider-independent) ----------------------

    def _narrate(self, question: str, result, rows: list[dict]) -> str:
        fallback = _summarize_rows(result.columns, rows)
        if not rows:
            return "The query ran successfully but returned no rows for that question."
        preview = rows[:8]
        try:
            prompt = (
                "You are ARIA Copilot, a business analyst explaining data in plain business English. "
                "Answer the user's business question in 2-4 concise sentences using ONLY the query result below. "
                "Start with the headline finding, then one supporting detail and, when useful, one suggested action. "
                "Round numbers to human scale (42.8K, 18%, ~12.5K). "
                "Never mention SQL, tables, columns, schemas, models, R-squared, RMSE, p-values, coefficients, or embeddings. "
                "Use honest uncertainty words (suggests, appears to, may be) and never claim causation from correlation.\n\n"
                f"QUESTION: {question}\n"
                f"RESULT COLUMNS: {', '.join(result.columns)}\n"
                f"RESULT ROWS ({len(rows)} total, showing {len(preview)}):\n"
                f"{json.dumps(preview, default=str)[:2000]}\n"
            )
            text = self.llm.complete("story", prompt, temperature=0.2,
                                     num_predict=200)
            text = strip_text(text)
            if text and len(text) >= 3:
                return text
            log.info("Copilot narration LLM empty (LLM_EMPTY); using deterministic answer.")
        except Exception as exc:
            log.info("Copilot narration LLM failed (%s); using deterministic answer.", exc)
        return (f"ARIA identified the following data pattern, but the AI explanation "
                f"is currently unavailable. {fallback}")

    # -- analytical reasoning over SQL evidence (t.txt §§24, 27) ---------

    def _narrate_analytical(self, question: str, result, rows: list[dict],
                            analysis: dict, intent: str) -> str:
        """Ground the LLM on SQL evidence + deterministic analysis.

        The analysis (not the raw rows alone) is passed in, with intent-specific
        instructions: hypotheses labeled as possibilities, forecasts directional
        with uncertainty, recommendations actionable. Falls back to the
        deterministic builder when the LLM is unavailable.
        """
        role_hint = {
            HYPOTHESIS: ("Propose 1-2 POSSIBLE explanations for the pattern, each "
                         "starting with 'Possible explanation:'. Distinguish observed "
                         "facts from hypotheses; never claim causation."),
            FORECAST: ("Give a directional outlook ('likely to continue/increase/"
                       "decline') with honest uncertainty ('if the recent pattern "
                       "continues', 'actual results may vary'). Never guarantee."),
            ANOMALY: ("Point out what looks unusual and why it stands out. If "
                      "nothing stands out, say so plainly instead of inventing."),
            RECOMMENDATION: ("End with one concrete suggested action tied to the "
                             "finding. Never invent impact numbers."),
            MIXED: ("First give a possible explanation (labeled as such), then one "
                    "concrete suggested action. Keep each to one sentence."),
            HEALTH: ("Summarize the 2-3 most important findings, then what to "
                     "watch next. Plain business language throughout."),
        }.get(intent, "Answer in 2-4 concise business sentences.")
        findings = "; ".join(analysis.get("findings", [])[:6]) or "no strong pattern"
        preview = rows[:8]
        try:
            prompt = (
                "You are ARIA Copilot, a business analyst explaining data in plain business English. "
                f"{role_hint} "
                "Round numbers to human scale (42.8K, 18%, ~12.5K). "
                "Never mention SQL, tables, columns, schemas, models, R-squared, RMSE, p-values, coefficients, or embeddings. "
                "Use honest uncertainty words (suggests, appears to, may be) and never claim causation from correlation.\n\n"
                f"QUESTION: {question}\n"
                f"DATA SUMMARY: {findings}\n"
                f"RESULT COLUMNS: {', '.join(result.columns)}\n"
                f"RESULT ROWS ({len(rows)} total, showing {len(preview)}):\n"
                f"{json.dumps(preview, default=str)[:2000]}\n"
            )
            text = self.llm.complete("story", prompt, temperature=0.2,
                                     num_predict=220)
            text = strip_text(text)
            if text and len(text) >= 3:
                return text
            log.info("Copilot analytical narration LLM empty (LLM_EMPTY).")
        except Exception as exc:
            log.info("Copilot analytical narration LLM failed (%s).", exc)
        return (f"ARIA identified the following data pattern, but the AI explanation "
                f"is currently unavailable. {_fallback_analytical(question, analysis, intent)}")

    # -- forecasting via the prediction engine (t.txt §27) ----------------

    def _narrate_forecast(self, question: str, rows: list[dict],
                          analysis: dict) -> str | None:
        """Run SQL evidence through PredictionAgent (walk-forward validated).

        Returns a grounded forecast narrative, or None when the engine is
        unavailable/inapplicable so the caller falls back to the directional
        trend outlook. Never raises.
        """
        try:
            from prediction.agent import PredictionAgent
        except Exception as exc:
            log.info("Copilot forecast engine unavailable (%s); trend fallback.", exc)
            return f"Forecast unavailable: {exc}"
        try:
            payload = PredictionAgent().forecast(
                {"data": rows, "user_goal": question})
        except Exception as exc:
            log.info("Copilot forecast run failed (%s); trend fallback.", exc)
            return "The forecast could not run. Check the selected measure and available history, then retry."
        if not isinstance(payload, dict) or not payload.get("valid"):
            return "Forecast returned invalid or empty result."
        points = payload.get("points") or []
        if not points:
            return "Forecast returned no data points."
        try:
            return self._forecast_narrative(question, payload, analysis)
        except Exception as exc:
            log.info("Copilot forecast narration failed (%s); trend fallback.", exc)
            return "The forecast explanation is unavailable. Please retry."

    def _forecast_narrative(self, question: str, payload: dict,
                            analysis: dict) -> str:
        """Business outlook grounded on validated engine output."""
        col = payload.get("column") or "value"
        pts = payload.get("points") or []
        horizon = payload.get("horizon") or len(pts)
        proj = payload.get("projected_pct_change")
        acc = payload.get("accuracy_pct")
        rel = str(payload.get("reliability") or "").lower()
        conf = ("High" if rel in ("excellent", "good", "high") else "Moderate"
                if rel in ("fair", "moderate") else "Low")
        first, last = pts[0], pts[-1]
        trend = ""
        if isinstance(proj, (int, float)):
            trend = (f"about {proj:+.0f}% vs the latest actual" if abs(proj) > 1
                     else "roughly flat vs the latest actual")
        det = (f"{col} is projected {trend} over the next {horizon} "
               f"{payload.get('period') or 'periods'} "
               f"({_biz_num(first.get('value'))} → {_biz_num(last.get('value'))}). "
               f"Reliability: {conf}"
               + (f" ({acc:.0f}% backtest accuracy)" if acc is not None else "")
               + ". Actual results may vary — treat distant points as directional.")
        try:
            prompt = (
                "You are ARIA Copilot, a business analyst explaining data in plain business English. "
                "Rewrite the outlook below in 2-3 concise sentences for a business owner: "
                "what is likely, how reliable it is, and one preparation suggestion. "
                "Round numbers to human scale. Never mention models, MAPE, R-squared, RMSE, "
                "p-values, tables, columns, or embeddings. Never guarantee the outcome.\n\n"
                f"QUESTION: {question}\n"
                f"VALIDATED OUTLOOK: {det}\n"
            )
            text = self.llm.complete("story", prompt, temperature=0.2,
                                     num_predict=200)
            text = strip_text(text)
            if text and len(text) >= 3:
                return text
        except Exception as exc:
            log.info("Copilot forecast LLM polish failed (%s).", exc)
        return det

    def _guided_next_step(self) -> str:
        """Data-relevant next step naming real tables (generic, no fixed data)."""
        try:
            tables = sorted((self.mapping or {}).get("tables", {}) or {})
            if tables:
                named = ", ".join(tables[:3])
                return (f" Try asking about {named} — for example a total, "
                        f"a breakdown, or how it changed over time.")
        except Exception:
            pass
        return (" Try asking about a specific table, measure, or time period "
                "— for example a total, a breakdown, or a trend over time.")

    def _explain_failure(self, result) -> str:
        err = (result.error or "").lower()
        if "rate limit" in err or "429" in err:
            return ("The language model is rate-limited right now. Please retry in a "
                    "moment or switch the LLM provider in Settings.")
        if "timed out" in err or "timeout" in err or "deadline" in err:
            return ("The analysis is taking longer than usual on this machine. "
                    "Try a narrower question (one metric, one period), then retry — "
                    "your data was not changed.")
        if "connect" in err or "refused" in err:
            return ("I couldn't reach the language model backend. Check that Ollama is "
                    "running (Local mode) or that the API key is valid (Cloud mode).")
        if "unknown table" in err or "unknown column" in err:
            return ("The generated query referenced tables/columns that don't exist in "
                    "this schema. Ask with a clearer subject or measure."
                    + self._guided_next_step())
        if "only read-only" in err:
            return "The generated query was rejected because it was not read-only."
        return ("I couldn't answer that question from the connected data."
                + self._guided_next_step())


def _lexical_score(question: str, text: str) -> int:
    """Token-overlap score for document chunks (stopword-filtered)."""
    stop = {"what", "does", "the", "this", "that", "about", "say", "says",
            "and", "for", "with", "from", "there", "their", "have", "has",
            "was", "were", "are", "our", "your", "how", "why", "when",
            "which", "who", "whom", "a", "an", "of", "to", "in", "on", "is",
            "it", "its", "by", "do", "did", "any", "me", "tell"}
    qtok = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in stop}
    if not qtok:
        return 0
    ttok = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(qtok & ttok)


def _biz_num(v) -> str:
    """Human-scale number for deterministic analysis sentences."""
    try:
        n = float(str(v).replace(",", "").replace("$", "").replace("%", ""))
    except (TypeError, ValueError):
        return str(v)
    a = abs(n)
    if a >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if a >= 10_000:
        return f"{n / 1_000:.1f}K"
    if a >= 100:
        return f"{round(n):,}"
    if a >= 1:
        return f"{round(n, 1)}"
    return f"{round(n, 2)}"


def _analyze_rows(columns: list[str], rows: list[dict], intent: str) -> dict:
    """Deterministic analysis over SQL evidence (no LLM).

    Finds: row count, top contributor + share, first-half vs second-half
    momentum, and whether evidence is thin. The LLM reasons ON TOP of this;
    it never sees raw rows alone for analytical intents.
    """
    out: dict = {"findings": [], "thin": False, "top": None,
                 "momentum_pct": None, "row_count": len(rows)}
    if not rows or not columns:
        out["thin"] = True
        out["findings"].append("the query returned no rows")
        return out
    sample = rows[0]
    numeric_cols = [c for c in columns if _numeric(sample.get(c))]
    label_cols = [c for c in columns if c not in numeric_cols]
    out["findings"].append(f"{len(rows)} result row{'s' if len(rows) != 1 else ''}")
    if label_cols and numeric_cols:
        label, measure = label_cols[0], numeric_cols[-1]
        try:
            vals = [(r.get(label), float(str(r.get(measure)).replace(",", "")))
                    for r in rows]
            vals = [(k, v) for k, v in vals if k is not None]
        except (TypeError, ValueError):
            vals = []
        if vals:
            top = max(vals, key=lambda kv: kv[1])
            total = sum(v for _, v in vals)
            share = (top[1] / total * 100) if total else 0
            out["top"] = {"label": str(top[0]), "value": top[1], "share": share}
            out["findings"].append(
                f"{top[0]} leads {measure} at {_biz_num(top[1])}"
                + (f" ({share:.0f}% of the total)" if len(vals) > 1 else ""))
    if numeric_cols and len(rows) >= 4:
        m = numeric_cols[-1]
        # Momentum is only meaningful on a time axis: require a date-like
        # column and sort by it first. On ranked/grouped return order the
        # "second half vs first half" comparison is just sort order, not a
        # trend — report None instead of a fabricated direction.
        date_keys = [k for k in (rows[0].keys() if rows else []) if _datetime_name(k)]
        if not date_keys:
            out["momentum_pct"] = None
        else:
            try:
                ordered = sorted(rows, key=lambda r: str(r.get(date_keys[0])))
                seq = [float(str(r.get(m)).replace(",", "")) for r in ordered]
                half = len(seq) // 2
                early = sum(seq[:half]) / max(1, half)
                late = sum(seq[half:]) / max(1, len(seq) - half)
                if early:
                    pct = (late - early) / abs(early) * 100
                    out["momentum_pct"] = pct
                    direction = "up" if pct > 1 else "down" if pct < -1 else "flat"
                    out["findings"].append(
                        f"recent {m} trends {direction} ({pct:+.0f}% second half vs first half)")
            except (TypeError, ValueError, ZeroDivisionError):
                out["momentum_pct"] = None
    if len(rows) < 2:
        out["thin"] = True
        out["findings"].append("limited evidence — a single data point")
    return out


def _fallback_analytical(question: str, analysis: dict, intent: str) -> str:
    """Deterministic business answer when the LLM is unavailable."""
    findings = "; ".join(analysis.get("findings", [])) or "the available data"
    top = analysis.get("top") or {}
    mom = analysis.get("momentum_pct")
    if intent == FORECAST:
        if mom is None:
            return ("There isn't enough history in the connected data to project "
                    "a direction reliably. Connect more periods and try again.")
        direction = ("increasing" if mom > 1 else "declining" if mom < -1
                     else "holding steady")
        return (f"Based on {findings}, the recent pattern points {direction} "
                f"if current conditions continue — actual results may vary, "
                f"so treat this as directional.")
    if intent == HYPOTHESIS:
        why = (f" A possible explanation is that {top.get('label')} — the largest "
               f"contributor — is driving the pattern." if top else "")
        return (f"Based on {findings}.{why} This is a possible explanation, not "
                f"proof of cause; the data shows association only.")
    if intent == ANOMALY:
        if top and top.get("share", 0) >= 60:
            return (f"One thing stands out: {top.get('label')} accounts for "
                    f"{top.get('share'):.0f}% of the total across {findings}. "
                    f"Worth a closer look.")
        if mom is not None and abs(mom) >= 20:
            return (f"Unusual movement: the second half differs by {mom:+.0f}% "
                    f"from the first half ({findings}). Worth investigating.")
        return (f"I reviewed {findings} and nothing stands out as clearly unusual. "
                f"No strong anomaly signal in this slice.")
    if intent in (RECOMMENDATION, HEALTH, MIXED):
        action = (f" Consider focusing on {top.get('label')}, the largest "
                  f"contributor." if top else " Consider digging into the top segment.")
        if intent == MIXED:
            return (f"Based on {findings}, a possible explanation is concentration "
                    f"in the top contributor.{action}")
        if intent == HEALTH:
            return (f"Overall: {findings}.{action} Watch whether the recent "
                    f"momentum holds next period.")
        return (f"Based on {findings}.{action}")
    return f"Based on {findings}."


def strip_text(raw: str) -> str:
    """Light cleanup for free-text LLM output (no code fences)."""
    text = (raw or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _sanitize_rows(rows: list[dict]) -> list[dict]:
    """Make rows JSON-safe (datetimes/Decimals -> str/float)."""
    out = []
    for r in rows or []:
        clean = {}
        for k, v in r.items():
            if v is None or isinstance(v, (str, int, float, bool)):
                clean[k] = v
            else:
                clean[k] = str(v)
        out.append(clean)
    return out


def create_pipeline(llm, schema_mapping: dict, db_uri: str, dialect: str = "postgresql",
                    user_id="anon", embedder=None, provider_name: str | None = None,
                    document_id: str | None = None, linked: dict | None = None) -> CopilotPipeline:
    """Factory used by the API route. The LLM is the existing provider object."""
    return CopilotPipeline(
        llm=llm,
        schema_mapping=schema_mapping,
        db_uri=db_uri,
        dialect=dialect,
        user_id=user_id,
        embedder=embedder,
        provider_name=provider_name or getattr(llm, "provider", None),
        document_id=document_id,
        linked=linked,
    )