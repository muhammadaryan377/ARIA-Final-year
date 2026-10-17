"""Structured-database RAG pipeline stages for ARIA Copilot (n.txt §5).

Expands Copilot RAG beyond ``question → embeddings → chunks → LLM`` into the
full pipeline::

    question → intent classification → knowledge-source decision
        (db / document / hybrid / general)
      → retrieve schema + metadata + docs (bounded: never the whole DB)
      → retrieve database information (sample values, executed rows)
      → grounded context → LLM → answer-grounding verification

This module owns the *retrieval/context* half. Verification lives in
``copilot.answer_grounding``. Everything here is deterministic and
provider-independent; the LLM only ever sees the bounded grounded prompt.

Reuses:
- ``copilot.intent.classify`` for the source decision,
- ``copilot.retriever.SchemaRetriever`` ordering (plus an optional
  ``insight_rag.reranker.LocalCrossEncoderReranker`` pass, fail-open),
- ``insight_rag.enterprise_quality`` signal philosophy (bounded, auditable).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

MAX_GROUNDED_TABLES = 4
MAX_PROMPT_CHARS = 4000
MAX_SAMPLE_COLUMNS_PER_TABLE = 4
MAX_SAMPLE_VALUES = 3
MAX_PATTERNS = 2

_DB_WORDS = re.compile(
    r"\b(total|sales|revenue|orders?|customers?|products?|territor|region|"
    r"category|categories|trend|average|top|bottom|best|monthly|yearly|"
    r"count|sum|by|per|each|compare|share|trend|growth|amount|profit|cost)\b",
    re.IGNORECASE,
)


def decide_knowledge_source(question: str) -> dict:
    """Map a question to the knowledge source it needs (deterministic).

    Reuses ``copilot.intent.classify`` so routing stays identical to the
    pipeline's tool plan. Returns ``{"source": ..., "intent": ...}`` where
    source is one of ``db`` | ``document`` | ``hybrid`` | ``general`` |
    ``schema``.
    """
    from .intent import COMBINED, DOCUMENT, GENERAL, SCHEMA, classify

    route = classify(question or "")
    if route.intent == COMBINED:
        source = "hybrid"
    elif route.intent == DOCUMENT:
        source = "document"
    elif route.intent == GENERAL:
        source = "general"
    elif route.intent == SCHEMA:
        source = "schema"
    else:
        source = "db"
    return {"source": source, "intent": route.intent,
            "tools": list(route.tools), "confidence": route.confidence}


@dataclass
class GroundedDbContext:
    """Bounded structured context for one question (never a whole-DB dump)."""

    tables: list[str] = field(default_factory=list)
    intent: str = "data"
    source: str = "db"
    blocks: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def to_prompt(self, max_tables: int = MAX_GROUNDED_TABLES) -> str:
        text = "\n".join(self.blocks[: max(1, max_tables * 2)])
        return text[:MAX_PROMPT_CHARS]

    def is_empty(self) -> bool:
        return not self.tables


def _table_info(mapping: dict, table: str) -> dict:
    tables = (mapping or {}).get("tables", {}) or {}
    info = tables.get(table, {})
    return info if isinstance(info, dict) else {}


def _columns_of(info: dict) -> list[dict]:
    cols = info.get("columns", [])
    out: list[dict] = []
    if isinstance(cols, dict):
        for name, meta in cols.items():
            dtype = meta.get("data_type", "") if isinstance(meta, dict) else ""
            desc = meta.get("description", "") if isinstance(meta, dict) else ""
            out.append({"column": str(name), "data_type": str(dtype),
                        "description": str(desc or "")})
    elif isinstance(cols, list):
        for c in cols:
            if isinstance(c, dict):
                out.append({
                    "column": str(c.get("column") or c.get("name") or ""),
                    "data_type": str(c.get("data_type", "")),
                    "description": str(c.get("description") or c.get("business_meaning") or ""),
                })
            elif isinstance(c, str):
                out.append({"column": c, "data_type": "", "description": ""})
    return [c for c in out if c["column"]]


def _describe_table(table: str, info: dict) -> str | None:
    for key in ("description", "business_meaning", "business_description",
                "table_description"):
        value = info.get(key)
        if value:
            return str(value)[:300]
    return None


def _metric_defs(mapping: dict, tables: list[str]) -> list[str]:
    defs: list[str] = []
    global_defs = (mapping or {}).get("metric_definitions") or (mapping or {}).get("metrics") or {}
    if isinstance(global_defs, dict):
        for name, desc in global_defs.items():
            text = f"{name}: {desc}"
            if any(t.lower() in text.lower() for t in tables) or len(tables) <= 2:
                defs.append(str(text)[:200])
    for table in tables:
        info = _table_info(mapping, table)
        for key in ("metric_definitions", "metrics"):
            local = info.get(key)
            if isinstance(local, dict):
                for name, desc in local.items():
                    defs.append(f"{table}.{name}: {desc}"[:200])
            elif isinstance(local, list):
                defs.extend(str(v)[:200] for v in local)
    return defs[:4]


def _validated_patterns(mapping: dict, tables: list[str],
                        intent_terms: list[str]) -> list[str]:
    patterns = (mapping or {}).get("validated_query_patterns") or []
    if not isinstance(patterns, list):
        return []
    scored: list[tuple[int, str]] = []
    selected = set(tables)
    for pat in patterns:
        if not isinstance(pat, dict):
            continue
        ptables = set(pat.get("tables", []) or [])
        text = f"{pat.get('name', '')} {pat.get('description', '')} {pat.get('sql', '')}"
        overlap = len(selected & ptables)
        term_hit = sum(1 for t in (intent_terms or [])
                       if t and t.lower() in text.lower())
        score = overlap * 2 + min(term_hit, 3)
        if overlap or (selected and not ptables):
            label = pat.get("name") or "pattern"
            sql = str(pat.get("sql", ""))[:300]
            scored.append((score, f"{label}: {sql}"))
    scored.sort(key=lambda kv: (-kv[0], kv[1]))
    return [s for _, s in scored[:MAX_PATTERNS]]


def _sample_values(db_uri: str | None, dialect: str, table: str,
                   columns: list[str]) -> dict[str, list[str]]:
    """Bounded DISTINCT sample values per column (fail-open, read-only)."""
    out: dict[str, list[str]] = {}
    if not db_uri or str(db_uri).startswith("document://"):
        return out
    try:
        from sqlalchemy import create_engine, text as _text

        quote = "`" if dialect == "mysql" else '"'
        _cargs = {} if "sqlite" in str(db_uri) else {"connect_timeout": 5}
        try:
            # Schema-qualified keys quote per segment ("sales"."employee").
            _tsegs = [s for s in str(table or "").split(".") if s]
            _qtable = (".".join(f"{quote}{s}{quote}" for s in _tsegs)
                       if len(_tsegs) > 1 else f"{quote}{table}{quote}")
        except Exception:
            _qtable = f"{quote}{table}{quote}"
        engine = create_engine(db_uri, pool_pre_ping=True,
                               connect_args=_cargs)
        try:
            with engine.connect() as conn:
                for col in columns[:MAX_SAMPLE_COLUMNS_PER_TABLE]:
                    try:
                        sql = (f"SELECT DISTINCT {quote}{col}{quote} FROM "
                               f"{_qtable} LIMIT {MAX_SAMPLE_VALUES}")
                        rows = conn.execute(_text(sql)).fetchmany(MAX_SAMPLE_VALUES)
                        vals = [str(r[0])[:24] for r in rows if r[0] is not None]
                        if vals:
                            out[col] = vals
                    except Exception:
                        continue
        finally:
            engine.dispose()
    except Exception as exc:
        log.info("Grounded sample values unavailable for %s (%s).", table, exc)
    return out


def _quote(name: str, dialect: str) -> str:
    # Schema-qualified keys quote per segment ("sales"."employee").
    try:
        segs = [s for s in str(name or "").split(".") if s]
        if len(segs) > 1:
            q = "`" if dialect == "mysql" else '"'
            return ".".join(f"{q}{s}{q}" for s in segs)
    except Exception:
        pass
    q = "`" if dialect == "mysql" else '"'
    return f"{q}{name}{q}"


def rerank_schema_tables(question: str, tables: list[str],
                         table_texts: dict[str, str]) -> list[str]:
    """Optional cross-encoder rerank of the selected tables (fail-open).

    Reuses ``insight_rag.reranker.LocalCrossEncoderReranker`` when explicitly
    opted in via ``ARIA_SCHEMA_RERANK=1`` (the model download is expensive, so
    retriever order is the default). Never raises.
    """
    if len(tables) <= 1:
        return list(tables)
    import os

    if os.getenv("ARIA_SCHEMA_RERANK", "0") != "1":
        return list(tables)
    try:
        from langchain_core.documents import Document

        from insight_rag.reranker import LocalCrossEncoderReranker

        docs = [Document(page_content=table_texts.get(t, t),
                         metadata={"table": t}) for t in tables]
        ranked = LocalCrossEncoderReranker.rerank(question, docs, top_k=len(docs))
        ordered = [d.metadata.get("table") for d in ranked if d.metadata.get("table") in tables]
        ordered += [t for t in tables if t not in ordered]
        return ordered
    except Exception as exc:
        log.info("Schema rerank unavailable (%s); keeping retriever order.", exc)
        return list(tables)


def build_grounded_db_context(
    question: str,
    retrieved_ctx,
    mapping: dict,
    *,
    db_uri: str | None = None,
    dialect: str = "postgresql",
    with_samples: bool = True,
) -> GroundedDbContext:
    """Build the bounded grounded context (n.txt §5, DB half).

    Includes, per selected table only: table/column descriptions, business
    meanings, data types, relationships, sample values, metric definitions and
    validated query patterns. Never dumps the whole schema.
    """
    decision = decide_knowledge_source(question)
    tables = list((getattr(retrieved_ctx, "tables", None) or [])[:MAX_GROUNDED_TABLES])
    intent_terms = list(getattr(retrieved_ctx, "intent_terms", None) or [])
    table_texts = {t: f"Table {t}" for t in tables}
    tables = rerank_schema_tables(question or "", tables, table_texts)

    blocks: list[str] = []
    provenance: dict = {"tables": list(tables), "intent": decision["intent"],
                        "source": decision["source"], "bounded": True,
                        "max_tables": MAX_GROUNDED_TABLES}
    sample_provenance: dict[str, list[str]] = {}

    for table in tables:
        info = _table_info(mapping, table)
        cols = _columns_of(info)
        col_types = {c["column"]: c["data_type"] for c in cols}
        lines = [f"TABLE {table}:"]
        desc = _describe_table(table, info)
        if desc:
            lines.append(f"  description: {desc}")
        pk = info.get("primary_key") or info.get("inferred_primary_key") or []
        if pk:
            lines.append(f"  primary key: {', '.join(pk)}")
        # Column roles come from the retriever typing (measure/dimension/date/key).
        measures = set((getattr(retrieved_ctx, "measures", None) or {}).get(table, []))
        dims = set((getattr(retrieved_ctx, "dimensions", None) or {}).get(table, []))
        dates = set((getattr(retrieved_ctx, "dates", None) or {}).get(table, []))
        keys = set((getattr(retrieved_ctx, "keys", None) or {}).get(table, []))
        col_bits = []
        for c in cols[:12]:
            name, dtype = c["column"], c["data_type"] or col_types.get(c["column"], "")
            role = ("measure" if name in measures else "date" if name in dates
                    else "key" if name in keys else "dimension" if name in dims else "")
            bit = f"{name} ({dtype or role or 'col'})".strip()
            if c["description"]:
                bit += f" — {c['description'][:80]}"
            col_bits.append(bit)
        lines.append("  columns: " + ("; ".join(col_bits) if col_bits else "(none)"))
        row_count = info.get("row_count")
        if row_count is not None:
            lines.append(f"  rows: {row_count}")
        if with_samples:
            sample_cols = ([c for c in cols if c["column"] in (measures | dims | dates)][:MAX_SAMPLE_COLUMNS_PER_TABLE]
                           or [c["column"] for c in cols[:MAX_SAMPLE_COLUMNS_PER_TABLE]])
            samples = _sample_values(db_uri, dialect, table, sample_cols)
            if samples:
                sample_provenance[table] = list(samples.keys())
                for col, vals in samples.items():
                    lines.append(f"  sample {col}: {', '.join(vals)}")
        blocks.append("\n".join(lines))

    rels = list((getattr(retrieved_ctx, "relationships", None) or []))[:8]
    if rels:
        rel_lines = [f"  {r.get('left_table')}.{_q(r.get('left_column'), dialect)} = "
                     f"{r.get('right_table')}.{_q(r.get('right_column'), dialect)}"
                     for r in rels]
        blocks.append("RELATIONSHIPS (join only via these):\n" + "\n".join(rel_lines))
        provenance["relationships"] = len(rels)
    else:
        # Single-table case (n.txt §1): relationships = none is valid.
        blocks.append("RELATIONSHIPS: none (single-table or unrelated slice — do not invent joins).")
        provenance["relationships"] = 0

    metrics = _metric_defs(mapping, tables)
    if metrics:
        blocks.append("METRIC DEFINITIONS:\n" + "\n".join(f"  - {m}" for m in metrics))
        provenance["metric_defs"] = len(metrics)

    patterns = _validated_patterns(mapping, tables, intent_terms)
    if patterns:
        blocks.append("VALIDATED QUERY PATTERNS (prefer these shapes):\n"
                      + "\n".join(f"  - {p}" for p in patterns))
        provenance["patterns"] = len(patterns)

    if sample_provenance:
        provenance["sampled_columns"] = sample_provenance
    return GroundedDbContext(tables=tables, intent=decision["intent"],
                             source=decision["source"], blocks=blocks,
                             provenance=provenance)


def _q(name: object, dialect: str) -> str:
    # Schema-qualified keys render per segment (sales.employee."col").
    try:
        segs = [s for s in str(name or "").split(".") if s]
        if len(segs) > 1:
            return ".".join(_quote(s, dialect) for s in segs)
    except Exception:
        pass
    return _quote(str(name or ""), dialect)


class GroundedCtxView:
    """Retriever-compatible view exposing the grounded prompt to SqlEngine.

    Delegates table/column typing to the underlying RetrievedContext while
    ``to_prompt()`` returns the enriched bounded context. Keeps the SQL
    generator on retrieved tables only (no whole-schema dump).
    """

    def __init__(self, retrieved_ctx, grounded: GroundedDbContext):
        self._ctx = retrieved_ctx
        self._grounded = grounded

    @property
    def tables(self):
        return self._grounded.tables or self._ctx.tables

    @property
    def measures(self):
        return getattr(self._ctx, "measures", {})

    @property
    def dimensions(self):
        return getattr(self._ctx, "dimensions", {})

    @property
    def dates(self):
        return getattr(self._ctx, "dates", {})

    @property
    def keys(self):
        return getattr(self._ctx, "keys", {})

    @property
    def relationships(self):
        return getattr(self._ctx, "relationships", [])

    @property
    def intent_terms(self):
        return getattr(self._ctx, "intent_terms", [])

    def is_empty(self) -> bool:
        return not self.tables

    def to_prompt(self, max_tables: int = MAX_GROUNDED_TABLES) -> str:
        base = self._ctx.to_prompt(max_tables=max_tables) if hasattr(self._ctx, "to_prompt") else ""
        extra = self._grounded.to_prompt(max_tables=max_tables)
        combined = (extra + ("\n" + base if base and base not in extra else ""))
        return combined[:MAX_PROMPT_CHARS]

    def to_state(self) -> dict:
        state = self._ctx.to_state() if hasattr(self._ctx, "to_state") else {"tables": self.tables}
        state["grounded"] = True
        return state
