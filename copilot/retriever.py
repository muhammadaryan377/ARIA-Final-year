"""Provider-independent schema retriever for Copilot RAG.

Combines lexical (always available) and semantic (embedding) scores, then
expands the selected tables through the schema's relationship graph so the
context handed to the LLM contains the tables AND the joins needed to answer
the question — never the whole schema dump.

All scoring is deterministic and schema-agnostic. No LLM is involved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .embeddings import Embedder, cosine
from .schema_index import SchemaIndex, SchemaDocument

log = logging.getLogger(__name__)

_REL_WORDS = {
    "compare", "comparison", "versus", "vs", "difference", "between", "among",
    "breakdown", "by", "per", "each", "trend", "over", "time", "monthly",
    "yearly", "top", "bottom", "highest", "lowest", "rank", "share",
    "percentage", "distribution", "average", "total", "sum", "count",
}


@dataclass
class RetrievedContext:
    """The bundle of schema knowledge selected for one question."""

    tables: list[str] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    measures: dict[str, list[str]] = field(default_factory=dict)
    dimensions: dict[str, list[str]] = field(default_factory=dict)
    dates: dict[str, list[str]] = field(default_factory=dict)
    keys: dict[str, list[str]] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    intent_terms: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.tables

    def to_prompt(self, max_tables: int = 6) -> str:
        """Render a compact DDL-like context: only the selected tables."""
        lines: list[str] = []
        for t in self.tables[:max_tables]:
            cols = []
            cols += [f"{c} (measure)" for c in self.measures.get(t, [])]
            cols += [f"{c} (dimension)" for c in self.dimensions.get(t, [])]
            cols += [f"{c} (date)" for c in self.dates.get(t, [])]
            cols += [f"{c} (key)" for c in self.keys.get(t, [])]
            lines.append(f"TABLE {t}: " + (", ".join(cols) if cols else "(no typed columns)"))
        if self.relationships:
            rel_lines = []
            for r in self.relationships[:20]:
                rel_lines.append(
                    f"  {r['left_table']}.{r['left_column']} = "
                    f"{r['right_table']}.{r['right_column']} [{r.get('trust', 'declared')}]"
                )
            lines.append("RELATIONSHIPS:\n" + "\n".join(rel_lines))
        return "\n".join(lines)

    def to_state(self) -> dict:
        """Compact context saved into conversation memory for follow-ups."""
        return {
            "tables": self.tables[:4],
            "measures": [c for t in self.tables for c in self.measures.get(t, [])][:4],
            "group_by": [c for t in self.tables for c in self.dimensions.get(t, [])][:4],
        }


class SchemaRetriever:
    """Hybrid retriever over a SchemaIndex. Deterministic and offline-capable."""

    def __init__(self, index: SchemaIndex, embedder: Embedder | None = None,
                 semantic_weight: float = 0.6):
        self.index = index
        self.embedder = embedder
        self.semantic_weight = float(semantic_weight)
        self._doc_embeddings: list[list[float]] | None = None
        self._doc_ids: list[str] = []

    # -- embedding cache -------------------------------------------------

    def _ensure_doc_embeddings(self) -> None:
        if self._doc_embeddings is not None or self.embedder is None:
            return
        docs = self.index.documents
        self._doc_ids = [d.doc_id for d in docs]
        try:
            self._doc_embeddings = self.embedder.embed([d.text for d in docs])
        except Exception as exc:
            log.info("Embedding schema docs failed (%s); lexical-only retrieval.", exc)
            self._doc_embeddings = None

    # -- scoring ---------------------------------------------------------

    @staticmethod
    def _intent_terms(query: str) -> list[str]:
        from .schema_index import SchemaIndex as _SI
        return sorted(t for t in _SI._tokens(query) if t not in _REL_WORDS)

    def _semantic_scores(self, query: str) -> dict[str, float]:
        if self.embedder is None:
            return {}
        self._ensure_doc_embeddings()
        if not self._doc_embeddings:
            return {}
        try:
            qv = self.embedder.embed_one(query)
        except Exception:
            return {}
        out: dict[str, float] = {}
        for doc_id, dv in zip(self._doc_ids, self._doc_embeddings):
            out[doc_id] = cosine(qv, dv)
        return out

    def retrieve(self, question: str, top_tables: int = 4,
                 followup_context: dict | None = None) -> RetrievedContext:
        """Select the most relevant tables + their relationships for `question`."""
        query = question or ""
        ctx = RetrievedContext()
        ctx.intent_terms = self._intent_terms(query)

        lexical = self.index.lexical_search(query, limit=max(30, top_tables * 8))
        semantic = self._semantic_scores(query)

        max_lex = max((s for s, _ in lexical), default=0.0) or 1.0
        table_scores: dict[str, float] = {}
        doc_scores: dict[str, float] = {}

        for score, doc in lexical:
            lex_norm = score / max_lex
            sem = semantic.get(doc.doc_id, 0.0)
            combined = (1 - self.semantic_weight) * lex_norm + self.semantic_weight * max(sem, 0.0)
            doc_scores[doc.doc_id] = combined
            if doc.kind == "relationship":
                for t in (doc.payload.get("left_table"), doc.payload.get("right_table")):
                    if t:
                        table_scores[t] = table_scores.get(t, 0.0) + combined * 0.5
            else:
                table_scores[doc.table] = table_scores.get(doc.table, 0.0) + combined

        # Semantic-only hits not in the lexical list still count.
        for doc_id, sem in semantic.items():
            if doc_id in doc_scores or sem <= 0:
                continue
            for d in self.index.documents:
                if d.doc_id == doc_id:
                    if d.kind != "relationship":
                        table_scores[d.table] = table_scores.get(d.table, 0.0) + self.semantic_weight * sem
                    break

        # Follow-up: keep previous tables in play so "what about last year?"
        # targets the same subject rather than drifting.
        if followup_context:
            for t in followup_context.get("tables", []) or []:
                if t in self.index.tables:
                    table_scores[t] = table_scores.get(t, 0.0) + 0.75

        ranked = sorted(table_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        selected = [t for t, _ in ranked if _ > 0][:top_tables]

        # Cold start / very large schema with no lexical match: fall back to the
        # most connected tables so the LLM still gets a usable slice.
        if not selected and self.index.tables:
            selected = self._most_connected(top_tables)

        ctx.tables = selected
        ctx.scores = {t: round(table_scores.get(t, 0.0), 4) for t in selected}
        self._fill_columns(ctx, selected)
        ctx.relationships = self._relationships_for(set(selected))
        return ctx

    def _most_connected(self, n: int) -> list[str]:
        degree: dict[str, int] = {}
        for e in self.index.edges():
            degree[e["left_table"]] = degree.get(e["left_table"], 0) + 1
            degree[e["right_table"]] = degree.get(e["right_table"], 0) + 1
        tables = sorted(self.index.all_tables(),
                        key=lambda t: (-degree.get(t, 0), t))
        return tables[:n]

    def _fill_columns(self, ctx: RetrievedContext, tables: list[str]) -> None:
        for t in tables:
            doc = self.index.table_doc(t)
            payload = doc.payload if doc else {}
            columns = payload.get("columns", [])
            types = payload.get("column_types", {})
            measures, dims, dates, keys = [], [], [], []
            from .schema_index import classify_type, _is_id_like  # local import
            for c in columns:
                dt = types.get(c, "")
                cat = classify_type(dt)
                if c in payload.get("measures", []):
                    measures.append(c)
                elif c in payload.get("dates", []) or cat == "datetime":
                    dates.append(c)
                elif _is_id_like(c) or c in payload.get("primary_key", []):
                    keys.append(c)
                elif cat in ("text", "other"):
                    dims.append(c)
                elif cat == "numeric":
                    measures.append(c)
            ctx.measures[t] = measures
            ctx.dimensions[t] = dims
            ctx.dates[t] = dates
            ctx.keys[t] = keys

    def _relationships_for(self, tables: set[str]) -> list[dict]:
        out = []
        for e in self.index.edges():
            if e["left_table"] in tables and e["right_table"] in tables:
                out.append(e)
        return out

    def stats(self) -> dict:
        return {**self.index.stats(),
                "embedder": getattr(self.embedder, "name", None)}