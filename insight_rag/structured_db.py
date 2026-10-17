"""Structured-database awareness for Insight RAG (n.txt §5).

Insight RAG is a *document* capability. This module gives it just enough
structured-DB understanding for AdventureWorks-style schemas so that:

- database/analytics questions are recognised deterministically (no LLM),
- they are never answered from PDF evidence (which would hallucinate),
- the caller gets a bounded structured context (via Copilot's grounded
  context builder) or an honest handoff to Copilot/Data instead.

No PDF paths are modified. All helpers are deterministic and import-light
(copilot imports are lazy to avoid cycles).
"""

from __future__ import annotations

import re

_DOCUMENT_WORDS = re.compile(
    r"\b(pdf|document|report|invoice|contract|uploaded|page|pages|chapter|"
    r"section|paragraph|quote|cite|citation)\b",
    re.IGNORECASE,
)

# Shape-based intent (generic): aggregation/comparison questions about
# structured data, regardless of domain nouns (sales OR churn OR defects
# OR readmissions). Never a noun list — "churn by plan" and "defects by
# line" must route the same as "sales by region".
_DB_ANALYTICS = re.compile(
    r"\btop\s+\d+\b|\bbreakdown\b|\btrend\b|\bover\s+time\b|"
    r"\b(by|per|across|versus|vs\.?)\b.{0,24}\b|\b.{0,24}\b(by|per|across)\b|"
    r"\b(sum|total|average|avg|count|how\s+many|max|min|median|mean)\b|"
    r"\bcompare\b|\bhighest(?=[-\s?!.]|$)|\bbest(?=[-\s?!.]|$)|\blowest(?=[-\s?!.]|$)|\bworst(?=[-\s?!.]|$)|"
    r"\bgrowth\b|\bdecline\b|\bgrew\b|\bfell\b|\bchange\b|\bdistribut\w+\b|"
    r"\b(monthly|weekly|daily|quarterly|yearly)\b",
    re.IGNORECASE,
)


def is_structured_db_question(question: str) -> bool:
    """True for analytics questions needing structured DB context."""
    text = (question or "").strip()
    if len(text) < 8:
        return False
    if _DOCUMENT_WORDS.search(text):
        return False
    return bool(_DB_ANALYTICS.search(text))


def describe_structured_handoff(question: str) -> str:
    """Honest handoff answer: DB questions need Copilot/Data, not PDFs."""
    return (
        "That looks like a database analytics question "
        f"(“{(question or '').strip()[:120]}”). "
        "I answer from uploaded PDFs, so I can't compute it from document "
        "text. Ask it in Copilot (or the Data page) against the connected "
        "database — e.g. “Total sales by territory”, “Top 10 products”, "
        "“Monthly sales trend” — where the SQL result is authoritative."
    )


def build_structured_context(question: str, mapping: dict,
                             *, dialect: str = "postgresql",
                             db_uri: str | None = None) -> dict:
    """Build bounded structured DB context without dumping the whole DB.

    Runs Copilot retrieval + grounded-context construction over the supplied
    schema mapping. Returns ``{"tables": [...], "prompt": ..., "source": ...}``.
    Never raises; empty tables mean "no relevant slice found".
    """
    try:
        from copilot.grounded_context import build_grounded_db_context
        from copilot.schema_index import build_schema_index
        from copilot.retriever import SchemaRetriever

        index = build_schema_index(mapping or {}, dialect=dialect)
        if not index.tables:
            return {"tables": [], "prompt": "", "source": "db", "bounded": True}
        ctx = SchemaRetriever(index).retrieve(question or "")
        if ctx.is_empty():
            return {"tables": [], "prompt": "", "source": "db", "bounded": True}
        grounded = build_grounded_db_context(
            question or "", ctx, mapping or {},
            db_uri=db_uri, dialect=dialect, with_samples=db_uri is not None,
        )
        return {"tables": list(grounded.tables), "prompt": grounded.to_prompt(),
                "source": grounded.source, "intent": grounded.intent,
                "provenance": dict(grounded.provenance), "bounded": True,
                "full_table_count": len(index.tables)}
    except Exception:
        return {"tables": [], "prompt": "", "source": "db", "bounded": True}
