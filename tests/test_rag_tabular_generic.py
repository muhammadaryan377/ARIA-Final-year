"""Generic tabular-RAG regressions (no dataset/table/column hardcoding).

Proves the RAG fix works for *any* CSV/DB row set using a small synthetic
frame with neutral column names:

- tabular ingestion emits a full-coverage summary + sampled row chunks,
- identifier sub-token matching finds columns from plain-word queries,
- table reasoning falls back to inferred operations with correct citations,
- large tables stay bounded (sibling cap, sampled chunks, summary scope),
- structured-DB handoff recognises generic analytics phrasing.
"""

import pandas as pd
from langchain_core.documents import Document

from insight_rag.retrieval import bm25_search, terms
from insight_rag.service import InsightPDFRAG as CoreRAG
from insight_rag.structured_db import is_structured_db_question
from insight_rag.table_reasoner import build_table_facts, infer_table_operations
from insight_rag.tabular import dataframe_to_table_documents


def _frame(rows: int = 8) -> pd.DataFrame:
    return pd.DataFrame({
        "group": [f"g{i % 3}" for i in range(rows)],
        "period": [f"2024-{(i % 4) + 1:02d}-01" for i in range(rows)],
        "amount": [100.0 + 10 * i for i in range(rows)],
        "units": [1 + (i % 5) for i in range(rows)],
    })


def _sources(docs):
    return [
        {
            "source_id": f"S{i}",
            "document_id": doc.metadata["document_id"],
            "filename": doc.metadata["filename"],
            "page": doc.metadata["page"],
            "content_type": doc.metadata.get("content_type"),
            "table_index": doc.metadata.get("table_index"),
            "chunk_id": doc.metadata.get("chunk_id"),
        }
        for i, doc in enumerate(docs, start=1)
    ]


def test_tabular_ingest_emits_summary_plus_row_chunks():
    docs, stats = dataframe_to_table_documents(
        _frame(), document_id="doc-tab", filename="generic.csv",
    )
    assert stats["rows"] == 8
    assert docs[0].metadata["table_summary"] is True
    assert docs[0].metadata["content_type"] == "table"
    assert docs[0].metadata["page"] == 1
    aggs = docs[0].metadata["table_summary_aggregates"]
    assert aggs["amount"]["sum"] == sum(100.0 + 10 * i for i in range(8))
    assert aggs["amount"]["count"] == 8
    assert all(doc.metadata["table_index"] == 1 for doc in docs)
    assert len(docs) >= 2  # summary + at least one row chunk


def test_identifier_subtokens_match_plain_words():
    assert "sales" in terms("net_sales")
    docs = [
        Document(page_content="| net_sales | group |\n| --- | --- |\n| 10 | g1 |",
                 metadata={"document_id": "d", "chunk_id": "c1"}),
        Document(page_content="unrelated prose about weather",
                 metadata={"document_id": "d", "chunk_id": "c2"}),
    ]
    ranked = bm25_search("total sales", docs, limit=2)
    assert ranked and ranked[0].metadata["chunk_id"] == "c1"


def test_inferred_sum_fallback_with_citations():
    docs, _ = dataframe_to_table_documents(
        _frame(), document_id="doc-tab", filename="generic.csv",
    )
    facts = build_table_facts(
        "What is the total amount?", docs,
        operations=(), sources=_sources(docs),
    )
    expected = sum(100.0 + 10 * i for i in range(8))
    assert f"amount sum={expected:g}" in facts
    assert "[S1]" in facts  # summary provenance, no invented labels


def test_no_aggregation_intent_stays_empty():
    docs, _ = dataframe_to_table_documents(
        _frame(), document_id="doc-tab", filename="generic.csv",
    )
    assert build_table_facts("describe the groups", docs, operations=()) == ""


def test_infer_operations_is_generic():
    ops, _, _, _ = infer_table_operations("What is the average units?")
    assert "MEAN" in ops
    ops, filter_op, filter_value, _ = infer_table_operations("amount over 150")
    assert "FILTER" in ops and filter_op in {"GT", "GTE"} and filter_value == 150
    ops, _, _, top_n = infer_table_operations("top 3 groups by amount")
    assert "RANK" in ops and top_n == 3
    assert infer_table_operations("describe the groups")[0] == ()


def test_large_table_stays_bounded_with_full_scope_summary():
    big = pd.DataFrame({
        "group": [f"g{i % 45}" for i in range(3000)],
        "amount": [float(i) for i in range(3000)],
    })
    docs, stats = dataframe_to_table_documents(
        big, document_id="doc-big", filename="big.csv",
    )
    assert stats["sampled"] is True
    assert len(docs) <= 14  # summary + capped row chunks
    facts = build_table_facts(
        "What is the total amount?", docs,
        operations=("SUM",), sources=_sources(docs),
    )
    assert f"amount sum={sum(range(3000)):g}" in facts  # full 3000-row scope
    assert "3000 rows (full-table summary)" in facts


def test_sibling_expansion_is_bounded():
    table_docs = []
    for i in range(40):
        table_docs.append(Document(
            page_content=f"| group | amount |\n| --- | --- |\n| g{i} | {i} |",
            metadata={
                "document_id": "doc-big", "filename": "big.csv", "page": 1,
                "content_type": "table", "table_index": 1,
                "table_chunk_index": i, "table_chunk_count": 40,
                "chunk_id": f"chunk-{i}",
            },
        ))

    class FakeMeta:
        def load_chunks(self, document_id):
            return list(table_docs)

    rag = CoreRAG.__new__(CoreRAG)
    rag.metadata = FakeMeta()
    restored = rag._expand_table_siblings(table_docs[:1])
    assert len(restored) <= 1 + 16
    assert any(doc.metadata.get("retrieval_sibling") for doc in restored[1:])


def test_structured_detector_catches_generic_analytics():
    assert is_structured_db_question("What are total weekly sales?")
    assert is_structured_db_question("What are total sales by store?")
    assert is_structured_db_question("Show monthly sales trend.")
    assert is_structured_db_question("What is the average weekly sales?")
    assert not is_structured_db_question("What does the report say about sales?")
    assert not is_structured_db_question("Hi, how are you?")
