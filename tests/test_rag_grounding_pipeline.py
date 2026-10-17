"""RAG-as-pipeline + answer-grounding tests (n.txt §§5-6).

Covers: knowledge-source decision, bounded grounded context (AdventureWorks
style + single-table), grounding verdicts (numeric match vs SQL, unsupported
claims, question coverage), pipeline repair on invented numbers, and
insight_rag structured-DB awareness. Offline: SQLite + FakeLLM only.
"""

import sqlite3

import pytest

from copilot.answer_grounding import check_answer_grounding
from copilot.grounded_context import (
    build_grounded_db_context,
    decide_knowledge_source,
)
from copilot.pipeline import CopilotPipeline
from copilot.schema_index import build_schema_index
from copilot.retriever import SchemaRetriever
from insight_rag.structured_db import (
    build_structured_context,
    describe_structured_handoff,
    is_structured_db_question,
)


# ── Fixtures: AdventureWorks-style relational slice ──────────────────────

@pytest.fixture()
def aw_mapping():
    return {
        "database": "adventureworks",
        "metric_definitions": {
            "total_sales": "SUM of SalesOrderDetail.LineTotal",
            "average_order_value": "total_sales / COUNT(DISTINCT SalesOrderID)",
        },
        "validated_query_patterns": [
            {"name": "sales_by_territory",
             "description": "total sales grouped by sales territory",
             "tables": ["SalesOrderHeader", "SalesTerritory"],
             "sql": "SELECT t.Name, SUM(h.TotalDue) FROM SalesOrderHeader h "
                    "JOIN SalesTerritory t ON h.TerritoryID = t.TerritoryID GROUP BY t.Name"},
            {"name": "top_products",
             "description": "top products by sales amount",
             "tables": ["SalesOrderDetail", "Product"],
             "sql": "SELECT p.Name, SUM(d.LineTotal) FROM SalesOrderDetail d "
                    "JOIN Product p ON d.ProductID = p.ProductID GROUP BY p.Name ORDER BY 2 DESC LIMIT 10"},
        ],
        "tables": {
            "Customer": {
                "columns": [
                    {"column": "CustomerID", "data_type": "INTEGER"},
                    {"column": "CustomerName", "data_type": "TEXT",
                     "description": "Full customer display name"},
                    {"column": "TerritoryID", "data_type": "INTEGER"},
                ],
                "primary_key": ["CustomerID"],
                "description": "People and stores who buy products",
                "foreign_keys": [{"column": "TerritoryID",
                                  "referenced_table": "SalesTerritory",
                                  "referenced_column": "TerritoryID"}],
                "row_count": 100,
            },
            "Product": {
                "columns": [
                    {"column": "ProductID", "data_type": "INTEGER"},
                    {"column": "Name", "data_type": "TEXT"},
                    {"column": "Category", "data_type": "TEXT"},
                    {"column": "ListPrice", "data_type": "REAL"},
                ],
                "primary_key": ["ProductID"],
                "description": "Sellable goods catalogue",
                "foreign_keys": [],
                "row_count": 50,
            },
            "SalesOrderHeader": {
                "columns": [
                    {"column": "SalesOrderID", "data_type": "INTEGER"},
                    {"column": "CustomerID", "data_type": "INTEGER"},
                    {"column": "TerritoryID", "data_type": "INTEGER"},
                    {"column": "OrderDate", "data_type": "DATE"},
                    {"column": "TotalDue", "data_type": "REAL"},
                ],
                "primary_key": ["SalesOrderID"],
                "description": "One row per customer order",
                "foreign_keys": [
                    {"column": "CustomerID", "referenced_table": "Customer",
                     "referenced_column": "CustomerID"},
                    {"column": "TerritoryID", "referenced_table": "SalesTerritory",
                     "referenced_column": "TerritoryID"},
                ],
                "row_count": 500,
            },
            "SalesOrderDetail": {
                "columns": [
                    {"column": "SalesOrderID", "data_type": "INTEGER"},
                    {"column": "ProductID", "data_type": "INTEGER"},
                    {"column": "OrderQty", "data_type": "INTEGER"},
                    {"column": "LineTotal", "data_type": "REAL"},
                ],
                "primary_key": ["SalesOrderID"],
                "description": "Line items within each order",
                "foreign_keys": [
                    {"column": "SalesOrderID", "referenced_table": "SalesOrderHeader",
                     "referenced_column": "SalesOrderID"},
                    {"column": "ProductID", "referenced_table": "Product",
                     "referenced_column": "ProductID"},
                ],
                "row_count": 1200,
            },
            "SalesTerritory": {
                "columns": [
                    {"column": "TerritoryID", "data_type": "INTEGER"},
                    {"column": "Name", "data_type": "TEXT"},
                ],
                "primary_key": ["TerritoryID"],
                "description": "Sales regions",
                "foreign_keys": [],
                "row_count": 10,
            },
        },
    }


@pytest.fixture()
def aw_db(tmp_path):
    path = tmp_path / "aw.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE Product (ProductID INTEGER PRIMARY KEY, Name TEXT,
            Category TEXT, ListPrice REAL);
        CREATE TABLE SalesOrderHeader (SalesOrderID INTEGER PRIMARY KEY,
            CustomerID INTEGER, TerritoryID INTEGER, OrderDate TEXT, TotalDue REAL);
        CREATE TABLE SalesOrderDetail (SalesOrderID INTEGER, ProductID INTEGER,
            OrderQty INTEGER, LineTotal REAL);
        INSERT INTO Product VALUES (1,'Mountain Bike','Bikes',1000.0),(2,'Helmet','Accessories',50.0);
        INSERT INTO SalesOrderHeader VALUES (1,1,1,'2024-01-05',1050.0),(2,2,1,'2024-02-10',50.0);
        INSERT INTO SalesOrderDetail VALUES (1,1,1,1000.0),(1,2,1,50.0),(2,2,1,50.0);
    """)
    conn.commit()
    conn.close()
    return f"sqlite:///{path}"


class FakeLLM:
    def __init__(self, sql, story, provider="local"):
        self.provider = provider
        self._sql, self._story = sql, story
        self.calls = []

    def complete(self, role, prompt, temperature=0.1, num_predict=400, timeout=None):
        self.calls.append((role, prompt))
        return self._sql if role == "sql" else self._story

    def chat(self, role, messages, temperature=0.1, num_predict=400, timeout=None):
        return "ok"


# ── Item 5: knowledge-source decision ────────────────────────────────────

def test_source_decision_db_document_hybrid_general():
    assert decide_knowledge_source("total sales by territory")["source"] == "db"
    assert decide_knowledge_source("what does the uploaded invoice contain?")["source"] == "document"
    assert decide_knowledge_source(
        "compare the report explanation with the actual sales data")["source"] == "hybrid"
    assert decide_knowledge_source("what is gross margin?")["source"] == "general"


# ── Item 5: bounded grounded context ─────────────────────────────────────

def test_grounded_context_bounded_never_whole_db(aw_mapping):
    idx = build_schema_index(aw_mapping)
    ctx = SchemaRetriever(idx, embedder=None).retrieve("top 10 products by sales")
    grounded = build_grounded_db_context("top 10 products by sales", ctx,
                                         aw_mapping, with_samples=False)
    assert len(grounded.tables) <= 4 < len(aw_mapping["tables"])
    prompt = grounded.to_prompt()
    # Grounded slice: relationships + business meaning present, whole-DB dump absent.
    assert "RELATIONSHIPS" in prompt
    assert "SalesOrderDetail" in prompt or "Product" in prompt
    assert "VALIDATED QUERY PATTERNS" in prompt
    assert "METRIC DEFINITIONS" in prompt
    assert grounded.provenance.get("bounded") is True


def test_grounded_context_single_table_relationships_none():
    mapping = {"database": "one", "tables": {
        "sales": {"columns": [
            {"column": "id", "data_type": "INTEGER"},
            {"column": "amount", "data_type": "REAL"},
            {"column": "sold_at", "data_type": "DATE"}],
            "primary_key": ["id"], "foreign_keys": [], "row_count": 7}}}
    idx = build_schema_index(mapping)
    ctx = SchemaRetriever(idx, embedder=None).retrieve("what are my sales?")
    grounded = build_grounded_db_context("what are my sales?", ctx, mapping,
                                         with_samples=False)
    assert grounded.tables == ["sales"]
    assert "none" in grounded.to_prompt().lower()  # relationships = none is valid


def test_grounded_prompt_includes_types_and_descriptions(aw_mapping):
    idx = build_schema_index(aw_mapping)
    ctx = SchemaRetriever(idx, embedder=None).retrieve("how many customers do we have?")
    grounded = build_grounded_db_context("how many customers do we have?", ctx,
                                         aw_mapping, with_samples=False)
    prompt = grounded.to_prompt()
    assert "Customer" in prompt
    assert "People and stores" in prompt  # table business meaning preserved
    assert "INTEGER" in prompt or "TEXT" in prompt  # data types preserved


# ── Item 6: grounding verdicts ───────────────────────────────────────────

def test_grounding_accepts_sql_numbers():
    rows = [{"total": 1100.0}]
    out = check_answer_grounding(
        question="What are total sales?", answer="Total sales were 1100.",
        retrieved_tables=["SalesOrderHeader"], retrieved_columns=["TotalDue"],
        sql="SELECT SUM(TotalDue)", rows=rows, columns=["total"], intent="data")
    assert out["verdict"] == "grounded"
    assert out["checks"]["numeric_match_vs_sql"] is True
    assert out["checks"]["used_retrieved_context"] is True
    assert out["checks"]["answers_question"] is True


def test_grounding_rejects_invented_numbers():
    rows = [{"total": 1100.0}]
    out = check_answer_grounding(
        question="What are total sales?", answer="Total sales were $9,999,999, a record quarter.",
        retrieved_tables=["SalesOrderHeader"], retrieved_columns=["TotalDue"],
        sql="SELECT SUM(TotalDue)", rows=rows, columns=["total"], intent="data")
    assert out["verdict"] == "ungrounded"
    assert out["reason"] == "numeric_mismatch_vs_sql"
    assert out["checks"]["unsupported_claims"]


def test_grounding_empty_result_with_numbers_is_ungrounded():
    out = check_answer_grounding(
        question="Total sales?", answer="Total sales were $500.",
        retrieved_tables=["t"], sql="SELECT 1", rows=[], columns=["total"],
        intent="data")
    assert out["verdict"] == "ungrounded"


# ── Pipeline: SQL authoritative, repair invented numbers ─────────────────

def test_pipeline_repairs_hallucinated_total(aw_db, aw_mapping):
    llm = FakeLLM(
        sql="SELECT SUM(LineTotal) AS total FROM SalesOrderDetail",
        story="Total sales were $9,999,999 across the period.",
    )
    pipe = CopilotPipeline(llm=llm, schema_mapping=aw_mapping, db_uri=aw_db,
                           dialect="sqlite", user_id="u_ground_fix", embedder=None)
    ans = pipe.ask("What are total sales?")
    assert ans.ok
    assert "9999999" not in ans.answer.replace(",", "")
    assert "1150" not in ans.answer.replace(",", "")  # sanity: no invented figure
    assert "1100" in ans.answer.replace(",", "")  # true SUM(LineTotal)
    assert ans.grounding and ans.grounding["verdict"] == "grounded"
    assert ans.to_public()["grounding"]["verdict"] == "grounded"


def test_pipeline_grounded_answer_passes_through(aw_db, aw_mapping):
    llm = FakeLLM(
        sql="SELECT SUM(LineTotal) AS total FROM SalesOrderDetail",
        story="Total sales were 1100 in SalesOrderDetail.",
    )
    pipe = CopilotPipeline(llm=llm, schema_mapping=aw_mapping, db_uri=aw_db,
                           dialect="sqlite", user_id="u_ground_ok", embedder=None)
    ans = pipe.ask("What are total sales?")
    assert ans.ok
    assert ans.grounding["verdict"] == "grounded"
    assert ans.grounding["repaired"] is False


# ── insight_rag structured-DB awareness ──────────────────────────────────

def test_structured_detector_routes_db_not_docs():
    assert is_structured_db_question("What is our monthly sales trend?")
    assert is_structured_db_question("Which territory performs best?")
    assert not is_structured_db_question("What does the report say about sales?")
    assert not is_structured_db_question("Hi, how are you?")


def test_structured_context_bounded(aw_mapping):
    out = build_structured_context("Which category generated the most sales?",
                                   aw_mapping, dialect="sqlite")
    assert out["bounded"] is True
    assert len(out["tables"]) <= 4 < out["full_table_count"]


def test_structured_handoff_points_to_copilot():
    text = describe_structured_handoff("What is our monthly sales trend?")
    assert "Copilot" in text and "PDF" in text
