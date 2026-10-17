"""Intent router + multi-tool orchestration tests (t.txt).

Deterministic: no LLM needed for routing; pipeline branches use the same
FakeLLM + SQLite pattern as test_copilot_rag.py.
"""

import sqlite3

import pytest

from copilot.intent import classify
from copilot.pipeline import CopilotPipeline, _analyze_rows


# ── Router ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("q", [
    "How many orders did we have last month?",
    "What was our total revenue?",
    "Which product sold the most?",
    "Compare sales between January and February.",
    "Show me the sales trend.",
])
def test_data_questions_use_sql_path(q):
    r = classify(q)
    assert r.intent == "data"
    assert r.tools == ["sql"]


def test_schema_questions_skip_sql():
    r = classify("What data do I have?")
    assert r.intent == "schema"
    assert r.tools == ["schema"]
    r = classify("Which table contains customer information?")
    assert r.intent == "schema"


def test_document_questions_never_sql():
    r = classify("What does this invoice say?")
    assert r.intent == "document"
    assert "sql" not in r.tools


def test_forecast_routes_to_forecast_tools():
    r = classify("What will sales look like next month?")
    assert r.intent == "forecast"
    assert "forecast" in r.tools
    assert "sql" in r.tools  # history evidence still retrieved


def test_why_routes_to_hypothesis():
    r = classify("Why are sales dropping?")
    assert r.intent == "hypothesis"
    assert "hypothesis" in r.tools


def test_recommendation_routes_correctly():
    r = classify("What should I do about declining sales?")
    assert r.intent == "recommendation"


def test_anomaly_routes_correctly():
    r = classify("Is anything unusual happening?")
    assert r.intent == "anomaly"


def test_health_check_routes_correctly():
    r = classify("Give me a quick health check of my business.")
    assert r.intent == "health"


def test_mixed_why_plus_action():
    r = classify("Sales have fallen recently. Why, and what should I do?")
    assert r.intent == "mixed"
    assert "hypothesis" in r.tools and "recommendation" in r.tools


def test_general_concept_never_sql():
    r = classify("What is a cash flow statement?")
    assert r.intent == "general"
    assert "sql" not in r.tools


def test_data_scoped_what_is_stays_sql():
    r = classify("What is our total revenue?")
    assert r.intent == "data"


# ── Analysis helper ──────────────────────────────────────────────────────

def test_analyze_rows_finds_top_and_momentum():
    cols = ["region", "total"]
    rows = [
        {"region": "North", "total": 100.0},
        {"region": "South", "total": 300.0},
        {"region": "East", "total": 200.0},
        {"region": "West", "total": 400.0},
    ]
    a = _analyze_rows(cols, rows, "hypothesis")
    assert a["top"]["label"] == "West"
    # Ranked/grouped rows carry no time axis: momentum must stay None
    # instead of fabricating a trend from return order.
    assert a["momentum_pct"] is None
    assert not a["thin"]


def test_analyze_rows_momentum_needs_date_axis():
    cols = ["month", "total"]
    rows = [{"month": f"2026-0{i + 1}", "total": 100.0 + i * 50.0} for i in range(6)]
    a = _analyze_rows(cols, rows, "forecast")
    assert a["momentum_pct"] is not None
    assert a["momentum_pct"] > 0


def test_analyze_rows_flags_thin_evidence():
    a = _analyze_rows(["total"], [{"total": 5}], "data")
    assert a["thin"]


# ── Pipeline branches ────────────────────────────────────────────────────

@pytest.fixture()
def sqlite_db(tmp_path):
    path = tmp_path / "shop.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            full_name TEXT,
            region TEXT
        );
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            order_date TEXT,
            amount REAL
        );
        INSERT INTO customers VALUES (1,'Acme','North'),(2,'Beta','South'),(3,'Cyan','North');
        INSERT INTO orders VALUES
            (1,1,'2024-01-05',100.0),
            (2,1,'2024-02-10',150.0),
            (3,2,'2024-01-20',300.0),
            (4,3,'2024-03-01',50.0);
    """)
    conn.commit()
    conn.close()
    return f"sqlite:///{path}"


@pytest.fixture()
def mapping():
    return {
        "database": "shop",
        "tables": {
            "customers": {
                "columns": [
                    {"column": "customer_id", "data_type": "INTEGER"},
                    {"column": "full_name", "data_type": "TEXT"},
                    {"column": "region", "data_type": "TEXT"},
                ],
                "primary_key": ["customer_id"],
                "foreign_keys": [],
                "row_count": 3,
            },
            "orders": {
                "columns": [
                    {"column": "order_id", "data_type": "INTEGER"},
                    {"column": "customer_id", "data_type": "INTEGER"},
                    {"column": "order_date", "data_type": "TEXT"},
                    {"column": "amount", "data_type": "REAL"},
                ],
                "primary_key": ["order_id"],
                "foreign_keys": [
                    {"column": "customer_id", "referenced_table": "customers",
                     "referenced_column": "customer_id"}
                ],
                "row_count": 4,
            },
        },
    }


class FakeLLM:
    def __init__(self, sql="SELECT 1", provider="local"):
        self.provider = provider
        self._sql = sql
        self.calls = []

    def complete(self, role, prompt, temperature=0.1, num_predict=400, timeout=None):
        self.calls.append((role, prompt))
        if role == "sql":
            return self._sql
        return "Business answer from fake LLM."

    def chat(self, role, messages, temperature=0.1, num_predict=400, timeout=None):
        return "ok"


def _pipe(sqlite_db, mapping, sql="SELECT 1", user="u_intent"):
    return CopilotPipeline(
        llm=FakeLLM(sql), schema_mapping=mapping, db_uri=sqlite_db,
        dialect="sqlite", user_id=user, embedder=None, provider_name="local",
    )


def test_schema_branch_never_calls_sql(sqlite_db, mapping):
    pipe = _pipe(sqlite_db, mapping, user="u_schema")
    ans = pipe.ask("What data do I have?")
    assert ans.ok
    assert ans.intent == "schema"
    assert ans.sql is None
    assert "customers" in ans.answer and "orders" in ans.answer
    assert not any(role == "sql" for role, _ in pipe.llm.calls)


def test_document_branch_never_runs_sql(sqlite_db, mapping):
    pipe = _pipe(sqlite_db, mapping, user="u_doc")
    ans = pipe.ask("What does this invoice say?")
    assert ans.intent == "document"
    assert ans.sql is None
    assert not any(role == "sql" for role, _ in pipe.llm.calls)


def test_general_branch_never_runs_sql(sqlite_db, mapping):
    pipe = _pipe(sqlite_db, mapping, user="u_gen")
    ans = pipe.ask("What is a cash flow statement?")
    assert ans.intent == "general"
    assert ans.sql is None
    assert not any(role == "sql" for role, _ in pipe.llm.calls)


def test_why_branch_labels_hypothesis(sqlite_db, mapping):
    sql = ("SELECT c.region AS region, SUM(o.amount) AS total FROM customers c "
           "JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.region")
    pipe = _pipe(sqlite_db, mapping, sql=sql, user="u_why")
    ans = pipe.ask("Why are sales dropping across regions?")
    assert ans.ok
    assert ans.intent == "hypothesis"
    assert ans.sql  # SQL used as evidence tool, not the whole answer


def test_mixed_branch_runs_full_pipeline(sqlite_db, mapping):
    sql = ("SELECT c.region AS region, SUM(o.amount) AS total FROM customers c "
           "JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.region")
    pipe = _pipe(sqlite_db, mapping, sql=sql, user="u_mixed")
    ans = pipe.ask("Sales have fallen recently. Why, and what should I do?")
    assert ans.ok
    assert ans.intent == "mixed"
    assert "intent" in ans.to_public()


def test_data_path_unchanged(sqlite_db, mapping):
    sql = ("SELECT c.region AS region, SUM(o.amount) AS total FROM customers c "
           "JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.region")
    pipe = _pipe(sqlite_db, mapping, sql=sql, user="u_data")
    ans = pipe.ask("total order amount by customer region")
    assert ans.ok
    assert ans.intent == "data"


@pytest.fixture()
def spaced_csv_db(tmp_path):
    """CSV-style source: column names with spaces, TEXT dates, mixed types."""
    path = tmp_path / "orders.csv.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE orders (
            "Order ID" INTEGER PRIMARY KEY,
            "Order Date" TEXT,
            "Customer Name" TEXT,
            "Amount" REAL
        );
        INSERT INTO orders VALUES
            (1,'2024-01-05','Acme',100.0),
            (2,'2024-02-10','Acme',150.0),
            (3,'2024-01-20','Beta',300.0),
            (4,'2024-03-01','Cyan',50.0);
    """)
    conn.commit()
    conn.close()
    return f"sqlite:///{path}"


@pytest.fixture()
def spaced_mapping():
    return {
        "database": "orders",
        "tables": {
            "orders": {
                "columns": [
                    {"column": "Order ID", "data_type": "INTEGER"},
                    {"column": "Order Date", "data_type": "TEXT"},
                    {"column": "Customer Name", "data_type": "TEXT"},
                    {"column": "Amount", "data_type": "REAL"},
                ],
                "primary_key": ["Order ID"],
                "foreign_keys": [],
                "row_count": 4,
            },
        },
    }


def test_csv_spaced_columns_answer(spaced_csv_db, spaced_mapping):
    sql = 'SELECT "Customer Name" AS name, SUM("Amount") AS total FROM orders GROUP BY "Customer Name"'
    pipe = CopilotPipeline(
        llm=FakeLLM(sql), schema_mapping=spaced_mapping, db_uri=spaced_csv_db,
        dialect="sqlite", user_id="u_spaced", embedder=None, provider_name="local",
    )
    ans = pipe.ask("Which customer spent the most?")
    assert ans.ok, ans.error
    assert ans.rows
    assert ans.intent == "data"


def test_csv_followup_reuses_context(spaced_csv_db, spaced_mapping):
    sql = 'SELECT "Customer Name" AS name, SUM("Amount") AS total FROM orders GROUP BY "Customer Name"'
    pipe = CopilotPipeline(llm=FakeLLM(sql), schema_mapping=spaced_mapping, db_uri=spaced_csv_db,
                           dialect="sqlite", user_id="u_spaced_fu", embedder=None, provider_name="local")
    first = pipe.ask("Which customer spent the most?")
    assert first.ok
    again = pipe.ask("why?")
    # Follow-up resolves against prior context; must not error.
    assert again.answer


def test_suggestions_are_data_driven(spaced_mapping):
    from api.copilot_routes import _suggest_questions
    out = _suggest_questions(spaced_mapping)
    assert len(out) == 4
    blob = " ".join(out).lower()
    assert "amount" in blob or "order" in blob


def test_suggestions_fallback_without_schema():
    from api.copilot_routes import _suggest_questions
    out = _suggest_questions({})
    assert out == []


def test_forecast_branch_is_honest(sqlite_db, mapping):
    sql = ("SELECT o.order_date AS month, SUM(o.amount) AS total FROM orders o "
           "GROUP BY o.order_date ORDER BY o.order_date")
    pipe = _pipe(sqlite_db, mapping, sql=sql, user="u_fc")
    ans = pipe.ask("What will sales look like next month?")
    assert ans.intent == "forecast"
    # Either a validated engine outlook or an honest insufficient-history note.
    assert ans.ok
    assert ans.sql  # history evidence still retrieved via SQL tool


def test_document_guidance_without_docs(sqlite_db, mapping):
    pipe = _pipe(sqlite_db, mapping, user="u_doc2")
    ans = pipe.ask("What does this invoice say?")
    assert ans.intent == "document"
    assert ans.sql is None
    assert "document" in ans.answer.lower()


def test_lexical_score_matches_terms():
    from copilot.pipeline import _lexical_score
    assert _lexical_score("What does the invoice say about revenue?",
                          "Total revenue for Q1 was 42K.") > 0
    assert _lexical_score("What does the invoice say?", "the and of to") == 0


@pytest.mark.parametrize("q", [
    "Compare the report's explanation with the actual sales data.",
    "Does the management report explain why sales fell?",
    "Why did sales decline according to both the data and the report?",
])
def test_combined_router(q):
    r = classify(q)
    assert r.intent == "combined"
    assert "sql" in r.tools and "document" in r.tools


def test_combined_downgrades_without_document(sqlite_db, mapping):
    sql = ("SELECT c.region AS region, SUM(o.amount) AS total FROM customers c "
           "JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.region")
    pipe = _pipe(sqlite_db, mapping, sql=sql, user="u_comb1")
    ans = pipe.ask("Why did sales decline according to both the data and the report?")
    assert ans.intent == "combined"
    # No document connected: honest database-only answer, no fabrication.
    assert ans.ok
    assert ans.sql


def test_combined_full_with_document(sqlite_db, mapping, tmp_path, monkeypatch):
    import insight_rag.storage as storage
    monkeypatch.setattr(storage, "user_dir", lambda uid: tmp_path / str(uid))
    from insight_rag.storage import RAGMetadataStore
    store = RAGMetadataStore("u_comb2")
    doc_id = "testdoc-12345678"
    store.put_document({"document_id": doc_id, "filename": "q2-report.pdf",
                        "created_at": "2026-01-01"})
    from langchain_core.documents import Document
    store.save_chunks(doc_id, [Document(
        page_content="Sales revenue declined because supply delays slowed deliveries in Q2.",
        metadata={"page": 3})])
    sql = ("SELECT c.region AS region, SUM(o.amount) AS total FROM customers c "
           "JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.region")

    class NoStoryLLM(FakeLLM):
        def complete(self, role, prompt, **kw):
            self.calls.append((role, prompt))
            if role == "sql":
                return self._sql
            raise RuntimeError("no story model")

    from copilot.pipeline import CopilotPipeline
    pipe = CopilotPipeline(
        llm=NoStoryLLM(sql), schema_mapping=mapping, db_uri=sqlite_db,
        dialect="sqlite", user_id="u_comb2", embedder=None,
        provider_name="local", document_id=doc_id,
    )
    ans = pipe.ask("Does the management report explain why sales fell?")
    assert ans.intent == "combined"
    assert ans.ok
    # Deterministic fallback attributes both sources explicitly.
    assert "supply delays" in ans.answer
    assert "database" in ans.answer.lower()
