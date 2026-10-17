"""Copilot RAG tests: provider-independence, dialect handling, retrieval,
SQL validation, execution, response and chart inference.

Uses SQLite as the execution backend so the whole pipeline (minus the LLM)
runs offline in CI. The LLM is a fake exposing the same interface as
llm_provider.LLMProvider, so the SAME pipeline code path is exercised for
both "local" and "cloud" provider names.
"""

import sqlite3

import pytest

from copilot.embeddings import HashingEmbedder, create_embedder, cosine
from copilot.schema_index import SchemaIndex, build_schema_index
from copilot.retriever import SchemaRetriever
from copilot.conversation import ConversationMemory
from copilot.sql_engine import SqlEngine, is_read_only, clean_sql
from copilot.pipeline import CopilotPipeline, infer_chart
from copilot.answer_grounding import check_answer_grounding


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture()
def sqlite_db(tmp_path):
    """A small unseen schema (not a known demo DB) to prove dynamism."""
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
    """Same interface as llm_provider.LLMProvider, deterministic SQL."""

    def __init__(self, sql: str, provider: str = "local"):
        self.provider = provider
        self._sql = sql
        self.calls = []

    def complete(self, role, prompt, temperature=0.1, num_predict=400, timeout=None):
        self.calls.append((role, prompt))
        if role == "sql":
            return self._sql
        return "Acme leads with 250 in total order value."

    def chat(self, role, messages, temperature=0.1, num_predict=400, timeout=None):
        return "ok"


# ── Embeddings ───────────────────────────────────────────────────────────

def test_hashing_embedder_is_deterministic_and_normalized():
    emb = HashingEmbedder()
    a = emb.embed_one("total sales by region")
    b = emb.embed_one("total sales by region")
    assert a == b
    assert abs(cosine(a, a) - 1.0) < 1e-6


def test_create_embedder_never_raises():
    emb = create_embedder()
    assert emb.embed_one("hello") is not None


# ── Schema index / retrieval ─────────────────────────────────────────────

def test_index_is_built_dynamically(mapping):
    idx = build_schema_index(mapping)
    assert set(idx.all_tables()) == {"customers", "orders"}
    kinds = {d.kind for d in idx.documents}
    assert {"table", "column", "relationship"} <= kinds
    assert idx.stats()["relationships"] >= 1


def test_retriever_selects_relevant_tables_only(mapping):
    idx = build_schema_index(mapping)
    r = SchemaRetriever(idx, embedder=None)
    ctx = r.retrieve("total order amount by customer region")
    assert "orders" in ctx.tables
    assert "customers" in ctx.tables
    assert ctx.measures.get("orders")  # amount detected as a measure
    assert any(c == "region" for c in ctx.dimensions.get("customers", []))


# ── Conversation memory ──────────────────────────────────────────────────

def test_followup_detection_and_resolution():
    mem = ConversationMemory("u1")
    mem.clear()
    assert ConversationMemory.is_followup("what about last year?")
    assert not ConversationMemory.is_followup(
        "Show me the monthly revenue trend for the last two years by region")
    mem.append(question="top customers by amount",
               context={"tables": ["orders", "customers"],
                        "measures": ["amount"], "group_by": ["full_name"]})
    resolved = mem.resolve("what about last year?")
    assert "Previous question" in resolved
    assert "orders" in resolved


# ── SQL engine ───────────────────────────────────────────────────────────

def test_read_only_guard_rejects_mutations():
    assert is_read_only("SELECT * FROM t")
    assert is_read_only("WITH x AS (SELECT 1) SELECT * FROM x")
    assert not is_read_only("DELETE FROM t")
    assert not is_read_only("DROP TABLE t")
    assert not is_read_only("UPDATE t SET a=1")
    assert not is_read_only("")
    # A destructive keyword inside a string literal is not destructive.
    assert is_read_only("SELECT * FROM t WHERE name = 'drop table'")


def test_clean_sql_strips_fences_and_explanation():
    raw = "```sql\nSELECT 1;\n```\nThis query returns one."
    assert clean_sql(raw) == "SELECT 1"


def test_engine_validates_unknown_tables(sqlite_db, mapping):
    engine = SqlEngine(sqlite_db, dialect="sqlite")
    ok, reason = engine.validate("SELECT nope FROM does_not_exist", mapping["tables"])
    assert not ok
    assert "Unknown table" in reason


def test_engine_executes_real_query(sqlite_db, mapping):
    engine = SqlEngine(sqlite_db, dialect="sqlite")
    res = engine.execute(
        "SELECT region, SUM(amount) AS total FROM customers c "
        "JOIN orders o ON c.customer_id = o.customer_id GROUP BY region",
        mapping["tables"],
    )
    assert res.ok, res.error
    assert res.row_count >= 1
    assert "total" in res.columns


def test_limit_branch_keeps_text_cast_rewrite(sqlite_db, mapping):
    """Plain-row query: the appended LIMIT must be built from safe_sql,
    so the TEXT-cast rewrite (bare CAST crashes on '₹399'-style values)
    survives in the SQL that actually executes."""
    engine = SqlEngine(sqlite_db, dialect="sqlite", max_rows=1000)
    res = engine.execute(
        "SELECT order_id, CAST(order_date AS REAL) AS d FROM orders",
        mapping["tables"],
    )
    assert res.ok, res.error
    assert "NULLIF" in res.sql          # rewritten cast present
    assert "LIMIT 1000" in res.sql      # LIMIT present
    assert "CAST(order_date AS REAL)" not in res.sql  # original cast replaced


# ── Chart inference (dynamic) ────────────────────────────────────────────

def test_infer_chart_ranking_is_bar():
    cols = ["full_name", "total"]
    rows = [{"full_name": "Acme", "total": 250.0}, {"full_name": "Beta", "total": 300.0}]
    spec = infer_chart(cols, rows, "top customers by total amount")
    assert spec["type"] in ("bar", "bar_horizontal")
    assert spec["x"] == "full_name" and spec["y"] == "total"


def test_infer_chart_time_is_line_or_area():
    cols = ["month", "total"]
    rows = [{"month": "2024-01", "total": 10}, {"month": "2024-02", "total": 20}]
    spec = infer_chart(cols, rows, "monthly revenue trend")
    assert spec["type"] in ("line", "area")


# ── Pipeline: provider independence ──────────────────────────────────────

@pytest.mark.parametrize("provider_name", ["local", "cloud"])
def test_pipeline_runs_identically_for_both_providers(
        sqlite_db, mapping, provider_name):
    llm = FakeLLM(
        "SELECT c.region AS region, SUM(o.amount) AS total "
        "FROM customers c JOIN orders o ON c.customer_id = o.customer_id "
        "GROUP BY c.region ORDER BY total DESC",
        provider=provider_name,
    )
    pipe = CopilotPipeline(
        llm=llm, schema_mapping=mapping, db_uri=sqlite_db,
        dialect="sqlite", user_id=f"u_{provider_name}", embedder=None,
        provider_name=provider_name,
    )
    answer = pipe.ask("total order amount by customer region")
    assert answer.ok, answer.error
    assert answer.rows
    assert answer.chart is not None
    assert answer.provider == provider_name
    # The SQL role was used for generation, and only read-only SQL ran.
    assert any(role == "sql" for role, _ in llm.calls)


def test_pipeline_handles_unanswerable_question(sqlite_db, mapping):
    llm = FakeLLM("SELECT * FROM nonexistent_table")
    pipe = CopilotPipeline(
        llm=llm, schema_mapping=mapping, db_uri=sqlite_db,
        dialect="sqlite", user_id="u_err", embedder=None, provider_name="local",
    )
    answer = pipe.ask("total order amount by region")
    assert answer.ok is False
    assert answer.error


def test_pipeline_without_schema_is_graceful():
    llm = FakeLLM("SELECT 1")
    pipe = CopilotPipeline(
        llm=llm, schema_mapping={}, db_uri="sqlite:///:memory:",
        dialect="sqlite", user_id="u_empty", embedder=None,
    )
    answer = pipe.ask("anything")
    assert answer.ok is False
    assert "schema" in answer.answer.lower()


def test_dialect_rules_differ_by_database():
    from copilot.sql_engine import build_sql_prompt

    class Ctx:
        def to_prompt(self, max_tables=6):
            return "TABLE t: a (measure)"

    pg = build_sql_prompt("q", Ctx(), "postgresql")
    my = build_sql_prompt("q", Ctx(), "mysql")
    assert "date_trunc" in pg
    assert "DATE_FORMAT" in my
    assert pg != my


# ── Answer grounding: no magnitude exemption for invented numbers ────────

def _ground(answer, rows, **kw):
    base = dict(question="What is total sales?", answer=answer,
                retrieved_tables=["sales"], retrieved_columns=["total"],
                sql="SELECT SUM(total) AS total FROM sales",
                rows=rows, columns=["total"], intent="data")
    base.update(kw)
    return check_answer_grounding(**base)


def test_grounding_flags_small_invented_number():
    out = _ground("Total sales is 999", [{"total": 50000}])
    assert out["verdict"] == "ungrounded"
    assert out["reason"] == "numeric_mismatch_vs_sql"
    assert 999 in out["checks"]["unsupported_claims"]
    assert out["checks"]["numeric_match_vs_sql"] is False


def test_grounding_accepts_small_number_present_in_rows():
    out = _ground("Total sales is 999", [{"total": 999}])
    assert out["verdict"] == "grounded"
    assert not out["checks"]["unsupported_claims"]


def test_grounding_allows_plain_prose_year():
    out = _ground("Total sales was 50,000 in 2024", [{"total": 50000}])
    assert out["verdict"] == "grounded"
    assert 2024 not in out["checks"]["unsupported_claims"]


def test_grounding_money_looking_like_year_is_not_a_year():
    out = _ground("Total sales was $1,999", [{"total": 50000}])
    assert out["verdict"] == "ungrounded"


def test_grounding_check4_never_auto_passes_on_invented_numbers():
    # Numbers are present, but none is backed by the rows: check 4 must not
    # short-circuit to True on "any number present".
    out = _ground("999", [{"total": 50000}])
    assert out["checks"]["answers_question"] is False
    assert out["verdict"] == "ungrounded"
    # The same shape with a row-backed number still auto-passes.
    ok = _ground("50000", [{"total": 50000}])
    assert ok["checks"]["answers_question"] is True
    assert ok["verdict"] == "grounded"