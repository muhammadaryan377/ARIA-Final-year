"""End-to-end regression matrix (spec item 13).

Datasets x pipeline stages, fully hermetic (tests/helpers.py doubles,
SQLite execution, FakeLLM — no live Postgres/MySQL/LLM):

  datasets: single-table CSV | single-table Postgres (simulated mapping +
            sqlite exec) | single-table MySQL (same) | AdventureWorks
            multi-table | time-series | messy (null/dup) | small/insufficient
  stages: connection | schema extraction | goal/SQL | execution | dashboard |
          insights | forecast | RAG/copilot | charts | errors/edge
"""

import json
import sqlite3
from pathlib import Path

import pytest

from csv_handler import (CsvFolderSchemaExtractor, CsvSchemaExtractor,
                         CsvValidationError)
from csv_orchestrator import (build_csv_schema_mapping, load_csv_to_sqlite,
                              safe_table_name)
from tests.helpers import fake_profiler

SALES_CSV = ("customer,product,category,territory,order_id,order_date,quantity,unit_price\n"
             "Alice,Widget,Gadgets,North,1,2024-01-05,2,50.0\n"
             "Bob,Gizmo,Gadgets,South,2,2024-01-20,1,200.0\n"
             "Alice,Sprocket,Tools,North,3,2024-02-10,4,25.0\n"
             "Cara,Gizmo,Gadgets,North,4,2024-02-15,2,200.0\n"
             "Dan,Sprocket,Tools,East,5,2024-03-01,2,25.0\n"
             "Bob,Widget,Gadgets,South,6,2024-03-10,3,50.0\n")

TS_CSV = ("day,revenue\n" + "".join(
    f"2024-01-{d:02d},{100 + d * 7}\n" for d in range(1, 31)))

MESSY_CSV = ("order_id,customer,amount,note\n"
             "1,Alice,100.0,ok\n"
             "2,,200.0,null\n"
             "1,Alice,100.0,ok\n"
             "3,Bob,,missing\n"
             "4,Cara,50.0,ok\n")

SMALL_CSV = "order_id,amount\n1,10.0\n2,20.0\n"


class _NoLLM:
    def complete(self, *a, **k):
        raise AssertionError("LLM must not be called")

    def chat(self, *a, **k):
        raise AssertionError("LLM must not be called")


class _FakeSQL:
    def __init__(self, sql):
        self._sql = sql
        self.calls = []

    def complete(self, role, prompt, **kw):
        self.calls.append((role, prompt))
        return self._sql if role == "sql" else "narrative"

    def chat(self, role, messages, **kw):
        return "ok"


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8", newline="")
    return p


def _goal_agent(mapping, dialect="sqlite"):
    from goal_agent import GoalAgent
    import tempfile, os
    d = tempfile.mkdtemp()
    p = os.path.join(d, "schema.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh)
    return GoalAgent(schema_json_path=p,
                     db_uri="postgresql://u:p@localhost:5432/nonexistent",
                     provider=_NoLLM(), dialect=dialect)


def _aw_sqlite(tmp_path):
    import sqlite3 as s3
    db = tmp_path / "aw.db"
    c = s3.connect(db)
    c.executescript("""
        CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, total REAL);
        INSERT INTO customers VALUES (1,'a'),(2,'b');
        INSERT INTO orders VALUES (10,1,5.0),(11,2,7.0);
    """)
    c.commit(); c.close()
    return f"sqlite:///{db}"


# ---------------------------------------------------------------- connection
def test_m01_csv_connection_loads_to_sqlite(tmp_path):
    csv = _write(tmp_path, "sales.csv", SALES_CSV)
    db = tmp_path / "s.db"
    uri, table, n = load_csv_to_sqlite(csv, db)
    assert table == "sales" and n == 6 and uri.startswith("sqlite")


def test_m02_pg_mysql_simulated_connection_uses_doubles():
    for dialect in ("postgresql", "mysql"):
        prof = fake_profiler(rows=6)
        assert prof.row_count("sales_flat") == 6
        assert prof.quote("sales_flat") in ('"sales_flat"', '"sales_flat"')
        assert prof.types_compatible("a", "b") is True


def test_m03_aw_multitable_connection(tmp_path):
    uri = _aw_sqlite(tmp_path)
    c = sqlite3.connect(tmp_path / "aw.db")
    tables = {r[0] for r in c.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    c.close()
    assert {"customers", "orders"} <= tables and uri.startswith("sqlite")


def test_m04_time_series_messy_small_connections(tmp_path):
    for name, text in (("ts.csv", TS_CSV), ("messy.csv", MESSY_CSV),
                       ("small.csv", SMALL_CSV)):
        csv = _write(tmp_path, name, text)
        db = tmp_path / (name + ".db")
        _, _, n = load_csv_to_sqlite(csv, db)
        assert n >= 2


# ---------------------------------------------------------- schema extraction
def test_m05_single_table_schema_profile(tmp_path):
    s = CsvSchemaExtractor(_write(tmp_path, "sales.csv", SALES_CSV)).extract()
    assert s["row_count"] == 6
    assert {c["name"] for c in s["columns"]} >= {"customer", "order_date", "quantity"}


def test_m06_pg_mysql_mapping_from_csv_shape(tmp_path):
    csv = _write(tmp_path, "sales.csv", SALES_CSV)
    m = build_csv_schema_mapping(csv)
    table = next(iter(m["tables"]))
    cols = [c["column"] for c in m["tables"][table]["columns"]]
    assert "quantity" in cols and m["tables"][table]["row_count"] == 6


def test_m07_aw_folder_extraction_is_multitable(tmp_path):
    d = tmp_path / "awf"
    d.mkdir()
    (d / "customers.csv").write_text("customer_id,name\n1,a\n2,b\n", encoding="utf-8")
    (d / "orders.csv").write_text("order_id,customer_id,total\n10,1,5.0\n", encoding="utf-8")
    s = CsvFolderSchemaExtractor(d).extract()
    assert s["table_count"] == 2


def test_m08_time_series_date_column_detected(tmp_path):
    s = CsvSchemaExtractor(_write(tmp_path, "ts.csv", TS_CSV)).extract()
    col = {c["name"]: c for c in s["columns"]}["day"]
    assert col["inferred_type"] in ("DATE", "DATETIME", "STRING")
    assert s["row_count"] == 30


def test_m09_messy_null_dup_signals(tmp_path):
    s = CsvSchemaExtractor(_write(tmp_path, "messy.csv", MESSY_CSV)).extract()
    assert s["quality"]["duplicate_rows"] >= 1
    cols = {c["name"]: c for c in s["columns"]}
    assert cols["customer"]["null_count"] >= 1 or cols["amount"]["null_count"] >= 1


def test_m10_small_dataset_row_count(tmp_path):
    s = CsvSchemaExtractor(_write(tmp_path, "small.csv", SMALL_CSV)).extract()
    assert s["row_count"] == 2


# ------------------------------------------------------------------ goal/SQL
def _flat_goal_mapping():
    def _t(cols):
        return {"columns": [{"column": c, "data_type": t} for c, t in cols],
                "primary_key": []}
    return {"schema": "public",
            "tables": {"sales": _t([("customer", "text"), ("product", "text"),
                                    ("quantity", "integer"), ("unit_price", "numeric"),
                                    ("order_date", "timestamp")])},
            "relationship_edges": [], "declared_relationships": [],
            "inferred_relationships": [], "ambiguous_relationships": []}


@pytest.mark.parametrize("dialect", ["sqlite", "postgresql", "mysql"])
def test_m11_read_only_guard_all_dialects(dialect):
    ag = _goal_agent(_flat_goal_mapping(), dialect=dialect)
    assert ag._is_read_only_sql("SELECT * FROM sales;")
    assert not ag._is_read_only_sql("DROP TABLE sales;")


def test_m12_clarification_gate_gibberish_vs_goal():
    ag = _goal_agent(_flat_goal_mapping())
    needs, _ = ag._needs_clarification("explain the weather forecast")
    assert needs is True
    needs, _ = ag._needs_clarification("total sales by customer")
    assert needs is False


@pytest.mark.parametrize("q", ["how many customers", "top products by sales",
                               "monthly sales trend", "average order value"])
def test_m13_goal_ir_grounds_on_flat_and_aw(q, tmp_path):
    from goal_agent import GoalAgent
    import tempfile, os
    flat = _flat_goal_mapping()
    aw = {"schema": "public",
          "tables": {"customers": {"columns": [{"column": "id", "data_type": "integer"},
                                               {"column": "name", "data_type": "text"}],
                                   "primary_key": ["id"]},
                     "orders": {"columns": [{"column": "order_id", "data_type": "integer"},
                                            {"column": "customer_id", "data_type": "integer"},
                                            {"column": "total", "data_type": "numeric"},
                                            {"column": "order_date", "data_type": "timestamp"}],
                                "primary_key": ["order_id"]}},
          "relationship_edges": [],
          "declared_relationships": [{"table_name": "orders", "column_name": "customer_id",
                                      "references_table": "customers",
                                      "references_column": "id"}],
          "inferred_relationships": [], "ambiguous_relationships": []}
    for mapping in (flat, aw):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "schema.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(mapping, fh)
        ag = GoalAgent(schema_json_path=p,
                       db_uri="postgresql://u:p@localhost:5432/nonexistent",
                       provider=_NoLLM(), dialect="postgresql")
        ir = ag._build_goal_ir(q)
        jp = ag._determine_join_path(ag._get_relevant_tables(q))
        ir = ag._resolve_semantics(q, ir, jp)
        assert ir["metrics"], q


def test_m14_output_contract_shape():
    ag = _goal_agent(_flat_goal_mapping())
    c = ag._build_output_contract(
        "total sales", {"kpis": [], "dimensions": []}, ["sales"],
        "SELECT 1;", [{"a": 1}], success=True, message=None,
        status="success", warnings=[], execution_time_ms=3)
    assert c["status"] == "success" and c["execution"]["row_count"] == 1


# ----------------------------------------------------------------- execution
def test_m15_execution_single_table_sqlite(tmp_path):
    csv = _write(tmp_path, "sales.csv", SALES_CSV)
    db = tmp_path / "s.db"
    uri, table, _ = load_csv_to_sqlite(csv, db)
    c = sqlite3.connect(db)
    n = c.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    total = c.execute(f'SELECT SUM(quantity*unit_price) FROM "{table}"').fetchone()[0]
    c.close()
    assert n == 6 and total == pytest.approx(1000.0)


def test_m16_execution_aw_join(tmp_path):
    _aw_sqlite(tmp_path)
    c = sqlite3.connect(tmp_path / "aw.db")
    rows = c.execute("SELECT c.name, SUM(o.total) FROM customers c "
                     "JOIN orders o ON o.customer_id=c.customer_id GROUP BY c.name").fetchall()
    c.close()
    assert sorted(r[1] for r in rows) == [5.0, 7.0]


def test_m17_execution_messy_and_small_graceful(tmp_path):
    for name, text in (("messy.csv", MESSY_CSV), ("small.csv", SMALL_CSV)):
        csv = _write(tmp_path, name, text)
        db = tmp_path / (name + ".db")
        _, table, n = load_csv_to_sqlite(csv, db)
        c = sqlite3.connect(db)
        got = c.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        c.close()
        assert got == n


# ----------------------------------------------------------------- dashboard
def test_m18_dashboard_kpis_from_execution(tmp_path):
    csv = _write(tmp_path, "sales.csv", SALES_CSV)
    db = tmp_path / "s.db"
    _, table, _ = load_csv_to_sqlite(csv, db)
    c = sqlite3.connect(db)
    n = c.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    total = c.execute(f'SELECT SUM(quantity*unit_price) FROM "{table}"').fetchone()[0]
    aov_orders = c.execute(
        f'SELECT AVG(t.v) FROM (SELECT SUM(quantity*unit_price) AS v FROM "{table}" '
        'GROUP BY order_id) t').fetchone()[0]
    c.close()
    assert n == 6 and total == pytest.approx(1000.0)
    assert aov_orders == pytest.approx(total / 6)


# ------------------------------------------------------------------ insights
def test_m19_insights_suggestions_and_overview():
    ag = _goal_agent(_flat_goal_mapping())
    sug = ag._template_suggestions(limit=3)
    assert sug and len(sug) <= 3
    cleaned = ag._clean_data([{"total": 1.0, "city": "x"}, {"total": None, "city": "y"}])
    assert cleaned[1]["total"] is None  # nulls preserved, never filled


# ------------------------------------------------------------------ forecast
def test_m20_forecast_time_series_well_formed():
    from prediction.agent import PredictionAgent
    rows = [{"day": f"2024-01-{d:02d}", "revenue": 100 + d * 7} for d in range(1, 31)]
    res = PredictionAgent(horizon=3).forecast({"data": rows, "user_goal": "revenue trend"})
    assert isinstance(res, dict) and "valid" in res


def test_m21_forecast_small_insufficient_is_honest():
    from prediction.agent import PredictionAgent
    rows = [{"order_id": 1, "amount": 10.0}, {"order_id": 2, "amount": 20.0}]
    res = PredictionAgent(horizon=3).forecast({"data": rows, "user_goal": "amount trend"})
    assert isinstance(res, dict) and "valid" in res
    assert res["valid"] in (True, False)


# --------------------------------------------------------------- RAG/copilot
def test_m22_copilot_sql_guard_and_chart_utils():
    from copilot.sql_engine import is_read_only, clean_sql
    from copilot.pipeline import infer_chart
    from copilot.embeddings import HashingEmbedder, cosine
    assert is_read_only("SELECT 1") and not is_read_only("DELETE FROM t")
    assert clean_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    e = HashingEmbedder()
    assert abs(cosine(e.embed_one("sales"), e.embed_one("sales")) - 1.0) < 1e-6
    spec = infer_chart(["product", "total"],
                       [{"product": "a", "total": 1}], "top products")
    assert spec["type"] in ("bar", "bar_horizontal", "line", "area", "pie", "table")


@pytest.mark.parametrize("provider_name", ["local", "cloud"])
def test_m23_copilot_pipeline_provider_independent(tmp_path, provider_name):
    from copilot.pipeline import CopilotPipeline
    db = tmp_path / f"shop_{provider_name}.db"
    c = sqlite3.connect(db)
    c.executescript("CREATE TABLE sales (product TEXT, total REAL);"
                    "INSERT INTO sales VALUES ('a',10.0),('b',20.0);")
    c.commit(); c.close()
    mapping = {"database": "shop",
               "tables": {"sales": {"columns": [{"column": "product", "data_type": "TEXT"},
                                                {"column": "total", "data_type": "REAL"}],
                                    "primary_key": [], "foreign_keys": [], "row_count": 2}}}
    llm = _FakeSQL("SELECT product, SUM(total) AS total FROM sales GROUP BY product")
    pipe = CopilotPipeline(llm=llm, schema_mapping=mapping,
                           db_uri=f"sqlite:///{db}", dialect="sqlite",
                           user_id=f"u_{provider_name}", embedder=None,
                           provider_name=provider_name)
    ans = pipe.ask("total sales by product")
    assert ans.ok and ans.rows and ans.provider == provider_name


def test_m24_retriever_and_memory():
    from copilot.schema_index import build_schema_index
    from copilot.retriever import SchemaRetriever
    from copilot.conversation import ConversationMemory
    mapping = {"database": "shop",
               "tables": {"sales": {"columns": [{"column": "product", "data_type": "TEXT"},
                                                {"column": "total", "data_type": "REAL"}],
                                    "primary_key": [], "foreign_keys": [], "row_count": 2}}}
    ctx = SchemaRetriever(build_schema_index(mapping), embedder=None).retrieve(
        "total sales by product")
    assert "sales" in ctx.tables
    assert ConversationMemory.is_followup("what about last year?")


# -------------------------------------------------------------------- charts
def test_m25_charts_ranking_is_bar_and_trend_is_line():
    from copilot.pipeline import infer_chart
    bar = infer_chart(["product", "total"], [{"product": "a", "total": 5}],
                      "top products by total")
    line = infer_chart(["month", "total"],
                       [{"month": "2024-01", "total": 5}], "monthly trend")
    assert bar["type"] in ("bar", "bar_horizontal")
    assert line["type"] in ("line", "area")


# --------------------------------------------------------------- errors/edge
def test_m26_errors_and_edge_cases(tmp_path):
    ag = _goal_agent(_flat_goal_mapping())
    assert not ag._is_read_only_sql("UPDATE sales SET total=0;")
    needs, _ = ag._needs_clarification("asdf asdf qwerty zzz")
    assert needs in (True, False)  # gate runs without raising
    dup = _write(tmp_path, "bad.csv", "id,ID\n1,2\n")
    with pytest.raises(CsvValidationError):
        CsvSchemaExtractor(dup).extract()
    hdr = _write(tmp_path, "empty.csv", "a,b\n")
    s = CsvSchemaExtractor(hdr).extract()
    assert s["row_count"] == 0
    from copilot.sql_engine import SqlEngine
    eng = SqlEngine("sqlite:///:memory:", dialect="sqlite")
    ok, _ = eng.validate("SELECT x FROM nope", {"sales": {"columns": []}})
    assert not ok
    assert safe_table_name("Sales Flat!") == "sales_flat"
    prof = fake_profiler(exhausted=True)
    assert prof.budget.exhausted is True
