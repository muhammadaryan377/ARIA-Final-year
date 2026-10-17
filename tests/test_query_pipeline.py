"""Query-generation pipeline tests (n.txt item 2).

Hermetic (sqlite + FakeLLM, no network): covers the post-generation gate —
nonexistent column (deterministic repair), invalid JOIN (LLM repair), bad
aggregation (nested / incompatible datatype) — and the never-blind UI rule
(result validation refuses broken frames).
"""

import json
import os
import tempfile
import unittest

from sqlalchemy import create_engine, text

from core.validation import (
    find_incompatible_aggregations,
    find_nested_aggregates,
    is_bare_select_star,
    validate_generated_sql,
)
from goal_agent import GoalAgent

DUMMY_URI = "postgresql://u:p@localhost:5432/nonexistent"


def write_mapping(mapping, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh)
    return path


def shop_mapping():
    return {
        "schema": "public",
        "tables": {
            "customers": {
                "columns": [
                    {"column": "id", "data_type": "integer"},
                    {"column": "name", "data_type": "text"},
                ],
                "primary_key": ["id"],
            },
            "orders": {
                "columns": [
                    {"column": "order_id", "data_type": "integer"},
                    {"column": "customer_id", "data_type": "integer"},
                    {"column": "total", "data_type": "numeric"},
                ],
                "primary_key": ["order_id"],
            },
            "order_details": {
                "columns": [
                    {"column": "order_id", "data_type": "integer"},
                    {"column": "quantity", "data_type": "integer"},
                ],
                "primary_key": ["order_id"],
            },
        },
        "relationship_edges": [],
        "declared_relationships": [
            {"table_name": "orders", "column_name": "customer_id",
             "references_table": "customers", "references_column": "id"},
            {"table_name": "order_details", "column_name": "order_id",
             "references_table": "orders", "references_column": "order_id"},
        ],
        "inferred_relationships": [],
        "ambiguous_relationships": [],
    }


def make_agent(mapping, tmpdir, provider=None, db_uri=None):
    schema_file = write_mapping(mapping, os.path.join(tmpdir, "schema.json"))
    return GoalAgent(
        schema_json_path=schema_file,
        db_uri=db_uri or DUMMY_URI,
        provider=provider or object(),
        dialect="sqlite" if db_uri else "postgresql",
    )


class NoLLM:
    def complete(self, *a, **k):
        raise AssertionError("LLM must not be called on this path")

    def chat(self, *a, **k):
        raise AssertionError("LLM must not be called on this path")


class RepairLLM:
    """Returns fixed SQL for the invalid-JOIN case, records repair prompts."""

    def __init__(self, fixed_sql):
        self.fixed_sql = fixed_sql
        self.calls = 0

    def complete(self, role, prompt, **kw):
        self.calls += 1
        return self.fixed_sql


class TestPipelineValidationHelpers(unittest.TestCase):
    def test_nested_aggregate_flagged(self):
        self.assertTrue(find_nested_aggregates("SELECT AVG(SUM(total)) FROM orders;"))

    def test_bare_select_star_flagged(self):
        self.assertTrue(is_bare_select_star("SELECT * FROM orders;"))
        self.assertFalse(is_bare_select_star("SELECT SUM(total) FROM orders GROUP BY x;"))

    def test_incompatible_datatype_sum_over_text(self):
        types = {"customers": {"name": "text", "id": "integer"}}
        issues = find_incompatible_aggregations(
            "SELECT SUM(customers.name) FROM customers;", types)
        self.assertTrue(any("incompatible" in i for i in issues), issues)
        ok = find_incompatible_aggregations(
            "SELECT COUNT(*) FROM customers;", types)
        self.assertEqual(ok, [])

    def test_validate_generated_sql_catches_unknown_table_and_column(self):
        cols = {"orders": ["order_id", "total"]}
        issues = validate_generated_sql(
            "SELECT orders.bogus FROM order_items;",
            schema_tables=["orders"], columns_by_table=cols)
        self.assertTrue(any("nonexistent table" in i for i in issues), issues)
        self.assertTrue(any("nonexistent column" in i for i in issues), issues)

    def test_agent_quote_escapes_identifiers(self):
        with tempfile.TemporaryDirectory() as d:
            ag = make_agent(shop_mapping(), d)
            self.assertEqual(ag._quote('orders'), '"orders"')
            self.assertEqual(ag._quote('a"b'), '"a""b"')


class TestInvalidColumnRepair(unittest.TestCase):
    """`orders.quantity` is hallucinated (quantity lives on order_details):
    the deterministic pass must rewrite + inject the FK JOIN with no LLM."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = os.path.join(self._tmp.name, "t.db")
        eng = create_engine(f"sqlite:///{self.db}")
        with eng.begin() as conn:
            conn.execute(text("CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, total REAL)"))
            conn.execute(text("CREATE TABLE order_details (order_id INTEGER PRIMARY KEY, quantity INTEGER)"))
            conn.execute(text("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)"))
            conn.execute(text("INSERT INTO orders VALUES (1, 1, 10.0)"))
            conn.execute(text("INSERT INTO order_details VALUES (1, 3)"))
            conn.execute(text("INSERT INTO customers VALUES (1, 'a')"))
        eng.dispose()
        self.agent = make_agent(shop_mapping(), self._tmp.name,
                                provider=NoLLM(), db_uri=f"sqlite:///{self.db}")

    def tearDown(self):
        try:
            self.agent.engine.dispose()
        except Exception:
            pass

    def test_invalid_column_repaired_deterministically(self):
        bad = "SELECT orders.quantity FROM orders;"
        rows, final_sql, note = self.agent._execute_with_retry(
            "show quantities", ["orders"], {"kpis": [], "dimensions": []}, bad)
        self.assertIn("order_details", final_sql)
        self.assertNotIn("orders.quantity", final_sql)
        self.assertTrue(rows)
        self.assertEqual(rows[0].get("quantity"), 3.0)


class TestInvalidJoinRepair(unittest.TestCase):
    """PK=PK join (customers.id = orders.order_id) must be repaired via LLM
    into the real FK join; the broken SQL is never returned as-is."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = os.path.join(self._tmp.name, "t.db")
        eng = create_engine(f"sqlite:///{self.db}")
        with eng.begin() as conn:
            conn.execute(text("CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, total REAL)"))
            conn.execute(text("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)"))
            conn.execute(text("INSERT INTO orders VALUES (1, 1, 10.0)"))
            conn.execute(text("INSERT INTO customers VALUES (1, 'a')"))
        eng.dispose()
        fixed = ("SELECT c.name, SUM(o.total) AS v FROM orders o "
                 "JOIN customers c ON o.customer_id = c.id GROUP BY c.name;")
        self.llm = RepairLLM(fixed)
        self.agent = make_agent(shop_mapping(), self._tmp.name,
                                provider=self.llm, db_uri=f"sqlite:///{self.db}")

    def tearDown(self):
        try:
            self.agent.engine.dispose()
        except Exception:
            pass

    def test_invalid_join_repaired_not_passed_through(self):
        bad = ("SELECT c.name FROM orders o JOIN customers c "
               "ON o.order_id = c.id;")
        # Sanity: the bad join is actually detected as suspicious.
        self.assertTrue(self.agent._find_suspicious_pk_joins(bad))
        rows, final_sql, note = self.agent._execute_with_retry(
            "total sales by customer", ["orders", "customers"],
            {"kpis": [], "dimensions": []}, bad)
        self.assertGreaterEqual(self.llm.calls, 1)
        self.assertIn("customer_id", final_sql)
        self.assertNotIn("o.order_id = c.id", final_sql)
        self.assertTrue(rows)


class TestResultValidation(unittest.TestCase):
    def test_broken_frame_refused(self):
        with tempfile.TemporaryDirectory() as d:
            ag = make_agent(shop_mapping(), d)
            with self.assertRaises(RuntimeError):
                ag._validate_result_shape("q", "SELECT 1;", [{"a": None}], [])

    def test_empty_columns_refused(self):
        with tempfile.TemporaryDirectory() as d:
            ag = make_agent(shop_mapping(), d)
            with self.assertRaises(RuntimeError):
                ag._validate_result_shape("q", "SELECT 1;", [{"a": 1}], [])


if __name__ == "__main__":
    unittest.main()
