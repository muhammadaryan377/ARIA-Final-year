"""P0 data-foundation tests (n.txt items 1, 11, 12).

Hermetic: no live LLM or database required.

- Item 1: a single-table dataset is first-class through the SHARED path
  (PostgreSQL / MySQL / CSV-SQLite shapes): table name, columns+types, PK,
  nullable, categorical vs numerical vs datetime, measures, dimensions, date
  columns, row count, relationships == none (valid). The Goal Agent answers
  "What are my sales?" from that one table with NO JOIN.
- Item 11: the universal Data Intelligence Layer contract.
- Item 12: capability detection enables/disables analyses from actual data.
"""

import json
import os
import tempfile
import unittest

from core.data_intelligence import (
    build_data_intelligence,
    classify_column,
    detect_capabilities,
    profile_table,
)
from goal_agent import GoalAgent

DUMMY_URI = "postgresql://u:p@localhost:5432/nonexistent"


def write_mapping(mapping, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh)
    return path


def single_table_mapping(db_flavor="postgresql"):
    """One sales table; dtypes vary per flavor to prove the shared path."""
    dtype = {
        "postgresql": {"id": "integer", "amount": "numeric",
                       "sale_date": "timestamp without time zone",
                       "region": "character varying", "qty": "integer"},
        "mysql": {"id": "int", "amount": "decimal",
                  "sale_date": "datetime",
                  "region": "varchar", "qty": "int"},
        "sqlite": {"id": "INTEGER", "amount": "REAL",
                   "sale_date": "TEXT", "region": "TEXT", "qty": "INTEGER"},
    }[db_flavor]
    if db_flavor == "sqlite":
        # CSV/SQLite keeps ISO dates as TEXT; the name hint still classifies it.
        dtype = dict(dtype)
    return {
        "schema": "public",
        "tables": {
            "sales": {
                "columns": [
                    {"column": "id", "data_type": dtype["id"], "nullable": False},
                    {"column": "region", "data_type": dtype["region"], "nullable": True},
                    {"column": "amount", "data_type": dtype["amount"], "nullable": True},
                    {"column": "qty", "data_type": dtype["qty"], "nullable": True},
                    {"column": "sale_date", "data_type": dtype["sale_date"], "nullable": True},
                ],
                "primary_key": ["id"],
                "null_stats": {
                    "id": {"total_rows": 50, "nulls": 0, "null_pct": 0.0},
                    "region": {"total_rows": 50, "nulls": 5, "null_pct": 10.0},
                    "amount": {"total_rows": 50, "nulls": 0, "null_pct": 0.0},
                    "qty": {"total_rows": 50, "nulls": 2, "null_pct": 4.0},
                    "sale_date": {"total_rows": 50, "nulls": 0, "null_pct": 0.0},
                },
                "row_count": 50,
            }
        },
        "declared_relationships": [],
        "inferred_relationships": [],
        "relationship_edges": [],
        "ambiguous_relationships": [],
    }


def make_agent(mapping):
    tmpdir = tempfile.mkdtemp()
    schema_file = write_mapping(mapping, os.path.join(tmpdir, "schema.json"))
    return GoalAgent(schema_json_path=schema_file, db_uri=DUMMY_URI,
                     provider=object(), dialect="postgresql")


class TestSingleTableFirstClass(unittest.TestCase):
    def test_checklist_all_flavors(self):
        for flavor in ("postgresql", "mysql", "sqlite"):
            with self.subTest(flavor=flavor):
                profile = profile_table(
                    "sales", single_table_mapping(flavor)["tables"]["sales"])
                self.assertEqual(profile["table_name"], "sales")
                self.assertEqual(profile["column_count"], 5)
                self.assertEqual(profile["primary_key"], ["id"])
                self.assertIn("region", profile["nullable_columns"])
                self.assertNotIn("id", profile["nullable_columns"])
                self.assertIn("region", profile["categorical_columns"])
                self.assertIn("amount", profile["numerical_columns"])
                self.assertIn("qty", profile["numerical_columns"])
                self.assertIn("sale_date", profile["datetime_columns"])
                self.assertIn("sale_date", profile["date_columns"])
                self.assertIn("amount", profile["measures"])
                self.assertIn("region", profile["dimensions"])
                self.assertEqual(profile["row_count"], 50)

    def test_relationships_none_is_valid(self):
        intel = build_data_intelligence(single_table_mapping())
        self.assertTrue(intel["dataset_connected"])
        self.assertTrue(intel["schema_valid"])
        self.assertEqual(intel["table_count"], 1)
        self.assertEqual(intel["total_rows"], 50)
        self.assertTrue(intel["date_column_available"])
        self.assertTrue(intel["numeric_measures_available"])
        self.assertTrue(intel["categorical_dimensions_available"])
        self.assertTrue(intel["relationships"]["none"])
        self.assertEqual(intel["relationships"]["edges"], 0)

    def test_column_roles_shared(self):
        self.assertEqual(classify_column("customer_id", "integer"), "identifier")
        self.assertEqual(classify_column("amount", "numeric"), "numerical")
        self.assertEqual(classify_column("price", "REAL"), "numerical")
        self.assertEqual(classify_column("sale_date", "TEXT"), "datetime")
        self.assertEqual(classify_column("created_at", "timestamp"), "datetime")
        self.assertEqual(classify_column("region", "character varying"), "categorical")
        self.assertEqual(classify_column("is_active", "boolean"), "boolean")


class TestCapabilityDetection(unittest.TestCase):
    def test_single_table_capabilities(self):
        caps = detect_capabilities(build_data_intelligence(single_table_mapping()))
        got = caps["capabilities"]
        for key in ("kpis", "distributions", "categorical_analysis",
                    "numerical_analysis", "forecasting", "trends",
                    "product_analysis"):
            self.assertTrue(got[key]["enabled"], key)
        # No relationships exist: relational answers must disable, not error.
        self.assertFalse(got["relational_analysis"]["enabled"])
        self.assertFalse(got["cross_table_insights"]["enabled"])
        self.assertIn("forecasting", caps["suitable_analyses"])

    def test_multi_table_relational_enabled(self):
        mapping = single_table_mapping()
        mapping["tables"]["customers"] = {
            "columns": [
                {"column": "id", "data_type": "integer", "nullable": False},
                {"column": "name", "data_type": "text", "nullable": True},
            ],
            "primary_key": ["id"],
            "null_stats": {"id": {"total_rows": 10, "nulls": 0, "null_pct": 0.0}},
            "row_count": 10,
        }
        mapping["relationship_edges"] = [{
            "source_table": "sales", "source_column": "id",
            "target_table": "customers", "target_column": "id",
            "type": "declared",
        }]
        caps = detect_capabilities(build_data_intelligence(mapping))
        self.assertTrue(caps["capabilities"]["relational_analysis"]["enabled"])
        self.assertTrue(caps["capabilities"]["cross_table_insights"]["enabled"])

    def test_no_date_no_forecast(self):
        mapping = single_table_mapping()
        cols = mapping["tables"]["sales"]["columns"]
        mapping["tables"]["sales"]["columns"] = [
            c for c in cols if c["column"] != "sale_date"]
        caps = detect_capabilities(build_data_intelligence(mapping))
        self.assertFalse(caps["capabilities"]["forecasting"]["enabled"])
        self.assertTrue(caps["capabilities"]["kpis"]["enabled"])


class TestSingleTableNoJoin(unittest.TestCase):
    def test_sales_question_needs_no_join(self):
        agent = make_agent(single_table_mapping())
        needs, _ = agent._needs_clarification("What are my sales?")
        self.assertFalse(needs)
        relevant = agent._get_relevant_tables("What are my sales?")
        self.assertEqual(relevant, ["sales"])
        join_path = agent._determine_join_path(relevant)
        self.assertEqual(join_path, ["sales"])
        plan = agent._plan_joins(
            agent._required_tables_from_ir(None, join_path))
        self.assertEqual(plan["nodes"], ["sales"])
        self.assertEqual(plan["edges"], [])

    def test_disconnected_tables_never_forced_into_path(self):
        mapping = single_table_mapping()
        mapping["tables"]["customers"] = {
            "columns": [{"column": "id", "data_type": "integer", "nullable": False}],
            "primary_key": ["id"],
        }
        agent = make_agent(mapping)
        path = agent._determine_join_path(["sales", "customers"])
        # No FK between them -> the unconnected table must not be forced in.
        self.assertEqual(path, ["sales"])

    def test_agent_exposes_intelligence(self):
        agent = make_agent(single_table_mapping())
        self.assertIsInstance(agent.data_intelligence, dict)
        self.assertTrue(agent.data_intelligence.get("dataset_connected"))
        self.assertIn("forecasting",
                      agent.capabilities.get("suitable_analyses", []))


class TestCsvMappingIntelligence(unittest.TestCase):
    def test_csv_mapping_carries_intelligence(self):
        import csv as _csv
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "sales.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = _csv.writer(fh)
                writer.writerow(["id", "region", "amount", "sale_date"])
                writer.writerow(["1", "north", "10.5", "2024-01-15"])
                writer.writerow(["2", "south", "20", "2024-02-15"])
            from csv_orchestrator import build_csv_schema_mapping
            mapping = build_csv_schema_mapping(csv_path)
            self.assertIn("data_intelligence", mapping)
            self.assertIn("capabilities", mapping)
            intel = mapping["data_intelligence"]
            self.assertTrue(intel["dataset_connected"])
            self.assertEqual(intel["table_count"], 1)
            self.assertTrue(intel["relationships"]["none"])


if __name__ == "__main__":
    unittest.main()
