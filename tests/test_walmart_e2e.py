"""Walmart end-to-end test on the real Kaggle retail CSV (hermetic).

Dataset: data/kaggle_retail_sales/mikhail1681__walmart-sales/Walmart_Sales.csv
  6435 rows, single table ``walmart``:
  Store, Date (TEXT DD-MM-YYYY), Weekly_Sales, Holiday_Flag,
  Temperature, Fuel_Price, CPI, Unemployment.

Hermetic: real CSV file -> temp SQLite via csv_orchestrator (no live DB,
no network, no LLM). Ground truth is computed with pandas/SQLite directly;
the Goal Agent, KPI helpers, PredictionAgent and Copilot RAG pipeline are
exercised against the SAME single table to prove the 8 conceptual questions
work with zero JOINs.

The 8 conceptual questions:
  Q1 total sales                       (SUM Weekly_Sales)
  Q2 sales by store                    (GROUP BY Store)
  Q3 monthly trend                     (strftime %Y-%m on normalized Date)
  Q4 holiday vs non-holiday            (GROUP BY Holiday_Flag)
  Q5 average weekly sales              (AVG Weekly_Sales)
  Q6 KPI values match SQL              (profile/score vs direct SQL)
  Q7 forecast needs date + measure     (PredictionAgent contract)
  Q8 RAG grounding is SQL-authoritative (answer_grounding verdicts)

Plus structural coverage: connection/shape, PK inference (Store+Date
composite or graceful none), date-text -> datetime detection.
"""

import os
import re
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
WALMART_CSV = (
    REPO / "data" / "kaggle_retail_sales"
    / "mikhail1681__walmart-sales" / "Walmart_Sales.csv"
)

EXPECTED_COLUMNS = ["Store", "Date", "Weekly_Sales", "Holiday_Flag",
                    "Temperature", "Fuel_Price", "CPI", "Unemployment"]

HAVE_DATASET = WALMART_CSV.exists()


def _iso_date_re():
    return re.compile(r"^\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2})?$")


class FakeSQLLLM:
    """Deterministic LLM: canned SQL for role='sql', short prose otherwise."""

    def __init__(self, sql: str):
        self._sql = sql
        self.calls: list = []

    def complete(self, role, prompt, **kw):
        self.calls.append((role, prompt))
        if role == "sql":
            return self._sql
        return "Business answer from fake LLM."

    def chat(self, role, messages, **kw):
        return "ok"


@unittest.skipUnless(HAVE_DATASET, f"Walmart CSV not found: {WALMART_CSV}")
class WalmartE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from csv_orchestrator import (
            build_csv_schema_mapping,
            load_csv_to_sqlite,
        )

        cls._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.workdir = Path(cls._tmp.name)
        # Structural mapping (Goal-Agent compatible) + SQLite load.
        cls.mapping = build_csv_schema_mapping(WALMART_CSV, table_name="walmart")
        cls.db_uri, cls.table, cls.row_count = load_csv_to_sqlite(
            WALMART_CSV, cls.workdir / "walmart.db", table_name="walmart"
        )
        cls.db_path = str(cls.workdir / "walmart.db")
        # Ground truth straight from the CSV (independent of the pipeline).
        cls.frame = pd.read_csv(WALMART_CSV, dtype=str, keep_default_na=False)
        cls.sales = pd.to_numeric(cls.frame["Weekly_Sales"], errors="coerce")
        # Normalized Date column as stored in SQLite.
        conn = sqlite3.connect(cls.db_path)
        try:
            cls.sqlite_dates = [
                r[0] for r in conn.execute('SELECT "Date" FROM "walmart"').fetchall()
            ]
            cls.sqlite_rows = conn.execute('SELECT COUNT(*) FROM "walmart"').fetchone()[0]
        finally:
            conn.close()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # -- helpers ------------------------------------------------------
    def _query(self, sql, params=()):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(sql, params)
            cols = [d[0] for d in cur.description] if cur.description else []
            return cols, [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

    def _goal_agent(self, provider=None):
        import json

        from goal_agent import GoalAgent

        schema_file = self.workdir / f"schema_{os.getpid()}.json"
        if not schema_file.exists():
            schema_file.write_text(json.dumps(self.mapping), encoding="utf-8")
        return GoalAgent(
            schema_json_path=str(schema_file),
            db_uri=f"sqlite:///{self.db_path}",
            provider=provider or FakeSQLLLM("SELECT 1"),
            dialect="sqlite",
        )

    def _copilot(self, sql, user="u_walmart"):
        from copilot.pipeline import CopilotPipeline

        return CopilotPipeline(
            llm=FakeSQLLLM(sql),
            schema_mapping=self.mapping,
            db_uri=f"sqlite:///{self.db_path}",
            dialect="sqlite",
            user_id=user,
            embedder=None,
            provider_name="local",
        )

    # -- 1. connection / shape ---------------------------------------
    def test_01_connection_shape(self):
        self.assertEqual(self.table, "walmart")
        self.assertEqual(self.row_count, 6435)
        self.assertEqual(self.sqlite_rows, 6435)
        self.assertIn("walmart", self.mapping["tables"])
        info = self.mapping["tables"]["walmart"]
        cols = [c["column"] for c in info["columns"]]
        self.assertEqual(cols, EXPECTED_COLUMNS)
        self.assertEqual(info["row_count"], 6435)
        self.assertFalse(info.get("empty", True))
        # SQLite really holds all 8 columns with numeric measures typed.
        conn = sqlite3.connect(self.db_path)
        try:
            pragma = conn.execute('PRAGMA table_info("walmart")').fetchall()
        finally:
            conn.close()
        self.assertEqual(len(pragma), 8)
        by_name = {r[1]: r[2] for r in pragma}
        self.assertEqual(by_name["Weekly_Sales"], "REAL")
        self.assertEqual(by_name["Store"], "INTEGER")

    # -- 2. PK inference ----------------------------------------------
    def test_02_pk_inference_composite_or_graceful_none(self):
        """No single column is a valid PK; (Store, Date) is unique.

        The CSV handler only suggests single-column candidate keys, so the
        honest outcome is inferred_primary_key == [] (graceful none) while
        the composite (Store, Date) covers all 6435 rows uniquely.
        """
        info = self.mapping["tables"]["walmart"]
        self.assertEqual(info.get("primary_key", []), [])
        # Graceful none: no single-column key is forced.
        self.assertEqual(info.get("inferred_primary_key", []), [])
        _, dup = self._query(
            'SELECT COUNT(*) AS n, COUNT(DISTINCT "Store" || \'|\' || "Date") AS u '
            'FROM "walmart"'
        )
        self.assertEqual(dup[0]["n"], 6435)
        self.assertEqual(dup[0]["u"], 6435)
        # Neither member alone is unique (Store ~143 rows each, Date repeats).
        _, store_u = self._query(
            'SELECT COUNT(DISTINCT "Store") AS s, COUNT(DISTINCT "Date") AS d '
            'FROM "walmart"'
        )
        self.assertEqual(store_u[0]["s"], 45)
        self.assertLess(store_u[0]["d"], 6435)

    # -- 3. date-text -> datetime detection ----------------------------
    def test_03_date_text_to_datetime_detection(self):
        """Date arrives as DD-MM-YYYY text; SQLite must hold ISO dates."""
        from csv_handler import CsvSchemaExtractor

        profile = CsvSchemaExtractor(WALMART_CSV).extract()
        by_name = {c["name"]: c for c in profile["columns"]}
        # Conservative inference: non-ISO text stays STRING, never guessed.
        self.assertEqual(by_name["Date"]["inferred_type"], "STRING")
        # But the orchestrator normalizes DD-MM-YYYY -> ISO YYYY-MM-DD so
        # time SQL works. Every stored value must be ISO-parseable.
        iso = _iso_date_re()
        bad = [v for v in self.sqlite_dates if not iso.match(str(v).strip())]
        self.assertEqual(bad, [], f"non-ISO dates remain: {bad[:3]}")
        parsed = pd.to_datetime(pd.Series(self.sqlite_dates), errors="coerce")
        self.assertGreaterEqual(parsed.notna().mean(), 0.99)
        # SchemaIntelligence falls back to date-like TEXT names.
        from core.schema_intel import SchemaIntelligence

        intel = SchemaIntelligence(self.mapping)
        self.assertEqual(intel.find_best_datetime("walmart"), "Date")
        # PredictionAgent sees a real datetime axis (not a fake sequence).
        from prediction.agent import _datetime_cols

        df = pd.DataFrame({"Date": list(self.sqlite_dates),
                           "Weekly_Sales": self.sales.tolist()})
        self.assertIn("Date", _datetime_cols(df))

    # -- Q1 total sales ------------------------------------------------
    def test_04_q1_total_sales_no_join(self):
        from copilot.sql_engine import SqlEngine

        sql = 'SELECT SUM("Weekly_Sales") AS total FROM "walmart"'
        self.assertNotIn("JOIN", sql.upper())
        engine = SqlEngine(f"sqlite:///{self.db_path}", dialect="sqlite")
        res = engine.execute(sql, self.mapping["tables"])
        self.assertTrue(res.ok, res.error)
        expected = float(self.sales.sum())
        self.assertAlmostEqual(float(res.rows[0]["total"]), expected, delta=0.01 * max(1.0, expected))
        pipe = self._copilot(sql, user="u_wm_q1")
        ans = pipe.ask("What are total weekly sales?")
        self.assertTrue(ans.ok, ans.error)
        self.assertIsNotNone(ans.sql)
        self.assertNotIn("JOIN", (ans.sql or "").upper())
        self.assertTrue(ans.rows)
        self.assertAlmostEqual(float(ans.rows[0]["total"]), expected,
                               delta=0.01 * max(1.0, expected))

    # -- Q2 sales by store ---------------------------------------------
    def test_05_q2_sales_by_store_no_join(self):
        sql = ('SELECT "Store" AS store, SUM("Weekly_Sales") AS total '
               'FROM "walmart" GROUP BY "Store" ORDER BY total DESC')
        cols, rows = self._query(sql)
        self.assertEqual(len(rows), 45)
        truth = self.frame.copy()
        truth["Weekly_Sales"] = pd.to_numeric(truth["Weekly_Sales"], errors="coerce")
        top = truth.groupby("Store")["Weekly_Sales"].sum().sort_values(ascending=False)
        self.assertEqual(str(rows[0]["store"]), str(top.index[0]))
        self.assertAlmostEqual(float(rows[0]["total"]), float(top.iloc[0]), delta=1.0)
        pipe = self._copilot(sql, user="u_wm_q2")
        ans = pipe.ask("What are total sales by store?")
        self.assertTrue(ans.ok, ans.error)
        self.assertNotIn("JOIN", (ans.sql or "").upper())
        self.assertEqual(len(ans.rows), 45)

    # -- Q3 monthly trend ----------------------------------------------
    def test_06_q3_monthly_trend_no_join(self):
        sql = ('SELECT strftime(\'%Y-%m\', "Date") AS month, '
               'SUM("Weekly_Sales") AS total FROM "walmart" '
               'GROUP BY month ORDER BY month')
        cols, rows = self._query(sql)
        self.assertTrue(len(rows) >= 12)
        months = [r["month"] for r in rows]
        self.assertEqual(months, sorted(months))
        self.assertTrue(all(re.fullmatch(r"\d{4}-\d{2}", m or "") for m in months))
        # Cross-check one month against pandas on normalized dates.
        norm = pd.to_datetime(pd.Series(self.sqlite_dates), errors="coerce")
        exp_first = float(self.sales[norm.dt.strftime("%Y-%m") == months[0]].sum())
        self.assertAlmostEqual(float(rows[0]["total"]), exp_first, delta=1.0)
        pipe = self._copilot(sql, user="u_wm_q3")
        ans = pipe.ask("Show monthly sales trend.")
        self.assertTrue(ans.ok, ans.error)
        self.assertNotIn("JOIN", (ans.sql or "").upper())
        self.assertTrue(ans.rows)
        if ans.chart is not None:
            self.assertIn(ans.chart.get("type"), ("line", "area", "bar", "bar_horizontal"))

    # -- Q4 holiday vs non-holiday -------------------------------------
    def test_07_q4_holiday_vs_nonholiday_no_join(self):
        sql = ('SELECT "Holiday_Flag" AS holiday, SUM("Weekly_Sales") AS total, '
               'COUNT(*) AS n FROM "walmart" GROUP BY "Holiday_Flag" ORDER BY holiday')
        cols, rows = self._query(sql)
        self.assertEqual(len(rows), 2)
        by_flag = {str(r["holiday"]): (float(r["total"]), int(r["n"])) for r in rows}
        self.assertEqual(set(by_flag), {"0", "1"})
        truth = self.frame.copy()
        truth["Weekly_Sales"] = pd.to_numeric(truth["Weekly_Sales"], errors="coerce")
        for flag in ("0", "1"):
            exp = truth.loc[truth["Holiday_Flag"] == flag, "Weekly_Sales"]
            self.assertAlmostEqual(by_flag[flag][0], float(exp.sum()), delta=1.0)
            self.assertEqual(by_flag[flag][1], int(len(exp)))
        pipe = self._copilot(sql, user="u_wm_q4")
        ans = pipe.ask("Compare holiday vs non-holiday sales.")
        self.assertTrue(ans.ok, ans.error)
        self.assertNotIn("JOIN", (ans.sql or "").upper())
        self.assertEqual(len(ans.rows), 2)

    # -- Q5 average weekly sales ----------------------------------------
    def test_08_q5_avg_weekly_sales_no_join(self):
        sql = 'SELECT AVG("Weekly_Sales") AS avg_sales FROM "walmart"'
        cols, rows = self._query(sql)
        expected = float(self.sales.mean())
        self.assertAlmostEqual(float(rows[0]["avg_sales"]), expected, delta=1e-6 * max(1.0, expected))
        pipe = self._copilot(sql, user="u_wm_q5")
        ans = pipe.ask("What is the average weekly sales?")
        self.assertTrue(ans.ok, ans.error)
        self.assertNotIn("JOIN", (ans.sql or "").upper())
        self.assertAlmostEqual(float(ans.rows[0]["avg_sales"]), expected,
                               delta=1e-6 * max(1.0, expected))

    # -- Q6 KPI values match SQL -----------------------------------------
    def test_09_q6_kpi_values_match_sql(self):
        from sqlalchemy import create_engine

        from api.kpi_routes import _profile_numeric, _score_kpi
        from core.schema_intel import SchemaIntelligence

        intel = SchemaIntelligence(self.mapping)
        self.assertEqual(intel.find_best_measure("walmart"), "Weekly_Sales")
        eng = create_engine(f"sqlite:///{self.db_path}")
        try:
            with eng.connect() as conn:
                prof = _profile_numeric(conn, "walmart", "Weekly_Sales", "sqlite")
        finally:
            eng.dispose()
        self.assertIsNotNone(prof)
        _, total = self._query('SELECT SUM("Weekly_Sales") AS s FROM "walmart"')
        _, avg = self._query('SELECT AVG("Weekly_Sales") AS a FROM "walmart"')
        self.assertAlmostEqual(float(prof["sum"]), float(total[0]["s"]), delta=1.0)
        self.assertAlmostEqual(float(prof["avg"]), float(avg[0]["a"]), delta=1e-6)
        self.assertEqual(int(prof["n"]), 6435)
        score, reason = _score_kpi("outcome", "Weekly_Sales", prof,
                                   has_date=True, is_currency=True)
        self.assertGreaterEqual(score, 0.35, reason)
        self.assertTrue(reason)

    # -- Q7 forecast needs date + measure ---------------------------------
    def test_10_q7_forecast_needs_date_and_measure(self):
        from prediction.agent import PredictionAgent

        # Monthly history built from the normalized Date axis.
        cols, mrows = self._query(
            'SELECT strftime(\'%Y-%m-%d\', "Date") AS d, '
            'SUM("Weekly_Sales") AS amount FROM "walmart" '
            'GROUP BY strftime(\'%Y-%m\', "Date") ORDER BY d'
        )
        history = [{"d": r["d"], "amount": float(r["amount"])} for r in mrows]
        self.assertGreaterEqual(len(history), 12)
        out = PredictionAgent().forecast({"data": history, "user_goal": "forecast amount"})
        for key in ("historical", "forecast", "target", "model", "metrics"):
            self.assertIn(key, out, f"missing forecast key {key}")
        self.assertEqual(out["target"], "amount")
        self.assertTrue(out["historical"])
        # Category-only aggregate (no date axis) must degrade honestly.
        _, srows = self._query(
            'SELECT "Store" AS store, SUM("Weekly_Sales") AS total '
            'FROM "walmart" GROUP BY "Store"'
        )
        cat = [{"store": r["store"], "total": float(r["total"])} for r in srows]
        refused = PredictionAgent().forecast({"data": cat, "user_goal": "forecast total"})
        self.assertFalse(refused["valid"])
        self.assertEqual(refused["forecast"], [])
        self.assertIn(refused.get("reliability"), ("not_applicable", "no_data", "insufficient_data"))

    # -- Q8 RAG grounding is SQL-authoritative ------------------------------
    def test_11_q8_rag_grounding_uses_sql_as_authoritative(self):
        from copilot.answer_grounding import check_answer_grounding

        sql = 'SELECT SUM("Weekly_Sales") AS total FROM "walmart"'
        cols, rows = self._query(sql)
        total = float(rows[0]["total"])
        grounded_answer = f"Total weekly sales are {total:,.2f}."
        verdict = check_answer_grounding(
            question="What are total weekly sales?",
            answer=grounded_answer,
            retrieved_tables=["walmart"],
            retrieved_columns=["Weekly_Sales"],
            sql=sql, rows=rows, columns=cols, intent="data",
        )
        self.assertEqual(verdict["verdict"], "grounded")
        self.assertTrue(verdict["checks"]["sql_authoritative"])
        self.assertTrue(verdict["checks"]["numeric_match_vs_sql"])
        invented = check_answer_grounding(
            question="What are total weekly sales?",
            answer=f"Total weekly sales are {total * 3.7:,.2f}.",
            retrieved_tables=["walmart"],
            retrieved_columns=["Weekly_Sales"],
            sql=sql, rows=rows, columns=cols, intent="data",
        )
        self.assertEqual(invented["verdict"], "ungrounded")
        self.assertEqual(invented["reason"], "numeric_mismatch_vs_sql")
        # End-to-end through the pipeline: SQL rows stay authoritative.
        pipe = self._copilot(sql, user="u_wm_q8")
        ans = pipe.ask("What are total weekly sales?")
        self.assertTrue(ans.ok, ans.error)
        pipe_verdict = check_answer_grounding(
            question="What are total weekly sales?",
            answer=f"Total weekly sales are {float(ans.rows[0]['total']):,.2f}.",
            retrieved_tables=ans.tables_used or ["walmart"],
            sql=ans.sql, rows=ans.rows, columns=ans.columns, intent="data",
        )
        self.assertEqual(pipe_verdict["verdict"], "grounded")

    # -- single-table Goal Agent behaviour (no JOINs) -------------------------
    def test_12_goal_agent_single_table_needs_no_joins(self):
        agent = self._goal_agent()
        try:
            for goal in ("total weekly sales by store",
                         "monthly sales trend",
                         "compare holiday vs non-holiday sales"):
                tables = agent._get_relevant_tables(goal)
                self.assertIn("walmart", tables)
                plan = agent._plan_joins(["walmart"])
                self.assertEqual(plan["edges"], [])
                self.assertEqual(plan["nodes"], ["walmart"])
                needs, _ = agent._needs_clarification(goal)
                self.assertFalse(needs, goal)
            # Suspicious-JOIN detector stays silent on single-table SQL.
            single = 'SELECT "Store", SUM("Weekly_Sales") FROM "walmart" GROUP BY "Store"'
            self.assertEqual(agent._find_suspicious_pk_joins(single), [])
            self.assertEqual(agent._warnings_for_sql(single), [])
        finally:
            try:
                agent.engine.dispose()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
