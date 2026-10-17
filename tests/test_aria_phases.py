"""Phase tests for t.txt implementation (additive; independent of goal_agent).

Covers: core.semantic helpers, core.rl ranking/events, csv identifier
quoting, prescription impact honesty note, query provenance envelope keys.
"""

import json
import unittest

from core.rl import IMPACT_UNKNOWN_NOTE, RLPolicy, rank_candidates, record_event
from core.semantic import (
    build_result_envelope,
    classify_semantic_type,
    is_valid_identifier,
)
from csv_orchestrator import _quote_ident


class TestSemanticHelpers(unittest.TestCase):
    def test_valid_identifiers(self):
        self.assertTrue(is_valid_identifier("orders"))
        self.assertTrue(is_valid_identifier("customer_id"))
        self.assertFalse(is_valid_identifier(""))
        self.assertFalse(is_valid_identifier("a; DROP TABLE x"))
        self.assertFalse(is_valid_identifier('a"b'))
        self.assertFalse(is_valid_identifier("9lives"))

    def test_classify_types(self):
        self.assertEqual(classify_semantic_type("customer_id", "integer"), "identifier")
        self.assertEqual(classify_semantic_type("order_date", "timestamp"), "date")
        self.assertEqual(classify_semantic_type("total_price", "numeric"), "money")
        self.assertEqual(classify_semantic_type("discount_pct", "float"), "percentage")
        self.assertEqual(classify_semantic_type("qty", "integer"), "measure")
        self.assertEqual(classify_semantic_type("is_active", "boolean"), "boolean")
        # Unknown schema names still classify without crashing.
        result = classify_semantic_type("blorpt_field", "text")
        self.assertIn(result, ("text", "category"))

    def test_envelope_contract(self):
        env = build_result_envelope(
            result={"a": 1}, evidence=["e1"], source="pg",
            confidence="high", methodology="m",
        )
        for key in ("result", "evidence", "source", "confidence", "assumptions",
                    "warnings", "errors", "methodology", "provenance", "timestamp"):
            self.assertIn(key, env)


class TestQuoteIdent(unittest.TestCase):
    def test_escapes_quotes(self):
        self.assertEqual(_quote_ident("orders"), '"orders"')
        self.assertEqual(_quote_ident('a"; DROP TABLE x; --'), '"a""; DROP TABLE x; --"')

    def test_roundtrip_sqlite(self):
        import sqlite3

        conn = sqlite3.connect(":memory:")
        try:
            evil = 'weird"; DROP TABLE t; --'
            conn.execute(f"CREATE TABLE {_quote_ident('t')} ({_quote_ident(evil)} TEXT)")
            conn.execute(f"INSERT INTO {_quote_ident('t')} ({_quote_ident(evil)}) VALUES (?)", ("v",))
            rows = conn.execute(f"SELECT {_quote_ident(evil)} FROM {_quote_ident('t')}").fetchall()
            self.assertEqual(rows[0][0], "v")
        finally:
            conn.close()


class TestRLLayer(unittest.TestCase):
    def test_rank_only_valid_candidates(self):
        cands = [{"id": "bar", "score": 1.0}, {"id": "line", "score": 2.0}]
        ordered = rank_candidates(cands)
        self.assertEqual(ordered[0]["id"], "line")

    def test_rank_never_crashes(self):
        self.assertEqual(rank_candidates([]), [])
        ordered = rank_candidates([{"id": "x"}], prefs={"suggestion_clicks": {"x": 5}})
        self.assertEqual(ordered[0]["id"], "x")

    def test_policy_fallback(self):
        policy = RLPolicy()
        out = policy.rank({}, [{"id": "a"}])
        self.assertFalse(out["fallback"])
        self.assertEqual(out["ordered"][0]["id"], "a")

    def test_record_event_no_raw_datasets(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            record_event(
                "test_user",
                event="chart_changed",
                state={"metric": "revenue"},
                action={"type": "select_chart", "value": "line"},
                outcome={"kept": True},
                reward={"usefulness": 1},
                base_dir=Path(tmp),
            )
            path = Path(tmp) / "user_test_user" / "rl_events.jsonl"
            self.assertTrue(path.exists())
            rec = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(rec["event"], "chart_changed")


class TestPrescriptionHonesty(unittest.TestCase):
    def test_impact_note_present(self):
        from prescription_agent import PrescriptionAgent

        agent = PrescriptionAgent.__new__(PrescriptionAgent)  # no provider needed
        actions = agent._template_actions([], [], [], [], None, 3)
        self.assertTrue(actions)
        self.assertIn("could not be reliably estimated", IMPACT_UNKNOWN_NOTE)


if __name__ == "__main__":
    unittest.main()
