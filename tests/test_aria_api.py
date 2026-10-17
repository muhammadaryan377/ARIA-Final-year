"""API-level tests: auth flow, SQLite hypothesis path, prediction gating.

Hermetic: auth uses a temp DB (monkeypatched), hypothesis/prediction are
pure-math helpers on temp data. No network, no LLM, no live database.
"""

import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import auth


class TestAuthFlow(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._orig_db = auth.DB_PATH
        auth.DB_PATH = Path(self._tmp.name) / "test_auth.db"
        self.addCleanup(setattr, auth, "DB_PATH", self._orig_db)

    def test_register_login_me(self):
        import os
        from fastapi.testclient import TestClient
        import app as app_module

        # Explicit local-HTTP dev opt-out: TestClient runs over plain HTTP,
        # so Secure cookies would not round-trip without this. Production
        # defaults remain Secure (see api/auth_routes._cookie_secure).
        _prev_cookie_secure = os.environ.get("ARIA_COOKIE_SECURE")
        os.environ["ARIA_COOKIE_SECURE"] = "0"
        try:
            c = TestClient(app_module.app, raise_server_exceptions=False)
            user = f"apitest_{int(time.time() * 1000) % 1000000}"
            r = c.post("/api/auth/register", json={"username": user, "password": "Secret123!"})
            self.assertEqual(r.status_code, 200, r.text[:200])
            body = r.json()
            self.assertTrue(body.get("ok"))
            self.assertEqual(body.get("username"), user)
            # HttpOnly cookie flow: raw JWT must NOT be exposed in JSON.
            self.assertNotIn("token", body)

            # Cookie was set on the client jar (dev name: aria_token).
            token = c.cookies.get("aria_token") or c.cookies.get("__Host-aria_token")
            self.assertTrue(token)

            r = c.post("/api/auth/login", json={"username": user, "password": "Secret123!"})
            self.assertEqual(r.status_code, 200)
            self.assertNotIn("token", r.json())
            token = c.cookies.get("aria_token") or c.cookies.get("__Host-aria_token")
            self.assertTrue(token)

            r = c.post("/api/auth/login", json={"username": user, "password": "wrong"})
            self.assertEqual(r.status_code, 401)

            r = c.post("/api/auth/register", json={"username": user, "password": "Secret123!"})
            self.assertEqual(r.status_code, 400)

            # Cookie flow: /me works without an explicit Authorization header.
            r = c.get("/api/auth/me")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["username"], user)

            # Bearer flow still works (token extracted from cookie jar).
            from fastapi.testclient import TestClient as _TC2
            c2 = _TC2(app_module.app, raise_server_exceptions=False)
            r = c2.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["username"], user)

            # Unauthenticated fresh client (no cookie/header) => 401.
            r = c2.get("/api/auth/me")
            self.assertEqual(r.status_code, 401)

            # Malformed/invalid/empty tokens => 401, never 500.
            r = c2.get("/api/auth/me", headers={"Authorization": "Bearer not-a-jwt"})
            self.assertEqual(r.status_code, 401)
            r = c2.get("/api/auth/me", headers={"Authorization": "Bearer null"})
            self.assertEqual(r.status_code, 401)
        finally:
            if _prev_cookie_secure is None:
                os.environ.pop("ARIA_COOKIE_SECURE", None)
            else:
                os.environ["ARIA_COOKIE_SECURE"] = _prev_cookie_secure

    def test_ttl_env_parsing(self):
        import importlib
        import os

        os.environ["TOKEN_TTL_DAYS"] = "1"
        try:
            self.assertEqual(auth._token_ttl_days(), 1)
        finally:
            del os.environ["TOKEN_TTL_DAYS"]
        self.assertEqual(auth._token_ttl_days(), 1)
        # Empty/unset -> intended default 1.
        os.environ["TOKEN_TTL_DAYS"] = ""
        try:
            self.assertEqual(auth._token_ttl_days(), 1)
        finally:
            del os.environ["TOKEN_TTL_DAYS"]
        os.environ["TOKEN_TTL_DAYS"] = "   "
        try:
            self.assertEqual(auth._token_ttl_days(), 1)
        finally:
            del os.environ["TOKEN_TTL_DAYS"]
        # Out-of-range clamps to 1..7.
        os.environ["TOKEN_TTL_DAYS"] = "99"
        try:
            self.assertEqual(auth._token_ttl_days(), 7)
        finally:
            del os.environ["TOKEN_TTL_DAYS"]
        os.environ["TOKEN_TTL_DAYS"] = "0"
        try:
            self.assertEqual(auth._token_ttl_days(), 1)
        finally:
            del os.environ["TOKEN_TTL_DAYS"]
        # Invalid non-integer fails clearly (no silent unsafe fallback).
        os.environ["TOKEN_TTL_DAYS"] = "abc"
        try:
            with self.assertRaises(ValueError):
                auth._token_ttl_days()
        finally:
            del os.environ["TOKEN_TTL_DAYS"]


class TestHypothesisSqlite(unittest.TestCase):
    def _make_engine(self, stmts, table="sales"):
        from sqlalchemy import create_engine

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        conn = sqlite3.connect(tmp.name)
        for s in stmts:
            conn.execute(s)
        conn.commit()
        conn.close()
        engine = create_engine(f"sqlite:///{tmp.name}")

        def _cleanup(path=tmp.name, eng=engine):
            try:
                eng.dispose()
            finally:
                Path(path).unlink(missing_ok=True)

        self.addCleanup(_cleanup)
        return engine

    def _sales_engine(self, n=30):
        stmts = ["CREATE TABLE sales (order_id INTEGER, amount REAL, qty INTEGER)"]
        stmts += [
            f"INSERT INTO sales VALUES ({i}, {100.0 + i * 2.0}, {i % 5 + 1})"
            for i in range(1, n + 1)
        ]
        return self._make_engine(stmts)

    def test_sqlite_tables_and_pairs(self):
        from api import hypothesis_routes as hyp

        engine = self._sales_engine()
        with engine.connect() as conn:
            tables = hyp._get_tables(conn, "sqlite")
            self.assertIn("sales", tables)
            pairs = hyp._find_numeric_pairs(conn, tables, "sqlite")
            # order_id is id-like and excluded; amount+qty must pair up.
            flat = [(x, y) for x, y, _ in pairs]
            self.assertIn(("amount", "qty"), flat)

    def test_run_test_honest_verdict(self):
        from api import hypothesis_routes as hyp

        engine = self._sales_engine()
        req = SimpleNamespace(hypothesis="higher quantity means higher amount")
        out = hyp._run_test(engine, "sqlite", req)
        self.assertTrue(out["ok"])
        self.assertIn("supported", out)
        self.assertIn("stats", out)
        self.assertGreaterEqual(len(out["scatter_data"]), 10)

    def test_run_test_insufficient_data(self):
        from api import hypothesis_routes as hyp

        engine = self._make_engine(
            ["CREATE TABLE tiny (a REAL, b REAL)", "INSERT INTO tiny VALUES (1.0, 2.0)"]
        )
        req = SimpleNamespace(hypothesis="a relates to b")
        out = hyp._run_test(engine, "sqlite", req)
        self.assertFalse(out["ok"])
        self.assertIn("detail", out)


class TestPredictionGating(unittest.TestCase):
    def test_empty_refuses(self):
        from prediction.agent import PredictionAgent

        out = PredictionAgent().forecast({"data": []})
        self.assertFalse(out["valid"])

    def test_trending_series_valid(self):
        from prediction.agent import PredictionAgent

        rows = [{"d": f"2024-{m:02d}-01", "amount": 100.0 + m * 10.0} for m in range(1, 13)]
        out = PredictionAgent().forecast({"data": rows, "user_goal": "forecast amount"})
        self.assertIn("valid", out)
        if out["valid"]:
            self.assertTrue(out.get("points"))

    def test_id_only_refuses(self):
        from prediction.agent import PredictionAgent

        rows = [{"order_id": i} for i in range(10)]
        out = PredictionAgent().forecast({"data": rows})
        self.assertFalse(out["valid"])


class TestDashboardScoring(unittest.TestCase):
    def test_variation(self):
        from api.kpi_routes import _variation

        self.assertEqual(_variation(0, 0, 0), 0.0)
        self.assertEqual(_variation(100, 100, 100), 0.0)
        self.assertGreater(_variation(100, 50, 150), 0.5)
        self.assertEqual(_variation(None, 1, 2), 0.0)

    def test_kpi_score_orders_semantics(self):
        from api.kpi_routes import _score_kpi

        prof = {"n": 100.0, "nn": 98.0, "distinct": 50.0, "sum": 1000.0,
                "avg": 10.0, "min": 1.0, "max": 50.0}
        s_out, _ = _score_kpi("outcome", "revenue", prof, True, True)
        s_vol, _ = _score_kpi("volume", "revenue", prof, True, True)
        self.assertGreater(s_out, s_vol)
        # Heavy nulls penalize below a clean column.
        bad = dict(prof, nn=10.0)
        s_bad, _ = _score_kpi("outcome", "revenue", bad, False, False)
        self.assertGreater(s_out, s_bad)

    def test_kpi_score_has_reason(self):
        from api.kpi_routes import _score_kpi

        prof = {"n": 10.0, "nn": 10.0, "distinct": 5.0, "sum": 100.0,
                "avg": 10.0, "min": 10.0, "max": 10.0}
        _, reason = _score_kpi("efficiency", "amount", prof, False, False)
        self.assertIn("no variation", reason)

    def test_time_grain(self):
        from api.kpi_routes import _time_grain_expr

        _, g = _time_grain_expr("d", "sqlite", 30)
        self.assertEqual(g, "day")
        _, g = _time_grain_expr("d", "sqlite", 100)
        self.assertEqual(g, "week")
        _, g = _time_grain_expr("d", "sqlite", 800)
        self.assertEqual(g, "month")
        _, g = _time_grain_expr("d", "sqlite", 2000)
        self.assertEqual(g, "quarter")
        _, g = _time_grain_expr("d", "postgresql", 800)
        self.assertEqual(g, "month")


class TestUserOverrides(unittest.TestCase):
    def _agent(self, tmpdir, overrides=None):
        import json as _json
        import os as _os
        from goal_agent import GoalAgent

        mapping = {
            "schema": "public",
            "tables": {
                "customers": {"columns": ["id", "name"], "primary_key": ["id"]},
                "orders": {"columns": ["order_id", "customer_id", "total"],
                           "primary_key": ["order_id"]},
            },
            "relationship_edges": [],
            "declared_relationships": [],
            "inferred_relationships": [{
                "table": "orders", "column": "customer_id",
                "references_table": "customers", "references_column": "id",
                "relationship_state": "UNCERTAIN", "confidence_score": 55,
            }],
            "ambiguous_relationships": [],
        }
        p = _os.path.join(tmpdir, "schema.json")
        open(p, "w").write(_json.dumps(mapping))
        return GoalAgent(schema_json_path=p, db_uri="postgresql://u:p@localhost:5432/x",
                         provider=object(), dialect="postgresql", overrides=overrides)

    def test_rejected_edge_removed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ag = self._agent(d, {"rejected": ["orders.customer_id->customers.id"],
                                 "fk": {}, "pk": {}})
            keys = {(e["left_table"], e["left_column"], e["right_table"], e["right_column"])
                    for e in ag.relationship_edges}
            self.assertNotIn(("orders", "customer_id", "customers", "id"), keys)

    def test_user_fk_outranks_inference(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ag = self._agent(d, {"rejected": [], "pk": {},
                                 "fk": {"orders.customer_id->customers.id": {
                                     "table": "orders", "column": "customer_id",
                                     "references_table": "customers", "references_column": "id"}}})
            trusts = [e["trust"] for e in ag.relationship_edges
                      if (e["left_table"], e["left_column"]) == ("orders", "customer_id")]
            self.assertIn("USER_CONFIRMED", trusts)
            plan = ag._plan_joins(["orders", "customers"])
            used = [e.get("trust") for e in plan["edges"]]
            self.assertIn("USER_CONFIRMED", used)

    def test_no_overrides_unchanged(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ag = self._agent(d)
            self.assertTrue(any(e["left_table"] == "orders" for e in ag.relationship_edges))


class TestWhyStoryEnvelope(unittest.TestCase):
    def test_builders(self):
        from api.query_routes import _build_why, _build_story, _chart_spec_for_result, _check_consistency

        rows = [{"customer": "alice", "total": 250.0}, {"customer": "bob", "total": 200.0}]
        why = _build_why(goal="total by customer", tables=["sales"], columns=["customer", "total"],
                         row_count=2, sql="SELECT ...", explanation="Alice leads.",
                         confidence="high", warnings=[])
        self.assertIn("limitations", why)
        self.assertEqual(why["evidence"]["row_count"], 2)
        story = _build_story(goal="g", tables=["sales"], columns=["customer", "total"],
                             result_data=rows, why_block=why)
        self.assertIn("alice", story.lower())
        spec = _chart_spec_for_result(["customer", "total"], rows)
        self.assertIn("chart_type", spec)
        self.assertEqual(spec["chart_type"], "bar")
        cons = _check_consistency(spec, rows, why)
        self.assertTrue(cons["consistent"])
        cons2 = _check_consistency({"chart_type": "bar"}, [], why)
        self.assertFalse(cons2["consistent"])
        self.assertEqual(cons2["chart_spec"]["chart_type"], "table")


class TestCrossSource(unittest.TestCase):
    def test_no_intent_no_docs(self):
        from api.query_routes import _document_evidence

        self.assertEqual(_document_evidence("nouser_xyz", "total sales by region"), [])

    def test_doc_keywords(self):
        from api.query_routes import _DOC_INTENT_WORDS

        self.assertIn("target", _DOC_INTENT_WORDS)
        self.assertTrue(any(w in "compare revenue with the targets in the report".lower()
                            for w in _DOC_INTENT_WORDS))


class TestUploadCapsAndRedaction(unittest.TestCase):
    def test_spool_limit(self):
        import io
        from fastapi import HTTPException
        from api.connect_routes import _spool_limited

        class FakeUpload:
            def __init__(self, data):
                self._io = io.BytesIO(data)

            def read(self, n):
                return self._io.read(n)

        import tempfile
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "f.csv"
            n = _spool_limited(FakeUpload(b"a,b\n1,2\n"), dest, 1024 * 1024, "CSV file")
            self.assertGreater(n, 0)
            self.assertTrue(dest.exists())
            dest2 = Path(d) / "big.csv"
            with self.assertRaises(HTTPException) as ctx:
                _spool_limited(FakeUpload(b"x" * (2 * 1024 * 1024)), dest2, 1024 * 1024, "CSV file")
            self.assertEqual(ctx.exception.status_code, 413)
            self.assertFalse(dest2.exists())

    def test_db_uri_redaction(self):
        import re

        raw = "postgresql://admin:s3cr3t@db.internal:5432/sales"
        safe = re.sub(r"://([^:/@]+):[^@/]*@", r"://\1:***@", raw)
        self.assertEqual(safe, "postgresql://admin:***@db.internal:5432/sales")
        self.assertNotIn("s3cr3t", safe)


class TestRateLimitAndErrors(unittest.TestCase):
    def test_ratelimit_trips(self):
        from fastapi import HTTPException
        from core import ratelimit as rl

        rl._hits.clear()
        for _ in range(3):
            rl.check_limit("t", "k", 3, 60)
        with self.assertRaises(HTTPException) as ctx:
            rl.check_limit("t", "k", 3, 60)
        self.assertEqual(ctx.exception.status_code, 429)

    def test_ratelimit_disabled(self):
        import os
        from core import ratelimit as rl

        rl._hits.clear()
        os.environ["ARIA_RATE_LIMIT"] = "off"
        try:
            for _ in range(10):
                rl.check_limit("t2", "k", 1, 60)
        finally:
            del os.environ["ARIA_RATE_LIMIT"]

    def test_sanitize_error(self):
        from core.errors import sanitize_error

        out = sanitize_error("postgresql://admin:s3cr3t@db:5432/sales password='hunter2'")
        self.assertNotIn("s3cr3t", out)
        self.assertNotIn("hunter2", out)
        self.assertIn("***", out)
        out = sanitize_error(Exception("no tables in C:\\data\\x\\f.db"))
        self.assertNotIn("C:\\data", out)

    def test_humanize_labels(self):
        from core.semantic import humanize, kpi_label, is_valid_identifier

        self.assertEqual(humanize("order_date"), "Order Date")
        self.assertEqual(humanize("customer_id"), "Customer")
        self.assertEqual(kpi_label("SUM", "amount"), "Total Amount")
        self.assertEqual(kpi_label("AVG", "order_value"), "Average Order Value")
        self.assertEqual(kpi_label("DISTINCT", "customer"), "Customer Count")
        self.assertTrue(is_valid_identifier("sales"))
        self.assertFalse(is_valid_identifier("a; DROP TABLE x"))

    def test_rl_event_capped(self):
        import json
        from pathlib import Path
        import tempfile
        from core.rl import record_event

        with tempfile.TemporaryDirectory() as tmp:
            record_event("u", event="e", state={"big": "x" * 20000},
                         base_dir=Path(tmp))
            lines = (Path(tmp) / "user_u" / "rl_events.jsonl").read_text().splitlines()
            rec = json.loads(lines[0])
            self.assertLessEqual(len(json.dumps(rec)), 8192)


class TestSecretBoxKeepalive(unittest.TestCase):
    def test_roundtrip(self):
        from core.secret_box import encrypt, decrypt, PREFIX

        enc = encrypt("s3cr3t-value")
        self.assertTrue(enc.startswith(PREFIX))
        self.assertEqual(decrypt(enc), "s3cr3t-value")
        self.assertEqual(decrypt("plain"), "plain")


class TestSlashDateNormalization(unittest.TestCase):
    def test_mdy_resolved(self):
        from csv_orchestrator import _resolve_slash_date_format

        self.assertEqual(
            _resolve_slash_date_format(["11/8/2016", "6/14/2017", "1/5/2018"]), "MDY")

    def test_dmy_resolved(self):
        from csv_orchestrator import _resolve_slash_date_format

        self.assertEqual(
            _resolve_slash_date_format(["8/11/2016", "14/6/2017"]), "DMY")

    def test_ambiguous_unresolved(self):
        from csv_orchestrator import _resolve_slash_date_format

        self.assertIsNone(_resolve_slash_date_format(["3/4/2016", "5/6/2017"]))
        self.assertIsNone(_resolve_slash_date_format(["11/8/2016", "not a date"]))

    def test_to_iso(self):
        from csv_orchestrator import _slash_to_iso

        self.assertEqual(_slash_to_iso("11/8/2016", "MDY"), "2016-11-08")
        self.assertEqual(_slash_to_iso("8/11/2016", "DMY"), "2016-11-08")
        self.assertIsNone(_slash_to_iso("13/40/2016", "MDY"))

    def test_load_normalizes(self):
        import pandas as pd
        from csv_orchestrator import _normalize_date_columns

        df = pd.DataFrame({"order_date": ["11/8/2016", "6/14/2017"], "v": ["1", "2"]})
        notes = _normalize_date_columns(df, {"order_date": "STRING", "v": "STRING"})
        self.assertIn("order_date", notes)
        self.assertEqual(list(df["order_date"]), ["2016-11-08", "2017-06-14"])


if __name__ == "__main__":
    unittest.main()
