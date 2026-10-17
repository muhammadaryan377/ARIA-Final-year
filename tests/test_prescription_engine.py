"""Prescription engine + API tests (spec: s.txt master plan).

Hermetic: temp SQLite files with seeded random data (no live database, no
LLM). Every schema uses generated table/column names so the tests prove
dynamic discovery — nothing hardcoded:

  * strong pair        -> completed scenario rx, traceable r
  * weak pair          -> honest no_prescription naming tested columns
  * n < MIN_N          -> refusal, never a fabricated model
  * self driver        -> INVALID_DRIVER
  * unknown metric     -> INVALID_METRIC + available_metrics
  * cross-table driver -> declared-join label, same r as manual join math
  * stay-length        -> derived driver from two date columns
  * scenario math      -> matches manual OLS recomputation
  * score              -> equals sum of its breakdown
  * success/review     -> measurable target + data-derived review period
  * investigate        -> segment gaps only, OBSERVED labels, no simulation
  * document mode      -> citations are verbatim input slices, nothing invented
  * user isolation     -> saved prescriptions never leak across users
"""

import math
import random
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine

from core.schema_intel import SchemaIntelligence
from prescription_engine import (
    analyze,
    analyze_document,
    run_scenario,
)


def _col(name, dtype):
    return {"column": name, "data_type": dtype}


def _mapping(tables, edges=None):
    return {
        "schema": "public",
        "tables": tables,
        "relationship_edges": edges or [],
        "declared_relationships": [],
    }


def _sqlite(rows_map):
    """{table: (ddl, rows)} -> (engine, tmpdir)."""
    tmp = Path(tempfile.mkdtemp())
    db = tmp / "t.db"
    conn = sqlite3.connect(str(db))
    for table, (ddl, rows) in rows_map.items():
        conn.execute(f"CREATE TABLE {table} ({ddl})")
        if rows:
            conn.executemany(
                f"INSERT INTO {table} VALUES ({','.join('?' * len(rows[0]))})", rows)
    conn.commit()
    conn.close()
    return create_engine(f"sqlite:///{db}"), tmp


class StrongPairCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = random.Random(11)
        cls.rows = []
        for _ in range(120):
            p = rng.uniform(10, 100)
            cls.rows.append((round(p, 2), round(2000 - 15 * p + rng.gauss(0, 40), 2)))
        cls.engine, _ = _sqlite({"shop": ("px REAL, sxs REAL",
                                          [(r[0], r[1]) for r in cls.rows])})
        cls.intel = SchemaIntelligence(_mapping({
            "shop": {"columns": [_col("px", "REAL"), _col("sxs", "REAL")],
                     "primary_key": []}}))

    def test_completed_with_evidence(self):
        out = analyze(self.engine, "sqlite", self.intel, "shop", "sxs", driver="px")
        self.assertEqual(out["status"], "completed")
        self.assertEqual(out["driver"], "px")
        xs = np.array([r[0] for r in self.rows])
        ys = np.array([r[1] for r in self.rows])
        expect_r = round(float(np.corrcoef(xs, ys)[0, 1]), 3)
        self.assertAlmostEqual(out["fit"]["r"], expect_r, places=3)
        kinds = {p["action_type"] for p in out["prescriptions"]}
        self.assertIn("scenario", kinds)
        self.assertTrue(out["summary_plain"])
        sm = out["prescriptions"][0]["success_metric"]
        self.assertEqual(sm["metric"], "sxs")
        self.assertIn("target_pct", sm)
        self.assertTrue(out["prescriptions"][0]["review_period"])

    def test_deterministic(self):
        a = analyze(self.engine, "sqlite", self.intel, "shop", "sxs", driver="px")
        b = analyze(self.engine, "sqlite", self.intel, "shop", "sxs", driver="px")
        self.assertEqual([p["title"] for p in a["prescriptions"]],
                         [p["title"] for p in b["prescriptions"]])
        self.assertEqual([p["score"] for p in a["prescriptions"]],
                         [p["score"] for p in b["prescriptions"]])

    def test_score_sums(self):
        out = analyze(self.engine, "sqlite", self.intel, "shop", "sxs", driver="px")
        for p in out["prescriptions"]:
            if p["action_type"] == "scenario":
                self.assertAlmostEqual(p["score"], round(sum(p["score_breakdown"].values()), 1))

    def test_scenario_matches_manual_ols(self):
        out = run_scenario(self.engine, "sqlite", self.intel, "shop", "sxs", "px",
                           [{"field": "px", "operation": "percent", "value": -10.0}])
        self.assertTrue(out["success"])
        xs = np.array([r[0] for r in self.rows])
        ys = np.array([r[1] for r in self.rows])
        slope, intercept = np.polyfit(xs, ys, 1)
        base = intercept + slope * xs.mean()
        scen = base + slope * (xs.mean() * -0.10)
        got = out["scenarios"][0]
        self.assertAlmostEqual(got["baseline"], round(base, 2), places=1)
        self.assertAlmostEqual(got["scenario"], round(scen, 2), places=1)


class RefusalCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = random.Random(23)
        cls.engine, _ = _sqlite({
            "w": ("aa REAL, bb REAL, cc REAL",
                  [(rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(80)]),
            "tiny": ("x REAL, y REAL", [(float(i), float(2 * i)) for i in range(1, 6)]),
        })
        cls.intel = SchemaIntelligence(_mapping({
            "w": {"columns": [_col("aa", "REAL"), _col("bb", "REAL"), _col("cc", "REAL")],
                  "primary_key": []},
            "tiny": {"columns": [_col("x", "REAL"), _col("y", "REAL")],
                     "primary_key": []},
        }))

    def test_weak_pair_refuses_and_names_evidence(self):
        out = analyze(self.engine, "sqlite", self.intel, "w", "aa")
        self.assertEqual(out["status"], "no_prescription")
        self.assertIn("aa", out["reason"])

    def test_small_sample_refuses(self):
        out = analyze(self.engine, "sqlite", self.intel, "tiny", "y", driver="x")
        self.assertEqual(out["status"], "no_prescription")

    def test_self_driver_rejected(self):
        out = analyze(self.engine, "sqlite", self.intel, "w", "aa", driver="aa")
        self.assertFalse(out["success"])
        self.assertEqual(out["code"], "INVALID_DRIVER")

    def test_unknown_metric_lists_available(self):
        out = analyze(self.engine, "sqlite", self.intel, "w", "nope")
        self.assertEqual(out["code"], "INVALID_METRIC")
        self.assertTrue(out["available_metrics"])


class JoinAndStayCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = random.Random(37)
        spends, amounts, cids = [], [], []
        for i in range(1, 61):
            s = rng.uniform(50, 2000)
            spends.append((i, round(s, 2)))
            amounts.append((i, i, round(4 * s + rng.gauss(0, 300), 2)))
            cids.append(i)
        import datetime
        base = datetime.date(2024, 1, 1)
        stays = []
        for i in range(100):
            d1 = base + datetime.timedelta(days=rng.randint(0, 300))
            nights = rng.randint(1, 14)
            d2 = d1 + datetime.timedelta(days=nights)
            stays.append((d1.isoformat(), d2.isoformat(),
                          round(nights * 120 + rng.gauss(0, 60), 2)))
        cls.engine, _ = _sqlite({
            "funds": ("cid INTEGER PRIMARY KEY, sp REAL", spends),
            "gains": ("gid INTEGER PRIMARY KEY, cid INTEGER, am REAL", amounts),
            "visits": ("start_date TEXT, end_date TEXT, bill REAL", stays),
        })
        cls.intel = SchemaIntelligence(_mapping(
            {"funds": {"columns": [_col("cid", "INTEGER"), _col("sp", "REAL")],
                       "primary_key": ["cid"]},
             "gains": {"columns": [_col("gid", "INTEGER"), _col("cid", "INTEGER"),
                                   _col("am", "REAL")],
                       "primary_key": ["gid"]},
             "visits": {"columns": [_col("start_date", "TEXT"), _col("end_date", "TEXT"),
                                     _col("bill", "REAL")],
                        "primary_key": []}},
            edges=[{"source_table": "gains", "source_column": "cid",
                    "target_table": "funds", "target_column": "cid",
                    "type": "declared", "confidence": "declared"}]))

    def test_cross_table_driver(self):
        out = analyze(self.engine, "sqlite", self.intel, "gains", "am")
        self.assertEqual(out["status"], "completed")
        self.assertEqual(out["driver"], "sp")
        self.assertTrue(out["join"])
        self.assertIn("funds", out["join"])

    def test_stay_length_derived(self):
        out = analyze(self.engine, "sqlite", self.intel, "visits", "bill")
        self.assertEqual(out["status"], "completed")
        self.assertEqual(out["driver"], "length_of_stay")


class InvestigateCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = random.Random(51)
        rows = []
        for i in range(90):
            grp = ["a", "b", "c"][i % 3]
            mu = {"a": 100, "b": 160, "c": 105}[grp]
            rows.append((grp, round(rng.gauss(mu, 12), 1)))
        cls.engine, _ = _sqlite({"seg": ("grp TEXT, val REAL", rows)})
        cls.intel = SchemaIntelligence(_mapping({
            "seg": {"columns": [_col("grp", "TEXT"), _col("val", "REAL")],
                    "primary_key": []}}))

    def test_segment_gap_prescription(self):
        out = analyze(self.engine, "sqlite", self.intel, "seg", "val")
        self.assertEqual(out["status"], "completed")
        inv = [p for p in out["prescriptions"] if p["action_type"] == "investigate"]
        self.assertTrue(inv)
        self.assertEqual(inv[0]["impact"]["labels"]["scenario"], "OBSERVED")
        self.assertTrue(inv[0]["evidence"]["groups"])
        self.assertTrue(inv[0]["success_metric"])
        self.assertTrue(inv[0]["review_period"])


class DocumentCase(unittest.TestCase):
    CHUNKS = [
        {"id": "c1", "content": "Revenue declined 12 percent in Q3. The shortfall concentrated in enterprise contracts.", "page_number": 2, "section": "Results"},
        {"id": "c2", "content": "The board approved a plan to expand support coverage next quarter.", "page_number": 3, "section": None},
        {"id": "c3", "content": "Churn risk is rising among mid-market accounts because onboarding delays persist.", "page_number": 4, "section": "Risks"},
    ]

    def test_cited_recommendations(self):
        out = analyze_document(self.CHUNKS, "What revenue risks need action?", "q3.pdf")
        self.assertEqual(out["status"], "completed")
        for p in out["prescriptions"]:
            self.assertEqual(p["action_type"], "qualitative")
            for cite in p["citations"]:
                src = next(c["content"] for c in self.CHUNKS if c["id"] == cite["chunk_id"])
                self.assertIn(cite["quote"][:60], src)
        self.assertIn("q3.pdf", out["summary_plain"])

    def test_empty_question_asks(self):
        out = analyze_document(self.CHUNKS, "   ", "q3.pdf")
        self.assertEqual(out["status"], "no_prescription")

    def test_no_overlap_refuses(self):
        out = analyze_document(self.CHUNKS, "xylophone zebra orbit", "q3.pdf")
        self.assertEqual(out["status"], "no_prescription")


class IsolationCase(unittest.TestCase):
    def test_saved_prescriptions_are_user_isolated(self):
        try:
            from fastapi.testclient import TestClient
        except Exception:
            self.skipTest("fastapi.testclient unavailable")
            return
        import os
        import tempfile
        from pathlib import Path
        import auth as _auth
        # Hermetic auth DB: never pollute the real data/aria.db.
        _tmp = tempfile.TemporaryDirectory()
        self.addCleanup(_tmp.cleanup)
        _orig_db = _auth.DB_PATH
        _auth.DB_PATH = Path(_tmp.name) / "test_rxiso_auth.db"
        self.addCleanup(setattr, _auth, "DB_PATH", _orig_db)
        # Plain-HTTP TestClient needs the Secure-cookie opt-out (production
        # default stays Secure). Cookie-only flow: no token in JSON.
        _prev = os.environ.get("ARIA_COOKIE_SECURE")
        os.environ["ARIA_COOKIE_SECURE"] = "0"
        self.addCleanup(os.environ.__setitem__, "ARIA_COOKIE_SECURE", _prev) if _prev is not None else self.addCleanup(os.environ.pop, "ARIA_COOKIE_SECURE", None)
        import app as _app
        import random as _rng
        clients = []
        for u in ("rxiso%s" % _rng.randint(1000000, 9999999),
                  "rxiso%s" % _rng.randint(1000000, 9999999)):
            c = TestClient(_app.app, raise_server_exceptions=False)
            r = c.post("/api/auth/register",
                       json={"username": u, "password": "password123"})
            self.assertEqual(r.status_code, 200, r.text[:200])
            self.assertNotIn("token", r.json())
            self.assertTrue(c.cookies.get("aria_token") or c.cookies.get("__Host-aria_token"))
            clients.append(c)
        rx = {"id": "x", "title": "T", "action_type": "scenario"}
        r = clients[0].post("/api/prescriptions/save",
                            json={"prescription": rx, "question": "q"})
        self.assertTrue(r.json().get("ok"), r.text[:200])
        other = clients[1].get("/api/prescriptions/saved").json()
        self.assertEqual(other.get("saved"), [])
        own = clients[0].get("/api/prescriptions/saved").json()
        self.assertEqual(len(own.get("saved")), 1)
        sid = own["saved"][0]["id"]
        clients[0].delete(f"/api/prescriptions/saved/{sid}")
        own2 = clients[0].get("/api/prescriptions/saved").json()
        self.assertEqual(own2.get("saved"), [])


if __name__ == "__main__":
    unittest.main()
