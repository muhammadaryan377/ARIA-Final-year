"""Prescription top-notch coverage (FYP core).

Hermetic: temp SQLite files with seeded data (no live database, no LLM).
Proves the broadened honest coverage without weakened honesty:

  * segment-focus fires on a dominant-share breakdown (share >= 30%)
  * monitor fires on strong trend momentum (>= 15% over >= 6 points)
  * refusal preserved for unsupported shapes (tiny n, no signal anywhere)
  * engine validation drops empty-evidence cards and counts the drops
  * refresh (re-analyze) + run_scenario live on a sqlite lever fixture
"""

import datetime
import random
import sqlite3
import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine

from core.schema_intel import SchemaIntelligence
from prescription_engine import analyze, run_scenario
from prescription_engine.engine import _validate_prescriptions


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
        conn.execute("CREATE TABLE %s (%s)" % (table, ddl))
        if rows:
            conn.executemany(
                "INSERT INTO %s VALUES (%s)"
                % (table, ",".join("?" * len(rows[0]))), rows)
    conn.commit()
    conn.close()
    return create_engine("sqlite:///%s" % db), tmp


class SegmentFocusCase(unittest.TestCase):
    """Dominant-share breakdown (equal means, so no mean-difference test
    could fire) still earns an honest segment-focus card."""

    @classmethod
    def setUpClass(cls):
        rng = random.Random(99)
        rows = []
        # 'big' holds 60/120 rows (50% share) with the SAME mean as the rest.
        for i in range(120):
            grp = "big" if i % 2 == 0 else ("mid" if i % 4 == 1 else "small")
            rows.append((grp, round(rng.gauss(100, 5), 1)))
        cls.engine, _ = _sqlite({"seg": ("grp TEXT, val REAL", rows)})
        cls.intel = SchemaIntelligence(_mapping({
            "seg": {"columns": [_col("grp", "TEXT"), _col("val", "REAL")],
                    "primary_key": []},
        }))

    def test_segment_focus_fires_on_dominant_share(self):
        out = analyze(self.engine, "sqlite", self.intel, "seg", "val")
        self.assertEqual(out["status"], "completed")
        seg = [p for p in out["prescriptions"]
               if p.get("scenario_label") == "segment-focus"]
        self.assertTrue(seg, "expected a segment-focus card, got %s"
                        % [p.get("scenario_label")
                           for p in out["prescriptions"]])
        top = seg[0]
        self.assertTrue(top["title"].startswith("Review big"))
        self.assertGreaterEqual(top["evidence"]["share_pct"], 30.0)
        # Honesty: no invented impact — explicit note present.
        self.assertTrue(top.get("impact_note"))
        self.assertIn("could not be reliably estimated",
                      " ".join(top.get("limitations", [])).lower())
        self.assertTrue(top["evidence"]["n"] >= 10)


class MonitorCase(unittest.TestCase):
    """Strong trend momentum over >= 6 points earns a monitor card with
    metric + direction and no causal claim."""

    @classmethod
    def setUpClass(cls):
        base = datetime.date(2024, 1, 1)
        rows = [((base + datetime.timedelta(days=30 * i)).isoformat(),
                 round(100 + i * 9, 1)) for i in range(12)]
        cls.engine, _ = _sqlite({"m": ("sale_date TEXT, v REAL", rows)})
        cls.intel = SchemaIntelligence(_mapping({
            "m": {"columns": [_col("sale_date", "TEXT"), _col("v", "REAL")],
                  "primary_key": []},
        }))

    def test_monitor_fires_on_strong_momentum(self):
        out = analyze(self.engine, "sqlite", self.intel, "m", "v")
        self.assertEqual(out["status"], "completed")
        mon = [p for p in out["prescriptions"]
               if p.get("action_type") == "monitor"]
        self.assertTrue(mon)
        self.assertIn("up", mon[0]["title"])
        self.assertGreaterEqual(
            abs(mon[0]["evidence"]["momentum_pct"]), 15.0)
        self.assertGreaterEqual(mon[0]["evidence"]["n_points"], 6)
        self.assertTrue(mon[0].get("impact_note"))

    def test_flat_trend_refuses_monitor(self):
        rng = random.Random(5)
        base = datetime.date(2024, 1, 1)
        rows = [((base + datetime.timedelta(days=30 * i)).isoformat(),
                 round(100 + rng.gauss(0, 1), 1)) for i in range(12)]
        engine, _ = _sqlite({"f": ("sale_date TEXT, v REAL", rows)})
        intel = SchemaIntelligence(_mapping({
            "f": {"columns": [_col("sale_date", "TEXT"), _col("v", "REAL")],
                  "primary_key": []},
        }))
        out = analyze(engine, "sqlite", intel, "f", "v")
        mon = [p for p in (out.get("prescriptions") or [])
               if p.get("action_type") == "monitor"]
        self.assertFalse(mon)


class RefusalPreservedCase(unittest.TestCase):
    """Unsupported shapes still refuse: tiny n with no dates, dims, or
    levers anywhere."""

    def test_tiny_metric_only_refuses(self):
        engine, _ = _sqlite({
            "tiny": ("x REAL", [(float(i),) for i in range(1, 6)]),
        })
        intel = SchemaIntelligence(_mapping({
            "tiny": {"columns": [_col("x", "REAL")],
                     "primary_key": []},
        }))
        out = analyze(engine, "sqlite", intel, "tiny", "x")
        self.assertEqual(out["status"], "no_prescription")


class ValidationCase(unittest.TestCase):
    def test_drops_empty_evidence_and_counts(self):
        recs = [
            {"id": "good", "evidence": {"metric": "v", "n": 50},
             "impact": {"baseline": 1.0, "scenario": 2.0, "delta": 1.0,
                        "rel_pct": 100.0}},
            {"id": "no-evidence", "evidence": {},
             "impact": {"baseline": 1.0, "scenario": 2.0}},
            {"id": "no-impact-no-note",
             "evidence": {"metric": "v", "n": 50},
             "impact": {"baseline": None, "scenario": None, "delta": None,
                        "rel_pct": None}},
            {"id": "honest-note", "evidence": {"metric": "v", "n": 50},
             "impact": {"baseline": None, "scenario": None, "delta": None,
                        "rel_pct": None},
             "impact_note": "Impact could not be reliably estimated "
                            "from the available data."},
            "not-a-dict",
        ]
        kept, dropped = _validate_prescriptions(recs)
        self.assertEqual({r["id"] for r in kept}, {"good", "honest-note"})
        self.assertEqual(dropped, 3)


class RefreshAndScenarioCase(unittest.TestCase):
    """Re-analyze (refresh) is deterministic and run_scenario works live on
    a sqlite lever fixture."""

    @classmethod
    def setUpClass(cls):
        rng = random.Random(11)
        cls.rows = []
        for _ in range(120):
            p = rng.uniform(10, 100)
            cls.rows.append(
                (round(p, 2), round(2000 - 15 * p + rng.gauss(0, 40), 2)))
        cls.engine, _ = _sqlite({"shop": ("px REAL, sxs REAL", cls.rows)})
        cls.intel = SchemaIntelligence(_mapping({
            "shop": {"columns": [_col("px", "REAL"), _col("sxs", "REAL")],
                     "primary_key": []}}))

    def test_refresh_is_deterministic(self):
        a = analyze(self.engine, "sqlite", self.intel, "shop", "sxs",
                    driver="px")
        b = analyze(self.engine, "sqlite", self.intel, "shop", "sxs",
                    driver="px")
        self.assertEqual(a["status"], "completed")
        self.assertEqual([p["title"] for p in a["prescriptions"]],
                         [p["title"] for p in b["prescriptions"]])
        kinds = {p["action_type"] for p in a["prescriptions"]}
        self.assertIn("scenario", kinds)
        self.assertIn("diagnostics", a)
        self.assertIn("scenario_meta", a)

    def test_scenario_live(self):
        out = run_scenario(self.engine, "sqlite", self.intel, "shop", "sxs",
                           "px", [{"field": "px", "operation": "percent",
                                   "value": -10.0}])
        self.assertTrue(out["success"])
        sc = out["scenarios"][0]
        for k in ("baseline", "scenario", "delta"):
            self.assertIsNotNone(sc[k])
        self.assertTrue(out["assumptions"])


if __name__ == "__main__":
    unittest.main()
