"""Prescription display-safe contract tests (dynamic only).

Every schema below uses generated table/column names — nothing is hardcoded,
so the suite proves the display contract is derived from the live schema
mapping + engine evidence on any dataset:

  * display keys present on every prescription (title/action_text/
    outcome_text/confidence_label/why_summary) + details (evidence_pass/
    checks/numbers/limitations), additive over existing keys
  * outcome_text humanized + rounded ($312K, +10%, never >2 decimals)
  * why_summary plain: no r/p/n statistics
  * confidence_label derived from the existing confidence vocabulary
  * route responses carry no internal keys (scenario_label, dedupe_key,
    risk_reasons, fit internals, sql?) — stripped + logged server-side
  * refusal alternatives are computed (metric/driver/label, max 3) and valid:
    re-running each one returns a structured analyze result
  * dynamic: two sqlite fixtures with completely different column names
"""

import json
import random
import re
import sqlite3
import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine

from core.schema_intel import SchemaIntelligence
from prescription_engine import analyze


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


def _strong_lever_fixture(table, driver, metric, seed, base, slope,
                          lo=5.0, hi=60.0, noise=1.0, n=150):
    """Seeded strong driver->metric pair at money scale (dynamic names)."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x = rng.uniform(lo, hi)
        span = abs(base) or 1.0
        rows.append((round(x, 2),
                     round(base + slope * x + rng.gauss(0, span * noise), 2)))
    engine, _ = _sqlite({table: (f"{driver} REAL, {metric} REAL", rows)})
    intel = SchemaIntelligence(_mapping({
        table: {"columns": [_col(driver, "REAL"), _col(metric, "REAL")],
                "primary_key": []}}))
    return engine, intel


class DisplayContractCaseA(unittest.TestCase):
    """Fixture A column names: promo_spend -> gross_revenue."""

    @classmethod
    def setUpClass(cls):
        cls.table, cls.driver, cls.metric = (
            "alpha_store", "promo_spend", "gross_revenue")
        cls.engine, cls.intel = _strong_lever_fixture(
            cls.table, cls.driver, cls.metric, seed=4242,
            base=300000.0, slope=-2500.0, noise=0.03)
        cls.out = analyze(cls.engine, "sqlite", cls.intel,
                          cls.table, cls.metric, driver=cls.driver)

    def test_completed(self):
        self.assertEqual(self.out["status"], "completed")
        self.assertTrue(self.out["prescriptions"])

    def test_display_keys_present(self):
        for p in self.out["prescriptions"]:
            disp = p.get("display")
            self.assertIsInstance(disp, dict)
            for k in ("title", "action_text", "outcome_text",
                      "confidence_label", "why_summary"):
                self.assertTrue(str(disp.get(k) or "").strip(),
                                f"display.{k} missing on {p.get('id')}")
            self.assertIn(disp.get("confidence_label"),
                          ("High", "Moderate", "Early"))
            det = p.get("details")
            self.assertIsInstance(det, dict)
            self.assertIsInstance(det.get("evidence_pass"), bool)
            self.assertIsInstance(det.get("checks"), list)
            self.assertIsInstance(det.get("numbers"), dict)
            self.assertIsInstance(det.get("limitations"), list)
            # Top-level mirrors for the route allowlist.
            self.assertTrue(str(p.get("why_summary") or "").strip())
            self.assertIn(p.get("confidence_label"),
                          ("High", "Moderate", "Early"))
            self.assertIsInstance(p.get("evidence_pass"), bool)

    def test_existing_keys_kept(self):
        for p in self.out["prescriptions"]:
            for k in ("id", "title", "evidence", "impact", "confidence", "risk"):
                self.assertIn(k, p, f"existing key {k} removed")

    def test_outcome_rounded_money(self):
        for p in self.out["prescriptions"]:
            if str(p.get("action_type") or "").lower() != "scenario":
                continue
            oc = str(p["display"]["outcome_text"])
            self.assertNotRegex(oc, r"\d+\.\d{3,}",
                                f"more than 2 decimals in {oc!r}")
            self.assertIn("$", oc)  # money metric -> money-aware outcome
            self.assertRegex(oc, r"\$[\d.,]+[KM]")

    def test_why_summary_plain(self):
        for p in self.out["prescriptions"]:
            why = str(p["display"]["why_summary"])
            self.assertNotRegex(why, r"\br\s*=")
            self.assertNotRegex(why, r"\bp\s*=")
            self.assertNotRegex(why, r"\bn\s*=")

    def test_confidence_label_derived(self):
        from prescription_engine.engine import confidence_label_for
        for p in self.out["prescriptions"]:
            self.assertEqual(p["display"]["confidence_label"],
                             confidence_label_for(p.get("confidence")))


class DisplayContractCaseB(unittest.TestCase):
    """Fixture B: completely different column names (ad_budget -> net_sales)."""

    @classmethod
    def setUpClass(cls):
        cls.table, cls.driver, cls.metric = (
            "beta_shop", "ad_budget", "net_sales")
        cls.engine, cls.intel = _strong_lever_fixture(
            cls.table, cls.driver, cls.metric, seed=777,
            base=5000.0, slope=30.0, noise=0.03)
        cls.out = analyze(cls.engine, "sqlite", cls.intel,
                          cls.table, cls.metric, driver=cls.driver)

    def test_dynamic_names_both_complete(self):
        self.assertEqual(self.out["status"], "completed")
        self.assertTrue(self.out["prescriptions"])
        # Different physical names prove nothing is hardcoded.
        self.assertNotEqual(
            {DisplayContractCaseA.table, DisplayContractCaseA.driver,
             DisplayContractCaseA.metric},
            {self.table, self.driver, self.metric})

    def test_display_keys_present_and_rounded(self):
        for p in self.out["prescriptions"]:
            disp = p.get("display")
            self.assertIsInstance(disp, dict)
            for k in ("title", "action_text", "outcome_text",
                      "confidence_label", "why_summary"):
                self.assertTrue(str(disp.get(k) or "").strip())
            self.assertNotRegex(str(disp["outcome_text"]), r"\d+\.\d{3,}")
            self.assertNotRegex(str(disp["why_summary"]), r"\br\s*=")
            self.assertIn("$", str(disp["outcome_text"]))


class RoundingHelpersCase(unittest.TestCase):
    def test_money_percent_labels(self):
        from prescription_engine.engine import (
            _fmt_money_short, _fmt_pct_short, confidence_label_for)
        self.assertEqual(_fmt_money_short(312000), "$312K")
        self.assertEqual(_fmt_pct_short(10.0), "+10%")
        self.assertEqual(_fmt_pct_short(-2.46), "-2.5%")
        self.assertEqual(confidence_label_for("high"), "High")
        self.assertEqual(confidence_label_for("moderate"), "Moderate")
        self.assertEqual(confidence_label_for("medium"), "Moderate")
        self.assertEqual(confidence_label_for("low"), "Early")
        self.assertEqual(confidence_label_for("limited"), "Early")


class RefusalAlternativesCase(unittest.TestCase):
    """Single-measure table: honest refusal with COMPUTED alternatives."""

    @classmethod
    def setUpClass(cls):
        rng = random.Random(99)
        rows = [(round(rng.uniform(100, 900), 2), "only-zone")
                for _ in range(60)]
        cls.table, cls.metric, cls.dim = "gamma_log", "total_orders", "zone"
        cls.engine, _ = _sqlite(
            {cls.table: (f"{cls.metric} REAL, {cls.dim} TEXT", rows)})
        cls.intel = SchemaIntelligence(_mapping({
            cls.table: {"columns": [_col(cls.metric, "REAL"),
                                    _col(cls.dim, "TEXT")],
                        "primary_key": []}}))
        cls.out = analyze(cls.engine, "sqlite", cls.intel,
                          cls.table, cls.metric)

    def test_refusal_contract(self):
        self.assertEqual(self.out["status"], "no_prescription")
        refusal = self.out.get("refusal")
        self.assertIsInstance(refusal, dict)
        headline = str(refusal.get("headline") or "")
        self.assertTrue(headline.strip())
        self.assertNotIn("_", headline)  # dynamic humanized names only
        self.assertTrue(str(refusal.get("reason_code") or "").strip())
        alts = refusal.get("alternatives")
        self.assertIsInstance(alts, list)
        self.assertLessEqual(len(alts), 3)
        for a in alts:
            self.assertTrue(str(a.get("metric") or "").strip())
            self.assertTrue(str(a.get("driver") or "").strip())
            self.assertTrue(str(a.get("label") or "").strip())

    def test_alternatives_rerun_valid(self):
        for a in (self.out.get("refusal") or {}).get("alternatives", []):
            rerun = analyze(self.engine, "sqlite", self.intel,
                            self.table, a["metric"], driver=a["driver"])
            self.assertIsInstance(rerun, dict)
            self.assertIn(rerun.get("status"),
                          ("completed", "no_prescription"))
            if rerun.get("status") == "completed":
                for p in rerun["prescriptions"]:
                    self.assertIn("display", p)
                    self.assertIn("details", p)


class RouteStripCase(unittest.TestCase):
    """Route payloads carry no internal keys; display contract survives."""

    @classmethod
    def setUpClass(cls):
        cls.engine, cls.intel = _strong_lever_fixture(
            "delta_mart", "clearance_spend", "daily_revenue", seed=31337,
            base=90000.0, slope=-700.0, noise=0.03)
        cls.out = analyze(cls.engine, "sqlite", cls.intel,
                          "delta_mart", "daily_revenue",
                          driver="clearance_spend")
        assert cls.out["status"] == "completed"

    def test_no_internal_keys_in_route_response(self):
        from api.prescription_routes import _strip_prescription_response
        import copy
        out = copy.deepcopy(self.out)
        # Engine (pre-strip) still carries the internals the audit targets.
        pre = json.dumps(out, default=str)
        self.assertIn("scenario_label", pre)
        self.assertIn("risk_reasons", pre)
        cleaned, stripped = _strip_prescription_response(out)
        self.assertTrue(stripped)
        post = json.dumps(cleaned, default=str)
        self.assertNotIn("scenario_label", post)
        self.assertNotIn("risk_reasons", post)
        self.assertNotIn("dedupe_key", post)
        self.assertNotIn('"fit"', post)
        self.assertNotIn("sql", post.lower().replace("why is", ""))
        # Display-safe contract survives the strip.
        for p in cleaned["prescriptions"]:
            self.assertIn("display", p)
            self.assertIn("details", p)
            self.assertIn("why_summary", p)
            self.assertIn("confidence_label", p)
            self.assertIn("evidence_pass", p)
        # Scenario variant labels survive under the renamed key.
        for p in cleaned["prescriptions"]:
            for opt in p.get("scenario_options") or []:
                self.assertNotIn("scenario_label", opt)
                self.assertIn("label", opt)

    def test_stripped_keys_logged(self):
        from api.prescription_routes import _strip_prescription_response
        import copy
        import logging
        out = copy.deepcopy(self.out)
        with self.assertLogs("aria.prescription", level="DEBUG") as cm:
            _strip_prescription_response(out)
        self.assertTrue(any("scenario_label" in m or "risk_reasons" in m
                            for m in cm.output))


class IntegrityGuardsCase(unittest.TestCase):
    """Integrity: sanitized doc titles + always-on sampling provenance."""

    def test_doc_title_strips_markup(self):
        from prescription_engine.engine import _doc_title, analyze_document
        t = _doc_title({"section": "<script>alert(1)</script> Section title",
                        "content": "x", "id": "c1"})
        self.assertNotIn("<script>", t)
        self.assertNotIn(">", t.replace("…", ""))
        out = analyze_document(
            [{"id": "c1", "content": "risk of delay in shipments hurts delivery",
              "page_number": 2, "section": "<img src=x onerror=alert(1)> Hi"}],
            "delay risk", "<script>evil.doc</script> Report")
        title = out["prescriptions"][0]["title"]
        self.assertNotIn("<script>", title)
        self.assertNotIn("<img", title)
        self.assertNotIn("<script>", out["summary_plain"])

    def test_sampling_provenance_always_disclosed(self):
        from prescription_engine.engine import _details_for
        rec = {"evidence": {"metric": "m", "n": 100, "p_value": 0.001,
                            "effect_r": 0.5},
               "impact": {"baseline": 1, "scenario": 2, "delta": 1,
                          "rel_pct": 10},
               "confidence": "high", "action_type": "investigate"}
        d = _details_for(rec, sampling={"sampled": False, "sample_n": 42,
                                        "table_n": 42})
        self.assertIs(d.get("sampled"), False)
        self.assertEqual(d.get("sample_n"), 42)
        self.assertEqual(d.get("table_n"), 42)
        d2 = _details_for(rec, sampling={"sampled": True, "sample_n": 5000,
                                         "table_n": 20000})
        self.assertIs(d2.get("sampled"), True)
        self.assertEqual(d2.get("sample_n"), 5000)
        self.assertEqual(d2.get("table_n"), 20000)
        self.assertTrue(str(d2.get("sample_note") or "").strip())

    def test_saved_question_sanitized(self):
        from api.prescription_routes import _sanitize_saved_text
        self.assertNotIn("<script>", _sanitize_saved_text(
            "<script>alert(1)</script> hello", 500))


if __name__ == "__main__":
    unittest.main()
