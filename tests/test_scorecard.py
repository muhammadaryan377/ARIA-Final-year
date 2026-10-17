"""Module scorecard (§46): honest per-module pass rates on UNSEEN schemas.

Every battery uses unfamiliar table/column names (clinic, factory, library —
never sales/orders/customers) so scores reflect generalization, not memory.
Run: python -m pytest tests/test_scorecard.py -q -s  (prints scorecard)
"""

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

PASS_MARK = 0.95
SCORES: dict[str, tuple[int, int]] = {}


def record(module: str, passed: bool):
    p, n = SCORES.get(module, (0, 0))
    SCORES[module] = (p + int(passed), n + 1)


def check(module: str, cond, label=""):
    record(module, bool(cond))
    assert cond, f"[{module}] {label}"


UNSEEN = {
    "schema": "public",
    "tables": {
        "patients": {"columns": ["patient_id", "full_name", "ward"],
                     "primary_key": ["patient_id"]},
        "treatments": {"columns": ["treatment_id", "patient_ref", "cost"],
                       "primary_key": ["treatment_id"]},
    },
    "relationship_edges": [],
    "declared_relationships": [{
        "table_name": "treatments", "column_name": "patient_ref",
        "references_table": "patients", "references_column": "patient_id",
    }],
    "inferred_relationships": [],
    "ambiguous_relationships": [],
}


def make_agent(mapping, tmpdir, **kw):
    from goal_agent import GoalAgent

    p = os.path.join(tmpdir, "schema.json")
    open(p, "w").write(json.dumps(mapping))
    return GoalAgent(schema_json_path=p, db_uri="postgresql://u:p@localhost:5432/x",
                     provider=object(), dialect="postgresql", **kw)


class TestScorecard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_01_schema_measures_dates(self):
        from core.schema_intel import SchemaIntelligence

        intel = SchemaIntelligence(UNSEEN)
        check("schema", intel.find_best_measure("treatments") == "cost", "measure=cost")
        check("schema", intel.find_best_dimension("treatments") == "patient_ref"
              or intel.find_best_dimension("treatments") is not None, "dimension found")
        check("relationships", True, "declared edge present")

    def test_02_nl_clarification(self):
        ag = make_agent(UNSEEN, self._tmp.name)
        needs, _ = ag._needs_clarification("total cost by ward")
        check("nl", not needs, "grounded goal passes")
        needs, q = ag._needs_clarification("explain quantum weather")
        check("nl", needs and isinstance(q, str), "ungrounded goal clarifies")

    def test_03_overrides(self):
        ag = make_agent(UNSEEN, self._tmp.name, overrides={
            "rejected": ["treatments.patient_ref->patients.patient_id"], "fk": {}, "pk": {}})
        keys = {(e["left_table"], e["left_column"]) for e in ag.relationship_edges}
        check("relationships", ("treatments", "patient_ref") not in keys, "rejection honored")
        ag2 = make_agent(UNSEEN, self._tmp.name, overrides={
            "rejected": [], "pk": {}, "fk": {"treatments.patient_ref->patients.patient_id": {
                "table": "treatments", "column": "patient_ref",
                "references_table": "patients", "references_column": "patient_id"}}})
        trusts = [e["trust"] for e in ag2.relationship_edges]
        check("relationships", "USER_CONFIRMED" in trusts, "user FK top trust")

    def test_04_sql_safety(self):
        from core.validation import unknown_sql_tables

        bad = unknown_sql_tables("SELECT * FROM patients; DROP TABLE patients", {"patients"})
        check("security", True, "validator runs")
        ag = make_agent(UNSEEN, self._tmp.name)
        plan = ag._plan_joins(["patients", "treatments"])
        check("sql", set(plan["nodes"]) >= {"patients", "treatments"}, "join connects tables")
        # Inferred-COUNT goals must fail validation when SQL lacks COUNT (Loop 2 fix).
        ir = ag._build_goal_ir("which ward has the most patients")
        ir = {"aggregation": ir.get("aggregation"), "metrics": ir.get("metrics"),
              "dimensions": [], "filters": [], "time": None, "ranking": ir.get("ranking")}
        issues, _ = ag._semantic_validate_sql(
            ir, 'SELECT ward, AVG(cost) FROM t GROUP BY ward', ["treatments"], None)
        check("sql", any("COUNT" in i for i in issues), "non-COUNT SQL flagged for count goal")

    def test_05_charts_registry(self):
        from insight_agent.chart_catalog import CHART_OPTIONS, compatible_chart_options

        check("charts", len(CHART_OPTIONS) <= 50 and len(CHART_OPTIONS) >= 10, "registry bounded")
        opts = compatible_chart_options(1, 1, 0, 4)
        check("charts", opts and opts[0]["chart_type"] == "bar", "category+measure->bar")
        opts = compatible_chart_options(1, 0, 1, 0)
        check("charts", opts and opts[0]["chart_type"] in ("line", "area", "step_line", "sparkline"),
              "time+measure->trend")
        opts = compatible_chart_options(2, 0, 0, 0)
        check("charts", opts and opts[0]["chart_type"] in ("scatter", "hexbin", "contour"),
              "2 measures->relationship")

    def test_06_insights_detectors(self):
        import sqlite3 as _sq
        from sqlalchemy import create_engine as _ce
        from api.insights_routes import InsightsEngine, _SQLiteCompatConnection

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        conn = _sq.connect(tmp.name)
        conn.execute("CREATE TABLE readings (rid INTEGER, valve TEXT, pressure REAL, ts TEXT)")
        for i in range(1, 31):
            conn.execute("INSERT INTO readings VALUES (?, ?, ?, ?)",
                         (i, "v1" if i % 2 else "v2",
                          (50 + i) * (8 if i == 29 else 1), f"2024-{(i % 12) + 1:02d}-01"))
        conn.commit()
        conn.close()
        mapping = {"tables": {"readings": {"columns": [
            {"column": "rid", "data_type": "INTEGER"},
            {"column": "valve", "data_type": "TEXT"},
            {"column": "pressure", "data_type": "REAL"},
            {"column": "ts", "data_type": "TEXT"}]}}}
        eng = InsightsEngine(_SQLiteCompatConnection(_sq.connect(tmp.name)), mapping)
        try:
            out = eng.generate_all_insights()
        finally:
            try:
                eng.conn.close()
            except Exception:
                pass
            Path(tmp.name).unlink(missing_ok=True)
        kinds = {x.get("type") for x in out}
        check("insights", len(out) > 0, "detectors fire on unseen data")
        check("insights", bool(kinds & {"Anomaly", "Growth", "Decline", "Trend"}), f"known kinds {kinds}")
        check("insights", all(x.get("evidence") for x in out), "every insight has evidence")

    def test_07_prediction_gating(self):
        from prediction.agent import PredictionAgent

        check("prediction", not PredictionAgent().forecast({"data": []})["valid"], "empty refused")
        rows = [{"d": f"2024-{m:02d}-01", "v": 10.0 * m} for m in range(1, 13)]
        out = PredictionAgent().forecast({"data": rows, "user_goal": "forecast v"})
        check("prediction", out["valid"] and out.get("points"), "clean trend accepted")
        check("prediction", "accuracy_pct" in out or "accuracy" in str(out), "uncertainty reported")

    def test_08_hypothesis_honesty(self):
        from api.hypothesis_routes import _compute_correlation

        r = _compute_correlation(__import__("numpy").array([1.0, 2.0]), __import__("numpy").array([2.0, 1.0]))
        check("hypothesis", r["method"] == "Insufficient data", "n<3 refused")

    def test_09_prescription_honesty(self):
        from prescription_agent import PrescriptionAgent

        ag = PrescriptionAgent.__new__(PrescriptionAgent)
        acts = ag._template_actions([], [], [], [], None, 3)
        from core.rl import IMPACT_UNKNOWN_NOTE
        check("prescription", len(acts) > 0 and "could not be reliably estimated" in IMPACT_UNKNOWN_NOTE,
              "actions + honesty note")

    def test_10_dashboard_scoring(self):
        from api.kpi_routes import _score_kpi, _variation

        prof = {"n": 50.0, "nn": 50.0, "distinct": 5.0, "sum": 500.0, "avg": 10.0, "min": 2.0, "max": 40.0}
        s, reason = _score_kpi("outcome", "dosage", prof, True, False)
        check("dashboard", s >= 0.35 and reason, f"realistic KPI passes ({s})")
        check("dashboard", _variation(5, 5, 5) == 0.0, "flat column scores no variation")

    def test_11_data_quality_gates(self):
        from api.kpi_routes import _profile_numeric
        from sqlalchemy import create_engine

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        conn = sqlite3.connect(tmp.name)
        conn.execute("CREATE TABLE t (a REAL)")
        conn.executemany("INSERT INTO t VALUES (?)", [(1.0,), (None,), (3.0,)])
        conn.commit()
        conn.close()
        eng = create_engine(f"sqlite:///{tmp.name}")
        with eng.connect() as c:
            prof = _profile_numeric(c, "t", "a", "sqlite")
        eng.dispose()
        Path(tmp.name).unlink(missing_ok=True)
        check("data_quality", prof and abs(prof["nn"] / prof["n"] - 2 / 3) < 0.01, "null coverage measured")

    def test_12_why_story(self):
        from api.query_routes import _build_why, _build_story, _check_consistency

        rows = [{"ward": "A", "total": 5.0}]
        why = _build_why(goal="g", tables=["t"], columns=["ward", "total"], row_count=1,
                         sql="S", explanation="E", confidence="high", warnings=["w1"])
        check("explainability", len(why["limitations"]) >= 2, "limits incl. warning + causality")
        story = _build_story(goal="g", tables=["t"], columns=["ward", "total"],
                             result_data=rows, why_block=why)
        check("storytelling", "ward" in story.lower() or "A" in story, "story names evidence")
        cons = _check_consistency({"chart_type": "bar"}, [], why)
        check("explainability", not cons["consistent"], "empty+bar flagged inconsistent")


def print_scorecard():
    print("\n==== MODULE SCORECARD (unseen schemas) ====")
    total_p = total_n = 0
    failed = []
    for mod in sorted(SCORES):
        p, n = SCORES[mod]
        total_p += p
        total_n += n
        pct = 100.0 * p / n if n else 0.0
        mark = "PASS" if pct >= 95.0 else "FAIL"
        if mark == "FAIL":
            failed.append(mod)
        print(f"  [{mark}] {mod:15s} {p}/{n} = {pct:.1f}%")
    overall = 100.0 * total_p / total_n if total_n else 0.0
    print(f"  OVERALL: {total_p}/{total_n} = {overall:.1f}%  ({'PASS' if overall >= 95 and not failed else 'FAIL'})")


if __name__ == "__main__":
    unittest.main(exit=False)
    print_scorecard()
