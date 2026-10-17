"""Tests for the independent Intelligence validation layer.

Covers: tampered total / change_pct / fabricated previous detection,
birth_date date_sanity (pure check), happy-path on a temp sqlite dataset,
fail-closed wiring in run_explore, and warn-only wiring in run_whatif.
"""

import copy
import json
import sqlite3

import pytest

EXPECTED_CHECKS = {"total_check", "previous_check", "buckets_sum",
                   "breakdown_sum", "arithmetic", "date_sanity",
                   "sparkline_check"}

MAPPING = {
    "tables": {
        "orders": {
            "columns": [
                {"column": "order_id", "data_type": "integer", "nullable": False},
                {"column": "revenue", "data_type": "numeric", "nullable": True},
                {"column": "quantity", "data_type": "integer", "nullable": True},
                {"column": "product_category", "data_type": "text", "nullable": True},
                {"column": "order_date", "data_type": "date", "nullable": True},
                {"column": "region", "data_type": "text", "nullable": True},
            ],
            "primary_key": ["order_id"],
            "row_count": 100,
        },
    }
}

CUR = ("2024-01-01", "2024-12-31")
PREV = ("2023-01-01", "2023-12-31")


def _seed_db(db_path, years=(2023, 2024)):
    conn = sqlite3.connect(str(db_path))
    conn.execute("DROP TABLE IF EXISTS orders")
    conn.execute("""CREATE TABLE orders (
        order_id INTEGER PRIMARY KEY,
        revenue REAL, quantity INTEGER,
        product_category TEXT, order_date TEXT, region TEXT)""")
    oid = 1
    for y in years:
        for m in range(1, 13):
            for cat, base, step in (("Bikes", 100, 10), ("Clothing", 50, 5)):
                rev = (base if y == 2023 else base + 20) + m * step
                conn.execute(
                    "INSERT INTO orders VALUES (?,?,?,?,?,?)",
                    (oid, float(rev), 2, cat, f"{y}-{m:02d}-15", "West"))
                oid += 1
    conn.commit()
    conn.close()


@pytest.fixture
def ctx(tmp_path):
    """Temp sqlite dataset (2023+2024) + session wired to mapping + db."""
    db_path = tmp_path / "val.db"
    _seed_db(db_path)
    schema_path = tmp_path / "schema_mapping_latest.json"
    schema_path.write_text(json.dumps(MAPPING), encoding="utf-8")
    session = {"provider": "local", "source_type": "csv",
               "db_uri": f"sqlite:///{db_path}",
               "db_name": "val", "schema_path": str(schema_path)}
    from sqlalchemy import create_engine
    engine = create_engine(f"sqlite:///{db_path}")
    return {"session": session, "engine": engine}


def _vargs(engine):
    """Validator query context matching the ctx dataset (single-table)."""
    return {
        "engine": engine, "dialect": "sqlite",
        "qm": '"revenue"', "qt": '"orders"',
        "wc": ' WHERE "order_date" >= :_vc_s AND "order_date" <= :_vc_e',
        "wp": ' WHERE "order_date" >= :_vp_s AND "order_date" <= :_vp_e',
        "params": {"_vc_s": CUR[0], "_vc_e": CUR[1],
                   "_vp_s": PREV[0], "_vp_e": PREV[1]},
        "date_col": "order_date", "qd": '"order_date"',
        "cur_start": "2024-01-01", "cur_end": "2024-12-31",
        "prev_start": "2023-01-01", "prev_end": "2023-12-31",
        "do_compare": True,
    }


def _validate(engine, result, **over):
    from services.intelligence_validation import validate_explore
    a = _vargs(engine)
    a.update(over)
    return validate_explore(a["engine"], a["dialect"], a["qm"], a["qt"],
                            a["wc"], a["wp"], a["params"], result,
                            date_col=a["date_col"], qd=a["qd"],
                            cur_start=a["cur_start"], cur_end=a["cur_end"],
                            prev_start=a["prev_start"], prev_end=a["prev_end"],
                            do_compare=a["do_compare"])


def _failed_names(vr):
    return {c["name"] for c in vr["checks"] if not c["passed"]}


class TestValidationLayer:
    def test_happy_path_passes(self, ctx):
        from services.intelligence_service import run_explore
        res = run_explore(ctx["session"], {"metric": "revenue",
                                           "breakdown": "product_category"})
        assert res["ok"] is True, res.get("detail")
        assert res.get("validation", {}).get("passed") is True
        assert EXPECTED_CHECKS <= set(res["validation"]["checks"])

    def test_tampered_total_detected(self, ctx):
        from services.intelligence_service import run_explore
        res = run_explore(ctx["session"], {"metric": "revenue",
                                           "breakdown": "product_category"})
        assert res["ok"] is True
        bad = copy.deepcopy(res)
        bad["total"] = round(float(bad["total"]) * 2, 2)
        vr = _validate(ctx["engine"], bad)
        assert vr["passed"] is False
        assert "total_check" in _failed_names(vr)

    def test_tampered_change_pct_detected(self, ctx):
        from services.intelligence_service import run_explore
        res = run_explore(ctx["session"], {"metric": "revenue",
                                           "breakdown": "product_category"})
        assert res["ok"] is True
        bad = copy.deepcopy(res)
        assert bad["rows"] and bad["rows"][0]["change_pct"] is not None
        bad["rows"][0]["change_pct"] = round(
            float(bad["rows"][0]["change_pct"]) + 25.0, 2)
        vr = _validate(ctx["engine"], bad)
        assert vr["passed"] is False
        assert "arithmetic" in _failed_names(vr)

    def test_fabricated_previous_detected(self, tmp_path):
        """Prior window empty (only-2024 data) but previous_total nonzero."""
        db_path = tmp_path / "cur_only.db"
        _seed_db(db_path, years=(2024,))
        schema_path = tmp_path / "schema_mapping_latest.json"
        schema_path.write_text(json.dumps(MAPPING), encoding="utf-8")
        session = {"provider": "local", "source_type": "csv",
                   "db_uri": f"sqlite:///{db_path}",
                   "db_name": "cur", "schema_path": str(schema_path)}
        from sqlalchemy import create_engine
        engine = create_engine(f"sqlite:///{db_path}")
        from services.intelligence_service import run_explore
        res = run_explore(session, {"metric": "revenue",
                                    "breakdown": "product_category"})
        assert res["ok"] is True
        bad = copy.deepcopy(res)
        # Fabricate a nonzero previous while keeping arithmetic consistent
        # so that previous_check is the check that fires.
        bad["previous_total"] = 12345.0
        bad["change"] = round(float(bad["total"]) - 12345.0, 2)
        bad["change_pct"] = round(
            (float(bad["total"]) - 12345.0) / 12345.0 * 100, 2)
        vr = _validate(engine, bad)
        assert vr["passed"] is False
        assert "previous_check" in _failed_names(vr)

    def test_birth_date_window_detected_pure(self):
        from services.intelligence_validation import (
            check_date_sanity, is_suspicious_date_col)
        assert is_suspicious_date_col("birth_date") is True
        assert is_suspicious_date_col("order_date") is False
        ok, detail = check_date_sanity(
            "birth_date", "1966-01-01", "1966-12-31",
            "1965-01-01", "1965-12-31", "2023-01-01", "2024-12-31")
        assert ok is False
        assert "birth" in detail.lower() or "hire" in detail.lower()
        ok2, _ = check_date_sanity(
            "order_date", "2024-01-01", "2024-12-31",
            "2023-01-01", "2023-12-31", "2023-01-01", "2024-12-31")
        assert ok2 is True

    def test_fail_closed_wiring(self, ctx, monkeypatch):
        """Forced validator failure flips run_explore to VALIDATION_FAILED."""
        import services.intelligence_service as svc
        import services.intelligence_validation as val

        def _boom(*a, **k):
            return {"passed": False, "checks": [
                {"name": "total_check", "passed": False, "detail": "boom"}],
                "corrected": False}

        monkeypatch.setattr(val, "validate_explore", _boom)
        monkeypatch.setattr(svc, "validate_explore", _boom,
                            raising=False)
        # run_explore imports validate_explore lazily from the
        # services.intelligence_validation module object, so patch it there.
        res = svc.run_explore(ctx["session"], {"metric": "revenue",
                                               "breakdown": "product_category"})
        assert res["ok"] is False
        assert res.get("code") == "VALIDATION_FAILED"
        assert res.get("validation", {}).get("passed") is False
        assert "total_check" in res["validation"].get("failed", [])

    def test_whatif_validation_warn_only(self, ctx):
        from services.intelligence_service import run_whatif
        res = run_whatif(ctx["session"], {"variable": "revenue",
                                          "change_type": "increase",
                                          "change_value": 10})
        assert res["ok"] is True
        assert res.get("validation", {}).get("passed") is True
        # Tampered what-if is detected by the pure checker.
        from services.intelligence_validation import validate_whatif
        bad = copy.deepcopy(res)
        bad["projected_value"] = float(bad["projected_value"]) * 3
        vr = validate_whatif(bad)
        assert vr["passed"] is False


class TestTruncationScope:
    """FAILURE A regression: capped breakdowns carry breakdown_scope;
    truncated -> containment, non-truncated -> strict equality (unweakened)."""

    def test_check_truncated_rows_pure(self):
        from services.intelligence_validation import _check_truncated_rows
        rows = [
            {"category": "A", "current": 60.0, "previous": 50.0},
            {"category": "B", "current": 30.0, "previous": 40.0},
        ]
        ok, detail = _check_truncated_rows(rows, 100.0, 100.0)
        assert ok is True, detail
        assert "top share" in detail
        # subset of total still passes
        ok, _ = _check_truncated_rows(rows, 200.0, 200.0)
        assert ok is True
        # unsorted rows fail
        ok, detail = _check_truncated_rows(list(reversed(rows)), 100.0, 100.0)
        assert ok is False
        assert "sorted" in detail.lower()
        # rows exceeding the total fail
        ok, _ = _check_truncated_rows(rows, 50.0, 50.0)
        assert ok is False
        # previous subset exceeding previous_total fails
        ok, _ = _check_truncated_rows(rows, 200.0, 10.0)
        assert ok is False
        # previous None skips the previous leg
        ok, detail = _check_truncated_rows(rows, 200.0, None)
        assert ok is True
        assert "skipped" in detail.lower()

    def test_truncated_scope_passes_validator(self, ctx):
        from services.intelligence_service import run_explore
        res = run_explore(ctx["session"], {"metric": "revenue",
                                           "breakdown": "product_category"})
        assert res["ok"] is True
        assert res["breakdown_scope"]["truncated"] is False
        bad = copy.deepcopy(res)
        # Simulate a capped top-N: same full-scope totals, truncated flag.
        bad["breakdown_scope"] = {"returned": len(bad["rows"]),
                                  "total_groups": len(bad["rows"]) + 40,
                                  "truncated": True}
        vr = _validate(ctx["engine"], bad)
        assert vr["passed"] is True, _failed_names(vr)

    def test_nontruncated_equality_still_strict(self, ctx):
        from services.intelligence_service import run_explore
        res = run_explore(ctx["session"], {"metric": "revenue",
                                           "breakdown": "product_category"})
        assert res["ok"] is True
        bad = copy.deepcopy(res)
        bad["rows"][0]["current"] = round(float(bad["rows"][0]["current"]) + 100.0, 2)
        vr = _validate(ctx["engine"], bad)
        assert vr["passed"] is False
        assert "breakdown_sum" in _failed_names(vr)


class TestTolerantDates:
    """FAILURE B regression: ONE shared parser covers odd date storage."""

    def test_shared_parser_shapes(self):
        from services.intelligence_validation import to_date
        from services import intelligence_service as svc
        # ONE parser: the service delegates to the validation module.
        assert svc._shared_to_date is to_date
        assert to_date("2024-03-15") == __import__("datetime").date(2024, 3, 15)
        assert to_date("05-02-2010").isoformat() == "2010-05-02"  # MM-DD-YYYY
        assert to_date("05/02/2010").isoformat() == "2010-05-02"  # MM/DD/YYYY
        assert to_date("05-Feb-2010").isoformat() == "2010-02-05"  # DD-Mon-YYYY
        assert to_date("February 5, 2010").isoformat() == "2010-02-05"
        assert to_date("20100205").isoformat() == "2010-02-05"  # YYYYMMDD
        assert to_date(1265222400).year == 2010  # epoch seconds
        assert to_date(1265222400000).year == 2010  # epoch millis
        assert to_date("not a date") is None
        assert to_date(None) is None
        # service and validator agree on every shape
        for v in ("05-02-2010", "05-Feb-2010", "20100205", 1265222400):
            assert svc._shared_to_date(v) == to_date(v)

    def test_text_dates_bucketed_to_12_points(self, tmp_path):
        """TEXT MM-DD-YYYY storage -> Python bucketing, 12 monthly points."""
        import json as _json
        import sqlite3 as _sqlite3
        db_path = tmp_path / "textdates.db"
        conn = _sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE sales (id INTEGER PRIMARY KEY, amount REAL, d TEXT)")
        oid = 1
        for y in (2023, 2024):
            for m in range(1, 13):
                conn.execute("INSERT INTO sales VALUES (?,?,?)",
                             (oid, float(100 + m), f"{m:02d}-15-{y}"))
                oid += 1
        conn.commit()
        conn.close()
        mapping = {"tables": {"sales": {
            "columns": [
                {"column": "id", "data_type": "integer", "nullable": False},
                {"column": "amount", "data_type": "double precision", "nullable": True},
                {"column": "d", "data_type": "text", "nullable": True},
            ],
            "primary_key": ["id"], "row_count": 24}}}
        schema_path = tmp_path / "schema_mapping_latest.json"
        schema_path.write_text(_json.dumps(mapping), encoding="utf-8")
        session = {"provider": "local", "source_type": "csv",
                   "db_uri": f"sqlite:///{db_path}",
                   "db_name": "td", "schema_path": str(schema_path)}
        from services.intelligence_service import run_explore
        res = run_explore(session, {"metric": "amount"})
        assert res["ok"] is True, res.get("detail")
        assert len(res["rows"]) == 1  # overall mode: single row
        assert len((res.get("trend") or {}).get("points", [])) == 12
        assert res.get("validation", {}).get("passed") is True


class TestOverallMode:
    """FAILURE E regression: no breakdown -> single overall row + 12 points."""

    def test_overall_mode_end_to_end(self, ctx):
        from services.intelligence_service import run_explore
        res = run_explore(ctx["session"], {"metric": "revenue"})
        assert res["ok"] is True, res.get("detail")
        assert len(res.get("rows", [])) == 1
        assert len((res.get("trend") or {}).get("points", [])) == 12
        assert res.get("validation", {}).get("passed") is True
        assert res["breakdown_scope"] == {"returned": 1, "total_groups": 1,
                                          "truncated": False}


class TestCountFallback:
    """FAILURE D regression: zero metrics -> synthetic COUNT(*) metric."""

    def test_count_meta_and_explore(self, tmp_path):
        import json as _json
        import sqlite3 as _sqlite3
        db_path = tmp_path / "garbage.db"
        conn = _sqlite3.connect(str(db_path))
        conn.execute('CREATE TABLE "day_sell" ("date_zn_sb_tax_marza" REAL)')
        for v in (8.0, 62.0, 8.0, 15.0, 62.0, 8.0, 9.0):
            conn.execute('INSERT INTO "day_sell" VALUES (?)', (v,))
        conn.commit()
        conn.close()
        mapping = {"tables": {"day_sell": {
            "columns": [
                {"column": "date_zn_sb_tax_marza", "data_type": "double precision",
                 "nullable": True},
            ],
            "row_count": 7}}}
        schema_path = tmp_path / "schema_mapping_latest.json"
        schema_path.write_text(_json.dumps(mapping), encoding="utf-8")
        session = {"provider": "local", "source_type": "csv",
                   "db_uri": f"sqlite:///{db_path}",
                   "db_name": "gb", "schema_path": str(schema_path)}
        from services.intelligence_service import (
            get_intelligence_meta, run_explore)
        meta = get_intelligence_meta(session)
        assert meta["metrics"] and meta["metrics"][0]["column"] == "__count__"
        assert "record counts" in (meta.get("hint") or "").lower()
        res = run_explore(session, {"metric": "__count__"})
        assert res["ok"] is True, res.get("detail")
        assert float(res["total"]) == 7.0
        assert res["previous_total"] is None  # snapshot honesty
        assert res.get("validation", {}).get("passed") is True
