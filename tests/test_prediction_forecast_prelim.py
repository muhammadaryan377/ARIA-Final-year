"""Forecast confidence-threshold + validation/preparation contract.

Covers: <50 preliminary, =50 full, >50 full, constant series, no-date,
missing dates, duplicate timestamps, irregular intervals, horizon override,
extended metrics, data-derived drivers, graceful errors.

Hermetic: pure-math pipeline on synthetic rows. No network, no LLM, no DB.
"""

import unittest

from prediction.agent import (
    ANALYZE_SUGGESTION,
    PRELIMINARY_MESSAGE,
    PRELIMINARY_RECOMMENDATION,
    PredictionAgent,
)


def monthly_rows(n, start_year=2021, start_month=1, fn=None):
    rows = []
    y, m = start_year, start_month
    for i in range(n):
        val = fn(i) if fn else 100.0 + i * 5.0
        rows.append({"d": f"{y}-{m:02d}-01", "amount": float(val)})
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return rows


def daily_rows(dates, fn=None):
    rows = []
    for i, d in enumerate(dates):
        val = fn(i) if fn else 100.0 + i * 2.0 + (i % 5)
        rows.append({"d": d, "amount": float(val)})
    return rows


class TestPreliminaryThreshold(unittest.TestCase):
    def test_under_50_is_preliminary(self):
        out = PredictionAgent().forecast(
            {"data": monthly_rows(12), "user_goal": "forecast amount"})
        self.assertTrue(out.get("ok"))
        self.assertTrue(out.get("points"), "preliminary must still forecast")
        self.assertEqual(out.get("forecast_stage"), "preliminary")
        self.assertEqual(out.get("reliability"), "limited")
        self.assertEqual(out.get("message"), PRELIMINARY_MESSAGE)
        self.assertEqual(out.get("n_obs"), 12)
        self.assertEqual(out.get("recommendation"), PRELIMINARY_RECOMMENDATION)
        # confidence interval present and ordered
        self.assertTrue(out.get("forecast"))
        for f in out["forecast"]:
            self.assertLessEqual(f["lower"], f["value"])
            self.assertLessEqual(f["value"], f["upper"])

    def test_exactly_50_is_full(self):
        out = PredictionAgent().forecast(
            {"data": monthly_rows(50), "user_goal": "forecast amount"})
        self.assertTrue(out.get("ok"))
        self.assertTrue(out.get("points"))
        self.assertEqual(out.get("forecast_stage"), "full")
        self.assertIn(out.get("reliability"), ("high", "moderate", "limited"))
        self.assertEqual(out.get("n_obs"), 50)
        self.assertIsNone(out.get("message"))

    def test_over_50_is_full(self):
        out = PredictionAgent().forecast(
            {"data": monthly_rows(60), "user_goal": "forecast amount"})
        self.assertTrue(out.get("ok"))
        self.assertEqual(out.get("forecast_stage"), "full")
        self.assertIn(out.get("reliability"), ("high", "moderate", "limited"))
        self.assertEqual(out.get("n_obs"), 60)


class TestUnforecastable(unittest.TestCase):
    def test_constant_series_refuses(self):
        rows = monthly_rows(12, fn=lambda i: 42.0)
        out = PredictionAgent().forecast(
            {"data": rows, "user_goal": "forecast amount"})
        self.assertFalse(out.get("ok"))
        self.assertFalse(out.get("valid"))
        self.assertEqual(out.get("forecast"), [], "NEVER fabricate")
        self.assertEqual(out.get("points"), [])
        self.assertTrue(out.get("reason"))
        self.assertTrue(out.get("missing_field"))
        self.assertEqual(out.get("suggestion"), ANALYZE_SUGGESTION)

    def test_no_date_refuses(self):
        rows = [{"store": s, "total": float(v)} for s, v in
                (("a", 10.0), ("b", 20.0), ("c", 15.0), ("d", 30.0),
                 ("e", 25.0), ("f", 12.0))]
        out = PredictionAgent().forecast(
            {"data": rows, "user_goal": "forecast total"})
        self.assertFalse(out.get("ok"))
        self.assertFalse(out.get("valid"))
        self.assertEqual(out.get("forecast"), [])
        self.assertEqual(out.get("missing_field"), "date")
        self.assertEqual(out.get("suggestion"), ANALYZE_SUGGESTION)
        self.assertIn(out.get("reliability"),
                      ("not_applicable", "no_data", "insufficient_data"))

    def test_empty_is_graceful(self):
        out = PredictionAgent().forecast({"data": []})
        self.assertFalse(out.get("ok"))
        self.assertFalse(out.get("valid"))
        self.assertEqual(out.get("suggestion"), ANALYZE_SUGGESTION)
        dumped = str(out)
        self.assertNotIn("Traceback", dumped)
        self.assertNotIn("File \"", dumped)


class TestValidationPreparation(unittest.TestCase):
    def test_frequency_derived_monthly(self):
        out = PredictionAgent().forecast({"data": monthly_rows(12)})
        prep = out.get("preparation") or {}
        self.assertEqual(prep.get("frequency"), "monthly")
        self.assertEqual(prep.get("target_column"), "amount")
        self.assertEqual(prep.get("time_column"), "d")
        self.assertEqual(prep.get("n_obs"), 12)
        self.assertIn("median_delta_days", prep.get("frequency_detection") or {})

    def test_missing_dates_detected(self):
        rows = monthly_rows(8)
        rows.pop(5)  # one missing month
        out = PredictionAgent().forecast({"data": rows})
        prep = out.get("preparation") or {}
        self.assertGreaterEqual(prep.get("missing_periods", 0), 1)
        self.assertEqual(len(out["historical"]), 8)

    def test_duplicates_merged(self):
        rows = monthly_rows(8)
        rows.append(dict(rows[3]))
        out = PredictionAgent().forecast({"data": rows})
        prep = out.get("preparation") or {}
        self.assertGreaterEqual(prep.get("duplicate_timestamps_merged", 0), 1)
        diag = out["model_notes"].get("series_diagnostics", {})
        self.assertEqual(diag.get("duplicate_timestamps_merged"), 1)

    def test_irregular_intervals_flagged(self):
        from datetime import date, timedelta
        base = date(2024, 1, 1)
        dates = [(base + timedelta(days=i)).isoformat() for i in range(10)]
        dates += [(base + timedelta(days=70 + i)).isoformat() for i in range(6)]
        out = PredictionAgent().forecast({"data": daily_rows(dates)})
        prep = out.get("preparation") or {}
        self.assertTrue(prep.get("irregular_intervals"))
        self.assertTrue(out.get("ok"), "irregular still forecasts, flagged")


class TestModelSelectionAndMetrics(unittest.TestCase):
    def test_model_selected_and_candidates_scored(self):
        out = PredictionAgent().forecast({"data": monthly_rows(12)})
        self.assertTrue(out.get("model_selected"))
        cands = out.get("candidates") or []
        self.assertTrue(len(cands) >= 2)
        for c in cands:
            self.assertIn("model", c)
            self.assertIn("mae", c)
            self.assertIn("rmse", c)
        # winner is first
        self.assertEqual(cands[0]["model"], out["model_selected"])
        # time-series validation, never random split
        sel = (out.get("details") or {}).get("selection") or {}
        self.assertIn("holdout", str(sel.get("selection_metric", "")))

    def test_extended_metrics_present(self):
        out = PredictionAgent().forecast({"data": monthly_rows(30)})
        for key in ("mae", "rmse", "mape"):
            self.assertIn(key, out.get("metrics") or {}, f"missing {key}")
        ext = ((out.get("details") or {}).get("metrics_extended")) or {}
        for key in ("mae", "rmse", "mape", "smape", "wape", "bias"):
            self.assertIn(key, ext, f"missing extended {key}")
        self.assertIn(out.get("reliability"), ("high", "moderate", "limited"))
        acc = out.get("accuracy") or {}
        self.assertIn("summary", acc)
        self.assertTrue(acc.get("summary"))
        self.assertIn("details", out)

    def test_drivers_derived_from_data(self):
        rows = monthly_rows(20)
        for i, r in enumerate(rows):
            r["region"] = "north" if i % 3 else "south"
        out = PredictionAgent().forecast({"data": rows})
        drivers = out.get("drivers") or []
        self.assertGreaterEqual(len(drivers), 3)
        for d in drivers:
            self.assertIn("name", d)
            self.assertIn("evidence", d)
            self.assertIn("metric_value", d)
            self.assertTrue(d["evidence"], "evidence must be concrete, not generic")
        names = " ".join(d["name"] for d in drivers).lower()
        self.assertIn("trend", names)
        self.assertIn("volatility", names)


class TestHorizon(unittest.TestCase):
    def test_default_suggested_dynamically(self):
        out12 = PredictionAgent().forecast({"data": monthly_rows(12)})
        out60 = PredictionAgent().forecast({"data": monthly_rows(60)})
        # dynamic formula: min(12, max(3, n // 4))
        self.assertEqual(out12.get("horizon_suggested"), min(12, max(3, 12 // 4)))
        self.assertEqual(out60.get("horizon_suggested"), min(12, max(3, 60 // 4)))
        self.assertNotEqual(out12.get("horizon_suggested"),
                            out60.get("horizon_suggested"),
                            "suggestion must vary with history length")

    def test_user_override_respected(self):
        out = PredictionAgent().forecast({"data": monthly_rows(12)}, horizon=5)
        self.assertEqual(len(out["forecast"]), 5)
        self.assertEqual(out.get("horizon"), 5)


if __name__ == "__main__":
    unittest.main()
