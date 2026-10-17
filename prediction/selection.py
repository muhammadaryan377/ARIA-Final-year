"""Intelligent model selection layer for ARIA forecasting (n.txt item 4).

Rule-based shortlist from series diagnostics, then honest holdout backtest
of the competing models — the winner is picked on holdout error and the
result carries the statement "based on best holdout".

Rules (applied in order):
- very little history (< 8 points, or < 2 full seasons when seasonal)
  -> naive / seasonal-naive (+ mean/drift as sanity baselines).
- clear seasonality (autocorrelation at the seasonal lag >= 0.5)
  -> seasonal set: seasonal_naive, holt_winters, auto_ets, sarima, tbats.
- trend + seasonality -> ETS / SARIMA / ML-capable
  (auto_ets, sarima, auto_arima, holt_linear_damped, theta).
- feature-rich (extra regressor columns supplied alongside the series)
  -> ML path note: univariate shortlist + statement that the encoded
  tabular model (prediction/tabular.py) should be tried for the drivers.
- irregular (high CV, no trend, no seasonality)
  -> simple baselines only + "unreliable" explanation; never a complex fit.

The final choice is ALWAYS empirical: every shortlisted candidate is scored
on a trailing holdout (last h points, h = min(3, max(1, n // 4))) with
mae/rmse/mape, and the lowest holdout MAPE (MAE fallback for all-zero
holdouts) wins. Walk-forward MAPE from prediction/evaluate.py is reported
alongside as the honest accuracy.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from . import models as M
from .evaluate import (
    holdout_backtest,
    one_step_ahead_mae,
    one_step_ahead_mape,
    walk_forward_coverage,
)

log = logging.getLogger("aria.prediction.selection")

HOLDOUT_STATEMENT = "based on best holdout"
SELECTION_METRIC_NOTE = (
    "train on earlier periods, validate on later periods (trailing holdout + "
    "expanding walk-forward, never random split); winner by lowest holdout "
    "MAPE with MAE/RMSE as tiebreak and fallback for all-zero holdouts"
)

# Shortlists reference candidate CLASSES from prediction/models.py so the
# caller instantiates + fits the winner on the full series (uniform with
# prediction/evaluate.select_model). Names mirror M.CANDIDATES entries.
_BASELINES = ["naive", "mean", "median", "drift"]
_SEASONAL = ["seasonal_naive", "holt_winters_additive", "auto_ets", "sarima", "tbats"]
_TREND = ["holt_linear_damped", "theta", "linear_trend", "exponential_smoothing"]
_TREND_SEASON = ["auto_ets", "sarima", "auto_arima", "holt_winters_additive",
                 "holt_linear_damped", "theta"]
_SIMPLE = ["naive", "mean", "median", "moving_average", "drift"]

_BY_NAME = {cls.name: cls for cls in M.CANDIDATES}


def _autocorr(values, lag):
    """Lag-k autocorrelation (-1..1), 0.0 when undefined."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n <= lag or lag < 1:
        return 0.0
    a, b = v[:-lag], v[lag:]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _trend_strength(values):
    """|slope| scaled by std per step: ~0 flat, >~0.3 clearly trended."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 3:
        return 0.0
    slope = float(np.polyfit(np.arange(n, dtype=float), v, 1)[0])
    scale = float(np.std(v)) or abs(float(np.mean(v))) or 1.0
    return abs(slope) / scale


def describe_series(values, season_period=None):
    """Diagnostics driving the rule-based shortlist."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    mean = float(np.mean(v)) if n else 0.0
    std = float(np.std(v)) if n else 0.0
    cv = (std / abs(mean)) if abs(mean) > 1e-12 else (float("inf") if std > 0 else 0.0)
    seas_ac = _autocorr(v, season_period) if season_period and season_period >= 2 else 0.0
    trend = _trend_strength(v)
    zero_frac = float(np.mean(v == 0.0)) if n else 0.0
    two_seasons = bool(season_period and season_period >= 2 and n >= 2 * season_period)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "cv": cv,
        "season_period": season_period,
        "seasonal_ac": round(seas_ac, 3),
        "clear_seasonality": bool(two_seasons and seas_ac >= 0.5),
        "trend_strength": round(trend, 3),
        "has_trend": bool(trend >= 0.15 and n >= 4),
        "zero_frac": round(zero_frac, 3),
        "very_little_history": bool(n < 8 or (season_period and n < 2 * season_period)),
        "irregular": bool(n >= 8 and cv > 1.0 and trend < 0.15 and seas_ac < 0.3),
    }


def shortlist_candidates(features, n_extra_features=0):
    """Rule-based candidate names + human-readable reason.

    Returns (names, reason). Names are resolved against available M.CANDIDATES
    by the caller (missing optional deps are skipped, never fail).
    """
    if features["very_little_history"]:
        reason = (
            f"only {features['n']} history points — too little to estimate "
            "trend/seasonality, so naive/seasonal-naive baselines compete"
        )
        names = (["seasonal_naive", "naive", "mean", "drift"]
                 if features.get("season_period") else ["naive", "mean", "drift"])
        return names, reason
    if n_extra_features > 0:
        # Feature-rich: ML (encoded tabular) owns the drivers; the univariate
        # shortlist stays trend/seasonal but the statement says so.
        if features["clear_seasonality"] and features["has_trend"]:
            names = list(_TREND_SEASON)
            reason = (f"{n_extra_features} extra feature(s) available — ML path "
                      "recommended for drivers; trend+seasonal univariate models "
                      "(ETS/SARIMA) compete for the pure time-series signal")
        elif features["clear_seasonality"]:
            names = list(_SEASONAL)
            reason = (f"{n_extra_features} extra feature(s) available — ML path "
                      "recommended for drivers; seasonal models compete for the "
                      "time-series signal")
        else:
            names = list(_TREND) + ["auto_arima"]
            reason = (f"{n_extra_features} extra feature(s) available — ML path "
                      "recommended for drivers; trend models compete for the "
                      "time-series signal")
        return names, reason
    if features["clear_seasonality"] and features["has_trend"]:
        return (list(_TREND_SEASON),
                f"trend + seasonality detected (seasonal autocorr "
                f"{features['seasonal_ac']}) — ETS/SARIMA/theta compete")
    if features["clear_seasonality"]:
        return (list(_SEASONAL),
                f"clear seasonality detected (seasonal autocorr "
                f"{features['seasonal_ac']}) — seasonal models compete")
    if features["has_trend"]:
        return (list(_TREND) + ["auto_arima"],
                f"trend detected (strength {features['trend_strength']}) "
                "without clear seasonality — trend models compete")
    if features["irregular"]:
        return (list(_SIMPLE),
                f"irregular series (CV {features['cv']:.2f}, no trend/season) — "
                "simple baselines only; forecast is unreliable, use directionally")
    return (list(_BASELINES) + ["moving_average", "exponential_smoothing",
                               "holt_linear_damped", "theta", "auto_arima",
                               "auto_ets", "seasonal_naive", "holt_winters_additive"],
            "no dominant structure — broad baseline-to-seasonal shootout")


def _resolve(names, n, season_period):
    """Names -> eligible candidate classes (skips missing deps / min_points)."""
    out = []
    for name in names:
        cls = _BY_NAME.get(name)
        if cls is None:
            continue
        available = getattr(cls, "available", None)
        if callable(available) and not available():
            continue
        if n < getattr(cls, "min_points", 0):
            continue
        if getattr(cls, "needs_real_period", False) and not season_period:
            continue
        if cls not in out:
            out.append(cls)
    if not out:
        out = [M.NaiveModel]
    return out


def select_intelligent(values, season_period=None, n_extra_features=0,
                       min_points=5):
    """Full selection: diagnose -> shortlist -> holdout backtest -> winner.

    Returns (model_or_class, metrics{mae,rmse,mape}, meta). `model_or_class`
    is a candidate CLASS to fit on the full series (None when n < min_points).
    meta carries shortlist/reason/ranking + the HOLDOUT_STATEMENT.
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < min_points:
        return None, {"mae": None, "rmse": None, "mape": None}, {
            "reason": "insufficient history for validation",
            "statement": HOLDOUT_STATEMENT,
        }
    features = describe_series(v, season_period)
    names, reason = shortlist_candidates(features, n_extra_features)
    candidates = _resolve(names, n, season_period)

    from .evaluate import backtest_competing  # local: keep import graph flat
    ranking = backtest_competing(candidates, v, season_period)
    winner = ranking[0] if ranking else None
    if winner is None:
        return M.NaiveModel, {"mae": None, "rmse": None, "mape": None}, {
            "features": features, "shortlist": names, "reason": reason,
            "statement": HOLDOUT_STATEMENT,
        }
    best_name = winner["name"]
    best_cls = next(c for c in candidates if c.name == best_name)

    # Honest walk-forward accuracy for the winner (out-of-sample, all origins).
    oos_mape = one_step_ahead_mape(best_cls, v, season_period)
    oos_mae = one_step_ahead_mae(best_cls, v, season_period)
    metrics = {
        "mae": round(winner["mae"], 2) if winner["mae"] is not None else (
            round(oos_mae, 2) if oos_mae is not None else None),
        "rmse": round(winner["rmse"], 2) if winner["rmse"] is not None else None,
        "mape": round(winner["mape"], 2) if winner["mape"] is not None else (
            round(oos_mape, 2) if oos_mape is not None else None),
    }
    coverage = walk_forward_coverage(v, season_period)
    meta = {
        "features": features,
        "shortlist": [c.name for c in candidates],
        "shortlist_reason": reason,
        "winner": best_name,
        "holdout_points": winner["holdout_points"],
        "selection_metric": SELECTION_METRIC_NOTE,
        "walk_forward_mape": round(oos_mape, 2) if oos_mape is not None else None,
        "walk_forward_mae": round(oos_mae, 2) if oos_mae is not None else None,
        "walk_forward_origins": coverage["origins"],
        "walk_forward_evaluated": coverage["evaluated"],
        "zero_actuals_skipped": coverage["zero_skipped"],
        "ranking": [
            {"model": r["name"],
             "mae": round(r["mae"], 2) if r["mae"] is not None else None,
             "rmse": round(r["rmse"], 2) if r["rmse"] is not None else None,
             "mape": round(r["mape"], 2) if r["mape"] is not None else None,
             "holdout_points": r["holdout_points"]}
            for r in ranking
        ],
        "statement": (
            f"Selected {best_name} {HOLDOUT_STATEMENT} "
            f"(holdout MAPE {metrics['mape']}%, "
            f"{winner['holdout_points']} points; {reason})."
        ),
    }
    if n_extra_features > 0:
        if n >= 20:
            meta["ml_note"] = (
                f"{n_extra_features} extra feature column(s) present — for driver "
                "analysis use the encoded tabular ML path (prediction/tabular.py); "
                "the time-series winner above covers the pure temporal signal."
            )
            meta["statement"] += " ML path recommended for drivers."
        else:
            # ML fallback only with enough data: too few rows to train anything
            # beyond the univariate time-series shortlist.
            meta["ml_note"] = (
                f"{n_extra_features} extra feature column(s) present but only {n} "
                "history points — not enough data for the ML path, so the "
                "univariate time-series winner above stands alone."
            )
    if features["irregular"]:
        meta["reliability_note"] = (
            "irregular series without stable trend or seasonality — "
            "treat the forecast as unreliable/directional."
        )
        meta["statement"] += " Irregular history: treat as directional."
    return best_cls, metrics, meta


def accuracy_from_mape(mape_value):
    if mape_value is None or not math.isfinite(mape_value):
        return None
    return round(max(0.0, 100.0 - mape_value), 1)
