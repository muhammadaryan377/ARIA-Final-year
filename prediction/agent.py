"""PredictionAgent - ARIA predictive analysis.

Turns a Goal-Agent `processed_data.json` payload into a structured forecast.

Forecastability contract (dynamic, never hardcoded per dataset):

- ``50`` is a *confidence threshold*, not a hard block. With ``n_obs >= 50``
  the full pipeline runs (model selection + train-earlier/validate-later
  time-series validation). With fewer observations but a mathematically
  meaningful series (valid date column, temporal variation, numeric target,
  non-constant) ARIA still produces a PRELIMINARY forecast marked
  ``reliability: limited`` with an explicit message, ``n_obs``, confidence
  interval and a recommendation to connect more history.
- Genuinely unforecastable data (no date field, no temporal variation, no
  numeric target, constant series) returns ``ok: False`` with a reason, the
  missing field and a suggestion — never a fabricated forecast.
- Pre-forecast validation + preparation (time column, target, frequency from
  the median timestamp delta, missing dates/values, duplicate timestamps
  aggregated by sum, constant/near-constant check, outlier flags for info
  only, irregular intervals, aggregation level) is exposed as the
  ``preparation`` object — every value derived from the data.
- Model selection compares naive / seasonal-naive / moving-average /
  exponential smoothing / Holt-Winters / ARIMA-SARIMA families (statsmodels
  when installed, skipped gracefully otherwise; ML only with enough data)
  via trailing-holdout + walk-forward scoring and picks the best by
  MAPE with MAE/RMSE tiebreak. ``model_selected`` + ``candidates[]`` carry
  the scores.
- Metrics (MAE, RMSE, MAPE, sMAPE, WAPE, bias) when history allows;
  ``reliability: high|moderate|limited`` + concise ``accuracy`` summary up
  front, full detail behind the expandable ``details`` object.
- Horizon is auto-suggested from frequency and ``n_obs``
  (``min(cap, max(floor, n_obs // 4))`` style logic), user-overridable via
  the ``horizon`` request parameter.
- Drivers are derived from the dataset (trend slope, seasonality strength,
  top categorical contributor, volatility) — each with ``name``,
  ``evidence`` and ``metric_value``.

Legacy response keys are preserved; new keys are additive only. Every error
path returns structured graceful JSON (``core/errors.py::sanitize_error``),
never a stack trace.
"""

import logging
import math

import numpy as np
import pandas as pd

from .evaluate import (
    accuracy_from_mape,
    bias as _bias,
    mae as _mae,
    mape as _mape,
    reliability_label as _legacy_reliability_label,
    rmse as _rmse,
    smape as _smape,
    wape as _wape,
)
from . import models as M
from . import selection as _selection

try:  # graceful when used outside the full ARIA stack
    from core.errors import sanitize_error as _sanitize_error
except Exception:  # pragma: no cover
    def _sanitize_error(exc, max_len=300):
        try:
            return str(exc)[:max_len]
        except Exception:
            return "An error occurred."

log = logging.getLogger("aria.prediction")

# 50 is a confidence threshold, NOT a hard block: below it forecasts are
# still produced when mathematically meaningful, but marked preliminary.
CONFIDENCE_THRESHOLD_N = 50
PRELIMINARY_MESSAGE = (
    "ARIA can generate a preliminary forecast from the available data, "
    "but fewer than 50 observations are available. "
    "Forecast confidence is limited."
)
PRELIMINARY_RECOMMENDATION = (
    "Connect more historical data to improve forecast reliability."
)
ANALYZE_SUGGESTION = "Ask ARIA to analyze this dataset instead."


def _numeric_cols(df):
    """Numeric columns, including locale-formatted numeric strings.

    Uses locale-aware parsing ("1,234.56" vs "1.234,56", "%", currency)
    so formatted measures are recognised; the parse ratio (>=0.8) gates
    inclusion. Pure numeric dtypes always qualify.
    """
    cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
            continue
        sample = df[c].dropna()
        if not len(sample):
            continue
        try:
            _, ratio = _coerce_locale_series(sample)
        except Exception:
            continue
        if ratio >= 0.8:
            cols.append(c)
    return cols


def _datetime_cols(df):
    cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().sum() >= max(2, int(len(df) * 0.8)):
                cols.append(c)
        except Exception:
            continue
    return cols


def _infer_period(dates):
    """Choose the aggregation grain from the median timestamp delta.

    Supports hourly (h) / daily (D) / weekly (W) / monthly (MS) /
    quarterly (Q) / yearly (YS). The median delta drives the grain and a
    coverage sanity check coarsens a grain that is too fine for the
    observed span (fixes yearly/quarterly/hourly series mislabeled
    monthly). Returns a pandas Grouper-compatible code.
    """
    try:
        uniq = pd.Series(
            pd.unique(pd.to_datetime(dates, errors="coerce"))
        ).sort_values().dropna()
    except Exception:
        return "MS"
    if len(uniq) < 2:
        return "D"
    try:
        gaps_h = (uniq.diff().dropna().dt.total_seconds() / 3600.0).astype(float)
        gaps_h = gaps_h[np.isfinite(gaps_h)]
    except Exception:
        return "MS"
    if len(gaps_h) == 0:
        return "D"
    try:
        med_h = float(gaps_h.median())
    except Exception:
        return "MS"
    if not math.isfinite(med_h) or med_h < 0:
        return "MS"
    med_d = med_h / 24.0
    if med_h <= 3.0:
        grain = "h"
    elif med_d <= 2.0:
        grain = "D"
    elif med_d <= 10.0:
        grain = "W"
    elif med_d <= 45.0:
        grain = "MS"
    elif med_d <= 120.0:
        grain = "Q"
    elif med_d <= 270.0:
        # ~half-yearly spacings still aggregate cleanly at quarterly grain.
        grain = "Q"
    else:
        grain = "YS"
    # Sanity: a grain implying far more periods than observed points is
    # too fine (e.g. two close transactions in a yearly series). Coarsen
    # until the observed coverage is plausible.
    try:
        _NOMINAL_D = {"h": 1.0 / 24.0, "D": 1.0, "W": 7.0, "MS": 30.44,
                      "Q": 91.31, "YS": 365.25}
        _ORDER = ["h", "D", "W", "MS", "Q", "YS"]
        span_d = float((uniq.iloc[-1] - uniq.iloc[0]).total_seconds() / 86400.0)
        n_uniq = int(len(uniq))
        if span_d > 0 and n_uniq >= 2:
            idx = _ORDER.index(grain)
            for cand in _ORDER[idx:]:
                nominal = _NOMINAL_D[cand]
                implied = span_d / nominal + 1.0
                coverage = (n_uniq / implied) if implied > 0 else 1.0
                if coverage >= 0.25 or cand == "YS":
                    grain = cand
                    break
    except Exception:
        pass
    return grain


_PERIOD_NAMES = {"D": "daily", "W": "weekly", "MS": "monthly",
                 "Q": "quarterly", "YS": "yearly", "h": "hourly"}
_PERIOD_GROUPER = {"D": "D", "W": "W", "MS": "MS", "Q": "QS",
                   "YS": "YS", "h": "h"}
# Pandas Period aliases for gap detection (None where Period is unusable).
_PERIOD_FREQ = {"MS": "M", "D": "D", "W": "W", "Q": "Q", "YS": "Y", "h": "h"}
_NOMINAL_DAYS = {"h": 1.0 / 24.0, "D": 1.0, "W": 7.0, "MS": 30.44,
                 "Q": 91.31, "YS": 365.25}


def _parse_locale_value(raw):
    """Parse one locale-formatted number (thousands/decimal/percent/currency).

    Handles "1,234.56" vs "1.234,56", thin spaces/apostrophes as
    thousands separators, and trailing "%". Returns float or None.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float, np.integer, np.floating)):
        try:
            f = float(raw)
            return f if math.isfinite(f) else None
        except Exception:
            return None
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "null", "-", "--", "n/a", "na", "#n/a"):
        return None
    is_pct = s.endswith("%")
    if is_pct:
        s = s[:-1].strip()
    # Strip currency/scale symbols but keep digits, signs, separators.
    s = s.replace("$", "").replace("€", "").replace("£", "").replace("¥", "").strip()
    s = s.replace("\u00a0", " ").replace("\u202f", " ")
    s = s.replace(" ", "").replace("'", "").replace("’", "")
    if not s:
        return None
    try:
        if "." in s and "," in s:
            # Rightmost separator is the decimal mark.
            if s.rfind(".") > s.rfind(","):
                s = s.replace(",", "")
            else:
                s = s.replace(".", "").replace(",", ".")
        elif "," in s:
            # Only commas: thousands ("1,234") vs decimal ("12,5").
            import re as _re
            if _re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
                s = s.replace(",", "")
            elif _re.fullmatch(r"-?\d+,\d{1,2}", s):
                s = s.replace(",", ".")
            elif s.count(",") == 1:
                head, tail = s.split(",")
                if len(tail) == 3 and head and head.lstrip("-+").isdigit():
                    s = head + tail  # thousands
                else:
                    s = head + "." + tail
            else:
                s = s.replace(",", "")
        f = float(s)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _coerce_locale_series(values):
    """Coerce a column with locale-aware parsing. Returns (float Series, ratio)."""
    try:
        s = pd.Series(list(values))
    except Exception:
        return pd.Series(dtype=float), 0.0
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype(float), 1.0
    parsed = [_parse_locale_value(v) if pd.notna(v) else None for v in s.tolist()]
    out = pd.Series(parsed, dtype=float)
    try:
        denom = int(s.notna().sum()) or 0
        ratio = float(out.notna().sum() / denom) if denom else 0.0
    except Exception:
        ratio = 0.0
    return out, round(ratio, 4)


_GAUGE_LAST_HINTS = ("balance", "stock", "inventory", "level", "position",
                     "headcount", "occupancy")
_GAUGE_AVG_HINTS = ("price", "rate", "ratio", "pct", "percent", "temp",
                    "score", "rating", "index", "avg", "average", "close",
                    "closing", "open", "unit_cost", "unitcost")


def _infer_aggregation(numeric_col, df=None):
    """Additive-flow vs gauge inference for time-grain aggregation.

    Flow measures (sales, amounts, counts, quantities) SUM across
    timestamps; gauge measures (prices, rates, balances, levels) must
    never SUM — they average (AVG) or take the last observation (LAST).
    Returns (pandas_agg, method_label, reason).
    """
    low = str(numeric_col or "").lower().strip()
    for hint in _GAUGE_LAST_HINTS:
        if hint in low:
            return ("last",
                    "last",
                    f"column '{numeric_col}' looks like a gauge/level "
                    f"(hint '{hint}') — aggregating duplicate timestamps by "
                    f"LAST observation, not SUM")
    for hint in _GAUGE_AVG_HINTS:
        if hint in low:
            return ("mean",
                    "avg",
                    f"column '{numeric_col}' looks like a gauge/rate "
                    f"(hint '{hint}') — aggregating duplicate timestamps by "
                    f"AVERAGE, not SUM")
    return ("sum",
            "sum",
            f"column '{numeric_col}' looks additive — aggregating duplicate "
            f"timestamps by SUM")


def _headline_pct_change(points, values, season_period=None):
    """Headline change vs a trailing baseline average (not last-actual).

    Baseline = mean of the last min(n, max(3, season)) periods. Returns
    (pct_or_None, baseline_or_None, suppressed_bool). Suppressed when the
    baseline is near-zero (a pct would explode) — callers must render no
    headline pct in that case.
    """
    try:
        pts = np.asarray(points, dtype=float)
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        pts = pts[np.isfinite(pts)]
        if len(v) == 0 or len(pts) == 0:
            return None, None, True
        try:
            window = max(3, int(season_period)) if season_period else 4
        except Exception:
            window = 4
        window = max(1, min(int(len(v)), int(window)))
        baseline = float(np.mean(v[-window:]))
        try:
            scale = max(1.0, abs(float(np.mean(np.abs(v)))))
        except Exception:
            scale = 1.0
        if not math.isfinite(baseline) or abs(baseline) < 1e-6 * scale:
            return None, (round(baseline, 2) if math.isfinite(baseline) else None), True
        pct = round(float(pts[-1] - baseline) / abs(baseline) * 100.0, 2)
        return pct, round(baseline, 2), False
    except Exception:
        return None, None, True

# Column names that indicate a natural period/ordinal axis for forecasting even
# without a datetime column (e.g. "sales by year", "churn per month").
_PERIOD_KEYWORDS = {
    "period", "periods", "year", "years", "yr", "month", "months",
    "day", "days", "week", "weeks", "quarter", "quarters", "qtr",
    "seq", "sequence", "index", "idx", "order", "rank", "time",
    "date", "dates", "timestamp", "datetime",
}


def _suggest_horizon(frequency_label, n_obs):
    """Dynamic horizon default from frequency + history length.

    ``min(cap, max(floor, n_obs // 4))`` logic — caps/floors are method
    bounds, the actual value is computed per dataset. A caller-supplied
    ``horizon`` always overrides this.
    """
    n = max(int(n_obs or 0), 0)
    if frequency_label == "daily":
        return min(30, max(7, n // 4))
    if frequency_label == "weekly":
        return min(12, max(4, n // 4))
    if frequency_label == "monthly":
        return min(12, max(3, n // 4))
    if frequency_label == "quarterly":
        return min(8, max(2, n // 4))
    if frequency_label == "yearly":
        return min(5, max(2, n // 4))
    if frequency_label == "hourly":
        return min(48, max(12, n // 4))
    return min(12, max(3, n // 4))


def _reliability_level(accuracy, preliminary=False):
    """high|moderate|limited from backtested accuracy.

    Preliminary (sub-threshold history) forecasts are always ``limited`` —
    the ``forecast_stage`` + message say they are preliminary.
    """
    if preliminary:
        return "limited"
    if accuracy is None:
        return "limited"
    if accuracy >= 85:
        return "high"
    if accuracy >= 70:
        return "moderate"
    return "limited"


def _lag_autocorr(values, lag):
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n <= lag or lag < 1:
        return 0.0
    a, b = v[:-lag], v[lag:]
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    try:
        return float(np.corrcoef(a, b)[0, 1])
    except Exception:
        return 0.0


def _forecast_labels(last_label, steps, period_code):
    if period_code is None:
        return [f"t+{k}" for k in range(1, steps + 1)]
    last_ts = pd.to_datetime(last_label)
    step = {
        "D": pd.Timedelta(days=1),
        "W": pd.Timedelta(weeks=1),
        "MS": pd.DateOffset(months=1),
        "Q": pd.DateOffset(months=3),
        "YS": pd.DateOffset(years=1),
        "h": pd.Timedelta(hours=1),
    }.get(period_code, pd.DateOffset(months=1))
    out = []
    for k in range(1, max(1, int(steps or 1)) + 1):
        try:
            ts = last_ts + k * step
            out.append(pd.Timestamp(ts).strftime(
                "%Y-%m-%d %H:%M" if period_code == "h" else "%Y-%m-%d"))
        except Exception:
            out.append(f"t+{k}")
    return out


def _series_varies(vals) -> bool:
    """True when consecutive points actually move (not a flat level line).

    Relative bar: range must exceed 1e-6 of the series scale, so genuine
    micro-variation counts but float dust does not.
    """
    try:
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        if len(v) < 2:
            return False
        scale = max(1.0, abs(float(np.mean(np.abs(v)))))
        return bool(float(np.max(v) - np.min(v)) > 1e-6 * scale)
    except Exception:
        return False


def _forecast_varies(vals) -> bool:
    """Alias kept at the forecast call-site: same varying-series rule."""
    return _series_varies(vals)


def _history_varies(vals) -> bool:
    return _series_varies(vals)


def _substitute_flat_winner(values, season_period, model_spec, sel_meta,
                            sel_metrics, steps):
    """Swap a flat-projecting winner for a varying scored alternate.

    When the winner's own refit projects a constant level on genuinely
    varying history, prefer the best-ranked validated candidate whose refit
    actually varies (still a backtested model; the swap is disclosed in the
    returned selection statement) — never invented wiggle. No varying
    scored alternate -> inputs returned unchanged with note None.

    Returns (model_spec, sel_meta, sel_metrics, sub_note).
    """
    from .evaluate import _make as _mk
    try:
        _probe = max(1, min(int(steps or 1), len(values)))
        if season_period:
            _probe = max(_probe, min(int(season_period), len(values)))
        _winner = getattr(model_spec, "name", "?") \
            if isinstance(model_spec, type) else "ensemble"
        try:
            _wpts = np.asarray(
                _mk(model_spec, season_period).fit(values).predict(
                    _probe), dtype=float) \
                if isinstance(model_spec, type) else \
                np.asarray(model_spec.predict(_probe), dtype=float)
        except Exception:
            log.debug("flat-line guard: winner probe refit failed",
                      exc_info=True)
            return model_spec, sel_meta, sel_metrics, None
        if _wpts is None or _forecast_varies(_wpts) \
                or not _history_varies(values):
            return model_spec, sel_meta, sel_metrics, None
        for _row in (sel_meta.get("ranking") or [])[1:]:
            _alt_name = _row.get("model")
            if not _alt_name or _alt_name == _winner:
                continue
            if (_row.get("mae") is None and _row.get("mape") is None):
                continue  # unscored candidates cannot substitute
            _alt = next(
                (c for c in M.CANDIDATES if c.name == _alt_name),
                None)
            if _alt is None or _alt is model_spec:
                continue
            if len(values) < getattr(_alt, "min_points", 0):
                continue
            if (getattr(_alt, "needs_real_period", False)
                    and not season_period):
                continue
            try:
                _alt_pts = np.asarray(
                    _mk(_alt, season_period).fit(values).predict(
                        _probe), dtype=float)
            except Exception:
                continue
            if _forecast_varies(_alt_pts):
                _new_metrics = {
                    "mae": _row.get("mae"),
                    "rmse": _row.get("rmse"),
                    "mape": _row.get("mape"),
                }
                _new_meta = dict(sel_meta)
                _new_meta["winner"] = _alt_name
                _note = (
                    f"{_winner} validated best but projected a "
                    f"flat level line on varying history; using "
                    f"{_alt_name} (holdout MAPE {_row.get('mape')}%) "
                    f"so the projection carries the observed dynamics."
                )
                _new_meta["statement"] = (
                    str(_new_meta.get("statement") or "")
                    + " " + _note).strip()
                return _alt, _new_meta, _new_metrics, _note
        log.debug("flat-line guard: winner %s flat on varying history but "
                  "no varying scored alternate found", _winner)
        return model_spec, sel_meta, sel_metrics, None
    except Exception:
        log.debug("flat-line guard failed", exc_info=True)
        return model_spec, sel_meta, sel_metrics, None


def _wave_period(values, season_period, period_code):
    """Generic period for the trend+seasonal safety net (no hardcoded tables).

    Prefers the validated ``season_period``; otherwise falls back to the
    frequency grain (daily->7, weekly->4, monthly->12 need only ONE full
    season) and finally to the strongest autocorrelation lag in
    2..min(12, n//2). Returns None only when history is too short to
    estimate any periodicity.
    """
    try:
        if season_period and int(season_period) >= 2:
            return int(season_period)
    except Exception:
        pass
    try:
        n = int(len(values))
    except Exception:
        return None
    if period_code == "MS" and n >= 12:
        return 12
    if period_code == "D" and n >= 7:
        return 7
    if period_code == "W" and n >= 4:
        return 4
    if period_code == "Q" and n >= 8:
        return 4
    if period_code == "h" and n >= 48:
        return 24
    if period_code == "YS":
        return None
    if n >= 5:
        try:
            hi = min(12, n // 2)
            best_lag, best_ac = 0, 0.0
            for lag in range(2, hi + 1):
                ac = _lag_autocorr(values, lag)
                if abs(ac) > abs(best_ac):
                    best_lag, best_ac = lag, ac
            if best_lag:
                return int(best_lag)
        except Exception:
            pass
    return None


def _projection_slope(values, wave_period):
    """Trend slope of the generic trend+seasonal projection.

    OLS over the whole window, falling back to first->last drift when the
    history is shorter than two full seasons (OLS is phase-biased with a
    single cycle). Shared by the projection and its in-sample fitted curve so
    the two can never describe different trends.
    """
    v = np.asarray(values, dtype=float)
    n = int(len(v))
    try:
        t = np.arange(n, dtype=float)
        ols = float(np.polyfit(t, v, 1)[0]) if n >= 2 else 0.0
    except Exception:
        ols = 0.0
    try:
        drift = float((v[-1] - v[0]) / (n - 1)) if n > 1 else 0.0
    except Exception:
        drift = 0.0
    if wave_period and n < 2 * int(wave_period):
        slope = drift
    else:
        slope = ols
    if not math.isfinite(slope):
        slope = 0.0
    return slope


def _trend_seasonal_projection(values, steps, wave_period):
    """Generic trend + seasonal/periodic projection from actual history.

    - Trend slope: drift (first->last) when history is shorter than two full
      seasons (OLS is phase-biased with a single cycle), else least-squares.
      The level is always anchored to the last actual so the projection is
      continuous with history.
    - Seasonality: mean detrended residual per phase (zero-mean ripple), so
      future steps carry the observed periodic shape. Linear history yields
      ~zero ripple (straight trend preserved); wavy history yields waves.
    Returns (forecast ndarray, ripple_amplitude, slope).
    """
    v = np.asarray(values, dtype=float)
    n = int(len(v))
    steps = max(1, int(steps or 1))
    try:
        t = np.arange(n, dtype=float)
    except Exception:
        t = None
    slope = _projection_slope(v, wave_period)
    p = max(2, int(wave_period)) if wave_period else 0
    if not p:
        return v[-1] + slope * np.arange(1, steps + 1, dtype=float), 0.0, slope
    try:
        if t is None:
            raise ValueError("time index unavailable")
        resid = v - slope * t
        s = np.zeros(p)
        for ph in range(p):
            idx = [k for k in range(n) if k % p == ph]
            s[ph] = float(np.mean([resid[k] for k in idx])) if idx else 0.0
        ripple = s - float(np.mean(s))
        amp = float(np.max(ripple) - np.min(ripple))
        last_ph = (n - 1) % p
        out = np.empty(steps)
        for i in range(steps):
            fut = (n + i) % p
            out[i] = float(v[-1] + slope * (i + 1) + (ripple[fut] - ripple[last_ph]))
        return out, amp, slope
    except Exception:
        return v[-1] + slope * np.arange(1, steps + 1, dtype=float), 0.0, slope


def _ensure_wavy_forecast(values, points, season_period, period_code):
    """Guarantee a non-flat, wave-carrying projection on varying history.

    Replaces a flat level (or a straight line that suppresses clear observed
    seasonality) with the generic trend+seasonal projection above — derived
    only from the actual history, never invented. Returns (points, note).
    """
    try:
        v = np.asarray(values, dtype=float)
        pts = np.asarray(points, dtype=float)
        if len(pts) < 2 or not _history_varies(v):
            return points, None
        wp = _wave_period(v, season_period, period_code)
        if not wp:
            return points, None
        proj, amp, slope = _trend_seasonal_projection(v, len(pts), wp)
        if not _forecast_varies(proj):
            return points, None
        try:
            scale = max(1.0, abs(float(np.mean(np.abs(v[np.isfinite(v)])))))
        except Exception:
            scale = 1.0
        std_vals = float(np.std(v)) if len(v) > 1 else 0.0
        std_pts = float(np.std(pts)) if len(pts) > 1 else 0.0
        flat = not _forecast_varies(pts)
        thresh = max(1e-6 * scale, 0.15 * std_vals)
        seasonal = bool(amp > thresh)
        if flat or (seasonal and std_pts < 0.7 * std_vals):
            note = (
                f"level/straight projection carried no seasonal shape on varying "
                f"history; using generic trend+seasonal projection (period {wp}, "
                f"slope {slope:+.4f}/period, seasonal amplitude {amp:.2f}) from "
                f"observed history."
            )
            return proj, note
        return points, None
    except Exception:
        log.debug("wave guard failed", exc_info=True)
        return points, None


def _trend_seasonal_fitted(values, wave_period):
    """In-sample curve of the trend+seasonal projection (same slope rule).

    ``slope*t`` plus the mean detrended residual per phase — the exact
    decomposition the projection continues from, so ``values - fitted`` is
    the within-phase deviation around the trend the shipped points carry.
    """
    v = np.asarray(values, dtype=float)
    n = int(len(v))
    if not n:
        return np.array([], dtype=float)
    t = np.arange(n, dtype=float)
    slope = _projection_slope(v, wave_period)
    base = slope * t
    p = max(2, int(wave_period)) if wave_period else 0
    if not p:
        return base + float(np.mean(v - base))
    try:
        resid = v - base
        s = np.zeros(p)
        for ph in range(p):
            idx = [k for k in range(n) if k % p == ph]
            s[ph] = float(np.mean([resid[k] for k in idx])) if idx else 0.0
        return base + s[np.arange(n) % p]
    except Exception:
        return base + float(np.mean(v - base))


def _wave_projection_cls(season_period, period_code):
    """Model class for the projection the wave guard substitutes in.

    Honors the ``BaseModel`` contract the walk-forward helpers expect
    (``_make(cls, period).fit(train).predict(1)``) and is driven by the SAME
    ``_wave_period``/``_trend_seasonal_projection`` functions that produced
    the shipped points, so scoring it scores the forecast that actually
    ships — never the winner whose points were discarded.
    """

    class _WaveProjection(M.BaseModel):
        name = "trend_seasonal_projection"
        min_points = 5
        needs_period = False
        needs_real_period = False

        @classmethod
        def available(cls):
            return True

        def __init__(self, period=None):
            self.period = period or season_period

        def fit(self, values):
            v = np.asarray(values, dtype=float)
            self.values = v
            self.wave_period = _wave_period(v, self.period, period_code)
            return self

        def predict(self, steps):
            proj, _, _ = _trend_seasonal_projection(
                self.values, steps, self.wave_period)
            return proj

        def fitted(self):
            return _trend_seasonal_fitted(self.values, self.wave_period)

    return _WaveProjection


def _wave_metrics(values, season_period, period_code):
    """Walk-forward MAPE/MAE + in-sample residuals for the shipped projection.

    Rebuild path used when the wave guard replaces ``points``: the same
    ``one_step_ahead_*`` helpers the flat-line branch re-runs for a
    substituted winner, applied to the projection that replaces it.
    Returns ``(oos_mape, oos_mae, resid)`` (either out-of-sample value may
    be ``None`` when walk-forward cannot run — accuracy then honestly reads
    not-applicable instead of stale).
    """
    from .evaluate import one_step_ahead_mape as _oos_mape
    from .evaluate import one_step_ahead_mae as _oos_mae
    spec = _wave_projection_cls(season_period, period_code)
    wave_mape = _oos_mape(spec, values, season_period)
    wave_mae = _oos_mae(spec, values, season_period)
    resid = np.asarray(spec().fit(values).residuals(values), dtype=float)
    return wave_mape, wave_mae, resid


def _widening_sigmas(sigma, steps, values):
    """Per-horizon band scales that strictly widen: sigma*sqrt(h).

    Floored at 0.5% of the series scale so a perfect in-sample fit still
    yields an honest widening band instead of a zero-width line.
    """
    try:
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        scale = max(1.0, abs(float(np.mean(np.abs(v))))) if len(v) else 1.0
    except Exception:
        scale = 1.0
    try:
        base = float(sigma)
    except Exception:
        base = 0.0
    if not math.isfinite(base) or base < 0:
        base = 0.0
    eff = max(base, 0.005 * scale)
    return [eff * math.sqrt(i + 1) for i in range(max(1, int(steps or 1)))]


class PredictionAgent:
    def __init__(self, horizon=6, min_points=5, valid_threshold=70.0,
                 confidence_threshold=50, auto_horizon=True):
        self.horizon = int(horizon)
        self.min_points = int(min_points)
        # Minimum out-of-sample accuracy (%) for a forecast to be presented as
        # valid. Below this the prediction is kept in the payload for transparency
        # but flagged invalid so the story, prescriptions and UI do not use it.
        self.valid_threshold = float(valid_threshold)
        # Confidence threshold (NOT a hard block): below it forecasts are
        # marked preliminary/limited but still produced when meaningful.
        self.confidence_threshold = int(confidence_threshold)
        # When True (default) a forecast() call without an explicit horizon
        # uses the dynamically suggested horizon; otherwise self.horizon.
        self.auto_horizon = bool(auto_horizon)

    def forecast(self, processed_data, numeric_col=None, datetime_col=None, horizon=None):
        try:
            return self._forecast_inner(processed_data, numeric_col, datetime_col, horizon)
        except Exception as exc:
            log.warning("Forecast failed gracefully: %s", exc)
            out = self._empty(_sanitize_error(exc), "insufficient_data")
            out.update({"ok": False, "missing_field": "internal",
                        "suggestion": ANALYZE_SUGGESTION})
            return out

    def _forecast_inner(self, processed_data, numeric_col=None, datetime_col=None, horizon=None):
        rows = (processed_data or {}).get("data") or []
        if not rows:
            out = self._empty("empty dataset", "no_data")
            out.update({"ok": False, "missing_field": "data",
                        "suggestion": ANALYZE_SUGGESTION})
            return out

        df = pd.DataFrame(rows)
        numeric_cols = _numeric_cols(df)
        if not numeric_cols:
            out = self._empty("no numeric columns found", "no_data")
            out.update({"ok": False, "missing_field": "numeric_target",
                        "suggestion": ANALYZE_SUGGESTION})
            return out
        # Coerce locale-formatted string columns so downstream aggregation
        # and models compute correctly (parse ratios disclosed per series).
        for c in numeric_cols:
            if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
                try:
                    _coerced_c, _ = _coerce_locale_series(df[c])
                    df[c] = _coerced_c.values
                except Exception:
                    pass

        if numeric_col is None:
            goal = str(processed_data.get("user_goal") or "").lower()
            # Identifier-like columns (ids/keys/codes) are never forecast targets:
            # they are synthetic keys, not measures.
            def _is_id_like(name):
                try:
                    from core.identifiers import is_id_like as _canon
                    return bool(_canon(name))
                except Exception:
                    low = str(name or "").lower().strip()
                    return low.endswith("_id") or low.endswith("_key") or low in ("id", "key", "code", "codeid")

            measure_cols = [c for c in numeric_cols if not _is_id_like(c)]
            if not measure_cols:
                # Only identifier-like numeric columns exist (e.g. an aggregate
                # that only selected keys). Forecasting a synthetic key is
                # meaningless, so degrade gracefully instead of guessing.
                out = self._empty(
                    "no measure column to forecast (only id/key/code columns are numeric)",
                    "not_applicable",
                )
                out.update({"ok": False, "missing_field": "numeric_target",
                            "suggestion": ANALYZE_SUGGESTION})
                return out
            numeric_col = next(
                (c for c in measure_cols if c.lower() in goal or goal in c.lower()),
                measure_cols[-1],  # measures usually come after id columns in SELECT order
            )
        if numeric_col not in df.columns:
            _avail = ", ".join([str(c) for c in list(df.columns)[:8]])
            out = self._empty(
                f"numeric column '{numeric_col}' not found "
                f"(available columns: {_avail or 'none'}). "
                f"Specify one of the available numeric measures as the "
                f"forecast target.", "no_data")
            out.update({"ok": False, "missing_field": "numeric_target",
                        "suggestion": ANALYZE_SUGGESTION})
            return out

        if datetime_col is None:
            dcols = _datetime_cols(df)
            datetime_col = dcols[0] if dcols else None

        if datetime_col is not None:
            built = self._build_series(df, numeric_col, datetime_col)
            if built is not None:
                values, labels, period_code, diagnostics = built
                if len(values) >= 2:
                    season_period = self._season_period(period_code, len(values))
                    return self._build(
                        values, labels, numeric_col,
                        _PERIOD_NAMES[period_code], period_code, horizon,
                        season_period, diagnostics,
                        n_extra_features=self._extra_feature_count(
                            df, numeric_col, exclude=[datetime_col]),
                        raw_df=df, time_col=datetime_col,
                    )

        # No usable time column: a forecast is only meaningful when the rows carry
        # a natural period/ordinal axis (year, month, period, ...). Plain category
        # aggregates (e.g. "count by city") have no time dimension to forecast over
        # and are handled gracefully instead of being forced through a fake axis.
        seq_col = self._sequence_col(df)
        if seq_col is None:
            _avail = ", ".join([str(c) for c in list(df.columns)[:8]])
            out = self._empty(
                "no time dimension - a forecast is not applicable to this aggregate "
                f"question (forecast target '{numeric_col}'; available columns: "
                f"{_avail or 'none'}). Provide a date/timestamp column or a "
                f"period axis (year, month, week, quarter, period) to forecast over.",
                "not_applicable",
            )
            out.update({"ok": False, "missing_field": "date",
                        "suggestion": ANALYZE_SUGGESTION,
                        "target": numeric_col, "column": numeric_col})
            return out

        seq = df[seq_col]
        _raw_coerced, _seq_parse_ratio = _coerce_locale_series(df[numeric_col])
        raw_vals = _raw_coerced.astype(float)
        n_missing_vals = int(raw_vals.isna().sum())
        values = raw_vals.values
        mask = pd.notna(values)
        values = values[mask]
        seq = np.asarray(seq)[mask]
        if len(values) < 2:
            out = self._empty(f"not enough numeric values in '{numeric_col}'", "insufficient_data")
            out.update({"ok": False, "missing_field": "history",
                        "suggestion": ANALYZE_SUGGESTION,
                        "target": numeric_col, "column": numeric_col})
            return out
        labels = [str(v) for v in seq.tolist()]
        seq_dups = int(pd.Series(labels).duplicated().sum())
        diagnostics = {
            "time_grain": "sequence",
            "duplicate_timestamps_merged": seq_dups,
            "missing_periods": 0,
            "missing_periods_interpolated": 0,
            "missing_periods_masked": 0,
            "missing_periods_filled": 0,
            "gap_method": "ordinal_sequence_no_gaps",
            "median_delta_days": None,
            "frequency": "sequence",
            "n_obs_raw": int(len(df)),
            "n_obs": int(len(values)),
            "missing_values": n_missing_vals,
            "parse_ratio": round(float(_seq_parse_ratio), 4),
            "aggregation_level": "ordinal_sequence_no_date_aggregation",
            "aggregation_method": "none",
            "irregular_intervals": False,
            "tz_normalized": False,
            "tz_note": "no datetime axis — no timezone normalization",
        }
        return self._build(values, labels, numeric_col, "sequence", None, horizon, None,
                           diagnostics=diagnostics,
                           n_extra_features=self._extra_feature_count(
                               df, numeric_col, exclude=[seq_col]),
                           raw_df=df, time_col=seq_col)

    def _sequence_col(self, df):
        """Find a column that provides a natural period/ordinal axis."""
        for c in df.columns:
            low = str(c).lower().strip()
            if low in _PERIOD_KEYWORDS or low.startswith(("year", "month", "week", "qtr", "quarter", "period")):
                if df[c].notna().sum() >= 2 and df[c].nunique() >= 2:
                    return c
        return None

    def _season_period(self, period_code, n):
        """Season length for seasonal models, when enough history exists."""
        if period_code == "D" and n >= 21:
            return 7
        if period_code == "MS" and n >= 24:
            return 12
        if period_code == "W" and n >= 12:
            return 4
        if period_code == "Q" and n >= 12:
            return 4
        if period_code == "h" and n >= 72:
            return 24
        # Yearly grain has no sub-year season to estimate.
        return None

    def _extra_feature_count(self, df, target_col, exclude=None):
        """Other usable columns alongside the target (feature-richness hint).

        The time axis itself is not a predictive feature and is excluded."""
        skip = {target_col} | set(exclude or [])
        return sum(1 for c in df.columns if c not in skip and df[c].notna().sum() >= 2)

    def _build_series(self, df, numeric_col, datetime_col):
        """Data-first series construction.

        detect datetime column -> target/measure -> aggregate to time grain
        -> clean missing/duplicates -> history check.

        Returns (values, labels, period_code, diagnostics) or None when the
        datetime column cannot serve as a time axis. Labels are backend
        timestamps ("%Y-%m-%d"); the frontend must plot these verbatim and
        never invent dates.
        """
        ts_raw = pd.to_datetime(df[datetime_col], errors="coerce")
        # TZ normalization is disclosed, never silent: aware -> UTC naive.
        tz_original, tz_normalized = None, False
        try:
            _tz = getattr(ts_raw.dt, "tz", None)
            if _tz is not None:
                tz_original = str(_tz)
                ts = ts_raw.dt.tz_convert("UTC").dt.tz_localize(None)
                tz_normalized = True
            else:
                # Mixed offsets without a uniform tz still normalize per-value.
                try:
                    _sample = pd.Series(pd.unique(ts_raw.dropna().astype(str))).head(3).tolist()
                except Exception:
                    _sample = []
                if any(("+" in str(x) or str(x).rstrip().endswith("Z")
                        or "GMT" in str(x)) for x in _sample):
                    tz_normalized = True
                    tz_original = "mixed offsets -> UTC"
                ts = ts_raw
                try:
                    ts = ts.dt.tz_localize(None)
                except Exception:
                    pass
        except Exception:
            ts = ts_raw
            try:
                ts = ts.dt.tz_localize(None)
            except Exception:
                pass
        n_bad_time = int(ts.isna().sum())
        mask = ts.notna()
        if mask.sum() < 3:
            return None
        temp = df[mask].copy()
        temp["_ts"] = ts[mask]
        # Chronological order is enforced here; everything downstream
        # (grain inference, aggregation, labels) sees sorted time.
        temp = temp.sort_values("_ts").reset_index(drop=True)
        # Target/measure: locale-aware coercion; parse_ratio is disclosed.
        _coerced, _parse_ratio = _coerce_locale_series(temp[numeric_col])
        temp["_val"] = _coerced.values
        n_bad_value = int(pd.Series(temp["_val"]).isna().sum())
        temp = temp[pd.notna(temp["_val"])]
        if len(temp) < 2:
            return None
        # Raw gap profile (before aggregation) drives the irregular-interval
        # flag and the frequency evidence — all derived from observed data.
        uniq_ts = pd.Series(pd.unique(temp["_ts"])).sort_values()
        raw_gaps = uniq_ts.diff().dropna().dt.total_seconds() / 86400.0
        median_delta = float(raw_gaps.median()) if len(raw_gaps) else 0.0
        if len(raw_gaps) >= 3:
            gap_mean = float(raw_gaps.mean())
            gap_std = float(raw_gaps.std())
            gap_cv = (gap_std / gap_mean) if gap_mean > 1e-12 else 0.0
            irregular = bool(gap_cv > 0.75 or float(raw_gaps.max()) > 3 * median_delta)
        else:
            irregular = False
        # Time grain from the observed (unique) timestamps.
        period_code = _infer_period(temp["_ts"])
        if period_code not in _PERIOD_GROUPER:
            period_code = "MS"
        group_freq = _PERIOD_GROUPER[period_code]
        per_freq = _PERIOD_FREQ.get(period_code)
        # Additive-vs-gauge aggregation inference (never SUM a gauge).
        _agg_func, _agg_method, _agg_reason = _infer_aggregation(
            numeric_col, temp)
        n_rows = len(temp)
        n_unique = int(temp["_ts"].nunique())
        try:
            grouped = temp.groupby(
                pd.Grouper(key="_ts", freq=group_freq))["_val"].agg(_agg_func)
        except Exception:
            grouped = temp.groupby(
                pd.Grouper(key="_ts", freq=group_freq))["_val"].sum()
            _agg_method = "sum"
        duplicates_merged = max(0, n_rows - n_unique)
        # Gap accounting from observed grain periods vs the full range.
        try:
            if per_freq:
                observed = set(temp["_ts"].dt.to_period(per_freq))
                full = pd.period_range(temp["_ts"].min(), temp["_ts"].max(),
                                       freq=per_freq)
                missing = [p for p in full if p not in observed]
            else:
                missing = []
        except Exception:
            observed, full, missing = set(), [], []
        missing_periods = len(missing)
        # Never 0-fill for fitting: reindex keeps NaN for missing periods.
        # Short interior runs (<=2) are linearly interpolated; longer runs
        # are linearly bridged for continuity but MASKED (counted
        # separately, disclosed, and excluded from honest error metrics
        # where possible) — never filled with 0.
        try:
            full_idx = pd.date_range(grouped.index.min(), grouped.index.max(),
                                     freq=group_freq)
            grouped = grouped.reindex(full_idx)
        except Exception:
            pass
        grouped = grouped.astype(float)
        n_interpolated, n_masked = 0, 0
        if missing_periods:
            try:
                asfreq_idx = pd.period_range(grouped.index.min(),
                                             grouped.index.max(), freq=per_freq)
                vals = grouped.copy()
                n = min(len(asfreq_idx), len(vals))
                vals.index = asfreq_idx[:n]
                run, prev, runs = [], None, []
                for p in missing:
                    if prev is not None and p.ordinal == prev.ordinal + 1:
                        run.append(p)
                    else:
                        if run:
                            runs.append(run)
                        run = [p]
                    prev = p
                if run:
                    runs.append(run)
                for run in runs:
                    try:
                        lo_pos = max(0, full.get_loc(run[0]) - 1)
                        hi_pos = min(len(full) - 1, full.get_loc(run[-1]) + 1)
                    except Exception:
                        continue
                    if full[lo_pos] in observed and full[hi_pos] in observed:
                        try:
                            lo_v = float(vals.loc[full[lo_pos]])
                            hi_v = float(vals.loc[full[hi_pos]])
                        except Exception:
                            continue
                        span = hi_pos - lo_pos
                        for k, p in enumerate(run, start=1):
                            try:
                                vals.loc[p] = lo_v + (hi_v - lo_v) * (k / span)
                            except Exception:
                                continue
                            if len(run) <= 2:
                                n_interpolated += 1
                            else:
                                n_masked += 1
                    else:
                        n_masked += len(run)
                vals.index = grouped.index
                grouped = vals.astype(float)
            except Exception:
                log.debug("gap interpolation failed", exc_info=True)
        # Any residual NaN (e.g. edge effects) is bridged linearly for model
        # continuity and counted as masked — still never 0.
        try:
            if grouped.isna().any():
                n_masked += int(grouped.isna().sum())
                grouped = grouped.interpolate(method="linear", limit_direction="both")
                if grouped.isna().any():
                    grouped = grouped.ffill().bfill()
        except Exception:
            pass
        values = np.asarray(grouped.values, dtype=float)
        # Safety: no NaN/0-fill may reach the fitter.
        try:
            if np.isnan(values).any():
                _fill = float(np.nanmedian(values)) if np.isfinite(
                    np.nanmedian(values)) else 0.0
                values = np.where(np.isnan(values), _fill, values)
                n_masked += 1
        except Exception:
            pass
        if period_code == "h":
            labels = [pd.Timestamp(t).strftime("%Y-%m-%d %H:%M")
                      for t in grouped.index]
        else:
            labels = [pd.Timestamp(t).strftime("%Y-%m-%d") for t in grouped.index]
        # Post-aggregation quality flags — info for validation, never silent.
        std = float(np.std(values)) if len(values) else 0.0
        mean = float(np.mean(values)) if len(values) else 0.0
        constant = bool(len(values) >= 2 and (std < 1e-9 or pd.Series(values).nunique() <= 1))
        cv = (std / abs(mean)) if abs(mean) > 1e-12 else (float("inf") if std > 0 else 0.0)
        near_constant = bool(not constant and len(values) >= 3 and math.isfinite(cv) and cv < 1e-3)
        outliers = self._outlier_count(values)
        if n_rows > len(values):
            agg_level = (f"transaction_rows_aggregated_to_{period_code}_"
                         f"by_{_agg_method}")
        elif duplicates_merged:
            agg_level = (f"duplicate_timestamps_aggregated_to_{period_code}_"
                         f"by_{_agg_method}")
        else:
            agg_level = f"already_at_{period_code}_grain"
        diagnostics = {
            "time_grain": period_code,
            "duplicate_timestamps_merged": duplicates_merged,
            "missing_periods": missing_periods,
            "missing_periods_interpolated": n_interpolated,
            "missing_periods_masked": n_masked,
            "missing_periods_filled": missing_periods,
            "gap_method": ("interpolate_short_runs_max2_then_mask_long "
                           "(never 0-fill for fitting)"),
            "median_delta_days": round(median_delta, 2),
            "frequency": _PERIOD_NAMES.get(period_code, period_code),
            "n_obs_raw": int(n_rows),
            "n_obs": int(len(values)),
            "missing_values": int(n_bad_time + n_bad_value),
            "parse_ratio": round(float(_parse_ratio), 4),
            "outlier_count": outliers,
            "outlier_method": "IQR+zscore union, info only - values kept",
            "irregular_intervals": bool(irregular),
            "aggregation_level": agg_level,
            "aggregation_method": _agg_method,
            "aggregation_reason": _agg_reason,
            "tz_normalized": bool(tz_normalized),
            "tz_original": tz_original,
            "tz_note": ("timestamps normalized to UTC-naive"
                        if tz_normalized else "no timezone normalization needed"),
            "constant": constant,
            "near_constant": near_constant,
            "temporal_variation": bool(not constant and len(values) >= 2),
        }
        return values, labels, period_code, diagnostics

    @staticmethod
    def _outlier_count(values):
        """IQR + z-score union count. Info only — values are never removed."""
        v = np.asarray(values, dtype=float)
        if len(v) < 4:
            return 0
        try:
            q1, q3 = float(np.percentile(v, 25)), float(np.percentile(v, 75))
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            iqr_mask = (v < lo) | (v > hi)
        except Exception:
            iqr_mask = np.zeros(len(v), dtype=bool)
        try:
            std = float(np.std(v))
            mean = float(np.mean(v))
            z_mask = np.abs(v - mean) > 3 * std if std > 1e-12 else np.zeros(len(v), dtype=bool)
        except Exception:
            z_mask = np.zeros(len(v), dtype=bool)
        return int(np.sum(iqr_mask | z_mask))

    def _holdout_extended(self, values, model_cls, season_period):
        """Extended metrics on a train-earlier/validate-later trailing holdout.

        Fits on values[:-h], scores against values[-h:] — never a random
        split. Returns {mae, rmse, mape, smape, wape, bias, holdout_points}
        with None where unmeasurable.
        """
        from .evaluate import holdout_backtest as _hb
        n = len(values)
        h = min(3, max(1, n // 4)) if n > 1 else 0
        if h < 1 or (n - h) < 3:
            return {"mae": None, "rmse": None, "mape": None, "smape": None,
                    "wape": None, "bias": None, "holdout_points": 0}
        base = _hb(model_cls, values, season_period, h)
        try:
            from .evaluate import _make as _mk
            pred = np.asarray(_mk(model_cls, season_period).fit(values[:-h]).predict(h),
                              dtype=float)
            actual = np.asarray(values[-h:], dtype=float)
            return {
                "mae": base.get("mae"),
                "rmse": base.get("rmse"),
                "mape": base.get("mape"),
                "smape": _smape(actual, pred),
                "wape": _wape(actual, pred),
                "bias": round(_bias(actual, pred), 4) if _bias(actual, pred) is not None else None,
                "holdout_points": int(h),
            }
        except Exception:
            return {"mae": base.get("mae"), "rmse": base.get("rmse"),
                    "mape": base.get("mape"), "smape": None, "wape": None,
                    "bias": None, "holdout_points": int(h)}

    def _compute_drivers(self, values, season_period, target_col, period_name,
                         raw_df=None, time_col=None):
        """Dataset-derived drivers: trend, seasonality, volatility, top
        categorical contributor. Every string embeds measured numbers —
        nothing generic or hardcoded."""
        v = np.asarray(values, dtype=float)
        n = len(v)
        drivers = []
        mean = float(np.mean(v)) if n else 0.0
        std = float(np.std(v)) if n else 0.0
        # 1. trend slope (least squares, per-period).
        try:
            slope = float(np.polyfit(np.arange(n, dtype=float), v, 1)[0]) if n >= 2 else 0.0
        except Exception:
            slope = 0.0
        direction = "upward" if slope > 0 else ("downward" if slope < 0 else "flat")
        drivers.append({
            "name": f"trend in {target_col}",
            "evidence": (
                f"least-squares slope {slope:+.4f} per {period_name} period "
                f"over {n} periods (direction: {direction})"
            ),
            "metric_value": round(slope, 4),
        })
        # 2. seasonality strength at the seasonal lag (or best scanned lag).
        if season_period and season_period >= 2:
            ac = _lag_autocorr(v, season_period)
            drivers.append({
                "name": f"seasonality (lag {season_period})",
                "evidence": (
                    f"autocorrelation {ac:.3f} at seasonal lag {season_period} "
                    f"for this {period_name} series of {n} periods"
                ),
                "metric_value": round(ac, 3),
            })
        else:
            best_lag, best_ac = 0, 0.0
            for lag in range(2, min(12, n // 2) + 1):
                ac = _lag_autocorr(v, lag)
                if abs(ac) > abs(best_ac):
                    best_lag, best_ac = lag, ac
            drivers.append({
                "name": f"seasonality (lag {best_lag})" if best_lag else "seasonality (no lag scannable)",
                "evidence": (
                    f"strongest autocorrelation {best_ac:.3f} at lag {best_lag} "
                    f"scanning lags 2..{min(12, n // 2)} over {n} periods"
                    if best_lag else
                    f"only {n} periods — too short to scan seasonal lags"
                ),
                "metric_value": round(best_ac, 3),
            })
        # 3. volatility (coefficient of variation).
        cv = (std / abs(mean)) if abs(mean) > 1e-12 else (float("inf") if std > 0 else 0.0)
        drivers.append({
            "name": f"volatility of {target_col}",
            "evidence": (
                f"coefficient of variation "
                f"{cv:.3f}" if math.isfinite(cv) else "coefficient of variation undefined (zero mean)"
            ) + f" (std {std:.2f} vs mean {mean:.2f}) over {n} periods",
            "metric_value": round(cv, 3) if math.isfinite(cv) else None,
        })
        # 4. top categorical contributor (contribution analysis on raw rows).
        contrib = self._top_contributor(raw_df, target_col, exclude=time_col)
        if contrib:
            col, val, share, top_sum, total, ngroups = contrib
            drivers.append({
                "name": f"top contributor: {col}",
                "evidence": (
                    f"{col}='{val}' contributes {share:.1f}% of total {target_col} "
                    f"({top_sum:,.2f} of {total:,.2f} across {ngroups} groups)"
                ),
                "metric_value": round(share, 2),
            })
        return drivers

    @staticmethod
    def _top_contributor(raw_df, target_col, exclude=None):
        """Highest-share category by summed target across usable cat columns."""
        if raw_df is None or target_col not in raw_df.columns:
            return None
        try:
            try:
                totals, _ = _coerce_locale_series(raw_df[target_col])
            except Exception:
                totals = pd.to_numeric(raw_df[target_col], errors="coerce")
            grand = float(totals.sum())
            if not math.isfinite(grand) or abs(grand) < 1e-12:
                return None
            best = None
            for c in raw_df.columns:
                if c == target_col or (exclude is not None and c == exclude):
                    continue
                s = raw_df[c]
                if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_object_dtype(s):
                    continue
                try:
                    nunique = int(s.nunique(dropna=True))
                except Exception:
                    continue
                if nunique < 2 or nunique > 30:
                    continue
                try:
                    grouped = totals.groupby(s.fillna("∅")).sum()
                    if grouped.empty:
                        continue
                    top_val = grouped.abs().idxmax()
                    top_sum = float(grouped.loc[top_val])
                    share = abs(top_sum) / abs(grand) * 100.0
                    if best is None or share > best[2]:
                        best = (str(c), str(top_val), share, top_sum, grand, nunique)
                except Exception:
                    continue
            return best
        except Exception:
            return None

    def _build(self, values, labels, col, period_name, period_code, horizon,
               season_period=None, diagnostics=None, n_extra_features=0,
               raw_df=None, time_col=None):
        conf_thresh = int(getattr(self, "confidence_threshold", None)
                          or CONFIDENCE_THRESHOLD_N)
        values = np.asarray(values, dtype=float)
        frequency_label = _PERIOD_NAMES.get(period_code, "sequence") \
            if period_code else "sequence"
        n_obs = int(len(values))
        horizon_suggested = _suggest_horizon(frequency_label, n_obs)
        auto_h = bool(getattr(self, "auto_horizon", True))
        if horizon is not None:
            horizon_eff = horizon
        elif auto_h:
            horizon_eff = horizon_suggested
        else:
            horizon_eff = getattr(self, "horizon", horizon_suggested)
        try:
            horizon_eff = int(horizon_eff)
        except Exception:
            horizon_eff = horizon_suggested
        horizon_eff = max(1, horizon_eff)
        # Validation (Phases 10/14): sufficient unique history + meaningful
        # variance. Either failure returns an explanation with NO forecast
        # points — never a fake-looking projection from inadequate data.
        if len(values) < self.min_points:
            return self._insufficient(
                col, period_name, period_code, horizon_eff,
                f"only {len(values)} history period(s) found — "
                f"at least {self.min_points} are needed for a reliable forecast. "
                "Try a different number with more history, or connect a "
                "longer series and try again.",
                diagnostics=diagnostics,
                missing_field="history",
            )
        if float(np.std(values)) < 1e-9 or pd.Series(values).nunique() <= 1:
            out = self._insufficient(
                col, period_name, period_code, horizon_eff,
                f"'{col}' shows no meaningful variation over "
                f"{len(values)} periods (constant history) — there is no "
                "trend to project. Try a different number with changing "
                "values instead.",
                diagnostics=diagnostics,
                missing_field="temporal_variation",
            )
            out.update({"ok": False, "suggestion": ANALYZE_SUGGESTION})
            return out
        preliminary = bool(n_obs < conf_thresh)
        # Intelligent model selection: diagnose the series, shortlist by rule
        # (little history -> naive/seasonal-naive; seasonality -> seasonal;
        # trend+seasonality -> ETS/SARIMA; feature-rich -> ML note;
        # irregular -> unreliable), backtest the shortlist on a trailing
        # holdout and pick the winner on holdout error.
        model_spec, sel_metrics, sel_meta = _selection.select_intelligent(
            values, season_period,
            n_extra_features=n_extra_features,
            min_points=self.min_points,
        )
        validated = model_spec is not None
        if not validated:
            # Defensive: select_intelligent only declines when n < min_points
            # (handled above), so reaching here means no candidate could be
            # scored — explain instead of emitting an unvalidated projection.
            return self._insufficient(
                col, period_name, period_code, horizon_eff,
                "no candidate model could be validated on this history "
                "(too little data for holdout/walk-forward scoring), "
                "so no forecast was generated.",
                diagnostics=diagnostics,
                missing_field="history",
            )
        # Flat-line guard (pre-refit probe): a validated winner that
        # projects a constant level on genuinely varying history draws a
        # flat line users read as "no forecast". A post-refit safety net
        # below re-checks the FINAL points (probe and final fit can differ).
        _probe_steps = max(1, min(int(horizon_eff or 1), len(values)))
        if season_period:
            _probe_steps = max(_probe_steps,
                               min(int(season_period), len(values)))
        try:
            _orig_winner = sel_meta.get("winner") or getattr(model_spec, "name", "none")
        except Exception:
            _orig_winner = "none"
        model_spec, sel_meta, sel_metrics, _sub_note = \
            _substitute_flat_winner(values, season_period, model_spec,
                                    sel_meta, sel_metrics, _probe_steps)
        from .evaluate import one_step_ahead_mae as _oos_mae
        from .evaluate import one_step_ahead_mape as _oos_mape
        oos_mape = _oos_mape(model_spec, values, season_period) \
            if isinstance(model_spec, type) else sel_meta.get("walk_forward_mape")
        oos_mae = _oos_mae(model_spec, values, season_period) \
            if isinstance(model_spec, type) else sel_meta.get("walk_forward_mae")

        # An EnsembleModel arrives already fitted, but its .fit(values) is also a
        # no-op that records the series for fitted()/residuals(); classes are
        # fitted here on the full series — via the same constructor the
        # backtests used, so a seasonal winner keeps its real period
        # (seasonal_naive refit with a default period 7 on monthly data, or
        # auto_ets refit without its period, silently mis-forecast).
        if isinstance(model_spec, type):
            from .evaluate import _make as _mk_final
            model = _mk_final(model_spec, season_period).fit(values)
        else:
            model = model_spec.fit(values)
        # Horizon is capped at history length (extrapolating further is not
        # supported). The cap is now disclosed via horizon_capped/details
        # instead of silent min() — numeric behavior unchanged.
        try:
            _req_steps = max(1, int(horizon_eff))
        except (TypeError, ValueError):
            _req_steps = 1
        steps = max(1, min(_req_steps, len(values)))
        horizon_capped = _req_steps > len(values)
        horizon_note = (
            f"Requested horizon {_req_steps} exceeds history length {len(values)}; "
            f"capped to {steps}."
            if horizon_capped else None
        )
        points = np.asarray(model.predict(steps), dtype=float)

        try:
            resid = model.residuals(values)
        except Exception:
            try:
                resid = np.asarray(values, dtype=float) - np.asarray(model.fitted(), dtype=float)
            except Exception:
                resid = np.zeros_like(values, dtype=float)
        std = float(np.std(resid)) if len(resid) > 1 else 0.0
        # Prefer the honest out-of-sample error for the confidence band width;
        # fall back to in-sample std only when no walk-forward could be run.
        if oos_mae:
            # MAE ~= 0.8 * sigma for a Gaussian; rescale to an approximate sigma.
            sigma = oos_mae / 0.8
        else:
            sigma = std
        # Post-refit safety net: the pre-refit probe can disagree with the
        # final fit (refit nondeterminism, ensemble path). If the FINAL
        # points are flat on varying history, substitute and rebuild
        # points/bands from the alternate — a flat line is never shipped
        # silently.
        if (not _forecast_varies(points) and _history_varies(values)):
            _post_spec, _post_meta, _post_met, _post_note = \
                _substitute_flat_winner(values, season_period, model_spec,
                                        sel_meta, sel_metrics, steps)
            if _post_note is not None and _post_spec is not model_spec:
                try:
                    from .evaluate import _make as _mk_post
                    model_spec, sel_meta, sel_metrics = \
                        _post_spec, _post_meta, _post_met
                    model = _mk_post(model_spec, season_period).fit(values) \
                        if isinstance(model_spec, type) \
                        else model_spec.fit(values)
                    points = np.asarray(model.predict(steps), dtype=float)
                    try:
                        resid = model.residuals(values)
                    except Exception:
                        try:
                            resid = np.asarray(values, dtype=float) - np.asarray(
                                model.fitted(), dtype=float)
                        except Exception:
                            resid = np.zeros_like(values, dtype=float)
                    std = float(np.std(resid)) if len(resid) > 1 else 0.0
                    if isinstance(model_spec, type):
                        oos_mape = _oos_mape(model_spec, values, season_period)
                        oos_mae = _oos_mae(model_spec, values, season_period)
                    else:
                        oos_mape = sel_meta.get("walk_forward_mape")
                        oos_mae = sel_meta.get("walk_forward_mae")
                    sigma = oos_mae / 0.8 if oos_mae else std
                    _sub_note = ((_sub_note + " ") if _sub_note else "") + _post_note
                    log.info("flat-line guard (post-refit): %s", _post_note)
                except Exception:
                    log.debug("flat-line guard post-refit rebuild failed",
                              exc_info=True)
        # Generic wave guard: a flat level or a straight line that suppresses
        # clear observed seasonality still reads as "no forecast" on unseen
        # databases. Replace it with a trend+seasonal projection derived only
        # from the actual history (never invented wiggle). Disclosed below.
        _wave_note = None
        try:
            _wavy, _wave_note = _ensure_wavy_forecast(
                values, points, season_period, period_code)
            if _wave_note is not None:
                points = np.asarray(_wavy, dtype=float)
                _sub_note = ((_sub_note + " ") if _sub_note else "") + _wave_note
                log.info("wave guard: %s", _wave_note)
                # Points now come from the substituted projection, so the
                # accuracy/band inputs computed for the original winner are
                # stale. Rebuild oos_mape/oos_mae/resid/sigma from the
                # projection that SHIPS (same one-step helpers the flat-line
                # branch re-runs): validation.accuracy_pct and the 80/95%
                # bands must never describe a model whose points were
                # discarded. On rebuild failure accuracy becomes
                # not-applicable rather than stale.
                try:
                    oos_mape, oos_mae, resid = _wave_metrics(
                        values, season_period, period_code)
                    std = float(np.std(resid)) if len(resid) > 1 else 0.0
                    sigma = oos_mae / 0.8 if oos_mae else std
                except Exception:
                    log.debug("wave guard metric rebuild failed",
                              exc_info=True)
                    oos_mape = oos_mae = None
        except Exception:
            log.debug("wave guard failed", exc_info=True)
        last_actual = float(values[-1])
        accuracy = accuracy_from_mape(oos_mape)
        # Headline change vs a trailing baseline average — never vs a
        # near-zero last actual (which would explode to ±inf%).
        projected_pct, _proj_baseline, _proj_suppressed = \
            _headline_pct_change(points, values, season_period)

        # Validation gate: only forecasts whose backtested accuracy clears the
        # threshold are "valid". Poor/unknown accuracy -> not usable downstream.
        # Zero-actual periods are excluded from MAPE (undefined at 0), so a
        # sparse series must also show enough evaluated origins — otherwise a
        # 95% accuracy on 2 of 10 periods would mislead.
        from prediction.evaluate import walk_forward_coverage as _coverage
        cov = _coverage(values, season_period)
        zero_frac = (cov["zero_skipped"] / cov["origins"]) if cov["origins"] else 0.0
        enough_coverage = cov["evaluated"] >= 3 and zero_frac < 0.5
        valid = accuracy is not None and accuracy >= self.valid_threshold and enough_coverage
        if not validated:
            reason = "insufficient history for walk-forward validation"
        elif accuracy is None:
            reason = "accuracy could not be measured (no walk-forward errors)"
        elif not enough_coverage:
            reason = (
                f"accuracy {accuracy}% covers only {cov['evaluated']}/{cov['origins']} "
                f"periods ({cov['zero_skipped']} zero-demand periods excluded from MAPE)"
            )
        else:
            reason = (
                f"backtest accuracy {accuracy}% {'meets' if valid else 'below'} "
                f"the {self.valid_threshold}% threshold"
            )
        forecast_stage = "preliminary" if preliminary else "full"
        reliability = _reliability_level(accuracy, preliminary=preliminary)
        validation = {
            "method": "expanding-window walk-forward (one-step-ahead, out-of-sample)",
            "metric": "MAPE",
            "accuracy_pct": accuracy,
            "threshold_pct": self.valid_threshold,
            "passed": valid,
            "reason": reason,
            "walk_forward_origins": cov["origins"],
            "walk_forward_evaluated": cov["evaluated"],
            "zero_actuals_skipped": cov["zero_skipped"],
            "n_obs": n_obs,
            "confidence_threshold": conf_thresh,
            "forecast_stage": forecast_stage,
        }

        # Data contract (Phase 12 / §16): backend timestamps only. `historical`
        # carries date->value pairs, `forecast` carries future timestamps with
        # a confidence interval. `confidence_interval` mirrors the 80/95% bands
        # per forecast date and `timestamps` lists every backend date in plot
        # order (history then forecast). The frontend plots these verbatim and
        # never invents dates; legacy `history`/`points` stay for compatibility.
        model_name = getattr(model, "name", "none")
        frequency = period_code or "sequence"
        fwd_labels = _forecast_labels(labels[-1], steps, period_code)
        # Structured guard disclosure (additive): which projection replaced
        # what and why. None when the validated winner stands unchanged.
        try:
            _final_name = getattr(model, "name", None) or sel_meta.get("winner") or model_name
        except Exception:
            _final_name = model_name
        _wave_fired = bool(_wave_note)
        _sub_fired = bool(_sub_note) and not _wave_fired
        _adjusted_fired = bool(_sub_note or _wave_note)
        if _wave_fired:
            _adj_to = f"trend+seasonal projection ({_final_name} base)"
        elif _sub_fired:
            _adj_to = str(_final_name)
        else:
            _adj_to = str(_final_name)
        forecast_adjusted = None
        if _adjusted_fired:
            forecast_adjusted = {
                "from": str(_orig_winner),
                "to": _adj_to,
                "reason": str((_sub_note or "").strip()),
            }
        # Confidence bands widen with horizon: sigma_h = sigma_eff * sqrt(h).
        # n>=30: empirical residual quantiles scale the bands (honest,
        # distribution-free); n<30: gaussian sigma with an unreliable flag.
        _sigmas = _widening_sigmas(sigma, steps, values)
        _n_for_intervals = int(n_obs)
        _use_empirical = bool(_n_for_intervals >= 30)
        _q10 = _q90 = _q025 = _q975 = None
        _hw80_emp = _hw95_emp = None
        if _use_empirical:
            try:
                _r = np.asarray(resid, dtype=float)
                _r = _r[np.isfinite(_r)]
                if len(_r) >= 30:
                    _q10, _q90 = float(np.quantile(_r, 0.10)), float(np.quantile(_r, 0.90))
                    _q025, _q975 = float(np.quantile(_r, 0.025)), float(np.quantile(_r, 0.975))
                    _hw80_emp, _hw95_emp = abs(_q90 - _q10) / 2.0, abs(_q975 - _q025) / 2.0
                else:
                    _use_empirical = False
            except Exception:
                _use_empirical = False
        if _use_empirical:
            interval_method = "empirical_residual_quantiles_sqrt_h"
            interval_unreliable = False
        else:
            interval_method = "gaussian_sigma_sqrt_h"
            interval_unreliable = True

        def _band_hw(i, which):
            import math as _m
            _w = _m.sqrt(i + 1)
            if _use_empirical and _hw80_emp is not None and _hw95_emp is not None:
                base = _hw80_emp if which == 80 else _hw95_emp
                try:
                    _v = np.asarray(values, dtype=float)
                    _scale = max(1.0, abs(float(np.mean(np.abs(_v[np.isfinite(_v)])))))
                except Exception:
                    _scale = 1.0
                base = max(float(base), 0.005 * float(_scale))
                return float(base) * _w
            if which == 80:
                return float(1.28 * _sigmas[i])
            return float(1.96 * _sigmas[i])

        forecast_points = [
            {
                "label": fwd_labels[i],
                "value": round(float(points[i]), 2),
                "forecast": round(float(points[i]), 2),
                "lower80": round(float(points[i] - _band_hw(i, 80)), 2),
                "upper80": round(float(points[i] + _band_hw(i, 80)), 2),
                "lower95": round(float(points[i] - _band_hw(i, 95)), 2),
                "upper95": round(float(points[i] + _band_hw(i, 95)), 2),
                "interval_method": interval_method,
                **({"interval_unreliable": True} if interval_unreliable else {}),
            }
            for i in range(steps)
        ]
        # Trailing actuals so the UI can draw one continuous
        # history + projection chart (last 24 periods max).
        history = [
            {"label": lab, "actual": round(float(v), 2)}
            for lab, v in zip(labels[-24:], values[-24:])
        ]
        historical = [
            {"date": lab, "value": round(float(v), 2)}
            for lab, v in zip(labels[-24:], values[-24:])
        ]
        forecast_series = [
            {
                "date": fwd_labels[i],
                "value": round(float(points[i]), 2),
                "lower": round(float(points[i] - _band_hw(i, 95)), 2),
                "upper": round(float(points[i] + _band_hw(i, 95)), 2),
            }
            for i in range(steps)
        ]
        confidence_interval = [
            {
                "date": fwd_labels[i],
                "lower80": round(float(points[i] - _band_hw(i, 80)), 2),
                "upper80": round(float(points[i] + _band_hw(i, 80)), 2),
                "lower95": round(float(points[i] - _band_hw(i, 95)), 2),
                "upper95": round(float(points[i] + _band_hw(i, 95)), 2),
                "interval_method": interval_method,
                **({"interval_unreliable": True} if interval_unreliable else {}),
            }
            for i in range(steps)
        ]
        # Metrics: holdout winner scores first, honest walk-forward as the
        # fallback so mae/rmse/mape are always populated when measurable.
        rep_mape = sel_metrics.get("mape")
        if rep_mape is None and oos_mape is not None:
            rep_mape = round(oos_mape, 2)
        rep_mae = sel_metrics.get("mae")
        if rep_mae is None and oos_mae is not None:
            rep_mae = round(oos_mae, 2)
        rep_rmse = sel_metrics.get("rmse")
        if rep_rmse is None:
            try:
                rep_rmse = round(_rmse(values, np.asarray(model.fitted(), dtype=float)), 2)
            except Exception:
                rep_rmse = None
        # Extended holdout metrics (train-earlier/validate-later split).
        extended = self._holdout_extended(values, model_spec if isinstance(model_spec, type)
                                          else model_spec.__class__, season_period)
        metrics = {"mae": rep_mae, "rmse": rep_rmse, "mape": rep_mape}
        selection_statement = sel_meta.get("statement") or (
            f"Selected {model_name} { _selection.HOLDOUT_STATEMENT}."
            if validated else "insufficient history for holdout validation")
        try:
            if _wave_note:
                selection_statement = (selection_statement + " " + _wave_note).strip()
        except Exception:
            pass
        model_notes = dict(sel_meta)
        try:
            model_notes.setdefault("sigma", round(float(_sigmas[0]), 2))
        except Exception:
            model_notes.setdefault("sigma", round(float(sigma), 2))
        if diagnostics:
            model_notes.setdefault("series_diagnostics", diagnostics)
        # Sparse / intermittent-demand hint (additive): high zero share
        # means MAPE covers only non-zero periods — flag directionally.
        try:
            _zero_frac_hist = float(np.mean(np.asarray(values) == 0.0)) if len(values) else 0.0
        except Exception:
            _zero_frac_hist = 0.0
        sparse_hint = None
        if _zero_frac_hist >= 0.3 or zero_frac >= 0.3:
            sparse_hint = (
                f"sparse/intermittent demand: "
                f"{round(_zero_frac_hist * 100.0, 1)}% zero periods in history "
                f"({cov.get('zero_skipped', 0)}/{cov.get('origins', 0)} walk-forward "
                f"origins excluded from MAPE) — treat the projection as "
                f"directional, not precise."
            )
            if "reliability_note" in model_notes and model_notes["reliability_note"]:
                model_notes["reliability_note"] = (
                    str(model_notes["reliability_note"]) + " " + sparse_hint)
            else:
                model_notes["sparse_demand_note"] = sparse_hint

        # Perfect-fit honesty: a zero-width interval is observed error, not
        # certainty. The bands stay as computed; the note tells every
        # consumer (story, prescriptions, UI) to treat it as uncertain.
        try:
            perfect_fit = bool(valid) and float(sigma) <= 1e-12
        except (TypeError, ValueError):
            perfect_fit = False
        uncertainty_note = (
            "Very low observed error — treat as uncertain, not guaranteed"
            if perfect_fit else None
        )
        # Documented flat line: no validated candidate varies on this
        # history, so the level projection stands — the note (surfaced in
        # the UI) says why instead of leaving a flat line unexplained.
        try:
            if (uncertainty_note is None and _sub_note is None
                    and not _forecast_varies(points)
                    and _history_varies(values)):
                uncertainty_note = (
                    "Validated models project a steady level on this history — "
                    "the flat line is the backtested signal, not missing data."
                )
        except Exception:
            pass

        # Candidates with scores (best first) + the winner — all derived
        # from the same trailing-holdout backtest, never invented.
        ranking = sel_meta.get("ranking") or []
        candidates = [
            {"model": r.get("model"), "mae": r.get("mae"), "rmse": r.get("rmse"),
             "mape": r.get("mape"), "holdout_points": r.get("holdout_points")}
            for r in ranking
        ]
        model_selected = sel_meta.get("winner") or model_name
        drivers = self._compute_drivers(values, season_period, col, period_name,
                                        raw_df=raw_df, time_col=time_col)
        preparation = {
            "time_column": time_col,
            "target_column": col,
            "frequency": frequency_label,
            "frequency_detection": {
                "method": "median timestamp delta + coverage sanity (h/D/W/MS/Q/YS)",
                "median_delta_days": (diagnostics or {}).get("median_delta_days"),
                "time_grain": (diagnostics or {}).get("time_grain"),
            },
            "n_obs_raw": (diagnostics or {}).get("n_obs_raw", n_obs),
            "n_obs": n_obs,
            "missing_periods": (diagnostics or {}).get("missing_periods", 0),
            "missing_periods_interpolated": (diagnostics or {}).get(
                "missing_periods_interpolated", 0),
            "missing_periods_masked": (diagnostics or {}).get(
                "missing_periods_masked", 0),
            "missing_periods_filled": (diagnostics or {}).get("missing_periods_filled", 0),
            "gap_method": (diagnostics or {}).get("gap_method"),
            "missing_values": (diagnostics or {}).get("missing_values", 0),
            "parse_ratio": (diagnostics or {}).get("parse_ratio", 1.0),
            "duplicate_timestamps_merged": (diagnostics or {}).get(
                "duplicate_timestamps_merged", 0),
            "duplicate_aggregation": (diagnostics or {}).get("aggregation_method", "sum"),
            "aggregation_method": (diagnostics or {}).get("aggregation_method", "sum"),
            "aggregation_reason": (diagnostics or {}).get("aggregation_reason"),
            "constant": (diagnostics or {}).get("constant", False),
            "near_constant": (diagnostics or {}).get("near_constant", False),
            "temporal_variation": (diagnostics or {}).get("temporal_variation", True),
            "outlier_count": (diagnostics or {}).get("outlier_count", 0),
            "outlier_method": (diagnostics or {}).get(
                "outlier_method", "IQR+zscore union, info only - values kept"),
            "irregular_intervals": (diagnostics or {}).get("irregular_intervals", False),
            "aggregation_level": (diagnostics or {}).get("aggregation_level"),
            "tz_normalized": (diagnostics or {}).get("tz_normalized", False),
            "tz_note": (diagnostics or {}).get("tz_note"),
            "sparse_demand_hint": sparse_hint,
        }
        if accuracy is not None:
            accuracy_summary = (
                f"Backtested accuracy {accuracy}% (walk-forward MAPE "
                f"{round(oos_mape, 2) if oos_mape is not None else 'n/a'}%) over "
                f"{cov['evaluated']}/{cov['origins']} origins — {reliability} reliability."
                + (" Preliminary: fewer than "
                   f"{conf_thresh} observations (n={n_obs})." if preliminary else "")
            )
        else:
            accuracy_summary = (
                "Accuracy could not be measured on this history — "
                "treat the projection as directional."
            )
        details = {
            "metrics_extended": {
                "mae": rep_mae, "rmse": rep_rmse, "mape": rep_mape,
                "smape": extended.get("smape"), "wape": extended.get("wape"),
                "bias": extended.get("bias"),
                "holdout_points": extended.get("holdout_points"),
            },
            "coverage": cov,
            "selection": {k: v for k, v in sel_meta.items()},
            "diagnostics": dict(diagnostics or {}),
            "horizon": {"requested": horizon, "suggested": horizon_suggested,
                        "used": steps, "frequency": frequency_label,
                        "capped": horizon_capped, "note": horizon_note},
            "confidence_threshold": conf_thresh,
            "sigma": round(float(_sigmas[0]), 2) if "_sigmas" in locals() else round(float(sigma), 2),
            "interval_widening": "sqrt(h)",
            "interval_method": interval_method,
            "interval_unreliable": interval_unreliable,
            "interval_quantiles": (
                {"q10": round(_q10, 2), "q90": round(_q90, 2),
                 "q2_5": round(_q025, 2), "q97_5": round(_q975, 2)}
                if _use_empirical else None),
            "forecast_adjusted": forecast_adjusted,
            "projected_baseline": _proj_baseline,
            "projected_pct_suppressed": bool(_proj_suppressed),
            "legacy_reliability_label": _legacy_reliability_label(accuracy),
        }

        return {
            "ok": True,
            "column": col,
            "period": period_name,
            "horizon": steps,
            "horizon_suggested": horizon_suggested,
            "horizon_capped": horizon_capped,
            "horizon_note": horizon_note,
            "forecast_stage": forecast_stage,
            "method": model_name,
            "model_selected": model_selected,
            "candidates": candidates,
            "accuracy_pct": accuracy,
            "accuracy": {"pct": accuracy, "summary": accuracy_summary},
            "mape_pct": round(oos_mape, 2) if oos_mape is not None else None,
            "reliability": reliability,
            "valid": valid,
            "reason": None if valid else reason,
            "message": PRELIMINARY_MESSAGE if preliminary else None,
            "recommendation": PRELIMINARY_RECOMMENDATION if preliminary else None,
            "n_obs": n_obs,
            "uncertainty_note": uncertainty_note,
            "interval_method": interval_method,
            "interval_unreliable": interval_unreliable,
            "interval_quantiles": details.get("interval_quantiles"),
            "forecast_adjusted": forecast_adjusted,
            "projected_baseline": _proj_baseline,
            "projected_pct_suppressed": bool(_proj_suppressed),
            "sparse_demand_hint": sparse_hint,
            "validation": validation,
            "preparation": preparation,
            "model_notes": model_notes,
            "last_actual": {"label": labels[-1], "value": round(last_actual, 2)},
            "projected_pct_change": projected_pct,
            "points": forecast_points,
            "history": history,
            "drivers": drivers,
            "details": details,
            # ---- data contract (Phase 12 / §16) ----
            "historical": historical,
            "forecast": forecast_series,
            "confidence_interval": confidence_interval,
            "timestamps": [h["date"] for h in historical]
            + [f["date"] for f in forecast_series],
            "frequency": frequency,
            "target": col,
            "model": model_name,
            "metrics": metrics,
            "selection_statement": selection_statement,
        }

    def _insufficient(self, col, period_name, period_code, horizon, reason,
                      diagnostics=None, missing_field="history"):
        """Contract-safe insufficient-history response: explanation, no
        forecast points — never a fake-looking projection."""
        out = self._empty(reason, "insufficient_data")
        out.update({
            "column": col,
            "period": period_name,
            "horizon": int(horizon or 0),
            "frequency": period_code or "sequence",
            "target": col,
            "selection_statement": None,
            "ok": False,
            "missing_field": missing_field,
            "suggestion": ANALYZE_SUGGESTION,
        })
        if diagnostics:
            out["model_notes"] = {"series_diagnostics": diagnostics}
            out["preparation"] = {
                "time_column": None,
                "target_column": col,
                "frequency": _PERIOD_NAMES.get(period_code, "sequence")
                if period_code else "sequence",
                "n_obs": diagnostics.get("n_obs", 0),
                "n_obs_raw": diagnostics.get("n_obs_raw", 0),
                "missing_periods": diagnostics.get("missing_periods", 0),
                "missing_values": diagnostics.get("missing_values", 0),
                "duplicate_timestamps_merged": diagnostics.get(
                    "duplicate_timestamps_merged", 0),
                "constant": diagnostics.get("constant", False),
                "near_constant": diagnostics.get("near_constant", False),
                "temporal_variation": diagnostics.get("temporal_variation", False),
                "irregular_intervals": diagnostics.get("irregular_intervals", False),
                "aggregation_level": diagnostics.get("aggregation_level"),
            }
        return out

    def _empty(self, reason, reliability):
        return {
            "ok": False,
            "column": None, "period": None, "horizon": 0,
            "horizon_suggested": None,
            "forecast_stage": "none",
            "method": "none", "model_selected": None, "candidates": [],
            "accuracy_pct": None, "accuracy": {"pct": None, "summary": str(reason)},
            "mape_pct": None,
            "reliability": reliability, "reason": reason,
            "message": None, "recommendation": None,
            "missing_field": None, "suggestion": ANALYZE_SUGGESTION,
            "n_obs": 0,
            "points": [], "history": [],
            "drivers": [], "details": {},
            "valid": False,
            "uncertainty_note": None,
            "interval_method": None,
            "interval_unreliable": True,
            "interval_quantiles": None,
            "forecast_adjusted": None,
            "projected_baseline": None,
            "projected_pct_suppressed": False,
            "sparse_demand_hint": None,
            "validation": {
                "method": None, "metric": "MAPE",
                "accuracy_pct": None, "threshold_pct": self.valid_threshold,
                "passed": False, "reason": reason,
            },
            "preparation": {
                "time_column": None, "target_column": None, "frequency": None,
                "n_obs": 0, "n_obs_raw": 0,
            },
            "last_actual": None, "projected_pct_change": None,
            "historical": [], "forecast": [], "confidence_interval": [],
            "timestamps": [], "frequency": None,
            "target": None, "model": "none",
            "metrics": {"mae": None, "rmse": None, "mape": None},
            "selection_statement": None,
            "model_notes": {},
        }

    def decision_scenario_support(self, decision: dict) -> dict:
        """PHASE 11 hook: honest forecast-backed scenario capability.

        What-If itself runs in the prescription engine (OLS lever model);
        the forecast agent only reports whether a time-series outlook can
        contextualize the decision. Never fabricates a prediction: returns
        {supported, reason} from the decision's own forecast payload.
        """
        d = decision if isinstance(decision, dict) else {}
        fc = d.get("forecast")
        if not isinstance(fc, dict) or not fc:
            return {"supported": False,
                    "reason": "No validated forecast attached to this decision."}
        pts = fc.get("points") or fc.get("forecast") or []
        if isinstance(pts, list) and len(pts) >= 2:
            return {"supported": True, "reason": None,
                    "horizon": len(pts)}
        return {"supported": False,
                "reason": "Attached forecast has no usable projection."}
