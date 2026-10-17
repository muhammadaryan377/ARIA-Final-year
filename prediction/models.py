"""Forecasting math models for ARIA predictive analysis.

Baselines (numpy only): Naive / Mean / Median / Moving average / Linear trend /
Exponential smoothing / Holt linear (damped) / Drift / Seasonal naive /
Holt-Winters additive / Theta.

Statistical models (optional deps, skipped gracefully when missing):
- AutoARIMA / SARIMA via pmdarima + statsmodels (order picked by AIC).
- AutoETS via statsmodels (error/trend/season grid picked by AIC).
- TBATS via the tbats package.
- GARCH(1,1) via the arch package (constant-mean point forecast).

Neural models (torch, CPU):
- LSTM / GRU: tiny single-layer nets, seeded and deterministic.

Models expose a uniform interface:
    fit(values) -> self
    predict(steps) -> np.ndarray (point forecasts)
    fitted() -> np.ndarray (in-sample one-step-ahead fits, same length as values)
    residuals(values) -> np.ndarray (values - fitted)

Class knobs used by prediction/evaluate.py:
    min_points  - minimum series length to even try this model.
    max_origins - cap on walk-forward origins (expensive models).
    needs_period / needs_real_period - seasonal handling.
    available() - False when an optional dependency is missing.
"""

import logging
import math

import numpy as np

log = logging.getLogger("aria.prediction.models")

try:
    import pmdarima as _pm
    _PMDARIMA_OK = True
except Exception:
    _PMDARIMA_OK = False

try:
    import statsmodels.api as _sm  # noqa: F401
    from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel as _ETSModel
    _STATSMODELS_OK = True
except Exception:
    _STATSMODELS_OK = False

try:
    from tbats import TBATS as _TBATS, BATS as _BATS
    _TBATS_OK = True
except Exception:
    _TBATS_OK = False

try:
    from arch import arch_model as _arch_model
    _ARCH_OK = True
except Exception:
    _ARCH_OK = False

try:
    import torch as _torch
    import torch.nn as _nn
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False


def _patch_sklearn_check_array():
    """Compat shim: tbats 1.1.3 calls sklearn's check_array with the removed
    `force_all_finite` kwarg (renamed to `ensure_all_finite` in sklearn 1.6+).
    Maps the old name to the new one; semantics are identical. Patches both
    the sklearn module and tbats' own imported reference to it."""
    try:
        import sklearn.utils.validation as _skv
    except Exception:
        return
    if getattr(_skv.check_array, "_aria_patched", False):
        _compat = _skv.check_array
    else:
        _orig = _skv.check_array

        def _compat(*args, **kwargs):
            if "force_all_finite" in kwargs and "ensure_all_finite" not in kwargs:
                kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
            return _orig(*args, **kwargs)

        _compat._aria_patched = True
        _skv.check_array = _compat
    try:
        import sys as _sys
        for _mod_name in ("tbats.abstract.Estimator", "tbats.abstract.Model",
                          "tbats.abstract.Case", "tbats.abstract.ParamsOptimizer"):
            _mod = _sys.modules.get(_mod_name)
            if _mod is not None and hasattr(_mod, "check_array"):
                _mod.check_array = _compat
    except Exception:
        pass


class BaseModel:
    name = "base"

    def fit(self, values):
        raise NotImplementedError

    def predict(self, steps):
        raise NotImplementedError

    def fitted(self):
        raise NotImplementedError

    def residuals(self, values):
        values = np.asarray(values, dtype=float)
        return values - np.asarray(self.fitted(), dtype=float)


class NaiveModel(BaseModel):
    """Forecast = the last observed value. The hard baseline to beat."""

    name = "naive"

    def fit(self, values):
        values = np.asarray(values, dtype=float)
        self.values = values
        self.last = float(values[-1]) if len(values) else 0.0
        return self

    def predict(self, steps):
        return np.full(steps, self.last)

    def fitted(self):
        if len(self.values) == 0:
            return np.array([])
        fitted = np.empty(len(self.values))
        fitted[0] = self.values[0]
        fitted[1:] = self.values[:-1]
        return fitted


class MeanModel(BaseModel):
    """Forecast = the historical mean. Baseline for flat series."""

    name = "mean"

    def fit(self, values):
        values = np.asarray(values, dtype=float)
        self.values = values
        self.mean = float(values.mean()) if len(values) else 0.0
        return self

    def predict(self, steps):
        return np.full(steps, self.mean)

    def fitted(self):
        return np.full(len(self.values), self.mean)


class MovingAverageModel(BaseModel):
    """Short-window rolling mean; flat forecast at the last smoothed value."""

    name = "moving_average"

    def __init__(self, window=3):
        self.window = int(window)

    def fit(self, values):
        self.values = np.asarray(values, dtype=float)
        return self

    def _smooth(self):
        n = len(self.values)
        out = np.empty(n)
        for i in range(n):
            lo = max(0, i - self.window + 1)
            out[i] = self.values[lo:i + 1].mean()
        return out

    def predict(self, steps):
        if len(self.values) == 0:
            return np.full(steps, 0.0)
        return np.full(steps, self._smooth()[-1])

    def fitted(self):
        return self._smooth()


class LinearTrendModel(BaseModel):
    """Least-squares linear trend via numpy polyfit."""

    name = "linear_trend"

    def fit(self, values):
        values = np.asarray(values, dtype=float)
        self.values = values
        n = len(values)
        t = np.arange(n, dtype=float)
        if n >= 2:
            self.coef = np.polyfit(t, values, 1)
        else:
            self.coef = np.array([0.0, values[0] if n else 0.0])
        return self

    def predict(self, steps):
        n = len(self.values)
        t = np.arange(n, n + steps, dtype=float)
        return np.polyval(self.coef, t)

    def fitted(self):
        n = len(self.values)
        return np.polyval(self.coef, np.arange(n, dtype=float))


class ExponentialSmoothingModel(BaseModel):
    """Level-only exponential smoothing; alpha grid-searched to minimize SSE.

    fitted() uses proper one-step-ahead forecasts (state from the previous step),
    so the in-sample errors reflect real predictive skill, not smoothing.
    """

    name = "exponential_smoothing"

    def fit(self, values):
        values = np.asarray(values, dtype=float)
        self.values = values
        n = len(values)
        best_alpha, best_sse = 0.5, np.inf
        for alpha in np.arange(0.05, 1.0, 0.05):
            fitted, level = self._run(alpha)
            sse = float(np.sum((values - fitted) ** 2))
            if sse < best_sse:
                best_alpha, best_sse = alpha, sse
        self.alpha = best_alpha
        self._fitted, self.level = self._run(self.alpha)
        return self

    def _run(self, alpha):
        n = len(self.values)
        fitted = np.empty(n)
        if n == 0:
            return fitted, 0.0
        level = float(self.values[0])
        fitted[0] = level
        for i in range(1, n):
            fitted[i] = level
            level = alpha * float(self.values[i]) + (1 - alpha) * level
        return fitted, level

    def predict(self, steps):
        return np.full(steps, self.level)

    def fitted(self):
        return self._fitted


class HoltLinearModel(BaseModel):
    """Holt's linear method with a damped trend (phi < 1) to avoid explosive
    long-horizon extrapolation. alpha/beta grid-searched to minimize SSE of
    one-step-ahead fits."""

    name = "holt_linear_damped"

    def __init__(self, damp=True, phi=0.9):
        self.damp = damp
        self.phi = float(phi)

    def fit(self, values):
        values = np.asarray(values, dtype=float)
        self.values = values
        best = (0.3, 0.1, np.inf)
        for alpha in np.arange(0.1, 1.0, 0.1):
            for beta in np.arange(0.05, 0.55, 0.05):
                fitted, _, _ = self._run(alpha, beta)
                sse = float(np.sum((values - fitted) ** 2))
                if sse < best[2]:
                    best = (alpha, beta, sse)
        self.alpha, self.beta = best[0], best[1]
        self._fitted, self._level, self._trend = self._run(self.alpha, self.beta)
        return self

    def _run(self, alpha, beta):
        n = len(self.values)
        fitted = np.empty(n)
        if n == 0:
            return fitted, 0.0, 0.0
        level = float(self.values[0])
        trend = float(self.values[1] - self.values[0]) if n > 1 else 0.0
        fitted[0] = level
        for i in range(1, n):
            forecast = level + self.phi * trend
            fitted[i] = forecast
            new_level = alpha * float(self.values[i]) + (1 - alpha) * forecast
            trend = beta * (new_level - level) + (1 - beta) * self.phi * trend
            level = new_level
        return fitted, level, trend

    def predict(self, steps):
        out = np.empty(steps)
        level, trend = self._level, self._trend
        for i in range(steps):
            trend = self.phi * trend
            level = level + trend
            out[i] = level
        return out

    def fitted(self):
        return self._fitted


class MedianModel(BaseModel):
    """Forecast = the historical median. Robust baseline for flat / skewed series."""

    name = "median"

    def fit(self, values):
        values = np.asarray(values, dtype=float)
        self.values = values
        self.median = float(np.median(values)) if len(values) else 0.0
        return self

    def predict(self, steps):
        return np.full(steps, self.median)

    def fitted(self):
        return np.full(len(self.values), self.median)


class DriftModel(BaseModel):
    """Forecast by projecting the average drift from the first to the last value."""

    name = "drift"

    def fit(self, values):
        values = np.asarray(values, dtype=float)
        self.values = values
        n = len(values)
        self.start = float(values[0]) if n else 0.0
        self.drift = (float(values[-1]) - self.start) / (n - 1) if n > 1 else 0.0
        return self

    def predict(self, steps):
        n = len(self.values)
        return self.start + self.drift * (n - 1 + np.arange(1, steps + 1))

    def fitted(self):
        n = len(self.values)
        return self.start + self.drift * np.arange(n)


class SeasonalNaiveModel(BaseModel):
    """Forecast each step as the value observed one full season earlier.

    The natural baseline for strongly seasonal series (e.g. day-of-week or
    month-of-year patterns)."""

    name = "seasonal_naive"
    needs_period = True

    def __init__(self, period=7):
        self.period = max(2, int(period))

    def fit(self, values):
        self.values = np.asarray(values, dtype=float)
        return self

    def predict(self, steps):
        v = self.values
        p = self.period
        n = len(v)
        if n == 0:
            return np.zeros(steps)
        out = np.empty(steps)
        for i in range(steps):
            idx = n - p + ((i) % p)
            out[i] = v[max(0, min(n - 1, idx))]
        return out

    def fitted(self):
        v = self.values
        p = self.period
        n = len(v)
        out = np.empty(n)
        for i in range(n):
            out[i] = v[i - p] if i >= p else v[i]
        return out


class HoltWintersModel(BaseModel):
    """Additive Holt-Winters (level + trend + season) with grid-searched
    alpha/beta/gamma. Season period is required and is set by the caller based on
    the detected frequency (e.g. 7 for daily, 12 for monthly)."""

    name = "holt_winters_additive"
    needs_period = True

    def __init__(self, period=7):
        self.period = max(2, int(period))

    def fit(self, values):
        values = np.asarray(values, dtype=float)
        self.values = values
        n = len(values)
        best = None
        for alpha in np.arange(0.1, 0.9, 0.1):
            for beta in np.arange(0.05, 0.45, 0.1):
                for gamma in np.arange(0.05, 0.45, 0.1):
                    try:
                        fitted = self._run(alpha, beta, gamma)[0]
                        sse = float(np.sum((values - fitted) ** 2))
                    except Exception:
                        continue
                    if best is None or sse < best[0]:
                        best = (sse, alpha, beta, gamma)
        if best is None:
            self.alpha, self.beta, self.gamma = 0.3, 0.1, 0.1
        else:
            self.alpha, self.beta, self.gamma = best[1], best[2], best[3]
        self._fitted, self._level, self._trend, self._season = self._run(
            self.alpha, self.beta, self.gamma
        )
        return self

    def _run(self, alpha, beta, gamma):
        values = self.values
        n = len(values)
        p = self.period
        if n == 0:
            return np.array([]), 0.0, 0.0, np.zeros(p)
        season = np.zeros(p)
        m = min(p, n)
        first_mean = float(values[:m].mean())
        season[:m] = values[:m] - first_mean
        level = first_mean
        trend = 0.0
        fitted = np.empty(n)
        for i in range(n):
            s_idx = i % p
            fitted[i] = level + trend + season[s_idx]
            new_level = alpha * (float(values[i]) - season[s_idx]) + (1 - alpha) * (level + trend)
            trend = beta * (new_level - level) + (1 - beta) * trend
            season[s_idx] = gamma * (float(values[i]) - new_level) + (1 - gamma) * season[s_idx]
            level = new_level
        return fitted, level, trend, season

    def predict(self, steps):
        out = np.empty(steps)
        level, trend = self._level, self._trend
        p = self.period
        n = len(self.values)
        for i in range(steps):
            out[i] = level + trend + self._season[(n + i) % p]
            level = level + trend
        return out

    def fitted(self):
        return self._fitted


class EnsembleModel(BaseModel):
    """Weighted blend of already-fitted member models (voting ensemble).

    Created by prediction/evaluate.py; `fit(values)` is a no-op because members
    are fitted externally on the full series."""

    name = "ensemble"

    def __init__(self, fitted_members=None):
        # fitted_members: list of (fitted_model_instance, weight)
        self.fitted_members = fitted_members or []
        self.values = np.array([])
        names = "+".join(m.name for m, _ in self.fitted_members[:3])
        if len(self.fitted_members) > 3:
            names += "+"
        self.name = f"ensemble({names})" if names else "ensemble"

    def fit(self, values):
        self.values = np.asarray(values, dtype=float)
        return self

    def predict(self, steps):
        preds = np.zeros(steps)
        for m, w in self.fitted_members:
            preds = preds + w * np.asarray(m.predict(steps), dtype=float)
        return preds

    def fitted(self):
        if len(self.values) == 0:
            return np.array([])
        preds = np.zeros(len(self.values))
        for m, w in self.fitted_members:
            preds = preds + w * np.asarray(m.fitted(), dtype=float)
        return preds

    def residuals(self, values):
        values = np.asarray(values, dtype=float)
        return values - np.asarray(self.fitted(), dtype=float)


class ThetaModel(BaseModel):
    """Classic Theta method: SES on the theta-0 (trend) and theta-2 lines,
    averaged. Strong, parameter-light performer on trended series."""

    name = "theta"

    def fit(self, values):
        values = np.asarray(values, dtype=float)
        self.values = values
        n = len(values)
        t = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(t, values, 1) if n >= 2 else (0.0, values[0] if n else 0.0)
        self._trend_line = intercept + slope * t
        self._theta2 = 2 * values - self._trend_line
        best_alpha, best_sse = 0.2, np.inf
        for alpha in np.arange(0.05, 1.0, 0.05):
            f0, _ = self._ses(self._trend_line, alpha)
            f2, _ = self._ses(self._theta2, alpha)
            sse = float(np.sum((values - 0.5 * (f0 + f2)) ** 2))
            if sse < best_sse:
                best_alpha, best_sse = alpha, sse
        self.alpha = best_alpha
        return self

    @staticmethod
    def _ses(series, alpha):
        n = len(series)
        out = np.empty(n)
        if n == 0:
            return out
        level = float(series[0])
        out[0] = level
        for i in range(1, n):
            out[i] = level
            level = alpha * float(series[i]) + (1 - alpha) * level
        return out, level

    def _level_of(self, series):
        _, level = self._ses(series, self.alpha)
        return level

    def predict(self, steps):
        l0 = self._level_of(self._trend_line)
        l2 = self._level_of(self._theta2)
        n = len(self.values)
        slope = (self._trend_line[-1] - self._trend_line[0]) / (n - 1) if n > 1 else 0.0
        # Theta-0 extrapolates the trend line; theta-2 continues from its level.
        f0 = l0 + slope * np.arange(1, steps + 1)
        f2 = np.full(steps, l2)
        return 0.5 * (f0 + f2)

    def fitted(self):
        f0, _ = self._ses(self._trend_line, self.alpha)
        f2, _ = self._ses(self._theta2, self.alpha)
        return 0.5 * (f0 + f2)


class AutoARIMAModel(BaseModel):
    """AutoARIMA via pmdarima (stepwise AIC search). Seasonal when a real
    season period is known and there are at least two full seasons."""

    name = "auto_arima"
    min_points = 10
    max_origins = 4

    @classmethod
    def available(cls):
        return _PMDARIMA_OK

    def __init__(self, period=None):
        self.period = int(period) if period else None

    def fit(self, values):
        if not _PMDARIMA_OK:
            raise RuntimeError("pmdarima is not installed")
        values = np.asarray(values, dtype=float)
        self.values = values
        seasonal = bool(self.period and self.period >= 2 and len(values) >= 2 * self.period)
        self._model = _pm.auto_arima(
            values,
            seasonal=seasonal,
            m=self.period if seasonal else 1,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_p=2, max_q=2, max_P=1, max_Q=1, max_d=1, max_D=1,
            information_criterion="aic",
        )
        return self

    def predict(self, steps):
        return np.asarray(self._model.predict(n_periods=int(steps)), dtype=float)

    def fitted(self):
        return np.asarray(self._model.predict_in_sample(), dtype=float)


class SARIMAModel(BaseModel):
    """Fixed-order seasonal ARIMA(1,1,1)x(1,1,1,s) via statsmodels. Only runs
    when a real season period is known."""

    name = "sarima"
    needs_period = True
    needs_real_period = True
    min_points = 12
    max_origins = 4

    @classmethod
    def available(cls):
        return _STATSMODELS_OK

    def __init__(self, period=7):
        self.period = max(2, int(period))

    def fit(self, values):
        if not _STATSMODELS_OK:
            raise RuntimeError("statsmodels is not installed")
        values = np.asarray(values, dtype=float)
        self.values = values
        self._res = _SARIMAX(
            values,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, self.period),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        return self

    def predict(self, steps):
        return np.asarray(self._res.forecast(steps=int(steps)), dtype=float)

    def fitted(self):
        return np.asarray(self._res.fittedvalues, dtype=float)


class AutoETSModel(BaseModel):
    """Automatic ETS via statsmodels: small error/trend/season grid, AIC pick."""

    name = "auto_ets"
    min_points = 8
    max_origins = 4

    @classmethod
    def available(cls):
        return _STATSMODELS_OK

    def __init__(self, period=None):
        self.period = int(period) if period else None

    def fit(self, values):
        if not _STATSMODELS_OK:
            raise RuntimeError("statsmodels is not installed")
        values = np.asarray(values, dtype=float)
        self.values = values
        seasonal = bool(self.period and self.period >= 2 and len(values) >= 2 * self.period)
        s = self.period if seasonal else 2
        best = None
        for error in ("add", "mul"):
            for trend in ("add", None):
                for damped in ((False,) if trend is None else (False, True)):
                    for seas in (("add",) if seasonal else (None,)):
                        try:
                            res = _ETSModel(
                                values, error=error, trend=trend,
                                damped_trend=damped, seasonal=seas,
                                seasonal_periods=s,
                            ).fit(disp=False)
                        except Exception:
                            continue
                        if best is None or res.aic < best[0]:
                            best = (res.aic, res)
        if best is None:
            raise RuntimeError("AutoETS could not fit any configuration")
        self._res = best[1]
        return self

    def predict(self, steps):
        return np.asarray(self._res.forecast(steps=int(steps)), dtype=float)

    def fitted(self):
        return np.asarray(self._res.fittedvalues, dtype=float)


class TBATSModel(BaseModel):
    """TBATS/BATS via the tbats package (multiple seasonality, Box-Cox,
    damped trend). Only runs when a real season period is known."""

    name = "tbats"
    needs_period = True
    needs_real_period = True
    min_points = 14
    max_origins = 2

    @classmethod
    def available(cls):
        return _TBATS_OK

    def __init__(self, period=7):
        self.period = max(2, int(period))

    def fit(self, values):
        if not _TBATS_OK:
            raise RuntimeError("tbats is not installed")
        values = np.asarray(values, dtype=float)
        self.values = values
        _patch_sklearn_check_array()
        estimator = _TBATS(
            seasonal_periods=[self.period],
            use_box_cox=False, use_trend=True, use_damped_trend=True,
            use_arma_errors=False,  # skip the ARMA-error grid: halves fit time
            show_warnings=False,
            n_jobs=1,  # single-process: child workers would miss the compat shim
        )
        self._fit = estimator.fit(values)
        return self

    def predict(self, steps):
        return np.asarray(self._fit.forecast(steps=int(steps)), dtype=float)

    def fitted(self):
        return np.asarray(self._fit.y_hat, dtype=float)


class GARCHModel(BaseModel):
    """GARCH(1,1) on demeaned data via the arch package. The point forecast
    is the conditional mean (constant); the model competes fairly and mostly
    matters for heteroskedastic series."""

    name = "garch"
    min_points = 20
    max_origins = 4

    @classmethod
    def available(cls):
        return _ARCH_OK

    def fit(self, values):
        if not _ARCH_OK:
            raise RuntimeError("arch is not installed")
        values = np.asarray(values, dtype=float)
        self.values = values
        self.mean = float(values.mean()) if len(values) else 0.0
        am = _arch_model(values - self.mean, mean="Zero", vol="Garch", p=1, q=1)
        self._res = am.fit(disp="off")
        return self

    def predict(self, steps):
        f = self._res.forecast(horizon=int(steps), reindex=False)
        return np.asarray(f.mean.iloc[-1].values, dtype=float) + self.mean

    def fitted(self):
        try:
            resid = np.asarray(self._res.resid, dtype=float)
            out = self.values - resid
            if len(out) == len(self.values):
                return out
        except Exception:
            pass
        return np.full(len(self.values), self.mean)


class _TorchRecurrent(BaseModel):
    """Shared tiny recurrent forecaster (LSTM/GRU). Seeded, CPU, deterministic."""

    kind = "lstm"
    min_points = 24
    max_origins = 3

    @classmethod
    def available(cls):
        return _TORCH_OK

    def __init__(self, hidden=16, lookback=6, epochs=20, lr=0.01):
        self.hidden = int(hidden)
        self.lookback = int(lookback)
        self.epochs = int(epochs)
        self.lr = float(lr)

    def _windows(self, values):
        w = min(self.lookback, max(1, len(values) - 1))
        X, y = [], []
        for i in range(w, len(values)):
            X.append(values[i - w:i])
            y.append(values[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), w

    def _build_net(self):
        cell = _nn.LSTM(1, self.hidden, batch_first=True) if self.kind == "lstm" \
            else _nn.GRU(1, self.hidden, batch_first=True)
        return _nn.Sequential(cell, _LambdaLast(), _nn.Linear(self.hidden, 1))

    def fit(self, values):
        if not _TORCH_OK:
            raise RuntimeError("torch is not installed")
        values = np.asarray(values, dtype=float)
        self.values = values
        mu, sigma = float(values.mean()), float(values.std()) or 1.0
        self.mu, self.sigma = mu, sigma
        z = (values - mu) / sigma
        X, y, self._w = self._windows(z)
        _torch.manual_seed(0)
        try:
            _torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        _torch.set_num_threads(1)
        net = self._build_net()
        opt = _torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = _nn.MSELoss()
        Xt = _torch.from_numpy(X).unsqueeze(-1)
        yt = _torch.from_numpy(y).unsqueeze(-1)
        net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(net(Xt), yt)
            loss.backward()
            opt.step()
        self._net = net
        return self

    def _predict_z(self, window):
        self._net.eval()
        with _torch.no_grad():
            t = _torch.from_numpy(np.asarray(window, dtype=np.float32)).unsqueeze(0).unsqueeze(-1)
            return float(self._net(t).item())

    def predict(self, steps):
        z = list((self.values - self.mu) / self.sigma)
        out = []
        for _ in range(int(steps)):
            w = z[-self._w:] if len(z) >= self._w else ([z[0]] * (self._w - len(z))) + z
            nxt = self._predict_z(w)
            out.append(nxt)
            z.append(nxt)
        return np.asarray(out) * self.sigma + self.mu

    def fitted(self):
        z = (self.values - self.mu) / self.sigma
        out = np.empty(len(self.values))
        for i in range(len(self.values)):
            if i < self._w:
                out[i] = z[i]
            else:
                out[i] = self._predict_z(z[i - self._w:i])
        return out * self.sigma + self.mu


class _LambdaLast(_nn.Module if _TORCH_OK else object):
    """Take the last time step from an RNN output sequence."""

    def forward(self, x):
        out, _ = (x, None) if not isinstance(x, tuple) else x
        return out[:, -1, :]


class LSTMModel(_TorchRecurrent):
    name = "lstm"
    kind = "lstm"


class GRUModel(_TorchRecurrent):
    name = "gru"
    kind = "gru"


# Candidate models for model selection, roughly simplest -> most complex.
CANDIDATES = [
    NaiveModel,
    MeanModel,
    MedianModel,
    DriftModel,
    MovingAverageModel,
    LinearTrendModel,
    ExponentialSmoothingModel,
    HoltLinearModel,
    ThetaModel,
    SeasonalNaiveModel,
    HoltWintersModel,
    AutoETSModel,
    AutoARIMAModel,
    SARIMAModel,
    TBATSModel,
    GARCHModel,
    LSTMModel,
    GRUModel,
]