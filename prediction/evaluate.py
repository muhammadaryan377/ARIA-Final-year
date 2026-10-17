"""Walk-forward evaluation, model selection and ensemble voting for ARIA.

Why this is honest (no overfitting):
- We never judge a model on the data it was fitted to. Each candidate is scored
  with an expanding-window walk-forward loop: fit on the first k points, predict
  point k+1, repeat. The reported accuracy is therefore out-of-sample.
- Overfit guard: a model whose out-of-sample error is much worse than its
  in-sample error (ratio > 1.5) has memorized the training window; it is dropped
  in favour of simpler models.
- Beat-naive guard: any candidate that performs worse than the naive baseline is
  excluded from the ensemble (it only adds noise). The reported ensemble error is
  itself measured by the same out-of-sample walk-forward loop, so an ensemble is
  only used when it genuinely beats the best single model.
- Accuracy = max(0, 100 - MAPE%). It is reported honestly per dataset.
"""

import logging
import numpy as np

from . import models as M


def mape(actual, predicted):
    """Mean Absolute Percentage Error (%). None if no valid (non-zero) actuals."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.abs(actual) > 1e-9
    if mask.sum() == 0:
        return None
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100.0)


def rmse(actual, predicted):
    """Root Mean Squared Error. None when there is nothing to compare."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) == 0 or len(predicted) == 0:
        return None
    n = min(len(actual), len(predicted))
    return float(np.sqrt(np.mean((actual[:n] - predicted[:n]) ** 2)))


def mae(actual, predicted):
    """Mean Absolute Error. None when there is nothing to compare."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) == 0 or len(predicted) == 0:
        return None
    n = min(len(actual), len(predicted))
    return float(np.mean(np.abs(actual[:n] - predicted[:n])))


def smape(actual, predicted):
    """Symmetric MAPE (%). None when nothing scoreable (all zero pairs)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) == 0 or len(predicted) == 0:
        return None
    n = min(len(actual), len(predicted))
    a, p = actual[:n], predicted[:n]
    denom = (np.abs(a) + np.abs(p)) / 2.0
    mask = denom > 1e-9
    if mask.sum() == 0:
        return None
    return float(np.mean(np.abs(a[mask] - p[mask]) / denom[mask]) * 100.0)


def wape(actual, predicted):
    """Weighted APE (%): sum|a-p| / sum|a|. None when no volume."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) == 0 or len(predicted) == 0:
        return None
    n = min(len(actual), len(predicted))
    a, p = actual[:n], predicted[:n]
    denom = float(np.sum(np.abs(a)))
    if denom < 1e-9:
        return None
    return float(np.sum(np.abs(a - p)) / denom * 100.0)


def bias(actual, predicted):
    """Mean signed error (predicted - actual). Positive = over-forecast."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) == 0 or len(predicted) == 0:
        return None
    n = min(len(actual), len(predicted))
    return float(np.mean(predicted[:n] - actual[:n]))


def holdout_backtest(model_cls, values, season_period=None, holdout=None):
    """Trailing-holdout backtest for ONE candidate.

    Trains on values[:-h], forecasts h steps, scores against values[-h:].
    Returns {"mae","rmse","mape","holdout_points"} with None metrics when the
    holdout cannot be scored (too little history, fit failure, all-zero days).
    Default holdout h = min(3, max(1, n // 4)) — small enough to keep training
    data, large enough to separate the candidates.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if holdout is None:
        holdout = min(3, max(1, n // 4))
    holdout = max(1, min(int(holdout), n - 1)) if n > 1 else 0
    if holdout < 1 or n < 5:
        return {"mae": None, "rmse": None, "mape": None, "holdout_points": 0}
    if not _eligible_cls(model_cls, n - holdout, season_period):
        # Not enough training history for this candidate's minimum: it cannot
        # compete on this holdout (keeps heavy models out of tiny series).
        train_ok = (n - holdout) >= max(3, getattr(model_cls, "min_points", 0) - holdout)
        if not train_ok:
            return {"mae": None, "rmse": None, "mape": None, "holdout_points": 0}
    try:
        pred = np.asarray(
            _make(model_cls, season_period).fit(values[:-holdout]).predict(holdout),
            dtype=float,
        )
    except Exception:
        return {"mae": None, "rmse": None, "mape": None, "holdout_points": holdout}
    actual = values[-holdout:]
    mask = np.abs(actual) > 1e-9
    return {
        "mae": mae(actual, pred),
        "rmse": rmse(actual, pred),
        "mape": mape(actual[mask], pred[mask]) if mask.any() else None,
        "holdout_points": int(holdout),
    }


def backtest_competing(candidates, values, season_period=None, holdout=None):
    """Score competing candidates on the same trailing holdout.

    Returns a ranked list of {"name","mae","rmse","mape","holdout_points"},
    best first (lowest holdout MAPE; MAE breaks ties / covers all-zero
    holdouts; naive-MAPE fallback keeps unscoreable candidates last).
    Candidates that cannot be scored at all are appended last, unranked.
    """
    values = np.asarray(values, dtype=float)
    scored, unscored = [], []
    for cls in candidates:
        try:
            res = holdout_backtest(cls, values, season_period, holdout)
        except Exception:
            continue
        entry = {"name": cls.name, "mae": res["mae"], "rmse": res["rmse"],
                 "mape": res["mape"], "holdout_points": res["holdout_points"]}
        if res["mae"] is None and res["mape"] is None:
            unscored.append(entry)
        else:
            scored.append(entry)

    def _key(r):
        mape_k = r["mape"] if r["mape"] is not None else float("inf")
        mae_k = r["mae"] if r["mae"] is not None else float("inf")
        return (mape_k, mae_k, r["name"])

    return sorted(scored, key=_key) + unscored


def _make(cls, season_period):
    """Instantiate a candidate, passing a season period to models that need one.

    Models with an *optional* season (e.g. auto_ets, which fits seasonal ETS
    when ``period`` is given) receive the real validated period too: a
    seasonal winner must never be validated as seasonal and then refit
    non-seasonal, which collapses to a flat level line on seasonal history.
    Classes that do not accept ``period`` fall back to a bare constructor.
    """
    if getattr(cls, "needs_period", False):
        return cls(period=season_period or 7)
    if season_period:
        try:
            return cls(period=season_period)
        except TypeError:
            pass
    try:
        return cls()
    except TypeError:
        return cls(period=season_period or 7)


def _eligible_cls(cls, n, season_period):
    """Whether a candidate may compete on this series at all."""
    available = getattr(cls, "available", None)
    if callable(available) and not available():
        return False
    if n < getattr(cls, "min_points", 0):
        return False
    if getattr(cls, "needs_real_period", False) and not season_period:
        return False
    return True


def _origins(n, model_cls):
    """Walk-forward origin indices, capped for expensive models."""
    all_k = list(range(max(3, n // 2), n))
    cap = getattr(model_cls, "max_origins", None)
    if cap and len(all_k) > cap:
        idx = np.linspace(0, len(all_k) - 1, cap).round().astype(int)
        return [all_k[i] for i in idx]
    return all_k


def walk_forward_errors(model_cls, values, season_period=None):
    """Expanding-window walk-forward one-step errors.

    Returns a list of (actual, abs_error) tuples for every point that could be
    predicted out-of-sample. This is the ground truth for both MAPE (used for
    accuracy) and MAE (used for honest confidence bands).
    NOTE: zero actuals are skipped (MAPE is undefined at 0) — see
    walk_forward_coverage() for the excluded fraction, reported alongside
    accuracy so sparse series cannot look precise.
    """
    n = len(values)
    if n < 5:
        return []
    if not _eligible_cls(model_cls, n, season_period):
        return []
    errors = []
    for k in _origins(n, model_cls):
        train = values[:k]
        actual = float(values[k])
        if abs(actual) < 1e-9:
            continue
        try:
            pred = _make(model_cls, season_period).fit(train).predict(1)[0]
        except Exception:
            continue
        errors.append((actual, abs(actual - pred)))
    return errors


def walk_forward_coverage(values, season_period=None):
    """How much of the walk-forward window MAPE actually evaluates.

    Returns {"origins": total, "zero_skipped": zeros, "evaluated": n}.
    A high zero_skipped fraction means the reported accuracy covers only
    the non-zero periods — callers must surface this, not just the MAPE.
    """
    n = len(values)
    if n < 5:
        return {"origins": 0, "zero_skipped": 0, "evaluated": 0}
    all_k = list(range(max(3, n // 2), n))
    zeros = sum(1 for k in all_k if abs(float(values[k])) < 1e-9)
    return {"origins": len(all_k), "zero_skipped": zeros, "evaluated": len(all_k) - zeros}


def one_step_ahead_mape(model_cls, values, season_period=None):
    """Expanding-window walk-forward one-step MAPE for a candidate model."""
    errors = walk_forward_errors(model_cls, values, season_period)
    if not errors:
        return None
    return float(np.mean([e / a * 100.0 for a, e in errors]))


def one_step_ahead_mae(model_cls, values, season_period=None):
    """Expanding-window walk-forward one-step MAE for a candidate model."""
    errors = walk_forward_errors(model_cls, values, season_period)
    if not errors:
        return None
    return float(np.mean([e for _, e in errors]))


def in_sample_mape(model_cls, values, season_period=None):
    """In-sample one-step-ahead MAPE (the fitted() curve). Used only by the
    overfit guard as a reference, never as the reported accuracy."""
    try:
        fitted = np.asarray(_make(model_cls, season_period).fit(values).fitted(), dtype=float)
    except Exception:
        return None
    return mape(values, fitted)


def _score_candidates(values, season_period, min_points):
    """Walk-forward score every candidate. Returns list of result dicts."""
    n = len(values)
    if n < min_points:
        return []
    results = []
    for cls in M.CANDIDATES:
        if not _eligible_cls(cls, n, season_period):
            continue
        oos = one_step_ahead_mape(cls, values, season_period)
        if oos is None:
            continue
        ins = in_sample_mape(cls, values, season_period)
        ratio = (oos / ins) if (ins and ins > 1e-9) else float("inf")
        overfit = ratio > 1.5
        results.append({
            "cls": cls, "oos": oos, "ins": ins, "ratio": ratio, "overfit": overfit,
        })
    return results


def _blend_errors(members, weights, values, season_period=None):
    """Honest out-of-sample errors of a weighted blend via walk-forward."""
    n = len(values)
    errors = []
    # Blend members were already selected as eligible; use the coarsest
    # origin cap among them so ensembles stay tractable.
    caps = [getattr(cls, "max_origins", None) for cls in members]
    caps = [c for c in caps if c]
    all_k = list(range(max(3, n // 2), n))
    if caps and len(all_k) > min(caps):
        idx = np.linspace(0, len(all_k) - 1, min(caps)).round().astype(int)
        all_k = [all_k[i] for i in idx]
    for k in all_k:
        train = values[:k]
        actual = float(values[k])
        if abs(actual) < 1e-9:
            continue
        preds = []
        ok = True
        for cls in members:
            try:
                preds.append(float(_make(cls, season_period).fit(train).predict(1)[0]))
            except Exception:
                ok = False
                break
        if not ok:
            continue
        blend = float(np.dot(weights, preds))
        errors.append((actual, abs(actual - blend)))
    return errors


def select_model(values, min_points=5, season_period=None):
    """Pick the best model for a series: the best single candidate or, when it
    genuinely improves out-of-sample accuracy, a weighted ensemble (voting).

    Returns (model, out_of_sample_mape, out_of_sample_mae, meta) where `model`
    is either a candidate CLASS (instantiate + fit on the full series) or an
    already-fitted EnsembleModel instance. Returns (None, None, None, None) when
    there is too little data to validate.
    """
    n = len(values)
    if n < min_points:
        return None, None, None, None

    results = _score_candidates(values, season_period, min_points)
    if not results:
        return None, None, None, None

    naive_mape = next((r["oos"] for r in results if r["cls"] is M.NaiveModel), None)

    # Non-overfit candidates that are not worse than naive (they earn their place).
    eligible = [
        r for r in results
        if not r["overfit"] and (naive_mape is None or r["oos"] <= naive_mape * 1.01)
    ]
    if not eligible:
        # If no overfit-free candidates beat naive, pick the one with lowest
        # OOS error (even if overfit) rather than re-enabling all overfit models.
        eligible = sorted(results, key=lambda r: r["oos"])
        if eligible and eligible[0]["overfit"]:
            logging.warning("All models overfit; selecting least-overfit model.")
    eligible.sort(key=lambda r: r["oos"])
    best = eligible[0]

    meta = {
        "walk_forward_mape": round(best["oos"], 2),
        "in_sample_mape": round(best["ins"], 2) if best["ins"] is not None else None,
        "overfit_ratio": round(best["ratio"], 2),
        "overfit_rejected": best["overfit"],
        "naive_mape": round(naive_mape, 2) if naive_mape is not None else None,
        "candidates": [r["cls"].name for r in results],
    }
    coverage = walk_forward_coverage(values, season_period)
    meta["walk_forward_origins"] = coverage["origins"]
    meta["walk_forward_evaluated"] = coverage["evaluated"]
    meta["zero_actuals_skipped"] = coverage["zero_skipped"]

    # ---- Ensemble voting ----
    top = eligible[:3]
    weights = None
    if len(top) >= 2:
        inv = [1.0 / max(r["oos"], 1e-9) for r in top]
        wsum = sum(inv)
        weights = [w / wsum for w in inv]
        blend_errors = _blend_errors([r["cls"] for r in top], weights, values, season_period)
        if blend_errors:
            ens_mape = float(np.mean([e / a * 100.0 for a, e in blend_errors]))
            ens_mae = float(np.mean([e for _, e in blend_errors]))
            # Only use the ensemble when it genuinely beats the best single model.
            if ens_mape < best["oos"]:
                members = []
                for r, w in zip(top, weights):
                    try:
                        members.append((_make(r["cls"], season_period).fit(values), w))
                    except Exception:
                        continue
                if members:
                    weights = [w / sum(w for _, w in members) for _, w in members]
                    members = [(m, w) for (m, _), w in zip(members, weights)]
                    model = M.EnsembleModel(members)
                    mae = ens_mae
                    meta.update({
                        "ensemble": {
                            "members": [{"name": r["cls"].name, "weight": round(w, 3)} for r, w in zip(top, weights)],
                            "walk_forward_mape": round(ens_mape, 2),
                            "walk_forward_mae": round(ens_mae, 2),
                        },
                    })
                    return model, ens_mape, mae, meta

    mae = one_step_ahead_mae(best["cls"], values, season_period)
    meta["walk_forward_mae"] = round(mae, 2) if mae is not None else None
    return best["cls"], best["oos"], mae, meta


def accuracy_from_mape(mape_value):
    """Accuracy percentage from a MAPE value (None-safe)."""
    if mape_value is None:
        return None
    return round(max(0.0, 100.0 - mape_value), 1)


def reliability_label(accuracy):
    if accuracy is None:
        return "insufficient_data"
    if accuracy >= 90:
        return "excellent"
    if accuracy >= 80:
        return "good"
    if accuracy >= 70:
        return "fair"
    return "poor"
