"""Hypothesis Engine core: intent, adequacy, selection, statistics, verdict.

Analysis types: correlation | group_comparison | segment | trend |
before_after | regression. Every branch computes real statistics with
honest gating (sample minimums, variance checks, null handling) and returns
a structured result the frontend renders directly — no raw SQL in the UI.
"""

import logging
import math
import re

import numpy as np
import pandas as pd

log = logging.getLogger("aria.hypothesis_engine")

try:
    from scipy import stats as _st
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False

MIN_N = 10
SAMPLE_LIMIT = 5000

_DIRECTION_UP = ("increase", "increases", "higher", "grow", "growth", "rise", "rises",
                 "more", "improve", "improves", "boost", "gain")
_DIRECTION_DOWN = ("decrease", "decreases", "lower", "reduce", "reduces", "reduction",
                   "decline", "declines", "drop", "drops", "less", "fewer", "fall", "falls")
_BEFORE_AFTER = ("after", "before", "since", "change", "changed", "event", "launch",
                 "campaign", "increase in", "decrease in", "price increase", "price cut")
_TREND_WORDS = ("trend", "trending", "over time", "overtime", "growing", "declining",
                "month over month", "year over year", "seasonal", "seasonality")


# --------------------------------------------------------------------------
# Structured error codes (§35/§49). Every failure carries a machine code the
# frontend maps to friendly copy — never a raw traceback in the UI.
# --------------------------------------------------------------------------
CODE_NO_CONNECTION = "NO_CONNECTION"
CODE_INVALID_SOURCE = "INVALID_SOURCE"
CODE_PARSING_FAILED = "PARSING_FAILED"
CODE_COLUMN_NOT_FOUND = "COLUMN_NOT_FOUND"
CODE_AMBIGUOUS_COLUMN = "AMBIGUOUS_COLUMN"
CODE_INCOMPATIBLE_DATA_TYPE = "INCOMPATIBLE_DATA_TYPE"
CODE_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
CODE_INVALID_GROUPING = "INVALID_GROUPING"
CODE_INVALID_DATE_RANGE = "INVALID_DATE_RANGE"
CODE_UNSUPPORTED_ANALYSIS = "UNSUPPORTED_ANALYSIS"
CODE_ANALYSIS_FAILED = "ANALYSIS_FAILED"
CODE_TIMEOUT = "TIMEOUT"
CODE_INTERNAL_ERROR = "INTERNAL_ERROR"


def _fail(verdict, code, detail, **extra):
    out = {"ok": False, "verdict": verdict, "code": code,
           "detail": detail, "recoverable": True}
    out.update(extra)
    return out


# --------------------------------------------------------------------------
# Intent
# --------------------------------------------------------------------------
def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def _col_tokens(name):
    return set(_norm(name).split())


# Business synonyms so "quantity" finds a qty column, "revenue" finds sales, etc.
_SYNONYMS = {
    "qty": ["quantity", "quantities", "units"],
    "quantity": ["qty", "units"],
    "revenue": ["sales", "income", "turnover"],
    "sales": ["revenue", "income"],
    "amount": ["revenue", "sales", "total", "value"],
    "price": ["prices", "cost", "rate"],
    "cost": ["price", "prices", "expense"],
    "count": ["number", "volume"],
    "profit": ["margin", "earnings"],
}


def _mentions(question, col):
    """True when the question names the column (full, humanized, plural, synonym)."""
    q = " " + _norm(question) + " "
    toks = set(_col_tokens(col))
    for t in list(toks):
        toks.add(t + "s")
        toks.add(t + "es")
        if t.endswith("s") and len(t) > 3:
            toks.add(t[:-1])
        for s in _SYNONYMS.get(t, []):
            toks.add(s)
    return any(len(t) > 2 and (" " + t + " " in q) for t in toks)


def _mention_pos(question, col):
    """First mention position of a column in the question (large = absent/late)."""
    q = " " + _norm(question) + " "
    toks = set(_col_tokens(col))
    for t in list(toks):
        toks.add(t + "s")
        toks.add(t + "es")
        if t.endswith("s") and len(t) > 3:
            toks.add(t[:-1])
        for s in _SYNONYMS.get(t, []):
            toks.add(s)
    best = None
    for t in toks:
        if len(t) > 2:
            p = q.find(" " + t + " ")
            if p >= 0 and (best is None or p < best):
                best = p
    return best if best is not None else 10 ** 9


def parse_intent(question, intel, tables):
    """Extract metric / driver / dimension / timeframe from the question."""
    intent = {"metric": None, "metric_table": None, "driver": None,
              "driver_table": None, "dimension": None, "dim_table": None,
              "time_col": None, "time_table": None, "direction": None,
              "weekend": False, "before_after": False, "trend": False,
              "metric_explicit": False, "driver_explicit": False,
              "dim_explicit": False, "time_explicit": False}
    ql = question.lower()
    if any(w in ql for w in _DIRECTION_UP):
        intent["direction"] = "up"
    if any(w in ql for w in _DIRECTION_DOWN):
        intent["direction"] = "down" if intent["direction"] is None else "mixed"
    intent["weekend"] = "weekend" in ql
    intent["before_after"] = any(w in ql for w in _BEFORE_AFTER)
    intent["trend"] = any(w in ql for w in _TREND_WORDS)

    cands = {"metric": [], "driver": [], "dim": [], "time": []}
    for t in tables:
        for c in intel.get_numeric_cols(t):
            low = c.lower()
            if low.endswith(("_id", "_key", "_no", "_code")) or low in ("id", "key", "code"):
                continue
            fks = {f[0].lower() for f in intel.get_fks(t)}
            if low in fks:
                continue
            if _mentions(question, c):
                cands["metric"].append((t, c))
        for c in intel.get_text_cols(t):
            if _mentions(question, c):
                cands["dim"].append((t, c))
        for c in intel.get_datetime_cols(t):
            if _mentions(question, c):
                cands["time"].append((t, c))
    # Driver: any other mentioned column. Numerics named alongside the
    # metric come first (the "X" in "Does X affect Y?").
    numcols = set()
    for t in tables:
        for c in intel.get_numeric_cols(t):
            numcols.add((t, c))
    for t in tables:
        for c in intel.get_numeric_cols(t) + intel.get_text_cols(t) + intel.get_datetime_cols(t):
            if _mentions(question, c) and (t, c) not in cands["driver"]:
                cands["driver"].append((t, c))
    cands["driver"].sort(key=lambda tc: 0 if tc in numcols else 1)
    if cands["metric"]:
        # The affected metric is usually named LAST ("Does X affect Y?"),
        # so a mentioned driver is never swallowed as the metric.
        cands["metric"].sort(key=lambda tc: _mention_pos(question, tc[1]), reverse=True)
        intent["metric_table"], intent["metric"] = cands["metric"][0]
        intent["metric_explicit"] = True
    if cands["dim"]:
        intent["dim_table"], intent["dimension"] = cands["dim"][0]
        intent["dim_explicit"] = True
    if cands["time"]:
        intent["time_table"], intent["time_col"] = cands["time"][0]
        intent["time_explicit"] = True
    if cands["driver"]:
        for t, c in cands["driver"]:
            if c != intent["metric"]:
                intent["driver_table"], intent["driver"] = t, c
                intent["driver_explicit"] = True
                break
    # Full candidate lists (§62 ambiguity): the caller decides whether the
    # first pick is confident or the user must choose.
    intent["metric_cands"] = cands["metric"]
    intent["driver_cands"] = [(t, c) for t, c in cands["driver"]]
    intent["dim_cands"] = cands["dim"]
    intent["time_cands"] = cands["time"]
    return intent


def _exact_hit(question, table, col):
    """The question names this column almost verbatim (not via synonym)."""
    q = " " + _norm(question) + " "
    name = _norm(col).replace("_", " ").strip()
    human = name
    return bool(name and (" " + name + " " in q or " " + human + " " in q))


def fill_defaults(intent, intel, tables):
    """Fill gaps from the schema: best measure, first dimension/time."""
    if intent["metric"] is None:
        for t in tables:
            m = intel.find_best_measure(t)
            if m:
                intent["metric"], intent["metric_table"] = m, t
                break
    mt = intent["metric_table"]
    if mt and intent["dimension"] is None:
        dims = intel.find_all_dimensions(mt)
        if dims:
            intent["dimension"], intent["dim_table"] = dims[0], mt
    if mt and intent["time_col"] is None:
        dt = intel.find_best_datetime(mt)
        if dt:
            intent["time_col"], intent["time_table"] = dt, mt
    return intent


# --------------------------------------------------------------------------
# Data access (bounded, dialect-aware)
# --------------------------------------------------------------------------
def _exec(engine, sql, params=None):
    from sqlalchemy import text as _t
    with engine.connect() as conn:
        return conn.execute(_t(sql), params or {}).fetchall()


def _table_count(engine, dialect, table):
    from core.schema_intel import _safe_name
    rows = _exec(engine, f"SELECT COUNT(*) FROM {_safe_name(table, dialect)}")
    return int(rows[0][0] or 0)


def _fetch_pairs(engine, dialect, table, col_a, col_b, limit=SAMPLE_LIMIT):
    from core.schema_intel import _safe_name
    # Deterministic sample: ORDER BY the sampled columns so identical
    # requests analyze identical rows (bare LIMIT is return-order dependent).
    return _exec(
        engine,
        f"SELECT {_safe_name(col_a, dialect)}, {_safe_name(col_b, dialect)} "
        f"FROM {_safe_name(table, dialect)} "
        f"WHERE {_safe_name(col_a, dialect)} IS NOT NULL "
        f"AND {_safe_name(col_b, dialect)} IS NOT NULL "
        f"ORDER BY {_safe_name(col_a, dialect)}, {_safe_name(col_b, dialect)} "
        f"LIMIT {int(limit)}")


def _fetch_single(engine, dialect, table, col, limit=SAMPLE_LIMIT):
    from core.schema_intel import _safe_name
    return [r[0] for r in _exec(
        engine,
        f"SELECT {_safe_name(col, dialect)} FROM {_safe_name(table, dialect)} "
        f"WHERE {_safe_name(col, dialect)} IS NOT NULL "
        f"ORDER BY {_safe_name(col, dialect)} LIMIT {int(limit)}")]


# --------------------------------------------------------------------------
# Statistics (scipy when available, numpy fallbacks)
# --------------------------------------------------------------------------
def _pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 1.0
    r = float(np.corrcoef(x, y)[0, 1])
    if abs(r) >= 1.0:
        return r, 0.0
    if _SCIPY_OK:
        try:
            rr, p = _st.pearsonr(x, y)
            return float(rr), float(p)
        except Exception:
            pass
    n = len(x)
    t = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
    p = 2 * (1 - _t_cdf(abs(t), n - 2))
    return r, p


def _t_cdf(t, df):
    if _SCIPY_OK:
        try:
            return float(_st.t.cdf(t, df))
        except Exception:
            pass
    from core.math_stats import regularized_incomplete_beta as _regularized_incomplete_beta
    x = df / (df + t * t)
    return 1.0 - 0.5 * _regularized_incomplete_beta(df / 2.0, 0.5, x)


def _spearman(x, y):
    if _SCIPY_OK:
        try:
            rr, p = _st.spearmanr(x, y)
            return float(rr), float(p)
        except Exception:
            pass
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return _pearson(rx, ry)


def _welch(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx, my = float(x.mean()), float(y.mean())
    if _SCIPY_OK:
        try:
            _, p = _st.ttest_ind(x, y, equal_var=False)
            p = float(p)
        except Exception:
            p = 1.0
    else:
        p = 1.0
    nx, ny = len(x), len(y)
    vx, vy = float(x.var(ddof=1)) if nx > 1 else 0.0, float(y.var(ddof=1)) if ny > 1 else 0.0
    pooled = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / max(1, nx + ny - 2))
    d = (mx - my) / pooled if pooled > 0 else 0.0
    return mx, my, p, float(d)


def _mann_whitney(x, y):
    if _SCIPY_OK:
        try:
            _, p = _st.mannwhitneyu(x, y, alternative="two-sided")
            return float(p)
        except Exception:
            pass
    return 1.0


def _kruskal(groups):
    if _SCIPY_OK:
        try:
            _, p = _st.kruskal(*groups)
            return float(p)
        except Exception:
            pass
    return 1.0


def _ols(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return (0.0, 0.0, 0.0, 1.0)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # p-value for the slope via t on Pearson(x, y).
    r, p = _pearson(x, y)
    return float(slope), float(intercept), float(r2), p


def _mean_ci(vals):
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0.0
    m = float(vals.mean())
    se = float(vals.std(ddof=1)) / math.sqrt(n) if n > 1 else 0.0
    return m, m - 1.96 * se, m + 1.96 * se


# --------------------------------------------------------------------------
# Analysis runners — each returns (result_dict, limitations)
# --------------------------------------------------------------------------
def run_correlation(engine, dialect, table, driver, metric):
    rows = _fetch_pairs(engine, dialect, table, driver, metric)
    x = np.array([float(r[0]) for r in rows])
    y = np.array([float(r[1]) for r in rows])
    if len(x) < MIN_N:
        return None, [f"Only {len(x)} usable rows for {driver} vs {metric} — at least {MIN_N} are needed."]
    r, p = _pearson(x, y)
    rs, ps = _spearman(x, y)
    slope, intercept, r2, _ = _ols(x, y)
    n = len(x)
    if n <= 3:
        return None, [f"Only {n} usable rows for {driver} vs {metric} — at least {MIN_N} are needed."]
    z = 0.5 * math.log((1 + r) / (1 - r)) if abs(r) < 1 else 0.0
    se = 1.0 / math.sqrt(max(1, n - 3))
    if abs(r) < 1:
        if _SCIPY_OK and n - 2 > 0:
            t_crit = _st.t.ppf(0.975, df=max(1, n - 2))
        else:
            t_crit = 1.96
        ci = (round(math.tanh(z - t_crit * se), 3), round(math.tanh(z + t_crit * se), 3))
    else:
        ci = (r, r)
    lim = []
    if abs(r) < 0.3:
        lim.append("The relationship is weak — the driver explains little of the metric on its own.")
    lim.append("Correlation does not prove causation; hidden factors may drive both.")
    pts = [{"x": round(float(x[i]), 3), "y": round(float(y[i]), 3)}
           for i in np.linspace(0, n - 1, min(n, 200)).astype(int)]
    return {
        "analysis": "correlation",
        "driver": driver, "metric": metric, "table": table, "n": n,
        "r": round(r, 3), "p_value": p, "spearman_r": round(rs, 3),
        "slope": round(slope, 4), "intercept": round(intercept, 3), "r_squared": round(r2, 3),
        "driver_mean": round(float(x.mean()), 3), "metric_mean": round(float(y.mean()), 3),
        "ci": list(ci),
        "strength": _strength(abs(r)),
        "direction": "up" if r > 0 else "down",
        "viz": {"type": "scatter", "x": driver, "y": metric, "points": pts,
                "trend": {"slope": round(slope, 4), "intercept": round(intercept, 3)}},
    }, lim


def run_group_comparison(engine, dialect, table, dim, metric, groups=None):
    rows = _fetch_pairs(engine, dialect, table, dim, metric)
    df = pd.DataFrame(rows, columns=["g", "v"])
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    df = df.dropna()
    df["g"] = df["g"].astype(str)
    if groups is None:
        top = df["g"].value_counts().head(2).index.tolist()
        if len(top) < 2:
            return None, [f"{dim} has fewer than two usable groups."]
        groups = top
    a = df.loc[df["g"] == groups[0], "v"].to_numpy()
    b = df.loc[df["g"] == groups[1], "v"].to_numpy()
    if len(a) < 5 or len(b) < 5:
        return None, [f"Groups are too small ({len(a)} vs {len(b)} rows) — at least 5 each are needed."]
    ma, la, ha = _mean_ci(a)
    mb, lb, hb = _mean_ci(b)
    _, _, p_welch, d = _welch(a, b)
    p_mw = _mann_whitney(a, b)
    p = min(p_welch, p_mw) if _SCIPY_OK else p_welch
    diff = ma - mb
    rel = diff / abs(mb) * 100 if mb else 0.0
    lim = ["Group comparison shows association, not proof of cause."]
    if abs(d) < 0.2:
        lim.append("The effect size is tiny — the gap may not matter in practice.")
    return {
        "analysis": "group_comparison",
        "driver": dim, "metric": metric, "table": table, "n": len(df),
        "groups": [
            {"name": groups[0], "mean": round(ma, 3), "n": len(a), "ci": [round(la, 3), round(ha, 3)]},
            {"name": groups[1], "mean": round(mb, 3), "n": len(b), "ci": [round(lb, 3), round(hb, 3)]},
        ],
        "diff": round(diff, 3), "rel_pct": round(rel, 1),
        "p_value": p, "cohens_d": round(d, 3),
        "strength": _strength(min(1.0, abs(d) / 2)),
        "direction": "up" if diff > 0 else "down",
        "viz": {"type": "bar", "groups": [
            {"label": groups[0], "value": round(ma, 3)},
            {"label": groups[1], "value": round(mb, 3)}]},
    }, lim


def run_segments(engine, dialect, table, dim, metric, top_k=6):
    rows = _fetch_pairs(engine, dialect, table, dim, metric)
    df = pd.DataFrame(rows, columns=["g", "v"])
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    df = df.dropna()
    df["g"] = df["g"].astype(str)
    counts = df["g"].value_counts()
    keep = counts[counts >= 5].head(top_k).index.tolist()
    if len(keep) < 3:
        return None, [f"{dim} has fewer than three usable segments."]
    df = df[df["g"].isin(keep)]
    arrs = [df.loc[df["g"] == g, "v"].to_numpy() for g in keep]
    p = _kruskal(arrs)
    overall = float(df["v"].mean())
    segs = []
    for g, arr in zip(keep, arrs):
        m, lo, hi = _mean_ci(arr)
        segs.append({"name": g, "mean": round(m, 3), "n": len(arr),
                     "ci": [round(lo, 3), round(hi, 3)],
                     "vs_overall_pct": round((m - overall) / abs(overall) * 100, 1) if overall else 0.0})
    segs.sort(key=lambda s: s["mean"], reverse=True)
    lim = ["Segment gaps are observational — other factors may explain them."]
    return {
        "analysis": "segment",
        "driver": dim, "metric": metric, "table": table, "n": len(df),
        "p_value": p, "segments": segs, "overall_mean": round(overall, 3),
        "strongest": segs[0]["name"], "weakest": segs[-1]["name"],
        "strength": _strength(0.6 if p < 0.05 else 0.2),
        "direction": "mixed",
        "viz": {"type": "grouped", "groups": [{"label": s["name"], "value": s["mean"]} for s in segs]},
    }, lim


def run_trend(engine, dialect, table, time_col, metric):
    rows = _fetch_pairs(engine, dialect, table, time_col, metric)
    raw_t = np.array([r[0] for r in rows])
    raw_v = np.array([r[1] for r in rows])
    ts_all = pd.to_datetime(raw_t, errors="coerce")
    order = np.argsort(ts_all.to_numpy())
    ts = pd.to_datetime(raw_t[order], errors="coerce")
    vals = pd.to_numeric(raw_v[order], errors="coerce")
    ok = np.asarray(ts.notna()) & np.asarray(pd.notna(vals))
    df = pd.DataFrame({"t": pd.to_datetime(np.asarray(ts)[ok]),
                       "v": pd.to_numeric(np.asarray(vals)[ok], errors="coerce")}).dropna()
    if len(df) < MIN_N:
        return None, [f"Only {len(df)} dated observations — at least {MIN_N} are needed for a trend."]
    ts, vals = df["t"], df["v"]
    x = np.arange(len(vals), dtype=float)
    slope, _, r2, p = _ols(x, vals.to_numpy())
    per = float(slope)
    rel = per / abs(float(vals.mean())) * 100 if float(vals.mean()) else 0.0
    pts = [{"label": str(ts.iloc[i])[:10], "actual": round(float(vals.iloc[i]), 3)}
           for i in range(len(vals))]
    lim = []
    if r2 < 0.3:
        lim.append("The trend line explains little of the ups and downs — history is noisy.")
    lim.append("Past direction does not guarantee the future.")
    return {
        "analysis": "trend",
        "driver": time_col, "metric": metric, "table": table, "n": len(vals),
        "slope_per_period": round(per, 4), "rel_pct_per_period": round(rel, 2),
        "r_squared": round(r2, 3), "p_value": p,
        "strength": _strength(r2),
        "direction": "up" if per > 0 else "down",
        "viz": {"type": "line", "x": time_col, "y": metric, "points": pts,
                "slope": round(per, 4)},
    }, lim


def run_before_after(engine, dialect, table, time_col, metric):
    rows = _fetch_pairs(engine, dialect, table, time_col, metric)
    ts = pd.to_datetime([r[0] for r in rows], errors="coerce")
    vals = pd.to_numeric([r[1] for r in rows], errors="coerce")
    df = pd.DataFrame({"t": ts, "v": vals}).dropna().sort_values("t")
    if len(df) < MIN_N:
        return None, [f"Only {len(df)} dated observations — at least {MIN_N} are needed."]
    cut = df["t"].quantile(0.5)
    before = df.loc[df["t"] <= cut, "v"].to_numpy()
    after = df.loc[df["t"] > cut, "v"].to_numpy()
    if len(before) < 5 or len(after) < 5:
        return None, ["Too few observations on one side of the midpoint to compare."]
    ma, la, ha = _mean_ci(before)
    mb, lb, hb = _mean_ci(after)
    _, _, p, d = _welch(after, before)
    diff = mb - ma
    rel = diff / abs(ma) * 100 if ma else 0.0
    lim = ["The midpoint split is a neutral default — a real event date would be sharper.",
           "Before/after gaps can reflect seasonality or other changes, not just the event."]
    return {
        "analysis": "before_after",
        "driver": time_col, "metric": metric, "table": table, "n": len(df),
        "cut_label": str(cut)[:10],
        "groups": [
            {"name": "Before", "mean": round(ma, 3), "n": len(before), "ci": [round(la, 3), round(ha, 3)]},
            {"name": "After", "mean": round(mb, 3), "n": len(after), "ci": [round(lb, 3), round(hb, 3)]},
        ],
        "diff": round(diff, 3), "rel_pct": round(rel, 1),
        "p_value": p, "cohens_d": round(d, 3),
        "strength": _strength(min(1.0, abs(d) / 2)),
        "direction": "up" if diff > 0 else "down",
        "viz": {"type": "comparison",
                "groups": [{"label": "Before", "value": round(ma, 3)},
                           {"label": "After", "value": round(mb, 3)}]},
    }, lim


def _strength(x):
    x = abs(float(x))
    return "strong" if x > 0.7 else "moderate" if x > 0.4 else "weak"


# --------------------------------------------------------------------------
# Verdict + orchestration
# --------------------------------------------------------------------------
def _verdict(p, strength, n, direction_expected=None, direction_got=None):
    if n < MIN_N:
        return "NEEDS MORE DATA"
    if p >= 0.05:
        return "NOT SUPPORTED"
    if strength == "weak":
        return "INCONCLUSIVE"
    if direction_expected and direction_got and direction_expected == "mixed":
        pass
    if direction_expected and direction_got and direction_expected != direction_got:
        return "NOT SUPPORTED"
    return "SUPPORTED"


def build_hypothesis_text(question, intent, result=None):
    """Statement from the ACTUAL tested pair (result) when available, so the
    text never names columns that weren't analyzed."""
    if result:
        m = str(result.get("metric") or intent.get("metric") or "the key metric").replace("_", " ")
        d = str(result.get("driver") or intent.get("driver") or "?").replace("_", " ")
        a = result.get("analysis", "")
        if a == "correlation":
            _d = result.get("direction")
            if _d == "up":
                return f"Higher {d} is associated with higher {m}."
            if _d == "down":
                return f"Higher {d} is associated with lower {m}."
            return f"{d} is related to {m}."
        if a in ("group_comparison", "before_after") and isinstance(result.get("groups"), list) and len(result["groups"]) == 2:
            g0, g1 = result["groups"][0]["name"], result["groups"][1]["name"]
            return f"{m} differs between {g0} and {g1}."
        if a == "segment":
            return f"{m} differs across {d}."
        if a == "trend":
            return f"{m} changes over time."
        return f"{d} is related to {m}."
    m = (intent.get("metric") or "the key metric").replace("_", " ")
    d = (intent.get("driver") or intent.get("dimension") or "the main driver")
    d = (d or "").replace("_", " ")
    direction = intent.get("direction")
    if direction == "down":
        return f"Higher {d} is associated with lower {m}."
    if direction == "up":
        return f"Higher {d} is associated with higher {m}."
    if intent.get("dimension"):
        return f"{m} differs across {d}."
    return f"{d} is related to {m}."


def analyze_question(engine, dialect, intel, tables, question, table_hint=None,
                     metric_override=None, driver_override=None):
    """Full pipeline: intent -> adequacy -> selection -> stats -> verdict."""
    intent = parse_intent(question, intel, tables)
    intent = fill_defaults(intent, intel, tables)
    if metric_override:
        for t in tables:
            if metric_override in (intel.get_numeric_cols(t) + intel.get_text_cols(t)):
                intent["metric"], intent["metric_table"] = metric_override, t
                intent["metric_explicit"] = True
                break
    if driver_override:
        for t in tables:
            if driver_override in (intel.get_numeric_cols(t) + intel.get_text_cols(t)
                                   + intel.get_datetime_cols(t)):
                intent["driver"], intent["driver_table"] = driver_override, t
                intent["driver_explicit"] = True
                break
    metric, mtable = intent["metric"], intent["metric_table"]
    if table_hint and table_hint in tables:
        mtable = table_hint
        if intent["metric_table"] != table_hint:
            ms = [c for c in intel.get_numeric_cols(table_hint)]
            if ms:
                intent["metric"], intent["metric_table"] = ms[0], table_hint
                metric, mtable = ms[0], table_hint
    if metric is None or mtable is None:
        return dict(_fail("NOT TESTABLE", CODE_COLUMN_NOT_FOUND,
                          "No measurable number was found in this data. Hypothesis testing needs a numeric metric (not ids, keys or text).",
                          hypothesis=build_hypothesis_text(question, intent), intent=intent))

    total = _table_count(engine, dialect, mtable)
    if total == 0:
        return dict(_fail("NEEDS MORE DATA", CODE_INSUFFICIENT_DATA,
                          f"The {mtable} table is empty — connect data with rows first.",
                          hypothesis=build_hypothesis_text(question, intent), intent=intent))
    # Centralized global gate (core/evidence.py): rows < 10 -> insufficient
    # for ALL substantive conclusions. Additive evidence_state; keys kept.
    if total < MIN_N:
        try:
            from core.evidence import assess_evidence as _assess2
            _ev0 = _assess2(total, "hypothesis", dataset_name=mtable)
        except Exception:
            _ev0 = {"status": "insufficient_evidence",
                    "reason": (f"ARIA needs at least 10 usable records before it can reliably "
                               f"identify business patterns. Only {total} usable records were found."),
                    "row_count": total, "usable_records": total,
                    "min_required": MIN_N, "capability": "hypothesis"}
        return dict(_fail("NEEDS MORE DATA", CODE_INSUFFICIENT_DATA,
                          _ev0.get("reason") or
                          f"The {mtable} table is empty — connect data with rows first.",
                          hypothesis=build_hypothesis_text(question, intent), intent=intent,
                          evidence_state=_ev0,
                          usable_observations=total, min_required=MIN_N))

    driver, dtable = intent["driver"], intent["driver_table"]
    dim, dimtable = intent["dimension"], intent["dim_table"]
    tcol = intent["time_col"] if intent["time_table"] == mtable else None

    result, limitations = None, []
    auto_time = tcol or intel.find_best_datetime(mtable)
    # Selector: explicit time intents first, then an explicitly named numeric
    # driver (correlation), then an explicitly named dimension, and only then
    # auto-filled dimensions — an auto dimension must never hijack the analysis.
    if intent["weekend"] and auto_time:
        result, limitations = _run_weekend(engine, dialect, mtable, auto_time, metric)
    elif intent["before_after"] and auto_time:
        result, limitations = run_before_after(engine, dialect, mtable, auto_time, metric)
    elif intent["trend"] and auto_time:
        result, limitations = run_trend(engine, dialect, mtable, auto_time, metric)
    elif intent["driver_explicit"] and driver and dtable == mtable and driver != metric:
        dcat = intel._col_category.get(mtable, {}).get(driver, "")
        if dcat == "numeric":
            result, limitations = run_correlation(engine, dialect, mtable, driver, metric)
    if result is None and intent["dim_explicit"] and dim and dimtable == mtable:
        n_groups = _distinct_count(engine, dialect, mtable, dim)
        if 2 <= n_groups <= 12:
            if n_groups == 2:
                result, limitations = run_group_comparison(engine, dialect, mtable, dim, metric)
            else:
                result, limitations = run_segments(engine, dialect, mtable, dim, metric)
    if result is None and driver and dtable == mtable and driver != metric:
        dcat = intel._col_category.get(mtable, {}).get(driver, "")
        if dcat == "numeric":
            result, limitations = run_correlation(engine, dialect, mtable, driver, metric)
        elif dcat in ("text",):
            n_groups = _distinct_count(engine, dialect, mtable, driver)
            if 2 <= n_groups <= 12:
                if n_groups == 2:
                    result, limitations = run_group_comparison(engine, dialect, mtable, driver, metric)
                else:
                    result, limitations = run_segments(engine, dialect, mtable, driver, metric)
    if result is None:
        # An explicitly named driver that couldn't run must NEVER fall
        # through to an unrelated pair — that tests the wrong hypothesis.
        if intent.get("driver_explicit") and driver:
            if driver and dtable != mtable:
                return dict(_fail("NOT TESTABLE", CODE_UNSUPPORTED_ANALYSIS,
                                  f"ARIA can't compare {driver.replace('_', ' ')} with {metric.replace('_', ' ')} across different tables yet.",
                                  hypothesis=build_hypothesis_text(question, intent), intent=intent))
            if limitations:
                return dict(_fail("NEEDS MORE DATA", CODE_INSUFFICIENT_DATA,
                                  limitations[0],
                                  hypothesis=build_hypothesis_text(question, intent), intent=intent))
            return dict(_fail("NOT TESTABLE", CODE_UNSUPPORTED_ANALYSIS,
                              f"ARIA couldn't test {driver.replace('_', ' ')} against {metric.replace('_', ' ')} with this data.",
                              hypothesis=build_hypothesis_text(question, intent), intent=intent))
        # Default: correlate the metric with the most relevant numeric column
        # (real measures only — never keys or foreign keys).
        fks = {f[0].lower() for f in intel.get_fks(mtable)}
        pks = {p.lower() for p in intel.get_pk(mtable)}
        best = None
        for c in intel.get_numeric_cols(mtable):
            low = c.lower()
            if c == metric or low in fks or low in pks:
                continue
            if low.endswith(("_id", "_key", "_no", "_code")) or low in ("id", "key", "code"):
                continue
            best = c
            break
        if best:
            result, limitations = run_correlation(engine, dialect, mtable, best, metric)
    if result is None:
        return dict(_fail("NEEDS MORE DATA", CODE_INSUFFICIENT_DATA,
                          limitations[0] if limitations else
                          "There isn't enough usable data to test this idea yet.",
                          hypothesis=build_hypothesis_text(question, intent), intent=intent))

    n = result.get("n", 0)
    strength = result.get("strength", "weak")
    p = result.get("p_value", 1.0)
    verdict = _verdict(p, strength, n, intent.get("direction"), result.get("direction"))
    if verdict == "NEEDS MORE DATA" and limitations:
        return dict(_fail(verdict, CODE_INSUFFICIENT_DATA, limitations[0],
                          hypothesis=build_hypothesis_text(question, intent), intent=intent))

    segments = _top_segments(engine, dialect, intel, mtable, metric)
    sim = _whatif_basis(result)
    try:
        from core.evidence import assess_evidence as _assess3
        _ev_ok = _assess3(n, "hypothesis", dataset_name=mtable)
    except Exception:
        _ev_ok = {"status": "supported" if n >= MIN_N else "insufficient_evidence",
                  "reason": None, "row_count": n, "usable_records": n,
                  "min_required": MIN_N, "capability": "hypothesis"}
    return {
        "ok": True,
        "verdict": verdict,
        "hypothesis": build_hypothesis_text(question, intent, result),
        "observation": f"Question: {question.strip()}",
        "intent": {k: intent[k] for k in
                   ("metric", "metric_table", "driver", "dimension", "time_col", "direction")},
        "result": result,
        "segments": segments,
        "whatif": sim,
        "limitations": limitations,
        "explanation": _explain(result, verdict, metric),
        "evidence_state": _ev_ok,
        "usable_observations": n,
        "min_required": MIN_N,
    }


def _distinct_count(engine, dialect, table, col):
    from core.schema_intel import _safe_name
    rows = _exec(engine,
                 f"SELECT COUNT(DISTINCT {_safe_name(col, dialect)}) "
                 f"FROM {_safe_name(table, dialect)}")
    return int(rows[0][0] or 0)


def _run_weekend(engine, dialect, table, time_col, metric):
    rows = _fetch_pairs(engine, dialect, table, time_col, metric)
    ts = pd.to_datetime([r[0] for r in rows], errors="coerce")
    vals = pd.to_numeric([r[1] for r in rows], errors="coerce")
    df = pd.DataFrame({"t": ts, "v": vals}).dropna()
    if len(df) < MIN_N:
        return None, ["Not enough dated rows to compare weekends with weekdays."]
    df["g"] = np.where(df["t"].dt.dayofweek >= 5, "Weekend", "Weekday")
    if df["g"].nunique() < 2:
        return None, ["The data covers only weekdays or only weekends — nothing to compare."]
    a = df.loc[df["g"] == "Weekend", "v"].to_numpy()
    b = df.loc[df["g"] == "Weekday", "v"].to_numpy()
    if len(a) < 5 or len(b) < 5:
        return None, ["Too few weekend or weekday rows to compare reliably."]
    ma, la, ha = _mean_ci(a)
    mb, lb, hb = _mean_ci(b)
    _, _, p, d = _welch(a, b)
    diff = ma - mb
    rel = diff / abs(mb) * 100 if mb else 0.0
    return {
        "analysis": "group_comparison",
        "driver": "day of week", "metric": metric, "table": table, "n": len(df),
        "groups": [
            {"name": "Weekend", "mean": round(ma, 3), "n": len(a), "ci": [round(la, 3), round(ha, 3)]},
            {"name": "Weekday", "mean": round(mb, 3), "n": len(b), "ci": [round(lb, 3), round(hb, 3)]},
        ],
        "diff": round(diff, 3), "rel_pct": round(rel, 1),
        "p_value": p, "cohens_d": round(d, 3),
        "strength": _strength(min(1.0, abs(d) / 2)),
        "direction": "up" if diff > 0 else "down",
        "viz": {"type": "bar", "groups": [{"label": "Weekend", "value": round(ma, 3)},
                                          {"label": "Weekday", "value": round(mb, 3)}]},
    }, ["Weekend means Saturday–Sunday in your dates; holidays are not separated."]


def _top_segments(engine, dialect, intel, table, metric, max_dims=3):
    """Per-group means for the most useful categorical columns."""
    out = []
    for dim in intel.find_all_dimensions(table)[:max_dims]:
        try:
            if not (2 <= _distinct_count(engine, dialect, table, dim) <= 12):
                continue
            rows = _fetch_pairs(engine, dialect, table, dim, metric)
            df = pd.DataFrame(rows, columns=["g", "v"])
            df["v"] = pd.to_numeric(df["v"], errors="coerce")
            df = df.dropna()
            df["g"] = df["g"].astype(str)
            g = df.groupby("g")["v"].agg(["mean", "count"])
            g = g[g["count"] >= 3].sort_values("mean", ascending=False).head(6)
            if len(g) < 2:
                continue
            out.append({"dimension": dim,
                        "groups": [{"label": str(i), "value": round(float(r["mean"]), 3),
                                    "n": int(r["count"])} for i, r in g.iterrows()]})
        except Exception:
            continue
    return out


def _whatif_basis(result):
    """Simulation inputs when the analysis supports them (labeled SIMULATION)."""
    if result.get("analysis") == "correlation" and result.get("slope") is not None:
        return {"kind": "driver_pct", "driver": result["driver"],
                "slope": result["slope"], "label": "SIMULATION"}
    if result.get("analysis") in ("group_comparison", "before_after"):
        gs = result.get("groups", [])
        if len(gs) == 2:
            return {"kind": "group_shift", "groups": [gs[0]["name"], gs[1]["name"]],
                    "means": [gs[0]["mean"], gs[1]["mean"]], "label": "SIMULATION"}
    return None


def _explain(result, verdict, metric):
    m = metric.replace("_", " ")
    a = result.get("analysis", "")
    n = result.get("n", 0)
    if verdict == "SUPPORTED":
        base = f"The data backs this idea across {n:,} records. "
    elif verdict == "NOT SUPPORTED":
        base = f"The data does not back this idea across {n:,} records. "
    elif verdict == "INCONCLUSIVE":
        base = f"The evidence is mixed across {n:,} records — real but too weak to act on. "
    else:
        base = "There isn't enough evidence yet. "
    if a == "correlation":
        d = result.get("driver", "").replace("_", " ")
        which = "rise together" if result.get("direction") == "up" else "move in opposite directions"
        return (base + f"{d} and {m} {which} ({result.get('strength')} link). "
                       f"Knowing one gives a useful read on the other, but it doesn't prove cause.")
    if a in ("group_comparison", "before_after"):
        gs = result.get("groups", [])
        if len(gs) == 2:
            return (base + f"{gs[1]['name']} averages {gs[1]['mean']:,} vs {gs[0]['name']} at "
                           f"{gs[0]['mean']:,} — a {result.get('rel_pct', 0):+.1f}% gap.")
    if a == "segment":
        return (base + f"The effect is strongest in {result.get('strongest')} and "
                       f"weakest in {result.get('weakest')}.")
    if a == "trend":
        return (base + f"{m} moves {result.get('rel_pct_per_period', 0):+.2f}% per period on average.")
    return base + "See the numbers alongside for the full picture."


def _usable_measures(intel, table):
    """Numeric, non-key, non-FK columns — the only valid hypothesis metrics."""
    fks = {f[0].lower() for f in intel.get_fks(table)}
    pks = {p.lower() for p in intel.get_pk(table)}
    out = []
    for c in intel.get_numeric_cols(table):
        low = c.lower()
        if low in fks or low in pks:
            continue
        if low.endswith(("_id", "_key", "_no", "_code")) or low in ("id", "key", "code"):
            continue
        out.append(c)
    return out


def rank_prominent(engine, dialect, intel, tables, limit=5):
    """Find the most prominent testable patterns in the actual data.

    Scores real effects — strongest correlations, biggest group gaps,
    clearest trends — and returns them ranked with evidence, so every
    suggestion is guaranteed testable (pre-validated, never an error).
    """
    cands = []
    reserve = []
    for t in tables[:8]:
        measures = _usable_measures(intel, t)[:4]
        if not measures:
            continue
        dims = intel.find_all_dimensions(t)[:3]
        dts = intel.get_datetime_cols(t)[:2]
        if not dts:
            for c in intel.get_text_cols(t):
                if any(h in c.lower() for h in ("date", "time", "day", "month", "year", "period")):
                    dts.append(c)
                    break
        # 1) Correlations between measure pairs (pre-validated).
        for i in range(len(measures)):
            for j in range(i + 1, min(i + 3, len(measures))):
                try:
                    rows = _fetch_pairs(engine, dialect, t, measures[i], measures[j], limit=500)
                    x = np.array([float(r[0]) for r in rows])
                    y = np.array([float(r[1]) for r in rows])
                    if len(x) < MIN_N:
                        continue
                    r, p = _pearson(x, y)
                    a, b = measures[i].replace("_", " "), measures[j].replace("_", " ")
                    entry = {
                        "score": abs(r),
                        "text": f"Are {a} and {b} related?",
                        "kind": "correlation",
                        "evidence": f"Link r={r:+.2f} across {len(x):,} records",
                        "metric": measures[j], "driver": measures[i],
                    }
                    if p < 0.05 and abs(r) >= 0.3:
                        entry["evidence"] = f"Strong link (r={r:+.2f}) across {len(x):,} records"
                        cands.append(entry)
                    else:
                        reserve.append(entry)
                except Exception:
                    continue
        # 2) Biggest group gaps.
        m0 = measures[0]
        for d in dims[:2]:
            try:
                if not (2 <= _distinct_count(engine, dialect, t, d) <= 12):
                    continue
                rows = _fetch_pairs(engine, dialect, t, d, m0, limit=1000)
                df = pd.DataFrame(rows, columns=["g", "v"])
                df["v"] = pd.to_numeric(df["v"], errors="coerce")
                df = df.dropna()
                df["g"] = df["g"].astype(str)
                g = df.groupby("g")["v"].agg(["mean", "count"])
                g = g[g["count"] >= 5]
                if len(g) < 2:
                    continue
                top = g.sort_values("mean", ascending=False)
                gap = (top["mean"].iloc[0] - top["mean"].iloc[-1]) / abs(top["mean"].mean()) \
                    if top["mean"].mean() else 0.0
                entry = {
                    "score": min(1.0, abs(gap)),
                    "text": f"Does {m0.replace('_', ' ')} differ across {d.replace('_', ' ')}?",
                    "kind": "groups",
                    "evidence": (f"{top.index[0]} averages {top['mean'].iloc[0]:,.1f} vs "
                                 f"{top.index[-1]} at {top['mean'].iloc[-1]:,.1f}"),
                    "metric": m0, "dimension": d,
                }
                if abs(gap) >= 0.2:
                    cands.append(entry)
                else:
                    reserve.append(entry)
            except Exception:
                continue
        # 3) Clearest trend over time.
        if dts:
            try:
                rows = _fetch_pairs(engine, dialect, t, dts[0], m0, limit=1000)
                ts = pd.to_datetime([r[0] for r in rows], errors="coerce")
                vals = pd.to_numeric([r[1] for r in rows], errors="coerce")
                ok = ts.notna() & vals.notna()
                if ok.sum() >= MIN_N:
                    x = np.arange(ok.sum(), dtype=float)
                    slope, _, r2, p = _ols(x, vals[ok].to_numpy())
                    direction = "rising" if slope > 0 else "falling"
                    entry = {
                        "score": r2,
                        "text": f"Is {m0.replace('_', ' ')} {direction} over time?",
                        "kind": "trend",
                        "evidence": f"{direction.title()} pattern across {ok.sum():,} dated rows",
                        "metric": m0, "time_col": dts[0],
                    }
                    if p < 0.05 and r2 >= 0.3:
                        entry["evidence"] = f"{direction.title()} trend across {ok.sum():,} dated rows"
                        cands.append(entry)
                    else:
                        reserve.append(entry)
            except Exception:
                continue
        if len(cands) >= limit + 2:
            break
    cands.sort(key=lambda c: c["score"], reverse=True)
    reserve.sort(key=lambda c: c["score"], reverse=True)
    seen = set()
    out = []
    for c in cands + reserve:
        if c["text"] not in seen:
            seen.add(c["text"])
            out.append(c)
        if len(out) >= limit:
            break
    return out


def suggest_hypotheses(engine, dialect, intel, tables, limit=4):
    """Backward-compatible text suggestions (schema-only, no data scan)."""
    ranked = []
    try:
        ranked = rank_prominent(engine, dialect, intel, tables, limit=limit)
    except Exception:
        ranked = []
    if ranked:
        return [c["text"] for c in ranked]
    out = []
    for t in tables[:10]:
        measures = _usable_measures(intel, t)[:3]
        dims = intel.find_all_dimensions(t)[:3]
        if not measures:
            continue
        m = measures[0]
        mh = m.replace("_", " ")
        if len(measures) > 1:
            d = measures[1].replace("_", " ")
            out.append(f"Are {d} and {mh} related?")
        if dims:
            out.append(f"Does {mh} differ across {dims[0].replace('_', ' ')}?")
        if len(out) >= limit:
            break
    return out[:limit]


# --------------------------------------------------------------------------
# Preview (§33) + ambiguity (§62) + data context (§47)
# --------------------------------------------------------------------------
def _usable_count(engine, dialect, table, cols):
    """Rows where ALL listed columns are non-null (bounded, read-only)."""
    from core.schema_intel import _safe_name
    conds = " AND ".join(f"{_safe_name(c, dialect)} IS NOT NULL" for c in cols)
    try:
        rows = _exec(engine,
                     f"SELECT COUNT(*) FROM {_safe_name(table, dialect)} WHERE {conds}")
        return int(rows[0][0] or 0)
    except Exception:
        return 0


def _resolve_named(question, cands, tables, intel, kind):
    """Pick one column from candidates or report ambiguity.

    Returns (table, col, options). options is non-empty when the user must
    choose (§62) — exact verbatim mentions win, otherwise ambiguity.
    """
    if not cands:
        return None, None, []
    if len(cands) == 1:
        return cands[0][0], cands[0][1], []
    for t, c in cands:
        if _exact_hit(question, t, c):
            return t, c, []
    seen = []
    for t, c in cands:
        if (t, c) not in seen:
            seen.append((t, c))
    return None, None, seen[:6]


def preview_question(engine, dialect, intel, tables, question,
                     metric_hint=None, driver_hint=None):
    """Parse without running statistics: preview or clarification request.

    Returns either {"ok": True, "preview": {...}} or {"ok": False,
    "code": "AMBIGUOUS_COLUMN"/"COLUMN_NOT_FOUND", "options": {...}}.
    Never raises for analytical reasons — those are data, not crashes.
    """
    q = (question or "").strip()
    if not q:
        return _fail("NOT TESTABLE", CODE_PARSING_FAILED,
                     "Type a question first — for example, does price affect sales?")
    intent = parse_intent(q, intel, tables)

    # Explicit user picks (ambiguity resolution) outrank guessing.
    if metric_hint:
        mt = intent.get("metric_table")
        if mt and metric_hint in (intel.get_numeric_cols(mt) + intel.get_text_cols(mt)):
            intent["metric"], intent["metric_table"] = metric_hint, mt
            intent["metric_explicit"] = True
    if driver_hint:
        dt = intent.get("driver_table") or intent.get("metric_table")
        if dt and driver_hint in (intel.get_numeric_cols(dt) + intel.get_text_cols(dt)
                                  + intel.get_datetime_cols(dt)):
            intent["driver"], intent["driver_table"] = driver_hint, dt
            intent["driver_explicit"] = True

    if intent.get("metric") is None and not metric_hint:
        mt, mc, opts = _resolve_named(q, intent.get("metric_cands", []), tables, intel, "metric")
        if opts:
            return dict(_fail("NOT TESTABLE", CODE_AMBIGUOUS_COLUMN,
                              "ARIA needs a little clarification before testing.",
                              hypothesis=build_hypothesis_text(q, intent)),
                        options={"metrics": [{"table": t, "column": c} for t, c in opts],
                                 "drivers": []})
        if mt:
            intent["metric"], intent["metric_table"] = mt, mc
    if intent.get("metric") is None:
        avail = []
        for t in tables[:8]:
            for c in _usable_measures(intel, t)[:4]:
                avail.append({"table": t, "column": c})
        return dict(_fail("NOT TESTABLE", CODE_COLUMN_NOT_FOUND,
                          "ARIA couldn't identify a measurable metric for this question.",
                          hypothesis=build_hypothesis_text(q, intent)),
                    options={"metrics": avail[:8], "drivers": []})

    metric, mtable = intent["metric"], intent["metric_table"]
    # Driver: resolve ambiguity the same way, else leave for auto selection.
    driver, dtable = intent.get("driver"), intent.get("driver_table")
    if driver is None and not driver_hint:
        dcands = [(t, c) for t, c in intent.get("driver_cands", [])
                  if c != metric][:6]
        if len(dcands) > 1 and not any(_exact_hit(q, t, c) for t, c in dcands):
            mopts = [{"table": mtable, "column": metric}]
            return dict(_fail("NOT TESTABLE", CODE_AMBIGUOUS_COLUMN,
                              "Which driver should ARIA test against "
                              + metric.replace("_", " ") + "?",
                              hypothesis=build_hypothesis_text(q, intent)),
                        options={"metrics": mopts,
                                 "drivers": [{"table": t, "column": c} for t, c in dcands]})
    if intent.get("dimension") is None:
        intent = fill_defaults(intent, intel, tables)
        metric, mtable = intent["metric"], intent["metric_table"]

    # Predict the analysis kind with the same selector order as analyze_question.
    auto_time = intent.get("time_col") if intent.get("time_table") == mtable \
        else intel.find_best_datetime(mtable)
    if intent.get("weekend") and auto_time:
        kind = "group_comparison"
    elif intent.get("before_after") and auto_time:
        kind = "before_after"
    elif intent.get("trend") and auto_time:
        kind = "trend"
    elif driver and (intent.get("driver_table") == mtable):
        dcat = intel._col_category.get(mtable, {}).get(driver, "")
        kind = "correlation" if dcat == "numeric" else "group_comparison"
    elif intent.get("dimension"):
        kind = "group_comparison"
    else:
        kind = "correlation"

    cols = [metric] + ([driver] if driver else [])
    usable = _usable_count(engine, dialect, mtable, cols)
    try:
        total = _table_count(engine, dialect, mtable)
    except Exception:
        total = usable
    # Preview statement from the RESOLVED triple (never the auto dimension),
    # so it names only columns that will actually be analyzed.
    _pv_dir = intent.get("direction")
    _pv_result = {"metric": metric, "driver": driver,
                  "analysis": ("correlation" if kind == "correlation"
                               else "group_comparison" if kind in ("group_comparison", "before_after")
                               else kind),
                  "direction": ("up" if _pv_dir == "up" else
                                "down" if _pv_dir == "down" else None)}
    try:
        from core.evidence import assess_evidence as _assess
        _ev_state = _assess(usable, "hypothesis", dataset_name=mtable)
    except Exception:
        _ev_state = {"status": ("insufficient_evidence" if usable < MIN_N else "supported"),
                     "reason": None, "row_count": usable, "usable_records": usable,
                     "min_required": MIN_N, "capability": "hypothesis"}
    return {"ok": True, "preview": {
        "statement": build_hypothesis_text(q, intent, _pv_result),
        "metric": metric, "metric_table": mtable,
        "driver": driver, "driver_table": dtable,
        "dimension": intent.get("dimension"),
        "time_col": auto_time, "analysis": kind,
        "rows_available": total, "usable_observations": usable,
        "needs_more_data": usable < MIN_N, "min_required": MIN_N,
        "evidence_state": _ev_state,
    }}


def data_context(engine, dialect, intel, tables):
    """Dynamic hypothesis context (§47): metrics, drivers, dims, dates, ranges."""
    ctx = {"metrics": [], "drivers": [], "dimensions": [],
           "date_columns": [], "tables": [], "time_range": {}}
    for t in tables[:12]:
        try:
            n = _table_count(engine, dialect, t)
        except Exception:
            continue
        ctx["tables"].append({"table": t, "rows": n})
        for c in _usable_measures(intel, t)[:8]:
            ctx["metrics"].append({"table": t, "column": c})
            ctx["drivers"].append({"table": t, "column": c, "kind": "numeric"})
        for c in intel.find_all_dimensions(t)[:8]:
            ctx["dimensions"].append({"table": t, "column": c})
            ctx["drivers"].append({"table": t, "column": c, "kind": "categorical"})
        dts = intel.get_datetime_cols(t)[:2] or \
            ([intel.find_best_datetime(t)] if intel.find_best_datetime(t) else [])
        for c in dts:
            if c and not any(d["column"] == c and d["table"] == t for d in ctx["date_columns"]):
                ctx["date_columns"].append({"table": t, "column": c})
        if dts and dts[0]:
            try:
                from core.schema_intel import _safe_name
                rows = _exec(engine,
                             f"SELECT MIN({_safe_name(dts[0], dialect)}), "
                             f"MAX({_safe_name(dts[0], dialect)}) "
                             f"FROM {_safe_name(t, dialect)}")
                if rows and rows[0][0] is not None:
                    ctx["time_range"][t] = {"min": str(rows[0][0]), "max": str(rows[0][1])}
            except Exception:
                pass
    return ctx
