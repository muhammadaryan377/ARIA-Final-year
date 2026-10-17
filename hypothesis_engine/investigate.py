"""Evidence-based investigation workspace (t.txt: Hypothesis as investigation).

Pipeline: UNDERSTAND -> TARGET METRIC -> TIME PERIOD -> RETRIEVE ->
BASELINE -> FIND CHANGE -> BREAK DOWN -> RELATED PATTERNS -> ANOMALIES ->
DOCUMENT EVIDENCE -> CANDIDATES -> TEST -> RANK -> EXPLAIN.

Rules: the LLM generates candidate wordings and polishes explanations; DATA
validates them. Deterministic-first: without an LLM every candidate still
carries measured evidence, and insufficient evidence is reported honestly.
Never invents causes, never states hypotheses as facts.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

MIN_OBS = 10

_STRENGTH_ORDER = {"Strong evidence": 0, "Moderate evidence": 1,
                   "Limited evidence": 2, "Insufficient evidence": 3}


def _biz(v) -> str:
    try:
        n = float(v)
    except (TypeError, ValueError):
        return str(v)
    a = abs(n)
    if a >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if a >= 10_000:
        return f"{n / 1_000:.1f}K"
    if a >= 100:
        return f"{round(n):,}"
    if a >= 1:
        return f"{round(n, 1)}"
    return f"{round(n, 2)}"


def _plain(name) -> str:
    return str(name or "").replace("_", " ")


def _strength_for(share: float, n: int) -> str:
    """Evidence-strength label from measured support (not a probability)."""
    if n < MIN_OBS:
        return "Insufficient evidence"
    if share >= 0.7:
        return "Strong evidence"
    if share >= 0.4:
        return "Moderate evidence"
    if share > 0:
        return "Limited evidence"
    return "Insufficient evidence"


def _doc_score(question: str, text: str) -> int:
    stop = {"what", "does", "the", "this", "that", "about", "why", "did",
            "are", "our", "your", "how", "when", "which", "explain", "does",
            "and", "for", "with", "from", "have", "has", "was", "were",
            "is", "it", "its", "by", "do", "did", "any", "a", "an", "of",
            "to", "in", "on", "be", "may", "might", "could", "would"}
    qtok = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in stop}
    if not qtok:
        return 0
    ttok = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(qtok & ttok)


def investigate_question(engine, dialect, intel, tables, question,
                         documents=None, llm=None) -> dict:
    """Run a full evidence-based investigation. Never raises."""
    from .engine import (parse_intent, fill_defaults, _fetch_pairs,
                         run_trend, run_segments, _top_segments)
    try:
        return _investigate(engine, dialect, intel, tables, question,
                            documents or [], llm,
                            parse_intent, fill_defaults, _fetch_pairs,
                            run_trend, run_segments, _top_segments)
    except Exception as exc:
        log.info("Hypothesis investigation failed (%s).", exc)
        return {"ok": False, "code": "ANALYSIS_FAILED",
                "detail": "The investigation could not run. Try a narrower question."}


def _investigate(engine, dialect, intel, tables, question, documents, llm,
                 parse_intent, fill_defaults, _fetch_pairs,
                 run_trend, run_segments, _top_segments) -> dict:
    # 1-3. Understand + target metric + defaults. A data-grounded resolver
    # picks (table, metric) from the question; the legacy intent parser is
    # only a fallback so different questions yield different analyses.
    table, metric, count_mode = _resolve_target(
        engine, dialect, intel, tables, question, parse_intent, fill_defaults)
    if count_mode and table:
        return _investigate_count(engine, dialect, intel, table, question)
    if not table or not metric:
        return {"ok": True, "insufficient": True,
                "observation": None, "hypotheses": [],
                "detail": ("ARIA couldn't find a measurable numeric metric to investigate. "
                           "Connect data with numbers tracked over time for the strongest results."),
                "needed_data": ["a numeric metric (sales, orders, revenue, …)"],
                "suggestions": _validated_suggestions(intel, table or (tables or [None])[0], None, None)}
    if table not in (tables or []):
        table = tables[0]

    # 4-6. Retrieve + baseline + find change (recent vs previous split).
    import pandas as pd
    time_col = _time_col(intel, table)
    obs, series = _observe(engine, dialect, table, metric, time_col, pd, _fetch_pairs)
    if obs is None:
        return {"ok": True, "insufficient": True, "observation": None,
                "hypotheses": [],
                "detail": ("There isn't enough evidence to identify a reliable explanation. "
                           f"Only {series} usable observations were found."),
                "needed_data": ["more historical records for " + _plain(metric)],
                "suggestions": _validated_suggestions(intel, table, metric, time_col)}

    candidates = []
    # Observation chart: trend line when a time field exists.
    if obs.get("time_col"):
        try:
            _trend, _lim = run_trend(engine, dialect, table, obs["time_col"], metric)
            if _trend and (_trend.get("viz") or {}).get("points"):
                obs["chart"] = {"type": "line", "x": obs["time_col"], "y": metric,
                                "points": [{"label": p["label"], "actual": p["actual"]}
                                           for p in _trend["viz"]["points"]]}
        except Exception as exc:
            log.info("Investigation observation chart skipped (%s).", exc)
    # Validated next questions: built only from resolved table/metric/dims,
    # so every suggestion is structurally investigable (no dead ends).
    suggestions = _validated_suggestions(intel, table, metric, obs.get("time_col"))
    # 7. Break down the change across dimensions.
    candidates.extend(_breakdown(engine, dialect, intel, table, metric, obs,
                                 run_segments, _top_segments))
    # 7b. Contribution: whose change explains the total move?
    candidates.extend(_contribution(engine, dialect, intel, table, metric, obs,
                                    _top_segments))
    # 7c. Volume vs value decomposition (orders↓ + AOV→ pattern).
    candidates.extend(_volume_split(engine, dialect, table, metric, obs))    # 8-9. Related patterns + anomalies on the series.
    candidates.extend(_volume_candidate(obs))
    candidates.extend(_anomaly_notes(series))
    # 10. Document evidence.
    doc_cands = _document_candidates(question, metric, documents)
    candidates.extend(doc_cands)

    # 11. Trend fallback: a measured movement with no breakdown is still a
    # valid Moderate/Limited candidate — never return an empty workspace.
    if not candidates and obs.get("direction") in ("up", "down"):
        pct = abs(obs.get("change_pct") or 0)
        candidates.append({
            "title": f"Sustained {obs['direction']}ward movement in {_plain(metric)}",
            "strength": "Moderate evidence" if pct >= 10 else "Limited evidence",
            "explanation": (f"{_plain(metric)} moved {obs['direction']} about "
                            f"{pct:.0f}% between the earlier and recent halves of the "
                            f"observed period. The data confirms the direction but not the cause."),
            "supporting": [f"Earlier average {_biz(obs.get('prev_mean'))} → recent average {_biz(obs.get('recent_mean'))}",
                           f"{obs.get('n', 0):,} observations"],
            "metrics": {"change_pct": obs.get("change_pct"), "n": obs.get("n", 0)},
            "contradictory": [],
            "limitations": ["Direction is measured; the driver is not yet identified."],
            "needed_data": ["segment history (product, region, customer) to find where it moved"],
            "chart": None,
        })
    if not candidates and obs.get("direction") == "flat":
        candidates.append({
            "title": f"{_plain(metric)} held steady",
            "strength": "Limited evidence",
            "explanation": (f"{_plain(metric)} stayed close to its average across the "
                            f"observed period — no strong rise or fall to explain."),
            "supporting": [f"Average {_biz(obs.get('recent_mean'))} across {obs.get('n', 0):,} observations"],
            "metrics": {"change_pct": obs.get("change_pct"), "n": obs.get("n", 0)},
            "contradictory": [],
            "limitations": ["A flat average can hide offsetting moves in segments."],
            "needed_data": ["segment history to check for hidden shifts"],
            "chart": None,
        })

    # Rank: strongest evidence first, cap at 4.
    candidates.sort(key=lambda c: (_STRENGTH_ORDER.get(c["strength"], 3), c["title"]))
    candidates = candidates[:4]
    # Polish wording with the LLM when available; data stays authoritative.
    if llm is not None:
        _polish(candidates, obs, question, llm)

    return {
        "ok": True,
        "insufficient": all(c["strength"] == "Insufficient evidence" for c in candidates),
        "observation": obs,
        "hypotheses": candidates,
        "needed_data": _needed_data(candidates, time_col),
        "suggestions": suggestions,
        "detail": (None if candidates else
                   f"ARIA examined {_plain(metric)} across {obs.get('n', 0):,} records "
                   f"but found no testable pattern worth ranking as a hypothesis."),
    }


def _time_col(intel, table):
    try:
        if hasattr(intel, "find_best_datetime"):
            dt = intel.find_best_datetime(table)
            if dt:
                return dt
        dts = intel.get_datetime_cols(table) if hasattr(intel, "get_datetime_cols") else []
        return dts[0] if dts else None
    except Exception:
        return None


def _observe(engine, dialect, table, metric, time_col, pd, _fetch_pairs):
    """Baseline + change: split the series into previous vs recent halves."""
    series_n = 0
    if not time_col:
        try:
            n = _table_count(engine, dialect, table)
            series_n = int(n or 0)
        except Exception:
            pass
        return {"text": f"{_plain(metric)} was examined across {series_n:,} records.",
                "change_pct": None, "direction": "unknown", "n": series_n,
                "table": table, "metric": metric, "time_col": None}, series_n
    try:
        rows = _fetch_pairs(engine, dialect, table, time_col, metric, limit=2000)
    except TypeError:
        rows = _fetch_pairs(engine, dialect, table, time_col, metric)
    import pandas as _pd
    ts = _pd.to_datetime([r[0] for r in rows], errors="coerce")
    vals = _pd.to_numeric([r[1] for r in rows], errors="coerce")
    df = _pd.DataFrame({"t": ts, "v": vals}).dropna().sort_values("t")
    if len(df) < MIN_OBS:
        return None, len(df)
    half = len(df) // 2
    prev, recent = df["v"].iloc[:half], df["v"].iloc[half:]
    pm, rm = float(prev.mean()), float(recent.mean())
    pct = (rm - pm) / abs(pm) * 100 if pm else 0.0
    direction = "down" if pct < -1 else "up" if pct > 1 else "flat"
    first, last = df["t"].iloc[0], df["t"].iloc[-1]
    text = (f"{_plain(metric)} moved {direction} — about {pct:+.0f}% "
            f"in the recent half of the period ({str(first)[:10]} → {str(last)[:10]}).")
    if direction == "flat":
        text = (f"{_plain(metric)} held roughly steady across the period "
                f"({str(first)[:10]} → {str(last)[:10]}).")
    return {"text": text, "change_pct": round(pct, 1), "direction": direction,
            "n": len(df), "table": table, "metric": metric, "time_col": time_col,
            "prev_mean": round(pm, 2), "recent_mean": round(rm, 2)}, len(df)


def _table_count(engine, dialect, table):
    from .engine import _table_count as _tc
    return _tc(engine, dialect, table)


def _resolve_target(engine, dialect, intel, tables, question,
                    parse_intent, fill_defaults):
    """Pick (table, metric, count_mode) grounded in the question + data.

    Scores every table by name overlap, then every runtime-verified numeric
    column by name/synonym overlap. Falls back to record counts when the
    question names a table with no usable measure (e.g. "customers
    dropping"), and to the legacy parser only when nothing scores.
    Returns (table, metric, count_mode); metric is None when unresolvable.
    """
    qtok = _q_tokens(question)
    scored = []
    for t in tables or []:
        t_tok = _name_tokens(t)
        t_hit = len(qtok & t_tok)
        try:
            num_cols = intel.get_numeric_cols(t) or []
        except Exception:
            num_cols = []
        for c in num_cols:
            c_tok = _name_tokens(c)
            hit = len(qtok & (c_tok | _SYNONYMS.get(str(c).lower(), set())))
            for syn, members in _SYNONYMS.items():
                if syn in qtok and str(c).lower() in members:
                    hit += 2
            if t_hit:
                hit += 2 * t_hit
            if hit > 0:
                scored.append((hit, t, c))
    scored.sort(key=lambda x: -x[0])
    for _, t, c in scored:
        if _metric_is_numeric(engine, dialect, t, c):
            return t, c, False
    # Volume fallback: the question names a table with no numeric measure.
    for t in tables or []:
        if qtok & _name_tokens(t):
            return t, None, True
    # Legacy parser fallback.
    try:
        intent = fill_defaults(parse_intent(question, intel, tables), intel, tables)
        m, tb = intent.get("metric"), intent.get("metric_table")
        if m and tb and _metric_is_numeric(engine, dialect, tb, m):
            return tb, m, False
    except Exception as exc:
        log.info("Resolver legacy fallback skipped (%s).", exc)
    return None, None, False


def _q_tokens(question):
    stop = {"why", "did", "does", "do", "are", "is", "the", "a", "an", "of",
            "to", "in", "on", "what", "which", "how", "change", "changed",
            "changing", "drop", "dropping", "decline", "declining", "fall",
            "falling", "increase", "decreasing", "decrease", "rise", "poorly",
            "perform", "my", "our", "s", "t"}
    return {_singular(w) for w in re.findall(r"[a-z0-9]+", str(question or "").lower())
            if w not in stop}


def _singular(word):
    """Light plural normalization so 'products' matches 'product'."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("ses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _name_tokens(name):
    return {_singular(w) for w in re.findall(r"[a-z0-9]+", str(name or "").lower())}


_SYNONYMS = {
    "price": {"price", "unit_price", "amount"},
    "prices": {"price", "unit_price", "amount"},
    "pricing": {"price", "unit_price", "amount"},
    "sale": {"sales", "amount", "revenue", "total", "quantity"},
    "sales": {"sales", "amount", "revenue", "total", "quantity"},
    "revenue": {"revenue", "amount", "total", "sales"},
    "order": {"orders", "quantity", "qty"},
    "orders": {"orders", "quantity", "qty"},
    "customer": {"customers", "customer"},
    "customers": {"customers", "customer"},
    "product": {"products", "product"},
    "products": {"products", "product"},
    "profit": {"profit", "margin", "amount"},
    "cost": {"cost", "freight", "amount"},
    "freight": {"freight", "cost"},
    "quantity": {"quantity", "qty", "orders"},
}


def _investigate_count(engine, dialect, intel, table, question):
    """Volume investigation for tables with no usable measure.

    Tracks record counts over time (e.g. "customers dropping"): earlier vs
    recent halves, with an honest insufficient path when no time field
    exists. Never raises.
    """
    label = _plain(table)
    try:
        time_col = _time_col(intel, table)
        if not time_col:
            return {"ok": True, "insufficient": True, "observation": None,
                    "hypotheses": [],
                    "detail": (f"ARIA found {label} records but no date field to "
                               f"track change over time, so a drop cannot be measured."),
                    "needed_data": [f"a date field on {label} to track change"],
                    "suggestions": _validated_suggestions(intel, table, None, None)}
        from .engine import _fetch_single
        import pandas as _pd
        times = _pd.to_datetime(_fetch_single(engine, dialect, table, time_col, limit=5000),
                                errors="coerce").dropna().sort_values()
        if len(times) < MIN_OBS:
            return {"ok": True, "insufficient": True, "observation": None,
                    "hypotheses": [],
                    "detail": (f"Only {len(times)} dated {label} records exist — "
                               f"too few to judge a change."),
                    "needed_data": ["more historical records"],
                    "suggestions": _validated_suggestions(intel, table, None, time_col)}
        cut = len(times) // 2
        early, late = times.iloc[:cut], times.iloc[cut:]
        span_e = (early.iloc[-1] - early.iloc[0]).days or 1
        span_l = (late.iloc[-1] - late.iloc[0]).days or 1
        rate_e = len(early) / span_e
        rate_l = len(late) / span_l
        pct = (rate_l - rate_e) / abs(rate_e) * 100 if rate_e else 0.0
        direction = "down" if pct < -5 else "up" if pct > 5 else "flat"
        obs = {"text": (f"{label} records arrived {direction} — about {pct:+.0f}% "
                        f"per day in the recent half of the period."),
               "change_pct": round(pct, 1), "direction": direction,
               "n": len(times), "table": table, "metric": f"{label} records",
               "time_col": time_col}
        if direction == "flat":
            cands = [{"title": f"{label} volume held steady",
                      "strength": "Limited evidence",
                      "explanation": (f"New {label} records arrived at roughly the same "
                                      f"daily pace throughout the period."),
                      "supporting": [f"{len(times):,} dated records"],
                      "metrics": {"change_pct": round(pct, 1), "n": len(times)},
                      "contradictory": [],
                      "limitations": ["Record counts alone cannot show why pace changed."],
                      "needed_data": ["segment history to see where volume moved"],
                      "chart": None}]
        else:
            verb = "fell" if direction == "down" else "rose"
            cands = [{"title": f"Fewer new {label} records ({verb} {abs(pct):.0f}%)",
                      "strength": "Moderate evidence" if abs(pct) >= 15 else "Limited evidence",
                      "explanation": (f"The arrival pace of {label} records {verb} about "
                                      f"{abs(pct):.0f}% in the recent half of the period."),
                      "supporting": [f"{len(early):,} earlier vs {len(late):,} recent records"],
                      "metrics": {"change_pct": round(pct, 1), "n": len(times)},
                      "contradictory": [],
                      "limitations": ["Pace change is measured; the reason is not yet identified."],
                      "needed_data": ["segment history to see whose activity changed"],
                      "chart": None}]
        return {"ok": True,
                "insufficient": all(c["strength"] == "Insufficient evidence" for c in cands),
                "observation": obs, "hypotheses": cands,
                "needed_data": _needed_data(cands, time_col),
                "suggestions": _validated_suggestions(intel, table, None, time_col)}
    except Exception as exc:
        log.info("Count investigation failed (%s).", exc)
        return {"ok": True, "insufficient": True, "observation": None,
                "hypotheses": [],
                "detail": "The investigation could not run on this data.",
                "needed_data": [],
                "suggestions": []}


def _first_measure(intel, tables):
    """First usable numeric measure across tables (id-like columns excluded)."""
    try:
        from .engine import _usable_measures
        for t in tables or []:
            cols = _usable_measures(intel, t) or []
            if cols:
                return t, cols[0]
    except Exception:
        pass
    return None, None


def _metric_is_numeric(engine, dialect, table, metric, sample=50) -> bool:
    """Runtime check: does this column actually hold numbers?"""
    try:
        from .engine import _fetch_single
        vals = _fetch_single(engine, dialect, table, metric, limit=sample)
    except Exception:
        return True  # can't verify — don't block on it
    if not vals:
        return False
    import pandas as _pd
    good = _pd.Series(_pd.to_numeric(list(vals), errors="coerce")).notna().mean()
    return bool(good >= 0.5)


def _first_numeric_measure(engine, dialect, intel, tables, skip=None):
    """First runtime-verified numeric measure across tables."""
    try:
        from .engine import _usable_measures
        for t in tables or []:
            for c in _usable_measures(intel, t) or []:
                if c == skip:
                    continue
                if _metric_is_numeric(engine, dialect, t, c):
                    return t, c
    except Exception:
        pass
    return None, None


def _breakdown(engine, dialect, intel, table, metric, obs, run_segments, _top_segments):
    """Dimension breakdown: which groups drive the change."""
    out = []
    try:
        tops = _top_segments(engine, dialect, intel, table, metric, max_dims=3) or []
    except Exception as exc:
        log.info("Investigation breakdown skipped (%s).", exc)
        return out
    for entry in tops[:2]:
        dim = entry.get("dimension") if isinstance(entry, dict) else None
        if not dim:
            continue
        out.extend(_segment_candidate(engine, dialect, table, metric, obs,
                                      dim, run_segments))
    if not out:
        # High-cardinality dims (e.g. dozens of products) are skipped by the
        # strict segment rules; still compare the top groups descriptively
        # at Limited strength so driver questions get an answer.
        try:
            dims = intel.find_all_dimensions(table) or []
        except Exception:
            dims = []
        for dim in dims[:2]:
            cands = _segment_candidate(engine, dialect, table, metric, obs,
                                       dim, run_segments, relaxed=True)
            if cands:
                out.extend(cands)
                break
    return out


def _segment_candidate(engine, dialect, table, metric, obs, dim,
                       run_segments, relaxed=False):
    """One dimension's group comparison as a hypothesis candidate."""
    try:
        res, _lim = run_segments(engine, dialect, table, dim, metric, top_k=6)
    except Exception:
        return []
    if not res or not res.get("segments"):
        if not relaxed:
            return []
        return _descriptive_top(engine, dialect, table, metric, obs, dim)
    segs = res["segments"]
    total = sum(s["mean"] * s["n"] for s in segs)
    bottom = segs[-1]
    top3 = segs[-3:] if len(segs) >= 3 else segs
    cov = sum(s["mean"] * s["n"] for s in top3) / total if total else 0
    strength = _strength_for(abs(cov), res.get("n", 0))
    direction = obs.get("direction")
    title = f"Change concentrated in {_plain(dim)}"
    if direction == "down":
        expl = (f"The lowest {_plain(dim)} groups account for most of the total, "
                f"so weakness there drags the overall number down.")
    elif direction == "up":
        expl = (f"The strongest {_plain(dim)} groups carry most of the total, "
                f"so strength there lifts the overall number.")
    else:
        expl = (f"Performance varies across {_plain(dim)} — "
                f"{bottom['name']} trails at {_biz(bottom['mean'])}.")
    support = [f"{s['name']} — {_biz(s['mean'])} (n={s['n']:,})" for s in top3[::-1][:3]]
    return [{
        "title": title, "strength": strength, "explanation": expl,
        "supporting": support,
        "metrics": {"coverage_share": round(abs(cov), 2), "n": res.get("n", 0)},
        "contradictory": [],
        "limitations": ["Segment gaps are observational — other factors may explain them."],
        "needed_data": [f"more {_plain(dim)} history to confirm the pattern holds"],
        "chart": {"type": "hbar", "dimension": dim,
                  "groups": [{"label": s["name"], "value": s["mean"]} for s in segs[:8]]},
    }]


def _descriptive_top(engine, dialect, table, metric, obs, dim):
    """Descriptive top-groups comparison without statistical thresholds."""
    try:
        from core.schema_intel import _safe_name
        from .engine import _exec
        import pandas as _pd
        rows = _exec(engine,
                     f"SELECT {_safe_name(dim, dialect)}, {_safe_name(metric, dialect)} "
                     f"FROM {_safe_name(table, dialect)} "
                     f"WHERE {_safe_name(dim, dialect)} IS NOT NULL LIMIT 5000")
        df = _pd.DataFrame(rows, columns=["g", "v"])
        df["v"] = _pd.to_numeric(df["v"], errors="coerce")
        df = df.dropna()
        df["g"] = df["g"].astype(str)
        means = df.groupby("g")["v"].agg(["mean", "count"])
        means = means[means["count"] >= 3].sort_values("mean", ascending=False).head(6)
        if len(means) < 2:
            return []
        top = [(str(i), float(r["mean"]), int(r["count"])) for i, r in means.iterrows()]
        spread = (top[0][1] - top[-1][1]) / abs(top[0][1]) if top[0][1] else 0
        return [{
            "title": f"{_plain(dim)} varies widely for {_plain(metric)}",
            "strength": "Limited evidence",
            "explanation": (f"{top[0][0]} averages {_biz(top[0][1])} while "
                            f"{top[-1][0]} averages {_biz(top[-1][1])} — a "
                            f"{abs(spread):.0%} gap worth a closer look."),
            "supporting": [f"{n} — {_biz(m)} (n={c:,})" for n, m, c in top[:3]],
            "metrics": {"n": int(means["count"].sum())},
            "contradictory": [],
            "limitations": ["Descriptive comparison only — no change measured over time."],
            "needed_data": [f"{_plain(dim)} history over time to test direction"],
            "chart": {"type": "hbar", "dimension": dim,
                      "groups": [{"label": n, "value": m} for n, m, _c in top]},
        }]
    except Exception as exc:
        log.info("Descriptive top-groups skipped (%s).", exc)
        return []


def _volume_candidate(obs):
    """Order-volume style check is folded into the observation when thin."""
    if obs.get("n", 0) < MIN_OBS:
        return [{"title": "Too little history to judge",
                 "strength": "Insufficient evidence",
                 "explanation": ("Only a handful of observations exist, so no "
                                 "explanation can be tested reliably."),
                 "supporting": [], "metrics": {"n": obs.get("n", 0)},
                 "contradictory": [],
                 "limitations": ["Sample too small for any comparison."],
                 "needed_data": ["more historical records"],
                 "chart": None}]
    return []


def _volume_split(engine, dialect, table, metric, obs):
    """Decompose a move into transaction volume vs per-unit value.

    Counts rows per half vs sums per half: volume down + value down with
    steady implied unit value points at fewer transactions, not pricing.
    """
    if not obs.get("time_col") or obs.get("n", 0) < MIN_OBS:
        return []
    try:
        from core.schema_intel import _safe_name
        from .engine import _exec
        import pandas as _pd
        rows = _exec(engine,
                     f"SELECT {_safe_name(obs['time_col'], dialect)}, "
                     f"{_safe_name(metric, dialect)} "
                     f"FROM {_safe_name(table, dialect)} "
                     f"WHERE {_safe_name(obs['time_col'], dialect)} IS NOT NULL LIMIT 5000")
        df = _pd.DataFrame(rows, columns=["t", "v"])
        df["t"] = _pd.to_datetime(df["t"], errors="coerce")
        df["v"] = _pd.to_numeric(df["v"], errors="coerce")
        df = df.dropna().sort_values("t")
        if len(df) < MIN_OBS:
            return []
        cut = len(df) // 2
        early, late = df.iloc[:cut], df.iloc[cut:]
        en, ln = len(early), len(late)
        vol = (ln - en) / en * 100 if en else 0.0
        aov_e = early["v"].sum() / en if en else 0.0
        aov_l = late["v"].sum() / ln if ln else 0.0
        aov = (aov_l - aov_e) / abs(aov_e) * 100 if aov_e else 0.0
    except Exception as exc:
        log.info("Volume split skipped (%s).", exc)
        return []
    direction = obs.get("direction")
    if direction == "down" and vol <= -5 and abs(aov) < 5:
        return [{
            "title": "Lower transaction volume contributed to the decline",
            "strength": _strength_for(0.6, obs.get("n", 0)),
            "explanation": (f"Transaction volume fell about {abs(vol):.0f}% while the "
                            f"average value stayed roughly steady ({aov:+.0f}%), suggesting "
                            f"fewer transactions rather than lower prices."),
            "supporting": [f"Volume {vol:+.0f}%", f"Average value {aov:+.0f}%",
                           f"{obs.get('n', 0):,} observations"],
            "metrics": {"volume_change_pct": round(vol, 1),
                        "avg_value_change_pct": round(aov, 1), "n": obs.get("n", 0)},
            "contradictory": [],
            "limitations": ["Volume and value move together here; other linked factors may share the cause."],
            "needed_data": ["customer or channel history to see whose activity fell"],
            "chart": None,
        }]
    if direction == "down" and vol > -5:
        return [{
            "title": "Volume held steady — pricing or mix likely",
            "strength": "Limited evidence",
            "explanation": (f"Transaction volume barely moved ({vol:+.0f}%), so the "
                            f"decline more likely reflects lower values per transaction "
                            f"or a shift in mix than fewer transactions."),
            "supporting": [f"Volume {vol:+.0f}%", f"Average value {aov:+.0f}%"],
            "metrics": {"volume_change_pct": round(vol, 1),
                        "avg_value_change_pct": round(aov, 1), "n": obs.get("n", 0)},
            "contradictory": [],
            "limitations": ["Mix vs pricing cannot be separated without product detail."],
            "needed_data": ["product-level history to separate price from mix effects"],
            "chart": None,
        }]
    return []


def _contribution(engine, dialect, intel, table, metric, obs, _top_segments):
    """Share of the total period move explained by each group.

    Compares group sums in the earlier vs recent halves: the group whose
    delta dominates the total delta is the largest contributor.
    """
    if not obs.get("time_col") or obs.get("direction") == "flat":
        return []
    try:
        tops = _top_segments(engine, dialect, intel, table, metric, max_dims=2) or []
    except Exception as exc:
        log.info("Contribution dims skipped (%s).", exc)
        return []
    out = []
    for entry in tops[:1]:
        dim = entry.get("dimension") if isinstance(entry, dict) else None
        if not dim:
            continue
        try:
            parts = _group_half_deltas(engine, dialect, table, dim,
                                       obs["time_col"], metric)
        except Exception as exc:
            log.info("Contribution query skipped (%s).", exc)
            continue
        if not parts or parts["total_delta"] == 0:
            continue
        total = parts["total_delta"]
        ranked = sorted(parts["groups"], key=lambda g: abs(g["delta"])
                        * (1 if (g["delta"] > 0) == (total > 0) else -1),
                        reverse=True)
        top = ranked[0]
        share = abs(top["delta"] / total) if total else 0
        if share <= 0:
            continue
        strength = _strength_for(share, obs.get("n", 0))
        direction = "decline" if total < 0 else "increase"
        expl = (f"{top['label']} appears to be the largest contributor to the "
                f"{direction}, accounting for about {share:.0%} of the total move.")
        support = [f"{g['label']} {_biz(g['delta'])} ({g['delta'] / total:.0%} of move)"
                   for g in ranked[:3]]
        out.append({
            "title": f"{top['label']} drove the {direction}",
            "strength": strength, "explanation": expl,
            "supporting": support,
            "metrics": {"contribution_share": round(share, 2),
                        "group_delta": round(top["delta"], 2),
                        "total_delta": round(total, 2), "n": obs.get("n", 0)},
            "contradictory": [],
            "limitations": ["Contribution is measured association — other linked factors may share the cause."],
            "needed_data": [f"finer {_plain(dim)} history to confirm the timing"],
            "chart": {"type": "hbar", "dimension": dim,
                      "groups": [{"label": g["label"], "value": abs(g["delta"])}
                                 for g in ranked[:8]]},
        })
    return out


def _group_half_deltas(engine, dialect, table, dim, time_col, metric):
    """Per-group (recent-half sum − earlier-half sum) with row-count halves."""
    from core.schema_intel import _safe_name
    from .engine import _exec
    import pandas as _pd
    rows = _exec(engine,
                 f"SELECT {_safe_name(dim, dialect)}, {_safe_name(time_col, dialect)}, "
                 f"{_safe_name(metric, dialect)} "
                 f"FROM {_safe_name(table, dialect)} "
                 f"WHERE {_safe_name(dim, dialect)} IS NOT NULL "
                 f"AND {_safe_name(time_col, dialect)} IS NOT NULL LIMIT 5000")
    if not rows:
        return None
    df = _pd.DataFrame(rows, columns=["g", "t", "v"])
    df["t"] = _pd.to_datetime(df["t"], errors="coerce")
    df["v"] = _pd.to_numeric(df["v"], errors="coerce")
    df = df.dropna().sort_values("t")
    if len(df) < MIN_OBS:
        return None
    cut = len(df) // 2
    early, late = df.iloc[:cut], df.iloc[cut:]
    groups = []
    for g, sub in df.groupby(df["g"].astype(str)):
        in_early = sub["t"].isin(early["t"])
        in_late = sub["t"].isin(late["t"])
        if in_early.sum() >= 2 and in_late.sum() >= 2:
            groups.append({"label": str(g),
                           "delta": float(sub.loc[in_late, "v"].sum()
                                          - sub.loc[in_early, "v"].sum())})
    if not groups:
        return None
    return {"groups": groups, "total_delta": float(sum(g["delta"] for g in groups))}


def _anomaly_notes(n):
    return []


def _document_candidates(question, metric, documents):
    """Document passages as Limited-evidence candidates (never higher)."""
    scored = []
    for doc in documents or []:
        for ch in (doc.get("chunks") or [])[:20]:
            text = ch.get("text", "") if isinstance(ch, dict) else str(ch)
            s = _doc_score(question + " " + str(metric), text)
            if s > 0:
                scored.append((s, doc, text))
    if not scored:
        return []
    scored.sort(key=lambda t: t[0], reverse=True)
    s, doc, text = scored[0]
    snippet = " ".join(text.split())[:350]
    fname = doc.get("filename", "document")
    page = doc.get("page")
    label = f"{fname}" + (f" p.{page}" if page is not None else "")
    return [{
        "title": "External factor mentioned in documents",
        "strength": "Limited evidence",
        "explanation": (f"The connected report mentions a factor overlapping this "
                        f"period. The data shows association in time only — it does "
                        f"not establish a direct causal relationship."),
        "supporting": [f"From {label}: {snippet}"],
        "metrics": {},
        "contradictory": ["No measured link between the reported factor and the metric exists in the data."],
        "limitations": ["Documents describe claims, not measured effects."],
        "needed_data": ["delivery/inventory records covering the same period"],
        "chart": None,
        "source": {"type": "document", "label": label},
    }]


def _validated_suggestions(intel, table, metric, time_col):
    """Next questions guaranteed structurally investigable.

    Built only from resolved schema roles (real table/metric/dims), never
    from the user's phrasing — so every suggestion can actually run.
    Prefers metrics that have a time field, so suggestions lead somewhere.
    """
    out = []

    def _dims(t):
        try:
            return [d for d in (intel.find_all_dimensions(t) or [])][:2]
        except Exception:
            return []

    def _time(t):
        try:
            if hasattr(intel, "find_best_datetime"):
                return intel.find_best_datetime(t)
            dts = intel.get_datetime_cols(t) if hasattr(intel, "get_datetime_cols") else []
            return dts[0] if dts else None
        except Exception:
            return None

    # Prefer the current table when it has a time field; otherwise point at
    # the first table that does, so suggestions don't dead-end.
    tables = []
    try:
        tables = list((intel.tables or {}).keys())
    except Exception:
        pass
    ordered = []
    if table:
        ordered.append(table)
    ordered.extend([t for t in tables if t != table])
    # Consider every table for a time-capable metric (cheap metadata-only
    # checks); only the final two suggestions are kept.
    timed = []
    for t in ordered:
        m = _first_table_metric(intel, t)
        if m:
            timed.append((t, m, _time(t)))
        if len([x for x in timed if x[2] is not None]) >= 3 and len(timed) >= 6:
            break
    timed.sort(key=lambda x: (x[2] is None, x[0] != table))
    for t, m, dt in timed[:2]:
        out.append(f"Why did {_plain(m)} change?")
        for d in _dims(t):
            out.append(f"How does {_plain(m)} vary by {_plain(d)}?")
            break
        if dt:
            out.append(f"How has {_plain(m)} changed over time?")
            break
    seen, clean = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            clean.append(s)
    return clean[:4]


def _first_table_metric(intel, table):
    try:
        from .engine import _usable_measures
        cols = _usable_measures(intel, table) or []
        return cols[0] if cols else None
    except Exception:
        return None


def _needed_data(candidates, time_col):
    needed = []
    for c in candidates:
        needed.extend(c.get("needed_data", []))
    if not time_col:
        needed.append("a date field to enable period comparison")
    seen, out = set(), []
    for n in needed:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out[:4]


# --------------------------------------------------------------------------
# Automatic detection (t.txt §§6-7, 11, algorithm stack): no question needed.
# Change-point (CUSUM-style) + volume/value decomposition + contribution +
# anomalies, all deterministic. Returns detected events; the top event is
# investigated through the same pipeline as user questions.
# --------------------------------------------------------------------------

def auto_detect(engine, dialect, intel, tables, documents=None, llm=None,
                max_tables=3) -> dict:
    """Scan connected data for business events worth investigating."""
    from .engine import _usable_measures
    events = []
    try:
        for table in (tables or [])[:max_tables]:
            if intel.tables.get(table, {}).get("empty"):
                continue
            for metric in (_usable_measures(intel, table) or [])[:2]:
                ev = _scan_metric(engine, dialect, intel, table, metric)
                if ev:
                    events.append(ev)
    except Exception as exc:
        log.info("Auto-detection scan failed (%s).", exc)
    events.sort(key=lambda e: abs(e.get("change_pct") or 0), reverse=True)
    events = events[:5]
    return {"ok": True, "events": events}


def _monthly_series(engine, dialect, table, time_col, metric):
    """Bucket (count, sum) per month. Returns list of (label, n, total)."""
    from core.schema_intel import _safe_name
    from .engine import _exec
    if dialect == "mysql":
        gexpr = f"DATE_FORMAT({_safe_name(time_col, dialect)}, '%Y-%m')"
    elif dialect == "sqlite":
        gexpr = f"strftime('%Y-%m', {_safe_name(time_col, dialect)})"
    else:
        gexpr = f"TO_CHAR({_safe_name(time_col, dialect)}, 'YYYY-MM')"
    rows = _exec(engine,
                 f"SELECT {gexpr} AS p, COUNT(*) AS n, "
                 f"SUM({_safe_name(metric, dialect)}) AS v "
                 f"FROM {_safe_name(table, dialect)} "
                 f"WHERE {_safe_name(time_col, dialect)} IS NOT NULL "
                 f"GROUP BY p ORDER BY p LIMIT 36")
    out = []
    for r in rows or []:
        try:
            out.append((str(r[0]), int(r[1]), float(r[2]) if r[2] is not None else 0.0))
        except (TypeError, ValueError, IndexError):
            continue
    return out


def _change_point(months) -> dict | None:
    """CUSUM-style change point: where did behavior actually shift?"""
    import statistics as _st
    if len(months) < 6:
        return None
    vals = [m[2] for m in months]
    mean = _st.mean(vals)
    dev = [v - mean for v in vals]
    cum, best, idx = 0.0, 0.0, -1
    for i, d in enumerate(dev):
        cum += d
        if abs(cum) > abs(best):
            best, idx = cum, i
    if idx <= 0 or idx >= len(vals) - 1:
        return None
    try:
        sd = _st.pstdev(vals) or 1.0
    except _st.StatisticsError:
        return None
    significance = abs(best) / (sd * (len(vals) ** 0.5))
    if significance < 1.0:
        return None
    before = vals[:idx + 1]
    after = vals[idx + 1:]
    ma = sum(before) / len(before)
    mb = sum(after) / len(after)
    pct = (mb - ma) / abs(ma) * 100 if ma else 0.0
    return {"index": idx, "label": months[idx][0],
            "change_pct": round(pct, 1), "significance": round(significance, 2)}


def _scan_metric(engine, dialect, intel, table, metric):
    """Detect events for one metric: change-point, trend, volume split, anomaly."""
    time_col = _time_col(intel, table)
    if not time_col:
        return None
    try:
        months = _monthly_series(engine, dialect, table, time_col, metric)
    except Exception as exc:
        log.info("Auto scan query failed (%s).", exc)
        return None
    if len(months) < 4:
        return None
    cp = _change_point(months)
    recent = months[-3:]
    earlier = months[:-3]
    rv = sum(m[2] for m in recent)
    ev = sum(m[2] for m in earlier)
    if ev or rv:
        base = ev / len(earlier)
        # Zero baseline: percent change is undefined — never fabricate a
        # denominator (None displays as "n/a"; only genuine movement from
        # an all-zero baseline yields None, an all-zero series is 0.0).
        change = (rv / len(recent) - base) / abs(base) * 100 if base else None
    else:
        change = 0.0
    rn = sum(m[1] for m in recent)
    en = sum(m[1] for m in earlier) or 1
    vol_change = (rn / 3 - en / max(1, len(earlier))) / abs(en / max(1, len(earlier))) * 100
    # Anomaly: latest month z-score vs history.
    import statistics as _st
    hist = [m[2] for m in months[:-1]]
    anomaly = None
    if len(hist) >= 4:
        try:
            sd = _st.pstdev(hist)
            if sd > 0:
                z = (months[-1][2] - _st.mean(hist)) / sd
                if abs(z) >= 2.0:
                    anomaly = {"label": months[-1][0], "z": round(z, 2)}
        except _st.StatisticsError:
            pass
    # change is None only for a 0 -> non-zero transition (unbounded, not
    # small), so it must not suppress the event the way < 5% would.
    weak_change = abs(change) < 5 if change is not None else False
    if weak_change and not anomaly and (cp is None or abs(cp["change_pct"]) < 10):
        return None
    return {"table": table, "metric": metric, "time_col": time_col,
            "change_pct": round(change, 1) if change is not None else None,
            "volume_change_pct": round(vol_change, 1),
            "change_point": cp, "anomaly": anomaly,
            "periods": len(months), "last_period": months[-1][0]}


def auto_investigate(engine, dialect, intel, tables, documents=None, llm=None) -> dict:
    """Full automatic pass: detect events, investigate the top one."""
    found = auto_detect(engine, dialect, intel, tables, documents, llm)
    events = found.get("events", [])
    if not events:
        return {"ok": True, "events": [], "insufficient": True,
                "detail": ("ARIA scanned the connected data but found no strong "
                           "changes worth investigating. Ask a question to investigate manually."),
                "needed_data": ["more historical records to establish unusual movement"]}
    top = events[0]
    metric = _plain(top["metric"])
    question = f"Why did {metric} change?"
    out = investigate_question(engine, dialect, intel, tables, question,
                               documents=documents, llm=llm)
    out["events"] = events
    out["auto_question"] = question
    return out


_EXPL_NUM_RE = re.compile(r"\$?-?\d[\d,]*(?:\.\d+)?(?:\s*[kKmMbB])?(?![a-zA-Z])%?")


def _norm_num(tok: str) -> str:
    """Canonical numeric token: strip ',' '$' '%' and spaces, case-fold."""
    out = "".join(tok.replace(",", "").replace("$", "")
                  .replace("%", "").split()).lower()
    return out[:-2] if out.endswith(".0") else out


def _num_tokens(text) -> list[str]:
    """Normalized numeric tokens in ``text`` ('12.4%', '$1,234' -> '12.4', '1234')."""
    return [t for t in (_norm_num(m.group(0))
                        for m in _EXPL_NUM_RE.finditer(str(text or ""))) if t]


def _rewrite_numbers_ok(orig: list[str], allowed: list[str], text: str) -> bool:
    """Deterministic number check for an LLM rewrite.

    Mirrors ``insight_agent.agent._validate_polished_story``'s "unmapped
    numeric" rejection, scoped to this module's prose: every number of the
    computed explanation must survive unchanged, and the rewrite may only
    use numbers it was already shown (explanation + supporting evidence).
    """
    new = set(_num_tokens(text))
    orig_set = set(orig)
    if not orig_set <= new:
        return False
    return new <= (set(allowed) | orig_set)


def _polish(candidates, obs, question, llm):
    """LLM polishes wording only; measurements stay untouched.

    Numbers are extracted from the original explanation and its supporting
    fields BEFORE the call (the call replaces the explanation); a rewrite
    that drops, changes, or invents a number is discarded and the computed
    explanation is kept.
    """
    for c in candidates:
        try:
            original = c.get("explanation") or ""
            orig_nums = _num_tokens(original)
            allowed_nums = list(orig_nums)
            for s in c.get("supporting") or []:
                allowed_nums.extend(_num_tokens(s))
            metrics = c.get("metrics")
            if isinstance(metrics, dict):
                for v in metrics.values():
                    if isinstance(v, (int, float)) and v == v:
                        allowed_nums.append(_norm_num(f"{float(v):g}"))
                    else:
                        allowed_nums.extend(_num_tokens(v))
            prompt = (
                "You are ARIA, a business analyst. Rewrite the explanation below "
                "in one plain-English sentence for a business owner. Keep every "
                "number and fact exactly as written. Never add causes, never "
                "claim certainty, use 'may'/'appears' language.\n\n"
                f"EXPLANATION: {original}\n"
                f"SUPPORT: {'; '.join(c.get('supporting', [])[:4])}"
            )
            text = llm.complete("story", prompt, temperature=0.2,
                                num_predict=120, timeout=60)
            text = (text or "").strip()
            if text and len(text) >= 12:
                if orig_nums and not _rewrite_numbers_ok(orig_nums, allowed_nums, text):
                    log.info("Investigation polish rejected (numbers changed); "
                             "keeping the computed explanation.")
                    continue
                c["explanation"] = text
        except Exception as exc:
            log.info("Investigation polish skipped (%s).", exc)
            continue
