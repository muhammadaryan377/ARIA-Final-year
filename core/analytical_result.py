"""Analytical result reasoning for ARIA.

The Goal IR states what the user asked for; a raw SQL result does not. This
module is the single place that decides whether a returned result actually
answers the question, repairs transaction-level output into the requested
granularity using only the rows that were returned, and produces the answer,
story and visualization from the validated frame.

Rules:
- truth layer only: re-aggregation uses returned values, never invented ones;
- provider-agnostic and deterministic (identical on local and cloud LLMs);
- intent-aware: granularity, ranking, sorting and chart type follow the
  question, not the incidental shape of the returned columns.
"""

from __future__ import annotations

import math
import re
from datetime import datetime


_NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?$")
_DATETIME_NAME_RE = re.compile(
    r"(date|time|month|year|period|day|week|quarter|timestamp)", re.IGNORECASE
)

_GENERIC_CONCEPTS = {
    "general kpi", "aggregate total", "row count", "average value",
    "maximum value", "minimum value", "sales kpi", "revenue kpi",
    "profit kpi", "growth kpi",
}

_LABEL_PRIORITY = (
    "full_name", "fullname", "display_name", "name", "company_name", "company",
    "product_name", "customer_name", "employee_name", "title", "label",
    "description", "category_name", "category", "segment", "region", "country",
    "city", "state", "status",
)

_MEASURE_PRIORITY = (
    "total", "amount", "value", "revenue", "sales", "sum", "count", "quantity",
    "qty", "units", "price", "cost", "profit", "margin", "score", "rate",
)

_MEASURE_WORDS = {
    "value", "amount", "total", "revenue", "sales", "price", "quantity", "qty",
    "cost", "profit", "margin", "sum", "volume", "income",
}

_GROUPING_OPS = {"ranking", "aggregation", "distribution", "comparison", "summary"}

_TREND_OPS = {"trend"}


def _norm(value) -> str:
    return str(value or "").strip().lower()


def _singular(word) -> str:
    w = str(word or "")
    if w.endswith("ies") and len(w) > 3:
        return w[:-3] + "y"
    if w.endswith("ses") and len(w) > 3:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 1:
        return w[:-1]
    return w


def natural_label(identifier) -> str:
    """Identifier -> plain-English phrase: 'order_details' -> 'order detail'.

    Schema-agnostic: only the column part of 'table.column' is used, so a
    qualified reference like 'order_details.unit_price' reads as
    'unit price' — the table name never leaks into prose. No underscores,
    dots or SQL ever survive: output is plain words only.
    """
    text = str(identifier or "")
    # Drop any table qualifier: 'order_details.unit_price' -> 'unit_price'.
    if "." in text:
        text = text.split(".")[-1]
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", text) if p]
    if not parts:
        return re.sub(r"[_.\"]+", " ", str(identifier or "")).strip() or "value"
    _ALIASES = {"avg": "average", "qty": "quantity", "cnt": "count",
                "std": "standard deviation", "sum": "total",
                "min": "minimum", "max": "maximum"}
    out = []
    for p in parts:
        low = p.lower()
        if low in _ALIASES:
            out.append(_ALIASES[low])
        else:
            out.append(_singular(low))
    return " ".join(out)


def human_entity(identifier) -> str:
    """Plural human noun for a grouped entity: 'invoice_line' -> 'invoice lines'.

    Schema-agnostic pluralization of the last word only ('orders' stays
    'orders'). Used for entity nouns so prose never shows a raw table name.
    """
    base = natural_label(identifier)
    if not base:
        return "groups"
    words = base.split()
    words[-1] = _plural(words[-1])
    return " ".join(words)


def _scrub_prose(text: str) -> str:
    """Safety net: raw identifiers must never reach the user.

    Replaces any surviving snake_case / table.column / dotted reference with
    plain words and strips SQL keywords, so even an unexpected label cannot
    leak raw schema text into conversational prose.
    """
    s = str(text or "")
    # table.column -> column words
    s = re.sub(r"[A-Za-z_][A-Za-z0-9_]*\.([A-Za-z_][A-Za-z0-9_]*)",
               lambda m: natural_label(m.group(1)), s)
    # surviving snake_case -> spaces
    s = re.sub(r"[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+",
               lambda m: natural_label(m.group(0)), s)
    # Raw SQL leaks are conventionally UPPERCASE; everyday words like
    # "from"/"on" must survive in conversational prose, so this match is
    # deliberately case-sensitive.
    s = re.sub(r"\b(SELECT|FROM|WHERE|GROUP BY|ORDER BY|JOIN|LIMIT|OFFSET|SQL)\b",
               "the data", s)
    return re.sub(r"\s+", " ", s).strip()


def _plural(label: str) -> str:
    if not label:
        return label
    if label.endswith("y") and len(label) > 1 and label[-2] not in "aeiou":
        return label[:-1] + "ies"
    if label.endswith(("s", "x", "z", "ch", "sh")):
        return label + "es"
    return label + "s"


def _is_id_like(name) -> bool:
    low = _norm(name)
    if not low:
        return False
    # Whole-token / suffix match only — bare endswith("id") misfires on
    # ordinary words ("valid", "grid"). CamelCase is split first so
    # MediaTypeId still matches.
    import re as _re
    _split = _re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name or "")).lower()
    _toks = set(t for t in _re.split(r"[^a-z0-9]+", _split) if t)
    if _toks & {"id", "uuid", "guid", "key", "code"}:
        return True
    return (
        low in ("id", "key", "code", "codeid")
        or low.endswith("_id") or low.endswith("_key") or low.endswith("_code")
    )


def _to_float(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(value, str):
        s = value.strip().replace(",", "").replace("$", "").replace("%", "")
        if _NUMERIC_RE.match(s):
            try:
                return float(s)
            except ValueError:
                return None
    return None


def _looks_like_datetime(value) -> bool:
    if isinstance(value, datetime):
        return True
    if not isinstance(value, str):
        return False
    s = value.strip()
    if len(s) < 6:
        return False
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m",
                "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            datetime.strptime(s[:19], fmt)
            return True
        except ValueError:
            continue
    return bool(re.match(r"^\d{4}-\d{2}", s))


def _fmt(value) -> str:
    """Human-readable number formatting (reuses the insights-engine pattern).

    Large magnitudes use compact 1-decimal M/K suffixes (1.84M / 184K);
    smaller values keep grouped thousands with up to 1 decimal. Plain,
    jargon-free output for user-facing answer/story strings.
    """
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return str(value)
        if math.isnan(f) or math.isinf(f):
            return "—"
        a = abs(f)
        if a >= 1_000_000:
            return f"{f / 1_000_000:.1f}M"
        if a >= 1_000:
            return f"{f / 1_000:.1f}K"
        if isinstance(value, int):
            return f"{value:,}"
        if a >= 100:
            return f"{f:,.0f}"
        if a >= 1:
            return f"{f:,.1f}"
        return f"{f:,.2f}"
    return str(value)


def profile_columns(columns, rows) -> dict:
    """Classify returned columns into numeric / text / datetime / id-like."""
    sample = rows[:50]
    profile = {"numeric": [], "text": [], "datetime": [], "id_like": []}
    for col in columns or []:
        values = [r.get(col) for r in sample if isinstance(r, dict)]
        non_null = [v for v in values if v is not None and v != ""]
        numeric_hits = sum(1 for v in non_null if _to_float(v) is not None)
        dt_hits = sum(1 for v in non_null if _looks_like_datetime(v))
        if _is_id_like(col):
            profile["id_like"].append(col)
        if non_null and numeric_hits / len(non_null) >= 0.8:
            profile["numeric"].append(col)
        if _DATETIME_NAME_RE.search(str(col)) or (non_null and dt_hits / len(non_null) >= 0.8):
            profile["datetime"].append(col)
        if col not in profile["numeric"] and col not in profile["datetime"]:
            profile["text"].append(col)
    return profile


def _find_col(columns, *candidates):
    lowered = {_norm(c): c for c in columns or []}
    for cand in candidates:
        if cand and _norm(cand) in lowered:
            return lowered[_norm(cand)]
    return None


def _find_col_loose(columns, *candidates):
    exact = _find_col(columns, *candidates)
    if exact:
        return exact
    for cand in candidates:
        low = _norm(cand)
        if not low:
            continue
        for col in columns or []:
            cl = _norm(col)
            if cl.endswith(low) or low.endswith(cl):
                return col
    return None


def extract_intent(ir, goal="") -> dict:
    """Read the requested operation, entity, measure and ranking from the IR."""
    ir = ir or {}
    operation = _norm(ir.get("operation")) or _norm(ir.get("intent")) or "summary"
    ranking = ir.get("ranking") or {}
    direction = _norm(ranking.get("direction")) or "desc"
    limit = ranking.get("limit")

    entity = None
    for dim in ir.get("dimensions") or []:
        table = dim.get("resolved_table") or dim.get("concept")
        if not table:
            continue
        entity = {
            "table": table,
            "column": dim.get("resolved_column"),
            "concept": dim.get("concept") or table,
        }
        break

    metric = None
    metrics = ir.get("metrics") or []
    if metrics:
        m = metrics[0]
        agg = m.get("aggregation")
        expression = _norm(m.get("resolved_expression"))
        if not agg and expression == "count(*)":
            agg = "COUNT"
        metric = {
            "aggregation": agg or ("COUNT" if not m.get("resolved_column") else "SUM"),
            "column": m.get("resolved_column"),
            "table": m.get("resolved_table"),
            "expression": m.get("resolved_expression"),
            "concept": m.get("concept"),
            "inferred": bool(m.get("inferred")),
        }

    goal_l = _norm(goal) or _norm(ir.get("original_goal"))
    goal_words = set(re.findall(r"[a-z0-9]+", goal_l))
    try:
        goal_tokens = {_singular(w) for w in goal_words}
    except Exception:
        goal_tokens = set(goal_words)
    if metric and metric.get("aggregation") == "COUNT" and metric.get("inferred"):
        if goal_words & _MEASURE_WORDS:
            metric["aggregation"] = "SUM"

    wants_share = bool(goal_words & {"share", "percentage", "percent", "proportion",
                                     "composition", "pie", "donut", "doughnut"})

    return {
        "operation": operation,
        "intent": _norm(ir.get("intent")) or operation,
        "direction": direction,
        "limit": int(limit) if limit else (10 if operation == "ranking" else None),
        "entity": entity,
        "metric": metric,
        "wants_share": wants_share,
        "goal_words": goal_tokens,
    }


def _col_is_text(col, profile) -> bool:
    return (
        col not in profile["numeric"]
        and col not in profile["datetime"]
        and not _is_id_like(col)
    )


def _resolve_entity_key(intent, columns, rows, profile):
    table = (intent.get("entity") or {}).get("table")
    if table:
        t = _norm(table)
        found = _find_col_loose(columns, f"{t}_id", f"{t}id", f"{t}_key", f"id_{t}")
        if found:
            return found
    ratio = {}
    for col in profile["id_like"]:
        vals = {str(r.get(col)) for r in rows if isinstance(r, dict) and r.get(col) is not None}
        if vals:
            ratio[col] = len(vals) / max(1, len(rows))
    repeats = {c: v for c, v in ratio.items() if 0.02 <= v <= 0.9}
    if repeats:
        return min(repeats, key=lambda c: repeats[c])
    return None


def _resolve_label(intent, columns, rows, key_col, profile):
    dim_col = (intent.get("entity") or {}).get("column")
    if dim_col:
        found = _find_col_loose(columns, dim_col)
        if found and found != key_col and _col_is_text(found, profile):
            return found
    for cand in _LABEL_PRIORITY:
        found = _find_col_loose(columns, cand)
        if found and found != key_col and _col_is_text(found, profile):
            return found
    for col in columns or []:
        if col != key_col and _col_is_text(col, profile):
            return col
    return key_col


def _resolve_measure(intent, columns, rows, profile):
    metric = intent.get("metric") or {}
    if metric.get("aggregation") == "COUNT" and not metric.get("column"):
        return None
    col = metric.get("column")
    if col:
        found = _find_col_loose(columns, col)
        if found and found in profile["numeric"]:
            return found
    for cand in _MEASURE_PRIORITY:
        for c in profile["numeric"]:
            if cand in _norm(c):
                return c
    nums = [c for c in profile["numeric"] if not _is_id_like(c)]
    if nums:
        return nums[-1]
    return profile["numeric"][-1] if profile["numeric"] else None


def _aggregate_rows(rows, key_col, label_col, measure_col, aggregation):
    buckets, order = {}, []
    for r in rows:
        if not isinstance(r, dict):
            continue
        key = r.get(key_col)
        if key is None:
            continue
        skey = str(key)
        if skey not in buckets:
            buckets[skey] = {"label": None, "values": [], "count": 0}
            order.append(skey)
        b = buckets[skey]
        b["count"] += 1
        if b["label"] is None and label_col:
            lv = r.get(label_col)
            if lv not in (None, ""):
                b["label"] = lv
        if measure_col:
            v = _to_float(r.get(measure_col))
            if v is not None:
                b["values"].append(v)
    out = []
    for skey in order:
        b = buckets[skey]
        vals = b["values"]
        if aggregation == "COUNT" or not vals:
            value = float(b["count"])
        elif aggregation == "AVG":
            value = sum(vals) / len(vals)
        elif aggregation == "MIN":
            value = min(vals)
        elif aggregation == "MAX":
            value = max(vals)
        else:
            value = sum(vals)
        out.append({
            "key": skey,
            "label": b["label"] if b["label"] is not None else skey,
            "value": round(float(value), 2),
        })
    return out


def _frame(agg_rows, label_name, measure_name):
    return [{label_name: a["label"], measure_name: a["value"]} for a in agg_rows]


def _labels_unique(rows, label_col, columns) -> bool:
    """True when each result row is its own group (grouped shape, not raw
    transaction rows): the label column has no repeats. Lets clean small
    breakdowns on key-less sources (typical CSVs) validate honestly."""
    try:
        key = label_col or (columns or [None])[0]
        if not key:
            return False
        vals = [str(r.get(key)) for r in rows if isinstance(r, dict)]
        return bool(vals) and len(set(vals)) == len(vals)
    except Exception:
        return False


def _measure_label(intent, measure_col, aggregation) -> str:
    metric = intent.get("metric") or {}
    col = measure_col or metric.get("column")
    if aggregation == "COUNT" and not col:
        return "record count"
    if col:
        return natural_label(col)
    concept = metric.get("concept")
    if concept and _norm(concept) not in _GENERIC_CONCEPTS:
        return natural_label(concept)
    return "value"


def _entity_label(intent) -> str:
    """Human noun for the entity being grouped/ranked.

    Prefers the entity concept/table name, but when that is not a meaningful
    word (e.g. a CSV upload whose table is the filename stem 's'), falls back
    to the resolved grouping column so the answer reads "Region North …"
    rather than "S North …". Likewise, when the concept is merely the
    humanized table name (file-derived tables such as "diag sales"), the
    grouping column is the meaningful entity ("Product …", never
    "Diag sale Banana …"), so table names never leak into prose.
    """
    ent = intent.get("entity") or {}
    concept = ent.get("concept") or ent.get("table") or "group"
    col = ent.get("column")
    table = ent.get("table")
    if col and not _is_id_like(col):
        try:
            col_tokens = {_singular(w) for w in re.findall(r"[a-z0-9]+", _norm(col))}
            concept_tokens = {_singular(w) for w in re.findall(r"[a-z0-9]+", _norm(concept))}
            gw = set(intent.get("goal_words") or set())
            col_hit = bool(col_tokens & gw)
            concept_hit = bool(concept_tokens & gw)
            # The question names the grouping ("Which product…?", "top
            # customers…"): that wording wins, so table names never leak
            # into prose ("Product banana…", never "Diag sale banana…").
            if col_hit and not concept_hit:
                return natural_label(col)
            if concept_hit and not col_hit:
                return natural_label(concept)
        except Exception:
            pass
        if len(str(concept).strip()) < 3:
            return natural_label(col)
        try:
            if _norm(natural_label(concept)) == _norm(natural_label(table or "")):
                return natural_label(col)
        except Exception:
            pass
    if not col and table:
        try:
            if _norm(concept) == _norm(natural_label(table or "")):
                return "group"
        except Exception:
            pass
    return natural_label(concept)


def synthesize_answer(frame, measure_label, entity_label) -> str:
    """Conversational answer from executed rows only.

    Short sentences, human-scale numbers (42K, 18%), no raw identifiers.
    Every number comes from `frame` — never invented.
    """
    if not frame:
        return "No matching rows were found for this question."
    measure_label = natural_label(measure_label)
    entity_label = natural_label(entity_label)
    names = list(frame[0].keys())
    if len(names) >= 2:
        label_name, measure_name = names[0], names[1]
        top = frame[:3]
        lead = top[0]
        head = (f"{entity_label.capitalize()} {lead[label_name]} leads your "
                f"{_plural(entity_label)} — {_fmt(lead[measure_name])} in "
                f"{measure_label}.")
        if len(top) > 1:
            others = ", ".join(
                f"{r[label_name]} at {_fmt(r[measure_name])}" for r in top[1:]
            )
            return _scrub_prose(f"{head} Next come {others}. "
                                f"So {lead[label_name]} is the clear standout.")
        return _scrub_prose(head)
    if len(names) == 1:
        return _scrub_prose(f"{measure_label.capitalize()} is {_fmt(frame[0][names[0]])}.")
    return _scrub_prose(f"{names[0]} = {_fmt(frame[0][names[0]])}, {names[1]} = {_fmt(frame[0][names[1]])}.")


def _numeric_key(frame, names):
    """The frame column that actually holds the computed measure."""
    if not frame:
        return names[-1] if names else None
    for n in list(names)[1:]:
        vals = [_to_float(r.get(n)) for r in frame if r.get(n) is not None]
        if vals and len(vals) == len(frame) and all(v is not None for v in vals):
            return n
    return names[1] if len(names) > 1 else names[0]


_DECLINE_WORD_RE = re.compile(
    r"\b(decline|declined|declining|declines|decrease|decreased|"
    r"decreasing|decreases|drop|dropped|dropping|drops|fall|fell|"
    r"falling|falls)\b", re.IGNORECASE)

_MONTH_NAMES = ("January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November",
                "December")


def _pretty_period(label) -> str:
    """Human month/period label: '2024-03' -> 'March 2024', else as-is."""
    try:
        s = str(label or "").strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m"):
            try:
                dt = datetime.strptime(s[:10] if len(s) >= 10 and fmt.endswith("%d")
                                       else s[:7], fmt)
                if fmt in ("%Y-%m", "%Y/%m"):
                    return f"{_MONTH_NAMES[dt.month - 1]} {dt.year}"
                return f"{_MONTH_NAMES[dt.month - 1]} {dt.year}"
            except ValueError:
                continue
        # Bare month name fragments stay readable.
        low = s.lower()
        for m in _MONTH_NAMES:
            if low.startswith(m.lower()[:3]) and len(s) <= 12:
                return m
    except Exception:
        pass
    return str(label)


def _short_month(label) -> str:
    """Short month name for decline prose ('2024-03' -> 'March')."""
    try:
        s = str(label or "").strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m"):
            try:
                dt = datetime.strptime(s[:10] if len(s) >= 10 and fmt.endswith("%d")
                                       else s[:7], fmt)
                return _MONTH_NAMES[dt.month - 1]
            except ValueError:
                continue
    except Exception:
        pass
    return str(label)


def synthesize_decline_answer(frame, measure_label, goal="") -> str:
    """Month-over-month decline answer (deterministic, schema-agnostic).

    Conversational: short sentences, human-scale numbers, up/down direction
    words. Every number comes from the ordered monthly frame — never invented.
    """
    measure_label = natural_label(measure_label)
    if not frame:
        return "No matching rows were found for this question."
    names = list(frame[0].keys())
    if len(names) < 2:
        return synthesize_answer(frame, measure_label, "period")
    label_name = names[0]
    measure_name = _numeric_key(frame, names)
    values = [(_to_float(r.get(measure_name)) or 0.0) for r in frame]
    labels = [r.get(label_name) for r in frame]
    if len(frame) < 2:
        return _scrub_prose(
            f"{_pretty_period(labels[0])} came in at {_fmt(values[0])} in "
            f"{measure_label}. There is only one period, so no month-over-month "
            f"change can be measured.")
    changes = []
    for i in range(1, len(values)):
        prev, curr = values[i - 1], values[i]
        if prev:
            pct = (curr - prev) / abs(prev) * 100.0
        else:
            pct = None
        changes.append((i, pct, prev, curr))
    declines = [(i, p, pr, cu) for (i, p, pr, cu) in changes
                if p is not None and p < 0]
    if not declines:
        # Honest: the data shows no decline.
        first, last = values[0], values[-1]
        if first:
            pct = (last - first) / abs(first) * 100.0
            move = (f"up about {abs(pct):.0f}%"
                    if pct > 0 else "flat")
        else:
            move = f"from {_fmt(first)} to {_fmt(last)}"
        return _scrub_prose(
            f"Good news — no month fell. {measure_label.capitalize()} is {move} "
            f"across {len(frame)} periods, from {_pretty_period(labels[0])} to "
            f"{_pretty_period(labels[-1])}."
        )
    declines.sort(key=lambda t: t[1])
    wi, wpct, wprev, wcurr = declines[0]
    wlab, plab = labels[wi], labels[wi - 1]
    head = (f"{_short_month(wlab)} was your toughest month — {measure_label} "
            f"fell {abs(wpct):.0f}% to {_fmt(wcurr)}, down from {_fmt(wprev)} "
            f"in {_short_month(plab)}")
    if len(declines) > 1:
        others = ", ".join(
            f"{_short_month(labels[i])} down {abs(p):.0f}% to {_fmt(cu)}"
            for (i, p, _pr, cu) in declines[1:3])
        return _scrub_prose(
            f"{head}. Other down months: {others}. "
            f"That covers {len(frame)} months, {_pretty_period(labels[0])} to "
            f"{_pretty_period(labels[-1])}.")
    return _scrub_prose(
        f"{head}. That covers {len(frame)} months, "
        f"{_pretty_period(labels[0])} to {_pretty_period(labels[-1])}.")


def synthesize_trend_answer(frame, measure_label, period_label) -> str:
    """Conversational trend summary from executed rows only."""
    measure_label = natural_label(measure_label)
    if not frame:
        return "No matching rows were found for this question."
    names = list(frame[0].keys())
    if len(names) < 2:
        return synthesize_answer(frame, measure_label, period_label)
    label_name = names[0]
    measure_name = _numeric_key(frame, names)
    values = [(_to_float(r.get(measure_name)) or 0.0) for r in frame]
    labels = [r.get(label_name) for r in frame]
    first, last = values[0], values[-1]
    if first:
        pct = (last - first) / abs(first) * 100.0
        if pct > 0:
            movement = (f"{measure_label} increased by about {abs(pct):.0f}% — "
                        f"up from {_fmt(first)} to {_fmt(last)}")
        elif pct < 0:
            movement = (f"{measure_label} decreased by about {abs(pct):.0f}% — "
                        f"down from {_fmt(first)} to {_fmt(last)}")
        else:
            movement = f"{measure_label} stayed flat near {_fmt(first)}"
    else:
        movement = f"{measure_label} moved from {_fmt(first)} to {_fmt(last)}"
    peak_i = max(range(len(values)), key=lambda i: values[i])
    low_i = min(range(len(values)), key=lambda i: values[i])
    return _scrub_prose(
        f"Across {len(frame)} periods, {movement}. "
        f"The peak was {_pretty_period(labels[peak_i])} at {_fmt(values[peak_i])}. "
        f"The low was {_pretty_period(labels[low_i])} at {_fmt(values[low_i])}."
    )


def build_story(frame, measure_label, entity_label, based_on, op="") -> str:
    """Conversational story: headline, concentration, provenance — all traceable."""
    measure_label = natural_label(measure_label)
    entity_label = natural_label(entity_label)
    if not frame:
        return "No data matched this question, so no conclusion can be drawn."
    names = list(frame[0].keys())
    if len(names) < 2:
        return synthesize_answer(frame, measure_label, entity_label)
    label_name = names[0]
    measure_name = _numeric_key(frame, names)
    values = [(_to_float(r.get(measure_name)) or 0.0) for r in frame]
    total = sum(values)
    lead = frame[0]
    lead_share = values[0] / total * 100 if total else 0
    parts = [
        synthesize_answer(frame, measure_label, entity_label),
        (f"{lead[label_name]} alone is about {lead_share:.0f}% of all "
         f"{measure_label} across {len(frame)} {_plural(entity_label)}."),
    ]
    if len(frame) > 3:
        top3 = sum(values[:3])
        top3_share = top3 / total * 100 if total else 0
        parts.append(
            f"The top three {_plural(entity_label)} together hold about "
            f"{top3_share:.0f}% — so {measure_label} is concentrated in a few names."
        )
    if based_on:
        parts.append(f"This comes from {based_on} underlying records.")
    parts.append("These patterns describe this data only.")
    return _scrub_prose(" ".join(parts))


def _compatible(op, wants_share, n_groups):
    if op == "ranking":
        base = ["bar_horizontal", "bar", "pie"]
    elif op in _TREND_OPS:
        base = ["line", "area", "bar"]
    elif op in ("comparison", "period_over_period", "contribution",
                "change_detection"):
        base = ["bar", "bar_horizontal", "line"]
    elif op in ("correlation",):
        base = ["scatter", "bar", "line"]
    elif op in ("segmentation",):
        base = ["bar_stacked", "bar", "bar_horizontal"]
    elif op == "distribution":
        base = ["histogram", "bar", "bar_horizontal"]
    else:
        base = ["bar", "bar_horizontal"]
    if wants_share and n_groups <= 8:
        base = ["pie"] + [b for b in base if b != "pie"]
    return base


def select_visualization(intent, columns, frame, measure_label) -> dict:
    op = intent["operation"]
    if not frame:
        return {
            "chart_type": "table", "label": "Table",
            "reason": "no result rows to visualize",
            "x": None, "y": None, "results": [], "compatible": [],
        }
    if len(columns) < 2:
        profile = profile_columns(columns, frame)
        numeric = profile["numeric"]
        if numeric:
            return {
                "chart_type": "histogram", "label": "Distribution",
                "reason": "single measure — its distribution shows the spread",
                "x": numeric[0], "y": "count", "results": frame[:10],
                "compatible": ["histogram", "bar"], "measure_label": measure_label,
            }
        return {
            "chart_type": "table", "label": "Table",
            "reason": "no compatible chart for this shape",
            "x": None, "y": None, "results": frame[:10], "compatible": [],
            "measure_label": measure_label,
        }

    # Validation: x/y must match real column TYPES, not positions.
    # First row drives typing (numbers vs text vs date-ish names).
    first = frame[0] if frame and isinstance(frame[0], dict) else {}
    def _is_num(c):
        v = first.get(c)
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    num_cols = [c for c in columns if _is_num(c)]
    dt_cols = [c for c in columns if any(k in c.lower() for k in
               ("date", "time", "month", "year", "day", "period"))]
    cat_cols = [c for c in columns if c not in num_cols and first.get(c) is not None]
    # Prefer semantic slots over positions: time->x for trends, category->x + numeric->y otherwise.
    if dt_cols and num_cols:
        x, y = dt_cols[0], num_cols[0]
    elif cat_cols and num_cols:
        x, y = cat_cols[0], num_cols[0]
    else:
        x, y = columns[0], columns[1]
    n_groups = len(frame)
    # Shape guard: trend without time axis, or pie with 1 / 9+ slices, misleads — downgrade.
    if op in _TREND_OPS and not dt_cols:
        op = "comparison"  # render honest bars instead of a fake time line
    if intent.get("wants_share") and (n_groups < 2 or n_groups > 8):
        intent = dict(intent, wants_share=False)
    compatible = _compatible(op, intent.get("wants_share"), n_groups)
    reasons = {
        "bar_horizontal": "ranked comparison — names read best on a horizontal bar",
        "line": "ordered axis — a line shows the movement over time",
        "pie": "composition question over few groups — a share chart fits",
        "bar": "category comparison — bars compare the values directly",
        "histogram": "single measure — its distribution shows the spread",
    }
    if op == "ranking":
        chart_type = "bar_horizontal"
    elif op in _TREND_OPS:
        chart_type = "line"
    elif op == "correlation":
        chart_type = "scatter" if len(num_cols) >= 2 else "bar"
    elif op == "distribution":
        chart_type = "histogram"
    elif op == "segmentation":
        chart_type = "bar_stacked"
    elif op in ("comparison", "period_over_period", "contribution",
                "change_detection"):
        chart_type = "bar"
    elif intent.get("wants_share") and n_groups <= 8:
        chart_type = "pie"
    else:
        chart_type = "bar"
    labels = {
        "bar_horizontal": "Horizontal bar", "line": "Line", "pie": "Donut",
        "bar": "Bar", "histogram": "Histogram",
    }
    return {
        "chart_type": chart_type,
        "label": labels.get(chart_type, chart_type),
        "reason": reasons.get(chart_type, "best fit for the result shape"),
        "x": x,
        "y": y,
        "results": frame[:10],
        "compatible": [c for c in compatible if c != chart_type][:3],
        "measure_label": measure_label,
    }


def build_why(intent, frame, columns, entity_label, measure_label, based_on,
              confidence, issues) -> dict:
    """Conversational 'why' — plain English, human-scale numbers, no raw schema.

    `what_happened` reads like a person explaining, never 'Ranked N X by Y'.
    Table/column identifiers and SQL never surface in prose (scrubbed); the
    structured evidence keeps humanized names only.
    """
    entity_label = natural_label(entity_label)
    measure_label = natural_label(measure_label)
    limitations = list(issues)
    if not frame:
        limitations.append("The query returned no rows, so no conclusion can be drawn.")
    limitations.append("These patterns describe this data only.")
    if frame and len(columns) >= 2:
        names = list(frame[0].keys())
        lead_val = _to_float(frame[0][names[1]])
        total = sum((_to_float(r.get(names[1])) or 0.0) for r in frame)
        share = (lead_val / total * 100) if lead_val is not None and total else 0
        what = (
            f"You have {len(frame)} {_plural(entity_label)}. "
            f"{frame[0][names[0]]} leads with {_fmt(frame[0][names[1]])} in "
            f"{measure_label}"
            + (f" — about {share:.0f}% of everything measured." if total else ".")
            + " The rest trail behind."
        )
    elif frame:
        what = "The numbers boil down to a single total from your data."
    else:
        what = "The analysis produced no result to explain."
    factors = []
    if len(frame) > 1:
        factors = [f"some {_plural(entity_label)} simply bring in far more {measure_label} than others"]
    return {
        "what_happened": _scrub_prose(what),
        "why_it_matters": _scrub_prose(
            f"This answers your question straight from "
            f"{based_on or 'the returned'} records. "
            f"It shows where your {measure_label} is coming from."
        ),
        "evidence": {
            "tables": [entity_label],
            "columns": [natural_label(c) for c in (columns or [])[:10]],
            "row_count": len(frame),
            "sql": None,
        },
        "contributing_factors": [_scrub_prose(f) for f in factors],
        "comparisons": [],
        "confidence": confidence,
        "limitations": [_scrub_prose(str(l)) for l in limitations],
    }


def analyze_result(ir, columns, rows, goal="", sql="", row_count=0) -> dict:
    """Validate, repair and describe a result. Returns the canonical envelope."""
    columns = list(columns or [])
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    based_on = int(row_count) if row_count else len(rows)

    intent = extract_intent(ir, goal)
    profile = profile_columns(columns, rows)

    issues = []
    repaired = False
    final_rows = rows
    final_columns = columns
    expected_granularity = "row_level"
    op = intent["operation"]
    entity_label = _entity_label(intent)

    key_col = _resolve_entity_key(intent, columns, rows, profile)
    label_col = _resolve_label(intent, columns, rows, key_col, profile) if key_col else None
    measure_col = _resolve_measure(intent, columns, rows, profile)
    aggregation = (intent.get("metric") or {}).get("aggregation") or (
        "SUM" if measure_col else "COUNT"
    )
    # The result's own first label column ("region") beats a stale entity
    # noun ("product") when the question names it ("Show amount by region"):
    # values belong to the displayed groups, never to another column.
    try:
        _cand = next(
            (c for c in (columns or [])
             if "(" not in str(c) and "*" not in str(c)
             and not _is_id_like(c) and c != measure_col),
            None,
        )
        if _cand:
            _cand_tokens = {_singular(w) for w in re.findall(r"[a-z0-9]+", _norm(_cand))}
            _gw = set(intent.get("goal_words") or set())
            _ent_tokens = {_singular(w) for w in re.findall(r"[a-z0-9]+", _norm(entity_label))}
            if _cand_tokens & _gw and not (_ent_tokens & _gw):
                entity_label = natural_label(_cand)
    except Exception:
        pass

    if op in _GROUPING_OPS and intent.get("entity") is not None and key_col:
        expected_granularity = f"one row per {entity_label}"
        groups = {str(r.get(key_col)) for r in rows if r.get(key_col) is not None}
        n_groups = len(groups)
        if n_groups and len(rows) > n_groups:
            agg = _aggregate_rows(rows, key_col, label_col, measure_col, aggregation)
            agg.sort(key=lambda a: a["value"], reverse=(intent["direction"] != "asc"))
            if intent.get("limit"):
                agg = agg[: int(intent["limit"])]
            label_name = label_col or "entity"
            measure_name = measure_col or ("count" if aggregation == "COUNT" else "value")
            final_rows = _frame(agg, label_name, measure_name)
            final_columns = [label_name, measure_name]
            repaired = True
            issues.append(
                f"the query returned transaction-level rows ({len(rows)} rows / "
                f"{n_groups} {_plural(entity_label)}); the result was re-aggregated "
                f"to the requested {expected_granularity}"
            )
    elif op in _TREND_OPS and profile["datetime"]:
        expected_granularity = "one row per period"
    elif op == "distribution" and profile["numeric"] and not profile["text"]:
        expected_granularity = "distribution over the measure"

    if not repaired and op == "ranking" and measure_col and measure_col in final_columns:
        limit = intent.get("limit") or 10
        final_rows = sorted(
            final_rows,
            key=lambda r: _to_float(r.get(measure_col)) or 0,
            reverse=(intent["direction"] != "asc"),
        )[: int(limit)]
    elif not repaired and op in _TREND_OPS and profile["datetime"]:
        tcol = profile["datetime"][0]
        final_rows = sorted(final_rows, key=lambda r: str(r.get(tcol) or ""))

    valid = bool(final_rows) and not (
        op in _GROUPING_OPS
        and intent.get("entity")
        and not key_col
        # Key-less sources (typical CSVs) have no entity key by construction;
        # unique labels still prove a grouped shape, so honest small
        # breakdowns validate instead of alarming "insufficient".
        and not _labels_unique(final_rows, label_col, final_columns)
    )
    if not final_rows:
        issues.append("the query returned no rows")

    measure_label = _measure_label(intent, measure_col, aggregation)
    confidence = "high" if valid and not repaired and len(final_rows) >= 3 else (
        "medium" if valid else "low"
    )

    # Trend questions need a time axis: without any datetime column the SQL
    # result (e.g. SUM vs COUNT) is meaningless, so refuse with a clear
    # natural-English explanation instead of a fabricated "trend".
    if op in _TREND_OPS and not profile["datetime"]:
        _no_time_msg = (
            "This data has no time or date column, so change over time "
            "can't be shown. Try a breakdown or ranking instead."
        )
        issues.append("the connected data has no time or date column for a trend")
        return {
            "data": [],
            "columns": [],
            "valid": False,
            "repaired": repaired,
            "issues": issues,
            "granularity": {
                "expected": "one row per period",
                "actual": "row_level",
                "ok": False,
            },
            "intent": intent,
            "chart_spec": {"chart_type": "table", "label": "No chart",
                           "reason": "no time column", "compatible": []},
            "answer": _no_time_msg,
            "story": _no_time_msg,
            "why": {
                "what_happened": _no_time_msg,
                "why_it_matters": "Trend analysis needs dated records.",
                "evidence": {"tables": [], "columns": [],
                             "row_count": 0, "sql": sql or None},
                "contributing_factors": [],
                "comparisons": [],
                "confidence": "low",
                "limitations": [_no_time_msg],
            },
            "consistency": {"consistent": False, "issues": issues,
                            "chart_spec": {"chart_type": "table"}},
            "analysis": {
                "based_on": based_on,
                "result_count": 0,
                "entity": entity_label,
                "entity_count": None,
                "measure_label": measure_label,
                "measure_column": measure_col,
                "dimension_columns": [],
                "aggregation": aggregation,
                "operation": op,
                "granularity": "one row per period",
                "repaired": repaired,
                "confidence": "low",
            },
        }

    if op in _TREND_OPS:
        try:
            _decline_q = bool(_DECLINE_WORD_RE.search(str(goal or "")))
        except Exception:
            _decline_q = False
        if _decline_q:
            answer = synthesize_decline_answer(final_rows, measure_label, goal)
        else:
            answer = synthesize_trend_answer(final_rows, measure_label, entity_label)
    else:
        answer = synthesize_answer(final_rows, measure_label, entity_label)
    story = build_story(final_rows, measure_label, entity_label, based_on, op)
    chart_spec = select_visualization(intent, final_columns, final_rows, measure_label)
    why = build_why(
        intent, final_rows, final_columns, entity_label, measure_label,
        based_on, confidence, issues,
    )
    # SQL stays server-side for debugging — it must never surface in prose
    # or user-facing evidence. Provenance keeps humanized names only.
    why["evidence"]["sql"] = None

    analysis = {
        "based_on": based_on,
        "result_count": len(final_rows),
        "entity": entity_label,
        "entity_count": len(final_rows) if len(final_columns) >= 2 else None,
        "measure_label": measure_label,
        "measure_column": measure_col,
        "dimension_columns": [c for c in final_columns if c != measure_col],
        "aggregation": aggregation,
        "operation": op,
        "granularity": expected_granularity,
        "repaired": repaired,
        "confidence": confidence,
    }

    return {
        "data": final_rows,
        "columns": final_columns,
        "valid": valid,
        "repaired": repaired,
        "issues": issues,
        "granularity": {
            "expected": expected_granularity,
            "actual": "row_level" if len(final_columns) == len(columns) and not repaired else expected_granularity,
            "ok": valid,
        },
        "intent": intent,
        "chart_spec": chart_spec,
        "answer": answer,
        "story": story,
        "why": why,
        "analysis": analysis,
        "consistency": {
            "consistent": valid and not issues,
            "issues": issues,
            "chart_spec": chart_spec,
        },
    }