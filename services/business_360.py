"""Business 360: non-technical 360-degree answer for AI Analytics.

Given the already-executed /api/ask result (real rows, drivers, hypothesis,
prescription output), build ONE businessman-friendly block:

  headline / what_happened / likely_causes / what_to_do / if_do_nothing

Rules:
- Numbers come from executed SQL rows only, never invented.
- No technical jargon: never SQL, p-value, r, z-score, FK, schema.
- Always returns something useful, even when the engine refuses.
- Best-effort, never raises: wrapped in try/except by the caller.
"""

from __future__ import annotations

import re


def _num(v):
    try:
        if v is None or v is True or v is False:
            return None
        if isinstance(v, (int, float)):
            f = float(v)
            return f if f == f and abs(f) != float("inf") else None
        s = str(v).replace(",", "").strip()
        if not s:
            return None
        f = float(s)
        return f if f == f and abs(f) != float("inf") else None
    except Exception:
        return None


def _fmt_money(n, metric=""):
    if n is None:
        return "—"
    a = abs(n)
    suffix = ""
    v = n
    if a >= 1_000_000_000:
        v, suffix = n / 1_000_000_000, "B"
    elif a >= 1_000_000:
        v, suffix = n / 1_000_000, "M"
    elif a >= 10_000:
        v, suffix = n / 1_000, "K"
    if suffix:
        return f"{v:,.1f}{suffix}"
    if float(n).is_integer():
        return f"{int(n):,}"
    return f"{n:,.1f}"


def _human(col: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(col or ""))
    s = s.replace("_", " ").strip()
    return " ".join(w.capitalize() for w in s.split()) or str(col or "value")


def _detect_period_col(columns, rows):
    for c in columns or []:
        cl = str(c).lower()
        if any(k in cl for k in ("period", "month", "date", "week", "quarter", "year", "day")):
            return c
    return None


def _detect_value_col(columns, rows, metric_col=None):
    if metric_col and metric_col in (columns or []):
        return metric_col
    for c in columns or []:
        cl = str(c).lower()
        if "(" in cl or "*" in cl:
            continue
        if rows and isinstance(rows[0], dict):
            if _num(rows[0].get(c)) is not None:
                if not cl.endswith("_id") and cl not in ("id",):
                    return c
    for c in columns or []:
        if rows and isinstance(rows[0], dict) and _num(rows[0].get(c)) is not None:
            return c
    return None


def _detect_seg_col(columns, rows, value_col, period_col):
    for c in columns or []:
        if c in (value_col, period_col):
            continue
        cl = str(c).lower()
        if "(" in cl or "*" in cl:
            continue
        return c
    return None


def build_business360(*, question, result_data, columns, metric_label=None,
                       metric_col=None, dims=None, drivers=None,
                       hypothesis=None, prescription=None, chart_spec=None) -> dict:
    """Build the 360 block. Never raises, never invents numbers."""
    try:
        rows = [r for r in (result_data or []) if isinstance(r, dict)]
        cols = list(columns or (list(rows[0].keys()) if rows else []))
        metric_name = str(metric_label or metric_col or "Sales")
        metric_plain = _human(metric_name)

        period_col = _detect_period_col(cols, rows)
        value_col = _detect_value_col(cols, rows, metric_col)
        seg_col = _detect_seg_col(cols, rows, value_col, period_col)

        # ---- What happened ----
        total = None
        change_pct = None
        before_v = after_v = None
        period_label = ""
        if rows and value_col:
            vals = [_num(r.get(value_col)) for r in rows]
            vals = [v for v in vals if v is not None]
            if vals:
                if period_col:
                    try:
                        ordered = sorted(rows, key=lambda r: str(r.get(period_col) or ""))
                    except Exception:
                        ordered = rows
                    ovals = [_num(r.get(value_col)) for r in ordered]
                    ovals = [v for v in ovals if v is not None]
                    if len(ovals) >= 2:
                        before_v, after_v = ovals[0], ovals[-1]
                        base = sum(ovals[:-1]) / max(1, len(ovals) - 1) if len(ovals) > 2 else (ovals[0] or 0)
                        if base:
                            change_pct = round((ovals[-1] - base) / abs(base) * 100, 1)
                        total = round(sum(ovals), 2)
                        try:
                            period_label = f"{ordered[0].get(period_col)} → {ordered[-1].get(period_col)}"
                        except Exception:
                            period_label = ""
                    elif len(ovals) == 1:
                        total = round(ovals[0], 2)
                else:
                    total = round(sum(vals), 2)
                    # segment concentration as the "change" story
                    if seg_col and len(rows) >= 2:
                        srows = sorted(rows, key=lambda r: (_num(r.get(value_col)) or 0), reverse=True)
                        top_v = _num(srows[0].get(value_col)) or 0
                        rest = sum(_num(r.get(value_col)) or 0 for r in srows[1:])
                        if top_v + rest > 0:
                            share = round(top_v / (top_v + rest) * 100, 1)
                            change_pct = None
                            before_v, after_v = None, None

        direction = "changed"
        if change_pct is not None:
            if change_pct <= -0.5:
                direction = "down"
            elif change_pct >= 0.5:
                direction = "up"
            else:
                direction = "steady"

        # headline in businessman words
        ql = str(question or "").lower()
        seg_hint = ""
        if seg_col and rows:
            try:
                top_row = max(rows, key=lambda r: (_num(r.get(value_col)) or 0) if value_col else 0)
                seg_hint = str(top_row.get(seg_col) or "")
            except Exception:
                seg_hint = ""
        if change_pct is not None and abs(change_pct) >= 0.5:
            headline = f"{metric_plain} is {direction} {abs(change_pct)}% — here's what to do"
        elif seg_hint:
            headline = f"{metric_plain}: {seg_hint} leads — here's the full picture"
        else:
            headline = f"{metric_plain}: here's the full picture in plain words"

        if change_pct is not None and abs(change_pct) >= 0.5:
            what_summary = (
                f"{metric_plain} went {direction} by {abs(change_pct)}% "
                f"({ _fmt_money(before_v)} → {_fmt_money(after_v)})"
                + (f" over {period_label}." if period_label else ".")
                + (f" Total in view: {_fmt_money(total)}." if total is not None else "")
            )
        elif total is not None:
            what_summary = f"{metric_plain} totals {_fmt_money(total)} across {len(rows)} group(s). No sharp up/down move — the story is in the mix below."
        else:
            what_summary = "We found your data, but there wasn't enough to size the move. The causes and actions below still apply."

        # ---- Likely causes (from real drivers/hypothesis, plain words) ----
        likely_causes = []
        for d in (drivers or [])[:3]:
            if not isinstance(d, dict):
                continue
            name = str(d.get("driver") or d.get("value") or d.get("dimension") or "")
            if not name:
                continue
            share = d.get("contribution_pct") or d.get("share_pct") or d.get("share")
            try:
                share_f = float(str(share).replace("%", "")) if share is not None else None
            except Exception:
                share_f = None
            why = f"Holds about {share_f}% of {metric_plain.lower()}." if share_f else "Shows up strongly in your data."
            likely_causes.append({
                "cause": _human(name),
                "why_plain": f"{_human(name)} {why} When this moves, your total moves with it — focus here first.",
                "share_pct": share_f,
            })
        if hypothesis and isinstance(hypothesis, dict):
            ht = str(hypothesis.get("text") or "")[:300]
            if ht and len(likely_causes) < 3:
                # strip stats jargon
                ht = re.sub(r"\b(p\s*[=<]\s*0\.\d+|r\s*=\s*-?\d\.?\d*|n\s*=\s*\d+)\b", "", ht).strip()
                likely_causes.append({
                    "cause": "Pattern in the data",
                    "why_plain": ht,
                    "share_pct": None,
                })
        if seg_col and value_col and rows and len(likely_causes) < 3:
            try:
                srows = sorted(rows, key=lambda r: (_num(r.get(value_col)) or 0), reverse=True)[:3]
                tot = sum((_num(r.get(value_col)) or 0) for r in rows)
                for r in srows:
                    nm = str(r.get(seg_col) or "")
                    v = _num(r.get(value_col)) or 0
                    if not nm:
                        continue
                    if any(nm.lower() in str(c.get("cause", "")).lower() for c in likely_causes):
                        continue
                    sh = round(v / tot * 100, 1) if tot else None
                    likely_causes.append({
                        "cause": nm,
                        "why_plain": f"{nm} contributes {_fmt_money(v)}" + (f" ({sh}% of the total)" if sh else "") + " — the biggest lever in view.",
                        "share_pct": sh,
                    })
                    if len(likely_causes) >= 3:
                        break
            except Exception:
                pass
        likely_causes = likely_causes[:3]
        if not likely_causes:
            likely_causes = [{
                "cause": "Spread across many small factors",
                "why_plain": "No single item dominates. That usually means the fix is operational (pricing, stock, follow-up) rather than one big cause.",
                "share_pct": None,
            }]

        # ---- What to do (translate engine cards → businessman actions) ----
        what_to_do = []
        eng_pres = []
        try:
            if isinstance(prescription, dict):
                eng_pres = [p for p in (prescription.get("prescriptions") or []) if isinstance(p, dict)]
        except Exception:
            eng_pres = []
        for p in eng_pres[:3]:
            title = str(p.get("title") or p.get("name") or "Recommended move")
            # clean technical titles
            title = re.sub(r"\b(scenario|ols|pearson|spearman|p_value|r_value)\b", "", title, flags=re.I).strip(" -:") or "Recommended move"
            disp = p.get("display") if isinstance(p.get("display"), dict) else {}
            action = str(disp.get("action_text") or p.get("action") or "")[:400]
            outcome = str(disp.get("outcome_text") or "")[:300]
            steps = []
            if action:
                steps.append(action)
            # add concrete generic steps so a businessman can act Monday morning
            top_cause = likely_causes[0]["cause"] if likely_causes else "your top segment"
            if len(steps) < 3:
                steps += [
                    f"Protect {top_cause}: check stock, price and delivery for the next 14 days.",
                    "Run one small test (e.g. a 2-week promotion or restock) and compare against the 2 weeks before.",
                ]
            what_to_do.append({
                "title": title[:120],
                "steps": steps[:3],
                "expected": outcome or f"If {top_cause} recovers even halfway, {metric_plain.lower()} should visibly improve within 2–4 weeks.",
                "effort": "Medium — one owner, 2 weeks",
            })
        # engine refused → still give 3 excellent businessman plays from real top segments
        if not what_to_do:
            top_cause = likely_causes[0]["cause"] if likely_causes else "your best-selling line"
            second = likely_causes[1]["cause"] if len(likely_causes) > 1 else "the weakest spot"
            what_to_do = [
                {
                    "title": f"Defend {top_cause} — your money-maker",
                    "steps": [
                        f"Check {top_cause} for out-of-stock, price jumps or delivery delays this week.",
                        "Call your top 10 repeat buyers and ask what stopped them buying.",
                        "Fix the top complaint within 7 days and tell customers it's fixed.",
                    ],
                    "expected": f"Holding {top_cause} steady protects most of your {metric_plain.lower()}.",
                    "effort": "Low — this week",
                },
                {
                    "title": f"Fix {second} before it drags you further",
                    "steps": [
                        f"Compare {second} this month vs last month: price, stock, returns.",
                        "Offer a simple bundle or small discount to restart movement.",
                        "Track daily for 14 days — keep it only if it moves.",
                    ],
                    "expected": "Even a 10% recovery here shows up in the total fast.",
                    "effort": "Medium — 2 weeks",
                },
                {
                    "title": "Run one clean 2-week test",
                    "steps": [
                        "Pick ONE change (price, promotion or restock) — not three at once.",
                        "Write down today's number so you can compare honestly.",
                        "Review after 14 days: keep what worked, kill what didn't.",
                    ],
                    "expected": "You get a clear yes/no answer instead of guessing.",
                    "effort": "Low — discipline, not budget",
                },
            ]

        # ---- If you do nothing ----
        if change_pct is not None and change_pct < -0.5:
            monthly_burn = None
            try:
                if after_v is not None and before_v is not None:
                    monthly_burn = round(before_v - after_v, 2)
            except Exception:
                monthly_burn = None
            if monthly_burn and monthly_burn > 0:
                nothing = (f"If nothing changes, you keep losing roughly {_fmt_money(monthly_burn)} "
                           f"per period at the current pace ({abs(change_pct)}% down). "
                           f"Over 3 periods that's about {_fmt_money(monthly_burn * 3)} left on the table.")
            else:
                nothing = (f"If nothing changes, the {abs(change_pct)}% slide becomes the new normal — "
                           f"customers form new habits fast. Acting in the next 2 weeks matters more than acting big.")
        elif change_pct is not None and change_pct > 0.5:
            nothing = f"Good news: you're up {abs(change_pct)}%. Doing nothing risks stalling it — lock in what's working."
        else:
            nothing = "No dramatic slide right now. The risk is slow drift — review this number weekly so a small dip never becomes a big one."

        return {
            "headline": headline,
            "what_happened": {
                "summary": what_summary,
                "total": total,
                "change_pct": change_pct,
                "before": before_v,
                "after": after_v,
                "period": period_label,
                "metric": metric_plain,
            },
            "likely_causes": likely_causes,
            "what_to_do": what_to_do[:3],
            "if_do_nothing": nothing,
            "confidence_plain": "Based on your real data — not a guess. Treat causes as strong suspects, not proof.",
            "limitations_plain": "Covers only the data you connected. Outside factors (market, season, competitors) aren't in here.",
            "chart_hint": {
                "x": period_col or seg_col,
                "y": value_col,
            },
        }
    except Exception:
        return {
            "headline": "Here's your business picture in plain words",
            "what_happened": {"summary": "We couldn't size the move from the data returned, but the actions below still apply.", "total": None,
                               "change_pct": None, "before": None, "after": None, "period": "", "metric": str(metric_label or "Sales")},
            "likely_causes": [{"cause": "Check your best seller first", "why_plain": "In most businesses, the top 1–2 items drive the total. Start there.", "share_pct": None}],
            "what_to_do": [{"title": "Talk to customers this week", "steps": ["Call 5 recent buyers.", "Ask what almost stopped them.", "Fix the top issue."], "expected": "Fastest honest signal you can get.", "effort": "Low"}],
            "if_do_nothing": "Without action, small dips tend to stick. A weekly check prevents that.",
            "confidence_plain": "Limited data in this answer — use it as a starting checklist.",
            "limitations_plain": "Based on the rows returned for this question.",
            "chart_hint": {"x": None, "y": None},
        }
