"""Next-step question builder (hyp.txt sections 15-16).

Dynamic follow-ups from actual schema findings — never hardcoded
products/regions. Also builds confirm/disprove checklists so users see
what ARIA knows and doesn't know.
"""

from __future__ import annotations


def follow_ups(hyp: dict, dimensions: list[str] | None = None,
               metric: str | None = None, limit: int = 4) -> list[str]:
    dims = [str(d) for d in (dimensions or []) if str(d).strip()][:6]
    m = (metric or hyp.get("metric") or "the metric")
    m = str(m).replace("_", " ")
    qs: list[str] = []
    if dims:
        qs.append(f"Which {dims[0].replace('_', ' ')} drove the change in {m} most?")
        if len(dims) > 1:
            qs.append(f"Was the change concentrated in one {dims[1].replace('_', ' ')}?")
    else:
        title = str(hyp.get("title") or "")
        if title:
            qs.append(f"What segments explain: {title[:80]}?")
    qs.append("Did volume fall or did value per unit change?")
    qs.append("Did the pattern appear before or after related changes?")
    seen, out = set(), []
    for q in qs:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out[:limit]


def confirm_disprove(hyp: dict) -> dict:
    needed = list(hyp.get("needed_data") or [])[:3]
    against = list(hyp.get("contradictory") or [])[:2]
    strengthen = needed or ["Finer segment history to confirm the pattern holds."]
    weaken = against or ["A stable control segment would weaken this explanation."]
    return {"strengthen": strengthen, "weaken": weaken}
