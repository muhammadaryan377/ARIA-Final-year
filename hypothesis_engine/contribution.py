"""Contribution analysis (hyp.txt section 4).

Pure function: turns measured group breakdowns into $ + share-% tables
so Hypothesis can say "Electronics contributed 64% of the decline"
instead of "maybe customers buy less". No LLM, no invention.
"""

from __future__ import annotations


def build_table(groups: list[dict] | None, value_key: str = "value",
                label_key: str = "label", top_n: int = 6) -> dict:
    """groups: [{label, value}] measured from data. Returns ranked shares."""
    rows = []
    for g in (groups or [])[:top_n]:
        try:
            v = float(g.get(value_key))
        except (TypeError, ValueError):
            continue
        rows.append({"label": str(g.get(label_key, "?")), "value": v})
    if not rows:
        return {"rows": [], "total": 0.0}
    total = sum(r["value"] for r in rows)
    denom = abs(total) if total else 1.0
    for r in rows:
        r["share"] = round(abs(r["value"]) / denom, 4)
        r["share_pct"] = round(abs(r["value"]) / denom * 100, 1)
    rows.sort(key=lambda r: abs(r["value"]), reverse=True)
    return {"rows": rows, "total": round(total, 2),
            "top_share": rows[0]["share"] if rows else 0.0,
            "top_label": rows[0]["label"] if rows else None}
