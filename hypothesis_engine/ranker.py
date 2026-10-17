"""Deterministic hypothesis ranker (hyp.txt section 13).

Priority = evidence strength x contribution share x |effect|
           - contradictory penalty - uncertainty penalty.
No LLM ordering. Returns top 3-5 with explainable inputs.
"""

from __future__ import annotations

from .evidence_ladder import rank_key

_STRENGTH_W = {"Strong evidence": 1.0, "Moderate evidence": 0.65,
               "Limited evidence": 0.35, "Insufficient evidence": 0.1,
               "Contradictory evidence": 0.05}


def _effect(h: dict) -> float:
    for k in ("effect", "rel_change", "change_pct", "lift"):
        try:
            v = abs(float(h.get(k)))
            if v > 0:
                return min(v, 2.0)
        except (TypeError, ValueError):
            continue
    return 0.3


def score(h: dict, top_share: float = 0.0) -> tuple[float, dict]:
    s = h.get("strength") or h.get("evidence_strength") or "Limited evidence"
    w = _STRENGTH_W.get(s, 0.3)
    contrib = max(top_share, 0.0)
    eff = _effect(h)
    n_against = len(h.get("contradictory") or [])
    n_unknown = len(h.get("limitations") or []) + len(h.get("needed_data") or [])
    val = w * (0.5 + 0.5 * contrib) * (0.5 + 0.5 * eff) - 0.08 * n_against - 0.03 * n_unknown
    parts = {"evidence": round(w, 2), "contribution": round(contrib, 2),
             "effect": round(eff, 2), "against": n_against, "unknown": n_unknown}
    return round(max(0.0, val), 4), parts


def rank(hyps: list[dict], shares: dict[str, float] | None = None, top_n: int = 5) -> list[dict]:
    shares = shares or {}
    out = []
    for h in hyps or []:
        key = str(h.get("title") or h.get("id") or "")
        sc, parts = score(h, shares.get(key, 0.0))
        h2 = dict(h)
        h2["rank_score"] = sc
        h2["rank_inputs"] = parts
        out.append(h2)
    out.sort(key=lambda h: (-h["rank_score"], rank_key(h.get("strength") or "")))
    return out[: max(1, min(5, top_n))]
