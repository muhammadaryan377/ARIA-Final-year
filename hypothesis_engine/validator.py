"""Hypothesis validator (hyp.txt sections 7-8).

Splits measured signals into supporting / contradictory / unknown and
adds temporal-consistency + sufficiency notes. Deterministic; the LLM
never validates — it only phrases what this module decided.
"""

from __future__ import annotations


def split_evidence(hyp: dict) -> dict:
    supporting = list(hyp.get("supporting") or [])
    against = list(hyp.get("contradictory") or [])
    unknown = list(hyp.get("limitations") or []) + list(hyp.get("needed_data") or [])
    return {"supporting": supporting[:6], "contradictory": against[:4],
            "unknown": unknown[:4],
            "n_support": len(supporting), "n_against": len(against)}


def temporal_note(hyp: dict) -> str | None:
    """Worth-investigating timing language; never claims causation."""
    text = " ".join([str(hyp.get("explanation") or ""), str(hyp.get("title") or "")]).lower()
    if any(k in text for k in ("before", "preced", "earlier", "prior", "lead", "lag")):
        return ("The timing is worth investigating as a possible contributing factor; "
                "the data does not establish causation.")
    return None
