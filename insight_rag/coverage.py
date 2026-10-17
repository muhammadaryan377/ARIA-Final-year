"""Insight RAG coverage contribution (coverage only).

Reports which evidence signals the RAG layer actually examined for a
decision analysis run. No hardcodes: inputs are the real retrieval traces.
"""
from __future__ import annotations


def rag_coverage(retrievals: list) -> dict:
    """Summarize examined RAG signals from actual retrieval traces.

    retrievals: list of {source, metric, dimension, accepted: bool, reason}.
    """
    rows = [r for r in (retrievals or []) if isinstance(r, dict)]
    rejected: dict[str, int] = {}
    accepted = 0
    for r in rows:
        if r.get("accepted"):
            accepted += 1
        else:
            reason = str(r.get("reason") or "insufficient_evidence")
            rejected[reason] = rejected.get(reason, 0) + 1
    return {
        "candidate_classes_examined": sorted({str(r.get("source") or "rag") for r in rows}),
        "metrics_examined": sorted({str(r.get("metric")) for r in rows if r.get("metric")}),
        "dimensions_examined": sorted({str(r.get("dimension")) for r in rows if r.get("dimension")}),
        "signals": len(rows),
        "accepted": accepted,
        "rejected_by_reason": rejected,
    }
