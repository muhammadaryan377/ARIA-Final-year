"""Hypothesis engine coverage contribution (PHASE 15, coverage only).

Reports which candidate classes the hypothesis layer actually examined.
No hardcodes: callers pass the real investigation results in.
"""
from __future__ import annotations


def hypothesis_coverage(investigations: list) -> dict:
    """Summarize examined hypothesis classes from actual results.

    investigations: list of {analysis_type, metric, dimension, verdict}.
    Returns {candidate_classes, metrics, dimensions, signals} — all computed.
    """
    rows = [r for r in (investigations or []) if isinstance(r, dict)]
    return {
        "candidate_classes_examined": sorted({str(r.get("analysis_type") or "unknown") for r in rows}),
        "metrics_examined": sorted({str(r.get("metric")) for r in rows if r.get("metric")}),
        "dimensions_examined": sorted({str(r.get("dimension")) for r in rows if r.get("dimension")}),
        "signals": len(rows),
    }
