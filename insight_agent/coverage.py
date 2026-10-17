"""Insight agent coverage contribution (coverage only).

Reports which finding classes the insight layer actually examined.
No hardcodes: callers pass the real insight findings in.
"""
from __future__ import annotations


def insight_coverage(findings: list) -> dict:
    rows = [r for r in (findings or []) if isinstance(r, dict)]
    return {
        "candidate_classes_examined": sorted({str(r.get("kind") or r.get("type") or "finding") for r in rows}),
        "metrics_examined": sorted({str(r.get("metric") or r.get("column")) for r in rows if r.get("metric") or r.get("column")}),
        "dimensions_examined": sorted({str(r.get("dimension") or r.get("segment")) for r in rows if r.get("dimension") or r.get("segment")}),
        "signals": len(rows),
    }
