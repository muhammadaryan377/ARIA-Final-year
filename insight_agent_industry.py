"""Compatibility shim: canonical implementation lives in insight_agent.industry.

Preserved so existing imports (`from insight_agent_industry import ...`) keep
working. Do not add logic here.

.. deprecated::
    Import from ``insight_agent.industry`` (or ``insight_agent``) instead.
"""
import warnings

warnings.warn(
    "insight_agent_industry is deprecated; use 'from insight_agent.industry import IndustryInsightAgent' "
    "or 'from insight_agent import IndustryInsightAgent' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from insight_agent import sanitize_json  # noqa: F401,E402
from insight_agent.industry import IndustryInsightAgent  # noqa: F401,E402

InsightAgent = IndustryInsightAgent

__all__ = ["IndustryInsightAgent", "InsightAgent", "sanitize_json"]
