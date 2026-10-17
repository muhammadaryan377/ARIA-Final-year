"""Compatibility shim: canonical implementation lives in insight_agent.advanced.

Preserved so existing imports (`from insight_agent_advanced import ...`,
`from insight_agent_industry import ...`) keep working. Do not add logic here.

.. deprecated::
    Import from ``insight_agent.advanced`` (or ``insight_agent``) instead.
"""
import warnings

warnings.warn(
    "insight_agent_advanced is deprecated; use 'from insight_agent.advanced import AdvancedInsightAgent' "
    "or 'from insight_agent import AdvancedInsightAgent' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from insight_agent import sanitize_json  # noqa: F401,E402
from insight_agent.advanced import AdvancedInsightAgent  # noqa: F401,E402

InsightAgent = AdvancedInsightAgent

__all__ = ["AdvancedInsightAgent", "InsightAgent", "sanitize_json"]
