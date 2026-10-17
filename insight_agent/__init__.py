"""ARIA Insight Agent package.

Public API (canonical)::

    from insight_agent import InsightAgent, AdvancedInsightAgent, IndustryInsightAgent
    from insight_agent import sanitize_json

Canonical chain:
    insight_agent.agent.InsightAgent (base, VERSION 2.0, zscore)
    -> insight_agent.advanced.AdvancedInsightAgent (VERSION 2.1, iqr)
    -> insight_agent.industry.IndustryInsightAgent (VERSION 3.0, iqr)
"""

from .agent import InsightAgent, sanitize_json
from .advanced import AdvancedInsightAgent
from .industry import IndustryInsightAgent
from .chart_catalog import CHART_OPTIONS, compatible_chart_options

__all__ = [
    "InsightAgent",
    "AdvancedInsightAgent",
    "IndustryInsightAgent",
    "sanitize_json",
    "CHART_OPTIONS",
    "compatible_chart_options",
]
