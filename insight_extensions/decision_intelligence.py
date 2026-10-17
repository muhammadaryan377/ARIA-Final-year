"""Goal-aware decision intelligence for ARIA.

Canonical implementation lives at ``insight_agent.extensions.decision_intelligence``.
This module re-exports it for backward compatibility.

.. deprecated::
    Import from ``insight_agent.extensions.decision_intelligence`` instead.
"""
import warnings

warnings.warn(
    "insight_extensions.decision_intelligence is deprecated; use "
    "'from insight_agent.extensions.decision_intelligence import DecisionIntelligence' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from insight_agent.extensions.decision_intelligence import DecisionIntelligence  # noqa: F401,E402

__all__ = ["DecisionIntelligence"]
