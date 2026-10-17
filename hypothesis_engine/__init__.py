"""Hypothesis Engine: question -> intent -> adequacy -> analysis -> verdict.

LLM never decides the statistical result. The engine (pure Python statistics)
understands the question, inspects the schema/data, selects the analysis,
calculates the evidence and renders the verdict. An LLM may later be used only
to rephrase the explanation — the numbers always come from here.
"""
from .engine import analyze_question, suggest_hypotheses, preview_question, data_context

__all__ = ["analyze_question", "suggest_hypotheses", "preview_question", "data_context"]
