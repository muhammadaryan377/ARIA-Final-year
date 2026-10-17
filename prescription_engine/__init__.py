"""Prescription Engine: evidence-gated action recommendations.

Pipeline: evidence -> problem -> driver -> controllability -> candidates ->
scenarios -> rank. Numbers always come from deterministic statistics;
an LLM may only rephrase wording at the routes layer.
"""
from .engine import (analyze, run_scenario, classify_driver, improvement_up, analyze_document,
                     classify_semantic_role, assess_actionability, NO_SUPPORTED_SCENARIO,
                     LEVER_ROLES, NON_LEVER_ROLES, NON_OUTCOME_ROLES,
                     dedupe_key, dedupe_prescriptions,
                     recommendation_for_decision, scenario_supported_range)

__all__ = ["analyze", "run_scenario", "classify_driver", "improvement_up", "analyze_document",
           "classify_semantic_role", "assess_actionability", "NO_SUPPORTED_SCENARIO",
           "LEVER_ROLES", "NON_LEVER_ROLES", "NON_OUTCOME_ROLES",
           "dedupe_key", "dedupe_prescriptions",
           "recommendation_for_decision", "scenario_supported_range"]
