"""schema_engine — normalized relational-schema inference pipeline.

Layers (single acceptance authority, RULE 6):
    adapters       -> normalized model (Phase B)
    candidates     -> CandidateGenerator (never accepts)
    evidence       -> EvidenceCollector (never accepts)
    classifier     -> Classifier (scores evidence)
    policy         -> AcceptancePolicy (the one acceptance authority)
    registry       -> RelationshipRegistry (retains REJECTED candidates)
    serializer     -> legacy-compatible output
"""

from schema_engine.adapters import ADAPTERS, DatabaseAdapter
from schema_engine.extraction import extract_database
from schema_engine.relationships import infer_relationships

try:  # Universal Data Intelligence Layer (single shared contract, n.txt P0).
    from core.data_intelligence import (
        build_data_intelligence,
        classify_column,
        detect_capabilities,
        profile_table,
    )
except ImportError:  # pragma: no cover - core always ships with schema_engine
    build_data_intelligence = classify_column = detect_capabilities = profile_table = None

__all__ = [
    "ADAPTERS", "DatabaseAdapter", "extract_database", "infer_relationships",
    "build_data_intelligence", "classify_column", "detect_capabilities",
    "profile_table",
]