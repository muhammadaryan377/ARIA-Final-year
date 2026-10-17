"""PDF RAG capability used by ARIA's existing Insight Agent.

This module extends Insight Agent with conversational document intelligence; it
is intentionally not a fifth autonomous agent.

Patch order (P1 cleanup note — do not reorder without updating
provider_stability/answer_stability/verification_stability/summary_retry):
  1. provider_stability (null->[] for target_pages/document_keys/
     table_operations + scope aliases; explicit helper:
     ``provider_stability.normalize_null_arrays`` — prefer calling it
     explicitly from ``llm_provider.chat_structured`` consumers over new
     import-time patches),
  2. answer_stability, 3. verification_stability, 4. summary_retry.
Import-time patches remain for backward compat so the pipeline never breaks.
Query diagnostics always carry ``RAG_PIPELINE_VERSION`` (see config.py /
scope_layer._decorate_result / TRACE context).
"""

from .provider_stability import apply_provider_stability_patches, normalize_null_arrays

# Apply cloud-provider output compatibility before the public RAG class imports
# the semantic router/provider runtime.
apply_provider_stability_patches()

from .scope_layer import InsightPDFRAG
from .answer_stability import apply_answer_stability_patches
from .verification_stability import apply_verification_stability_patches
from .summary_retry import apply_summary_retry_patch

# Broad summaries use source-bound generation, while both evidence sufficiency
# and final answer auditing use provider-stable binary verdicts. A single bounded
# retry handles rare hosted-model false negatives without bypassing verification.
apply_answer_stability_patches()
apply_verification_stability_patches()
apply_summary_retry_patch()

__all__ = ["InsightPDFRAG", "normalize_null_arrays"]
