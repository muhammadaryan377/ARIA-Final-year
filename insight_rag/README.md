# Agentic document handling

Scope: text-based PDFs and extracted tables. OCR/handwriting is deferred. Scanned
PDFs are rejected with an actionable message; no legacy OCR processor is invoked
by upload or index-repair routes.

## Responsibilities

- `ingestion.py`: owned PDF validation, duplicate detection, extraction/index commit,
  rollback and repair from saved chunks. Does not require an LLM.
- `pdf_ingest.py`: page text and table extraction, stable chunk IDs and page metadata.
- `providers.py`: saved provider/key/model resolution and provider-specific RAG roles.
- `storage.py`, `embeddings.py`: persistent user-scoped metadata, history and PGVector.
- `service.py`, `retrieval.py`, `reranker.py`: hybrid retrieval and evidence context.
- `semantic_router.py`, `scope_layer.py`: LLM intent/scope decisions and bounded calls.
- `context_engine.py`, `conversation_layer.py`: current-conversation context and follow-ups.
- `friendly.py`, `grounding.py`, `table_reasoner.py`: evidence selection, table calculations,
  answer generation and support checks.
- `final_layer.py`: final citation/presentation checks.

Both `/api/connect_pdf` and `/api/insight/pdf/upload` call PDFIngestion. Successful
upload means canonical indexing completed. Tables remain document evidence for
RAG/table reasoning; upload no longer switches the session into CSV/SQL analytics.
Full CSV analytics remains a separate source workflow.

## Agent behaviour

The existing semantic router selects document scope, task, page restrictions,
query decomposition and previous-answer actions. Retrieval/reranking gathers
bounded evidence, table tools calculate supported values, and grounding checks
answers before returning citations. User instructions cannot broaden ownership.
History is persisted per user/conversation. This is bounded agentic RAG, not an
unlimited research loop or a guarantee of factual accuracy.

## Performance

One canonical index replaces duplicate FAISS processing plus metadata mirroring.
Repeated identical uploads reuse their indexed record. Embedding/reranker model
instances and vector clients already have process-level caches. Index repair uses
saved chunks without re-extracting PDFs or calling a chat model. LLM call/context
budgets remain enforced; verification is not skipped to make latency look better.
No live latency benchmark was run for this change.

## Setup and migration

Set ARIA_RAG_DATABASE_URL to a PostgreSQL database with pgvector and install the
repository RAG dependencies. Hosted chat requires cloud consent and a configured
provider key/model. Restart the backend after pulling; rebuild the frontend.

Old PDFs uploaded through the legacy path may only have saved text, with no
PGVector embeddings. POST `/api/documents/{document_id}/retry` rebuilds the owned
canonical index from those saved chunks. If chunks are missing, re-upload the
original text PDF. Existing history is retained. Do not delete old data to migrate.

## Verification

Run:

    python -m pytest tests/test_document_provider_contract.py tests/test_insight_rag_conversation_layer.py tests/test_insight_rag_reliability.py tests/test_rag_cloud_provider_stability.py tests/test_rag_single_pdf_routing_retry.py tests/test_insight_rag_table_reasoner.py -q

Manual live gate: upload a text PDF; inspect page/chunk counts; ask a cited fact;
ask a follow-up in the same chat; request a missing fact; compare two owned PDFs;
repeat the upload; interrupt database availability and confirm an actionable
error. Check the returned model matches the selected provider/model. Scanned PDFs
must fail clearly. Live provider/PGVector integration remains to be verified.
