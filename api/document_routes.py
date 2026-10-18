"""Document API routes: upload, list, search, retrieve, delete.

Consolidated to use the canonical ``insight_rag`` pipeline (PGVector + hybrid
RRF) instead of the legacy FAISS ``document_processor``.  The adapter class
below provides the thin interface these routes need (list / search / delete)
while delegating storage and retrieval to the enterprise RAG stack.

Upload and index repair both use insight_rag.ingestion. The legacy processor
is retained for compatibility but is not invoked by these document routes.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from core.config import DATA_DIR, PROCESSED_DIR, get_session
from core.deps import require_writable, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Canonical RAG adapter ─────────────────────────────────────────────
# Thin façade over insight_rag.storage so document_routes.py no longer
# creates a full DocumentProcessor (FAISS + MiniLM) just to list or search.
# The adapter is stateless per-request — it hits the shared PGVector store
# and the on-disk manifest that insight_rag already maintains.

class _CanonicalRAGAdapter:
    """Delegates list / search / delete to insight_rag storage primitives.

    This avoids instantiating the heavyweight ``InsightPDFRAG`` (which needs
    an InsightAgent + cloud LLM) while still routing all vector operations
    through the single canonical PGVector index.
    """

    def __init__(self, user_id: int | str):
        self.user_id = str(user_id)
        from insight_rag.storage import RAGMetadataStore, get_user_vector_store
        self._metadata = RAGMetadataStore(self.user_id)
        self._get_vector_store = get_user_vector_store

    # ── list ───────────────────────────────────────────────────────────
    def list_documents(self) -> list[dict]:
        """Return document records from the canonical manifest."""
        return self._metadata.list_documents()

    # ── search ─────────────────────────────────────────────────────────
    def search_documents(
        self,
        query: str,
        document_ids: list[str] | None = None,
        k: int = 10,
    ) -> list[dict]:
        """Hybrid retrieval across the canonical PGVector index."""
        try:
            store = self._get_vector_store(self.user_id)
            results = store.search(query, k=k, document_ids=document_ids)
        except Exception as exc:
            logger.warning("Canonical RAG search failed: %s", type(exc).__name__)
            raise HTTPException(status_code=503, detail="Document search is unavailable. Check the RAG index connection and retry.") from exc
        out: list[dict] = []
        for doc, score in results:
            meta = doc.metadata if hasattr(doc, "metadata") else {}
            out.append({
                "chunk_id": meta.get("chunk_id", ""),
                "document_id": meta.get("document_id", ""),
                "content": doc.page_content[:500] if hasattr(doc, "page_content") else "",
                "score": round(float(score), 6),
                "page_number": meta.get("page", meta.get("page_number")),
                "section": meta.get("section"),
                "content_type": meta.get("content_type", "text"),
                "filename": meta.get("filename"),
            })
        return out

    def search_single_document(
        self,
        query: str,
        document_id: str,
        k: int = 10,
    ) -> list[dict]:
        """Search within one specific document."""
        return self.search_documents(query, document_ids=[document_id], k=k)

    # ── delete ─────────────────────────────────────────────────────────
    def delete_document(self, document_id: str) -> bool:
        """Remove document from canonical storage."""
        try:
            with self._metadata.mutation_lock():
                record = self._metadata.get_document(document_id)
                if not record:
                    return False
                from insight_rag.storage import UserPGVectorStore
                UserPGVectorStore(self.user_id).delete_chunks(record.get("chunk_ids", []))
                self._metadata.remove_document(document_id)
                self._metadata.document_file_path(document_id).unlink(missing_ok=True)
                self._metadata.chunks_path(document_id).unlink(missing_ok=True)
                return True
        except Exception as exc:
            logger.error("Canonical RAG delete failed for %s: %s", document_id, exc)
            return False

_DOCUMENT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _sanitize_document_id(document_id: str) -> str:
    """Reject document IDs that could enable path traversal."""
    if not document_id or not _DOCUMENT_ID_RE.match(document_id):
        raise HTTPException(status_code=400, detail="Invalid document ID.")
    return document_id


class DocumentSearchRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    k: int = 10


class DocumentSearchResponse(BaseModel):
    ok: bool
    results: List[dict]


@router.get("/api/documents")
def list_documents(
    request: Request,
    user: dict = Depends(get_current_user),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    """List user's uploaded documents via canonical RAG manifest."""
    adapter = _CanonicalRAGAdapter(user["user_id"])
    documents = adapter.list_documents()
    if status:
        documents = [d for d in documents if d.get("status") == status]
    # Normalise keys so the frontend sees a consistent shape regardless of
    # which backend originally ingested the document.
    normalised = []
    for doc in documents:
        normalised.append({
            "id": doc.get("document_id"),
            "name": doc.get("filename"),
            "file_size": doc.get("size_bytes"),
            "page_count": doc.get("pages"),
            "document_type": doc.get("source_type", "pdf"),
            "status": doc.get("status", "ready"),
            "created_at": doc.get("created_at"),
            "completed_at": doc.get("created_at"),
            "element_count": doc.get("text_chars", 0),
            "chunk_count": doc.get("total_chunks", 0),
            "table_count": doc.get("tables", 0),
            "processing_method": doc.get("processing_method"),
            "error_message": doc.get("error_message"),
            "sha256": doc.get("sha256"),
            "pipeline_version": doc.get("rag_schema_version"),
        })
    return {"ok": True, "documents": normalised[:limit]}


@router.get("/api/documents/{document_id}")
def get_document(
    document_id: str,
    user: dict = Depends(get_current_user),
):
    """Get document details from canonical RAG manifest."""
    document_id = _sanitize_document_id(document_id)
    adapter = _CanonicalRAGAdapter(user["user_id"])
    from insight_rag.storage import RAGMetadataStore
    meta = RAGMetadataStore(user["user_id"])
    doc = meta.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"ok": True, "document": doc}


@router.post("/api/documents/{document_id}/search")
def search_document(
    document_id: str,
    request: DocumentSearchRequest,
    user: dict = Depends(get_current_user),
):
    """Search within a specific document using canonical RAG."""
    document_id = _sanitize_document_id(document_id)
    adapter = _CanonicalRAGAdapter(user["user_id"])
    from insight_rag.storage import RAGMetadataStore
    meta = RAGMetadataStore(user["user_id"])
    if not meta.get_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    results = adapter.search_single_document(
        query=request.query, document_id=document_id, k=request.k,
    )
    return {"ok": True, "results": results}


@router.post("/api/documents/search")
def search_all_documents(
    request: DocumentSearchRequest,
    user: dict = Depends(get_current_user),
):
    """Search across all user's documents using canonical RAG."""
    adapter = _CanonicalRAGAdapter(user["user_id"])
    results = adapter.search_documents(
        query=request.query,
        document_ids=request.document_ids,
        k=request.k,
    )
    return {"ok": True, "results": results}


@router.delete("/api/documents/{document_id}")
def delete_document(
    document_id: str,
    user: dict = Depends(require_writable),
):
    """Delete a document and its associated data via canonical RAG."""
    document_id = _sanitize_document_id(document_id)
    adapter = _CanonicalRAGAdapter(user["user_id"])
    from insight_rag.storage import RAGMetadataStore
    meta = RAGMetadataStore(user["user_id"])
    doc = meta.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    success = adapter.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete document")
    return {"ok": True, "message": "Document deleted"}


@router.post("/api/documents/{document_id}/retry")
def retry_document(
    document_id: str,
    user: dict = Depends(require_writable),
):
    """Repair the canonical index from owned saved text; never invoke OCR."""
    document_id = _sanitize_document_id(document_id)
    from insight_rag.ingestion import PDFIngestion
    try:
        return PDFIngestion(user["user_id"]).reindex(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Document index repair failed")
        raise HTTPException(status_code=503, detail="Document index repair failed. Check the RAG database and retry.") from exc
