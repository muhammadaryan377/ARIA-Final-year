"""PDF-RAG endpoints for ARIA's existing Industry Insight Agent.

RAG is an Insight Agent capability, not a fifth autonomous agent.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from core.deps import require_writable
from insight_agent.industry import IndustryInsightAgent as InsightAgent
from insight_rag import InsightPDFRAG
from insight_rag.config import (
    MAX_PDF_MB,
    RAG_DATABASE_URL,
    RAG_LLM_MODEL,
    RAG_PIPELINE_VERSION,
    validate_rag_config,
)
from insight_rag.storage import RAGMetadataStore, UserPGVectorStore
from llm_provider import create_provider

router = APIRouter(prefix="/api/insight/pdf", tags=["Insight PDF RAG"])


class PDFChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=80)
    document_ids: list[str] | None = Field(default=None, max_length=20)


def _capability(user_id: str | int) -> InsightPDFRAG:
    """Attach PDF-RAG to the session's selected provider (Cloud = user's Groq key, Local = Ollama fine-tunes).

    Falls back to DeepSeek only when the session has no usable provider.
    RAG chat roles (rag/rag_rewrite/...) are mapped onto the active
    provider's own models so Cloud/Local never receive another
    provider's model names.
    """
    from core.config import get_session, resolve_provider_key
    _session = get_session(int(user_id))
    _pname = (_session.get("provider_name") or "local")
    if _pname == "local" and not _session.get("provider_name"):
        logging.getLogger("aria.insight_rag").warning(
            "provider_name missing from session – falling back to local"
        )
    if _pname not in ("local", "cloud", "deepseek", "omniroute", "grok"):
        raise HTTPException(status_code=503, detail="Selected AI provider is not supported.")
    if _pname == "deepseek":
        provider = create_provider(
            "deepseek",
            api_key=resolve_provider_key("deepseek", _session.get("api_keys") or {}),
            models={
                "rag": RAG_LLM_MODEL,
                "rag_scope": RAG_LLM_MODEL,
                "rag_rewrite": RAG_LLM_MODEL,
                "rag_verify": RAG_LLM_MODEL,
                "rag_plan": RAG_LLM_MODEL,
            },
        )
    elif _pname == "local":
        from llm_provider import PROVIDERS as _PROVIDERS
        _chat_model = (_PROVIDERS["local"]["models"].get("semi")
                       or _PROVIDERS["local"]["models"].get("suggest")
                       or "aria-goal")
        provider = create_provider("local")
        provider.models.update({
            "rag": _chat_model,
            "rag_scope": _chat_model,
            "rag_rewrite": _chat_model,
            "rag_verify": _chat_model,
            "rag_plan": _chat_model,
        })
    else:
        from llm_provider import PROVIDERS as _PROVIDERS
        _key = resolve_provider_key(_pname, (_session.get("api_keys") or {}))
        provider = create_provider(_pname, api_key=_key)
        _fallback = provider.model_for("sql") or provider.model_for("story")
        if _fallback:
            provider.models.update({
                "rag": _fallback,
                "rag_scope": _fallback,
                "rag_rewrite": _fallback,
                "rag_verify": _fallback,
                "rag_plan": _fallback,
            })
    insight_agent = InsightAgent(provider=provider)
    return InsightPDFRAG(insight_agent=insight_agent, user_id=user_id)


@router.get("/health")
def pdf_rag_health(user: dict = Depends(require_writable)):
    """Authenticated readiness/quality snapshot without exposing secrets or PDF text."""
    config_error = None
    try:
        validate_rag_config()
    except RuntimeError as exc:
        config_error = str(exc)

    documents = RAGMetadataStore(user["user_id"]).list_documents()
    quality_counts = Counter(
        str((document.get("ingestion_quality") or {}).get("grade") or "unknown")
        for document in documents
    )
    from core.config import get_session, resolve_provider_key
    session = get_session(user["user_id"])
    selected_provider = session.get("provider_name") or "local"
    provider_configured = (selected_provider == "local" or bool(
        resolve_provider_key(selected_provider, session.get("api_keys") or {})))
    provider_configured = provider_configured and not bool(session.get("_provider_error"))
    database_configured = bool(RAG_DATABASE_URL)
    ready = not config_error and provider_configured and database_configured
    return {
        "ok": True,
        "ready": ready,
        "status": "ready" if ready else "degraded",
        "pipeline_version": RAG_PIPELINE_VERSION,
        "model": RAG_LLM_MODEL if selected_provider == "deepseek" else None,
        "provider_name": selected_provider,
        "provider_configured": provider_configured,
        "database_configured": database_configured,
        "config_valid": config_error is None,
        "config_error": config_error,
        "documents": len(documents),
        "ingestion_quality": dict(quality_counts),
    }


@router.post("/upload")
def upload_pdf(
    file: UploadFile = File(...),
    user: dict = Depends(require_writable),
):
    filename = Path(file.filename or "document.pdf").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported right now.")

    temp_path = None
    max_bytes = MAX_PDF_MB * 1024 * 1024
    written = 0
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            temp_path = Path(temp.name)
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"PDF exceeds the current {MAX_PDF_MB} MB limit.",
                    )
                temp.write(chunk)

        return _capability(user["user_id"]).ingest_pdf(
            temp_path,
            original_filename=filename,
        )
    except HTTPException:
        raise
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="PDF chat request or provider settings are invalid.")
    except RuntimeError:
        raise HTTPException(status_code=503, detail="PDF assistant is unavailable. Check its provider and storage settings.")
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="I couldn't index that PDF right now. Please check the RAG database/model connection and try again.",
        )
    finally:
        try:
            file.file.close()
        except Exception:
            pass
        if temp_path:
            temp_path.unlink(missing_ok=True)


@router.get("/documents")
def list_pdf_documents(user: dict = Depends(require_writable)):
    """List local PDF metadata without requiring a cloud LLM call."""
    documents = RAGMetadataStore(user["user_id"]).list_documents()
    public_docs = [
        {key: value for key, value in document.items() if key != "chunk_ids"}
        for document in documents
    ]
    return {"ok": True, "documents": public_docs, "pipeline_version": RAG_PIPELINE_VERSION}


@router.delete("/documents/{document_id}")
def delete_pdf_document(document_id: str, user: dict = Depends(require_writable)):
    """Delete owned PDF metadata + vector chunks without constructing an LLM."""
    metadata = RAGMetadataStore(user["user_id"])
    with metadata.mutation_lock():
        record = metadata.get_document(document_id)
        if not record:
            raise HTTPException(status_code=404, detail="Document not found.")
        try:
            UserPGVectorStore(user["user_id"]).delete_chunks(record.get("chunk_ids", []))
            metadata.remove_document(document_id)
            metadata.document_file_path(document_id).unlink(missing_ok=True)
            metadata.chunks_path(document_id).unlink(missing_ok=True)
            return {"ok": True, "document_id": document_id}
        except Exception:
            raise HTTPException(
                status_code=503,
                detail="I couldn't delete that PDF from the search index right now. Please try again.",
            )


@router.post("/chat")
def chat_with_pdfs(request: PDFChatRequest, user: dict = Depends(require_writable)):
    try:
        return _capability(user["user_id"]).chat(
            request.question,
            conversation_id=request.conversation_id,
            document_ids=request.document_ids,
        )
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="PDF chat request or provider settings are invalid.")
    except RuntimeError:
        raise HTTPException(status_code=503, detail="PDF assistant is unavailable. Check its provider and storage settings.")
    except Exception:
        # Never expose raw SDK/network/database exceptions to the user-facing chat.
        raise HTTPException(
            status_code=503,
            detail="I'm having trouble with the PDF assistant right now. Your PDFs are safe; please try again in a moment.",
        )
