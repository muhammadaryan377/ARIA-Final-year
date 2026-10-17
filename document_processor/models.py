"""
Document Processor Data Models
==============================
Core data structures for PDF document processing.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ProcessingStatus(str, Enum):
    """Document processing lifecycle states."""
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    VALIDATING = "validating"
    DETECTING = "detecting"
    EXTRACTING = "extracting"
    OCR = "ocr"
    HANDWRITING = "handwriting"
    UNDERSTANDING = "understanding"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"
    PARTIAL = "partial"
    DELETING = "deleting"
    DELETED = "deleted"


class DocumentType(str, Enum):
    """Detected document type."""
    DIGITAL = "digital"
    SCANNED = "scanned"
    HANDWRITTEN = "handwritten"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ElementType(str, Enum):
    """Extracted element types."""
    TEXT = "text"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    TABLE_CELL = "table_cell"
    IMAGE = "image"
    HANDWRITING = "handwriting"
    OCR_TEXT = "ocr_text"
    FIGURE = "figure"


@dataclass
class Document:
    """Document metadata and processing state."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    original_name: str = ""
    file_path: str = ""
    file_size: int = 0
    mime_type: str = "application/pdf"
    page_count: int = 0
    document_type: DocumentType = DocumentType.UNKNOWN
    status: ProcessingStatus = ProcessingStatus.UPLOADING
    user_id: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    processing_method: Optional[str] = None
    sha256: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Processing results
    elements: List[DocumentElement] = field(default_factory=list)
    chunks: List[DocumentChunk] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "original_name": self.original_name,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "page_count": self.page_count,
            "document_type": self.document_type.value,
            "status": self.status.value,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
            "processing_method": self.processing_method,
            "sha256": self.sha256,
            "metadata": self.metadata,
            "element_count": len(self.elements),
            "chunk_count": len(self.chunks),
            "table_count": len(self.tables),
        }


@dataclass
class DocumentElement:
    """A single extracted element from a document page."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    page_number: int = 0
    element_type: ElementType = ElementType.TEXT
    content: str = ""
    section: Optional[str] = None
    source_type: str = "digital"  # digital, ocr, handwriting, vision
    confidence: float = 1.0
    bounding_box: Optional[Dict[str, float]] = None  # x0, y0, x1, y1 normalized 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # For tables
    table_data: Optional[Dict[str, Any]] = None  # headers, rows, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "element_type": self.element_type.value,
            "content": self.content,
            "section": self.section,
            "source_type": self.source_type,
            "confidence": self.confidence,
            "bounding_box": self.bounding_box,
            "metadata": self.metadata,
            "table_data": self.table_data,
        }


@dataclass
class DocumentChunk:
    """A chunk of document content for embedding and retrieval."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    page_number: int = 0
    section: Optional[str] = None
    content: str = ""
    embedding: Optional[List[float]] = None
    source_type: str = "digital"
    element_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "section": self.section,
            "content": self.content,
            "source_type": self.source_type,
            "element_ids": self.element_ids,
            "metadata": self.metadata,
            "token_count": self.token_count,
            "has_embedding": self.embedding is not None,
        }


@dataclass
class ProcessingResult:
    """Result of document processing."""
    success: bool
    document: Optional[Document] = None
    error: Optional[str] = None
    failed_pages: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class PageProcessingResult:
    """Result of processing a single page."""
    page_number: int
    success: bool
    elements: List[DocumentElement] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    document_type: DocumentType = DocumentType.UNKNOWN
    confidence: float = 1.0
    error: Optional[str] = None
    processing_method: str = "digital"