"""
ARIA Document Processor
=======================
PDF document intelligence system supporting:
- Digital/text PDFs
- Scanned/image PDFs (OCR)
- Handwritten PDFs (vision)
- Mixed documents (page-level processing)
- Table extraction
- Chunking with metadata
- Embeddings and vector indexing
"""

from .processor import DocumentProcessor, create_document_processor
from .models import Document, DocumentElement, DocumentChunk, ProcessingStatus
from .detection import DocumentTypeDetector
from .extraction import DigitalExtractor
from .scanned_extraction import ScannedExtractor, HybridExtractor
from .handwriting_extraction import HandwritingExtractor
from .tables import TableExtractor
from .chunking import DocumentChunker
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore, create_vector_store

__all__ = [
    "DocumentProcessor",
    "create_document_processor",
    "Document",
    "DocumentElement",
    "DocumentChunk",
    "ProcessingStatus",
    "DocumentTypeDetector",
    "DigitalExtractor",
    "ScannedExtractor",
    "HybridExtractor",
    "HandwritingExtractor",
    "TableExtractor",
    "DocumentChunker",
    "EmbeddingGenerator",
    "VectorStore",
    "create_vector_store",
]