"""
Main Document Processor
=======================
Orchestrates the complete document processing pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from .models import (
    Document, DocumentElement, DocumentChunk, ProcessingStatus, 
    DocumentType, ElementType, ProcessingResult, PageProcessingResult
)
from .detection import DocumentTypeDetector, detect_document_type
from .extraction import DigitalExtractor, extract_digital_pdf
from .scanned_extraction import ScannedExtractor, HybridExtractor, extract_hybrid_pdf, OCR_AVAILABLE
from .handwriting_extraction import HandwritingExtractor, extract_handwriting_pdf, VISION_AVAILABLE
from .tables import TableExtractor, extract_tables
from .chunking import DocumentChunker, ChunkConfig, chunk_document_elements
from .embeddings import EmbeddingGenerator, create_embedding_generator
from .vector_store import VectorStore, create_vector_store, SearchResult

logger = logging.getLogger(__name__)

# Heavy-ops guards (P1): PDF 50 MB / 500 pages — enforced here AND at the
# upload spool layer (api/connect_routes.py::_spool_limited -> 413).
# Files within limits but large are processed inline with a warning log
# (no job-queue refactor in P1).
MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_PDF_PAGES = 500
LARGE_INLINE_WARN_BYTES = 10 * 1024 * 1024
LARGE_INLINE_WARN_PAGES = 100


class DocumentProcessor:
    """
    Main document processing orchestrator.
    
    Pipeline:
    1. Validate PDF
    2. Detect document type (per page)
    3. Extract content (digital/OCR/vision per page)
    4. Extract tables
    5. Normalize elements
    6. Chunk with metadata
    7. Generate embeddings
    8. Index in vector store
    """
    
    def __init__(
        self,
        upload_dir: str | Path,
        processed_dir: str | Path,
        vector_store: Optional[VectorStore] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        llm_provider=None,
        chunk_config: Optional[ChunkConfig] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        self.upload_dir = Path(upload_dir)
        self.processed_dir = Path(processed_dir)
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.llm_provider = llm_provider
        self.chunk_config = chunk_config or ChunkConfig()
        self.progress_callback = progress_callback
        
        # Ensure directories exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding generator if not provided
        if self.embedding_generator is None:
            cache_dir = self.processed_dir / "embeddings_cache"
            self.embedding_generator = create_embedding_generator(
                model="minilm",
                cache_dir=str(cache_dir),
            )
        
        # Initialize vector store if not provided
        if self.vector_store is None:
            index_path = self.processed_dir / "vector_index.faiss"
            db_path = self.processed_dir / "vector_metadata.db"
            self.vector_store = create_vector_store(
                dimension=self.embedding_generator.get_dimension(),
                index_path=str(index_path),
                db_path=str(db_path),
            )
    
    def _report_progress(self, stage: str, progress: float) -> None:
        """Report progress via callback."""
        if self.progress_callback:
            self.progress_callback(stage, progress)
        logger.info(f"Progress: {stage} - {progress:.0%}")
    
    def process_document(
        self,
        file_path: str | Path,
        document_id: Optional[str] = None,
        user_id: int = 0,
        original_name: Optional[str] = None,
    ) -> ProcessingResult:
        """
        Process a document through the complete pipeline.
        
        Args:
            file_path: Path to PDF file
            document_id: Optional document ID (generated if not provided)
            user_id: User ID for ownership
            original_name: Original filename
        
        Returns:
            ProcessingResult with document and any errors
        """
        file_path = Path(file_path)
        document_id = document_id or str(uuid.uuid4())
        original_name = original_name or file_path.name
        
        # Create document record
        document = Document(
            id=document_id,
            name=document_id,
            original_name=original_name,
            file_path=str(file_path),
            file_size=file_path.stat().st_size,
            user_id=user_id,
            status=ProcessingStatus.VALIDATING,
        )
        
        try:
            # Stage 1: Validate
            self._report_progress("Validating PDF", 0.05)
            validation_error = self._validate_pdf(file_path)
            if validation_error:
                document.status = ProcessingStatus.FAILED
                document.error_message = validation_error
                return ProcessingResult(success=False, document=document, error=validation_error)
            
            # Compute SHA256
            document.sha256 = self._compute_sha256(file_path)
            
            # Stage 2: Detect document type
            self._report_progress("Detecting document type", 0.1)
            doc_type, page_analyses, summary = detect_document_type(file_path)
            document.document_type = DocumentType(doc_type)
            document.page_count = summary.get("total_pages", 0)
            document.metadata["detection"] = summary
            document.metadata["page_types"] = [p.value for p in page_analyses] if hasattr(page_analyses[0], 'value') else page_analyses
            
            # Stage 3: Extract content (per page based on type)
            self._report_progress("Extracting content", 0.2)
            all_elements, all_tables, failed_pages = self._extract_content(
                file_path, document_id, doc_type, page_analyses
            )
            
            document.elements = all_elements
            document.tables = all_tables
            
            if failed_pages:
                document.status = ProcessingStatus.PARTIAL
                document.metadata["failed_pages"] = failed_pages
            else:
                document.status = ProcessingStatus.EXTRACTING
            
            # Stage 4: Chunk elements
            self._report_progress("Chunking document", 0.5)
            chunks = self._chunk_document(all_elements, document_id)
            document.chunks = chunks
            
            # Stage 5: Generate embeddings
            self._report_progress("Generating embeddings", 0.7)
            chunks_with_embeddings = self._generate_embeddings(chunks)
            
            # Stage 6: Index in vector store
            self._report_progress("Indexing in vector store", 0.9)
            self._index_chunks(chunks_with_embeddings, document_id)
            
            # Complete
            document.status = ProcessingStatus.READY if not failed_pages else ProcessingStatus.PARTIAL
            document.completed_at = datetime.now(timezone.utc).isoformat()
            document.processing_method = self._determine_processing_method(doc_type)
            
            # Save document metadata
            self._save_document(document)
            
            self._report_progress("Complete", 1.0)
            
            return ProcessingResult(
                success=True,
                document=document,
                failed_pages=failed_pages,
                warnings=[] if not failed_pages else [f"Pages failed: {failed_pages}"],
            )
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}", exc_info=True)
            document.status = ProcessingStatus.FAILED
            document.error_message = str(e)
            return ProcessingResult(success=False, document=document, error=str(e))
    
    def _validate_pdf(self, file_path: Path) -> Optional[str]:
        """Validate PDF file."""
        # Check file exists
        if not file_path.exists():
            return "File not found"
        
        # Check extension
        if file_path.suffix.lower() != ".pdf":
            return "Only PDF files are supported"

        # Check magic bytes (%PDF-) so renamed executables/scripts are rejected
        # before pdfplumber parsing. CSV path already rejects NUL/binary.
        try:
            with file_path.open("rb") as _f:
                header = _f.read(5)
            if header != b"%PDF-":
                return "File is not a valid PDF (missing %PDF- header)"
        except Exception as e:
            return f"Could not read uploaded file: {e}"
        
        # Check file size (50MB limit — matches ARIA_MAX_UPLOAD_PDF_MB /
        # RAG MAX_PDF_MB; connect_routes spool layer returns 413 first).
        size_bytes = file_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        if size_bytes > MAX_PDF_BYTES:
            return f"File too large: {size_mb:.1f}MB (max 50MB)"
        if size_bytes > LARGE_INLINE_WARN_BYTES:
            logger.warning(
                "Large PDF processed inline: %s (%.1fMB). "
                "Consider background job queue for files this size.",
                file_path.name, size_mb,
            )
        
        # Check PDF integrity
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                if page_count == 0:
                    return "PDF has no pages"
                if page_count > MAX_PDF_PAGES:
                    return f"Too many pages: {page_count} (max {MAX_PDF_PAGES})"
                if page_count > LARGE_INLINE_WARN_PAGES:
                    logger.warning(
                        "Large PDF processed inline: %s (%d pages). "
                        "Consider background job queue for files this size.",
                        file_path.name, page_count,
                    )
        except Exception as e:
            return f"Invalid or corrupted PDF: {e}"
        
        return None
    
    def _compute_sha256(self, file_path: Path) -> str:
        """Compute SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _extract_content(
        self,
        file_path: Path,
        document_id: str,
        doc_type: str,
        page_analyses: List,
    ) -> tuple:
        """Extract content based on document type."""
        all_elements = []
        all_tables = []
        failed_pages = []
        
        # Get page types
        page_types = []
        for p in page_analyses:
            if hasattr(p, 'is_likely_handwritten') and p.is_likely_handwritten:
                page_types.append("handwritten")
            elif hasattr(p, 'is_likely_scanned') and p.is_likely_scanned:
                page_types.append("scanned")
            else:
                page_types.append("digital")
        
        # Process each page with appropriate method
        for page_num, page_type in enumerate(page_types, 1):
            try:
                if page_type == "handwritten" and VISION_AVAILABLE and self.llm_provider:
                    # Use vision LLM for handwriting
                    result = self._extract_handwritten_page(file_path, document_id, page_num)
                elif page_type == "scanned" and OCR_AVAILABLE:
                    # Use OCR for scanned
                    result = self._extract_scanned_page(file_path, document_id, page_num)
                else:
                    # Use digital extraction
                    result = self._extract_digital_page(file_path, document_id, page_num)
                
                if result.success:
                    all_elements.extend(result.elements)
                    all_tables.extend(result.tables)
                else:
                    failed_pages.append(page_num)
                    logger.warning(f"Page {page_num} extraction failed: {result.error}")
                    
            except Exception as e:
                logger.error(f"Page {page_num} extraction error: {e}")
                failed_pages.append(page_num)
        
        # Also extract tables globally (may find tables missed by page-level)
        try:
            table_extractor = TableExtractor(file_path)
            global_tables = table_extractor.extract_all_tables()
            # Merge with page-level tables (deduplicate by page + index)
            existing_keys = {(t.get("page"), t.get("table_index")) for t in all_tables}
            for table in global_tables:
                key = (table.get("page"), table.get("table_index"))
                if key not in existing_keys:
                    all_tables.append(table)
        except Exception as e:
            logger.warning(f"Global table extraction failed: {e}")
        
        return all_elements, all_tables, failed_pages
    
    def _extract_digital_page(self, file_path: Path, document_id: str, page_num: int) -> PageProcessingResult:
        """Extract a single page using digital extraction."""
        with DigitalExtractor(file_path) as extractor:
            result = extractor.extract_page(page_num)
            # Set document_id on elements
            for elem in result.elements:
                elem.document_id = document_id
            return result
    
    def _extract_scanned_page(self, file_path: Path, document_id: str, page_num: int) -> PageProcessingResult:
        """Extract a single page using OCR."""
        extractor = ScannedExtractor(file_path)
        result = extractor.extract_page(page_num)
        for elem in result.elements:
            elem.document_id = document_id
        return result
    
    def _extract_handwritten_page(self, file_path: Path, document_id: str, page_num: int) -> PageProcessingResult:
        """Extract a single page using vision LLM."""
        extractor = HandwritingExtractor(file_path, llm_provider=self.llm_provider)
        result = extractor.extract_page(page_num)
        for elem in result.elements:
            elem.document_id = document_id
        return result
    
    def _chunk_document(self, elements: List[DocumentElement], document_id: str) -> List[DocumentChunk]:
        """Chunk document elements."""
        return chunk_document_elements(
            elements, 
            document_id,
            max_tokens=self.chunk_config.max_tokens,
            overlap_tokens=self.chunk_config.overlap_tokens,
        )
    
    def _generate_embeddings(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate embeddings for chunks."""
        if not chunks:
            return chunks
        
        # Prepare texts
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate(texts)
        
        # Assign to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return chunks
    
    def _index_chunks(self, chunks: List[DocumentChunk], document_id: str) -> None:
        """Index chunks in vector store."""
        if not self.vector_store or not chunks:
            return
        
        # Prepare for vector store
        chunk_data = []
        for chunk in chunks:
            if chunk.embedding is None:
                continue
            chunk_data.append({
                "chunk_id": chunk.id,
                "document_id": document_id,
                "content": chunk.content,
                "embedding": chunk.embedding,
                "page_number": chunk.page_number,
                "section": chunk.section,
                "source_type": chunk.source_type,
                "token_count": chunk.token_count,
                "metadata": chunk.metadata,
            })
        
        if chunk_data:
            self.vector_store.add_chunks(chunk_data)
    
    def _determine_processing_method(self, doc_type: str) -> str:
        """Determine processing method string."""
        methods = {
            "digital": "digital_extraction",
            "scanned": "ocr",
            "handwritten": "vision_llm",
            "mixed": "hybrid",
        }
        return methods.get(doc_type, "unknown")
    
    def _save_document(self, document: Document) -> None:
        """Save document metadata to disk."""
        try:
            doc_path = self.processed_dir / f"{document.id}.json"
            doc_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a serializable version of the document dict
            doc_dict = document.to_dict()
            
            # Sanitize metadata to ensure JSON serializable
            def make_serializable(obj):
                if hasattr(obj, '__dict__'):
                    return {k: make_serializable(v) for k, v in obj.__dict__.items()}
                elif isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [make_serializable(v) for v in obj]
                elif isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                else:
                    return str(obj)
            
            doc_dict['metadata'] = make_serializable(doc_dict.get('metadata', {}))
            
            doc_path.write_text(
                json.dumps(doc_dict, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"Failed to save document {document.id}: {e}")
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its data."""
        try:
            # Delete from vector store
            if self.vector_store:
                self.vector_store.delete_document(document_id)
            
            # Delete file if exists
            doc_path = self.processed_dir / f"{document_id}.json"
            if doc_path.exists():
                doc_path.unlink()
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    def get_document_status(self, document_id: str) -> Optional[Dict]:
        """Get document processing status."""
        doc_path = self.processed_dir / f"{document_id}.json"
        if doc_path.exists():
            import json
            return json.loads(doc_path.read_text())
        return None
    
    def search_documents(
        self,
        query: str,
        user_id: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
        k: int = 10,
    ) -> List[SearchResult]:
        """
        Search documents using RAG.
        
        Args:
            query: User query
            user_id: Optional user ID for filtering
            document_ids: Optional specific document IDs to search
            k: Number of results
        
        Returns:
            List of SearchResult with content and citations
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_single(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            k=k,
            document_ids=document_ids,
            min_score=0.3,
        )
        
        return results
    
    def get_document_context(
        self,
        document_id: str,
        query: str,
        k: int = 5,
    ) -> List[SearchResult]:
        """Get relevant context from a specific document for a query."""
        query_embedding = self.embedding_generator.generate_single(query)
        return self.vector_store.search(
            query_embedding=query_embedding,
            k=k,
            document_ids=[document_id],
            min_score=0.3,
        )


def create_document_processor(
    upload_dir: str | Path,
    processed_dir: str | Path,
    llm_provider=None,
    **kwargs
) -> DocumentProcessor:
    """Factory function to create document processor."""
    return DocumentProcessor(
        upload_dir=upload_dir,
        processed_dir=processed_dir,
        llm_provider=llm_provider,
        **kwargs
    )