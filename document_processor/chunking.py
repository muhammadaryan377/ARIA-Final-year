"""
Document Chunking
=================
Intelligent chunking of document elements for embedding and retrieval.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator

from .models import DocumentElement, DocumentChunk, ElementType

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for chunking."""
    max_tokens: int = 512
    overlap_tokens: int = 50
    min_chunk_tokens: int = 50
    respect_boundaries: bool = True  # Respect headings, sections, pages
    include_metadata: bool = True


class DocumentChunker:
    """
    Intelligently chunks document elements for retrieval.
    
    Strategies:
    - Respects heading/section boundaries
    - Keeps tables intact
    - Preserves page context
    - Handles overlap for continuity
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
    
    def chunk_elements(
        self, 
        elements: List[DocumentElement], 
        document_id: str
    ) -> List[DocumentChunk]:
        """Chunk a list of document elements."""
        if not elements:
            return []
        
        # Sort by page, then by position
        sorted_elements = sorted(elements, key=lambda e: (e.page_number, e.bounding_box.get("y0", 0) if e.bounding_box else 0))
        
        chunks = []
        current_chunk_elements = []
        current_tokens = 0
        current_page = sorted_elements[0].page_number if sorted_elements else 1
        current_section = None
        
        for element in sorted_elements:
            element_tokens = self._estimate_tokens(element.content)
            
            # Check if we should start a new chunk
            should_split = self._should_split_chunk(
                current_chunk_elements, element, current_tokens, element_tokens
            )
            
            if should_split and current_chunk_elements:
                # Create chunk from current elements
                chunk = self._create_chunk(
                    current_chunk_elements, document_id, len(chunks)
                )
                if chunk is not None:
                    chunks.append(chunk)
                
                # Start new chunk with overlap
                current_chunk_elements = self._get_overlap_elements(current_chunk_elements)
                current_tokens = sum(self._estimate_tokens(e.content) for e in current_chunk_elements)
            
            current_chunk_elements.append(element)
            current_tokens += element_tokens
            current_page = element.page_number
            if element.section:
                current_section = element.section
        
        # Don't forget the last chunk
        if current_chunk_elements:
            chunk = self._create_chunk(current_chunk_elements, document_id, len(chunks))
            chunks.append(chunk)
        
        # Filter out too-small chunks (merge with neighbors if possible)
        chunks = self._filter_small_chunks(chunks)
        
        return chunks
    
    def _should_split_chunk(
        self, 
        current_elements: List[DocumentElement], 
        new_element: DocumentElement,
        current_tokens: int,
        new_tokens: int
    ) -> bool:
        """Determine if we should split before adding new element."""
        # Always split if adding would exceed max tokens
        if current_tokens + new_tokens > self.config.max_tokens:
            return True
        
        if not self.config.respect_boundaries:
            return False
        
        # Split on heading boundaries
        if new_element.element_type == ElementType.HEADING and current_elements:
            # Check if current chunk has substantial content
            if current_tokens > self.config.min_chunk_tokens:
                return True
        
        # Split on page boundaries if chunk is getting large
        if current_elements and new_element.page_number != current_elements[-1].page_number:
            if current_tokens > self.config.max_tokens * 0.7:
                return True
        
        # Split on section change
        if (current_elements and new_element.section 
            and new_element.section != current_elements[-1].section
            and current_tokens > self.config.min_chunk_tokens):
            return True
        
        # Keep tables together - if current chunk ends with table start, don't split
        if current_elements and current_elements[-1].element_type == ElementType.TABLE:
            if new_element.element_type != ElementType.TABLE:
                return True
        
        return False
    
    def _get_overlap_elements(self, elements: List[DocumentElement]) -> List[DocumentElement]:
        """Get elements for overlap from end of previous chunk."""
        if not elements or self.config.overlap_tokens <= 0:
            return []
        
        overlap_elements = []
        tokens = 0
        
        # Take from end backwards
        for element in reversed(elements):
            element_tokens = self._estimate_tokens(element.content)
            if tokens + element_tokens > self.config.overlap_tokens:
                break
            overlap_elements.insert(0, element)
            tokens += element_tokens
        
        return overlap_elements
    
    def _create_chunk(
        self, 
        elements: List[DocumentElement], 
        document_id: str, 
        chunk_index: int
    ) -> DocumentChunk:
        """Create a DocumentChunk from elements."""
        if not elements:
            return None
        
        # Combine content
        content_parts = []
        for elem in elements:
            if elem.element_type == ElementType.HEADING:
                content_parts.append(f"## {elem.content}")
            elif elem.element_type == ElementType.TABLE:
                content_parts.append(f"\n{elem.content}\n")
            else:
                content_parts.append(elem.content)
        
        content = "\n\n".join(content_parts)
        
        # Determine primary page and section
        page_number = elements[0].page_number
        section = None
        for elem in elements:
            if elem.section:
                section = elem.section
                break
        
        # Determine source type (majority)
        source_types = [e.source_type for e in elements]
        source_type = max(set(source_types), key=source_types.count) if source_types else "digital"
        
        # Collect element IDs
        element_ids = [e.id for e in elements]
        
        # Estimate token count
        token_count = self._estimate_tokens(content)
        
        chunk = DocumentChunk(
            document_id=document_id,
            page_number=page_number,
            section=section,
            content=content,
            source_type=source_type,
            element_ids=element_ids,
            token_count=token_count,
            metadata={
                "chunk_index": chunk_index,
                "element_count": len(elements),
                "element_types": [e.element_type.value for e in elements],
                "pages_covered": list(set(e.page_number for e in elements)),
            } if self.config.include_metadata else {},
        )
        
        return chunk
    
    def _filter_small_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Merge or remove chunks that are too small."""
        if len(chunks) <= 1:
            return chunks
        
        filtered = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            
            if chunk.token_count < self.config.min_chunk_tokens and i < len(chunks) - 1:
                # Try to merge with next chunk
                next_chunk = chunks[i + 1]
                merged_content = chunk.content + "\n\n" + next_chunk.content
                merged_tokens = self._estimate_tokens(merged_content)
                
                if merged_tokens <= self.config.max_tokens:
                    # Merge them
                    merged = DocumentChunk(
                        document_id=chunk.document_id,
                        page_number=chunk.page_number,
                        section=chunk.section or next_chunk.section,
                        content=merged_content,
                        source_type=chunk.source_type,
                        element_ids=chunk.element_ids + next_chunk.element_ids,
                        token_count=merged_tokens,
                        metadata={
                            **chunk.metadata,
                            "merged_from": [chunk.metadata.get("chunk_index"), next_chunk.metadata.get("chunk_index")],
                        },
                    )
                    filtered.append(merged)
                    i += 2
                    continue
            
            filtered.append(chunk)
            i += 1
        
        return filtered
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token for English)."""
        return max(1, len(text) // 4)


def chunk_document_elements(
    elements: List[DocumentElement], 
    document_id: str,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
) -> List[DocumentChunk]:
    """Convenience function for chunking."""
    config = ChunkConfig(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    chunker = DocumentChunker(config)
    return chunker.chunk_elements(elements, document_id)