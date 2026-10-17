"""
Handwriting Extraction using Vision LLM
========================================
Processes handwritten PDFs using vision-capable LLM models.
"""

from __future__ import annotations

import base64
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

from .models import DocumentElement, ElementType, PageProcessingResult, DocumentType

logger = logging.getLogger(__name__)

# Optional imports
try:
    from pdf2image import convert_from_path
    from PIL import Image
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    convert_from_path = None
    Image = None


HANDWRITING_PROMPT = """You are an expert at transcribing handwritten documents. 
Analyze this handwritten page image and extract all text content.

Return a JSON object with this structure:
{
  "text": "Full transcribed text from the page",
  "lines": [
    {"text": "line 1", "bbox": [x0, y0, x1, y1], "confidence": 0.9},
    {"text": "line 2", "bbox": [x0, y0, x1, y1], "confidence": 0.8}
  ],
  "sections": ["section1", "section2"],
  "is_handwritten": true,
  "language": "en"
}

Bounding boxes should be normalized 0-1 (x0, y0, x1, y1).
Only include lines with actual content.
Confidence is your confidence in the transcription accuracy (0-1)."""


class HandwritingExtractor:
    """
    Extracts text from handwritten PDFs using vision LLM.
    
    Requires:
    - Cloud LLM provider with vision model (Groq qwen/qwen3.8-27b)
    - pdf2image for rendering pages
    - PIL for image processing
    """
    
    def __init__(self, pdf_path: str | Path, llm_provider=None, dpi: int = 300):
        self.pdf_path = Path(pdf_path)
        self.llm_provider = llm_provider
        self.dpi = dpi
        self._temp_dir = None
    
    def set_provider(self, provider):
        """Set the LLM provider with vision capability."""
        self.llm_provider = provider
    
    def extract_all(self) -> PageProcessingResult:
        """Extract handwriting from all pages."""
        if not VISION_AVAILABLE:
            return PageProcessingResult(
                page_number=0,
                success=False,
                error="Vision dependencies not available. Install pdf2image and PIL.",
                processing_method="handwriting",
            )
        
        if not self.llm_provider:
            return PageProcessingResult(
                page_number=0,
                success=False,
                error="No LLM provider with vision capability configured.",
                processing_method="handwriting",
            )
        
        # Check if provider supports vision
        if not hasattr(self.llm_provider, 'vision'):
            return PageProcessingResult(
                page_number=0,
                success=False,
                error="LLM provider does not support vision. Use cloud provider with vision model.",
                processing_method="handwriting",
            )
        
        all_elements = []
        
        try:
            images = convert_from_path(self.pdf_path, dpi=self.dpi)
            
            for i, image in enumerate(images):
                page_num = i + 1
                elements = self._process_handwritten_page(image, page_num)
                all_elements.extend(elements)
            
            return PageProcessingResult(
                page_number=0,
                success=True,
                elements=all_elements,
                tables=[],
                document_type=DocumentType.HANDWRITTEN,
                confidence=0.8,
                processing_method="handwriting_vision",
            )
        except Exception as e:
            logger.error(f"Handwriting extraction failed: {e}")
            return PageProcessingResult(
                page_number=0,
                success=False,
                error=str(e),
                processing_method="handwriting",
            )
    
    def extract_page(self, page_number: int) -> PageProcessingResult:
        """Extract handwriting from a specific page."""
        if not VISION_AVAILABLE:
            return PageProcessingResult(
                page_number=page_number,
                success=False,
                error="Vision dependencies not available",
                processing_method="handwriting",
            )
        
        if not self.llm_provider or not hasattr(self.llm_provider, 'vision'):
            return PageProcessingResult(
                page_number=page_number,
                success=False,
                error="No vision-capable LLM provider",
                processing_method="handwriting",
            )
        
        try:
            images = convert_from_path(
                self.pdf_path, 
                dpi=self.dpi, 
                first_page=page_number, 
                last_page=page_number
            )
            
            if not images:
                return PageProcessingResult(
                    page_number=page_number,
                    success=False,
                    error=f"Page {page_number} not found",
                )
            
            elements = self._process_handwritten_page(images[0], page_number)
            
            return PageProcessingResult(
                page_number=page_number,
                success=True,
                elements=elements,
                tables=[],
                document_type=DocumentType.HANDWRITTEN,
                confidence=0.8,
                processing_method="handwriting_vision",
            )
        except Exception as e:
            logger.error(f"Handwriting extraction failed for page {page_number}: {e}")
            return PageProcessingResult(
                page_number=page_number,
                success=False,
                error=str(e),
                processing_method="handwriting",
            )
    
    def _process_handwritten_page(self, image, page_number: int) -> List[DocumentElement]:
        """Process a single handwritten page with vision LLM."""
        elements = []
        
        try:
            # Convert image to base64
            img_buffer = tempfile.BytesIO()
            image.save(img_buffer, format="PNG")
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
            
            # Call vision LLM
            response = self.llm_provider.vision(
                prompt=HANDWRITING_PROMPT,
                images=[img_base64],
                temperature=0.1,
                num_predict=2000,
            )
            
            # Parse response
            import json
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    raise ValueError("Could not parse vision model response as JSON")
            
            # Convert to DocumentElements
            lines = result.get("lines", [])
            for line in lines:
                text = line.get("text", "").strip()
                if not text:
                    continue
                
                bbox = line.get("bbox", [0, 0, 0, 0])
                confidence = line.get("confidence", 0.7)
                
                element = DocumentElement(
                    document_id="",
                    page_number=page_number,
                    element_type=ElementType.HANDWRITING,
                    content=text,
                    section=None,
                    source_type="handwriting_vision",
                    confidence=confidence,
                    bounding_box={
                        "x0": bbox[0],
                        "y0": bbox[1],
                        "x1": bbox[2],
                        "y1": bbox[3],
                    },
                    metadata={
                        "extraction_method": "vision_llm",
                        "model": getattr(self.llm_provider, 'model_for', lambda x: 'unknown')("vision"),
                        "dpi": self.dpi,
                    },
                )
                elements.append(element)
            
            # Also add full page text as a single element for context
            full_text = result.get("text", "")
            if full_text and len(full_text) > 50:
                full_element = DocumentElement(
                    document_id="",
                    page_number=page_number,
                    element_type=ElementType.HANDWRITING,
                    content=full_text,
                    section=None,
                    source_type="handwriting_vision",
                    confidence=result.get("confidence", 0.7),
                    bounding_box={"x0": 0, "y0": 0, "x1": 1, "y1": 1},
                    metadata={
                        "extraction_method": "vision_llm",
                        "is_full_page": True,
                    },
                )
                elements.insert(0, full_element)
            
        except Exception as e:
            logger.error(f"Vision processing failed for page {page_number}: {e}")
            # Add error element
            elements.append(DocumentElement(
                document_id="",
                page_number=page_number,
                element_type=ElementType.HANDWRITING,
                content=f"[Handwriting extraction failed: {str(e)}]",
                section=None,
                source_type="handwriting_vision",
                confidence=0.0,
                metadata={"error": str(e)},
            ))
        
        return elements


def extract_handwriting_pdf(pdf_path: str | Path, llm_provider, dpi: int = 300) -> PageProcessingResult:
    """Convenience function for handwriting extraction."""
    if not VISION_AVAILABLE:
        return PageProcessingResult(
            page_number=0,
            success=False,
            error="Vision dependencies not available",
            processing_method="handwriting",
        )
    extractor = HandwritingExtractor(pdf_path, llm_provider=llm_provider, dpi=dpi)
    return extractor.extract_all()