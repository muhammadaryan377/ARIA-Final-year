"""
Document Type Detection
=======================
Automatically determines whether a PDF is digital, scanned, handwritten, or mixed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pdfplumber

logger = logging.getLogger(__name__)


@dataclass
class PageAnalysis:
    """Analysis result for a single page."""
    page_number: int
    has_text: bool
    text_length: int
    text_density: float  # chars per page area
    has_images: bool
    image_count: int
    image_coverage: float  # fraction of page covered by images
    is_likely_scanned: bool
    is_likely_handwritten: bool
    confidence: float


class DocumentTypeDetector:
    """
    Detects document type by analyzing page content.
    
    Uses measurable signals:
    - Amount of extractable text
    - Page image characteristics
    - Text density
    - Image coverage
    """
    
    # Thresholds for classification
    MIN_TEXT_CHARS_PER_PAGE = 50
    MIN_TEXT_DENSITY = 0.001  # chars per unit area
    MIN_IMAGE_COVERAGE_FOR_SCANNED = 0.5
    MAX_TEXT_FOR_SCANNED = 100
    
    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)
        self.pages_analysis: List[PageAnalysis] = []
    
    def detect(self) -> Tuple[str, List[PageAnalysis], dict]:
        """
        Detect document type.
        
        Returns:
            (document_type, page_analyses, summary)
        """
        self.pages_analysis = self._analyze_all_pages()
        
        if not self.pages_analysis:
            return "unknown", [], {"reason": "No pages analyzed"}
        
        # Count page types
        digital_pages = sum(1 for p in self.pages_analysis if p.has_text and p.text_length > self.MIN_TEXT_CHARS_PER_PAGE and not p.is_likely_scanned)
        scanned_pages = sum(1 for p in self.pages_analysis if p.is_likely_scanned)
        handwritten_pages = sum(1 for p in self.pages_analysis if p.is_likely_handwritten)
        mixed_pages = sum(1 for p in self.pages_analysis if p.has_text and p.is_likely_scanned)
        
        total = len(self.pages_analysis)
        
        # Determine overall document type
        if scanned_pages == total and handwritten_pages == 0:
            doc_type = "scanned"
        elif handwritten_pages > 0 and handwritten_pages >= scanned_pages:
            doc_type = "handwritten"
        elif digital_pages == total:
            doc_type = "digital"
        elif scanned_pages > 0 and digital_pages > 0:
            doc_type = "mixed"
        elif scanned_pages > 0:
            doc_type = "scanned"
        else:
            doc_type = "digital"
        
        summary = {
            "total_pages": total,
            "digital_pages": digital_pages,
            "scanned_pages": scanned_pages,
            "handwritten_pages": handwritten_pages,
            "mixed_pages": mixed_pages,
            "avg_text_per_page": sum(p.text_length for p in self.pages_analysis) / total if total > 0 else 0,
            "avg_image_coverage": sum(p.image_coverage for p in self.pages_analysis) / total if total > 0 else 0,
        }
        
        logger.info(f"Document type detected: {doc_type} ({summary})")
        return doc_type, self.pages_analysis, summary
    
    def _analyze_all_pages(self) -> List[PageAnalysis]:
        """Analyze all pages in the PDF."""
        analyses = []
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    analysis = self._analyze_page(page, i + 1)
                    analyses.append(analysis)
        except Exception as e:
            logger.error(f"Failed to analyze PDF pages: {e}")
        
        return analyses
    
    def _analyze_page(self, page, page_number: int) -> PageAnalysis:
        """Analyze a single page for content characteristics."""
        # Extract text
        text = page.extract_text() or ""
        text_length = len(text.strip())
        
        # Get page dimensions
        width = page.width or 612
        height = page.height or 792
        page_area = width * height
        
        # Calculate text density
        text_density = text_length / page_area if page_area > 0 else 0
        
        # Analyze images
        images = page.images or []
        image_count = len(images)
        
        # Calculate image coverage
        image_area = 0
        for img in images:
            img_width = img.get("width", 0)
            img_height = img.get("height", 0)
            image_area += img_width * img_height
        
        image_coverage = min(image_area / page_area, 1.0) if page_area > 0 else 0
        
        # Determine if likely scanned (image-heavy, little text)
        is_likely_scanned = (
            image_coverage > self.MIN_IMAGE_COVERAGE_FOR_SCANNED
            and text_length < self.MAX_TEXT_FOR_SCANNED
        )
        
        # Determine if likely handwritten (scanned but with specific characteristics)
        # For now, we'll use a heuristic: scanned pages with very little text
        # but high image coverage might be handwritten
        is_likely_handwritten = (
            is_likely_scanned
            and text_length < 20
            and image_coverage > 0.7
        )
        
        # Confidence based on how clear the signals are
        if is_likely_scanned and text_length < 10:
            confidence = 0.9
        elif text_length > 200 and image_coverage < 0.1:
            confidence = 0.9
        elif is_likely_handwritten:
            confidence = 0.7
        else:
            confidence = 0.5
        
        return PageAnalysis(
            page_number=page_number,
            has_text=text_length > 0,
            text_length=text_length,
            text_density=text_density,
            has_images=image_count > 0,
            image_count=image_count,
            image_coverage=image_coverage,
            is_likely_scanned=is_likely_scanned,
            is_likely_handwritten=is_likely_handwritten,
            confidence=confidence,
        )
    
    def get_page_types(self) -> List[str]:
        """Get the detected type for each page."""
        types = []
        for p in self.pages_analysis:
            if p.is_likely_handwritten:
                types.append("handwritten")
            elif p.is_likely_scanned:
                types.append("scanned")
            elif p.has_text and p.text_length > self.MIN_TEXT_CHARS_PER_PAGE:
                types.append("digital")
            else:
                types.append("unknown")
        return types


def detect_document_type(pdf_path: str | Path) -> Tuple[str, List[PageAnalysis], dict]:
    """Convenience function to detect document type."""
    detector = DocumentTypeDetector(pdf_path)
    return detector.detect()