"""
Digital PDF Extraction
======================
Extracts text, layout, tables, and structure from digital/text-based PDFs.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pdfplumber

from .models import DocumentElement, ElementType, PageProcessingResult, DocumentType

logger = logging.getLogger(__name__)


class DigitalExtractor:
    """
    Extracts content from digital PDFs using pdfplumber.
    
    Preserves:
    - Page numbers
    - Text with layout information
    - Headings (detected via font size/style)
    - Tables with structure
    - Sections
    """
    
    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)
        self.pdf = None
    
    def __enter__(self):
        self.pdf = pdfplumber.open(self.pdf_path)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pdf:
            self.pdf.close()
            self.pdf = None
    
    def extract_all(self) -> PageProcessingResult:
        """Extract content from all pages."""
        all_elements = []
        all_tables = []
        
        if not self.pdf:
            self.pdf = pdfplumber.open(self.pdf_path)
        
        try:
            for i, page in enumerate(self.pdf.pages):
                page_num = i + 1
                elements, tables = self._extract_page(page, page_num)
                all_elements.extend(elements)
                all_tables.extend(tables)
        finally:
            if self.pdf:
                self.pdf.close()
                self.pdf = None
        
        return PageProcessingResult(
            page_number=0,  # 0 means all pages
            success=True,
            elements=all_elements,
            tables=all_tables,
            document_type=DocumentType.DIGITAL,
            confidence=0.9,
            processing_method="digital",
        )
    
    def extract_page(self, page_number: int) -> PageProcessingResult:
        """Extract content from a specific page."""
        if not self.pdf:
            self.pdf = pdfplumber.open(self.pdf_path)
        
        try:
            if page_number < 1 or page_number > len(self.pdf.pages):
                return PageProcessingResult(
                    page_number=page_number,
                    success=False,
                    error=f"Page {page_number} out of range (1-{len(self.pdf.pages)})",
                )
            
            page = self.pdf.pages[page_number - 1]
            elements, tables = self._extract_page(page, page_number)
            
            return PageProcessingResult(
                page_number=page_number,
                success=True,
                elements=elements,
                tables=tables,
                document_type=DocumentType.DIGITAL,
                confidence=0.9,
                processing_method="digital",
            )
        finally:
            if self.pdf:
                self.pdf.close()
                self.pdf = None
    
    def _extract_page(self, page, page_number: int) -> Tuple[List[DocumentElement], List[Dict[str, Any]]]:
        """Extract elements and tables from a single page."""
        elements = []
        tables = []
        
        # Extract text with layout info
        text_elements = self._extract_text_with_layout(page, page_number)
        elements.extend(text_elements)
        
        # Extract tables
        page_tables = self._extract_tables(page, page_number)
        tables.extend(page_tables)
        
        # Add table elements to elements list
        for table in page_tables:
            table_element = DocumentElement(
                document_id="",  # Will be set by processor
                page_number=page_number,
                element_type=ElementType.TABLE,
                content=table.get("markdown", ""),
                section=table.get("section"),
                source_type="digital",
                confidence=0.95,
                bounding_box=table.get("bbox"),
                table_data=table,
            )
            elements.append(table_element)
        
        return elements, tables
    
    def _extract_text_with_layout(self, page, page_number: int) -> List[DocumentElement]:
        """Extract text preserving layout structure."""
        elements = []
        
        # Get characters with font info
        chars = page.chars
        if not chars:
            return elements
        
        # Group characters into lines by y-position (with tolerance)
        lines = self._group_chars_into_lines(chars)
        
        # Classify lines as headings or paragraphs
        for line in lines:
            if not line["text"].strip():
                continue
            
            element_type = self._classify_line(line)
            
            # Calculate bounding box (normalized)
            bbox = self._normalize_bbox(line["bbox"], page.width, page.height)
            
            element = DocumentElement(
                document_id="",
                page_number=page_number,
                element_type=element_type,
                content=line["text"].strip(),
                section=self._detect_section(line["text"]),
                source_type="digital",
                confidence=0.9,
                bounding_box=bbox,
                metadata={
                    "font_size": line.get("avg_font_size"),
                    "font_name": line.get("font_name"),
                    "is_bold": line.get("is_bold", False),
                },
            )
            elements.append(element)
        
        return elements
    
    def _group_chars_into_lines(self, chars: List[Dict], y_tolerance: float = 2.0) -> List[Dict]:
        """Group characters into lines based on y-position."""
        if not chars:
            return []
        
        # Sort by y-position (top to bottom), then x-position (left to right)
        sorted_chars = sorted(chars, key=lambda c: (round(c["y0"] / y_tolerance) * y_tolerance, c["x0"]))
        
        lines = []
        current_line = []
        current_y = None
        
        for char in sorted_chars:
            char_y = round(char["y0"] / y_tolerance) * y_tolerance
            
            if current_y is None:
                current_y = char_y
                current_line = [char]
            elif abs(char_y - current_y) <= y_tolerance:
                current_line.append(char)
            else:
                # Finish current line
                if current_line:
                    lines.append(self._build_line(current_line))
                current_line = [char]
                current_y = char_y
        
        # Don't forget the last line
        if current_line:
            lines.append(self._build_line(current_line))
        
        return lines
    
    def _build_line(self, chars: List[Dict]) -> Dict:
        """Build a line object from characters."""
        text = "".join(c["text"] for c in chars)
        font_sizes = [c.get("size", 12) for c in chars]
        font_names = [c.get("fontname", "") for c in chars]
        
        x0 = min(c["x0"] for c in chars)
        y0 = min(c["y0"] for c in chars)
        x1 = max(c["x1"] for c in chars)
        y1 = max(c["y1"] for c in chars)
        
        # Check if bold (heuristic: fontname contains Bold)
        is_bold = any("Bold" in fn or "bold" in fn for fn in font_names)
        
        return {
            "text": text,
            "chars": chars,
            "bbox": (x0, y0, x1, y1),
            "avg_font_size": sum(font_sizes) / len(font_sizes) if font_sizes else 12,
            "font_name": font_names[0] if font_names else "",
            "is_bold": is_bold,
        }
    
    def _classify_line(self, line: Dict) -> ElementType:
        """Classify a line as heading or paragraph."""
        text = line["text"].strip()
        font_size = line.get("avg_font_size", 12)
        is_bold = line.get("is_bold", False)
        
        # Heuristics for headings
        if len(text) < 100 and (font_size > 14 or is_bold):
            # Check for heading patterns
            if re.match(r"^\d+(\.\d+)*\s+", text):  # Numbered heading
                return ElementType.HEADING
            if text.isupper() and len(text) > 3:  # ALL CAPS
                return ElementType.HEADING
            if font_size > 16:  # Large font
                return ElementType.HEADING
        
        return ElementType.PARAGRAPH
    
    def _detect_section(self, text: str) -> Optional[str]:
        """Detect section from text content."""
        # Simple section detection - could be enhanced
        text = text.strip()
        if re.match(r"^\d+(\.\d+)*\s+", text):
            return text.split()[0]  # Return the numbering as section
        return None
    
    def _normalize_bbox(self, bbox: Tuple[float, float, float, float], page_width: float, page_height: float) -> Dict[str, float]:
        """Normalize bounding box to 0-1 range."""
        x0, y0, x1, y1 = bbox
        return {
            "x0": x0 / page_width if page_width else 0,
            "y0": y0 / page_height if page_height else 0,
            "x1": x1 / page_width if page_width else 0,
            "y1": y1 / page_height if page_height else 0,
        }
    
    def _extract_tables(self, page, page_number: int) -> List[Dict[str, Any]]:
        """Extract tables from a page."""
        tables = []
        
        try:
            page_tables = page.extract_tables()
            for idx, table_data in enumerate(page_tables):
                if not table_data or len(table_data) < 2:
                    continue
                
                # Convert to structured format
                headers = table_data[0] if table_data else []
                rows = table_data[1:] if len(table_data) > 1 else []
                
                # Clean headers
                clean_headers = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(headers)]
                
                # Build markdown representation
                markdown = self._table_to_markdown(clean_headers, rows)
                
                # Get table bbox if available
                table_bbox = None
                try:
                    table_objects = page.find_tables()
                    if table_objects and idx < len(table_objects):
                        t = table_objects[idx]
                        table_bbox = {
                            "x0": t.bbox[0] / page.width if page.width else 0,
                            "y0": t.bbox[1] / page.height if page.height else 0,
                            "x1": t.bbox[2] / page.width if page.width else 0,
                            "y1": t.bbox[3] / page.height if page.height else 0,
                        }
                except Exception:
                    pass
                
                table = {
                    "page": page_number,
                    "table_index": idx,
                    "headers": clean_headers,
                    "rows": [[str(c).strip() if c else "" for c in row] for row in rows],
                    "row_count": len(rows),
                    "col_count": len(clean_headers),
                    "markdown": markdown,
                    "bbox": table_bbox,
                    "section": self._detect_section(f"Table {idx + 1}"),
                }
                tables.append(table)
        except Exception as e:
            logger.warning(f"Table extraction failed on page {page_number}: {e}")
        
        return tables
    
    def _table_to_markdown(self, headers: List[str], rows: List[List[str]]) -> str:
        """Convert table to markdown format."""
        if not headers:
            return ""
        
        lines = []
        # Header
        lines.append("| " + " | ".join(headers) + " |")
        # Separator
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        # Rows
        for row in rows:
            # Pad row to match headers
            padded_row = row + [""] * (len(headers) - len(row))
            lines.append("| " + " | ".join(padded_row[:len(headers)]) + " |")
        
        return "\n".join(lines)


def extract_digital_pdf(pdf_path: str | Path) -> PageProcessingResult:
    """Convenience function to extract digital PDF."""
    with DigitalExtractor(pdf_path) as extractor:
        return extractor.extract_all()