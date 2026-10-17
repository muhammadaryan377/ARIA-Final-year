"""
Scanned PDF Extraction with OCR
===============================
Processes scanned/image PDFs using OCR.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .models import DocumentElement, ElementType, PageProcessingResult, DocumentType

logger = logging.getLogger(__name__)

# Optional imports for OCR
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    pytesseract = None
    convert_from_path = None
    Image = None

# Try to import pdfplumber for layout analysis
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


class ScannedExtractor:
    """
    Extracts text from scanned PDFs using OCR.
    
    Pipeline:
    1. Render PDF pages as images
    2. Run OCR on each page image
    3. Extract layout structure from OCR results
    4. Preserve page numbers and positional info
    """
    
    def __init__(self, pdf_path: str | Path, dpi: int = 300, lang: str = "eng"):
        self.pdf_path = Path(pdf_path)
        self.dpi = dpi
        self.lang = lang
        self._temp_dir = None
    
    def extract_all(self) -> PageProcessingResult:
        """Extract content from all pages using OCR."""
        if not OCR_AVAILABLE:
            return PageProcessingResult(
                page_number=0,
                success=False,
                error="OCR dependencies not available. Install pytesseract and pdf2image.",
                processing_method="ocr",
            )
        
        all_elements = []
        
        try:
            # Convert PDF pages to images
            images = convert_from_path(self.pdf_path, dpi=self.dpi)
            
            for i, image in enumerate(images):
                page_num = i + 1
                elements = self._ocr_page(image, page_num)
                all_elements.extend(elements)
            
            return PageProcessingResult(
                page_number=0,
                success=True,
                elements=all_elements,
                tables=[],
                document_type=DocumentType.SCANNED,
                confidence=0.7,
                processing_method="ocr",
            )
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return PageProcessingResult(
                page_number=0,
                success=False,
                error=str(e),
                processing_method="ocr",
            )
    
    def extract_page(self, page_number: int) -> PageProcessingResult:
        """Extract content from a specific page using OCR."""
        if not OCR_AVAILABLE:
            return PageProcessingResult(
                page_number=page_number,
                success=False,
                error="OCR dependencies not available",
                processing_method="ocr",
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
            
            elements = self._ocr_page(images[0], page_number)
            
            return PageProcessingResult(
                page_number=page_number,
                success=True,
                elements=elements,
                tables=[],
                document_type=DocumentType.SCANNED,
                confidence=0.7,
                processing_method="ocr",
            )
        except Exception as e:
            logger.error(f"OCR extraction failed for page {page_number}: {e}")
            return PageProcessingResult(
                page_number=page_number,
                success=False,
                error=str(e),
                processing_method="ocr",
            )
    
    def _ocr_page(self, image, page_number: int) -> List[DocumentElement]:
        """Run OCR on a single page image."""
        elements = []
        
        try:
            # Get detailed OCR data with bounding boxes
            ocr_data = pytesseract.image_to_data(
                image, 
                lang=self.lang, 
                output_type=pytesseract.Output.DICT
            )
            
            # Group words into lines
            lines = self._group_ocr_words(ocr_data)
            
            for line in lines:
                if not line["text"].strip():
                    continue
                
                # Create element
                element = DocumentElement(
                    document_id="",
                    page_number=page_number,
                    element_type=ElementType.OCR_TEXT,
                    content=line["text"].strip(),
                    section=None,
                    source_type="ocr",
                    confidence=line["confidence"] / 100.0,
                    bounding_box=line["bbox"],
                    metadata={
                        "ocr_engine": "tesseract",
                        "language": self.lang,
                        "dpi": self.dpi,
                    },
                )
                elements.append(element)
            
        except Exception as e:
            logger.error(f"OCR failed for page {page_number}: {e}")
        
        return elements
    
    def _group_ocr_words(self, ocr_data: Dict) -> List[Dict]:
        """Group OCR words into lines based on line_num."""
        lines = {}
        
        for i in range(len(ocr_data["text"])):
            text = ocr_data["text"][i].strip()
            if not text:
                continue
            
            line_num = ocr_data["line_num"][i]
            block_num = ocr_data["block_num"][i]
            par_num = ocr_data["par_num"][i]
            
            key = (block_num, par_num, line_num)
            
            if key not in lines:
                lines[key] = {
                    "words": [],
                    "confidences": [],
                    "bboxes": [],
                }
            
            lines[key]["words"].append(text)
            lines[key]["confidences"].append(ocr_data["conf"][i])
            lines[key]["bboxes"].append({
                "x0": ocr_data["left"][i],
                "y0": ocr_data["top"][i],
                "x1": ocr_data["left"][i] + ocr_data["width"][i],
                "y1": ocr_data["top"][i] + ocr_data["height"][i],
            })
        
        # Build line objects
        result = []
        for key, data in sorted(lines.items()):
            # Calculate combined bbox
            bboxes = data["bboxes"]
            if bboxes:
                x0 = min(b["x0"] for b in bboxes)
                y0 = min(b["y0"] for b in bboxes)
                x1 = max(b["x1"] for b in bboxes)
                y1 = max(b["y1"] for b in bboxes)
            else:
                x0 = y0 = x1 = y1 = 0
            
            # Normalize to 0-1 (assuming standard page size, will be corrected later)
            # We'll use image dimensions for normalization
            text = " ".join(data["words"])
            avg_conf = sum(data["confidences"]) / len(data["confidences"]) if data["confidences"] else 0
            
            result.append({
                "text": text,
                "confidence": avg_conf,
                "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            })
        
        return result


class HybridExtractor:
    """
    Hybrid extractor that tries digital extraction first,
    then falls back to OCR for pages with little text.
    """
    
    def __init__(self, pdf_path: str | Path, ocr_dpi: int = 300):
        self.pdf_path = Path(pdf_path)
        self.ocr_dpi = ocr_dpi
    
    def extract_all(self) -> PageProcessingResult:
        """Extract using best method per page."""
        if not PDFPLUMBER_AVAILABLE:
            return PageProcessingResult(
                page_number=0,
                success=False,
                error="pdfplumber not available",
            )
        
        import pdfplumber
        
        all_elements = []
        all_tables = []
        page_types = []
        
        with pdfplumber.open(self.pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                
                # Try digital extraction first
                text = page.extract_text() or ""
                text_length = len(text.strip())
                
                if text_length > 100:  # Sufficient text - use digital
                    elements, tables = self._extract_digital_page(page, page_num)
                    all_elements.extend(elements)
                    all_tables.extend(tables)
                    page_types.append("digital")
                else:
                    # Fall back to OCR
                    if OCR_AVAILABLE:
                        elements = self._ocr_page(page_num)
                        all_elements.extend(elements)
                        page_types.append("scanned")
                    else:
                        # No OCR available, use whatever digital extraction got
                        elements, tables = self._extract_digital_page(page, page_num)
                        all_elements.extend(elements)
                        all_tables.extend(tables)
                        page_types.append("digital_low_text")
        
        # Determine overall document type
        if all(t == "digital" for t in page_types):
            doc_type = DocumentType.DIGITAL
        elif all(t in ("scanned", "digital_low_text") for t in page_types):
            doc_type = DocumentType.SCANNED
        else:
            doc_type = DocumentType.MIXED
        
        return PageProcessingResult(
            page_number=0,
            success=True,
            elements=all_elements,
            tables=all_tables,
            document_type=doc_type,
            confidence=0.8,
            processing_method="hybrid",
        )
    
    def _extract_digital_page(self, page, page_number: int) -> Tuple[List[DocumentElement], List[Dict]]:
        """Extract using digital method."""
        from .extraction import DigitalExtractor
        
        # Use the digital extractor logic inline to avoid reopening PDF
        elements = []
        tables = []
        
        # Text extraction
        chars = page.chars
        if chars:
            lines = self._group_chars_into_lines(chars)
            for line in lines:
                if not line["text"].strip():
                    continue
                element_type = self._classify_line(line)
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
                )
                elements.append(element)
        
        # Table extraction
        try:
            page_tables = page.extract_tables()
            for idx, table_data in enumerate(page_tables):
                if not table_data or len(table_data) < 2:
                    continue
                headers = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(table_data[0])]
                rows = table_data[1:]
                markdown = self._table_to_markdown(headers, rows)
                table = {
                    "page": page_number,
                    "table_index": idx,
                    "headers": headers,
                    "rows": [[str(c).strip() if c else "" for c in row] for row in rows],
                    "row_count": len(rows),
                    "col_count": len(headers),
                    "markdown": markdown,
                    "section": f"Table {idx + 1}",
                }
                tables.append(table)
                
                # Add as element
                table_element = DocumentElement(
                    document_id="",
                    page_number=page_number,
                    element_type=ElementType.TABLE,
                    content=markdown,
                    section=f"Table {idx + 1}",
                    source_type="digital",
                    confidence=0.95,
                    table_data=table,
                )
                elements.append(table_element)
        except Exception:
            pass
        
        return elements, tables
    
    def _ocr_page(self, page_number: int) -> List[DocumentElement]:
        """OCR a specific page."""
        if not OCR_AVAILABLE:
            return []
        
        try:
            from pdf2image import convert_from_path
            import pytesseract
            
            images = convert_from_path(
                self.pdf_path, 
                dpi=self.ocr_dpi, 
                first_page=page_number, 
                last_page=page_number
            )
            
            if not images:
                return []
            
            elements = []
            image = images[0]
            
            ocr_data = pytesseract.image_to_data(
                image, 
                lang="eng", 
                output_type=pytesseract.Output.DICT
            )
            
            lines = self._group_ocr_words(ocr_data)
            
            for line in lines:
                if not line["text"].strip():
                    continue
                
                # Normalize bbox using image dimensions
                img_width, img_height = image.size
                bbox = line["bbox"]
                normalized_bbox = {
                    "x0": bbox["x0"] / img_width,
                    "y0": bbox["y0"] / img_height,
                    "x1": bbox["x1"] / img_width,
                    "y1": bbox["y1"] / img_height,
                }
                
                element = DocumentElement(
                    document_id="",
                    page_number=page_number,
                    element_type=ElementType.OCR_TEXT,
                    content=line["text"].strip(),
                    section=None,
                    source_type="ocr",
                    confidence=line["confidence"] / 100.0,
                    bounding_box=normalized_bbox,
                )
                elements.append(element)
            
            return elements
        except Exception as e:
            logger.error(f"OCR failed for page {page_number}: {e}")
            return []
    
    def _group_chars_into_lines(self, chars: List[Dict], y_tolerance: float = 2.0) -> List[Dict]:
        """Group characters into lines."""
        if not chars:
            return []
        
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
                if current_line:
                    lines.append(self._build_line(current_line))
                current_line = [char]
                current_y = char_y
        
        if current_line:
            lines.append(self._build_line(current_line))
        
        return lines
    
    def _build_line(self, chars: List[Dict]) -> Dict:
        """Build line from chars."""
        text = "".join(c["text"] for c in chars)
        font_sizes = [c.get("size", 12) for c in chars]
        font_names = [c.get("fontname", "") for c in chars]
        
        x0 = min(c["x0"] for c in chars)
        y0 = min(c["y0"] for c in chars)
        x1 = max(c["x1"] for c in chars)
        y1 = max(c["y1"] for c in chars)
        
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
        """Classify line as heading or paragraph."""
        text = line["text"].strip()
        font_size = line.get("avg_font_size", 12)
        is_bold = line.get("is_bold", False)
        
        import re
        if len(text) < 100 and (font_size > 14 or is_bold):
            if re.match(r"^\d+(\.\d+)*\s+", text):
                return ElementType.HEADING
            if text.isupper() and len(text) > 3:
                return ElementType.HEADING
            if font_size > 16:
                return ElementType.HEADING
        
        return ElementType.PARAGRAPH
    
    def _detect_section(self, text: str) -> Optional[str]:
        import re
        text = text.strip()
        if re.match(r"^\d+(\.\d+)*\s+", text):
            return text.split()[0]
        return None
    
    def _normalize_bbox(self, bbox: Tuple[float, float, float, float], page_width: float, page_height: float) -> Dict[str, float]:
        x0, y0, x1, y1 = bbox
        return {
            "x0": x0 / page_width if page_width else 0,
            "y0": y0 / page_height if page_height else 0,
            "x1": x1 / page_width if page_width else 0,
            "y1": y1 / page_height if page_height else 0,
        }
    
    def _table_to_markdown(self, headers: List[str], rows: List[List[str]]) -> str:
        if not headers:
            return ""
        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            padded_row = row + [""] * (len(headers) - len(row))
            lines.append("| " + " | ".join(padded_row[:len(headers)]) + " |")
        return "\n".join(lines)
    
    def _group_ocr_words(self, ocr_data: Dict) -> List[Dict]:
        """Group OCR words into lines."""
        lines = {}
        
        for i in range(len(ocr_data["text"])):
            text = ocr_data["text"][i].strip()
            if not text:
                continue
            
            line_num = ocr_data["line_num"][i]
            block_num = ocr_data["block_num"][i]
            par_num = ocr_data["par_num"][i]
            
            key = (block_num, par_num, line_num)
            
            if key not in lines:
                lines[key] = {
                    "words": [],
                    "confidences": [],
                    "bboxes": [],
                }
            
            lines[key]["words"].append(text)
            lines[key]["confidences"].append(ocr_data["conf"][i])
            lines[key]["bboxes"].append({
                "x0": ocr_data["left"][i],
                "y0": ocr_data["top"][i],
                "x1": ocr_data["left"][i] + ocr_data["width"][i],
                "y1": ocr_data["top"][i] + ocr_data["height"][i],
            })
        
        result = []
        for key, data in sorted(lines.items()):
            bboxes = data["bboxes"]
            if bboxes:
                x0 = min(b["x0"] for b in bboxes)
                y0 = min(b["y0"] for b in bboxes)
                x1 = max(b["x1"] for b in bboxes)
                y1 = max(b["y1"] for b in bboxes)
            else:
                x0 = y0 = x1 = y1 = 0
            
            text = " ".join(data["words"])
            avg_conf = sum(data["confidences"]) / len(data["confidences"]) if data["confidences"] else 0
            
            result.append({
                "text": text,
                "confidence": avg_conf,
                "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            })
        
        return result


def extract_scanned_pdf(pdf_path: str | Path, dpi: int = 300) -> PageProcessingResult:
    """Convenience function for scanned PDF extraction."""
    if not OCR_AVAILABLE:
        return PageProcessingResult(
            page_number=0,
            success=False,
            error="OCR not available. Install pytesseract and pdf2image.",
            processing_method="ocr",
        )
    extractor = ScannedExtractor(pdf_path, dpi=dpi)
    return extractor.extract_all()


def extract_hybrid_pdf(pdf_path: str | Path) -> PageProcessingResult:
    """Convenience function for hybrid extraction."""
    extractor = HybridExtractor(pdf_path)
    return extractor.extract_all()