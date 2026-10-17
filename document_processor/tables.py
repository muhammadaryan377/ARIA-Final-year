"""
Table Extraction
================
Advanced table extraction and preservation from PDFs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pdfplumber

logger = logging.getLogger(__name__)


class TableExtractor:
    """
    Extracts and preserves table structure from PDFs.
    
    Supports:
    - Header detection
    - Row/column structure
    - Merged cells (basic)
    - Table to markdown conversion
    - Metadata preservation
    """
    
    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)
    
    def extract_all_tables(self) -> List[Dict[str, Any]]:
        """Extract all tables from all pages."""
        tables = []
        
        with pdfplumber.open(self.pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                page_tables = self._extract_page_tables(page, page_num)
                tables.extend(page_tables)
        
        return tables
    
    def extract_page_tables(self, page_number: int) -> List[Dict[str, Any]]:
        """Extract tables from a specific page."""
        with pdfplumber.open(self.pdf_path) as pdf:
            if page_number < 1 or page_number > len(pdf.pages):
                return []
            page = pdf.pages[page_number - 1]
            return self._extract_page_tables(page, page_number)
    
    def _extract_page_tables(self, page, page_number: int) -> List[Dict[str, Any]]:
        """Extract tables from a single page using multiple strategies."""
        tables = []
        
        # Strategy 1: pdfplumber's built-in table detection
        try:
            table_objects = page.find_tables()
            for idx, table in enumerate(table_objects):
                extracted = table.extract()
                if extracted and len(extracted) >= 2:
                    structured = self._structure_table(extracted, page, page_number, idx, table.bbox)
                    if structured:
                        tables.append(structured)
        except Exception as e:
            logger.warning(f"pdfplumber table detection failed on page {page_number}: {e}")
        
        # Strategy 2: Fallback to extract_tables()
        if not tables:
            try:
                extracted_tables = page.extract_tables()
                for idx, table_data in enumerate(extracted_tables):
                    if table_data and len(table_data) >= 2:
                        structured = self._structure_table_from_data(table_data, page, page_number, idx)
                        if structured:
                            tables.append(structured)
            except Exception as e:
                logger.warning(f"extract_tables failed on page {page_number}: {e}")
        
        return tables
    
    def _structure_table(
        self, 
        extracted: List[List[str]], 
        page, 
        page_number: int, 
        table_index: int,
        bbox: Tuple[float, float, float, float]
    ) -> Optional[Dict[str, Any]]:
        """Structure extracted table data."""
        if not extracted or len(extracted) < 2:
            return None
        
        # Clean headers
        raw_headers = extracted[0]
        headers = [self._clean_cell(h) for h in raw_headers]
        headers = [h if h else f"col_{i}" for i, h in enumerate(headers)]
        
        # Clean rows
        rows = []
        for row_data in extracted[1:]:
            cleaned_row = [self._clean_cell(c) for c in row_data]
            # Pad to match headers
            cleaned_row = cleaned_row + [""] * (len(headers) - len(cleaned_row))
            rows.append(cleaned_row[:len(headers)])
        
        if not rows:
            return None
        
        # Generate markdown
        markdown = self._to_markdown(headers, rows)
        
        # Normalize bbox
        norm_bbox = self._normalize_bbox(bbox, page.width, page.height)
        
        return {
            "page": page_number,
            "table_index": table_index,
            "headers": headers,
            "rows": rows,
            "row_count": len(rows),
            "col_count": len(headers),
            "markdown": markdown,
            "bbox": norm_bbox,
            "extraction_method": "pdfplumber_find_tables",
        }
    
    def _structure_table_from_data(
        self, 
        table_data: List[List[str]], 
        page, 
        page_number: int, 
        table_index: int
    ) -> Optional[Dict[str, Any]]:
        """Structure table from extract_tables() output."""
        if not table_data or len(table_data) < 2:
            return None
        
        raw_headers = table_data[0]
        headers = [self._clean_cell(h) for h in raw_headers]
        headers = [h if h else f"col_{i}" for i, h in enumerate(headers)]
        
        rows = []
        for row_data in table_data[1:]:
            cleaned_row = [self._clean_cell(c) for c in row_data]
            cleaned_row = cleaned_row + [""] * (len(headers) - len(cleaned_row))
            rows.append(cleaned_row[:len(headers)])
        
        if not rows:
            return None
        
        markdown = self._to_markdown(headers, rows)
        
        return {
            "page": page_number,
            "table_index": table_index,
            "headers": headers,
            "rows": rows,
            "row_count": len(rows),
            "col_count": len(headers),
            "markdown": markdown,
            "bbox": None,
            "extraction_method": "pdfplumber_extract_tables",
        }
    
    def _clean_cell(self, cell: Any) -> str:
        """Clean a table cell value."""
        if cell is None:
            return ""
        text = str(cell).strip()
        # Replace multiple whitespace with single space
        import re
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def _to_markdown(self, headers: List[str], rows: List[List[str]]) -> str:
        """Convert table to markdown."""
        if not headers:
            return ""
        
        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        for row in rows:
            padded = row + [""] * (len(headers) - len(row))
            lines.append("| " + " | ".join(padded[:len(headers)]) + " |")
        
        return "\n".join(lines)
    
    def _normalize_bbox(self, bbox: Tuple[float, float, float, float], page_width: float, page_height: float) -> Dict[str, float]:
        """Normalize bbox to 0-1."""
        x0, y0, x1, y1 = bbox
        return {
            "x0": x0 / page_width if page_width else 0,
            "y0": y0 / page_height if page_height else 0,
            "x1": x1 / page_width if page_width else 0,
            "y1": y1 / page_height if page_height else 0,
        }
    
    def tables_to_dataframe(self, tables: List[Dict[str, Any]]) -> List[Any]:
        """Convert extracted tables to pandas DataFrames."""
        try:
            import pandas as pd
            dataframes = []
            for table in tables:
                df = pd.DataFrame(table["rows"], columns=table["headers"])
                dataframes.append(df)
            return dataframes
        except ImportError:
            logger.warning("pandas not available for DataFrame conversion")
            return []


def extract_tables(pdf_path: str | Path) -> List[Dict[str, Any]]:
    """Convenience function to extract all tables."""
    extractor = TableExtractor(pdf_path)
    return extractor.extract_all_tables()