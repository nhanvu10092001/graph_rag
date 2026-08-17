"""PyMuPDF-based PDF text extractor — simple, fast, no OCR fallback."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Tuple

from llm_utils_parser.base import BaseExtractor

logger = logging.getLogger(__name__)


class PyMuPDFExtractor(BaseExtractor):
    """Extract text from PDF using PyMuPDF (fitz) directly.

    Config (under extraction.pdf):
        extract_tables: false   # Extract tables via pdfplumber (optional)
    """

    def __init__(self, config=None):
        self.config = config or {}
        pdf_cfg = self.config.get("pdf", {})
        if not pdf_cfg and "extraction" in self.config:
            pdf_cfg = self.config.get("extraction", {}).get("pdf", {})
        self.extract_tables = pdf_cfg.get("extract_tables", False)

    def extract(self, file_path: Path, **kwargs) -> Tuple[str, Dict[str, Any]]:
        t0 = time.perf_counter()

        import pymupdf

        doc = pymupdf.open(str(file_path))
        pages = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                pages.append(text)
        page_count = len(doc)
        doc.close()

        full_text = "\n\n".join(pages)

        metadata: Dict[str, Any] = {
            "extractor": "pymupdf",
            "loader": "pymupdf",
            "pages": page_count,
            "file_size": file_path.stat().st_size,
            "ocr_used": False,
        }

        if self.extract_tables:
            tables_md, table_count = self._extract_tables(file_path)
            if tables_md:
                full_text = full_text + "\n\n" + tables_md
                metadata["tables_found"] = table_count

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        metadata["extraction_time_ms"] = elapsed_ms

        logger.info(
            "pymupdf_extractor: %d chars, %d pages in %.1fms",
            len(full_text), page_count, elapsed_ms,
        )

        return full_text, metadata

    @staticmethod
    def _extract_tables(file_path: Path) -> Tuple[str, int]:
        try:
            import pdfplumber

            tables_md = []
            with pdfplumber.open(str(file_path)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for table_idx, table in enumerate(page_tables):
                        if not table or not table[0]:
                            continue
                        md_lines = []
                        header = [str(cell or "") for cell in table[0]]
                        md_lines.append("| " + " | ".join(header) + " |")
                        md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
                        for row in table[1:]:
                            cells = [str(cell or "") for cell in row]
                            while len(cells) < len(header):
                                cells.append("")
                            md_lines.append("| " + " | ".join(cells[:len(header)]) + " |")
                        tables_md.append(
                            f"\n**Table (Page {page_num + 1}, #{table_idx + 1}):**\n"
                            + "\n".join(md_lines)
                        )
            return "\n\n".join(tables_md), len(tables_md)
        except ImportError:
            logger.debug("pdfplumber not installed, skipping table extraction")
            return "", 0
        except Exception as e:
            logger.warning("Table extraction failed: %s", e)
            return "", 0