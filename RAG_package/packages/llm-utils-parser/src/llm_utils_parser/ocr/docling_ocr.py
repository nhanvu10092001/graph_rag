"""Docling OCR provider — IBM's document understanding library."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from . import BaseOCR

logger = logging.getLogger(__name__)


class DoclingOCR(BaseOCR):
    """OCR using IBM Docling for advanced document understanding.

    Config (under extraction.ocr.docling):
        enable_table_extraction: true
        enable_ocr: true
        ocr_engine: "easyocr"  # Options: "easyocr", "tesseract"
        max_pages: 100
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.enable_tables = self.config.get("enable_table_extraction", True)
        self.enable_ocr = self.config.get("enable_ocr", True)
        self.ocr_engine = self.config.get("ocr_engine", "easyocr")
        self.max_pages = self.config.get("max_pages", 100)
        self._converter = None

    def _get_converter(self):
        """Lazy-init Docling converter."""
        if self._converter is not None:
            return self._converter

        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import PdfFormatOption

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = self.enable_ocr
            pipeline_options.do_table_structure = self.enable_tables

            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                    ),
                }
            )
            return self._converter

        except ImportError:
            raise ImportError(
                "Docling is required for this OCR provider. "
                "Install with: pip install docling"
            )

    def extract_text(self, file_path: str | Path) -> Tuple[str, Dict[str, Any]]:
        """Extract text using Docling.

        Returns Markdown-formatted text with tables preserved.
        """
        file_path = Path(file_path)
        t0 = time.perf_counter()

        converter = self._get_converter()
        result = converter.convert(str(file_path))

        # Export as Markdown (preserves tables, headings, lists)
        content = result.document.export_to_markdown()

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Collect metadata
        metadata = {
            "ocr_provider": "docling",
            "ocr_time_ms": elapsed_ms,
            "pages": getattr(result.document, "num_pages", None),
            "tables_found": len(result.document.tables) if hasattr(result.document, "tables") else 0,
        }

        logger.info(
            "docling: extracted %d chars in %.1fms from '%s'",
            len(content), elapsed_ms, file_path.name,
        )

        return content, metadata
