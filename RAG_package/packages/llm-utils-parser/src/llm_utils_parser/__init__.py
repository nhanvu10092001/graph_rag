"""LLM Utils Parser - Document text extraction and parsing utilities."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from llm_utils_parser.base import (
    BaseExtractor,
    ExtractorRegistry,
    FILE_TYPE_MAP,
    detect_file_type_from_extension,
    get_supported_extensions,
    is_extension_supported,
)

# Re-expose the extractors for programmatic imports if needed
from llm_utils_parser.text_extractor import TextExtractor
from llm_utils_parser.pdf_extractor import PDFExtractor
from llm_utils_parser.html_extractor import HTMLExtractor
from llm_utils_parser.office_extractor import OfficeExtractor
from llm_utils_parser.code_extractor import CodeExtractor
from llm_utils_parser.pymupdf_extractor import PyMuPDFExtractor
from llm_utils_parser.ocr_extractor import OCRExtractor
from llm_utils_parser.text_cleaner import TextCleaner


__version__ = "0.1.0"


def extract_text(filename: str, file_bytes: bytes, config: Optional[Dict[str, Any]] = None) -> str:
    """Extracts plain text content from a file by writing to a temporary file and running ExtractorRegistry."""
    suffix = os.path.splitext(filename)[1].lower()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file_path = tmp_file.name

    try:
        if config is None:
            config = {
                "pdf": {
                    "enable_ocr": True,
                    "primary_loader": "pypdf",
                    "min_text_threshold": 50,
                },
                "extraction": {
                    "ocr": {
                        "enabled": True,
                        "provider": "tesseract",
                        "tesseract": {
                            "lang": "vie+eng",
                        }
                    }
                }
            }
        registry = ExtractorRegistry(config)
        text, _ = registry.extract(Path(tmp_file_path))
        return text
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


__all__ = [
    "BaseExtractor",
    "ExtractorRegistry",
    "FILE_TYPE_MAP",
    "detect_file_type_from_extension",
    "get_supported_extensions",
    "is_extension_supported",
    "extract_text",
    "TextExtractor",
    "PDFExtractor",
    "HTMLExtractor",
    "OfficeExtractor",
    "PyMuPDFExtractor",
    "OCRExtractor",
    "CodeExtractor",
    "TextCleaner",
]

