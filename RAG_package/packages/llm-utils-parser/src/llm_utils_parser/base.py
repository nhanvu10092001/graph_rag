"""Base classes and registry for document extractors."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Centralized file type mapping (single source of truth) ──────────

FILE_TYPE_MAP: Dict[str, List[str]] = {
    "text": [
        ".txt", ".md", ".rst", ".log", ".csv", ".json",
        ".xml", ".yaml", ".yml", ".css", ".ini", ".cfg", ".toml",
    ],
    "pdf": [".pdf"],
    "image": [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"],
    "html": [".html", ".htm"],
    "docx": [".docx", ".doc"],
    "pptx": [".pptx", ".ppt"],
    "xlsx": [".xlsx", ".xls"],
    "code": [
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp",
        ".cxx", ".cc", ".h", ".hpp", ".cs", ".go", ".rs", ".rb",
        ".php", ".scala", ".kt", ".swift", ".r", ".m", ".mm",
        ".lua", ".pl", ".pm", ".sh", ".bash", ".zsh", ".fish",
        ".ps1", ".bat", ".cmd", ".clj", ".cljs", ".ex", ".exs",
        ".erl", ".hrl", ".hs", ".lhs", ".ml", ".mli", ".fs",
        ".fsi", ".fsx", ".vb", ".pas", ".pp", ".inc", ".asm",
        ".s", ".sql", ".elm", ".nim", ".cr", ".d", ".dart",
        ".groovy", ".jl", ".lisp", ".scm", ".rkt", ".tcl",
        ".v", ".sv", ".svh", ".vhd", ".vhdl", ".cob", ".cbl", ".tex",
    ],
}

# Reverse lookup: extension → file type
_EXTENSION_TO_TYPE: Dict[str, str] = {}
for _ftype, _exts in FILE_TYPE_MAP.items():
    for _ext in _exts:
        _EXTENSION_TO_TYPE[_ext] = _ftype


def detect_file_type_from_extension(filename: str) -> str:
    """Detect file type from filename extension.

    Args:
        filename: Filename or path string.

    Returns:
        File type string (e.g., "pdf", "text", "code").
    """
    ext = os.path.splitext(filename)[1].lower()
    return _EXTENSION_TO_TYPE.get(ext, "unknown")


def get_supported_extensions() -> List[str]:
    """Get all supported file extensions."""
    return list(_EXTENSION_TO_TYPE.keys())


def is_extension_supported(filename: str) -> bool:
    """Check if a file extension is supported."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in _EXTENSION_TO_TYPE


# ── Base Extractor ABC ──────────────────────────────────────────────


class BaseExtractor(ABC):
    """Abstract base class for document extractors.

    All extractors must implement extract() and aextract().
    """

    @abstractmethod
    def extract(self, file_path: Path, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Extract text content from a file synchronously.

        Args:
            file_path: Path to the file to extract.

        Returns:
            Tuple of (extracted_text, metadata_dict).
        """
        ...

    async def aextract(self, file_path: Path, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Extract text content asynchronously.

        Default: runs sync extract() in a thread pool.
        Override for true async implementations.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.extract(file_path, **kwargs))


# ── Extractor Registry ──────────────────────────────────────────────


class ExtractorRegistry:
    """Registry that maps file types to extractor instances.

    Usage:
        registry = ExtractorRegistry(config)
        text, meta = registry.extract(Path("doc.pdf"))
        text, meta = await registry.aextract(Path("doc.pdf"))
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._extractors: Dict[str, BaseExtractor] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register built-in extractors (lazy import)."""
        try:
            from llm_utils_parser.text_extractor import TextExtractor
            self._extractors["text"] = TextExtractor(self.config)
        except ImportError:
            pass

        pdf_cfg = self.config.get("extraction", {}).get("pdf", {})
        if not pdf_cfg:
            pdf_cfg = self.config.get("pdf", {})
        pdf_extractor_type = pdf_cfg.get("extractor", "full")
        logger.info("ExtractorRegistry: pdf extractor type = %s", pdf_extractor_type)
        if pdf_extractor_type == "ocr":
            try:
                from llm_utils_parser.ocr_extractor import OCRExtractor
                self._extractors["pdf"] = OCRExtractor(self.config)
            except ImportError:
                pass
        elif pdf_extractor_type == "pymupdf":
            try:
                from llm_utils_parser.pymupdf_extractor import PyMuPDFExtractor
                self._extractors["pdf"] = PyMuPDFExtractor(self.config)
            except ImportError:
                pass
        else:
            try:
                from llm_utils_parser.pdf_extractor import PDFExtractor
                self._extractors["pdf"] = PDFExtractor(self.config)
            except ImportError:
                pass

        try:
            from llm_utils_parser.html_extractor import HTMLExtractor
            self._extractors["html"] = HTMLExtractor(self.config)
        except ImportError:
            pass

        try:
            from llm_utils_parser.office_extractor import OfficeExtractor
            office = OfficeExtractor(self.config)
            self._extractors["docx"] = office
            self._extractors["pptx"] = office
            self._extractors["xlsx"] = office
        except ImportError:
            pass

        try:
            from llm_utils_parser.code_extractor import CodeExtractor
            self._extractors["code"] = CodeExtractor(self.config)
        except ImportError:
            pass

        try:
            from llm_utils_parser.ocr_extractor import OCRExtractor
            self._extractors["image"] = OCRExtractor(self.config)
        except ImportError:
            pass

    def register(self, file_type: str, extractor: BaseExtractor):
        """Register a custom extractor for a file type."""
        self._extractors[file_type] = extractor

    def get_extractor(self, file_type: str) -> Optional[BaseExtractor]:
        """Get extractor for a file type."""
        return self._extractors.get(file_type)

    def extract(self, file_path: Path, file_type: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Extract text from a file using the appropriate extractor.

        Args:
            file_path: Path to file.
            file_type: Override file type detection.

        Returns:
            Tuple of (text, metadata).
        """
        if file_type is None:
            file_type = detect_file_type_from_extension(file_path.name)

        extractor = self._extractors.get(file_type)
        if extractor is None:
            # Fallback to text extractor
            extractor = self._extractors.get("text")
            if extractor is None:
                raise ValueError(f"No extractor for file type: {file_type}")
            logger.warning("No extractor for '%s', falling back to text", file_type)

        return extractor.extract(file_path)

    async def aextract(self, file_path: Path, file_type: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Extract text asynchronously."""
        if file_type is None:
            file_type = detect_file_type_from_extension(file_path.name)

        extractor = self._extractors.get(file_type)
        if extractor is None:
            extractor = self._extractors.get("text")
            if extractor is None:
                raise ValueError(f"No extractor for file type: {file_type}")
            logger.warning("No extractor for '%s', falling back to text", file_type)

        return await extractor.aextract(file_path)
