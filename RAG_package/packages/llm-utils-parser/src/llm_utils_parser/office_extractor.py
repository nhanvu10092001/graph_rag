"""Office file extractor — .docx, .pptx, .xlsx."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from llm_utils_parser.base import BaseExtractor

logger = logging.getLogger(__name__)


class OfficeExtractor(BaseExtractor):
    """Extract text from Microsoft Office files (Word, PowerPoint, Excel)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def extract(self, file_path: Path, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Route to the appropriate Office extractor based on extension."""
        ext = file_path.suffix.lower()
        metadata = {
            "extractor": "office",
            "file_size": file_path.stat().st_size,
            "office_type": ext.lstrip("."),
        }

        if ext in (".docx", ".doc"):
            content = self._extract_docx(file_path)
        elif ext in (".pptx", ".ppt"):
            content = self._extract_pptx(file_path)
        elif ext in (".xlsx", ".xls"):
            content = self._extract_xlsx(file_path)
        else:
            raise ValueError(f"Unsupported office format: {ext}")

        return content, metadata

    @staticmethod
    def _extract_docx(file_path: Path) -> str:
        """Extract text from Word documents."""
        from langchain_community.document_loaders import UnstructuredWordDocumentLoader
        loader = UnstructuredWordDocumentLoader(str(file_path))
        documents = loader.load()
        return "\n\n".join(doc.page_content for doc in documents)

    @staticmethod
    def _extract_pptx(file_path: Path) -> str:
        """Extract text from PowerPoint presentations."""
        from langchain_community.document_loaders import UnstructuredPowerPointLoader
        loader = UnstructuredPowerPointLoader(str(file_path))
        documents = loader.load()
        return "\n\n".join(doc.page_content for doc in documents)

    @staticmethod
    def _extract_xlsx(file_path: Path) -> str:
        """Extract text from Excel spreadsheets."""
        from langchain_community.document_loaders import UnstructuredExcelLoader
        loader = UnstructuredExcelLoader(str(file_path))
        documents = loader.load()
        return "\n\n".join(doc.page_content for doc in documents)
