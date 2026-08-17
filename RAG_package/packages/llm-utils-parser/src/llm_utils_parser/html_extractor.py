"""HTML file extractor — .html, .htm."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from llm_utils_parser.base import BaseExtractor

logger = logging.getLogger(__name__)


class HTMLExtractor(BaseExtractor):
    """Extract text from HTML files with fallback chain."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def extract(self, file_path: Path, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Extract text using UnstructuredHTMLLoader → BSHTMLLoader fallback."""
        metadata = {
            "extractor": "html",
            "file_size": file_path.stat().st_size,
        }

        # Try UnstructuredHTMLLoader first
        try:
            from langchain_community.document_loaders import UnstructuredHTMLLoader
            loader = UnstructuredHTMLLoader(str(file_path))
            documents = loader.load()
            content = "\n\n".join(doc.page_content for doc in documents)
            metadata["loader"] = "unstructured"
            return content, metadata
        except Exception as e:
            logger.debug("UnstructuredHTMLLoader failed: %s", e)

        # Fallback to BSHTMLLoader
        try:
            from langchain_community.document_loaders import BSHTMLLoader
            loader = BSHTMLLoader(str(file_path))
            documents = loader.load()
            content = "\n\n".join(doc.page_content for doc in documents)
            metadata["loader"] = "beautifulsoup"
            return content, metadata
        except Exception as e2:
            raise RuntimeError(f"HTML extraction failed: {e2}")
