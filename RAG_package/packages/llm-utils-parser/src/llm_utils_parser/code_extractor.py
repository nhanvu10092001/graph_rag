"""Source code extractor — .py, .js, .ts, .java, etc."""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from llm_utils_parser.base import BaseExtractor

logger = logging.getLogger(__name__)


class CodeExtractor(BaseExtractor):
    """Extract content from source code files using LangChain LanguageParser."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def extract(self, file_path: Path, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Extract code content with language-aware parsing."""
        metadata = {
            "extractor": "code",
            "file_size": file_path.stat().st_size,
            "code_language": file_path.suffix.lower().lstrip("."),
        }

        try:
            content = self._extract_with_parser(file_path)
        except Exception as e:
            logger.debug("LanguageParser failed: %s, using plain text", e)
            content = self._extract_plain(file_path)

        return content, metadata

    def _extract_with_parser(self, file_path: Path) -> str:
        """Use LangChain LanguageParser for structured extraction."""
        from langchain_community.document_loaders.generic import GenericLoader
        from langchain_community.document_loaders.parsers import LanguageParser
        from langchain_text_splitters import Language

        ext_to_lang = {
            ".py": Language.PYTHON, ".js": Language.JS, ".ts": Language.TS,
            ".jsx": Language.JS, ".tsx": Language.TS, ".java": Language.JAVA,
            ".c": Language.C, ".cpp": Language.CPP, ".h": Language.C,
            ".hpp": Language.CPP, ".cs": Language.CSHARP, ".go": Language.GO,
            ".rs": Language.RUST, ".rb": Language.RUBY, ".php": Language.PHP,
            ".scala": Language.SCALA, ".kt": Language.KOTLIN,
            ".swift": Language.SWIFT, ".lua": Language.LUA,
            ".pl": Language.PERL, ".ex": Language.ELIXIR,
            ".hs": Language.HASKELL, ".cob": Language.COBOL,
            ".ps1": Language.POWERSHELL, ".tex": Language.LATEX,
            ".rst": Language.RST, ".proto": Language.PROTO,
            ".sol": Language.SOL, ".html": Language.HTML,
        }

        ext = file_path.suffix.lower()
        language = ext_to_lang.get(ext)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_file = os.path.join(tmp, file_path.name)
            shutil.copy2(file_path, tmp_file)

            parser = LanguageParser(
                language=language, parser_threshold=0
            )
            loader = GenericLoader.from_filesystem(
                tmp, glob=file_path.name, parser=parser
            )
            documents = loader.load()

            parts = []
            for doc in documents:
                doc_type = doc.metadata.get("content_type", "code")
                if doc_type and doc_type != "code":
                    parts.append(f"=== {doc_type.upper()} ===\n{doc.page_content}")
                else:
                    parts.append(doc.page_content)

            return "\n\n".join(parts) if parts else ""

    @staticmethod
    def _extract_plain(file_path: Path) -> str:
        """Fallback: plain text extraction with encoding detection."""
        for enc in ["utf-8", "latin1", "cp1252"]:
            try:
                return file_path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        return file_path.read_text(encoding="utf-8", errors="ignore")
