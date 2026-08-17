"""Code language-aware chunker.

Uses LangChain's language-specific separators to split source code
by functions, classes, and logical blocks rather than arbitrary character counts.
"""

import copy
import logging
import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from .base_chunker import BaseChunker

logger = logging.getLogger(__name__)

# Map string names → LangChain Language enum
LANGUAGE_MAP = {
    "python": Language.PYTHON,
    "javascript": Language.JS,
    "js": Language.JS,
    "typescript": Language.TS,
    "ts": Language.TS,
    "java": Language.JAVA,
    "go": Language.GO,
    "rust": Language.RUST,
    "ruby": Language.RUBY,
    "c": Language.C,
    "cpp": Language.CPP,
    "csharp": Language.CSHARP,
    "scala": Language.SCALA,
    "swift": Language.SWIFT,
    "markdown": Language.MARKDOWN,
    "latex": Language.LATEX,
    "html": Language.HTML,
    "sol": Language.SOL,
    "php": Language.PHP,
    "haskell": Language.HASKELL,
    "elixir": Language.ELIXIR,
    "kotlin": Language.KOTLIN,
    "lua": Language.LUA,
    "perl": Language.PERL,
    "powershell": Language.POWERSHELL,
    "proto": Language.PROTO,
    "rst": Language.RST,
    "cobol": Language.COBOL,
}

# File extension → language name
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".cs": "csharp",
    ".scala": "scala",
    ".swift": "swift",
    ".tex": "latex",
    ".html": "html",
    ".htm": "html",
    ".sol": "sol",
    ".php": "php",
    ".hs": "haskell",
    ".ex": "elixir",
    ".exs": "elixir",
    ".kt": "kotlin",
    ".lua": "lua",
    ".pl": "perl",
    ".pm": "perl",
    ".ps1": "powershell",
    ".proto": "proto",
    ".rst": "rst",
    ".cob": "cobol",
}


class CodeChunkerWrapper(BaseChunker):
    """Structure-aware chunker for source code.

    Usage:
        chunker = CodeChunkerWrapper(language="python", chunk_size=1500)
        chunks = chunker.split_documents(docs)

        # Auto-detect language from filename
        chunker = CodeChunkerWrapper(auto_detect_language=True)
    """

    def __init__(
        self,
        language: str = "python",
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        auto_detect_language: bool = False,
    ):
        self._default_language = language
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._auto_detect = auto_detect_language
        self._splitter = self._create_splitter(language)

    def _create_splitter(self, language: str) -> RecursiveCharacterTextSplitter:
        """Create a language-specific splitter."""
        lang_enum = LANGUAGE_MAP.get(language.lower())
        if lang_enum is None:
            logger.warning(
                "Language '%s' not supported, falling back to standard splitting.",
                language,
            )
            return RecursiveCharacterTextSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
        return RecursiveCharacterTextSplitter.from_language(
            language=lang_enum,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

    def _detect_language(self, doc: Document) -> str:
        """Detect programming language from document metadata."""
        filename = doc.metadata.get("filename", "")
        if filename:
            ext = os.path.splitext(filename)[1].lower()
            detected = EXTENSION_TO_LANGUAGE.get(ext)
            if detected:
                logger.info("Auto-detected language '%s' from file '%s'", detected, filename)
                return detected
        return self._default_language

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split code documents using language-aware separators."""
        all_chunks = []

        for doc in documents:
            language = self._detect_language(doc) if self._auto_detect else self._default_language
            splitter = self._create_splitter(language) if self._auto_detect else self._splitter
            chunks = splitter.split_documents([doc])

            for i, chunk in enumerate(chunks):
                self._enrich_metadata(
                    chunk, index=i, method="code",
                    source_metadata=doc.metadata,
                    extra={"code_language": language},
                )
                all_chunks.append(chunk)

        return all_chunks

    async def asplit_documents(self, documents: List[Document]) -> List[Document]:
        """Async version — code splitting is CPU-bound, so same as sync."""
        return self.split_documents(documents)
