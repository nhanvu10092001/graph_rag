"""Factory for creating chunker instances for RAG applications.

Supports auto-routing by file type to select the optimal chunker strategy.
"""

import logging
import os
from typing import Any, Dict, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLLM

logger = logging.getLogger(__name__)

# File extension → chunker provider mapping for auto-routing
FILE_TYPE_ROUTES: Dict[str, str] = {
    # Markdown
    ".md": "markdown",
    ".mdx": "markdown",
    # Code
    ".py": "code",
    ".js": "code",
    ".jsx": "code",
    ".ts": "code",
    ".tsx": "code",
    ".java": "code",
    ".go": "code",
    ".rs": "code",
    ".rb": "code",
    ".c": "code",
    ".h": "code",
    ".cpp": "code",
    ".hpp": "code",
    ".cs": "code",
    ".kt": "code",
    ".swift": "code",
    ".php": "code",
    ".lua": "code",
    ".scala": "code",
    ".ex": "code",
    ".exs": "code",
    # Documents → semantic
    ".pdf": "semantic",
    ".docx": "semantic",
    ".doc": "semantic",
    ".pptx": "semantic",
    # Plain text → standard
    ".txt": "standard",
    ".csv": "standard",
    ".log": "standard",
    # Web → markdown (HTML has header structure)
    ".html": "markdown",
    ".htm": "markdown",
    # LaTeX
    ".tex": "code",
    ".rst": "code",
}


class ChunkerFactory:
    """Factory for creating chunker instances for RAG applications.

    Supports:
    - Manual provider selection: "semantic", "standard", "contextual", "markdown", "code"
    - Auto-routing: "auto" — selects chunker based on file extension
    """

    _providers = {}

    @classmethod
    def register_provider(cls, name: str, factory_func):
        """Register a new chunker provider factory function."""
        cls._providers[name.lower()] = factory_func

    @classmethod
    def create_chunker(
        cls,
        config: Dict[str, Any],
        embeddings: Embeddings,
        llm: Optional[BaseLLM] = None,
        filename: Optional[str] = None,
        **kwargs: Any,
    ):
        """Create a chunker instance based on configuration.

        Args:
            config: Chunking configuration from rag_config.yaml.
            embeddings: Embeddings model for semantic chunking.
            llm: LLM for category detection / contextual chunking.
            filename: Source filename for auto-routing.
            **kwargs: Additional arguments.
        """
        provider = config.get("provider", "standard").lower()

        # Auto-routing: detect provider from filename
        if provider == "auto":
            provider = cls._resolve_auto_provider(config, filename)
            logger.info("Auto-routed chunker: '%s' for file '%s'", provider, filename or "unknown")

        provider_config = config.get(provider, {}).copy()

        if provider not in cls._providers:
            cls._try_register_common_providers()

        if provider not in cls._providers:
            raise ValueError(f"Unsupported chunker provider: {provider}")

        # Pass common arguments and provider-specific config as kwargs
        return cls._providers[provider](
            embeddings=embeddings,
            llm=llm,
            config=provider_config,
            **kwargs,
        )

    @classmethod
    def _resolve_auto_provider(cls, config: Dict[str, Any], filename: Optional[str]) -> str:
        """Resolve chunker provider from filename extension."""
        auto_config = config.get("auto", {})
        default_fallback = auto_config.get("default_fallback", "semantic")

        if not filename:
            return default_fallback

        ext = os.path.splitext(filename)[1].lower()
        resolved = FILE_TYPE_ROUTES.get(ext, default_fallback)
        return resolved

    @classmethod
    def _try_register_common_providers(cls):
        """Register common chunker providers."""

        # Try Semantic Chunker
        try:
            from ..utils.chunking.chunk_utils import SemanticChunkerCustom

            def create_semantic_chunker(
                embeddings: Embeddings,
                llm: Optional[BaseLLM] = None,
                config: Optional[Dict[str, Any]] = None,
                **kwargs,
            ):
                config = config or {}
                return SemanticChunkerCustom(
                    embeddings=embeddings,
                    llm=llm,
                    breakpoint_threshold_type=config.get("breakpoint_threshold_type", "percentile"),
                    breakpoint_threshold_amount=config.get("breakpoint_threshold_amount", None),
                    number_of_chunks=config.get("number_of_chunks", None),
                    enable_category_detection=config.get("enable_category_detection", True),
                    embedding_batch_size=config.get("embedding_batch_size", 20),
                    **kwargs
                )

            cls.register_provider("semantic", create_semantic_chunker)

        except ImportError:
            pass

        # Try Standard RecursiveCharacterTextSplitter
        try:
            from ..utils.chunking.base_chunker import BaseChunker

            from langchain_text_splitters import RecursiveCharacterTextSplitter

            class StandardChunkerWrapper(BaseChunker):
                """Wraps RecursiveCharacterTextSplitter with BaseChunker interface."""

                def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
                    self._splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )

                def split_documents(self, documents):
                    all_chunks = []
                    for doc in documents:
                        chunks = self._splitter.split_documents([doc])
                        for i, chunk in enumerate(chunks):
                            self._enrich_metadata(
                                chunk, index=i, method="standard",
                                source_metadata=doc.metadata,
                            )
                            all_chunks.append(chunk)
                    return all_chunks

                async def asplit_documents(self, documents):
                    return self.split_documents(documents)

            def create_standard_chunker(
                embeddings: Embeddings = None,
                llm: Optional[BaseLLM] = None,
                config: Optional[Dict[str, Any]] = None,
                **kwargs,
            ):
                config = config or {}
                return StandardChunkerWrapper(
                    chunk_size=config.get("chunk_size", 1000),
                    chunk_overlap=config.get("chunk_overlap", 200),
                )

            cls.register_provider("standard", create_standard_chunker)

        except ImportError:
            pass

        # Contextual Chunker (Late Chunking)
        try:
            def create_contextual_chunker(
                embeddings: Embeddings = None,
                llm: Optional[BaseLLM] = None,
                config: Optional[Dict[str, Any]] = None,
                **kwargs,
            ):
                from ..utils.chunking.contextual_chunker import ContextualChunkerWrapper

                config = config or {}
                return ContextualChunkerWrapper(
                    llm=llm,
                    chunk_size=config.get("chunk_size", 1000),
                    chunk_overlap=config.get("chunk_overlap", 200),
                    context_prompt=config.get("context_prompt", None),
                )

            cls.register_provider("contextual", create_contextual_chunker)

        except ImportError:
            pass

        # Markdown Header-Aware Chunker
        try:
            def create_markdown_chunker(
                embeddings: Embeddings = None,
                llm: Optional[BaseLLM] = None,
                config: Optional[Dict[str, Any]] = None,
                **kwargs,
            ):
                from ..utils.chunking.markdown_chunker import MarkdownChunkerWrapper

                config = config or {}
                return MarkdownChunkerWrapper(
                    chunk_size=config.get("chunk_size", 1000),
                    chunk_overlap=config.get("chunk_overlap", 200),
                    strip_headers=config.get("strip_headers", False),
                )

            cls.register_provider("markdown", create_markdown_chunker)

        except ImportError:
            pass

        # Code Language-Aware Chunker
        try:
            def create_code_chunker(
                embeddings: Embeddings = None,
                llm: Optional[BaseLLM] = None,
                config: Optional[Dict[str, Any]] = None,
                **kwargs,
            ):
                from ..utils.chunking.code_chunker import CodeChunkerWrapper

                config = config or {}
                return CodeChunkerWrapper(
                    language=config.get("language", "python"),
                    chunk_size=config.get("chunk_size", 1500),
                    chunk_overlap=config.get("chunk_overlap", 200),
                    auto_detect_language=config.get("auto_detect_language", True),
                )

            cls.register_provider("code", create_code_chunker)

        except ImportError:
            pass

        # Parent-Child Hierarchical Chunker
        try:
            def create_parent_child_chunker(
                embeddings: Embeddings = None,
                llm: Optional[BaseLLM] = None,
                config: Optional[Dict[str, Any]] = None,
                **kwargs,
            ):
                from ..utils.chunking.parent_child_chunker import ParentChildChunker

                config = config or {}
                return ParentChildChunker(
                    parent_chunk_size=config.get("parent_chunk_size", 2000),
                    parent_chunk_overlap=config.get("parent_chunk_overlap", 200),
                    child_chunk_size=config.get("child_chunk_size", 500),
                    child_chunk_overlap=config.get("child_chunk_overlap", 50),
                    config=config,
                )

            cls.register_provider("parent_child", create_parent_child_chunker)

        except ImportError:
            pass