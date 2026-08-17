"""RAG utils — thin re-export layer for backward compatibility.

Internal code is organized into sub-packages:
  utils/core/          — metrics, hardening, shared infra
  utils/chunking/      — all chunker strategies
  utils/extraction/    — file extractors, text cleaner, OCR
  utils/retrieval/     — retrieve chains, reranking
"""

# ── Chunking ───────────────────────────────────────────────────────
from .chunking import (
    BaseChunker,
    SemanticChunkerCustom,
    CodeChunkerWrapper,
    ContextualChunkerWrapper,
    MarkdownChunkerWrapper,
)

# ── Extraction & Cleaning ──────────────────────────────────────────
from .file_utils import (
    extract_text_from_file,
    extract_file_content_async,
    is_supported_file,
)
from llm_utils_parser import TextCleaner


# ── Retrieval ──────────────────────────────────────────────────────
from .retrieval import (
    CategoryAwareRetriever,
    SemanticRAGChain,
    create_semantic_rag_chain,
    RerankerFactory,
    rerank_documents,
)

# ── Core ───────────────────────────────────────────────────────────
from .core import (
    PipelineTimer,
    MetricsCollector,
    SemanticRAGMetrics,
    InputValidator,
)

__all__ = [
    # Chunking
    "BaseChunker",
    "SemanticChunkerCustom",
    "CodeChunkerWrapper",
    "ContextualChunkerWrapper",
    "MarkdownChunkerWrapper",
    # Extraction
    "extract_text_from_file",
    "extract_file_content_async",
    "is_supported_file",
    "TextCleaner",
    # Retrieval
    "CategoryAwareRetriever",
    "SemanticRAGChain",
    "create_semantic_rag_chain",
    "RerankerFactory",
    "rerank_documents",
    # Core
    "PipelineTimer",
    "MetricsCollector",
    "SemanticRAGMetrics",
    "InputValidator",
]