"""Factory for creating reranker instances for RAG applications.

Exposes RerankerFactory and rerank_documents from llm_utils_reranker for compatibility.
"""

from llm_utils_reranker import RerankerFactory, rerank_documents

__all__ = [
    "RerankerFactory",
    "rerank_documents",
]
