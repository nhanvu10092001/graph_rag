"""Retrieval sub-package — RAG retrieval chains, reranking, and advanced retrieval."""

from .retrieve_utils import CategoryAwareRetriever, SemanticRAGChain, create_semantic_rag_chain
# Re-export from factories (canonical location) for backward compat
from ...factories.reranker_factory import RerankerFactory, rerank_documents
from .bm25_retriever import BM25Retriever
from .hybrid_retriever import HybridRetriever
from .query_transforms import QueryTransformPipeline
from .adaptive_rag import AdaptiveRAGPipeline

__all__ = [
    "CategoryAwareRetriever",
    "SemanticRAGChain",
    "create_semantic_rag_chain",
    "RerankerFactory",
    "rerank_documents",
    "BM25Retriever",
    "HybridRetriever",
    "QueryTransformPipeline",
    "AdaptiveRAGPipeline",
]

