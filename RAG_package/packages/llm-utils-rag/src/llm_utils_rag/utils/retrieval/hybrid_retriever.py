"""Hybrid retriever — merges BM25 (sparse) + Dense (vector) results via RRF.

Reciprocal Rank Fusion (RRF) is a rank-aggregation algorithm that combines
ranked lists from multiple retrieval systems without needing score calibration.

Formula: RRF_score(d) = Σ 1 / (k + rank_i(d))
    where k=60 (constant), rank_i is the rank of doc d in retriever i.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining sparse (BM25) and dense (vector) retrieval.

    Uses Reciprocal Rank Fusion to merge results from both retrievers.

    Usage:
        hybrid = HybridRetriever(
            sparse_retriever=bm25_retriever,
            dense_retriever=vector_retriever,
            sparse_weight=0.3,
            dense_weight=0.7,
            k=5,
        )
        results = hybrid.invoke("my query")
    """

    sparse_retriever: BaseRetriever
    dense_retriever: BaseRetriever
    sparse_weight: float = 0.3
    dense_weight: float = 0.7
    k: int = 5
    rrf_k: int = 60  # RRF constant

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Retrieve from both sources and merge via weighted RRF."""
        # Retrieve from both
        try:
            sparse_docs = self.sparse_retriever.invoke(query)
        except Exception as e:
            logger.warning("BM25 retrieval failed: %s, using dense only", e)
            sparse_docs = []

        try:
            dense_docs = self.dense_retriever.invoke(query)
        except Exception as e:
            logger.warning("Dense retrieval failed: %s, using sparse only", e)
            dense_docs = []

        # Merge via weighted RRF
        merged = self._reciprocal_rank_fusion(
            sparse_docs, dense_docs
        )

        logger.info(
            "Hybrid retrieval: sparse=%d, dense=%d → merged=%d (returning top %d)",
            len(sparse_docs),
            len(dense_docs),
            len(merged),
            min(self.k, len(merged)),
        )
        return merged[: self.k]

    def _reciprocal_rank_fusion(
        self,
        sparse_docs: List[Document],
        dense_docs: List[Document],
    ) -> List[Document]:
        """Merge two ranked lists using weighted Reciprocal Rank Fusion.

        RRF_score(d) = w_sparse / (k + rank_sparse(d)) + w_dense / (k + rank_dense(d))
        """
        # Build score map: content_hash → (doc, rrf_score)
        doc_scores: Dict[int, tuple] = {}  # hash → (Document, score)

        # Score sparse results
        for rank, doc in enumerate(sparse_docs):
            content_hash = hash(doc.page_content.strip())
            rrf_score = self.sparse_weight / (self.rrf_k + rank + 1)
            if content_hash in doc_scores:
                existing_doc, existing_score = doc_scores[content_hash]
                doc_scores[content_hash] = (existing_doc, existing_score + rrf_score)
            else:
                doc_copy = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata},
                )
                doc_scores[content_hash] = (doc_copy, rrf_score)

        # Score dense results
        for rank, doc in enumerate(dense_docs):
            content_hash = hash(doc.page_content.strip())
            rrf_score = self.dense_weight / (self.rrf_k + rank + 1)
            if content_hash in doc_scores:
                existing_doc, existing_score = doc_scores[content_hash]
                # Merge metadata from dense (may have similarity_score)
                merged_meta = {**existing_doc.metadata, **doc.metadata}
                merged_doc = Document(
                    page_content=existing_doc.page_content,
                    metadata=merged_meta,
                )
                doc_scores[content_hash] = (merged_doc, existing_score + rrf_score)
            else:
                doc_copy = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata},
                )
                doc_scores[content_hash] = (doc_copy, rrf_score)

        # Sort by RRF score descending
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Attach hybrid scores to metadata
        results = []
        for doc, score in sorted_docs:
            doc.metadata["hybrid_rrf_score"] = round(score, 6)
            doc.metadata["retrieval_method"] = "hybrid"
            results.append(doc)

        return results
