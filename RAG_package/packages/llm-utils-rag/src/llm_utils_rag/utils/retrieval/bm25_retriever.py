"""BM25 sparse retriever for hybrid search.

Uses rank_bm25 for keyword-based retrieval as the sparse component
in a hybrid (sparse + dense) retrieval pipeline.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer with lowercasing."""
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    return tokens


class BM25Retriever(BaseRetriever):
    """BM25 keyword retriever backed by rank_bm25.

    Builds an in-memory BM25 index from a list of LangChain Documents.
    Designed to be used alongside a dense vector retriever for hybrid search.

    Usage:
        docs = vectorstore.similarity_search("", k=10000)  # all docs
        retriever = BM25Retriever.from_documents(docs, k=5)
        results = retriever.invoke("exact keyword query")
    """

    documents: List[Document] = []
    k: int = 5
    bm25_params: Dict[str, float] = {"k1": 1.5, "b": 0.75}

    # Private attrs — not serialized
    _index: Any = None
    _tokenized_corpus: List[List[str]] = []

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        k: int = 5,
        bm25_params: Optional[Dict[str, float]] = None,
    ) -> "BM25Retriever":
        """Build a BM25 index from a list of documents."""
        instance = cls(
            documents=documents,
            k=k,
            bm25_params=bm25_params or {"k1": 1.5, "b": 0.75},
        )
        instance._build_index()
        return instance

    @classmethod
    def from_vectorstore(
        cls,
        vectorstore: Any,
        k: int = 5,
        fetch_k: int = 10000,
        bm25_params: Optional[Dict[str, float]] = None,
    ) -> "BM25Retriever":
        """Build BM25 index by fetching all docs from a vectorstore.

        Args:
            vectorstore: LangChain VectorStore instance.
            k: Number of results to return per query.
            fetch_k: Max docs to fetch from vectorstore for indexing.
            bm25_params: BM25 tuning parameters.
        """
        # Fetch all documents from vectorstore
        try:
            docs = vectorstore.similarity_search("", k=fetch_k)
        except Exception:
            logger.warning("similarity_search('') failed, trying get()")
            try:
                result = vectorstore.get()
                docs = [
                    Document(page_content=text, metadata=meta)
                    for text, meta in zip(
                        result.get("documents", []),
                        result.get("metadatas", [{}] * len(result.get("documents", []))),
                    )
                ]
            except Exception as e:
                logger.error("Failed to fetch docs from vectorstore: %s", e)
                docs = []

        logger.info("BM25: indexed %d documents from vectorstore", len(docs))
        return cls.from_documents(docs, k=k, bm25_params=bm25_params)

    def _build_index(self):
        """Build the BM25 index from stored documents."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank_bm25 is required for BM25 retrieval. "
                "Install with: pip install rank-bm25"
            )

        self._tokenized_corpus = [_tokenize(doc.page_content) for doc in self.documents]
        self._index = BM25Okapi(
            self._tokenized_corpus,
            k1=self.bm25_params.get("k1", 1.5),
            b=self.bm25_params.get("b", 0.75),
        )
        logger.info("BM25 index built: %d documents", len(self.documents))

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Retrieve documents matching the query using BM25."""
        if self._index is None:
            self._build_index()

        if not self.documents:
            return []

        tokenized_query = _tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        # Pair docs with scores, sort descending
        scored = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for doc, score in scored[: self.k]:
            if score > 0:
                # Attach BM25 score to metadata
                doc_copy = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "bm25_score": float(score)},
                )
                results.append(doc_copy)

        logger.debug("BM25 returned %d docs for query: %s", len(results), query[:50])
        return results
