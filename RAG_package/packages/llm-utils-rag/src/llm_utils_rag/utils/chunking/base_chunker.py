"""Unified base interface for all document chunkers.

All chunker implementations should extend BaseChunker to ensure
consistent metadata enrichment and interface compatibility.
"""

from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """Abstract base class for document chunkers.

    Provides:
    - Unified split interface (sync + async)
    - Standard metadata enrichment for all chunk types
    - Chunk statistics collection

    Subclasses must implement split_documents() and asplit_documents().
    """

    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks synchronously.

        Args:
            documents: List of documents to split.

        Returns:
            List of chunked Document objects with enriched metadata.
        """
        ...

    @abstractmethod
    async def asplit_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks asynchronously.

        Args:
            documents: List of documents to split.

        Returns:
            List of chunked Document objects with enriched metadata.
        """
        ...

    def _enrich_metadata(
        self,
        chunk: Document,
        index: int,
        method: str,
        source_metadata: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """Apply standard metadata enrichment to a chunk.

        Args:
            chunk: The chunk Document to enrich.
            index: Chunk index within the parent document.
            method: Chunking method name (e.g., "semantic", "standard").
            source_metadata: Original document metadata to merge.
            extra: Additional metadata key-value pairs.

        Returns:
            The chunk with enriched metadata.
        """
        if source_metadata:
            merged = copy.deepcopy(source_metadata)
            merged.update(chunk.metadata)
            chunk.metadata = merged

        chunk.metadata.update({
            "chunk_id": index,
            "chunk_method": method,
            "chunk_size": len(chunk.page_content),
            "word_count": len(chunk.page_content.split()),
        })

        if extra:
            chunk.metadata.update(extra)

        return chunk

    @staticmethod
    def compute_stats(chunks: List[Document]) -> Dict[str, Any]:
        """Compute statistics for a list of chunks.

        Args:
            chunks: List of chunked documents.

        Returns:
            Dict with total_chunks, avg/min/max chunk sizes, total_chars.
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "total_chars": 0,
            }

        sizes = [len(c.page_content) for c in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": round(sum(sizes) / len(sizes)),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "total_chars": sum(sizes),
        }
