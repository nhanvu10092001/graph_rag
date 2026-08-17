"""Parent-Child chunker — creates hierarchical chunks for better context retrieval.

Creates a two-level hierarchy:
- **Parent chunks** (large): Provide broad context for LLM generation
- **Child chunks** (small): Optimized for precise vector similarity search

At query time, child chunks are retrieved via vector search, then expanded
to their parent chunk before being sent to the LLM, giving much richer context.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base_chunker import BaseChunker

logger = logging.getLogger(__name__)


class ParentChildChunker(BaseChunker):
    """Hierarchical chunker that creates parent/child document pairs.

    Usage:
        chunker = ParentChildChunker(
            parent_chunk_size=2000,
            parent_chunk_overlap=200,
            child_chunk_size=500,
            child_chunk_overlap=50,
        )
        # Returns CHILD chunks (with parent_id in metadata)
        child_chunks = chunker.split_documents(docs)

        # At query time, use expand_to_parents() to get full parent context
        parent_docs = chunker.expand_to_parents(retrieved_child_chunks)
    """

    def __init__(
        self,
        parent_chunk_size: int = 2000,
        parent_chunk_overlap: int = 200,
        child_chunk_size: int = 500,
        child_chunk_overlap: int = 50,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.config = config or {}

        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
        )

        # In-memory parent store for expansion at query time
        # Key: parent_id → Document
        self._parent_store: Dict[str, Document] = {}

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into parent/child hierarchy.

        Returns CHILD chunks (small, optimized for embedding search).
        Parent chunks are stored internally for expansion at query time.
        """
        all_children = []

        for doc in documents:
            source_meta = doc.metadata.copy()

            # Step 1: Create parent chunks
            parent_chunks = self._parent_splitter.split_documents([doc])

            for parent_idx, parent in enumerate(parent_chunks):
                parent_id = self._generate_parent_id(parent.page_content, parent_idx)

                # Store parent for later expansion
                parent_meta = {
                    **source_meta,
                    "parent_id": parent_id,
                    "chunk_method": "parent_child",
                    "hierarchy_level": "parent",
                    "parent_index": parent_idx,
                    "parent_chunk_size": len(parent.page_content),
                    "total_parents": len(parent_chunks),
                }
                parent_doc = Document(
                    page_content=parent.page_content,
                    metadata=parent_meta,
                )
                self._parent_store[parent_id] = parent_doc

                # Step 2: Split parent into child chunks
                child_chunks = self._child_splitter.split_documents(
                    [Document(page_content=parent.page_content, metadata={})]
                )

                for child_idx, child in enumerate(child_chunks):
                    child_id = f"{parent_id}_child_{child_idx}"
                    enriched = self._enrich_metadata(
                        chunk=child,
                        index=len(all_children),
                        method="parent_child",
                        source_metadata=source_meta,
                        extra={
                            "parent_id": parent_id,
                            "child_id": child_id,
                            "hierarchy_level": "child",
                            "child_index": child_idx,
                            "parent_index": parent_idx,
                            "siblings_count": len(child_chunks),
                            "parent_chunk_size": len(parent.page_content),
                        },
                    )
                    all_children.append(enriched)

        logger.info(
            "ParentChild chunking: %d docs → %d parents → %d children "
            "(parent=%d chars, child=%d chars)",
            len(documents),
            len(self._parent_store),
            len(all_children),
            self.parent_chunk_size,
            self.child_chunk_size,
        )
        return all_children

    async def asplit_documents(self, documents: List[Document]) -> List[Document]:
        """Async version — delegates to sync (CPU-bound work)."""
        return self.split_documents(documents)

    def expand_to_parents(
        self,
        child_documents: List[Document],
        deduplicate: bool = True,
    ) -> List[Document]:
        """Expand retrieved child chunks to their parent chunks.

        Used at query time: after vector search returns child chunks,
        this replaces them with their richer parent context.

        Args:
            child_documents: Retrieved child chunks.
            deduplicate: Remove duplicate parents (when multiple children
                        from same parent are retrieved).

        Returns:
            List of parent documents with enriched metadata.
        """
        parents = []
        seen_parent_ids = set()

        for child in child_documents:
            parent_id = child.metadata.get("parent_id")
            if not parent_id:
                # Not a parent-child chunk, return as-is
                parents.append(child)
                continue

            if deduplicate and parent_id in seen_parent_ids:
                continue

            parent_doc = self._parent_store.get(parent_id)
            if parent_doc:
                # Copy parent and add retrieval metadata
                expanded = Document(
                    page_content=parent_doc.page_content,
                    metadata={
                        **parent_doc.metadata,
                        # Preserve child retrieval scores
                        "retrieved_via_child": child.metadata.get("child_id"),
                        "child_similarity_score": child.metadata.get(
                            "similarity_score"
                        ),
                        "child_rerank_score": child.metadata.get("rerank_score"),
                    },
                )
                parents.append(expanded)
                seen_parent_ids.add(parent_id)
            else:
                logger.warning(
                    "Parent %s not found in store, returning child as-is", parent_id
                )
                parents.append(child)

        logger.info(
            "Parent expansion: %d children → %d parents",
            len(child_documents),
            len(parents),
        )
        return parents

    def get_parent(self, parent_id: str) -> Optional[Document]:
        """Get a specific parent document by ID."""
        return self._parent_store.get(parent_id)

    @property
    def parent_count(self) -> int:
        """Number of parent documents in store."""
        return len(self._parent_store)

    @staticmethod
    def _generate_parent_id(content: str, index: int) -> str:
        """Generate a deterministic parent ID from content hash."""
        content_hash = hashlib.md5(content[:200].encode()).hexdigest()[:8]
        return f"parent_{content_hash}_{index}"
