"""Contextual chunker — prepends document summary to each chunk.

Implements "late chunking" / "contextual retrieval" where each chunk
is enriched with document-level context for better retrieval accuracy.
"""

import copy
import logging
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base_chunker import BaseChunker

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_PROMPT = (
    "Provide a brief 2-sentence summary of the following document. "
    "Focus on the main topic, purpose, and key themes.\n\n"
    "Document:\n{text}\n\nSummary:"
)


class ContextualChunkerWrapper(BaseChunker):
    """Splits documents then prepends a document-level summary to each chunk.

    Includes a summary cache to avoid redundant LLM calls for documents
    with the same content prefix.

    Usage:
        chunker = ContextualChunkerWrapper(llm=llm, chunk_size=1000)
        chunks = chunker.split_documents(docs)
        chunks = await chunker.asplit_documents(docs)
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        context_prompt: Optional[str] = None,
    ):
        self.llm = llm
        self.context_prompt = context_prompt or DEFAULT_CONTEXT_PROMPT
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        # Cache summaries by content hash to avoid repeated LLM calls
        self._context_cache: Dict[int, str] = {}

    def _get_cache_key(self, text: str) -> int:
        """Generate cache key from first 500 chars of document."""
        return hash(text[:500].strip())

    def _generate_context(self, text: str) -> str:
        """Generate a document-level summary using the LLM (with cache)."""
        cache_key = self._get_cache_key(text)
        if cache_key in self._context_cache:
            logger.debug("Context cache hit")
            return self._context_cache[cache_key]

        if self.llm is None:
            # Fallback: use first 200 chars as context
            context = text[:200].strip()
        else:
            truncated = text[:3000]  # Limit input to avoid token overflow
            prompt = self.context_prompt.format(text=truncated)
            try:
                result = self.llm.invoke(prompt)
                content = result.content if hasattr(result, "content") else str(result)
                context = content.strip()
            except Exception as e:
                logger.warning("Context generation failed: %s. Using fallback.", e)
                context = text[:200].strip()

        self._context_cache[cache_key] = context
        return context

    async def _generate_context_async(self, text: str) -> str:
        """Generate a document-level summary asynchronously (with cache)."""
        cache_key = self._get_cache_key(text)
        if cache_key in self._context_cache:
            logger.debug("Context cache hit (async)")
            return self._context_cache[cache_key]

        if self.llm is None:
            context = text[:200].strip()
        else:
            truncated = text[:3000]
            prompt = self.context_prompt.format(text=truncated)
            try:
                result = await self.llm.ainvoke(prompt)
                content = result.content if hasattr(result, "content") else str(result)
                context = content.strip()
            except Exception as e:
                logger.warning("Async context generation failed: %s. Using fallback.", e)
                context = text[:200].strip()

        self._context_cache[cache_key] = context
        return context

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents and prepend context to each chunk."""
        all_chunks = []

        for doc in documents:
            context = self._generate_context(doc.page_content)
            chunks = self._splitter.split_documents([doc])

            for i, chunk in enumerate(chunks):
                chunk.page_content = f"[Document Context: {context}]\n\n{chunk.page_content}"
                self._enrich_metadata(
                    chunk, index=i, method="contextual",
                    source_metadata=doc.metadata,
                    extra={"has_context": True},
                )
                all_chunks.append(chunk)

        return all_chunks

    async def asplit_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents and prepend context asynchronously."""
        all_chunks = []

        for doc in documents:
            context = await self._generate_context_async(doc.page_content)
            chunks = self._splitter.split_documents([doc])

            for i, chunk in enumerate(chunks):
                chunk.page_content = f"[Document Context: {context}]\n\n{chunk.page_content}"
                self._enrich_metadata(
                    chunk, index=i, method="contextual",
                    source_metadata=doc.metadata,
                    extra={"has_context": True},
                )
                all_chunks.append(chunk)

        return all_chunks
