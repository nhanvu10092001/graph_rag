"""Indexing service — handles document ingestion into vector stores.

Extracts the sync/async indexing logic from RAGPlugin into a focused service
with shared preparation logic to eliminate code duplication.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llm_utils_llm import EmbeddingsFactory, LLMFactory
from llm_utils_vector import VectorStoreManager as VectorStoreFactory

from ..factories import ChunkerFactory
from ..utils import extract_file_content_async, extract_text_from_file, is_supported_file, TextCleaner
from ..utils.chunking import BaseChunker
from ..utils.core import PipelineTimer

logger = logging.getLogger(__name__)


class IndexingService:
    """Handles document indexing (chunking + vector store insertion).

    Usage:
        svc = IndexingService(config)
        result = svc.index_sync(context)
        result = await svc.index_async(context)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        cleaning_cfg = config.get("extraction", {}).get("cleaning", {})
        self._cleaner = TextCleaner(cleaning_cfg)

    # ── Shared preparation ─────────────────────────────────────────────

    def _validate_and_extract_file(self, context: Dict[str, Any]):
        """Validate file and return (file_obj, collection_name, filename)."""
        file_obj = context.get("context")
        if not is_supported_file(file_obj):
            raise ValueError(f"File type not supported: {getattr(file_obj, 'documents_name', 'unknown')}")
        default_coll = "documents"
        collection_name = getattr(file_obj, "collection_name", None) or context.get("collection_name") or default_coll
        filename = getattr(file_obj, "documents_name", "unknown") or "unknown"
        return file_obj, collection_name, filename

    def _create_embeddings(self):
        """Create embeddings model from config."""
        embeddings_config = self.config.get("embeddings", {})
        return EmbeddingsFactory.create_embeddings(embeddings_config), embeddings_config

    def _create_llm(self):
        """Create LLM from config."""
        llm_config = self.config.get("llm", {})
        return LLMFactory.create_llm(llm_config)

    def _get_vector_provider(self, context: Dict[str, Any]) -> str:
        """Resolve the vector store provider."""
        return context.get("vector_provider") or self.config.get(
            "vector_store", {}
        ).get("provider", "qdrant")

    def _apply_doc_metadata(self, texts: List[Document], filename: str):
        """Stamp each chunk with source filename metadata."""
        for text in texts:
            if not text.metadata:
                text.metadata = {}
            text.metadata["filename"] = filename

    # ── Chunking ───────────────────────────────────────────────────────

    def _chunk_documents(self, documents: List[Document], embeddings, llm, filename: str = "") -> List[Document]:
        """Chunk documents using the configured strategy (sync)."""
        chunking_config = self.config.get("chunking", {})
        try:
            chunker = ChunkerFactory.create_chunker(
                config=chunking_config, embeddings=embeddings, llm=llm,
                filename=filename,
            )
            return chunker.split_documents(documents)
        except ValueError:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunking_config.get("chunk_size", 1000),
                chunk_overlap=chunking_config.get("chunk_overlap", 200),
            )
            return splitter.split_documents(documents)

    async def _chunk_documents_async(self, documents: List[Document], embeddings, llm, filename: str = "") -> List[Document]:
        """Chunk documents using the configured strategy (async)."""
        chunking_config = self.config.get("chunking", {})
        try:
            chunker = ChunkerFactory.create_chunker(
                config=chunking_config, embeddings=embeddings, llm=llm,
                filename=filename,
            )
            if hasattr(chunker, "asplit_documents"):
                return await chunker.asplit_documents(documents)
            else:
                return chunker.split_documents(documents)
        except ValueError:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunking_config.get("chunk_size", 1000),
                chunk_overlap=chunking_config.get("chunk_overlap", 200),
            )
            return splitter.split_documents(documents)

    # ── Sync indexing ──────────────────────────────────────────────────

    def index_sync(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Index documents synchronously."""
        timer = PipelineTimer()
        file_obj, collection_name, filename = self._validate_and_extract_file(context)

        with timer.stage("extraction"):
            try:
                content_string, extraction_meta = extract_text_from_file(file_obj, config=self.config)
                metadata = file_obj.metadata if hasattr(file_obj, "metadata") else {}
            except Exception as e:
                raise ValueError(f"Error extracting content from File object: {e}")

        with timer.stage("cleaning"):
            raw_chars = len(content_string)
            content_string = self._cleaner.clean(content_string, file_type=extraction_meta.get("file_type", "text"))
            cleaned_chars = len(content_string)

        with timer.stage("embedding_init"):
            embeddings, _ = self._create_embeddings()
            llm = self._create_llm()

        vector_provider = self._get_vector_provider(context)
        documents = [Document(page_content=content_string, metadata=metadata)]

        with timer.stage("chunking"):
            texts = self._chunk_documents(documents, embeddings, llm=llm, filename=filename)
            self._apply_doc_metadata(texts, filename)

        with timer.stage("vectorstore_create"):
            vectorstore = VectorStoreFactory.create_vectorstore_with_config(
                provider=vector_provider,
                embeddings=embeddings,
                collection_name=collection_name,
                config=self.config,
            )

        with timer.stage("vectorstore_insert"):
            try:
                uuids = VectorStoreFactory.add_documents_to_vectorstore(
                    vectorstore=vectorstore, documents=texts
                )
                if not uuids:
                    uuids = []
            except Exception as e:
                logger.error("Failed to add documents to vectorstore: %s", e)
                raise
            finally:
                if hasattr(vectorstore, "close"):
                    vectorstore.close()

        logger.info(
            "indexing.completed collection=%s chunks=%d total=%.1fms",
            collection_name, len(texts), timer.total_ms,
        )

        return {
            "status": "success",
            "indexed_documents": len(texts),
            "collection_name": collection_name,
            "vector_provider": vector_provider,
            "uuids": uuids,
            "timings": timer.timings,
            "chunking_metadata": BaseChunker.compute_stats(texts),
            "extraction_metadata": {
                "raw_chars": raw_chars,
                "cleaned_chars": cleaned_chars,
                **extraction_meta,
            },
        }

    # ── Async indexing ─────────────────────────────────────────────────

    async def index_async(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Index documents asynchronously with batched concurrent insertion."""
        timer = PipelineTimer()
        file_obj, collection_name, filename = self._validate_and_extract_file(context)

        with timer.stage("extraction"):
            try:
                content_string, extraction_meta = await extract_file_content_async(file_obj, config=self.config)
                metadata = file_obj.metadata if hasattr(file_obj, "metadata") else {}
            except Exception as e:
                raise ValueError(f"Error extracting content from File object: {e}")

        with timer.stage("cleaning"):
            raw_chars = len(content_string)
            content_string = self._cleaner.clean(content_string, file_type=extraction_meta.get("file_type", "text"))
            cleaned_chars = len(content_string)

        with timer.stage("embedding_init"):
            embeddings, embeddings_config = self._create_embeddings()
            llm = self._create_llm()

        vector_provider = self._get_vector_provider(context)
        documents = [Document(page_content=content_string, metadata=metadata)]

        with timer.stage("chunking"):
            texts = await self._chunk_documents_async(documents, embeddings, llm=llm, filename=filename)
            self._apply_doc_metadata(texts, filename)

        with timer.stage("vectorstore_create"):
            vectorstore = await VectorStoreFactory.create_vectorstore_async_with_config(
                provider=vector_provider,
                embeddings=embeddings,
                collection_name=collection_name,
                config=self.config,
            )

        async def add_batch(batch_docs):
            return await VectorStoreFactory.add_documents_to_vectorstore_async(
                vectorstore=vectorstore, documents=batch_docs
            )

        batch_size = embeddings_config.get(
            embeddings_config.get("provider", ""), {}
        )
        if isinstance(batch_size, dict):
            batch_size = batch_size.get("indexing_batch_size", 100)
        else:
            batch_size = 100

        with timer.stage("vectorstore_insert"):
            batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
            batch_uuids = await asyncio.gather(*[add_batch(b) for b in batches])
            uuids = [uid for batch in batch_uuids if batch for uid in batch]

        if hasattr(vectorstore, "aclose"):
            await vectorstore.aclose()

        logger.info(
            "indexing.async.completed collection=%s chunks=%d total=%.1fms",
            collection_name, len(texts), timer.total_ms,
        )

        return {
            "status": "success",
            "indexed_documents": len(texts),
            "collection_name": collection_name,
            "vector_provider": vector_provider,
            "uuids": uuids or [],
            "timings": timer.timings,
            "chunking_metadata": BaseChunker.compute_stats(texts),
            "extraction_metadata": {
                "raw_chars": raw_chars,
                "cleaned_chars": cleaned_chars,
                **extraction_meta,
            },
        }
