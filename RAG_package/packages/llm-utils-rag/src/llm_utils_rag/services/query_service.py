"""Query service — handles RAG retrieval and answer generation.

Extracts the sync/async query logic from RAGPlugin into a focused service
with shared retriever-building and response-parsing logic.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from llm_utils_llm import EmbeddingsFactory, LLMFactory
from llm_utils_vector import VectorStoreManager as VectorStoreFactory

from ..factories import RetrieverFactory, RerankerFactory
from ..utils.retrieval import create_semantic_rag_chain
from ..utils.retrieval.query_transforms import QueryTransformPipeline
from ..utils.retrieval.adaptive_rag import AdaptiveRAGPipeline

logger = logging.getLogger(__name__)


class QueryService:
    """Handles document querying through the RAG pipeline."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    # ── Shared helpers ─────────────────────────────────────────────────

    def _validate_query_context(self, context: Dict[str, Any]):
        query = context.get("query")
        if not query:
            raise ValueError("query is required")
        collection_name = context.get("collection_name")
        if not collection_name:
            raise ValueError("collection_name is required")
        return query, collection_name

    def _get_vector_provider(self, context: Dict[str, Any]) -> str:
        return context.get("vector_provider") or self.config.get(
            "vector_store", {}
        ).get("provider", "qdrant")

    def _build_filename_filter(self, context: Dict[str, Any]) -> dict:
        fn = context.get("filename", None)
        return {"filename": fn} if fn is not None else {}

    def _build_retriever(self, context, vectorstore, llm):
        """Build a (possibly combined) retriever from vs_id list."""
        retriever_config = self.config.get(
            "retriever", {"provider": "vector_store_retriever"}
        )
        filename_filter = self._build_filename_filter(context)
        combined = []
        vs_id_list = context.get("vs_id", [])

        if not vs_id_list:
            metadata = {**filename_filter}
            retriever = RetrieverFactory.create_retriever(
                config=retriever_config,
                vectorstore=vectorstore,
                llm=llm,
                metadata=metadata,
            )
            combined.append(retriever)
        else:
            for vs_id_metadata in vs_id_list:
                metadata = {**filename_filter}
                if vs_id_metadata and vs_id_metadata.strip():
                    metadata["vs_id"] = vs_id_metadata
                retriever = RetrieverFactory.create_retriever(
                    config=retriever_config,
                    vectorstore=vectorstore,
                    llm=llm,
                    metadata=metadata,
                )
                combined.append(retriever)

        if len(combined) > 1:
            return RetrieverFactory.combined_retriever(combined)
        return combined[0]

    def _build_reranker(self, llm):
        """Build a reranker from config (or None if disabled)."""
        return RerankerFactory.create(self.config, llm=llm)

    def _create_rag_chain(self, retriever, llm: BaseLanguageModel):
        """Create a RAG chain (semantic or standard LCEL)."""
        rag_config = self.config.get("retrieval", {})
        use_semantic = rag_config.get("use_semantic_chain", False)
        reranker = self._build_reranker(llm)
        rerank_top_k = self.config.get("reranking", {}).get("top_k", 5)

        if use_semantic:
            chain = create_semantic_rag_chain(
                retriever=retriever,
                llm=llm,
                enable_category_aware_prompts=rag_config.get(
                    "enable_category_prompts", True
                ),
                enable_context_enhancement=rag_config.get(
                    "enable_context_enhancement", True
                ),
                enable_llm_generation=rag_config.get(
                    "enable_llm_generation", False
                ),
            )
            return chain.create_chain()

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        async def retrieve_and_rerank(question: str):
            """Retrieve documents and optionally rerank them."""
            docs = await retriever.ainvoke(question)
            if reranker and docs:
                docs = await reranker.rerank(question, docs, top_k=rerank_top_k)
            return docs

        return RunnablePassthrough() | {
            "context": RunnableLambda(lambda x: x["question"])
            | RunnableLambda(retrieve_and_rerank)
            | RunnableLambda(format_docs),
            "question": RunnableLambda(lambda x: x["question"]),
            "docs": RunnableLambda(lambda x: x["question"]) | RunnableLambda(retrieve_and_rerank),
        }

    def _build_query_transform_pipeline(self, llm):
        """Build query transform pipeline from config (or None if disabled)."""
        return QueryTransformPipeline.from_config(self.config, llm)

    def _build_adaptive_pipeline(self, retriever, llm):
        """Build adaptive RAG pipeline from config (or None if disabled)."""
        adaptive_config = self.config.get("adaptive", {})
        if not adaptive_config.get("enabled", False):
            return None
        return AdaptiveRAGPipeline(
            retriever=retriever,
            llm=llm,
            max_retries=adaptive_config.get("max_retries", 2),
            relevance_threshold=adaptive_config.get("relevance_threshold", 0.5),
            enable_hallucination_check=adaptive_config.get("hallucination_check", False),
        )

    def _expand_parent_chunks(self, docs):
        """Expand child chunks to parent chunks if parent-child chunking was used."""
        if not docs:
            return docs
        # Check if any docs have parent_id metadata (parent-child chunking)
        has_parents = any(d.metadata.get("parent_id") for d in docs)
        if not has_parents:
            return docs

        try:
            from ..utils.chunking.parent_child_chunker import ParentChildChunker
            chunker = ParentChildChunker()
            # If parent store is empty (fresh query), return docs as-is
            if chunker.parent_count == 0:
                return docs
            return chunker.expand_to_parents(docs)
        except Exception as e:
            logger.warning("Parent expansion failed: %s, returning original docs", e)
            return docs

    def _parse_response(self, response: dict) -> tuple:
        """Extract answer text and source citations from chain response.

        Returns:
            (answer_text, source_docs) where source_docs is the raw document list
        """
        result = ""
        docs = []
        if "docs" in response:
            docs = response["docs"]
            for doc in docs:
                result += doc.page_content + "\n"
        elif "documents" in response:
            docs = response["documents"]
            for doc in docs:
                result += doc.page_content + "\n"
            if "answer" in response:
                result = response["answer"]
        return result, docs

    def _build_sources(self, docs: list) -> List[Dict[str, Any]]:
        """Build citation list from retrieved documents."""
        sources = []
        for i, doc in enumerate(docs):
            meta = doc.metadata if hasattr(doc, "metadata") else {}
            sources.append({
                "filename": meta.get("filename", "unknown"),
                "chunk_id": meta.get("chunk_id", i),
                "chunk_method": meta.get("chunk_method", "unknown"),
                "category": meta.get("category"),
                "relevance_score": meta.get("similarity_score"),
            })
        return sources

    # ── Sync query ─────────────────────────────────────────────────────

    def query_sync(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Query documents synchronously."""
        query, collection_name = self._validate_query_context(context)

        embeddings = EmbeddingsFactory.create_embeddings(
            self.config.get("embeddings", {})
        )
        vector_provider = self._get_vector_provider(context)

        vectorstore = VectorStoreFactory.get_existing_vectorstore_with_config(
            provider=vector_provider,
            embeddings=embeddings,
            collection_name=collection_name,
            config=self.config,
        )

        llm = LLMFactory.create_llm(self.config.get("llm", {}))
        retriever = self._build_retriever(context, vectorstore, llm)
        chain = self._create_rag_chain(retriever, llm)

        rag_config = self.config.get("retrieval", {})
        use_semantic = rag_config.get("use_semantic_chain", False)

        t0 = time.perf_counter()

        if use_semantic:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import nest_asyncio
                    nest_asyncio.apply()
                response = loop.run_until_complete(
                    chain.ainvoke({"question": query})
                )
            except RuntimeError:
                response = asyncio.run(
                    chain.ainvoke({"question": query})
                )
        else:
            response = chain.invoke({"question": query})

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        result, docs = self._parse_response(response)

        if hasattr(vectorstore, "close"):
            vectorstore.close()

        reranking_meta = {}
        if docs and docs[0].metadata.get("rerank_provider"):
            reranking_meta = {
                "reranker": docs[0].metadata.get("rerank_provider"),
                "scores": [round(d.metadata.get("rerank_score", 0), 4) for d in docs],
            }

        return {
            "status": "success",
            "query": query,
            "answer": result,
            "collection_name": collection_name,
            "sources": self._build_sources(docs),
            "retrieval_metadata": {
                "chunks_returned": len(docs),
                "retrieval_time_ms": elapsed_ms,
                **reranking_meta,
            },
        }

    # ── Async query ────────────────────────────────────────────────────

    async def query_async(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Query documents asynchronously."""
        query, collection_name = self._validate_query_context(context)

        embeddings = EmbeddingsFactory.create_embeddings(
            self.config.get("embeddings", {})
        )
        vector_provider = self._get_vector_provider(context)

        vectorstore = (
            await VectorStoreFactory.get_existing_vectorstore_async_with_config(
                provider=vector_provider,
                embeddings=embeddings,
                collection_name=collection_name,
                config=self.config,
            )
        )

        llm = LLMFactory.create_llm(self.config.get("llm", {}))
        retriever = self._build_retriever(context, vectorstore, llm)
        chain = self._create_rag_chain(retriever, llm)

        t0 = time.perf_counter()
        response = await chain.ainvoke({"question": query})
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        result, docs = self._parse_response(response)

        # Expand parent chunks if applicable
        docs = self._expand_parent_chunks(docs)

        if hasattr(vectorstore, "aclose"):
            await vectorstore.aclose()

        reranking_meta = {}
        if docs and docs[0].metadata.get("rerank_provider"):
            reranking_meta = {
                "reranker": docs[0].metadata.get("rerank_provider"),
                "scores": [round(d.metadata.get("rerank_score", 0), 4) for d in docs],
            }

        return {
            "status": "success",
            "query": query,
            "answer": result,
            "collection_name": collection_name,
            "sources": self._build_sources(docs),
            "retrieval_metadata": {
                "chunks_returned": len(docs),
                "retrieval_time_ms": elapsed_ms,
                **reranking_meta,
            },
        }
