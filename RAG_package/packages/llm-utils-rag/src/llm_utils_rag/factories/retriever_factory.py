"""Factory for creating advanced retriever instances for RAG applications."""

from typing import Any, Dict, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.base import VectorStoreRetriever
from llm_utils_llm import LLMListwiseRerankCompressor


class RetrieverFactory:
    """Factory for creating advanced retriever instances for RAG applications."""

    _providers = {}

    @classmethod
    def register_provider(cls, name: str, factory_func):
        """Register a new retriever provider factory function."""
        cls._providers[name.lower()] = factory_func

    @classmethod
    def create_retriever(
        cls,
        config: Dict[str, Any],
        vectorstore: VectorStore,
        llm: BaseLanguageModel,
        metadata: dict = {},
        **kwargs: Any,
    ) -> VectorStoreRetriever:
        """Create a retriever instance based on configuration."""
        provider = config.get("provider", "vector_store_retriever").lower()
        provider_config = config.get(provider, {}).copy()

        if provider not in cls._providers:
            # Try to import common providers
            cls._try_register_common_providers()

        if provider not in cls._providers:
            raise ValueError(f"Unsupported retriever provider: {provider}")

        search_kwargs = {**provider_config.pop("search_kwargs", {}), "filter": metadata}
        if "filename" in search_kwargs.get("filter", {}):
            search_kwargs.update(
                {
                    "k": 100  # Large k to ensure filename filtering works effectively
                }
            )
        search_type = provider_config.pop("search_type", "similarity")

        # Pass common arguments and provider-specific config as kwargs
        return cls._providers[provider](
            llm=llm,
            vectorstore=vectorstore,
            cfg=provider_config,
            search_kwargs=search_kwargs,
            search_type=search_type,
            **kwargs,
        )

    @classmethod
    def combined_retriever(
        cls,
        retrievers: list[VectorStoreRetriever],
    ):
        from langchain.retrievers import EnsembleRetriever

        weights = [1 / len(retrievers)] * len(retrievers)
        return EnsembleRetriever(retrievers=retrievers, weights=weights)

    @classmethod
    def _try_register_common_providers(cls) -> VectorStoreRetriever:
        """Register common retriever providers."""

        # Try Hypothetical Retriever
        try:
            from langchain.chains import HypotheticalDocumentEmbedder

            def create_hypothetical(
                llm: Optional[BaseLanguageModel] = None,
                embeddings: Optional[BaseLanguageModel] = None,
                vectorstore: Optional[VectorStore] = None,
                cfg: Optional[Dict[str, Any]] = {},
                **kwargs,
            ):
                # Get embeddings from vectorstore if not provided
                base_embeddings = embeddings if embeddings else vectorstore.embeddings
                hyde_embedding_function = HypotheticalDocumentEmbedder.from_llm(
                    llm=llm, base_embeddings=base_embeddings, prompt_key="web_search"
                )
                _retriever: VectorStoreRetriever = vectorstore.as_retriever(
                    embedding_function=hyde_embedding_function,
                    **cfg,
                    **kwargs,
                )
                return _retriever

            cls.register_provider("hypothetical", create_hypothetical)

        except ImportError:
            pass

        # Try Multi Query Retriever
        try:
            from langchain.retrievers.multi_query import MultiQueryRetriever

            def create_multi_query_retriever(
                llm: Optional[BaseLanguageModel] = None,
                vectorstore: Optional[VectorStore] = None,
                cfg: Optional[Dict[str, Any]] = {},
                **kwargs,
            ) -> MultiQueryRetriever:
                _retriever: MultiQueryRetriever = MultiQueryRetriever.from_llm(
                    llm=llm,
                    retriever=vectorstore.as_retriever(**kwargs, **cfg),
                )
                return _retriever

            cls.register_provider("multi_query_retriever", create_multi_query_retriever)

        except ImportError:
            pass

        # Try Vector Store Retriever
        try:

            def create_vector_store_retriever(
                vectorstore: Optional[VectorStore] = None,
                cfg: Optional[Dict[str, Any]] = {},
                **kwargs,
            ) -> VectorStoreRetriever:
                _retriever: VectorStoreRetriever = vectorstore.as_retriever(
                    **kwargs, **cfg
                )
                return _retriever

            cls.register_provider(
                "vector_store_retriever", create_vector_store_retriever
            )

        except ImportError:
            pass

        # Try Contextual Compression Retriever - default
        try:
            from langchain.retrievers import ContextualCompressionRetriever
            from langchain.retrievers.document_compressors import LLMChainExtractor

            def create_contextual_compression_retriever(
                llm: Optional[BaseLanguageModel] = None,
                vectorstore: Optional[VectorStore] = None,
                cfg: Optional[Dict[str, Any]] = {},
                **kwargs,
            ) -> ContextualCompressionRetriever:
                compressor = LLMChainExtractor.from_llm(llm)
                _retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=vectorstore.as_retriever(**kwargs, **cfg),
                )
                return _retriever

            cls.register_provider(
                "contextual_compression_retriever",
                create_contextual_compression_retriever,
            )

        except ImportError:
            pass

        # Try Contextual Compression Retriever - LLMChainFilter
        try:
            from langchain.retrievers import ContextualCompressionRetriever
            from langchain.retrievers.document_compressors import LLMChainFilter

            def create_contextual_compression_retriever_LLMChainFilter(
                llm: Optional[BaseLanguageModel] = None,
                vectorstore: Optional[VectorStore] = None,
                cfg: Optional[Dict[str, Any]] = {},
                **kwargs,
            ) -> ContextualCompressionRetriever:
                compressor = LLMChainFilter.from_llm(llm)
                _retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=vectorstore.as_retriever(**kwargs, **cfg),
                )
                return _retriever

            cls.register_provider(
                "contextual_compression_retriever_LLMChainFilter",
                create_contextual_compression_retriever_LLMChainFilter,
            )

        except ImportError:
            pass

        # Try Contextual Compression Retriever - ReRanking
        try:
            from langchain.retrievers import ContextualCompressionRetriever

            def create_contextual_compression_retriever_ReRanking(
                llm: Optional[BaseLanguageModel] = None,
                vectorstore: Optional[VectorStore] = None,
                cfg: Optional[Dict[str, Any]] = {},
                **kwargs,
            ) -> ContextualCompressionRetriever:
                compressor = LLMListwiseRerankCompressor(llm=llm, top_n=1)
                _retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=vectorstore.as_retriever(**kwargs, **cfg),
                )
                return _retriever

            cls.register_provider(
                "contextual_compression_retriever_ReRanking",
                create_contextual_compression_retriever_ReRanking,
            )

        except ImportError:
            pass

        # Try WebSearch Retriever - needs GOOGLE_API_KEY and GOOGLE_CSE_ID
        try:
            from langchain.retrievers import ContextualCompressionRetriever
            from langchain.retrievers.document_compressors import LLMChainFilter
            from langchain_community.retrievers.web_research import WebResearchRetriever
            from langchain_community.utilities import GoogleSearchAPIWrapper
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            md_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=40,
                separators=[
                    "\n#### ",
                    "\n### ",
                    "\n## ",
                    "\n# ",
                    "\n- ",
                    "\n* ",
                    "\n\n",
                    "\n",
                    " ",
                    "",
                ],
            )

            def create_contextual_compression_retriever_Google_Search(
                llm: Optional[BaseLanguageModel] = None,
                vectorstore: Optional[VectorStore] = None,
                cfg: Optional[Dict[str, Any]] = {},
                **kwargs,
            ) -> ContextualCompressionRetriever:
                search = GoogleSearchAPIWrapper()
                _retriever: WebResearchRetriever = WebResearchRetriever.from_llm(
                    vectorstore=vectorstore,
                    llm=llm,
                    search=search,
                    num_search_results=2,
                    text_splitter=md_splitter,
                    allow_dangerous_requests=True,
                )
                _retriever = ContextualCompressionRetriever(
                    base_retriever=_retriever,
                    base_compressor=LLMChainFilter.from_llm(llm),
                    **kwargs,
                )

                return _retriever

            cls.register_provider(
                "contextual_compression_retriever_Google_Search",
                create_contextual_compression_retriever_Google_Search,
            )

        except ImportError:
            pass

        # Try Ensemble retriever
        try:
            from langchain.retrievers import EnsembleRetriever

            def create_ensemble_retriever(
                llm: Optional[BaseLanguageModel] = None,
                vectorstore: Optional[VectorStore] = None,
                cfg: Optional[Dict[str, Any]] = {},
                **kwargs,
            ) -> EnsembleRetriever:
                retrievers = []
                configs = cfg.get("retriever_configs", [])
                weights = cfg.get("weights", [])

                for config in configs:
                    retriever = cls.create_retriever(
                        llm=llm,
                        vectorstore=vectorstore,
                        config=config,
                        metadata=kwargs.get("search_kwargs", {}).get("filter", {}),
                    )
                    retrievers.append(retriever)
                return EnsembleRetriever(retrievers=retrievers, weights=weights)

            cls.register_provider("ensemble", create_ensemble_retriever)
        except ImportError:
            pass

        try:
            from langchain.chains.query_constructor.base import AttributeInfo
            from langchain.retrievers.self_query.base import SelfQueryRetriever

            def create_self_query_retriever(
                llm: Optional[BaseLanguageModel] = None,
                vectorstore: Optional[VectorStore] = None,
                cfg: Optional[Dict[str, Any]] = {},
                **kwargs,
            ) -> SelfQueryRetriever:

                # Get metadata field info and document content description
                metadata_field_info = cfg.get("metadata_field_info", [])
                document_content_description = cfg.get(
                    "document_content_description", "Document content"
                )

                # Convert dict metadata_field_info to AttributeInfo objects if needed
                if metadata_field_info and isinstance(metadata_field_info[0], dict):
                    attribute_info = []
                    for field in metadata_field_info:
                        attribute_info.append(
                            AttributeInfo(
                                name=field["name"],
                                description=field["description"],
                                type=field["type"],
                            )
                        )
                    metadata_field_info = attribute_info

                # Add verbose and enable_limit for better debugging and control
                retriever_kwargs = {
                    "verbose": True,
                    "enable_limit": True,
                    **kwargs
                }
                
                try:
                    _retriever = SelfQueryRetriever.from_llm(
                        llm=llm,
                        vectorstore=vectorstore,
                        document_contents=document_content_description,
                        metadata_field_info=metadata_field_info,
                        **retriever_kwargs,
                    )
                    return _retriever
                except Exception as e:
                    print(f"Warning: SelfQueryRetriever creation failed: {e}")
                    print("Falling back to basic vector store retriever...")
                    # Fallback to basic vector store retriever
                    return vectorstore.as_retriever(**retriever_kwargs)

            cls.register_provider("self_query_retriever", create_self_query_retriever)

        except ImportError:
            pass

        # ── Hybrid Search (Dense + Sparse BM25 with RRF) ──────────────────

        try:
            def create_hybrid_retriever(
                llm: Optional[BaseLanguageModel] = None,
                vectorstore: Optional[VectorStore] = None,
                cfg: Optional[Dict[str, Any]] = {},
                **kwargs,
            ):
                """Hybrid retriever combining dense vector search with BM25 sparse retrieval.

                Uses custom RRF (Reciprocal Rank Fusion) implementation.
                Config options (under retriever.hybrid):
                    dense_weight: float (default 0.7)
                    sparse_weight: float (default 0.3)
                    bm25_k: int (default 10)
                    bm25_params: dict (default {"k1": 1.5, "b": 0.75})
                """
                from ..utils.retrieval.bm25_retriever import BM25Retriever
                from ..utils.retrieval.hybrid_retriever import HybridRetriever

                dense_weight = cfg.get("dense_weight", 0.7)
                sparse_weight = cfg.get("sparse_weight", 0.3)
                bm25_k = cfg.get("bm25_k", 10)
                bm25_params = cfg.get("bm25_params", {"k1": 1.5, "b": 0.75})
                k = kwargs.get("search_kwargs", {}).get("k", 5)

                # Dense retriever
                dense = vectorstore.as_retriever(**kwargs)

                # Build BM25 index from vectorstore
                try:
                    sparse = BM25Retriever.from_vectorstore(
                        vectorstore, k=bm25_k, bm25_params=bm25_params
                    )
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(
                        "BM25 index failed (%s), falling back to dense-only", e
                    )
                    return dense

                return HybridRetriever(
                    sparse_retriever=sparse,
                    dense_retriever=dense,
                    sparse_weight=sparse_weight,
                    dense_weight=dense_weight,
                    k=k,
                )

            cls.register_provider("hybrid", create_hybrid_retriever)
            # Keep backward compat alias
            cls.register_provider("hybrid_search", create_hybrid_retriever)

        except Exception:
            pass

        # ── Phase 3: Cross-Encoder Reranking ───────────────────────────

        try:
            from langchain.retrievers import ContextualCompressionRetriever

            def create_cross_encoder_reranker(
                llm: Optional[BaseLanguageModel] = None,
                vectorstore: Optional[VectorStore] = None,
                cfg: Optional[Dict[str, Any]] = {},
                **kwargs,
            ):
                """Cross-encoder reranker for high-precision retrieval.

                Config options (under retriever.cross_encoder_reranker):
                    model: str (default "BAAI/bge-reranker-v2-m3")
                    top_n: int (default 5)
                """
                model_name = cfg.get("model", "BAAI/bge-reranker-v2-m3")
                top_n = cfg.get("top_n", 5)

                try:
                    from langchain.retrievers.document_compressors import (
                        CrossEncoderReranker,
                    )
                    from langchain_community.cross_encoders import (
                        HuggingFaceCrossEncoder,
                    )

                    model = HuggingFaceCrossEncoder(model_name=model_name)
                    compressor = CrossEncoderReranker(model=model, top_n=top_n)
                except ImportError:
                    # Fall back to LLM listwise reranker if HuggingFace not available
                    compressor = LLMListwiseRerankCompressor(llm=llm, top_n=top_n)

                base = vectorstore.as_retriever(**kwargs)
                return ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=base,
                )

            cls.register_provider("cross_encoder_reranker", create_cross_encoder_reranker)

        except ImportError:
            pass

        # ── Phase 3: HyDE (Hypothetical Document Embeddings) ───────────

        try:
            from langchain.chains import HypotheticalDocumentEmbedder

            def create_hyde_retriever(
                llm: Optional[BaseLanguageModel] = None,
                vectorstore: Optional[VectorStore] = None,
                cfg: Optional[Dict[str, Any]] = {},
                **kwargs,
            ):
                """HyDE retriever — generates a hypothetical answer, embeds it,
                then uses that embedding for similarity search.

                Config options (under retriever.hyde_retriever):
                    prompt_key: str (default "web_search")
                """
                prompt_key = cfg.get("prompt_key", "web_search")
                base_embeddings = vectorstore.embeddings

                hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
                    llm=llm,
                    base_embeddings=base_embeddings,
                    prompt_key=prompt_key,
                )
                return vectorstore.as_retriever(
                    embedding_function=hyde_embeddings, **kwargs
                )

            cls.register_provider("hyde_retriever", create_hyde_retriever)

        except ImportError:
            pass

        # ── BM25 Standalone (keyword-only retrieval) ───────────────────────

        try:
            def create_bm25_retriever(
                llm: Optional[BaseLanguageModel] = None,
                vectorstore: Optional[VectorStore] = None,
                cfg: Optional[Dict[str, Any]] = {},
                **kwargs,
            ):
                """BM25 keyword retriever (sparse only, no dense vector).

                Config options (under retriever.bm25):
                    bm25_k: int (default 10)
                    bm25_params: dict (default {"k1": 1.5, "b": 0.75})
                """
                from ..utils.retrieval.bm25_retriever import BM25Retriever

                bm25_k = cfg.get("bm25_k", 10)
                bm25_params = cfg.get("bm25_params", {"k1": 1.5, "b": 0.75})
                k = kwargs.get("search_kwargs", {}).get("k", 5)

                return BM25Retriever.from_vectorstore(
                    vectorstore, k=k, fetch_k=bm25_k * 100, bm25_params=bm25_params
                )

            cls.register_provider("bm25", create_bm25_retriever)

        except Exception:
            pass

        # ── Adaptive RAG (Self-Correcting Retrieval) ───────────────────────

        try:
            def create_adaptive_retriever(
                llm: Optional[BaseLanguageModel] = None,
                vectorstore: Optional[VectorStore] = None,
                cfg: Optional[Dict[str, Any]] = {},
                **kwargs,
            ):
                """Adaptive retriever with document grading and query rewriting.

                Config options (under retriever.adaptive):
                    max_retries: int (default 2)
                    relevance_threshold: float (default 0.5)
                    hallucination_check: bool (default false)
                """
                from ..utils.retrieval.adaptive_rag import AdaptiveRAGPipeline

                base_retriever = vectorstore.as_retriever(**kwargs)
                return AdaptiveRAGPipeline(
                    retriever=base_retriever,
                    llm=llm,
                    max_retries=cfg.get("max_retries", 2),
                    relevance_threshold=cfg.get("relevance_threshold", 0.5),
                    enable_hallucination_check=cfg.get("hallucination_check", False),
                )

            cls.register_provider("adaptive", create_adaptive_retriever)

        except Exception:
            pass

        # ── Query Transform Retriever (multi-angle retrieval) ──────────────

        try:
            def create_query_transform_retriever(
                llm: Optional[BaseLanguageModel] = None,
                vectorstore: Optional[VectorStore] = None,
                cfg: Optional[Dict[str, Any]] = {},
                **kwargs,
            ):
                """Retriever with query transformation pipeline.

                Transforms the query using configured strategies, then retrieves
                with each variant and merges results.

                Config options (under retriever.query_transform):
                    strategies: list (default ["step_back"])
                    max_sub_queries: int (default 3)
                """
                from ..utils.retrieval.query_transforms import QueryTransformPipeline

                # Build the pipeline config
                transform_config = {
                    "query_transform": {
                        "enabled": True,
                        "strategies": cfg.get("strategies", ["step_back"]),
                        "max_sub_queries": cfg.get("max_sub_queries", 3),
                    }
                }
                pipeline = QueryTransformPipeline.from_config(transform_config, llm)
                base_retriever = vectorstore.as_retriever(**kwargs)

                if pipeline is None:
                    return base_retriever

                # Wrap base retriever with query transform
                from ..utils.retrieval.hybrid_retriever import HybridRetriever
                from langchain_core.callbacks import CallbackManagerForRetrieverRun
                from langchain_core.retrievers import BaseRetriever
                from langchain_core.documents import Document
                import asyncio
                from typing import List as TList

                class QueryTransformRetriever(BaseRetriever):
                    """Retriever that transforms query before searching."""

                    class Config:
                        arbitrary_types_allowed = True

                    def _get_relevant_documents(
                        self,
                        query: str,
                        *,
                        run_manager=None,
                    ) -> TList[Document]:
                        """Sync version — runs async pipeline in event loop."""
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                # Already in async context
                                import concurrent.futures
                                with concurrent.futures.ThreadPoolExecutor() as pool:
                                    variants = pool.submit(
                                        asyncio.run, pipeline.transform(query)
                                    ).result()
                            else:
                                variants = loop.run_until_complete(pipeline.transform(query))
                        except RuntimeError:
                            variants = [query]

                        # Retrieve with each variant and merge
                        all_docs = {}
                        for variant in variants:
                            docs = base_retriever.invoke(variant)
                            for doc in docs:
                                key = hash(doc.page_content.strip())
                                if key not in all_docs:
                                    all_docs[key] = doc

                        return list(all_docs.values())

                return QueryTransformRetriever()

            cls.register_provider("query_transform", create_query_transform_retriever)

        except Exception:
            pass
