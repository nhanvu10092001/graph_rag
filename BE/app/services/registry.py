"""Lazy singleton registry for Graph RAG services.

All services are initialized on first access, not at import time.
Thread-safe via threading.Lock.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_instance: Optional["ServiceBundle"] = None


@dataclass
class ServiceBundle:
    llm: object
    embeddings: object
    graph_rag_plugin: object
    graph_query_service: object
    graph_indexing_service: object
    community_service: object
    global_search_service: object
    ark_query_service: object
    reranker: object
    graph_rag_config: dict


def get_services() -> ServiceBundle:
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is not None:
            return _instance
        _instance = _initialize_services()
        return _instance


def _initialize_services() -> ServiceBundle:
    from app.config import settings
    from app.agent.llm import create_llm
    from llm_utils_llm import EmbeddingsFactory
    from llm_utils_graph_rag.plugins.graph_rag_plugin import GraphRAGPlugin
    from llm_utils_reranker import RerankerFactory
    from llm_utils_graph_rag.services.query import GraphQueryService
    from llm_utils_graph_rag.services.indexing import GraphIndexingService
    from llm_utils_graph_rag.services.community import CommunityDetectionService
    from llm_utils_graph_rag.services.global_search import GlobalSearchService
    from llm_utils_graph_rag.services.ark_service import ArkQueryService

    graph_rag_cfg = settings.get_graph_rag_config()

    llm = create_llm(graph_rag_cfg["llm"])

    plugin_config = {
        "neo4j": graph_rag_cfg["neo4j"],
        "llm": {},
        "embeddings": graph_rag_cfg["embeddings"],
    }

    logger.info("Initializing Graph RAG Plugin...")
    graph_rag_plugin = GraphRAGPlugin(plugin_config)
    graph_rag_plugin.graph_store.connect()
    graph_rag_plugin.graph_store.ensure_fulltext_index()

    embeddings = EmbeddingsFactory.create_embeddings(plugin_config["embeddings"])

    reranker = RerankerFactory.create_reranker(graph_rag_cfg, llm=llm)

    graph_query_service = GraphQueryService(
        graph_rag_plugin.graph_store, embeddings, reranker=reranker
    )

    graph_indexing_service = GraphIndexingService(
        graph_rag_plugin.graph_store, llm, embeddings, config=graph_rag_cfg
    )

    community_config = graph_rag_cfg.get("community_detection", {})
    community_service = CommunityDetectionService(
        graph_store=graph_rag_plugin.graph_store,
        llm=llm,
        embeddings=embeddings,
        config=community_config,
    )

    global_search_service = GlobalSearchService(
        graph_store=graph_rag_plugin.graph_store,
        llm=llm,
        embeddings=embeddings,
        community_service=community_service,
    )

    ark_config = graph_rag_cfg.get("ark", {})
    ark_query_service = ArkQueryService(
        graph_store=graph_rag_plugin.graph_store, llm=llm, config=ark_config
    )

    logger.info("All Graph RAG services initialized.")

    return ServiceBundle(
        llm=llm,
        embeddings=embeddings,
        graph_rag_plugin=graph_rag_plugin,
        graph_query_service=graph_query_service,
        graph_indexing_service=graph_indexing_service,
        community_service=community_service,
        global_search_service=global_search_service,
        ark_query_service=ark_query_service,
        reranker=reranker,
        graph_rag_config=graph_rag_cfg,
    )
