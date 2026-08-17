"""TaskPlugin implementation for Neo4j Graph RAG."""

import logging
from typing import Any, Dict, Optional
from llm_utils_core import TaskPlugin
from llm_utils_llm import EmbeddingsFactory, LLMFactory

from ..config import Neo4jConfig
from ..graph_store import Neo4jGraphStore
from ..services.indexing import GraphIndexingService
from ..services.query import GraphQueryService
from ..services.ark_service import ArkQueryService
from ..workflow.graph import LangGraphRAGWorkflow

logger = logging.getLogger(__name__)


class GraphRAGPlugin(TaskPlugin):
    """Neo4j Graph RAG Plugin implementing the TaskPlugin contract."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Load neo4j config from settings or env
        neo4j_config_dict = self.config.get("neo4j", {})
        
        # Create Neo4jConfig instance (fallback to env)
        self.neo4j_config = Neo4jConfig(
            uri=neo4j_config_dict.get("uri") or "bolt://localhost:7687",
            username=neo4j_config_dict.get("username") or "neo4j",
            password=neo4j_config_dict.get("password") or "password",
            database=neo4j_config_dict.get("database") or "neo4j"
        )
        
        # Initialize graph store
        self.graph_store = Neo4jGraphStore(self.neo4j_config)

    def name(self) -> str:
        """Return plugin name."""
        return "graph_rag"

    def _prepare_services(self) -> tuple:
        """Helper to dynamically construct LLM, Embeddings and services."""
        # 1. Initialize LLM and Embeddings from config
        llm_config = self.config.get("llm", {})
        embeddings_config = self.config.get("embeddings", {})
        ark_config = self.config.get("ark", {})

        llm = LLMFactory.create_llm(llm_config)
        embeddings = EmbeddingsFactory.create_embeddings(embeddings_config)

        # 2. Services
        indexing_service = GraphIndexingService(self.graph_store, llm, embeddings)
        query_service = GraphQueryService(self.graph_store, embeddings)
        ark_query_service = ArkQueryService(self.graph_store, llm, config=ark_config)

        return llm, indexing_service, query_service, ark_query_service

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronously execute RAG task."""
        action = context.get("action", "query")

        try:
            llm, indexing_service, query_service, ark_query_service = self._prepare_services()

            if action == "index":
                text = context.get("text")
                if not text:
                    raise ValueError("text is required for action 'index'")
                return indexing_service.index_text(text)

            elif action == "query":
                query = context.get("query")
                if not query:
                    raise ValueError("query is required for action 'query'")

                # Build and invoke LangGraph workflow
                workflow_orchestrator = LangGraphRAGWorkflow(self.graph_store, query_service, llm)
                workflow = workflow_orchestrator.compile()

                # Run the workflow
                inputs = {"query": query}
                result = workflow.invoke(inputs)

                return {
                    "status": "success",
                    "routing": result.get("routing"),
                    "cypher_query": result.get("cypher_query"),
                    "response": result.get("response")
                }
            elif action == "ark_search":
                query = context.get("query")
                if not query:
                    raise ValueError("query is required for action 'ark_search'")
                allowed_docs = context.get("allowed_docs", None)

                result = ark_query_service.retrieve(query, allowed_docs=allowed_docs)
                return {
                    "status": "success",
                    "context": result.get("context"),
                    "entities": result.get("entities")
                }
            else:
                raise ValueError(f"Unknown action: {action}")

        except Exception as e:
            logger.error(f"Graph RAG plugin execution failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}

    async def arun(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Asynchronously execute RAG task (delegating to threadpool where appropriate)."""
        # Since neo4j's langchain wrapper is primarily synchronous, we wrap run in an async executor
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.run, context)
