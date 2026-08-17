"""Simplified RAG plugin — thin facade delegating to focused services.

This module maintains backward compatibility while internally using:
- IndexingService for document ingestion
- QueryService for RAG retrieval
- DeletionService for vector removal
"""

import logging
from typing import Any, Dict, List, Optional

from llm_utils_core import TaskPlugin

from ..services import DeletionService, IndexingService, QueryService
from ..utils.core import InputValidator, MetricsCollector

logger = logging.getLogger(__name__)


class RAGPlugin(TaskPlugin):
    """RAG plugin with proper synchronous and asynchronous execution paths.

    Delegates all work to focused service classes while maintaining
    the original TaskPlugin interface (run / arun).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize RAG plugin with configuration."""
        self.config = config or {}

        # Initialize focused services
        self._indexing = IndexingService(self.config)
        self._query = QueryService(self.config)
        self._deletion = DeletionService(self.config)
        self._metrics = MetricsCollector()

    def name(self) -> str:
        """Return the plugin name."""
        return "rag"

    # ── TaskPlugin interface ───────────────────────────────────────────

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute RAG task synchronously using LangChain's synchronous methods."""
        try:
            InputValidator.validate_context(context)
        except (ValueError, TypeError) as e:
            return {"status": "error", "error": str(e), "error_type": "ValidationError"}

        action = context.get("action", "query")

        try:
            if action == "index":
                result = self._indexing.index_sync(context)
                self._metrics.record_index(
                    collection=result.get("collection_name", ""),
                    num_chunks=result.get("indexed_documents", 0),
                    latency_ms=result.get("timings", {}).get("total_ms", 0),
                    provider=result.get("vector_provider", "unknown"),
                )
                return result
            elif action == "query":
                result = self._query.query_sync(context)
                self._metrics.record_query(
                    query=context.get("query", ""),
                    collection=result.get("collection_name", ""),
                    num_docs=result.get("retrieval_metadata", {}).get("chunks_returned", 0),
                    latency_ms=result.get("retrieval_metadata", {}).get("retrieval_time_ms", 0),
                    sources=result.get("sources"),
                )
                return result
            elif action == "delete":
                return self._deletion.delete_sync(context)
            else:
                raise ValueError(f"Unknown action: {action}")
        except Exception as e:
            self._metrics.record_error(action, str(e), context.get("collection_name", ""))
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}

    async def arun(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute RAG task asynchronously using LangChain's asynchronous methods."""
        try:
            InputValidator.validate_context(context)
        except (ValueError, TypeError) as e:
            return {"status": "error", "error": str(e), "error_type": "ValidationError"}

        action = context.get("action", "query")

        try:
            if action == "index":
                result = await self._indexing.index_async(context)
                self._metrics.record_index(
                    collection=result.get("collection_name", ""),
                    num_chunks=result.get("indexed_documents", 0),
                    latency_ms=result.get("timings", {}).get("total_ms", 0),
                    provider=result.get("vector_provider", "unknown"),
                )
                return result
            elif action == "query":
                result = await self._query.query_async(context)
                self._metrics.record_query(
                    query=context.get("query", ""),
                    collection=result.get("collection_name", ""),
                    num_docs=result.get("retrieval_metadata", {}).get("chunks_returned", 0),
                    latency_ms=result.get("retrieval_metadata", {}).get("retrieval_time_ms", 0),
                    sources=result.get("sources"),
                )
                return result
            elif action == "delete":
                return await self._deletion.delete_async(context)
            else:
                raise ValueError(f"Unknown action: {action}")
        except Exception as e:
            self._metrics.record_error(action, str(e), context.get("collection_name", ""))
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}

    # ── Observability ──────────────────────────────────────────────────

    def get_metrics(self) -> Dict[str, Any]:
        """Return aggregate RAG pipeline metrics.

        Returns dict with query stats, indexing stats, and recent errors.
        """
        return self._metrics.get_stats()

    def reset_metrics(self):
        """Clear all collected metrics."""
        self._metrics.reset()

    # ── MCP Tools ──────────────────────────────────────────────────────

    def create_mcp_tools(self, tool_type="query"):
        """Create MCP tools for this RAG plugin.

        Returns:
            List of MCP tools configured with this plugin
        """
        from ..tools import RAGQueryTool

        return [RAGQueryTool(rag_plugin=self)]

    def export_to_mcp_server(self, server):
        """Export RAG functionality as MCP tools to an MCP server.

        Args:
            server: MCPServer instance to add tools to

        Returns:
            List of tool names that were added
        """
        tools = self.create_mcp_tools()
        tool_names = []

        for tool in tools:
            server.add_tool(tool)
            tool_names.append(tool.get_metadata().name)

        return tool_names

    def close(self):
        """Release all service resources."""
        self._deletion.close()

