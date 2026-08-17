"""RAG Query Tool for MCP integration."""

from typing import Any, Dict, Optional, Type

from langchain_core.tools import BaseTool as LangChainBaseTool
from pydantic import BaseModel, Field, PrivateAttr

from ..plugins.rag_plugin import RAGPlugin


class RAGQueryInput(BaseModel):
    """Input schema for RAG Query Tool."""
    query: str = Field(description="The complete question or query to ask about the indexed documents. Pass the full user query for better context.")
    collection_name: str = Field(default="demo_collection", description="dont need to fill this parameter")
    vs_id: list = Field(default_factory=list, description="dont need to fill this parameter")
    file_name: str = Field(default="", description="The name of the file or document that user wants to retrieve. example: 'filename.txt'")


class RAGQueryTool(LangChainBaseTool):
    """MCP tool for querying indexed documents using RAG."""
    
    name: str = "Get_Answer_Tools"
    description: str = "This tool must always be called before responding to any user input. It provides the final answer, ensuring every query or action is processed through this tool first."
    args_schema: Type[BaseModel] = RAGQueryInput
    _rag_plugin: Optional[RAGPlugin] = PrivateAttr(default=None)

    def __init__(
        self,
        rag_plugin: Optional[RAGPlugin] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize RAG query tool.

        Args:
            rag_plugin: Pre-configured RAGPlugin instance
            config: Configuration for RAGPlugin if rag_plugin is None
        """
        super().__init__(**kwargs)
        # Initialize the RAG plugin after calling super().__init__
        self._rag_plugin = rag_plugin or RAGPlugin(config)

    @property
    def rag_plugin(self) -> RAGPlugin:
        """Get the RAG plugin instance."""
        if self._rag_plugin is None:
            self._rag_plugin = RAGPlugin()
        return self._rag_plugin

    def _run(self, query: str, collection_name: str = "demo_collection", vs_id: list = None, file_name: str = "", **kwargs) -> Dict[str, Any]:
        """Synchronous execution method required by LangChain BaseTool."""
        import asyncio
        return asyncio.run(self._arun(query, collection_name, vs_id, file_name, **kwargs))

    async def _arun(self, query: str, collection_name: str = "demo_collection", vs_id: list = None, file_name: str = "", **kwargs) -> Dict[str, Any]:
        """Execute document querying asynchronously.

        Args:
            query: The question or query to ask about the indexed documents
            collection_name: Collection name for the documents
            vs_id: Vector store IDs (unused parameter)
            file_name: Specific file name to filter results

        Returns:
            Dict containing status, query, answer, and collection name
        """
        try:
            # Validate required parameters
            if not query:
                return {
                    "status": "error",
                    "error": "query parameter is required",
                    "error_type": "ValueError",
                }

            if not collection_name:
                return {
                    "status": "error",
                    "error": "collection_name parameter is required",
                    "error_type": "ValueError",
                }

            # Prepare context for RAG plugin
            context = {
                "action": "query",
                "query": query,
                "collection_name": collection_name,
                "metadata": kwargs.get("metadata", {}),
            }

            # Add vs_id if provided (required by some RAG implementations)
            # If vs_id is empty or None, don't pass it to avoid indexing errors
            if vs_id is not None and len(vs_id) > 0:
                context["vs_id"] = vs_id

            # Add filename extraction parameter if provided
            if file_name:
                context["filename"] = file_name

            # Execute querying using RAG plugin
            result = await self.rag_plugin.arun(context)

            # Add tool-specific metadata
            if result.get("status") == "success":
                result["tool"] = "rag_query"

            return result

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "tool": "rag_query",
            }
