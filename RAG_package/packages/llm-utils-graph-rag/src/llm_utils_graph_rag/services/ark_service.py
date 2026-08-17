import logging
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.documents import Document

from llm_utils_graph_rag.graph_store import Neo4jGraphStore
from .ark_retriever import ArkRetriever

logger = logging.getLogger(__name__)

class ArkQueryService:
    """
    Service wrapper for ARK (Adaptive Retriever of Knowledge) retrieval.
    Initializes and provides access to the ArkRetriever LangChain component.
    """

    def __init__(
        self,
        graph_store: Neo4jGraphStore,
        llm: BaseChatModel,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.graph_store = graph_store
        self.llm = llm
        self.config = config or {}

        # Parse ARK config defaults
        self.n_agents = self.config.get("n_agents", 3)
        self.max_steps = self.config.get("t_max", 30)
        self.temperature = self.config.get("temperature", 0.7)
        self.top_k = self.config.get("top_k_fused", 10)

    def get_retriever(self, allowed_docs: Optional[List[str]] = None) -> ArkRetriever:
        """Instantiates the LangChain ArkRetriever with the given document filters."""
        return ArkRetriever(
            graph_store=self.graph_store,
            llm=self.llm,
            n_agents=self.n_agents,
            max_steps=self.max_steps,
            temperature=self.temperature,
            top_k=self.top_k,
            allowed_docs=allowed_docs
        )

    def retrieve(self, query: str, allowed_docs: Optional[List[str]] = None) -> Dict[str, Any]:
        """Sync wrapper for retrieving and formatting nodes."""
        retriever = self.get_retriever(allowed_docs)
        docs = retriever.invoke(query)
        return self._format_response(docs)

    async def aretrieve(self, query: str, allowed_docs: Optional[List[str]] = None) -> Dict[str, Any]:
        """Async wrapper for retrieving and formatting nodes."""
        retriever = self.get_retriever(allowed_docs)
        docs = await retriever.ainvoke(query)
        return self._format_response(docs)

    def _format_response(self, docs: List[Document]) -> Dict[str, Any]:
        """Formats the LangChain documents into the expected RAG plugin dict structure."""
        if not docs:
            return {"context": "No relevant entities found.", "entities": []}

        context_str = "\n\n".join([d.page_content for d in docs])
        entities = [d.metadata for d in docs]

        return {
            "context": context_str,
            "entities": entities
        }
