import asyncio
import logging
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from llm_utils_graph_rag.graph_store import Neo4jGraphStore
from .ark_tools import GlobalSearchTool, NeighborhoodExplorationTool, AddToAnswerTool, FinishTool
from .ark_agent import create_ark_agent_graph, ArkAgentState
from .ark_prompts import build_ark_system_prompt

logger = logging.getLogger(__name__)


class ArkRetriever(BaseRetriever):
    """
    ARK (Adaptive Retriever of Knowledge) Retriever.
    Runs multiple agentic trajectories in parallel to explore a Neo4j knowledge graph,
    then uses Voting Rank Fusion to aggregate explicitly selected nodes into Documents.
    """

    graph_store: Neo4jGraphStore = Field(exclude=True)
    llm: BaseChatModel = Field(exclude=True)
    n_agents: int = Field(default=3, description="Number of parallel stochastic agents.")
    max_steps: int = Field(default=30, description="Maximum steps per agent trajectory.")
    temperature: float = Field(default=0.7, description="Temperature for agent exploration diversity.")
    top_k: int = Field(default=10, description="Top-K fused nodes to return.")
    allowed_docs: Optional[List[str]] = Field(default=None, description="Allowed documents for isolation.")

    def _fetch_node_metadata(self, node_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch type and description for a list of node IDs from Neo4j."""
        if not node_ids:
            return {}
        cypher = """
        MATCH (n:Entity)
        WHERE n.id IN $node_ids
        RETURN n.id AS id, n.type AS type, n.description AS description
        """
        try:
            results = self.graph_store.query(cypher, {"node_ids": node_ids})
            return {r["id"]: r for r in results}
        except Exception as e:
            logger.error(f"Failed to fetch node metadata: {e}")
            return {}

    async def _run_single_agent(self, query: str, agent_idx: int, system_prompt: str) -> List[Dict[str, Any]]:
        """Runs a single ARK agent trajectory and returns explicitly selected nodes."""
        logger.info(f"Starting ARK Agent {agent_idx} for query: {query}")

        tools = [
            GlobalSearchTool(graph_store=self.graph_store),
            NeighborhoodExplorationTool(graph_store=self.graph_store),
            AddToAnswerTool(),
            FinishTool(),
        ]

        try:
            llm_instance = self.llm.bind(temperature=self.temperature)
        except Exception:
            llm_instance = self.llm

        graph = create_ark_agent_graph(llm_instance, tools)

        system_msg = SystemMessage(content=system_prompt)

        initial_state: ArkAgentState = {
            "messages": [system_msg, HumanMessage(content=f"Find nodes that answer the question: {query}")],
            "current_step": 0,
            "max_steps": self.max_steps,
            "selected_nodes": {},
            "is_finished": False,
        }

        try:
            final_state = await graph.ainvoke(initial_state)
            selected = final_state.get("selected_nodes", {})
            nodes = [
                {
                    "node_id": nid,
                    "reasoning": info.get("reasoning", ""),
                    "step": info.get("step", 0),
                }
                for nid, info in selected.items()
            ]
            logger.info(f"ARK Agent {agent_idx} completed. Selected {len(nodes)} node(s).")
            return nodes
        except Exception as e:
            logger.error(f"ARK Agent {agent_idx} failed: {e}")
            return []

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Async implementation of the ARK retrieval process."""

        # Build system prompt once (shared across all agents)
        system_prompt = build_ark_system_prompt(self.graph_store)

        # 1. Run parallel agents
        tasks = [self._run_single_agent(query, i, system_prompt) for i in range(self.n_agents)]
        agent_results = await asyncio.gather(*tasks)

        # 2. Voting Rank Fusion (only over explicitly selected nodes via add_to_answer)
        fused_nodes: Dict[str, Dict[str, Any]] = {}
        for agent_idx, nodes in enumerate(agent_results):
            for order_idx, node in enumerate(nodes):
                nid = node["node_id"]
                if nid not in fused_nodes:
                    fused_nodes[nid] = {
                        "node_id": nid,
                        "vote_count": 0,
                        "first_seen_order": order_idx,
                    }
                fused_nodes[nid]["vote_count"] += 1

        # Sort: Vote Count DESC, First Seen Order ASC (matches official paper)
        sorted_nodes = sorted(
            fused_nodes.values(),
            key=lambda x: (-x["vote_count"], x["first_seen_order"]),
        )

        top_nodes = sorted_nodes[: self.top_k]
        top_node_ids = [n["node_id"] for n in top_nodes]

        if not top_node_ids:
            return []

        # 3. Fetch full metadata from Neo4j for the top VRF nodes
        node_metadata = self._fetch_node_metadata(top_node_ids)

        # 4. Build LangChain Documents
        docs = []
        for rank, node in enumerate(top_nodes):
            nid = node["node_id"]
            meta = node_metadata.get(nid, {})
            node_type = meta.get("type", "UNKNOWN")
            description = meta.get("description", "")

            page_content = f"Entity: {nid} (Type: {node_type})\nDescription: {description}"
            metadata = {
                "node_id": nid,
                "type": node_type,
                "vote_count": node["vote_count"],
                "ark_rank": rank + 1,
            }
            docs.append(Document(page_content=page_content, metadata=metadata))

        return docs

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Sync fallback implementation."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(self._aget_relevant_documents(query, run_manager=run_manager))
        else:
            return loop.run_until_complete(self._aget_relevant_documents(query, run_manager=run_manager))
