"""LangChain tool definitions for querying the Knowledge Graph."""

import logging
import re
from typing import Optional, Literal

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_core.runnables import RunnableConfig

from app.services.registry import get_services

logger = logging.getLogger(__name__)


class SearchModeClassification(BaseModel):
    mode: Literal["local", "global", "ark"] = Field(
        description="Classification category: 'local' for specific entities/facts, 'global' for corpus-wide themes/summaries, 'ark' for multi-hop trajectory traversal"
    )


def _classify_search_mode(query: str) -> str:
    """Use LLM with structured output to classify whether a query needs local, global, or ark search."""
    services = get_services()
    llm = services.llm

    prompt = (
        "Classify the following user query into one of three categories:\n"
        '- "local": The query asks about specific entities, people, facts, or 1-hop relationships\n'
        '- "global": The query asks about big-picture themes, summaries, or corpus-wide analysis\n'
        '- "ark": The query asks about complex multi-hop relational dependencies\n\n'
        f"Query: {query}"
    )
    try:
        if hasattr(llm, "with_structured_output"):
            structured_llm = llm.with_structured_output(SearchModeClassification)
            res = structured_llm.invoke(prompt)
            mode_val = getattr(res, "mode", None) or (res.get("mode") if isinstance(res, dict) else None)
            if mode_val in ["local", "global", "ark"]:
                return mode_val
    except Exception as e:
        logger.warning(f"Structured output search mode classification failed ({e}), using string fallback.")

    try:
        res = llm.invoke(prompt)
        content = res.content
        if isinstance(content, list):
            content = str(content[0])
        result = (
            content.strip().lower()
            if hasattr(res, "content")
            else str(res).strip().lower()
        )
        result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
        if result in ["local", "global", "ark"]:
            return result
    except Exception:
        pass
    return "local"


class QueryKnowledgeGraphInput(BaseModel):
    """Input schema for the Knowledge Graph search tool."""

    query: str = Field(
        description=(
            "A natural language question to search the Knowledge Graph. "
            "Write a clear, specific question in plain language. "
            "NEVER use Cypher, SQL, or any query syntax — the system handles that internally. "
            "Examples: 'What companies did John Smith work for?', "
            "'What are the main themes discussed across the documents?', "
            "'How is entity A connected to entity B through intermediaries?'"
        )
    )
    mode: str = Field(
        default="auto",
        description=(
            "Search strategy to use. Choose based on the nature of the question:\n"
            "- 'auto' (default): Let the system automatically classify and pick the best strategy.\n"
            "- 'local': For questions about specific entities, facts, definitions, or direct 1-hop relationships. "
            "Example: 'Who is the CEO of Acme Corp?', 'What is the capital of France?'\n"
            "- 'global': For big-picture, thematic, or summary questions that span many documents. "
            "Example: 'What are the major trends in the dataset?', 'Summarize the key findings across all reports.'\n"
            "- 'ark': For complex multi-hop reasoning that requires tracing chains of relationships across multiple entities. "
            "Example: 'How did company A's acquisition of B affect C's market position?', "
            "'What is the connection path between person X and organization Y?'"
        ),
    )


def query_knowledge_graph(
    query: str, mode: str = "auto", config: Optional[RunnableConfig] = None
) -> str:
    """Search the Knowledge Graph with a plain-language question."""
    services = get_services()
    cfg = services.graph_rag_config
    query_config = cfg.get("query", {})
    default_search_mode = query_config.get("search_mode", "auto")
    global_search_config = query_config.get("global_search", {})
    community_config = cfg.get("community_detection", {})
    ark_config = cfg.get("ark", {})

    logger.info(f"Agent is querying Knowledge Graph for: '{query}'")
    try:
        allowed_docs = None

        search_mode = default_search_mode if mode == "auto" else mode
        if search_mode == "auto":
            search_mode = _classify_search_mode(query)
            logger.info(f"Auto-classified search mode: {search_mode}")

        if search_mode == "global" and community_config.get("enabled", False):
            logger.info("Using GLOBAL search (map-reduce over communities)")
            res = services.global_search_service.search(
                query=query,
                level=global_search_config.get("default_level", 0),
                max_communities=global_search_config.get("max_communities", 10),
                allowed_docs=allowed_docs,
            )
            response = res.get("response", "")
            communities_used = res.get("selected_communities", [])
            if not response:
                return f"Query: '{query}'. Result: No relevant community information found."
            result_text = f"Query: '{query}'. [Global Search - Communities: {', '.join(str(c) for c in communities_used)}]\n{response}"
            return result_text

        elif search_mode == "ark" and ark_config.get("enabled", True):
            logger.info(
                "Using ARK search (Adaptive Retriever of Knowledge trajectory agents)"
            )
            try:
                res = services.ark_query_service.retrieve(query, allowed_docs=allowed_docs)
                context_str = res.get("context", "")

                if (
                    not context_str or "No relevant entities found" in context_str
                ) and ark_config.get("fallback_to_local", True):
                    logger.warning(
                        "ARK search returned empty results. Falling back to LOCAL search."
                    )
                    search_mode = "local"
                else:
                    return f"Query: '{query}'. ARK Trajectory Search Results:\n{context_str}"
            except Exception as e:
                logger.error(f"Error executing ARK search: {e}")
                if ark_config.get("fallback_to_local", True):
                    logger.warning("Falling back to LOCAL search due to ARK error.")
                    search_mode = "local"
                else:
                    return f"Query: '{query}'. Result: Error executing ARK retrieval."

        logger.info("Using LOCAL search (vector + graph traversal)")
        res = services.graph_query_service.retrieve_relevant_subgraph(
            query, allowed_docs=allowed_docs
        )
        context_str = res.get("context_str", "")
        if (
            not context_str
            or "No matching knowledge graph entities found" in context_str
        ):
            return f"Query: '{query}'. Result: No matching entities or relationships found in the Knowledge Graph."
        return f"Query: '{query}'. Knowledge Graph Subgraph Results:\n{context_str}"

    except Exception as e:
        logger.error(f"Error querying knowledge graph tool: {e}")
        return f"Error occurred while querying Knowledge Graph: {str(e)}"


def get_agent_tools():
    """Build and return the list of LangChain tools for the agent."""
    graph_rag_tool = StructuredTool.from_function(
        func=query_knowledge_graph,
        name="query_knowledge_graph",
        description=(
            "Search the Knowledge Graph using a natural language question to find entities, "
            "relationships, factual details, and corpus-wide themes. "
            "Always write your question in plain language — NEVER use Cypher or any database query syntax. "
            "Supports three search strategies via the 'mode' parameter: "
            "'local' for specific entity lookups, 'global' for thematic summaries across documents, "
            "and 'ark' for multi-hop relational reasoning. Use 'auto' (default) to let the system choose."
        ),
        args_schema=QueryKnowledgeGraphInput,
    )
    return [graph_rag_tool]
