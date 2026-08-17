"""Constructs the moderation-wrapped LangGraph agent."""

import logging
import threading
from typing import Optional

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

_compiled_graph = None
_lock = threading.Lock()


def get_compiled_graph():
    """Lazily build and return the compiled moderation+agent StateGraph."""
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph
    with _lock:
        if _compiled_graph is not None:
            return _compiled_graph
        _compiled_graph = _build_graph()
        return _compiled_graph


def _build_graph():
    from app.config import settings
    from app.services.registry import get_services
    from app.agent.tools import get_agent_tools
    from app.agent.security import (
        check_prompt_injection,
        respond_unsafe,
        route_after_check,
        AgentState,
    )
    from deepagents import async_create_deep_agent

    services = get_services()
    cfg = services.graph_rag_config

    agent_tools = get_agent_tools()

    system_prompt = (
        "You are a helpful AI assistant (Deep Agent) equipped with access to a structured Neo4j Knowledge Graph.\n"
        "Your goal is to answer queries by querying the knowledge graph. When asked about entities (e.g., people, organizations, locations) "
        "or their relationships/connections, you MUST use the `query_knowledge_graph` tool to fetch accurate context.\n\n"
        "After retrieving graph nodes and relations, synthesize the connections in your final answer so the user understands "
        "how the facts link together (e.g., 'Alice works at Heligate, which is located in Hanoi, the capital of Vietnam.')."
    )

    subagents_cfg = cfg.get("subagents", {})
    subagents_list = subagents_cfg.get("agents", []) if subagents_cfg.get("enabled", True) else []

    logger.info(
        f"Creating LangGraph Deep Agent with LLM model: {settings.llm_model} "
        f"(builtin_tools=['write_todos'], subagents: {len(subagents_list)})"
    )
    deep_agent = async_create_deep_agent(
        model=services.llm,
        tools=agent_tools,
        instructions=system_prompt,
        builtin_tools=["write_todos"],
        subagents=subagents_list if subagents_list else None,
    )

    moderation_workflow = StateGraph(AgentState)

    moderation_workflow.add_node("check_prompt_injection", check_prompt_injection)
    moderation_workflow.add_node("execute_agent", deep_agent)
    moderation_workflow.add_node("respond_unsafe", respond_unsafe)

    moderation_workflow.set_entry_point("check_prompt_injection")

    moderation_workflow.add_conditional_edges(
        "check_prompt_injection",
        route_after_check,
        {"execute_agent": "execute_agent", "respond_unsafe": "respond_unsafe"},
    )

    moderation_workflow.add_edge("execute_agent", END)
    moderation_workflow.add_edge("respond_unsafe", END)

    compiled = moderation_workflow.compile()
    logger.info("Moderation agent graph compiled successfully.")
    return compiled
