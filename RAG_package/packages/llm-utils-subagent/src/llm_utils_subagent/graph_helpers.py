"""Graph helpers — utilities for building subagent graphs.

Provides ``SubagentGraphBuilder`` for quickly defining StateGraph-based
subagents with nodes, edges, and conditional routing.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from typing import Annotated, TypedDict

logger = logging.getLogger(__name__)


# ── Default State ────────────────────────────────────────────────────────────

class SubagentState(TypedDict):
    """Default state for subagent graphs — just messages."""
    messages: Annotated[list[BaseMessage], add_messages]


# ── SubagentGraphBuilder ─────────────────────────────────────────────────────

class SubagentGraphBuilder:
    """Fluent builder for creating subagent StateGraphs.

    Simplifies the boilerplate of building a LangGraph graph by providing
    a chainable API for adding nodes, edges, and conditional routing.

    Usage::

        from llm_utils_subagent.graph_helpers import SubagentGraphBuilder, SubagentState

        def build_my_graph(query, context, llm, tools, messages, **kw):
            builder = SubagentGraphBuilder(state_schema=SubagentState)

            async def step_one(state: SubagentState):
                response = await llm.ainvoke(state["messages"])
                return {"messages": [response]}

            async def step_two(state: SubagentState):
                # ... further processing
                return {"messages": [...]}

            graph = (
                builder
                .add_node("step_one", step_one)
                .add_node("step_two", step_two)
                .set_entry("step_one")
                .add_edge("step_one", "step_two")
                .set_finish("step_two")
                .compile()
            )
            return graph

        agent = create_subagent(
            name="my_agent",
            description="Custom pipeline",
            graph_builder=build_my_graph,
        )
    """

    def __init__(self, state_schema: type = SubagentState):
        self._graph = StateGraph(state_schema)
        self._entry_set = False

    def add_node(self, name: str, fn: Callable) -> "SubagentGraphBuilder":
        """Add a node to the graph."""
        self._graph.add_node(name, fn)
        return self

    def add_edge(self, from_node: str, to_node: str) -> "SubagentGraphBuilder":
        """Add a directed edge between two nodes."""
        self._graph.add_edge(from_node, to_node)
        return self

    def add_conditional_edges(
        self,
        from_node: str,
        condition: Callable,
        mapping: Dict[str, str],
    ) -> "SubagentGraphBuilder":
        """Add conditional routing from a node.

        Args:
            from_node: Source node name.
            condition: A function ``(state) -> str`` returning a key from mapping.
            mapping: ``{condition_result: target_node_name}``.
                     Use ``"__end__"`` for END.
        """
        # Replace "__end__" with the actual END sentinel
        resolved = {}
        for k, v in mapping.items():
            resolved[k] = END if v == "__end__" else v
        self._graph.add_conditional_edges(from_node, condition, resolved)
        return self

    def set_entry(self, node_name: str) -> "SubagentGraphBuilder":
        """Set the entry point of the graph."""
        self._graph.add_edge(START, node_name)
        self._entry_set = True
        return self

    def set_finish(self, node_name: str) -> "SubagentGraphBuilder":
        """Connect a node to END."""
        self._graph.add_edge(node_name, END)
        return self

    def compile(self, **kwargs) -> Any:
        """Compile and return the graph.

        Raises:
            ValueError: If no entry point was set.
        """
        if not self._entry_set:
            raise ValueError(
                "Graph has no entry point. Call .set_entry('node_name') first."
            )
        return self._graph.compile(**kwargs)

    @property
    def graph(self) -> StateGraph:
        """Access the underlying StateGraph for advanced configuration."""
        return self._graph


# ── Prebuilt graph patterns ──────────────────────────────────────────────────

def make_llm_node(
    llm: BaseChatModel,
    system_prompt: str = "",
    name: str = "llm_call",
) -> Callable:
    """Create a simple LLM node function for use in graphs.

    Returns an async function that invokes the LLM with the current messages
    and appends the response.

    Usage::

        builder = SubagentGraphBuilder()
        builder.add_node("think", make_llm_node(my_llm, "You are a thinker."))
    """
    async def _node(state: SubagentState) -> dict:
        messages = list(state["messages"])
        if system_prompt and not any(
            isinstance(m, SystemMessage) for m in messages
        ):
            messages.insert(0, SystemMessage(content=system_prompt))
        response = await llm.ainvoke(messages)
        return {"messages": [response]}

    _node.__name__ = name
    return _node


def make_tool_node(tools: list, name: str = "tools") -> Callable:
    """Create a ToolNode function for use in graphs.

    Usage::

        builder = SubagentGraphBuilder()
        builder.add_node("tools", make_tool_node([search_tool, calc_tool]))
    """
    from langgraph.prebuilt import ToolNode

    tool_node = ToolNode(tools)

    async def _node(state: SubagentState) -> dict:
        return await tool_node.ainvoke(state)

    _node.__name__ = name
    return _node
