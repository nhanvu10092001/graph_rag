"""Factory — 3 factory methods for creating subagents.

* ``create_simple_subagent()`` — direct LLM call (prompt → response)
* ``create_react_subagent()``  — ReAct agent loop with tools
* ``create_graph_subagent()``  — custom StateGraph per subagent
"""

import logging
from typing import Any, Callable, List, Optional, Type

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from .tool_wrapper import SubagentTool

logger = logging.getLogger(__name__)


def create_simple_subagent(
    name: str,
    description: str,
    system_prompt: str,
    llm: BaseChatModel,
    *,
    input_schema: Optional[Type[BaseModel]] = None,
    extra_messages: Optional[list] = None,
    output_parser: Optional[Callable] = None,
) -> SubagentTool:
    """Create a **simple** subagent — direct LLM call, no tools.

    Each invocation: ``[SystemMessage, HumanMessage]`` → LLM → text response.
    Stateless between calls.

    Args:
        name: Tool name visible to the coordinator.
        description: When should the coordinator call this?
        system_prompt: Persona / behaviour instructions.
        llm: Any LangChain chat model.
        input_schema: Custom Pydantic input schema (optional).
        extra_messages: Extra messages prepended before user query.
        output_parser: Post-process raw output string.

    Example::

        planner = create_simple_subagent(
            name="planner",
            description="Break a task into numbered steps",
            system_prompt="You are a planning agent. Output ONLY a numbered plan.",
            llm=ChatOpenAI(model="gpt-4o-mini"),
        )
    """
    tool = SubagentTool(
        name=name,
        description=description,
        system_prompt=system_prompt,
        llm=llm,
        mode="simple",
        input_schema=input_schema,
        extra_messages=extra_messages,
        output_parser=output_parser,
    )
    logger.info(f"Created simple subagent '{name}'")
    return tool


def create_react_subagent(
    name: str,
    description: str,
    system_prompt: str,
    llm: BaseChatModel,
    tools: List,
    *,
    input_schema: Optional[Type[BaseModel]] = None,
    extra_messages: Optional[list] = None,
    output_parser: Optional[Callable] = None,
) -> SubagentTool:
    """Create a **ReAct** subagent — agent loop with tools.

    Each invocation: builds a fresh ``create_react_agent(llm, tools)``
    → runs ReAct loop → returns final message. Stateless between calls.

    Args:
        name: Tool name visible to the coordinator.
        description: When should the coordinator call this?
        system_prompt: Persona / behaviour instructions.
        llm: Any LangChain chat model.
        tools: List of LangChain tools available to this subagent.
        input_schema: Custom Pydantic input schema (optional).
        extra_messages: Extra messages prepended before user query.
        output_parser: Post-process raw output string.

    Example::

        executor = create_react_subagent(
            name="executor",
            description="Execute a plan using RAG and web search",
            system_prompt="You are an execution agent.",
            llm=ChatOpenAI(model="gpt-4o"),
            tools=[rag_query_tool, web_search_tool],
        )
    """
    if not tools:
        logger.warning(
            f"Subagent '{name}': react mode with empty tools list. "
            "Consider using create_simple_subagent() instead."
        )

    tool = SubagentTool(
        name=name,
        description=description,
        system_prompt=system_prompt,
        llm=llm,
        mode="react",
        tools=tools,
        input_schema=input_schema,
        extra_messages=extra_messages,
        output_parser=output_parser,
    )
    logger.info(f"Created react subagent '{name}' (tools={len(tools)})")
    return tool


def create_graph_subagent(
    name: str,
    description: str,
    *,
    graph_builder: Optional[Callable] = None,
    compiled_graph: Any = None,
    system_prompt: str = "",
    llm: Optional[BaseChatModel] = None,
    tools: Optional[List] = None,
    input_schema: Optional[Type[BaseModel]] = None,
    extra_messages: Optional[list] = None,
    output_parser: Optional[Callable] = None,
) -> SubagentTool:
    """Create a **graph** subagent — custom StateGraph per subagent.

    Each invocation: builds (or re-uses) a compiled graph → invokes with
    fresh state → returns final message. Stateless between calls.

    Two patterns supported:

    1. **``graph_builder``** — a callable that builds a fresh graph per call.
       Signature: ``(query, context, llm, tools, messages, **kw) -> CompiledGraph``

    2. **``compiled_graph``** — a pre-compiled graph, invoked with fresh input.

    Args:
        name: Tool name visible to the coordinator.
        description: When should the coordinator call this?
        graph_builder: Callable that builds a fresh graph per invocation.
        compiled_graph: Pre-compiled CompiledStateGraph.
        system_prompt: Optional system prompt (prepended to messages).
        llm: LLM passed to graph_builder (optional).
        tools: Tools passed to graph_builder (optional).
        input_schema: Custom Pydantic input schema (optional).
        extra_messages: Extra messages prepended before user query.
        output_parser: Post-process raw output string.

    Example with graph_builder::

        from llm_utils_subagent.graph_helpers import SubagentGraphBuilder, make_llm_node

        def build_analysis(query, context, llm, tools, messages, **kw):
            builder = SubagentGraphBuilder()
            builder.add_node("think", make_llm_node(llm, "Analyze deeply."))
            builder.add_node("summarize", make_llm_node(llm, "Summarize."))
            builder.set_entry("think")
            builder.add_edge("think", "summarize")
            builder.set_finish("summarize")
            return builder.compile()

        analyst = create_graph_subagent(
            name="analyst",
            description="Multi-step analysis pipeline",
            graph_builder=build_analysis,
            llm=my_llm,
        )

    Example with compiled_graph::

        from langgraph.graph import StateGraph, START, END

        graph = StateGraph(MyState)
        graph.add_node(...)
        compiled = graph.compile()

        agent = create_graph_subagent(
            name="processor",
            description="Custom processing pipeline",
            compiled_graph=compiled,
        )
    """
    if not graph_builder and not compiled_graph:
        raise ValueError(
            f"Subagent '{name}': graph mode requires either "
            "'graph_builder' or 'compiled_graph'."
        )

    tool = SubagentTool(
        name=name,
        description=description,
        system_prompt=system_prompt,
        llm=llm,
        mode="graph",
        tools=tools,
        input_schema=input_schema,
        extra_messages=extra_messages,
        output_parser=output_parser,
        graph_builder=graph_builder,
        compiled_graph=compiled_graph,
    )
    logger.info(f"Created graph subagent '{name}'")
    return tool
