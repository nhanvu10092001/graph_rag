"""SubagentTool — LangChain BaseTool wrapper for subagents.

Each invocation creates a **fresh message history**, executes the subagent,
returns the result, and discards all state.  This guarantees stateless behavior
between coordinator calls (history lives only within one invocation).

Three execution modes:

* ``simple`` — direct LLM call (prompt → response)
* ``react``  — LangGraph ``create_react_agent`` loop with tools
* ``graph``  — custom LangGraph ``StateGraph``, compiled fresh per call
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Type

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool as LangChainBaseTool
from pydantic import BaseModel, Field, PrivateAttr

logger = logging.getLogger(__name__)


# ── Input Schemas ────────────────────────────────────────────────────────────

class SimpleSubagentInput(BaseModel):
    """Default input schema — single query string."""
    query: str = Field(description="The task or query to send to this subagent")


class ReactSubagentInput(BaseModel):
    """Input schema for ReAct/Graph subagents that may receive structured context."""
    query: str = Field(description="The task or query to send to this subagent")
    context: str = Field(default="", description="Additional context or prior results to consider")


# ── SubagentTool ─────────────────────────────────────────────────────────────

class SubagentTool(LangChainBaseTool):
    """A LangChain tool that wraps a subagent.

    **Stateless by design:** every ``_arun`` / ``_run`` call builds a fresh
    message list, invokes the LLM / ReAct agent / custom graph, extracts
    the final text, and returns it.  No conversation history is persisted
    between calls.

    Three execution modes:

    * ``simple`` — direct LLM call (prompt → response)
    * ``react``  — LangGraph ``create_react_agent`` loop with tools
    * ``graph``  — user-defined ``StateGraph`` compiled fresh per call
    """

    name: str = ""
    description: str = ""
    args_schema: Type[BaseModel] = SimpleSubagentInput

    # ── Private attributes (not serialized) ──────────────────────────────
    _llm: Optional[BaseChatModel] = PrivateAttr(default=None)
    _system_prompt: str = PrivateAttr(default="")
    _mode: str = PrivateAttr(default="simple")
    _tools: list = PrivateAttr(default_factory=list)
    _extra_messages: list = PrivateAttr(default_factory=list)
    _output_parser: Optional[Callable] = PrivateAttr(default=None)
    _graph_builder: Optional[Callable] = PrivateAttr(default=None)
    _compiled_graph: Any = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        name: str,
        description: str,
        system_prompt: str = "",
        llm: Optional[BaseChatModel] = None,
        mode: str = "simple",
        tools: Optional[List] = None,
        input_schema: Optional[Type[BaseModel]] = None,
        extra_messages: Optional[List] = None,
        output_parser: Optional[Callable] = None,
        graph_builder: Optional[Callable] = None,
        compiled_graph: Any = None,
        **kwargs: Any,
    ):
        """
        Args:
            name: Tool name visible to the coordinator.
            description: When should the coordinator call this subagent?
            system_prompt: Persona / instructions (used in simple/react modes).
            llm: LangChain chat model (required for simple/react, optional for graph).
            mode: "simple" | "react" | "graph"
            tools: Tools for react mode.
            input_schema: Custom Pydantic input schema.
            extra_messages: Extra messages prepended before user query.
            output_parser: Post-process the raw output string.
            graph_builder: A callable ``(query, context, **kwargs) -> CompiledStateGraph``
                          that builds a fresh graph per invocation. Use this when
                          each call needs a dynamically constructed graph.
            compiled_graph: A pre-compiled ``CompiledStateGraph``. The tool will
                           invoke it with fresh state each call. Use this when
                           the graph structure is static.
        """
        if mode == "graph":
            schema = input_schema or ReactSubagentInput
        elif mode == "react":
            schema = input_schema or ReactSubagentInput
        else:
            schema = input_schema or SimpleSubagentInput

        super().__init__(name=name, description=description, args_schema=schema, **kwargs)
        self._llm = llm
        self._system_prompt = system_prompt
        self._mode = mode
        self._tools = tools or []
        self._extra_messages = extra_messages or []
        self._output_parser = output_parser
        self._graph_builder = graph_builder
        self._compiled_graph = compiled_graph

    # ── Async execution (primary) ────────────────────────────────────────

    async def _arun(self, query: str, context: str = "", **kwargs: Any) -> str:
        """Execute the subagent with a fresh, ephemeral state."""

        # 1. Build fresh messages
        messages: list = []
        if self._system_prompt:
            messages.append(SystemMessage(content=self._system_prompt))
        messages.extend(self._extra_messages)

        user_content = query
        if context:
            user_content = f"{query}\n\n## Additional Context\n{context}"
        messages.append(HumanMessage(content=user_content))

        # 2. Execute based on mode
        if self._mode == "graph":
            result_text = await self._run_graph(messages, query, context, **kwargs)
        elif self._mode == "react" and self._tools:
            result_text = await self._run_react(messages)
        else:
            result_text = await self._run_simple(messages)

        # 3. Optional post-processing
        if self._output_parser:
            try:
                result_text = self._output_parser(result_text)
            except Exception as e:
                logger.warning(f"Subagent '{self.name}' output_parser failed: {e}")

        logger.debug(f"Subagent '{self.name}' completed ({self._mode} mode)")
        return result_text

    # ── Sync fallback ────────────────────────────────────────────────────

    def _run(self, query: str, context: str = "", **kwargs: Any) -> str:
        """Synchronous fallback — runs the async version in a new loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, self._arun(query, context, **kwargs)).result()
        return asyncio.run(self._arun(query, context, **kwargs))

    # ── Internal execution strategies ────────────────────────────────────

    async def _run_simple(self, messages: list) -> str:
        """Direct LLM call — no tool loop."""
        if not self._llm:
            raise ValueError(f"Subagent '{self.name}': LLM is required for simple mode")
        response = await self._llm.ainvoke(messages)
        return str(response.content)

    async def _run_react(self, messages: list) -> str:
        """ReAct agent loop with tools — fresh agent per call."""
        if not self._llm:
            raise ValueError(f"Subagent '{self.name}': LLM is required for react mode")

        try:
            from langchain.agents import create_agent
            agent = create_agent(self._llm, tools=self._tools)
        except ImportError:
            from langgraph.prebuilt import create_react_agent
            agent = create_react_agent(self._llm, tools=self._tools)

        result = await agent.ainvoke({"messages": messages})
        return self._extract_result(result)

    async def _run_graph(
        self,
        messages: list,
        query: str,
        context: str = "",
        **kwargs: Any,
    ) -> str:
        """Execute a custom LangGraph StateGraph — fresh state per call.

        Supports two patterns:
        1. ``compiled_graph`` — a pre-compiled graph, invoked with fresh input
        2. ``graph_builder`` — a callable that builds a fresh graph per call
        """
        graph = None

        # Pattern 1: Dynamic graph builder (called every invocation)
        if self._graph_builder:
            graph = self._graph_builder(
                query=query,
                context=context,
                llm=self._llm,
                tools=self._tools,
                messages=messages,
                **kwargs,
            )
            # If builder returns an uncompiled graph, compile it
            if hasattr(graph, "compile"):
                graph = graph.compile()

        # Pattern 2: Static pre-compiled graph
        elif self._compiled_graph:
            graph = self._compiled_graph

        if graph is None:
            raise ValueError(
                f"Subagent '{self.name}': graph mode requires either "
                "'compiled_graph' or 'graph_builder'"
            )

        # Invoke with fresh state — no history carried over
        result = await graph.ainvoke({"messages": messages})
        return self._extract_result(result)

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_result(result: dict) -> str:
        """Extract the final text from a graph/agent result dict."""
        if "messages" in result and result["messages"]:
            final_msg = result["messages"][-1]
            if hasattr(final_msg, "content"):
                return str(final_msg.content)
            return str(final_msg)

        # Fallback: try common output keys
        for key in ("output", "result", "answer", "response"):
            if key in result:
                return str(result[key])

        return str(result)
