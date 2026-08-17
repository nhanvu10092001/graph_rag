"""SubagentRegistry — manage and export multiple subagents.

Provides a central place to register subagents and export them as a list
of tools that can be passed directly to a coordinator agent.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Type

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from .factory import create_simple_subagent, create_react_subagent, create_graph_subagent
from .tool_wrapper import SubagentTool

logger = logging.getLogger(__name__)


class SubagentRegistry:
    """Registry for managing subagents.

    Usage::

        registry = SubagentRegistry()

        # 3 create methods — one per mode
        registry.create_simple(name="planner", ..., llm=my_llm)
        registry.create_react(name="executor", ..., llm=my_llm, tools=[...])
        registry.create_graph(name="analyst", ..., graph_builder=my_builder)

        # Or register an existing tool
        registry.register(my_existing_tool)

        # Export all as tools for a coordinator
        coordinator = create_react_agent(llm, tools=registry.as_tools())
    """

    def __init__(self) -> None:
        self._agents: Dict[str, BaseTool] = {}

    # ── Registration ─────────────────────────────────────────────────────

    def register(self, tool: BaseTool) -> None:
        """Register an existing BaseTool (or SubagentTool) by its name."""
        if tool.name in self._agents:
            logger.warning(f"Overwriting existing subagent '{tool.name}'")
        self._agents[tool.name] = tool
        logger.info(f"Registered subagent '{tool.name}'")

    # ── Create + Register (3 methods) ────────────────────────────────────

    def create_simple(
        self,
        name: str,
        description: str,
        system_prompt: str,
        llm: BaseChatModel,
        **kwargs: Any,
    ) -> SubagentTool:
        """Create a simple subagent (LLM call) and register it."""
        tool = create_simple_subagent(name, description, system_prompt, llm, **kwargs)
        self.register(tool)
        return tool

    def create_react(
        self,
        name: str,
        description: str,
        system_prompt: str,
        llm: BaseChatModel,
        tools: List,
        **kwargs: Any,
    ) -> SubagentTool:
        """Create a ReAct subagent (with tools) and register it."""
        tool = create_react_subagent(name, description, system_prompt, llm, tools, **kwargs)
        self.register(tool)
        return tool

    def create_graph(
        self,
        name: str,
        description: str,
        **kwargs: Any,
    ) -> SubagentTool:
        """Create a graph subagent (custom StateGraph) and register it."""
        tool = create_graph_subagent(name, description, **kwargs)
        self.register(tool)
        return tool

    # ── Retrieval ────────────────────────────────────────────────────────

    def get(self, name: str) -> BaseTool:
        """Get a registered subagent by name.

        Raises:
            KeyError: If no subagent with that name is registered.
        """
        if name not in self._agents:
            available = ", ".join(self._agents.keys()) or "(none)"
            raise KeyError(
                f"Subagent '{name}' not found. Available: {available}"
            )
        return self._agents[name]

    def unregister(self, name: str) -> None:
        """Remove a subagent from the registry."""
        if name in self._agents:
            del self._agents[name]
            logger.info(f"Unregistered subagent '{name}'")
        else:
            logger.warning(f"Cannot unregister '{name}': not found")

    # ── Export ────────────────────────────────────────────────────────────

    def as_tools(self) -> List[BaseTool]:
        """Export all registered subagents as a list of BaseTool.

        This is the primary way to pass subagents to a coordinator::

            coordinator = create_react_agent(llm, tools=registry.as_tools())
        """
        return list(self._agents.values())

    def list_agents(self) -> List[Dict[str, str]]:
        """Return metadata for all registered subagents."""
        return [
            {"name": t.name, "description": t.description}
            for t in self._agents.values()
        ]

    # ── Utilities ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        return name in self._agents

    def __repr__(self) -> str:
        names = ", ".join(self._agents.keys())
        return f"SubagentRegistry([{names}])"

    def clear(self) -> None:
        """Remove all registered subagents."""
        self._agents.clear()
