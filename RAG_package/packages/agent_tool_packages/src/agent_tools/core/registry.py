"""ServiceToolRegistry — Auto-collect and export all agent tools.

Design Pattern: Registry Pattern (variant of Service Locator)

Mirrors create_factory() in mq_packages — provides a single entry point
to access all registered tools without knowing their concrete types.

Usage:
    # Register via decorator (in tool files)
    @ServiceToolRegistry.register
    class BlurTool(AbstractServiceTool): ...

    # Register MCP tools
    await MCPToolBridge.register_all(mcp_url=...)

    # Get all tools for Deep Agent
    tools = ServiceToolRegistry.get_all_langchain_tools()
"""

from typing import Any

from langchain_core.tools import BaseTool


class ServiceToolRegistry:
    """Central registry for all agent tools.

    Supports two registration methods:
        1. @register decorator — for interface-based tools (AbstractBaseTool subclasses)
        2. register_raw() — for pre-built LangChain tools (e.g. from MCP server)
    """

    _tools: dict[str, Any] = {}  # name → AbstractBaseTool instance
    _raw_tools: list[BaseTool] = []  # pre-built LangChain tools

    @classmethod
    def register(cls, tool_class):
        """Decorator: register a tool class implementing AbstractBaseTool.

        Usage:
            @ServiceToolRegistry.register
            class MyTool(AbstractServiceTool):
                name = "my_tool"
                ...

        The class is instantiated immediately and stored by name.
        """
        instance = tool_class()
        cls._tools[instance.name] = instance
        return tool_class

    @classmethod
    def register_raw(cls, langchain_tool: BaseTool):
        """Register a pre-built LangChain tool (e.g. from MCP server).

        Args:
            langchain_tool: A LangChain BaseTool or StructuredTool instance.
        """
        cls._raw_tools.append(langchain_tool)

    @classmethod
    def get_tool(cls, name: str):
        """Get a registered interface-based tool by name.

        Args:
            name: The tool's name property.

        Returns:
            AbstractBaseTool instance, or None if not found.
        """
        return cls._tools.get(name)

    @classmethod
    def get_tools_by_names(cls, names: list[str]) -> list[BaseTool]:
        """Get a subset of registered tools as LangChain tools, filtered by name.

        Args:
            names: List of tool names to include.

        Returns:
            List of LangChain BaseTool instances matching the given names.
        """
        tools = []
        for name in names:
            tool = cls._tools.get(name)
            if tool:
                tools.append(tool.as_langchain_tool())
            else:
                # Check raw tools
                for raw in cls._raw_tools:
                    if raw.name == name:
                        tools.append(raw)
                        break
        return tools

    @classmethod
    def get_all_langchain_tools(cls) -> list[BaseTool]:
        """Export ALL tools as LangChain tools for Deep Agent registration.

        Combines:
            1. Interface-based tools → converted via as_langchain_tool()
            2. Raw LangChain tools → passed through as-is

        Returns:
            List of LangChain BaseTool instances ready for agent registration.
        """
        interface_tools = [t.as_langchain_tool() for t in cls._tools.values()]
        return interface_tools + list(cls._raw_tools)

    @classmethod
    def list_tools(cls) -> list[dict]:
        """List all registered tools with their metadata. Useful for debugging.

        Returns:
            List of dicts with name, description, and type for each tool.
        """
        result = []
        for name, tool in cls._tools.items():
            result.append(
                {
                    "name": name,
                    "description": tool.description,
                    "type": type(tool).__bases__[0].__name__,
                }
            )
        for tool in cls._raw_tools:
            result.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "type": "MCPTool (raw)",
                }
            )
        return result

    @classmethod
    def clear(cls):
        """Clear all registered tools. Useful for testing."""
        cls._tools.clear()
        cls._raw_tools.clear()
