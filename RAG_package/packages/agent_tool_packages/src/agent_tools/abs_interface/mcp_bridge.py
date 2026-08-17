"""MCPToolBridge — Adapter Pattern for MCP server tools.

Wraps MCP server tools (FastMCP protocol) into the same ServiceToolRegistry
so they sit alongside interface-based tools seamlessly.

Compatible with langchain-mcp-adapters >=0.1.0 which removed the context
manager API. Now uses ``await client.get_tools()`` directly.
"""

from typing import Any

from langchain_core.tools import StructuredTool


class MCPToolBridge:
    """Auto-wrap all MCP server tools into ServiceToolRegistry.

    Usage:
        tools = await MCPToolBridge.register_all(
            mcp_url="http://localhost:2211/sse",
            server_name="quotation",
        )
        # ... use tools ...
    """

    # Class-level storage for live MCP clients
    _clients: dict[str, Any] = {}

    @classmethod
    async def register_all(
        cls,
        mcp_url: str,
        server_name: str = "mcp",
    ) -> list[StructuredTool]:
        """Connect to MCP server, fetch all tools, register them.

        Args:
            mcp_url: SSE endpoint of the MCP server.
            server_name: Logical name for the MCP server.

        Returns:
            List of LangChain StructuredTools that were registered.
        """
        from langchain_mcp_adapters.client import MultiServerMCPClient

        from ..core.registry import ServiceToolRegistry

        client = MultiServerMCPClient(
            {server_name: {"url": mcp_url, "transport": "sse"}}
        )
        cls._clients[server_name] = client

        registered = []
        for tool in await client.get_tools():
            ServiceToolRegistry.register_raw(tool)
            registered.append(tool)

        return registered

    @classmethod
    async def shutdown(cls):
        """Clean up stored MCP client references."""
        cls._clients.clear()
