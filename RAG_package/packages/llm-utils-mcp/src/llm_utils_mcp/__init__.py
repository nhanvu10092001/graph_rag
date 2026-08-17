"""LLM Utils MCP - MCP server implementation with FastMCP and LangChain integration."""

from llm_utils_core import TaskPlugin

from .server import MCPServer, ServerConfig
from .tool_interface import ToolInterface, ToolParameter, ToolMetadata, BaseTool
from .tool_registry import ToolRegistry
from .exceptions import (
    MCPError,
    ToolRegistrationError,
    ValidationError,
    ExportError,
    DependencyError,
    ServerError,
    ConfigurationError,
)
from .validators import ValidatorRegistry
from .exporters import (
    FastMCPExporter,
    LangChainStructuredToolExporter,
    LangChainMCPAdapterExporter,
)

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "MCPServer",
    "ServerConfig",
    "ToolInterface",
    "ToolParameter",
    "ToolMetadata",
    "BaseTool",
    "ToolRegistry",
    # Exceptions
    "MCPError",
    "ToolRegistrationError",
    "ValidationError",
    "ExportError",
    "DependencyError",
    "ServerError",
    "ConfigurationError",
    # Validators
    "ValidatorRegistry",
    # Exporters
    "FastMCPExporter",
    "LangChainStructuredToolExporter",
    "LangChainMCPAdapterExporter",
    # Plugin integration
    "MCPServerPlugin",
]


class MCPServerPlugin(TaskPlugin):
    """Plugin implementation for integrating MCP server with llm-utils-core plugin system."""

    def name(self) -> str:
        """Return the name of the plugin."""
        return "mcp_server"

    def run(self, context: dict) -> dict:
        """Execute the plugin task synchronously.

        Args:
            context: Plugin execution context containing server configuration

        Returns:
            dict: Plugin execution result with server information
        """
        try:
            # Extract server configuration from context
            server_config = ServerConfig(
                name=context.get("server_name", "mcp_server"),
                description=context.get("server_description", "MCP Server"),
                version=context.get("server_version", "1.0.0"),
                host=context.get("host", "localhost"),
                port=context.get("port", 8000),
                debug=context.get("debug", False),
            )

            # Create MCP server
            server = MCPServer(server_config)

            # Add tools if provided in context
            tools = context.get("tools", [])
            for tool in tools:
                server.add_tool_sync(tool)

            # Return server information
            return {
                "status": "success",
                "server_info": server.get_server_info(),
                "export_capabilities": {
                    "fastmcp": True,
                    "langchain_structured": True,
                    "langchain_mcp_adapter": True,
                },
            }

        except Exception as e:
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}

    async def arun(self, context: dict) -> dict:
        """Execute the plugin task asynchronously.

        Args:
            context: Plugin execution context containing server configuration

        Returns:
            dict: Plugin execution result with server information
        """
        try:
            # Extract server configuration from context
            server_config = ServerConfig(
                name=context.get("server_name", "mcp_server"),
                description=context.get("server_description", "MCP Server"),
                version=context.get("server_version", "1.0.0"),
                host=context.get("host", "localhost"),
                port=context.get("port", 8000),
                debug=context.get("debug", False),
            )

            # Create MCP server
            server = MCPServer(server_config)

            # Add tools if provided in context
            tools = context.get("tools", [])
            for tool in tools:
                await server.add_tool(tool)

            # Return server information
            return {
                "status": "success",
                "server_info": server.get_server_info(),
                "export_capabilities": {
                    "fastmcp": True,
                    "langchain_structured": True,
                    "langchain_mcp_adapter": True,
                },
            }

        except Exception as e:
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}
