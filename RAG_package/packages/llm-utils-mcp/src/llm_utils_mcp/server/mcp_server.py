"""MCP Server implementation."""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import fastmcp
    import langchain_core.tools

from .server_config import ServerConfig
from ..tool_interface import ToolInterface, ToolMetadata
from ..exceptions import (
    ServerError,
    ToolRegistrationError,
    ExportError,
    DependencyError,
    ValidationError,
)

logger = logging.getLogger(__name__)


class MCPServer:
    """MCP Server with support for FastMCP, LangChain structured tools, and LangChain MCP adapters."""

    def __init__(self, config: ServerConfig):
        """Initialize MCP Server.

        Args:
            config: Server configuration

        Raises:
            ValidationError: If configuration is invalid
        """
        self.config = self._validate_config(config)
        self.tools: Dict[str, ToolInterface] = {}
        self._running = False

        # Setup logging
        if self.config.debug:
            logging.basicConfig(level=logging.DEBUG)

        logger.info(
            f"Initialized MCP Server '{self.config.name}' v{self.config.version}"
        )

    def _validate_config(self, config: ServerConfig) -> ServerConfig:
        """Validate server configuration.

        Args:
            config: Configuration to validate

        Returns:
            ServerConfig: Validated configuration

        Raises:
            ValidationError: If configuration is invalid
        """
        if not config.name:
            raise ValidationError(
                field="name",
                value=config.name,
                requirement="Server name cannot be empty",
            )

        if config.port < 1 or config.port > 65535:
            raise ValidationError(
                field="port",
                value=config.port,
                requirement="Port must be between 1 and 65535",
            )

        if config.max_tools < 1:
            raise ValidationError(
                field="max_tools",
                value=config.max_tools,
                requirement="max_tools must be at least 1",
            )

        if config.timeout <= 0:
            raise ValidationError(
                field="timeout",
                value=config.timeout,
                requirement="timeout must be positive",
            )

        return config

    def add_tool(self, tool: ToolInterface) -> None:
        """Add a tool to the server.

        Args:
            tool: Tool implementation to add

        Raises:
            ToolRegistrationError: If tool registration fails
            ServerError: If server limits exceeded
        """
        try:
            # Validate tool implementation
            tool.validate_implementation()
            metadata = tool.get_metadata()

            # Check server limits
            if len(self.tools) >= self.config.max_tools:
                raise ServerError(
                    operation="add_tool",
                    reason=f"Maximum number of tools ({self.config.max_tools}) exceeded",
                )

            # Check for name conflicts
            if metadata.name in self.tools:
                raise ToolRegistrationError(
                    tool_name=metadata.name,
                    reason=f"Tool with name '{metadata.name}' already exists",
                )

            # Register the tool
            self.tools[metadata.name] = tool
            logger.info(f"Successfully registered tool '{metadata.name}'")

        except (ValidationError, ToolRegistrationError):
            raise
        except Exception as e:
            raise ToolRegistrationError(
                tool_name=getattr(tool, "name", "<unknown>"),
                reason=f"Unexpected error during registration: {e}",
            ) from e

    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool from the server.

        Args:
            tool_name: Name of tool to remove

        Returns:
            bool: True if tool was removed, False if not found
        """
        if tool_name in self.tools:
            del self.tools[tool_name]
            logger.info(f"Removed tool '{tool_name}'")
            return True
        return False

    def get_tool_metadata(
        self, tool_name: Optional[str] = None
    ) -> Union[ToolMetadata, Dict[str, ToolMetadata]]:
        """Get metadata for tools.

        Args:
            tool_name: Specific tool name, or None for all tools

        Returns:
            ToolMetadata or Dict[str, ToolMetadata]: Tool metadata

        Raises:
            ValidationError: If tool not found
        """
        if tool_name:
            if tool_name not in self.tools:
                raise ValidationError(
                    field="tool_name",
                    value=tool_name,
                    requirement=f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}",
                )
            return self.tools[tool_name].get_metadata()

        return {name: tool.get_metadata() for name, tool in self.tools.items()}

    async def execute_tool(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool with given parameters.

        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters

        Returns:
            Dict[str, Any]: Tool execution result

        Raises:
            ValidationError: If tool not found or parameters invalid
            ServerError: If execution fails
        """
        if tool_name not in self.tools:
            raise ValidationError(
                field="tool_name",
                value=tool_name,
                requirement=f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}",
            )

        tool = self.tools[tool_name]

        try:
            # Validate parameters
            validated_params = tool.validate_parameters(parameters)

            # Execute with timeout
            result = await asyncio.wait_for(
                tool.execute(**validated_params), timeout=self.config.timeout
            )

            logger.debug(f"Successfully executed tool '{tool_name}'")
            return result

        except ValidationError:
            raise
        except asyncio.TimeoutError:
            raise ServerError(
                operation="execute_tool",
                reason=f"Tool '{tool_name}' execution timed out after {self.config.timeout}s",
            )
        except Exception as e:
            raise ServerError(
                operation="execute_tool",
                reason=f"Tool '{tool_name}' execution failed: {e}",
            ) from e

    def export_fastmcp(self) -> "fastmcp.FastMCP":
        """Export server as FastMCP server.

        Returns:
            fastmcp.FastMCP: FastMCP server instance

        Raises:
            DependencyError: If FastMCP not installed
            ExportError: If export fails
        """
        try:
            import fastmcp  # noqa: F401
        except ImportError:
            raise DependencyError(
                dependency="fastmcp",
                feature="FastMCP export",
                install_command="pip install fastmcp",
            )

        try:
            from ..exporters.fastmcp_exporter import FastMCPExporter

            exporter = FastMCPExporter(self)
            return exporter.export()
        except Exception as e:
            raise ExportError(
                export_type="FastMCP",
                reason=str(e),
                suggestion="Check that all tools are compatible with FastMCP format",
            ) from e

    def export_langchain_structured_tools(
        self,
    ) -> List["langchain_core.tools.StructuredTool"]:
        """Export tools as LangChain structured tools.

        Returns:
            List[StructuredTool]: List of LangChain structured tools

        Raises:
            DependencyError: If LangChain not installed
            ExportError: If export fails
        """
        try:
            from langchain_core.tools import StructuredTool  # noqa: F401
        except ImportError:
            raise DependencyError(
                dependency="langchain-core",
                feature="LangChain structured tools export",
                install_command="pip install langchain-core",
            )

        try:
            from ..exporters.langchain_structured_exporter import (
                LangChainStructuredToolExporter,
            )

            exporter = LangChainStructuredToolExporter(self)
            return exporter.export()
        except Exception as e:
            raise ExportError(
                export_type="LangChain structured tools",
                reason=str(e),
                suggestion="Check that all tools have valid parameter schemas",
            ) from e

    def export_langchain_mcp_adapter(self) -> Dict[str, Any]:
        """Export server configuration for LangChain MCP adapter.

        Returns:
            Dict[str, Any]: Configuration for langchain-mcp-adapters

        Raises:
            ExportError: If export fails
        """
        try:
            from ..exporters.langchain_mcp_adapter_exporter import (
                LangChainMCPAdapterExporter,
            )

            exporter = LangChainMCPAdapterExporter(self)
            return exporter.export()
        except Exception as e:
            raise ExportError(
                export_type="LangChain MCP adapter",
                reason=str(e),
                suggestion="Ensure server is properly configured for MCP adapter integration",
            ) from e

    def get_server_info(self) -> Dict[str, Any]:
        """Get server information and status.

        Returns:
            Dict[str, Any]: Server information
        """
        return {
            "name": self.config.name,
            "description": self.config.description,
            "version": self.config.version,
            "host": self.config.host,
            "port": self.config.port,
            "running": self._running,
            "tool_count": len(self.tools),
            "max_tools": self.config.max_tools,
            "tools": list(self.tools.keys()),
        }