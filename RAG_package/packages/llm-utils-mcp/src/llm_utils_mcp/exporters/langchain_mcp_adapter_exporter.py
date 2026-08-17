"""LangChain MCP adapter exporter using langchain-mcp-adapters."""

import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from pathlib import Path

from ..exceptions import ExportError, DependencyError

if TYPE_CHECKING:
    from ..server import MCPServer
    from ..tool_interface import ToolParameter

logger = logging.getLogger(__name__)


class LangChainMCPAdapterExporter:
    """Exports MCP server configuration for use with langchain-mcp-adapters."""

    def __init__(self, server: "MCPServer"):
        """Initialize LangChain MCP adapter exporter.

        Args:
            server: MCP server to export from
        """
        self.server = server

    def export(self) -> Dict[str, Any]:
        """Export server configuration for LangChain MCP adapter.

        Returns:
            Dict[str, Any]: Configuration for langchain-mcp-adapters

        Raises:
            ExportError: If export fails
        """
        try:
            config = {
                "server_name": self.server.config.name,
                "server_config": self._create_server_config(),
                "tools": self._create_tools_config(),
                "connection": self._create_connection_config(),
                "metadata": self._create_metadata_config(),
            }

            logger.info(
                f"Successfully created LangChain MCP adapter configuration for {len(self.server.tools)} tools"
            )
            return config

        except Exception as e:
            raise ExportError(
                export_type="LangChain MCP adapter configuration",
                reason=f"Failed to create adapter configuration: {e}",
                suggestion="Check server configuration and ensure all tools are properly defined",
            ) from e

    def _create_server_config(self) -> Dict[str, Any]:
        """Create server configuration section.

        Returns:
            Dict[str, Any]: Server configuration
        """
        return {
            "name": self.server.config.name,
            "description": self.server.config.description,
            "version": self.server.config.version,
            "transport": "stdio",  # Default transport for MCP
            "capabilities": {
                "tools": True,
                "resources": False,  # Can be extended later
                "prompts": False,  # Can be extended later
                "logging": self.server.config.debug,
            },
        }

    def _create_tools_config(self) -> Dict[str, Dict[str, Any]]:
        """Create tools configuration section.

        Returns:
            Dict[str, Dict[str, Any]]: Tools configuration
        """
        tools_config = {}

        for tool_name, tool in self.server.tools.items():
            try:
                metadata = tool.get_metadata()

                tool_config = {
                    "name": metadata.name,
                    "description": metadata.description,
                    "parameters": self._convert_parameters_for_adapter(
                        metadata.parameters
                    ),
                    "return_type": getattr(metadata, "return_type", None),
                    "examples": getattr(metadata, "examples", []),
                }

                tools_config[tool_name] = tool_config

            except Exception as e:
                logger.warning(
                    f"Failed to export tool '{tool_name}' for MCP adapter: {e}"
                )
                raise ExportError(
                    export_type="LangChain MCP adapter tool config",
                    reason=f"Failed to process tool '{tool_name}': {e}",
                    suggestion="Check tool metadata and parameter definitions",
                ) from e

        return tools_config

    def _convert_parameters_for_adapter(
        self, parameters: List["ToolParameter"]
    ) -> Dict[str, Any]:
        """Convert tool parameters for MCP adapter format.

        Args:
            parameters: Tool parameters

        Returns:
            Dict[str, Any]: Adapter-compatible parameters
        """
        adapter_params = {}

        for param in parameters:
            adapter_params[param.name] = {
                "type": param.type.__name__,
                "description": param.description,
                "required": param.required,
                "default": param.default if param.default is not None else None,
            }

        return adapter_params

    def _create_connection_config(self) -> Dict[str, Any]:
        """Create connection configuration section.

        Returns:
            Dict[str, Any]: Connection configuration
        """
        return {
            "transport": "stdio",
            "host": self.server.config.host,
            "port": self.server.config.port,
            "timeout": self.server.config.timeout,
            "max_retries": 3,
            "retry_delay": 1.0,
        }

    def _create_metadata_config(self) -> Dict[str, Any]:
        """Create metadata configuration section.

        Returns:
            Dict[str, Any]: Metadata configuration
        """
        return {
            "created_by": "llm-utils-mcp",
            "export_version": "1.0.0",
            "tool_count": len(self.server.tools),
            "server_info": self.server.get_server_info(),
            "compatible_with": {
                "langchain_mcp_adapters": ">=0.1.0",
                "mcp_protocol": ">=0.1.0",
            },
        }

    def export_as_client_config(
        self, client_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Export configuration for MultiServerMCPClient.

        Args:
            client_name: Optional client name, defaults to server name

        Returns:
            Dict[str, Any]: Client configuration
        """
        try:
            client_name = client_name or self.server.config.name

            config = {
                client_name: {
                    "command": "python",
                    "args": ["-m", "llm_utils_mcp.server"],
                    "transport": "stdio",
                    "env": {
                        "MCP_SERVER_NAME": self.server.config.name,
                        "MCP_SERVER_PORT": str(self.server.config.port),
                        "MCP_DEBUG": str(self.server.config.debug).lower(),
                    },
                    "capabilities": ["tools"],
                    "timeout": self.server.config.timeout,
                }
            }

            logger.info(
                f"Created MultiServerMCPClient configuration for '{client_name}'"
            )
            return config

        except Exception as e:
            raise ExportError(
                export_type="LangChain MCP client configuration",
                reason=f"Failed to create client configuration: {e}",
                suggestion="Check server configuration and client requirements",
            ) from e

    def save_config_file(
        self, file_path: Path, include_client_config: bool = True
    ) -> None:
        """Save configuration to a file.

        Args:
            file_path: Path to save configuration file
            include_client_config: Whether to include client configuration

        Raises:
            ExportError: If file saving fails
        """
        try:
            import yaml
        except ImportError:
            raise DependencyError(
                dependency="pyyaml",
                feature="YAML configuration file export",
                install_command="pip install pyyaml",
            )

        try:
            config = self.export()

            if include_client_config:
                config["client_config"] = self.export_as_client_config()

            with open(file_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)

            logger.info(f"Saved MCP adapter configuration to {file_path}")

        except Exception as e:
            raise ExportError(
                export_type="MCP adapter configuration file",
                reason=f"Failed to save configuration file: {e}",
                suggestion="Check file path permissions and disk space",
            ) from e

    def get_usage_example(self) -> str:
        """Get usage example for the exported configuration.

        Returns:
            str: Usage example code
        """
        client_name = self.server.config.name

        example = f"""
# Example usage with langchain-mcp-adapters

from langchain_mcp import MultiServerMCPClient

# Initialize client
client = MultiServerMCPClient({{
    "{client_name}": {{
        "command": "python",
        "args": ["-m", "llm_utils_mcp.server"],
        "transport": "stdio"
    }}
}})

# Load tools
tools = await client.get_tools()
print(f"Loaded {{len(tools)}} tools: {{[tool.name for tool in tools]}}")

# Use with LangGraph agent
from langgraph.prebuilt import create_react_agent

agent = create_react_agent("openai:gpt-4", tools)
result = await agent.ainvoke({{"messages": [("user", "Your query here")]}})
print(result)
"""
        return example.strip()

    def validate_adapter_compatibility(self) -> Dict[str, Any]:
        """Validate compatibility with langchain-mcp-adapters.

        Returns:
            Dict[str, Any]: Validation results
        """
        validation = {
            "compatible": True,
            "issues": [],
            "warnings": [],
            "tool_validation": {},
        }

        # Check server configuration
        if not self.server.config.name:
            validation["issues"].append("Server name is required")
            validation["compatible"] = False

        if not self.server.tools:
            validation["warnings"].append("No tools defined in server")

        # Check each tool
        for tool_name, tool in self.server.tools.items():
            tool_validation = {"compatible": True, "issues": []}

            try:
                metadata = tool.get_metadata()

                # Check tool name
                if not metadata.name:
                    tool_validation["issues"].append("Tool name is required")
                    tool_validation["compatible"] = False

                # Check parameters
                for param in metadata.parameters:
                    if not param.name:
                        tool_validation["issues"].append("Parameter name is required")
                        tool_validation["compatible"] = False

                    if not param.description:
                        tool_validation["warnings"] = tool_validation.get(
                            "warnings", []
                        )
                        tool_validation["warnings"].append(
                            f"Parameter '{param.name}' missing description"
                        )

            except Exception as e:
                tool_validation["issues"].append(f"Tool validation failed: {e}")
                tool_validation["compatible"] = False

            validation["tool_validation"][tool_name] = tool_validation

            if not tool_validation["compatible"]:
                validation["compatible"] = False

        return validation
