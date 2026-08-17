"""FastMCP exporter with clear error reporting."""

import logging
from typing import Dict, Any, List, TYPE_CHECKING
from dataclasses import dataclass

from ..exceptions import ExportError, DependencyError
from ..tool_interface import ToolInterface, ToolParameter

if TYPE_CHECKING:
    from ..server import MCPServer
    import fastmcp

logger = logging.getLogger(__name__)


@dataclass
class FastMCPToolConfig:
    """Configuration for FastMCP tool."""

    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]]


class FastMCPExporter:
    """Exports MCP server tools to FastMCP format."""

    def __init__(self, server: "MCPServer"):
        """Initialize FastMCP exporter.

        Args:
            server: MCP server to export from
        """
        self.server = server

    def export(self) -> "fastmcp.FastMCP":
        """Export server as FastMCP server.

        Returns:
            fastmcp.FastMCP: FastMCP server instance

        Raises:
            DependencyError: If FastMCP not installed
            ExportError: If export fails
        """
        try:
            import fastmcp
        except ImportError:
            raise DependencyError(
                dependency="fastmcp",
                feature="FastMCP server export",
                install_command="pip install fastmcp",
            )

        try:
            # Create FastMCP server
            fastmcp_server = fastmcp.FastMCP(name=self.server.config.name)

            # Add tools to FastMCP server
            for tool_name, tool in self.server.tools.items():
                try:
                    self._add_tool_to_fastmcp(fastmcp_server, tool_name, tool)
                except Exception as e:
                    logger.warning(
                        f"Failed to export tool '{tool_name}' to FastMCP: {e}"
                    )
                    raise ExportError(
                        export_type="FastMCP",
                        reason=f"Failed to export tool '{tool_name}': {e}",
                        suggestion="Check that tool parameters are compatible with FastMCP schema",
                    ) from e

            logger.info(
                f"Successfully exported {len(self.server.tools)} tools to FastMCP"
            )
            return fastmcp_server

        except DependencyError:
            raise
        except ExportError:
            raise
        except Exception as e:
            raise ExportError(
                export_type="FastMCP",
                reason=f"Unexpected error during export: {e}",
                suggestion="Check server configuration and tool implementations",
            ) from e

    def _add_tool_to_fastmcp(
        self, fastmcp_server: "fastmcp.FastMCP", tool_name: str, tool: ToolInterface
    ) -> None:
        """Add a single tool to FastMCP server.

        Args:
            fastmcp_server: FastMCP server instance
            tool_name: Name of tool to add
            tool: Tool implementation

        Raises:
            ExportError: If tool cannot be added
        """
        try:
            metadata = tool.get_metadata()

            # Convert parameters to FastMCP schema with types and required info

            # Create function with explicit parameters but pass schema via meta
            param_names = [p.name for p in metadata.parameters]
            func_params = ", ".join(param_names) if param_names else ""
            func_params = ", ".join(
                (
                    f"{p.name}: {p.type.__name__} = '{p.default}'"
                    if p.default is not None and p.type.__name__ == "str"
                    else (
                        f"{p.name}: {p.type.__name__} = {p.default}"
                        if p.default is not None
                        else f"{p.name}: {p.type.__name__}"
                    )
                )
                for p in metadata.parameters
            )

            # Dynamically create function with explicit parameters
            arg_desc = ", ".join(
                (
                    f"{p.name} ({p.type.__name__}): {p.description}. Defaults to '{p.default}'\n"
                    if p.default is not None and p.type.__name__ == "str"
                    else (
                        f"{p.name} ({p.type.__name__}): {p.description}. Defaults to {p.default}\n"
                        if p.default is not None
                        else f"{p.name} ({p.type.__name__})\n"
                    )
                )
                for p in metadata.parameters
            )
            func_desc = f"""{metadata.description}
Args:
{arg_desc}
"""
            func_body = f"""
async def tool_wrapper({func_params}):
    \"\"\"
    {func_desc}
    \"\"\"
    try:
        # Build kwargs from explicit parameters
        kwargs = {{{", ".join(f'"{name}": {name}' for name in param_names)}}}
        # Execute tool via server to get validation and error handling
        result = await self.server.execute_tool("{tool_name}", kwargs)
        return result
    except Exception as e:
        # Convert to FastMCP-compatible error
        raise RuntimeError(f"Tool execution failed: {{e}}") from e
"""

            # Execute the function definition in local scope
            globals_dict = {"self": self}
            exec(func_body, globals_dict)
            tool_wrapper = globals_dict["tool_wrapper"]

            # Use FastMCP's tool decorator with parameter schema in meta
            fastmcp_server.tool(
                name=metadata.name,
                description=func_desc,
            )(tool_wrapper)

            logger.debug(
                f"Added tool '{tool_name}' to FastMCP server with {len(metadata.parameters)} parameters and schema"
            )

        except Exception as e:
            raise ExportError(
                export_type="FastMCP tool",
                reason=f"Failed to add tool '{tool_name}': {e}",
                suggestion="Ensure tool has valid metadata and compatible with FastMCP",
            ) from e

    def _convert_parameters_to_fastmcp_schema(
        self, parameters: List[ToolParameter]
    ) -> Dict[str, Any]:
        """Convert tool parameters to FastMCP JSON schema.

        Args:
            parameters: Tool parameters

        Returns:
            Dict[str, Any]: FastMCP-compatible parameter schema

        Raises:
            ExportError: If parameter conversion fails
        """
        try:
            schema = {"type": "object", "properties": {}, "required": []}

            for param in parameters:
                # Convert Python type to JSON schema type
                json_type = self._python_type_to_json_type(param.type)

                schema["properties"][param.name] = {
                    "type": json_type,
                    "description": param.description,
                }

                # Add default value if provided
                if param.default is not None:
                    schema["properties"][param.name]["default"] = param.default

                # Add to required list if required
                if param.required:
                    schema["required"].append(param.name)

            return schema

        except Exception as e:
            raise ExportError(
                export_type="FastMCP parameter schema",
                reason=f"Failed to convert parameters: {e}",
                suggestion="Check that all parameter types are supported by JSON schema",
            ) from e

    def _python_type_to_json_type(self, python_type: type) -> str:
        """Convert Python type to JSON schema type.

        Args:
            python_type: Python type

        Returns:
            str: JSON schema type name

        Raises:
            ExportError: If type conversion fails
        """
        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        if python_type in type_mapping:
            return type_mapping[python_type]

        # Handle generic types
        if hasattr(python_type, "__origin__"):
            origin = python_type.__origin__
            if origin in type_mapping:
                return type_mapping[origin]

        # Default to string for unknown types
        logger.warning(f"Unknown type {python_type}, defaulting to string")
        return "string"

    def get_export_config(self) -> FastMCPToolConfig:
        """Get FastMCP export configuration.

        Returns:
            FastMCPToolConfig: Export configuration
        """
        parameters_schema = {}

        for tool_name, tool in self.server.tools.items():
            metadata = tool.get_metadata()
            tool_schema = self._convert_parameters_to_fastmcp_schema(
                metadata.parameters
            )
            parameters_schema[tool_name] = tool_schema

        return FastMCPToolConfig(
            name=self.server.config.name,
            description=self.server.config.description,
            parameters=parameters_schema,
        )
