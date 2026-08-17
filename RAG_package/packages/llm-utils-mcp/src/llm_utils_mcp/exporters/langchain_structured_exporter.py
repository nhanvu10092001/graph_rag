"""LangChain structured tool exporter with clear error reporting."""

import logging
from typing import Dict, Any, List, Type, TYPE_CHECKING
from pydantic import BaseModel, Field, create_model

from ..exceptions import ExportError, DependencyError
from ..tool_interface import ToolInterface, ToolParameter

if TYPE_CHECKING:
    from ..server import MCPServer
    import langchain_core.tools

logger = logging.getLogger(__name__)


class LangChainStructuredToolExporter:
    """Exports MCP server tools to LangChain structured tools."""

    def __init__(self, server: "MCPServer"):
        """Initialize LangChain structured tool exporter.

        Args:
            server: MCP server to export from
        """
        self.server = server

    def export(self) -> List["langchain_core.tools.StructuredTool"]:
        """Export tools as LangChain structured tools.

        Returns:
            List[StructuredTool]: List of LangChain structured tools

        Raises:
            DependencyError: If LangChain not installed
            ExportError: If export fails
        """
        try:
            from langchain_core.tools import StructuredTool
        except ImportError:
            raise DependencyError(
                dependency="langchain-core",
                feature="LangChain structured tools export",
                install_command="pip install langchain-core",
            )

        try:
            structured_tools = []

            for tool_name, tool in self.server.tools.items():
                try:
                    structured_tool = self._create_structured_tool(
                        tool_name, tool, StructuredTool
                    )
                    structured_tools.append(structured_tool)
                except Exception as e:
                    logger.warning(
                        f"Failed to export tool '{tool_name}' to LangChain: {e}"
                    )
                    raise ExportError(
                        export_type="LangChain structured tool",
                        reason=f"Failed to export tool '{tool_name}': {e}",
                        suggestion="Check that tool parameters are compatible with Pydantic models",
                    ) from e

            logger.info(
                f"Successfully exported {len(structured_tools)} tools to LangChain structured tools"
            )
            return structured_tools

        except DependencyError:
            raise
        except ExportError:
            raise
        except Exception as e:
            raise ExportError(
                export_type="LangChain structured tools",
                reason=f"Unexpected error during export: {e}",
                suggestion="Check server configuration and LangChain compatibility",
            ) from e

    def _create_structured_tool(
        self, tool_name: str, tool: ToolInterface, StructuredTool: Type
    ) -> "langchain_core.tools.StructuredTool":
        """Create a LangChain StructuredTool from MCP tool.

        Args:
            tool_name: Name of tool
            tool: Tool implementation
            StructuredTool: LangChain StructuredTool class

        Returns:
            StructuredTool: LangChain structured tool

        Raises:
            ExportError: If tool creation fails
        """
        try:
            metadata = tool.get_metadata()

            # Create Pydantic model for parameters
            pydantic_model = self._create_pydantic_model(tool_name, metadata.parameters)

            # Create async wrapper function
            async def tool_func(**kwargs) -> Dict[str, Any]:
                try:
                    # Execute tool via server for validation and error handling
                    result = await self.server.execute_tool(tool_name, kwargs)
                    return result
                except Exception as e:
                    # Convert to LangChain-compatible error
                    raise ValueError(f"Tool '{tool_name}' execution failed: {e}") from e

            # Create structured tool
            structured_tool = StructuredTool(
                name=metadata.name,
                description=metadata.description,
                args_schema=pydantic_model,
                func=tool_func,
                coroutine=tool_func,  # For async support
            )

            logger.debug(f"Created LangChain structured tool for '{tool_name}'")
            return structured_tool

        except Exception as e:
            raise ExportError(
                export_type="LangChain structured tool",
                reason=f"Failed to create structured tool for '{tool_name}': {e}",
                suggestion="Ensure tool has valid parameter schema and is compatible with Pydantic",
            ) from e

    def _create_pydantic_model(
        self, tool_name: str, parameters: List[ToolParameter]
    ) -> Type[BaseModel]:
        """Create Pydantic model from tool parameters.

        Args:
            tool_name: Tool name for model naming
            parameters: Tool parameters

        Returns:
            Type[BaseModel]: Pydantic model class

        Raises:
            ExportError: If model creation fails
        """
        try:
            # Build fields dictionary for create_model
            fields = {}

            for param in parameters:
                # Convert parameter to Pydantic field
                field_info = self._parameter_to_pydantic_field(param)
                fields[param.name] = field_info

            # Create dynamic Pydantic model
            model_name = f"{tool_name.replace('-', '_').replace(' ', '_').title()}Args"
            pydantic_model = create_model(model_name, **fields)

            return pydantic_model

        except Exception as e:
            raise ExportError(
                export_type="Pydantic parameter model",
                reason=f"Failed to create Pydantic model for tool '{tool_name}': {e}",
                suggestion="Check that all parameter types are supported by Pydantic",
            ) from e

    def _parameter_to_pydantic_field(self, param: ToolParameter) -> tuple:
        """Convert ToolParameter to Pydantic field definition.

        Args:
            param: Tool parameter

        Returns:
            tuple: (type, Field) tuple for Pydantic model
        """
        # Handle optional parameters
        param_type = param.type
        field_kwargs = {"description": param.description}

        # Set default value if provided
        if param.default is not None:
            field_kwargs["default"] = param.default
        elif not param.required:
            field_kwargs["default"] = None
            # Make type optional
            param_type = (
                param_type
                if param_type.__name__ == "Optional"
                else type(None) | param_type
            )

        return (param_type, Field(**field_kwargs))

    def _python_type_to_pydantic_type(self, python_type: type) -> type:
        """Convert Python type to Pydantic-compatible type.

        Args:
            python_type: Python type

        Returns:
            type: Pydantic-compatible type
        """
        # Most Python types are directly compatible with Pydantic
        # Special handling can be added here if needed
        return python_type

    def get_export_info(self) -> Dict[str, Any]:
        """Get information about LangChain export.

        Returns:
            Dict[str, Any]: Export information
        """
        tools_info = {}

        for tool_name, tool in self.server.tools.items():
            metadata = tool.get_metadata()

            # Get parameter info
            params_info = []
            for param in metadata.parameters:
                params_info.append(
                    {
                        "name": param.name,
                        "type": param.type.__name__,
                        "description": param.description,
                        "required": param.required,
                        "default": param.default,
                    }
                )

            tools_info[tool_name] = {
                "name": metadata.name,
                "description": metadata.description,
                "parameters": params_info,
                "parameter_count": len(metadata.parameters),
                "required_parameters": [
                    p.name for p in metadata.parameters if p.required
                ],
            }

        return {
            "server_name": self.server.config.name,
            "server_description": self.server.config.description,
            "tool_count": len(self.server.tools),
            "tools": tools_info,
            "export_type": "LangChain StructuredTool",
            "dependencies": ["langchain-core", "pydantic"],
        }
