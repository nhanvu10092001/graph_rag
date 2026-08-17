"""Tool registry for managing and discovering MCP tools with clear error reporting."""

import importlib
from typing import Dict, List, Type, Any, Optional
from pathlib import Path
import inspect

from llm_utils_core import load_plugins

from .tool_interface import ToolInterface, ToolMetadata
from .exceptions import ToolRegistrationError, ValidationError, ConfigurationError
from .logger import logger


class ToolRegistry:
    """Registry for managing MCP tools with comprehensive error handling."""

    def __init__(self):
        """Initialize tool registry."""
        self._tools: Dict[str, Type[ToolInterface]] = {}
        self._instances: Dict[str, ToolInterface] = {}
        self._metadata_cache: Dict[str, ToolMetadata] = {}

    def register_tool_class(self, tool_class: Type[ToolInterface]) -> None:
        """Register a tool class.

        Args:
            tool_class: Tool class to register

        Raises:
            ToolRegistrationError: If registration fails
        """
        try:
            # Validate tool class
            if not issubclass(tool_class, ToolInterface):
                raise ToolRegistrationError(
                    tool_name=getattr(tool_class, "__name__", "<unknown>"),
                    reason="Tool class must inherit from ToolInterface",
                )

            # Create temporary instance to get metadata
            try:
                temp_instance = tool_class()
                temp_instance.validate_implementation()
                metadata = temp_instance.get_metadata()
            except Exception as e:
                raise ToolRegistrationError(
                    tool_name=getattr(tool_class, "__name__", "<unknown>"),
                    reason=f"Failed to create tool instance: {e}",
                ) from e

            # Check for name conflicts
            if metadata.name in self._tools:
                existing_class = self._tools[metadata.name]
                raise ToolRegistrationError(
                    tool_name=metadata.name,
                    reason=f"Tool name '{metadata.name}' already registered by {existing_class.__name__}",
                )

            # Register the tool class
            self._tools[metadata.name] = tool_class
            self._metadata_cache[metadata.name] = metadata

            logger.info(
                f"Registered tool class '{tool_class.__name__}' as '{metadata.name}'"
            )

        except ToolRegistrationError:
            raise
        except Exception as e:
            raise ToolRegistrationError(
                tool_name=getattr(tool_class, "__name__", "<unknown>"),
                reason=f"Unexpected error during class registration: {e}",
            ) from e

    def register_tool_instance(self, tool_instance: ToolInterface) -> None:
        """Register a tool instance.

        Args:
            tool_instance: Tool instance to register

        Raises:
            ToolRegistrationError: If registration fails
        """
        try:
            # Validate tool instance
            tool_instance.validate_implementation()
            metadata = tool_instance.get_metadata()

            # Check for name conflicts
            if metadata.name in self._instances:
                raise ToolRegistrationError(
                    tool_name=metadata.name,
                    reason=f"Tool instance '{metadata.name}' is already registered",
                )

            # Register the tool instance
            self._instances[metadata.name] = tool_instance
            self._metadata_cache[metadata.name] = metadata

            logger.info(f"Registered tool instance '{metadata.name}'")

        except ToolRegistrationError:
            raise
        except Exception as e:
            raise ToolRegistrationError(
                tool_name=getattr(tool_instance, "name", "<unknown>"),
                reason=f"Unexpected error during instance registration: {e}",
            ) from e

    def create_tool_instance(self, tool_name: str, **kwargs) -> ToolInterface:
        """Create a tool instance from registered class.

        Args:
            tool_name: Name of tool to create
            **kwargs: Constructor arguments

        Returns:
            ToolInterface: Tool instance

        Raises:
            ToolRegistrationError: If creation fails
        """
        if tool_name not in self._tools:
            available_tools = list(self._tools.keys())
            raise ToolRegistrationError(
                tool_name=tool_name,
                reason=f"Tool class '{tool_name}' not found. Available tools: {available_tools}",
            )

        try:
            tool_class = self._tools[tool_name]
            instance = tool_class(**kwargs)
            instance.validate_implementation()
            return instance

        except Exception as e:
            raise ToolRegistrationError(
                tool_name=tool_name, reason=f"Failed to create tool instance: {e}"
            ) from e

    def get_tool_instance(self, tool_name: str) -> ToolInterface:
        """Get a registered tool instance.

        Args:
            tool_name: Name of tool to get

        Returns:
            ToolInterface: Tool instance

        Raises:
            ValidationError: If tool not found
        """
        if tool_name not in self._instances:
            available_tools = list(self._instances.keys())
            raise ValidationError(
                field="tool_name",
                value=tool_name,
                requirement=f"Tool instance '{tool_name}' not found. Available instances: {available_tools}",
            )

        return self._instances[tool_name]

    def get_tool_metadata(
        self, tool_name: Optional[str] = None
    ) -> Dict[str, ToolMetadata]:
        """Get metadata for tools.

        Args:
            tool_name: Specific tool name, or None for all tools

        Returns:
            Dict[str, ToolMetadata]: Tool metadata
        """
        if tool_name:
            if tool_name not in self._metadata_cache:
                available_tools = list(self._metadata_cache.keys())
                raise ValidationError(
                    field="tool_name",
                    value=tool_name,
                    requirement=f"Tool '{tool_name}' not found. Available tools: {available_tools}",
                )
            return {tool_name: self._metadata_cache[tool_name]}

        return self._metadata_cache.copy()

    def discover_tools_from_plugins(self) -> Dict[str, List[str]]:
        """Discover tools from registered plugins.

        Returns:
            Dict[str, List[str]]: Map of plugin names to discovered tool names

        Raises:
            ConfigurationError: If plugin discovery fails
        """
        discovered = {}
        errors = []

        try:
            plugins = load_plugins()
        except Exception as e:
            raise ConfigurationError(
                config_key="plugin_discovery",
                issue=f"Failed to load plugins: {e}",
                expected="Valid plugin entry points in setup.py or pyproject.toml",
            ) from e

        for plugin_name, plugin_class in plugins.items():
            try:
                # Check if plugin has MCP tools
                if hasattr(plugin_class, "get_mcp_tools"):
                    tools = plugin_class.get_mcp_tools()
                    tool_names = []

                    for tool in tools:
                        try:
                            self.register_tool_instance(tool)
                            tool_names.append(tool.get_metadata().name)
                        except ToolRegistrationError as e:
                            errors.append(f"Plugin '{plugin_name}' tool error: {e}")
                            logger.warning(
                                f"Failed to register tool from plugin '{plugin_name}': {e}"
                            )

                    if tool_names:
                        discovered[plugin_name] = tool_names

            except Exception as e:
                errors.append(f"Plugin '{plugin_name}' discovery error: {e}")
                logger.warning(
                    f"Failed to discover tools from plugin '{plugin_name}': {e}"
                )

        if errors and not discovered:
            raise ConfigurationError(
                config_key="plugin_tool_discovery",
                issue="No tools discovered from any plugins",
                expected=f"Plugins with get_mcp_tools() method. Errors: {'; '.join(errors)}",
            )

        return discovered

    def discover_tools_from_directory(
        self, directory: Path, pattern: str = "*.py"
    ) -> List[str]:
        """Discover tools from Python files in a directory.

        Args:
            directory: Directory to scan
            pattern: File pattern to match

        Returns:
            List[str]: Names of discovered tools

        Raises:
            ConfigurationError: If directory discovery fails
        """
        if not directory.exists():
            raise ConfigurationError(
                config_key="tool_directory",
                issue=f"Directory does not exist: {directory}",
                expected="Valid directory path",
            )

        if not directory.is_dir():
            raise ConfigurationError(
                config_key="tool_directory",
                issue=f"Path is not a directory: {directory}",
                expected="Valid directory path",
            )

        discovered_tools = []
        errors = []

        for file_path in directory.glob(pattern):
            if file_path.name.startswith("__"):
                continue  # Skip __init__.py and __pycache__

            try:
                # Import module
                module_name = file_path.stem
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find tool classes
                for name in dir(module):
                    obj = getattr(module, name)
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, ToolInterface)
                        and obj != ToolInterface
                    ):
                        try:
                            self.register_tool_class(obj)
                            discovered_tools.append(name)
                        except ToolRegistrationError as e:
                            errors.append(
                                f"File '{file_path.name}' class '{name}': {e}"
                            )

            except Exception as e:
                errors.append(f"File '{file_path.name}': Failed to import - {e}")

        if errors:
            logger.warning(f"Tool discovery errors: {'; '.join(errors)}")

        return discovered_tools

    def list_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """List all available tools with their status.

        Returns:
            Dict[str, Dict[str, Any]]: Tool information
        """
        tools_info = {}

        for tool_name, metadata in self._metadata_cache.items():
            tools_info[tool_name] = {
                "description": metadata.description,
                "parameter_count": len(metadata.parameters),
                "has_class": tool_name in self._tools,
                "has_instance": tool_name in self._instances,
                "parameters": [p.name for p in metadata.parameters],
            }

        return tools_info

    def clear_registry(self) -> None:
        """Clear all registered tools."""
        count = len(self._tools) + len(self._instances)
        self._tools.clear()
        self._instances.clear()
        self._metadata_cache.clear()
        logger.info(f"Cleared {count} tools from registry")
