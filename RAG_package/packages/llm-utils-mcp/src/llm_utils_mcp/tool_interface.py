"""Tool interface for MCP server with comprehensive validation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
import inspect

from .exceptions import ValidationError, ToolRegistrationError


@dataclass
class ToolParameter:
    """Represents a tool parameter with validation rules."""

    name: str
    type: type
    description: str
    required: bool = True
    default: Any = None
    validator: Optional[Callable[[Any], bool]] = None


@dataclass
class ToolMetadata:
    """Metadata describing a tool."""

    name: str
    description: str
    parameters: List[ToolParameter]
    return_type: Optional[type] = None
    examples: Optional[List[Dict[str, Any]]] = None


class ToolInterface(ABC):
    """Abstract interface for creating MCP tools with validation."""

    @abstractmethod
    def get_metadata(self) -> ToolMetadata:
        """Return tool metadata including parameters and description.

        Returns:
            ToolMetadata: Complete tool metadata

        Raises:
            ValidationError: If metadata is invalid
        """
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters.

        Args:
            **kwargs: Tool parameters

        Returns:
            Dict[str, Any]: Tool execution result

        Raises:
            ValidationError: If parameters are invalid
            Exception: If tool execution fails
        """
        pass

    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize input parameters.

        Args:
            parameters: Raw input parameters

        Returns:
            Dict[str, Any]: Validated and sanitized parameters

        Raises:
            ValidationError: If validation fails with specific details
        """
        metadata = self.get_metadata()
        validated = {}

        # Check required parameters
        for param in metadata.parameters:
            if param.required and param.name not in parameters:
                raise ValidationError(
                    field=param.name,
                    value="<missing>",
                    requirement=f"This parameter is required for tool '{metadata.name}'",
                )

        # Validate each parameter
        for param in metadata.parameters:
            if param.name in parameters:
                value = parameters[param.name]
                validated_value = self._validate_parameter(param, value)
                validated[param.name] = validated_value
            elif not param.required and param.default is not None:
                validated[param.name] = param.default

        # Check for unexpected parameters
        unexpected = set(parameters.keys()) - {p.name for p in metadata.parameters}
        if unexpected:
            raise ValidationError(
                field="parameters",
                value=list(unexpected),
                requirement=f"Unexpected parameters for tool '{metadata.name}'. "
                f"Valid parameters are: {[p.name for p in metadata.parameters]}",
            )

        return validated

    def _validate_parameter(self, param: ToolParameter, value: Any) -> Any:
        """Validate a single parameter.

        Args:
            param: Parameter definition
            value: Parameter value to validate

        Returns:
            Any: Validated parameter value

        Raises:
            ValidationError: If validation fails
        """
        # Type validation
        if not isinstance(value, param.type):
            try:
                # Try to convert the type
                value = param.type(value)
            except (ValueError, TypeError):
                raise ValidationError(
                    field=param.name,
                    value=value,
                    requirement=f"Must be of type {param.type.__name__}",
                )

        # Custom validation
        if param.validator and not param.validator(value):
            raise ValidationError(
                field=param.name,
                value=value,
                requirement=f"Failed custom validation for parameter '{param.name}'",
            )

        return value

    def validate_implementation(self) -> None:
        """Validate that this tool implementation is correct.

        Raises:
            ToolRegistrationError: If implementation is invalid
        """
        try:
            metadata = self.get_metadata()

            # Validate metadata
            if not metadata.name:
                raise ToolRegistrationError(
                    tool_name="<unknown>", reason="Tool name cannot be empty"
                )

            if not metadata.description:
                raise ToolRegistrationError(
                    tool_name=metadata.name, reason="Tool description cannot be empty"
                )

            # Validate execute method signature
            sig = inspect.signature(self.execute)
            if not any(
                param.kind == param.VAR_KEYWORD for param in sig.parameters.values()
            ):
                raise ToolRegistrationError(
                    tool_name=metadata.name,
                    reason="execute() method must accept **kwargs",
                )

        except ValidationError as e:
            raise ToolRegistrationError(
                tool_name=getattr(metadata, "name", "<unknown>"),
                reason=f"Invalid metadata: {e}",
            ) from e
        except Exception as e:
            raise ToolRegistrationError(
                tool_name="<unknown>", reason=f"Implementation validation failed: {e}"
            ) from e


class BaseTool(ToolInterface):
    """Base implementation of ToolInterface with common functionality."""

    def __init__(self, name: str, description: str, parameters: List[ToolParameter]):
        """Initialize base tool.

        Args:
            name: Tool name
            description: Tool description
            parameters: Tool parameters
        """
        self._metadata = ToolMetadata(
            name=name, description=description, parameters=parameters
        )

    def get_metadata(self) -> ToolMetadata:
        """Return tool metadata."""
        return self._metadata

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool - must be implemented by subclasses."""
        pass
