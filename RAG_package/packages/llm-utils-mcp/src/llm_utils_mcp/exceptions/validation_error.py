"""Validation exception class."""

from typing import Optional, Dict, Any
from .mcp_error import MCPError


class ValidationError(MCPError):
    """Raised when input validation fails."""

    def __init__(
        self,
        field: str,
        value: Any,
        requirement: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        message = f"Invalid value for '{field}': {value}. {requirement}"
        super().__init__(message, context)
        self.field = field
        self.value = value
        self.requirement = requirement