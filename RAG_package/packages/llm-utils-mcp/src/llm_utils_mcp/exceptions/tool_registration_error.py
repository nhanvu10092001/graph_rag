"""Tool registration exception class."""

from typing import Optional, Dict, Any
from .mcp_error import MCPError


class ToolRegistrationError(MCPError):
    """Raised when tool registration fails."""

    def __init__(
        self, tool_name: str, reason: str, context: Optional[Dict[str, Any]] = None
    ):
        message = f"Failed to register tool '{tool_name}': {reason}"
        super().__init__(message, context)
        self.tool_name = tool_name
        self.reason = reason