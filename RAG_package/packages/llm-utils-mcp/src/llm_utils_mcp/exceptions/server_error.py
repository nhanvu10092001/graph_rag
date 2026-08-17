"""Server exception class."""

from typing import Optional, Dict, Any
from .mcp_error import MCPError


class ServerError(MCPError):
    """Raised when server operations fail."""

    def __init__(
        self, operation: str, reason: str, context: Optional[Dict[str, Any]] = None
    ):
        message = f"Server {operation} failed: {reason}"
        super().__init__(message, context)
        self.operation = operation
        self.reason = reason