"""Base MCP exception class."""

from typing import Optional, Dict, Any


class MCPError(Exception):
    """Base exception for all MCP-related errors."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.context = context or {}
        super().__init__(message)