"""Configuration exception class."""

from typing import Optional, Dict, Any
from .mcp_error import MCPError


class ConfigurationError(MCPError):
    """Raised when configuration is invalid."""

    def __init__(
        self,
        config_key: str,
        issue: str,
        expected: str = "",
        context: Optional[Dict[str, Any]] = None,
    ):
        message = f"Configuration error for '{config_key}': {issue}"
        if expected:
            message += f"\nExpected: {expected}"
        super().__init__(message, context)
        self.config_key = config_key
        self.issue = issue
        self.expected = expected