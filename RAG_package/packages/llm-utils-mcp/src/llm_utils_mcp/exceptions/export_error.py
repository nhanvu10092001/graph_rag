"""Export exception class."""

from typing import Optional, Dict, Any
from .mcp_error import MCPError


class ExportError(MCPError):
    """Raised when export operations fail."""

    def __init__(
        self,
        export_type: str,
        reason: str,
        suggestion: str = "",
        context: Optional[Dict[str, Any]] = None,
    ):
        message = f"Failed to export as {export_type}: {reason}"
        if suggestion:
            message += f"\nSuggestion: {suggestion}"
        super().__init__(message, context)
        self.export_type = export_type
        self.reason = reason
        self.suggestion = suggestion