"""Dependency exception class."""

from typing import Optional, Dict, Any
from .mcp_error import MCPError


class DependencyError(MCPError):
    """Raised when required dependencies are missing."""

    def __init__(
        self,
        dependency: str,
        feature: str,
        install_command: str = "",
        context: Optional[Dict[str, Any]] = None,
    ):
        message = f"Missing dependency '{dependency}' required for {feature}"
        if install_command:
            message += f"\nInstall with: {install_command}"
        super().__init__(message, context)
        self.dependency = dependency
        self.feature = feature
        self.install_command = install_command