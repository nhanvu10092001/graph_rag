"""Configuration for MCP Server."""

from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Configuration for MCP Server."""

    name: str
    description: str = ""
    version: str = "1.0.0"
    host: str = "localhost"
    port: int = 8000
    max_tools: int = 100
    timeout: float = 30.0
    debug: bool = False