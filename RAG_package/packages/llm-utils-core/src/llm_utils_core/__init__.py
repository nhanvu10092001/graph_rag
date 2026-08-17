"""LLM Utils Core - Plugin architecture and core utilities."""

from .abstracts import TaskPlugin
from .loaders import load_plugins

__version__ = "0.1.0"

__all__ = [
    "TaskPlugin",
    "load_plugins",
]
