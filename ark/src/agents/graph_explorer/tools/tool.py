from abc import ABC, abstractmethod
from typing import Any

from src.core.graph import Graph, Node


class Tool(ABC):
    """Abstract base class for all tools"""

    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    @classmethod
    def name(cls) -> str:
        return NotImplemented

    @classmethod
    def description(cls) -> str:
        return NotImplemented

    @abstractmethod
    def __call__(self, args: dict[str, Any], current_node: Node, graph: Graph) -> dict[str, Any]:
        """__call__ the tool with given inputs and context"""
        pass

    @classmethod
    @abstractmethod
    def schema(cls) -> dict[str, Any]:
        """Return the full tool definition for LiteLLM"""
        pass
