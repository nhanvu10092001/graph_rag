"""Core plugin interface supporting both synchronous and asynchronous execution."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class TaskPlugin(ABC):
    """Abstract base class for all task plugins with both sync and async support.

    Plugins must implement both run() and arun() methods separately to provide
    proper synchronous and asynchronous execution paths.
    """

    @abstractmethod
    def name(self) -> str:
        """Return the name of the plugin."""
        pass

    @abstractmethod
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the plugin task synchronously.

        This method should implement the synchronous execution path using
        synchronous LangChain methods and operations.

        Args:
            context: Execution context containing task parameters

        Returns:
            dict: Task execution result
        """
        pass

    @abstractmethod
    async def arun(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the plugin task asynchronously.

        This method should implement the asynchronous execution path using
        asynchronous LangChain methods and operations.

        Args:
            context: Execution context containing task parameters

        Returns:
            dict: Task execution result
        """
        pass
