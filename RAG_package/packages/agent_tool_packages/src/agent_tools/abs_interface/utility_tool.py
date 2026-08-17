"""AbstractUtilityTool — For tools that call external APIs directly.

No MQ, no DB. Direct API calls with immediate response.
Examples: weather, time, web search, knowledge base search.
"""

from abc import abstractmethod
from typing import Any

from .base_tool import AbstractBaseTool


class AbstractUtilityTool(AbstractBaseTool):
    """Base class for tools that call external APIs directly.

    Subclass contract:
        MUST override:
            - name, description, input_schema (from AbstractBaseTool)
            - call_api(**kwargs) -> Any

        DO NOT override:
            - execute(**kwargs)
    """

    @abstractmethod
    async def call_api(self, **kwargs) -> Any:
        """Call the external API and return the result.

        Args:
            **kwargs: Tool input parameters.

        Returns:
            Structured result for the agent.
        """
        raise NotImplementedError

    async def execute(self, **kwargs) -> Any:
        """Execute the API call. Delegates to call_api()."""
        return await self.call_api(**kwargs)
