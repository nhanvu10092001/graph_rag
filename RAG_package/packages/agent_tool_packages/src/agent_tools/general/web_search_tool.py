"""WebSearchTool — Search the web for current information.

Wraps the same Tavily integration used in mcp_sever/mcp_sever.py::tavily_search_tool.
"""

import os
from typing import Any

from pydantic import BaseModel, Field

from ..abs_interface.utility_tool import AbstractUtilityTool
from ..core.registry import ServiceToolRegistry


class WebSearchInput(BaseModel):
    query: str = Field(description="Search query to look up on the web")
    k: int = Field(default=3, description="Number of search results to return (1-10)")


@ServiceToolRegistry.register
class WebSearchTool(AbstractUtilityTool):
    """Search the web for current information using Tavily."""

    @property
    def name(self) -> str:
        return "search_web"

    @property
    def description(self) -> str:
        return (
            "Search the web for current and up-to-date information. "
            "Use this when the user asks about recent events, news, facts, "
            "or anything you are unsure about and need to verify. "
            "Input: search query and number of results. "
            "Returns: list of search results with title, content, and URL."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        return WebSearchInput

    async def call_api(self, query: str, k: int = 3, **_) -> Any:
        api_key = os.getenv("TAVILY_API_KEY", "")
        if not api_key:
            return {"error": "TAVILY_API_KEY not configured"}

        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=api_key)
            response = client.search(query=query, max_results=k)
            results = []
            for item in response.get("results", []):
                results.append(
                    {
                        "title": item.get("title", ""),
                        "content": item.get("content", ""),
                        "url": item.get("url", ""),
                    }
                )
            return {"query": query, "results": results}
        except Exception as e:
            return {"error": f"Web search failed: {str(e)}"}
