from typing import Any

from src.agents.graph_explorer.tools.tool import Tool


class FinishTool(Tool):
    def __init__(self, graph):
        super().__init__(graph=graph)

    @classmethod
    def name(cls) -> str:
        return "finish"

    @classmethod
    def description(cls) -> str:
        return "Signal that exploration is complete. Use this when you have found all relevant nodes and added them to answers, or when you've exhausted useful exploration paths."

    def __call__(self, args: dict) -> str:
        comment: str = args.get("comment", "").strip()
        if comment:
            return f"Exploration finished. Agent's comment: {comment}"
        else:
            return "Exploration finished."

    @classmethod
    def schema(cls) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": cls.name(),
                "description": cls.description(),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "comment": {
                            "type": "string",
                            "description": "Optional comment explaining why the exploration is finished (e.g., 'Found all relevant drugs', 'No more useful paths to explore')",
                        },
                    },
                    "required": [],
                },
            },
        }
