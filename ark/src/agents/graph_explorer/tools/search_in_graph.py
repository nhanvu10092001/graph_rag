from typing import Any, Optional
from src.agents.graph_explorer.tools.tool import Tool


class SearchInGraphTool(Tool):
    MAX_RESULTS_SHOWN: int = 30

    def __init__(self, graph):
        super().__init__(graph=graph)
        self.MAX_RESULTS_SHOWN = 30

    @classmethod
    def name(cls) -> str:
        return "search_in_graph"

    @classmethod
    def description(cls) -> str:
        return "Search across the entire knowledge graph to find entities matching the given query. Returns nodes with their indices, names, types, and relevance scores."

    def _find_query_context(self, node, query: str) -> Optional[str]:
        """Find and highlight query matches in the node's summary."""
        if not node.summary:
            return None

        query_words = [word for word in query.split() if len(word) > 2]
        if not query_words:
            return None

        matched_contexts = []
        summary = node.summary

        for word in query_words:
            word_index = summary.lower().find(word.lower())
            if word_index != -1:
                context_start = max(0, word_index - 30)
                context_end = min(len(summary), word_index + len(word) + 30)
                context = summary[context_start:context_end]
                # Highlight the matched word
                highlighted_context = context.replace(
                    summary[word_index : word_index + len(word)],
                    f"**{summary[word_index:word_index + len(word)]}**",
                )
                matched_contexts.append(highlighted_context)
                break  # Only show first match per node to keep output clean

        return matched_contexts[0] if matched_contexts else None

    def __call__(self, args: dict) -> str:
        query: str = args["query"]
        size: int = args.get("size", 5)
        
        k = min(size, self.MAX_RESULTS_SHOWN)
        nodes, scores = self.graph.search_nodes(query, k=k)

        if not nodes:
            return f"No nodes found for query: '{query}'"

        result_lines = [
            f"Found {len(nodes)} node{'s' if len(nodes) != 1 else ''} for query: '{query}':\n"
        ]

        for i, (node, score) in enumerate(zip(nodes, scores)):
            node_info = f"[{node.index}] {node.name} ({node.type}) - Score: {score:.3f}"

            # Try to find context matches in the summary
            context = self._find_query_context(node, query)
            if context:
                result_lines.append(f"{node_info}\n    Context: {context}\n")
            else:
                result_lines.append(f"{node_info}\n")

        return "\n".join(result_lines)

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
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant entities across the entire knowledge graph. Use key terms from the question.",
                        },
                        "size": {
                            "type": "integer",
                            "description": "The number of results to return (default: 20, max: 20)",
                        },
                    },
                    "required": ["query"],
                },
            },
        }