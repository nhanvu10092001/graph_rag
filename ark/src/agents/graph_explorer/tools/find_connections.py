from typing import Any, Optional
from src.agents.graph_explorer.tools.tool import Tool
from src.core.graph import Node


class FindConnectionsTool(Tool):
    def __init__(self, graph):
        super().__init__(graph=graph)

    @classmethod
    def name(cls) -> str:
        return "find_connections"

    @classmethod
    def description(cls) -> str:
        return "Find all connections between two nodes in the knowledge graph. Returns paths up to length 2 with detailed relationship information."

    def _format_path_nicely(self, path) -> str:
        """Format a path in a more readable way with indentation and clear node information."""
        formatted_parts = []
        for i, element in enumerate(path.path_as_list):
            if isinstance(element, Node):
                # Format node with just name and type for readability
                node_info = f"[{element.index}] {element.name} ({element.type})"
                if i == 0:  # First node
                    formatted_parts.append(f"  {node_info}")
                elif i == len(path.path_as_list) - 1:  # Last node
                    formatted_parts.append(f"  └─ {node_info}")
                else:  # Middle node
                    formatted_parts.append(f"  ├─ {node_info}")
            else:  # Relationship/edge
                formatted_parts.append(f"     │ {element}")

        return "\n".join(formatted_parts)

    def __call__(self, args: dict) -> str:
        src_index, dst_index = args["src_node_index"], args["dst_node_index"]
        node_type = args.get("node_type", None)

        try:
            src_node = self.graph.get_node_by_index(src_index)
        except Exception as e:
            return f"Source node with index {src_index} not found in the graph. Please check the index and try again."
        try:
            dst_node = self.graph.get_node_by_index(dst_index)
        except Exception as e:
            return f"Destination node with index {dst_index} not found in the graph. Please check the index and try again."

        paths = self.graph.find_paths_of_length_2(src_node, dst_node)

        # Filter paths by node_type if provided
        if node_type is not None:
            filtered_paths = set()
            for path in paths:
                # Check if any node in the path (excluding src and dst) matches the node_type
                # For paths of length 2, we check the middle node
                path_nodes = [elem for elem in path.path_as_list if isinstance(elem, Node)]
                # Check intermediate nodes (not src or dst)
                if len(path_nodes) > 2:
                    # Path with mediator: check the middle node
                    if path_nodes[1].type == node_type:
                        filtered_paths.add(path)
                # Direct paths (length 1) don't have intermediate nodes, so we skip them
                # unless we want to include them - for now, we only filter by intermediate nodes
            paths = filtered_paths

        if not paths:
            tool_name = self.name()
            return f"{tool_name}({src_node}, {dst_node}) didn't return any results. Consider widening your search or changing the strategy.\n"

        formatted_paths = []
        paths_to_show = list(paths)[:30] if len(paths) > 30 else paths

        for i, path in enumerate(paths_to_show, 1):
            formatted_paths.append(f"Path {i}:")
            formatted_paths.append(self._format_path_nicely(path))
            formatted_paths.append("")  # Empty line between paths

        result_header = f"Found {len(paths)} path{'s' if len(paths) != 1 else ''} between [{src_node.index}] {src_node.name} and [{dst_node.index}] {dst_node.name}"
        if len(paths) > 30:
            result_header += f" (showing first 30)"
        result_header += ":\n"

        return (
            result_header
            + "\n".join(formatted_paths)
            + "\nConsider if any of these paths can help you answer the question."
        )

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
                        "src_node_index": {
                            "type": "integer",
                            "description": "The index of the source node to find connections from",
                        },
                        "dst_node_index": {
                            "type": "integer",
                            "description": "The index of the destination node to find connections to",
                        },
                        "node_type": {
                            "type": "string",
                            "description": "Optional filter to only return paths where intermediate nodes match this node type",
                        },
                    },
                    "required": ["src_node_index", "dst_node_index"],
                },
            },
        }

