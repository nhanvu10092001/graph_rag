from typing import Any
from src.agents.graph_explorer.tools.tool import Tool


class AddToAnswer(Tool):
    def __init__(self, graph):
        super().__init__(graph=graph)

    @classmethod
    def name(cls) -> str:
        return "add_to_answer"

    @classmethod
    def description(cls) -> str:
        return "Add nodes to the answer."

    def __call__(self, args: dict) -> str:
        agent, answer_nodes = args["agent"], args["answer_nodes"]

        if not answer_nodes:
            return "No nodes were provided to add to the answer."

        valid_nodes = []
        error_indices = []
        reasoning_map = {}

        for answer_entry in answer_nodes:
            index = answer_entry.get("node_index")
            reasoning = answer_entry.get("reasoning", "No reasoning provided")

            try:
                node = self.graph.get_node_by_index(index)
                valid_nodes.append(node)
                reasoning_map[index] = reasoning
            except ValueError:
                error_indices.append(index)

        for new_valid_node in valid_nodes:
            if new_valid_node.index not in [node.index for node in agent.selected_nodes]:
                agent.selected_nodes.append(new_valid_node)
                agent.selected_nodes_reasoning.append(reasoning_map[new_valid_node.index])

        if valid_nodes:
            res = "Added the following nodes to the answer:\n"
            for node in valid_nodes:
                reasoning = reasoning_map[node.index]
                res += (
                    f"- [{node.index}] {node.name} (type: {node.type})\n  Reasoning: {reasoning}\n"
                )
        else:
            res = "No valid nodes were added to the answer."

        if error_indices:
            res += f"\nThe following indices were not found in the graph: {error_indices}"

        return res

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
                        "answer_nodes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "node_index": {
                                        "type": "integer",
                                        "description": "The index of the node to add as an answer",
                                    },
                                    "reasoning": {
                                        "type": "string",
                                        "description": "Explanation of why this node is relevant to answering the question",
                                    },
                                },
                                "required": ["node_index", "reasoning"],
                            },
                            "description": "List of answer nodes with their indices and reasoning. Pass EVERY node that could be an answer to this question.",
                        },
                    },
                    "required": ["answer_nodes"],
                },
            },
        }
