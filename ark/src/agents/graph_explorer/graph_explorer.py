import re
import json
from typing import Any, Generator

from litellm import completion

from src.core.graph import Graph, Node
from src.core.logger import logger
from src.agents.graph_explorer.tools.search_in_graph import SearchInGraphTool
from src.agents.graph_explorer.tools.add_to_answers import AddToAnswer
from src.agents.graph_explorer.tools.search_in_neighborhood import SearchInNeighborhoodTool
from src.agents.graph_explorer.tools.finish import FinishTool
import time

class GraphExplorerAgent:

    def __init__(
        self,
        graph: Graph,
        model,
        system_prompt: str,
    ):

        self.tools = [
            SearchInGraphTool(graph),
            SearchInNeighborhoodTool(graph),
            AddToAnswer(graph),
            FinishTool(graph),
        ]

        self.model = model
        self.graph = graph

        self.step = 0
        self.step_times = []
        self.selected_nodes = []
        self.selected_nodes_reasoning = []
        self.message_history = [{"role": "system", "content": system_prompt}]

    def select_tools(self) -> Any:
        model_response = self.model.forward(
            self.message_history,
            [tool.schema() for tool in self.tools],
            tool_choice="required",
            enable_thinking=False,
        )

        json_response = {"role": "assistant"}
        if hasattr(model_response, "reasoning_content"):
            json_response["reasoning_content"] = model_response.reasoning_content
        if hasattr(model_response, "content"):
            json_response["content"] = model_response.content
        if hasattr(model_response, "tool_calls"):
            json_response["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in model_response.tool_calls
            ]
        self.message_history.append(json_response)

        if hasattr(model_response, "reasoning_content"):
            logger.debug(f"Reasoning content: {model_response.reasoning_content}")
        logger.debug(f"Content: {model_response.content}")

        for tool_call in model_response.tool_calls:
            logger.debug(f"Called function: {tool_call.function.name}")
            logger.debug(f"With arguments: {tool_call.function.arguments}")

        logger.debug("=" * 48)

        return model_response.tool_calls

    def call_tool(self, selected_tool: Any, tool_call_id) -> Any:
        args = json.loads(selected_tool.function.arguments) | {"agent": self}
        tool = next(
            filter(lambda tool: tool.name() == selected_tool.function.name, self.tools), None
        )

        if tool is None:
            return f"Tool {selected_tool.function.name} not found."

        tool_response = tool(args)

        # logger.info(f"Tool response: {tool_response}\n{'='*48}")

        self.message_history.append(
            {"role": "tool", "content": tool_response, "tool_call_id": tool_call_id}
        )

    def find_nodes(self, query: str, max_steps=30) -> list[Node]:
        self.message_history.append({"role": "user", "content": query})

        logger.debug("=" * 48)
        logger.debug(f"Starting graph exploration with query:\n{query}")
        logger.debug("=" * 48)

        tool_time = 0
        llm_time = 0

        start_time = time.time()
        while self.step < max_steps:
            self.step += 1
            llm_start_time = time.time()
            selected_tools = self.select_tools()
            llm_end_time = time.time()
            llm_time += llm_end_time - llm_start_time
            for selected_tool in selected_tools:
                tool_start_time = time.time()
                self.call_tool(selected_tool, selected_tool.id)
                tool_end_time = time.time()
                tool_time += tool_end_time - tool_start_time
                step_end_time = time.time() - start_time
                
                if selected_tool.function.name == "finish":
                    # logger.info(f"Tool time: {tool_time}")
                    # logger.info(f"LLM time: {llm_time}")
                    self.step_times.append(step_end_time)
                    return self.selected_nodes
            self.step_times.append(step_end_time)

        return self.selected_nodes
