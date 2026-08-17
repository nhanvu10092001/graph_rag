import logging
from typing import Annotated, Any, Dict, List, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)


def merge_selected_nodes(a: Dict[str, Dict[str, Any]], b: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    merged = dict(a)
    merged.update(b)
    return merged


class ArkAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    selected_nodes: Annotated[Dict[str, Dict[str, Any]], merge_selected_nodes]
    current_step: int
    max_steps: int
    is_finished: bool


def create_ark_agent_graph(llm: BaseChatModel, tools: List[BaseTool]) -> StateGraph:
    """
    Constructs a LangGraph state machine for a single ARK trajectory agent.
    Uses tool_choice="required" to force the LLM to call a tool every turn.
    The agent terminates when it calls 'finish' or reaches max_steps.
    """

    llm_with_tools = llm.bind_tools(tools, tool_choice="required")

    tool_map = {t.name: t for t in tools}

    def call_model(state: ArkAgentState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        response = llm_with_tools.invoke(messages)
        return {
            "messages": [response],
            "current_step": state.get("current_step", 0) + 1,
        }

    def execute_tools(state: ArkAgentState) -> Dict[str, Any]:
        messages = state["messages"]
        last_message = messages[-1]

        new_messages: List[ToolMessage] = []
        new_selected: Dict[str, Dict[str, Any]] = {}
        is_finished = False
        current_step = state.get("current_step", 0)

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            if tool_name == "finish":
                comment = tool_args.get("comment", "")
                content = f"Exploration complete. {comment}" if comment else "Exploration complete."
                new_messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))
                is_finished = True
                break

            elif tool_name == "add_to_answer":
                answer_nodes = tool_args.get("answer_nodes", [])
                added_ids = []
                for entry in answer_nodes:
                    node_id = entry.get("node_id", "") if isinstance(entry, dict) else ""
                    reasoning = entry.get("reasoning", "") if isinstance(entry, dict) else ""
                    if node_id:
                        new_selected[node_id] = {
                            "node_id": node_id,
                            "reasoning": reasoning,
                            "step": current_step,
                        }
                        added_ids.append(node_id)

                if added_ids:
                    content = f"Added {len(added_ids)} node(s) to answer: {added_ids}"
                else:
                    content = "No valid node IDs found in the provided answer_nodes."
                new_messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))

            else:
                tool = tool_map.get(tool_name)
                if tool is None:
                    new_messages.append(ToolMessage(
                        content=f"Error: Unknown tool '{tool_name}'",
                        tool_call_id=tool_call_id,
                    ))
                    continue
                try:
                    result = tool.invoke(tool_args)
                except Exception as e:
                    logger.error(f"Tool '{tool_name}' execution error: {e}")
                    result = f"Error executing {tool_name}: {e}"
                new_messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call_id,
                ))

        update: Dict[str, Any] = {"messages": new_messages}
        if new_selected:
            update["selected_nodes"] = new_selected
        if is_finished:
            update["is_finished"] = True
        return update

    def route_after_agent(state: ArkAgentState) -> str:
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        if not last_message or not getattr(last_message, "tool_calls", None):
            return "end"
        return "execute_tools"

    def route_after_tools(state: ArkAgentState) -> str:
        if state.get("is_finished", False):
            return "end"
        if state.get("current_step", 0) >= state.get("max_steps", 30):
            return "end"
        return "agent"

    workflow = StateGraph(ArkAgentState)

    workflow.add_node("agent", call_model)
    workflow.add_node("execute_tools", execute_tools)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {"execute_tools": "execute_tools", "end": END},
    )
    workflow.add_conditional_edges(
        "execute_tools",
        route_after_tools,
        {"agent": "agent", "end": END},
    )

    return workflow.compile()
