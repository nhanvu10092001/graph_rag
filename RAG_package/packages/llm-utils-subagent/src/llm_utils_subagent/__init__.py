"""LLM Utils Subagent — Pluggable subagent factory for multi-agent systems.

3 factory methods:
- ``create_simple_subagent()`` — direct LLM call
- ``create_react_subagent()``  — ReAct agent with tools
- ``create_graph_subagent()``  — custom StateGraph per subagent

Quick start::

    from llm_utils_subagent import create_simple_subagent, create_react_subagent

    planner = create_simple_subagent(
        name="planner", description="Plan tasks",
        system_prompt="You are a planner.", llm=my_llm,
    )

    executor = create_react_subagent(
        name="executor", description="Execute with tools",
        system_prompt="You are an executor.", llm=my_llm,
        tools=[rag_tool, search_tool],
    )

Graph subagent::

    from llm_utils_subagent import create_graph_subagent
    from llm_utils_subagent.graph_helpers import SubagentGraphBuilder, make_llm_node

    def build_graph(query, context, llm, tools, messages, **kw):
        builder = SubagentGraphBuilder()
        builder.add_node("step1", make_llm_node(llm, "Step 1 prompt"))
        builder.add_node("step2", make_llm_node(llm, "Step 2 prompt"))
        builder.set_entry("step1").add_edge("step1", "step2").set_finish("step2")
        return builder.compile()

    agent = create_graph_subagent(
        name="analyst", description="Multi-step analysis",
        graph_builder=build_graph, llm=my_llm,
    )
"""

from .factory import create_simple_subagent, create_react_subagent, create_graph_subagent
from .registry import SubagentRegistry
from .tool_wrapper import SubagentTool
from . import prompts

__version__ = "0.1.0"

__all__ = [
    "create_simple_subagent",
    "create_react_subagent",
    "create_graph_subagent",
    "SubagentRegistry",
    "SubagentTool",
    "prompts",
]
