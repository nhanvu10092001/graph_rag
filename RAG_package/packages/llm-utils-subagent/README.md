# llm-utils-subagent

Pluggable subagent factory for multi-agent systems.

## Features

- **`create_subagent()`** — create a subagent from a prompt, get back a LangChain `BaseTool`
- **`SubagentRegistry`** — register multiple subagents, export as tool list
- **Stateless** — each invocation creates fresh history, no state leaks between calls
- **Two modes** — `simple` (LLM call) and `react` (ReAct loop with tools)
- **Built-in prompts** — common roles (planner, executor, reviewer, etc.)

## Quick Start

```python
from llm_utils_subagent import create_subagent, SubagentRegistry, prompts
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", api_key="...")

# Single subagent
planner = create_subagent(
    name="planner",
    description="Break a task into steps",
    system_prompt=prompts.PLANNER_PROMPT,
    llm=llm,
)

# Registry pattern
registry = SubagentRegistry()
registry.create(name="planner", description="Plan tasks", system_prompt=prompts.PLANNER_PROMPT, llm=llm)
registry.create(name="executor", description="Execute plans", system_prompt=prompts.EXECUTOR_PROMPT, llm=llm, tools=[my_tool], mode="react")
registry.create(name="reviewer", description="Review results", system_prompt=prompts.REVIEWER_PROMPT, llm=llm)

# Export and use
from langgraph.prebuilt import create_react_agent
coordinator = create_react_agent(llm, tools=registry.as_tools())
```

## API

### `create_subagent(name, description, system_prompt, llm, **kwargs) -> BaseTool`

| Param | Type | Description |
|---|---|---|
| `name` | `str` | Tool name for the coordinator |
| `description` | `str` | When should coordinator call this? |
| `system_prompt` | `str` | Subagent persona/instructions |
| `llm` | `BaseChatModel` | Any LangChain chat model |
| `tools` | `list` | Tools for ReAct mode (optional) |
| `mode` | `str` | `"simple"` or `"react"` (auto-detected) |
| `output_parser` | `Callable` | Post-process output (optional) |

### `SubagentRegistry`

| Method | Description |
|---|---|
| `create(...)` | Create + register in one call |
| `register(tool)` | Register an existing tool |
| `get(name)` | Get a subagent by name |
| `as_tools()` | Export all as `list[BaseTool]` |
| `list_agents()` | Get metadata for all agents |

### Built-in Prompts (`prompts.*`)

`PLANNER_PROMPT`, `EXECUTOR_PROMPT`, `REVIEWER_PROMPT`, `SUMMARIZER_PROMPT`, `TRANSLATOR_PROMPT`, `CODE_ANALYST_PROMPT`
