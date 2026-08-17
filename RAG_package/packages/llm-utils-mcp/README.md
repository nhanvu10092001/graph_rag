# LLM Utils MCP

MCP (Model Context Protocol) server implementation with support for FastMCP, LangChain structured tools, and LangChain MCP adapters.

## Features

- **Object-Oriented Design**: Clean, professional architecture with each class in its own file
- **Triple Export Support**: 
  - FastMCP server
  - LangChain structured tools
  - LangChain MCP adapter configuration
- **Clear Error Handling**: Transparent error messages that help users understand and fix issues
- **Tool Interface**: Unified interface for creating tools with built-in validation
- **Plugin Integration**: Seamlessly integrates with the llm-utils-core plugin system

## Installation

```bash
# Basic installation
pip install llm-utils-mcp

# With LangChain support
pip install llm-utils-mcp[langchain]

# Development installation
pip install llm-utils-mcp[dev]
```

## Quick Start

### 1. Create a Custom Tool

```python
from llm_utils_mcp import BaseTool, ToolParameter
from llm_utils_mcp.validators import validate_positive_integer

class CalculatorTool(BaseTool):
    def __init__(self):
        parameters = [
            ToolParameter(
                name="a",
                type=int,
                description="First number",
                required=True,
                validator=validate_positive_integer
            ),
            ToolParameter(
                name="b", 
                type=int,
                description="Second number",
                required=True,
                validator=validate_positive_integer
            ),
            ToolParameter(
                name="operation",
                type=str,
                description="Math operation (add, subtract, multiply, divide)",
                required=True
            )
        ]
        
        super().__init__(
            name="calculator",
            description="Perform basic math operations",
            parameters=parameters
        )
    
    async def execute(self, **kwargs):
        a = kwargs["a"]
        b = kwargs["b"] 
        operation = kwargs["operation"]
        
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Cannot divide by zero")
            result = a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        return {
            "result": result,
            "operation": f"{a} {operation} {b} = {result}"
        }
```

### 2. Create and Configure Server

```python
from llm_utils_mcp import MCPServer, ServerConfig

# Create server configuration
config = ServerConfig(
    name="my_calculator_server",
    description="Calculator MCP Server", 
    version="1.0.0",
    host="localhost",
    port=8000
)

# Create server
server = MCPServer(config)

# Add tools
calculator = CalculatorTool()
await server.add_tool(calculator)
```

### 3. Export to Different Formats

#### FastMCP Server
```python
# Export as FastMCP server
fastmcp_server = server.export_fastmcp()
fastmcp_server.run()
```

#### LangChain Structured Tools
```python
# Export as LangChain structured tools
structured_tools = server.export_langchain_structured_tools()

# Use with LangChain agents
from langchain.agents import create_openai_tools_agent
agent = create_openai_tools_agent(llm, structured_tools, prompt)
```

#### LangChain MCP Adapter
```python
# Export configuration for langchain-mcp-adapters
adapter_config = server.export_langchain_mcp_adapter()

# Use with MultiServerMCPClient
from langchain_mcp import MultiServerMCPClient
client = MultiServerMCPClient({
    "calculator": adapter_config["client_config"]["my_calculator_server"]
})
tools = await client.get_tools()
```

## Error Handling

The package provides clear, actionable error messages:

```python
try:
    await server.add_tool(invalid_tool)
except ToolRegistrationError as e:
    print(f"Tool registration failed: {e}")
    # Output: Tool registration failed: Tool 'my_tool' failed to register: missing required method 'execute'

try:
    result = await server.execute_tool("calculator", {"a": "not_a_number", "b": 5})
except ValidationError as e:
    print(f"Validation error: {e}")
    # Output: Validation error: Invalid value for 'a': not_a_number. Must be of type int
```

## Plugin Integration

The package integrates with the llm-utils-core plugin system:

```python
from llm_utils_core import load_plugins

# Load all plugins including MCP server
plugins = load_plugins()
mcp_plugin = plugins["mcp_server"]

# Use plugin
context = {
    "server_name": "my_server",
    "tools": [calculator],
    "port": 8000
}
result = await mcp_plugin.run(context)
```

## Tool Discovery

The package includes tools for discovering and managing tools:

```python
from llm_utils_mcp import ToolRegistry

registry = ToolRegistry()

# Discover tools from plugins
discovered = registry.discover_tools_from_plugins()

# Discover tools from directory
tools = registry.discover_tools_from_directory(Path("./my_tools"))

# List available tools
available = registry.list_available_tools()
```

## Validation

Built-in validators help ensure data quality:

```python
from llm_utils_mcp.validators import (
    validate_positive_integer,
    validate_url,
    validate_email,
    ValidatorRegistry
)

# Use built-in validators
validate_positive_integer(42)  # ✓
validate_url("https://example.com")  # ✓

# Create custom validators
length_validator = ValidatorRegistry.length_between(3, 10)
range_validator = ValidatorRegistry.range_validator(0, 100)
```

## Advanced Configuration

### Custom Export Configurations

```python
# Get detailed export information
export_info = server.export_langchain_structured_tools()
print(f"Exported {len(export_info)} tools")

# Save MCP adapter configuration
from pathlib import Path
exporter = LangChainMCPAdapterExporter(server)
exporter.save_config_file(Path("mcp_config.yaml"))

# Validate compatibility
validation = exporter.validate_adapter_compatibility()
if not validation["compatible"]:
    print(f"Compatibility issues: {validation['issues']}")
```

### Server Management

```python
# Get server information
info = server.get_server_info()
print(f"Server: {info['name']} with {info['tool_count']} tools")

# Remove tools
server.remove_tool("calculator")

# List tool metadata
metadata = server.get_tool_metadata()
for tool_name, meta in metadata.items():
    print(f"{tool_name}: {meta.description}")
```

## Examples

See the project-level examples directory for complete working demos:
- `examples/demo_mcp.py` - Complete calculator tool example with all export modes

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
black .
isort .
mypy .
```

## License

This project is licensed under the MIT License.