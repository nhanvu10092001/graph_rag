# HLG LLM Utils

A modular LLM framework with plugin-based architecture for building AI applications. Built as a monorepo with 6 independently installable packages.

## About Project

This monorepo contains:

- **`llm-utils-core`** - Core plugin framework (TaskPlugin, load_plugins)
- **`llm-utils-llm`** - LLM and Embeddings factories (Ollama, OpenAI)
- **`llm-utils-vector`** - Vector store implementations (Qdrant, PGVector)
- **`llm-utils-rag`** - RAG (Retrieval-Augmented Generation) functionality
- **`llm-utils-mcp`** - Model Context Protocol (MCP) server implementation
- **`llm-utils-mcp-io`** - MCP-IO tools for file/directory operations

## Installation

Clone and setup using Makefile commands:

```bash
# Clone repository
git clone <repository-url>
cd hlg-llm-utils

# Setup MCP features (includes FastMCP server, exporters)
make setup-mcp

# Setup RAG features (includes LLM providers, vector stores)
make setup-rag

# Get help for all available commands
make help
```

## Demo

Run demos using Makefile commands:

```bash
# MCP Server Demo
make run-fastmcp            # FastMCP server on port 8000

# MCP-IO Tools Demo
make mcp-io-demo           # MCP-IO server on port 8002

# RAG Interactive Demo
make test-rag              # Interactive RAG demo

# Convert MCP to OpenAPI
make cvt-fastmcp-2-openapi # OpenAPI server on port 8001
```

---

**Built with [uv](https://docs.astral.sh/uv/) workspaces for modern Python monorepo development.**