# CLAUDE.md

## Project Overview
**HLG LLM Utils** is a modular LLM framework with a plugin-based architecture for building AI applications. It is structured as a Python monorepo using **uv workspaces**, containing 8 core packages and an additional agent tool library.
---

## Architecture Summary
The project follows a decoupling strategy, separating core utilities, LLM integration, vector stores, RAG functionality, and multi-agent coordination into separate packages:

```
                  ┌──────────────────┐
                  │  llm-utils-core  │
                  └────────┬─────────┘
                           │ (Plugin Base & Loader)
                           ▼
 ┌─────────────────┬───────────────┬────────────────┐
 │ llm-utils-llm   │ llm-utils-vec │ llm-utils-sub  │
 └─────────────────┼───────────────┼────────────────┘
                   │               │
                   ▼               ▼
         ┌──────────────────┐  ┌──────────────────┐
         │  llm-utils-rag   │  │  llm-utils-mcp   │
         └──────────────────┘  └──────────────────┘
                   │
                   ▼
         ┌──────────────────┐
         │llm-utils-graph-rag│
         └──────────────────┘
```

1. **`llm-utils-core`**: Defines the base plugin interface (`TaskPlugin`) and handles dynamic loading of plugins using Python entry points.
2. **`llm-utils-llm`**: Factory implementations (`LLMFactory`, `EmbeddingsFactory`) for language models and embedding models, primarily supporting OpenAI.
3. **`llm-utils-vector`**: Decoupled interface (`VectorStoreProvider`) and concrete implementations for vector search databases, specifically `Qdrant` and `PostgreSQL (pgvector)`.
4. **`llm-utils-rag`**: End-to-end Retrieval-Augmented Generation capabilities. Contains specific services (`IndexingService`, `QueryService`, `DeletionService`) and complex retrieval features (BM25, hybrid retrieval, query transforms, reranking, and adaptive routing).
5. **`llm-utils-graph-rag`**: Graph RAG capabilities built using **Neo4j** and **LangGraph**, enabling entity/relationship extraction, Cypher query generation with auto-correction, and hybrid search.
6. **`llm-utils-subagent`**: Framework for creating coordinate-ready subagents. Supports:
   - **Simple Subagents**: Single prompt-response LLM calls.
   - **ReAct Subagents**: Loops with access to custom tools.
   - **Graph Subagents**: Structured multi-step pipelines built on top of state graphs.
7. **`llm-utils-mcp`**: Server/client implementations for Model Context Protocol (MCP), exposing local tools/exporters to FastMCP, LangChain structured tools, and LangChain MCP adapters.
8. **`agent_tool_packages`**: A standalone tool library (`agent-tool-library`) implementing standard utility tools (weather, time, web search, OCR, image processing) compatible with LangChain.

---

## Directory Structure
```
graph_rag/
├── BE/                            # Backend FastAPI Application
│   ├── app/                       # Application source package
│   │   ├── services/              # Business logic & storage services
│   │   ├── routers/               # FastAPI routers (chat, config, documents)
│   │   ├── database.py            # SQLAlchemy session setup
│   │   ├── models.py              # Database models (e.g. Document)
│   │   ├── parser.py              # File parsing & text extraction
│   │   └── agent.py               # LangGraph agent & tool calling config
│   ├── alembic/                   # Database schema migrations
│   ├── alembic.ini                # Alembic settings
│   ├── main.py                    # Root bootstrapper script
│   └── requirements.txt           # Backend python dependencies
├── FE/                            # Frontend Web Application (React + Vite)
├── RAG_package/                   # Monorepo packages container
│   ├── Makefile                   # Developer shortcuts for demos and setup
│   ├── pyproject.toml             # UV workspace and dependencies configuration
│   ├── setup.py                   # Legacy setup script
│   ├── uv.lock                    # Dependency lockfile
│   └── packages/                  # Workspace package members
│       ├── llm-utils-core/        # Plugin contracts and plugin loader
│       ├── llm-utils-llm/         # LLM/Embeddings factory (OpenAI)
│       ├── llm-utils-vector/      # Qdrant and PGVector providers
│       ├── llm-utils-rag/         # Query/Indexing services, hybrid search, RAG plugin
│       ├── llm-utils-graph-rag/   # Graph RAG using Neo4j and LangGraph
│       ├── llm-utils-subagent/    # Stateless subagent wrapper (simple, react, graph)
│       ├── llm-utils-mcp/         # MCP server, adapters, and exporters
│       ├── llm-utils-parser/      # Document parser and text extraction package
│       └── agent_tool_packages/   # General-purpose agent tool library (standalone)
├── .agents/                       # Agent rule sets & customizations
└── README.md                      # General repository introduction
```


---

## Technology Stack
- **Language**: Python >= 3.9 (workspace set up for Python 3.9, tool package requires Python >= 3.10)
- **Monorepo Manager**: [uv](https://docs.astral.sh/uv/) Workspaces
- **AI Framework**: LangChain (Core, OpenAI adapters, adapters for MCP)
- **State Machine**: LangGraph (used for ReAct agents, custom graph subagents, and Cypher self-correction)
- **Databases**: Qdrant (Vector Store), PostgreSQL with `pgvector`, Neo4j (Graph Store)
- **Web API**: FastAPI, Uvicorn, MCPO (OpenAPI generator for MCP)

---

## Setup & Running Demos

### 1. Basic Setup
```bash
cd RAG_package

# Setup entire workspace (MCP, RAG, and Graph RAG packages with dev dependencies)
make setup-all
```

### 2. Provider Prerequisites
- **OpenAI**: Save API key in a `.env` file at the `RAG_package/` root:
  ```env
  OPENAI_API_KEY=sk-your-openai-api-key
  ```
- **Databases**:
  - Run **Qdrant**: `docker run -p 6333:6333 qdrant/qdrant`
  - Run **PostgreSQL**: Install PostgreSQL 12+ and activate the `pgvector` extension.
  - Run **Neo4j**:
    ```bash
    docker run -d --name neo4j-rag -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
    ```
    Add connection info to `.env`:
    ```env
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USERNAME=neo4j
    NEO4J_PASSWORD=password
    ```

### 3. Demo Commands
```bash
# Start a simple FastMCP Calculator Server (port 8000)
make run-fastmcp

# Start a simple FastMCP Server with basic utility tools (port 8000)
make simple-fastmcp

# Start a FastMCP Server with 20+ file-based MCP-IO tools (port 8002)
make mcp-io-demo

# Start a FastMCP Server with RAG indexing and querying tools (port 8003)
make rag-mcp-demo

# Start a HTTP-based RAG document indexing API (port 9999)
make rag-indexing-api

# Run standard RAG interactive console demo
make test-rag

# Run Neo4j Graph RAG interactive console demo
make run-graph-rag-demo

# Run a LangChain conversational agent connected to MCP tools
make langchain-mcp-demo

# Convert an active MCP server's tools to an OpenAPI server (port 8001)
make cvt-fastmcp-2-openapi
```

### 4. Main Application (FE & BE)
- **Start Database & Storage Services (Docker)**:
  ```bash
  docker start graph_rag_postgres graph_rag_neo4j graph_rag_minio
  ```
- **Start Backend API (FastAPI - Port 8000)**:
  ```bash
  cd BE
  uv run alembic upgrade head
  uv run python main.py
  ```
- **Start Frontend UI (Express/Vite - Port 3000)**:
  ```bash
  cd FE
  # Copy .env.example to .env and configure GEMINI_API_KEY if needed
  cp .env.example .env
  npm run dev
  ```
- **Run Chat/Security Verification Tests (Playwright)**:
  ```bash
  cd FE
  node verify_chat.cjs
  ```
- **Run Group and Modal Deletion E2E Tests (Playwright)**:
  ```bash
  cd FE
  node verify_custom_modals.cjs
  ```

---

## Development & Test Commands
Navigate to the package or run from the workspace root:

```bash
# Run tests for the agent tools package
cd packages/agent_tool_packages
pytest tests/test_integration.py

# Run tests for the Neo4j Graph RAG package
cd packages/llm-utils-graph-rag
pytest tests/test_graph_rag.py

# Format code
black .

# Run linter
ruff check .

# Run static type checking
mypy .
```

---

## Coding Conventions
- **Dynamic Extensibility**: Rely on `TaskPlugin` class implementations for new actions. Dynamic package tools are registered via `llm_utils.plugins` entry points.
- **Stateless Subagents**: The `SubagentTool` class wraps subagents (Simple, ReAct, Graph) and keeps them stateless by rebuilding the execution graph and fresh memory history on every single `run`/`arun` invocation.
- **Service-Oriented Core**: Keep plugins thin. Delegate logic to dedicated services (e.g. `QueryService` / `GraphQueryService` for searching, `IndexingService` / `GraphIndexingService` for indexing, `DeletionService` for removal).
- **Type Hinting**: All new Python classes, methods, and functions must be properly typed.
