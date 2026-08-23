# CLAUDE.md

## Agent Rules & Workflow
- **Before Starting Any Task**: The agent MUST read `CLAUDE.md` first to gain a complete understanding of the project overview, architecture, directory structure, technology stack, and existing conventions before taking action.
- **After Completing Any Task**: The agent MUST update `CLAUDE.md` if the task introduced changes to architecture, packages/dependencies, configuration, setup/run commands, directory structure, or project conventions, ensuring future sessions remain accurate and up to date.

---

## Project Overview
**HLG LLM Utils** is a modular, high-performance Graph RAG and multi-agent AI application framework. It is structured as a Python monorepo managed via **uv workspaces**, comprising **10 packages** (9 workspace member packages and 1 standalone agent tool library), alongside a FastAPI backend application and a React 19 / Vite 6 frontend interface.

---

## Architecture Summary

### Monorepo Packages Overview
```
                         ┌──────────────────┐
                         │  llm-utils-core  │
                         └────────┬─────────┘
                                  │ (Plugin Base & Loader)
                                  ▼
 ┌────────────────┬───────────────┬────────────────┬─────────────────┐
 │ llm-utils-llm  │ llm-utils-vec │ llm-utils-parse│llm-utils-rerank │
 └────────────────┼───────────────┼────────────────┴─────────────────┘
                  │               │
                  ▼               ▼
        ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
        │  llm-utils-rag   │  │  llm-utils-sub   │  │  llm-utils-mcp   │
        └────────┬─────────┘  └──────────────────┘  └──────────────────┘
                 │
                 ▼
        ┌──────────────────┐  ┌───────────────────────┐
        │llm-utils-graph-rag│  │ agent_tool_packages   │
        └──────────────────┘  └───────────────────────┘
```

1. **`llm-utils-core`**: Defines plugin base interfaces (`TaskPlugin`) and dynamic loader logic via Python entry points.
2. **`llm-utils-llm`**: Model provider factory (`LLMFactory`, `EmbeddingsFactory`) for OpenAI and Anthropic models.
3. **`llm-utils-vector`**: Provider implementations for vector stores, including Qdrant and PostgreSQL (`pgvector`).
4. **`llm-utils-parser`**: Multi-format document parser supporting PDF (PyMuPDF), Office documents, code files, and OCR engine integrations (Tesseract, PaddleOCR, DeepSeek, Docling).
5. **`llm-utils-reranker`**: Second-stage reranking capabilities including FlashRank, CrossEncoder, and LLM-based rerankers.
6. **`llm-utils-rag`**: Core RAG package providing `IndexingService`, `QueryService`, `DeletionService`, BM25/hybrid search, query rewriting, and adaptive retrieval.
7. **`llm-utils-graph-rag`**: Neo4j and LangGraph-backed Graph RAG package with entity/relationship extraction, self-correcting Cypher query generation, hierarchical Leiden/Louvain community detection, LLM community summaries, map-reduce global search, and ARK (Adaptive Retriever of Knowledge).
8. **`llm-utils-subagent`**: Wrapper framework for building stateless subagents:
   - **Simple Subagents**: Direct prompt-response execution.
   - **ReAct Subagents**: Tool-calling execution loop.
   - **Graph Subagents**: Multi-step state graph execution workflows.
9. **`llm-utils-mcp`**: Model Context Protocol (MCP) server, FastMCP integrations, and OpenAPI transformation utilities.
10. **`agent_tool_packages`**: Standalone tool library (`agent-tool-library`) implementing utility tools (web search, time, weather, OCR, media processing) compatible with LangChain.

### Backend Architecture (`BE/`)
- **Agent Subsystem (`BE/app/agent/`)**:
  - `graph.py`: Main LangGraph agent state machine executing multi-turn workflows and invoking tools.
  - `llm.py`: Unified LLM factory supporting Anthropic and OpenAI backends.
  - `security.py`: Guardrail engine checking for prompt injection, jailbreak attempts, and malicious input.
  - `tools.py`: LangChain structured tool declarations exposing Graph RAG query modes (`auto`, `local`, `global`, `ark`).
- **Services Layer (`BE/app/services/`)**:
  - `registry.py`: Thread-safe, lazy-initialized singleton manager (`GraphRAGServiceRegistry`) managing shared Graph RAG service lifecycle.
  - `document_service.py`: Manages PostgreSQL document metadata and async background processing pipelines.
  - `file_storage.py`: Handles raw file upload storage and retrieval using MinIO S3 object storage.

### Frontend Architecture (`FE/`)
- **Stack**: React 19, TypeScript 5.8, Vite 6, Tailwind CSS v4, Express 4 server (`server.ts` proxying API and WebSocket traffic to FastAPI).
- **Core Components**:
  - `ThinkingPanel.tsx`: Displays model reasoning and step-by-step thinking output.
  - `ToolCallBadge.tsx` & `ToolCallGroup.tsx`: Renders tool execution progress and call arguments.
  - `CommunityPanel.tsx`: Interactive viewer for Leiden community detection hierarchies and summaries.
  - Custom Modals: Confirmation dialogs for document deletion and group management.
  - KaTeX & Markdown: Math notation rendering (`rehype-katex`, `remark-math`) and formatted markdown rendering.
- **Verification**: Node Playwright scripts (`verify_chat.cjs`, `verify_custom_modals.cjs`, `verify_groups.cjs`, `test_group_api.cjs`) for E2E validation.

---

## Directory Structure
```
graph_rag/
├── BE/                            # Backend FastAPI Application
│   ├── app/                       # Application source package
│   │   ├── agent/                 # LangGraph Agent & Guardrails Engine
│   │   │   ├── graph.py           # LangGraph workflow definition & state graph
│   │   │   ├── llm.py             # LLM provider factory (Anthropic / OpenAI)
│   │   │   ├── security.py        # Security & prompt injection guardrails
│   │   │   └── tools.py           # Graph RAG structured tools (auto/local/global/ark)
│   │   ├── services/              # Application logic & service registry
│   │   │   ├── registry.py        # Thread-safe lazy Graph RAG service registry
│   │   │   ├── document_service.py# Document metadata & background indexing
│   │   │   └── file_storage.py    # MinIO S3 object storage service
│   │   ├── routers/               # FastAPI route definitions (chat, config, docs, community)
│   │   ├── schemas/               # Pydantic request/response models
│   │   ├── config.py              # Configuration manager & YAML parser
│   │   ├── database.py            # SQLAlchemy database sessions
│   │   └── models.py              # ORM models (Document, Group, etc.)
│   ├── alembic/                   # PostgreSQL schema migration scripts
│   ├── graph_rag_config.yaml      # Unified system configuration file
│   ├── main.py                    # FastAPI server entry point
│   └── requirements.txt           # Python dependencies for BE
├── FE/                            # Frontend Web Application
│   ├── src/                       # React 19 source code
│   │   ├── components/            # UI components (ThinkingPanel, CommunityPanel, etc.)
│   │   └── App.tsx                # Main frontend application component
│   ├── server.ts                  # Express 4 API & WebSocket proxy server
│   ├── verify_chat.cjs            # Chat E2E test script
│   ├── verify_custom_modals.cjs   # Deletion modal verification script
│   ├── verify_groups.cjs          # Group API test script
│   ├── package.json               # Node.js dependencies & scripts
│   └── vite.config.ts             # Vite build configuration
├── RAG_package/                   # Monorepo Workspace Container
│   ├── Makefile                   # Makefile shortcuts for demos and environment setup
│   ├── pyproject.toml             # UV workspace configuration
│   ├── packages/                  # Workspace Member Packages
│   │   ├── llm-utils-core/        # Plugin base & loader
│   │   ├── llm-utils-llm/         # LLM & Embeddings factory
│   │   ├── llm-utils-vector/      # Qdrant & PGVector providers
│   │   ├── llm-utils-parser/      # Multi-format parser & OCR package
│   │   ├── llm-utils-reranker/    # Reranking engines (FlashRank, CrossEncoder, LLM)
│   │   ├── llm-utils-rag/         # Traditional RAG, hybrid search & services
│   │   ├── llm-utils-graph-rag/   # Neo4j Graph RAG, Leiden community detection & ARK
│   │   ├── llm-utils-subagent/    # Simple, ReAct, and Graph subagents
│   │   ├── llm-utils-mcp/         # MCP server, adapters & OpenAPI generator
│   │   └── agent_tool_packages/   # Standalone general agent tools library
│   └── uv.lock                    # Monorepo lockfile
├── .agents/                       # Agent rule sets & customizations
└── README.md                      # General repository documentation
```

---

## Technology Stack
- **Languages**: Python >= 3.9 (BE / Monorepo), TypeScript 5.8 & Node.js (FE)
- **Monorepo Manager**: [uv Workspaces](https://docs.astral.sh/uv/)
- **Frontend Engine**: React 19, Vite 6, Tailwind CSS v4, Express 4 (Proxy Server)
- **AI & Graph Frameworks**: LangChain, LangGraph (State Graphs, ReAct loops), Neo4j Cypher auto-correction
- **Databases & Storage**:
  - **Vector Store**: Qdrant, PostgreSQL with `pgvector`
  - **Graph Store**: Neo4j 5+
  - **Relational DB**: PostgreSQL 12+ (Alembic migrations)
  - **Object Storage**: MinIO S3-compatible storage
- **Document Processing & Reranking**:
  - **Parsers**: PyMuPDF, Docling, Office/Code parsers
  - **OCR Engines**: Tesseract, PaddleOCR, DeepSeek OCR
  - **Rerankers**: FlashRank, CrossEncoder, LLMReranker
- **Graph Clustering**: `igraph` + `leidenalg` (Leiden algorithm) / Louvain

---

## Setup & Running Demos

### 1. Workspace Installation
```bash
cd RAG_package

# Install and link all packages in the uv workspace
make setup-all
```

### 2. Infrastructure Setup (Docker Services)
Start PostgreSQL, Neo4j, and MinIO storage services:
```bash
docker start graph_rag_postgres graph_rag_neo4j graph_rag_minio
```
Or initialize Neo4j manually:
```bash
docker run -d --name neo4j-rag -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

### 3. Application Launch Commands

- **Backend Server (FastAPI on Port 8000)**:
  ```bash
  cd BE
  uv run alembic upgrade head
  uv run python main.py
  ```

- **Frontend Application (Express / Vite on Port 3000)**:
  ```bash
  cd FE
  cp .env.example .env
  npm run dev
  ```

- **Frontend Verification Scripts**:
  ```bash
  cd FE
  node verify_chat.cjs
  node verify_custom_modals.cjs
  node verify_groups.cjs
  ```

---

## Configuration

The main configuration is located at `BE/graph_rag_config.yaml` (loaded via `BE/app/config.py`):
- **`graph_rag`**: Neo4j connection parameters, extraction prompt templates, embedding models.
- **`reranking`**: Active reranker engine selection (`flashrank`, `cross_encoder`, `llm`), model path, `top_k`.
- **`community_detection`**: Leiden algorithm parameters (resolution, max depth levels, auto-rebuild flags).
- **`query`**: Search mode configuration (`auto`, `local`, `global`, `ark`), community map-reduce thresholds.
- **`chunking`**: Token size, overlap ratio, chunk strategy.
- **`parsing`**: Parser selection, OCR engine configuration (Tesseract, PaddleOCR, DeepSeek).
- **`storage`**: MinIO bucket name, endpoint, access credentials, PostgreSQL connection URI.

---

## Coding Conventions
- **Dynamic Extensibility**: Rely on `TaskPlugin` class implementations for new actions. Dynamic package tools register via `llm_utils.plugins` entry points.
- **Lazy Singleton Registry**: Backend services access core RAG components via `GraphRAGServiceRegistry` in `BE/app/services/registry.py` to ensure thread-safe single initialization.
- **Stateless Subagents**: Subagents wrapped in `SubagentTool` must remain stateless by re-instantiating state graphs and memory history on every `run`/`arun` call.
- **Service-Oriented Decoupling**: Keep routers and tools light by delegating query, indexing, community detection, and storage operations to specialized service classes.
- **Strict Typing**: All new Python classes, methods, parameters, and return types must include complete type annotations.
