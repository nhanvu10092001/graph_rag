# HLG LLM Utils - High-Performance Graph RAG & Multi-Agent Framework

[![Docker Compose](https://img.shields.io/badge/Docker-Compose_Ready-blue?logo=docker)](./docker-compose.yml)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)](./BE)
[![React 19](https://img.shields.io/badge/Frontend-React_19-61DAFB?logo=react)](./FE)
[![Neo4j](https://img.shields.io/badge/Graph_DB-Neo4j_5-008CC1?logo=neo4j)](https://neo4j.com)
[![pgvector](https://img.shields.io/badge/Vector_DB-PostgreSQL_pgvector-336791?logo=postgresql)](https://github.com/pgvector/pgvector)
[![vLLM](https://img.shields.io/badge/Serving-vLLM-blueviolet)](https://github.com/vllm-project/vllm)

**HLG LLM Utils** is a modular, high-performance Graph RAG (Retrieval-Augmented Generation) and multi-agent AI framework. It features hierarchical Leiden/Louvain community detection, entity/relationship extraction, self-correcting Cypher query generation, map-reduce global search, adaptive knowledge retrieval (ARK), doc parsing with local PaddleOCR-VL vision OCR, self-hosted nomic embeddings via vLLM, and an interactive React 19 interface.

---

## 🏗️ Architecture Overview

```
                                  ┌──────────────────┐
                                  │   React 19 UI    │
                                  │   (Port 3000)    │
                                  └────────┬─────────┘
                                           │ (REST / WebSockets)
                                           ▼
                                  ┌──────────────────┐
                                  │  Express Proxy   │
                                  └────────┬─────────┘
                                           │
                                           ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                         FastAPI Backend (Port 8000)                         │
 │                                                                             │
 │ ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────────────┐ │
 │ │  LangGraph Agent │    │  Graph RAG Core  │    │  Doc Parsing & OCR      │ │
 │ └──────────────────┘    └──────────────────┘    └─────────────────────────┘ │
 └──────┬──────────────────────────┬──────────────────────────┬────────────────┘
        │                          │                          │
        ├──────────────────────────┼──────────────────────────┤
        ▼                          ▼                          ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│ PostgreSQL 15 │          │  Neo4j 5.18   │          │   MinIO S3    │
│  (pgvector)   │          │  (Graph DB)   │          │ (Doc Storage) │
└───────────────┘          └───────────────┘          └───────────────┘
        ▲                          ▲                          ▲
        │                          │                          │
 ┌──────┴──────────────────────────┴──────────────────────────┴────────────────┐
 │                      Local Model Serving (vLLM on GPU)                      │
 │                                                                             │
 │ ┌────────────────────────────────────┐ ┌──────────────────────────────────┐ │
 │ │ nomic-embed-text-v1.5 (Port 8082)  │ │   PaddleOCR-VL-1.5 (Port 8083)   │ │
 │ └────────────────────────────────────┘ └──────────────────────────────────┘ │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Table of Contents

- [Prerequisites](#-prerequisites)
- [Quick Start with Docker Compose (Recommended)](#-quick-start-with-docker-compose-recommended)
- [Local Development Setup](#-local-development-setup)
- [Services & Dashboard URLs](#-services--dashboard-urls)
- [Configuration Reference](#-configuration-reference)
- [Database Migrations & Management](#-database-migrations--management)
- [Verification & E2E Testing](#-verification--e2e-testing)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## ⚡ Prerequisites

Before getting started, ensure you have the following installed on your host machine:

- **Docker Desktop** or **Docker Engine & Docker Compose v2+** (with NVIDIA Container Toolkit for GPU acceleration)
- **Node.js 20+** and **npm** (for local frontend development)
- **Python 3.11+** and **`uv`** package manager (for local backend development)
- An **Anthropic API Key** (`sk-ant-...`) for Claude LLM inference

---

## 🚀 Quick Start with Docker Compose (Recommended)

Run the entire Graph RAG stack (Databases, vLLM Embedding, vLLM PaddleOCR-VL, FastAPI Backend, React Frontend) in isolated Docker containers with a single command.

### Step 1: Environment Setup

Clone the repository and create your Docker environment file from the provided template:

```bash
# Clone the repository
git clone https://github.com/vupcln/graph_rag.git
cd graph_rag

# Copy the Docker Compose environment template
cp .env.docker.example .env
```

### Step 2: Configure API Key

Open `.env` in your text editor and set your Anthropic API key:

```env
# Anthropic Claude Configuration
LLM_PROVIDER=claude
LLM_MODEL=claude-haiku-4-5-20251001
ANTHROPIC_API_KEY=sk-ant-your-actual-anthropic-api-key-here
```

### Step 3: Build & Launch Services

Start all containers in detached mode:

```bash
docker compose up --build -d
```

This command automatically launches:
1. **PostgreSQL 15 (pgvector)**: Relational and vector document metadata storage.
2. **Neo4j 5.18**: Graph database with APOC plugin for entity & relation knowledge graph.
3. **MinIO Object Storage**: S3-compatible raw file storage.
4. **vLLM Embeddings**: Serves `nomic-ai/nomic-embed-text-v1.5` on port `8082`.
5. **vLLM PaddleOCR-VL**: Serves `PaddlePaddle/PaddleOCR-VL-1.5` vision OCR on port `8083`.
6. **FastAPI Backend**: Executes database schema migrations (`alembic upgrade head`) and serves Graph RAG agent APIs on port `8000`.
7. **React 19 Frontend**: Serves the interactive user interface on port `3000`.

### Step 4: Check Service Status

Verify that all services are healthy and running:

```bash
docker compose ps
```

---

## 💻 Local Development Setup

If you wish to make live code modifications to the Backend or Frontend, run infrastructure databases in Docker while executing application code locally.

### Step 1: Launch Infrastructure Containers

Start PostgreSQL, Neo4j, and MinIO databases:

```bash
docker compose up postgres neo4j minio -d
```

### Step 2: Install Monorepo Workspace Packages

Set up all Python workspace packages in `RAG_package`:

```bash
cd RAG_package
make setup-all
cd ..
```

### Step 3: Set Up and Run Backend (`BE/`)

Create the backend environment configuration:

```bash
cd BE
cp .env.example .env  # Or edit .env directly
```

Run Alembic schema migrations and launch the FastAPI server:

```bash
# Execute database migrations
uv run alembic upgrade head

# Start FastAPI development server
uv run python main.py
```

The backend server will run at `http://localhost:8000`.

### Step 4: Set Up and Run Frontend (`FE/`)

In a separate terminal window:

```bash
cd FE
cp .env.example .env
npm install
npm run dev
```

The React frontend development server will run at `http://localhost:3000`.

---

## 🌐 Services & Dashboard URLs

| Service Name | Container Name | Host Port | Description & URL |
| :--- | :--- | :--- | :--- |
| **Frontend UI** | `graph_rag_frontend` | `3000` | [http://localhost:3000](http://localhost:3000) (Interactive Graph RAG Chat Interface) |
| **FastAPI Backend** | `graph_rag_backend` | `8000` | [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger API Documentation) |
| **Neo4j Browser** | `graph_rag_neo4j` | `7474` / `7687` | [http://localhost:7474](http://localhost:7474) (Auth: `neo4j` / `password`) |
| **MinIO Console** | `graph_rag_minio` | `9001` / `9000` | [http://localhost:9001](http://localhost:9001) (Auth: `minioadmin` / `minioadmin`) |
| **vLLM Embedding** | `graph_rag_embedding` | `8082` | [http://localhost:8082/health](http://localhost:8082/health) (`nomic-ai/nomic-embed-text-v1.5`) |
| **vLLM PaddleOCR-VL** | `graph_rag_paddleocr` | `8083` | [http://localhost:8083/health](http://localhost:8083/health) (`PaddlePaddle/PaddleOCR-VL-1.5`) |
| **PostgreSQL DB** | `graph_rag_postgres` | `5432` | `postgresql://postgres:postgres@localhost:5432/graph_rag_db` |

---

## ⚙️ Configuration Reference

Main configuration options are controlled via `BE/graph_rag_config.yaml` and environment overrides (`.env`):

- **Neo4j**: Graph database connection parameters (`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`).
- **LLM**: Anthropic Claude model configuration (`LLM_PROVIDER=claude`, `LLM_MODEL=claude-haiku-4-5-20251001`, `ANTHROPIC_API_KEY`).
- **Embeddings**: Self-hosted `nomic-ai/nomic-embed-text-v1.5` served via local vLLM endpoint (`http://localhost:8082/v1`).
- **OCR**: `PaddleOCR-VL-1.5` vision OCR served via local vLLM endpoint (`http://localhost:8083/v1`).
- **Reranking**: Local CrossEncoder model (`BAAI/bge-reranker-large`) or FlashRank.
- **Community Detection**: Algorithm (`leiden` or `louvain`), resolution level, hierarchical max depth.
- **Query Modes**:
  - `auto`: Autonomous selection based on query intent.
  - `local`: Vector search + entity graph traversal.
  - `global`: Map-reduce summarization across Leiden communities.
  - `ark`: Adaptive Retriever of Knowledge (multi-hop graph agent search).

---

## 🗄️ Database Migrations & Management

PostgreSQL schema migrations are managed via **Alembic**.

### Running Migrations Manually
```bash
cd BE
uv run alembic upgrade head
```

### Creating New Schema Migrations
When adding or updating SQLAlchemy ORM models in `BE/app/models.py`:
```bash
cd BE
uv run alembic revision --autogenerate -m "Add new column or table"
uv run alembic upgrade head
```

---

## 🧪 Verification & E2E Testing

Validate the end-to-end functionality of chat streams, document deletion, and community panel endpoints using Playwright verification scripts:

```bash
cd FE

# Verify chat streaming and backend response
node verify_chat.cjs

# Verify document deletion modals
node verify_custom_modals.cjs

# Verify community API endpoints
node verify_groups.cjs
```

---

## ❓ Troubleshooting & FAQs

### 1. Port Conflicts (e.g., Port 5432, 7474, or 8000 already in use)
If local PostgreSQL or Neo4j instances are running on your host machine, stop them or change mapped ports in `docker-compose.yml`:
```bash
# Check process holding port 5432
lsof -i :5432
```

### 2. Neo4j Authentication Error
If you cannot log into Neo4j at `http://localhost:7474`, default credentials set in `docker-compose.yml` are:
- Username: `neo4j`
- Password: `password`

To reset Neo4j volumes:
```bash
docker compose down -v
docker compose up neo4j -d
```

### 3. Viewing Container Logs
To inspect real-time logs for backend migrations or API queries:
```bash
# Follow backend logs
docker compose logs -f backend

# Follow embedding logs
docker compose logs -f embedding

# Follow OCR logs
docker compose logs -f paddleocr

# Follow frontend logs
docker compose logs -f frontend
```

---

## 📄 License

This repository is licensed under the Apache 2.0 License.
