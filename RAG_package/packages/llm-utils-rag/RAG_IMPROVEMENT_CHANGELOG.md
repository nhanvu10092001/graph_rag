# RAG Package Improvement Changelog

> Completed: 2026-05-10

## Overview

Comprehensive refactoring and enhancement of the `llm-utils-rag` package across 6 phases, transforming the 877-line monolithic `RAGPlugin` into a modular, observable, and production-hardened architecture.

---

## Phase 1 — Critical Bug Fixes ✅

### 1.1 Fixed Duplicate Indexing Bug
- **File**: `plugins/rag_plugin.py` (L128–143)
- **Problem**: Sync indexing path called `add_documents` twice — once via `VectorStoreFactory` then again in a fallback block. If the first call succeeded but returned empty UUIDs, documents got indexed twice.
- **Fix**: Removed the fallback `vectorstore.add_documents()` call. Added `try/finally` for guaranteed `vectorstore.close()`.

### 1.2 Fixed `context.pop("action")` Mutation
- **File**: `plugins/rag_plugin.py` (L47, L63)
- **Problem**: `context.pop("action")` mutated the caller's dict, causing side effects.
- **Fix**: Changed to `context.get("action", "query")` in both `run()` and `arun()`.

### 1.3 Removed Bare `except:` Clauses
- **File**: `plugins/rag_plugin.py` (L141, L713)
- **Problem**: Bare `except:` swallowed all errors silently, masking real issues.
- **Fix**: Replaced with `except Exception:` with proper logging.

### 1.4 Fixed Fake Async Delete
- **File**: `plugins/rag_plugin.py` (L842–846)
- **Problem**: `_delete_vectors_async` just called the sync version directly, blocking the event loop.
- **Fix**: Uses `asyncio.get_running_loop().run_in_executor()` to run sync deletion in a thread pool.

### 1.5 Fixed Inconsistent Metadata Handling
- **Files**: `plugins/rag_plugin.py` (L110–113 sync, L187–190 async)
- **Problem**: Sync path set `{"filename": ...}` in metadata, async path overwrote with the full metadata dict.
- **Fix**: Both paths now consistently set `{"filename": filename}`.

---

## Phase 2 — Architecture Refactoring ✅

### 2.1 Extracted God Class into Focused Services

The 877-line `rag_plugin.py` was split into:

| New Module | Responsibility | Lines |
|---|---|---|
| `services/indexing_service.py` | Document indexing (sync + async) | 232 |
| `services/query_service.py` | Query pipeline (sync + async) | 254 |
| `services/deletion_service.py` | Vector deletion (all providers) | 189 |
| `services/connection_manager.py` | DB/Qdrant connection pooling | 135 |
| `plugins/rag_plugin.py` | Thin facade (delegator) | 164 |

### 2.2 Eliminated Sync/Async Code Duplication

Shared helper methods extract common logic:
- `_validate_and_extract_file()` — file validation
- `_create_embeddings()` / `_create_llm()` — factory calls
- `_build_retriever()` — retriever construction
- `_parse_response()` — response extraction
- `_build_sources()` — citation building

### 2.3 Connection Pooling for Database Access

- **File**: `services/connection_manager.py`
- Uses `psycopg2.pool.ThreadedConnectionPool` (configurable 1–10 connections)
- Singleton Qdrant client per manager instance
- `get_pg_connection()` / `release_pg_connection()` API
- Configurable via `pool_min` / `pool_max` in pgvector config

---

## Phase 3 — Retrieval Pipeline Upgrades ✅

### 3.1 Hybrid Search (Dense + Sparse BM25)

- **File**: `factories/retriever_factory.py`
- **Provider**: `hybrid_search`
- Combines dense vector retrieval with BM25 sparse retrieval using `EnsembleRetriever`
- Configurable weights: `dense_weight` (default 0.6), `sparse_weight` (default 0.4)
- Falls back to dense-only if BM25 index creation fails

### 3.2 Cross-Encoder Reranking

- **File**: `factories/retriever_factory.py`
- **Provider**: `cross_encoder_reranker`
- Uses HuggingFace `BAAI/bge-reranker-v2-m3` cross-encoder model
- Falls back to `OllamaListwiseRerankCompressor` if HuggingFace not installed
- Configurable: `model`, `top_n` (default 5)

### 3.3 HyDE Query Transformation

- **File**: `factories/retriever_factory.py`
- **Provider**: `hyde_retriever`
- Generates a hypothetical answer using LLM, embeds it, then searches for similar real documents
- Uses LangChain's `HypotheticalDocumentEmbedder`

### 3.4 Citation/Source Tracking

- **File**: `services/query_service.py`
- All query responses now include:

```json
{
  "sources": [
    {"filename": "doc.pdf", "chunk_id": 3, "chunk_method": "semantic", "category": "technical", "relevance_score": 0.92}
  ],
  "retrieval_metadata": {
    "chunks_returned": 5,
    "retrieval_time_ms": 230.4
  }
}
```

---

## Phase 4 — Advanced Chunking ✅

### 4.1 Contextual Chunker (Late Chunking)

- **File**: `utils/contextual_chunker.py` (120 lines)
- **Provider**: `contextual`
- Generates a 2-sentence LLM summary of each document
- Prepends `[Document Context: ...]` to every chunk
- Supports both sync and async via `split_documents()` / `asplit_documents()`

### 4.2 Markdown Header-Aware Chunker

- **File**: `utils/markdown_chunker.py` (86 lines)
- **Provider**: `markdown`
- Splits by `# ## ### ####` headers first, then applies recursive size splitting
- Preserves header hierarchy in metadata (`header_1`, `header_2`, etc.)

### 4.3 Code Language-Aware Chunker

- **File**: `utils/code_chunker.py` (103 lines)
- **Provider**: `code`
- Supports 25+ programming languages (Python, JS, TS, Java, Go, Rust, Ruby, C/C++, etc.)
- Uses language-specific separators (functions, classes, imports) for meaningful chunks
- Stamps `code_language` in chunk metadata

### Configuration Examples

```python
# Contextual chunking
config = {"chunking": {"provider": "contextual", "contextual": {"chunk_size": 1000}}}

# Markdown chunking
config = {"chunking": {"provider": "markdown", "markdown": {"strip_headers": False}}}

# Code chunking
config = {"chunking": {"provider": "code", "code": {"language": "python", "chunk_size": 1500}}}
```

---

## Phase 5 — Observability & Evaluation ✅

### 5.1 PipelineTimer

- **File**: `utils/pipeline_metrics.py`
- Context manager for timing individual pipeline stages:
  - `extraction`, `embedding_init`, `chunking`, `vectorstore_create`, `vectorstore_insert`
- Results included in every indexing response as `"timings"` dict

### 5.2 MetricsCollector

- **File**: `utils/pipeline_metrics.py`
- Aggregates query/indexing/error metrics across the plugin lifecycle
- Accessible via `plugin.get_metrics()`:

```json
{
  "queries": {"total": 42, "avg_latency_ms": 350.2, "max_latency_ms": 1200.0, "avg_docs_returned": 4.8},
  "indexing": {"total": 15, "avg_latency_ms": 2300.0, "total_chunks_indexed": 450},
  "errors": {"total": 2, "recent": [...]}
}
```

### 5.3 Structured Logging

- All services use `logging.getLogger(__name__)` with structured log messages
- Key log events: `indexing.completed`, `metrics.query`, `metrics.error`, `pipeline.stage.*`

---

## Phase 6 — Production Hardening ✅

### 6.1 Input Validation

- **File**: `utils/hardening.py`
- `InputValidator.validate_context()` runs before every `run()` / `arun()` call
- Validates: query length (max 10,000 chars), collection name format, action type, required fields
- Returns `ValidationError` responses instead of crashing

### 6.2 Retry Decorators

- **File**: `utils/hardening.py`
- `@retry(max_attempts=3)` — sync exponential backoff
- `@async_retry(max_attempts=3)` — async exponential backoff
- Configurable: `initial_delay`, `backoff_factor`, `retryable_exceptions`

### 6.3 Resource Cleanup Context Managers

- **File**: `utils/hardening.py`
- `managed_vectorstore(vs)` — sync context manager with guaranteed `close()`
- `managed_vectorstore_async(vs)` — async context manager with guaranteed `aclose()`

### 6.4 Query Caching (Deferred)

- Requires Redis or similar dependency decision — deferred for future implementation.

---

## File Inventory

### New Files Created (11)

| File | Lines | Purpose |
|---|---|---|
| `services/__init__.py` | 13 | Service layer exports |
| `services/indexing_service.py` | 232 | Sync/async document indexing with timing |
| `services/query_service.py` | 254 | Sync/async RAG queries with citations |
| `services/deletion_service.py` | 189 | Provider-dispatched vector deletion |
| `services/connection_manager.py` | 135 | Pooled PostgreSQL + Qdrant connections |
| `utils/pipeline_metrics.py` | 162 | PipelineTimer + MetricsCollector |
| `utils/hardening.py` | 209 | InputValidator, retry, cleanup |
| `utils/contextual_chunker.py` | 120 | Late chunking with LLM summaries |
| `utils/markdown_chunker.py` | 86 | Header-aware markdown splitting |
| `utils/code_chunker.py` | 103 | 25+ language-aware code splitting |

### Modified Files (5)

| File | Change |
|---|---|
| `plugins/rag_plugin.py` | Refactored from 877 → 164 lines (facade) |
| `factories/retriever_factory.py` | +3 providers (hybrid, cross-encoder, HyDE) |
| `factories/chunker_factory.py` | +3 providers (contextual, markdown, code) |
| `utils/__init__.py` | Added new exports |
| `__init__.py` | Added service exports |
