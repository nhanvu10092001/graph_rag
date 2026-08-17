# LLM Utils Vector

Vector store factories and implementations for LLM Utils.

## Features

- VectorStoreFactory for creating vector stores
- Standard LangChain PGVector implementation for PostgreSQL
- Support for Qdrant and PostgreSQL vector databases
- Graceful handling of missing dependencies

## Installation

```bash
# Basic installation (includes PostgreSQL support)
uv add llm-utils-vector

# With Qdrant support
uv add "llm-utils-vector[qdrant]"

# With all vector databases
uv add "llm-utils-vector[all]"
```

## Usage

```python
from llm_utils_vector import VectorStoreFactory
from langchain_core.documents import Document

# Create vector store
documents = [Document(page_content="Your text", metadata={})]
vectorstore = VectorStoreFactory.create_vectorstore(
    provider="pgvector",
    documents=documents,
    embeddings=your_embeddings,
    collection_name="my_collection"
)
```

## Supported Vector Stores

### PostgreSQL with pgvector
Uses the standard LangChain `langchain-postgres` implementation.

### Qdrant
Uses `langchain-qdrant` implementation (requires optional dependencies).

## Environment Variables

### PGVector
- `PGVECTOR_HOST` - PostgreSQL host (default: 172.16.6.31)
- `PGVECTOR_PORT` - PostgreSQL port (default: 5432)  
- `PGVECTOR_USER` - PostgreSQL user (default: heligate)
- `PGVECTOR_PASSWORD` - PostgreSQL password (required)
- `PGVECTOR_DATABASE` - PostgreSQL database (default: vector_db)

### Qdrant
- `QDRANT_HOST` - Qdrant host (default: 172.16.6.155)
- `QDRANT_PORT` - Qdrant port (default: 6333)
- `QDRANT_API_KEY` - Qdrant API key (required)