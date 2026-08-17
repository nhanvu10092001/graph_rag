# LLM Utils RAG

RAG (Retrieval-Augmented Generation) plugin for LLM Utils.

## Features

- Document indexing and chunking
- Vector similarity search
- RAG query processing with LLM integration
- Async document processing
- Support for multiple vector databases

## Installation

```bash
# Basic RAG functionality
uv add llm-utils-rag

# With all optional vector database support
uv add "llm-utils-rag[all]"
```

## Usage

```python
from llm_utils_rag import RAGPlugin

# Configure RAG plugin
config = {
    "llm": {
        "provider": "ollama",
        "ollama": {"model": "llama3:8b"}
    },
    "embeddings": {
        "provider": "ollama", 
        "ollama": {"model": "nomic-embed-text"}
    },
    "vector_store": {"provider": "pgvector"}
}

# Create and use RAG plugin
rag_plugin = RAGPlugin(config)

# Index documents
result = await rag_plugin.run({
    "action": "index",
    "documents_path": "path/to/docs.txt",
    "collection_name": "my_docs"
})

# Query documents
result = await rag_plugin.run({
    "action": "query",
    "query": "What is this document about?",
    "collection_name": "my_docs"
})
```

## Dependencies

This package depends on:
- `llm-utils-core` - Core plugin framework
- `llm-utils-vector` - Vector store support
- `langchain` - LangChain framework
- `langchain-community` - Community integrations
- `langchain-text-splitters` - Document chunking