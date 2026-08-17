# LLM Utils LLM

Factory classes for creating LLM and Embeddings instances with various providers.

## Features

- **LLMFactory**: Create LLM instances (Ollama, OpenAI, etc.)
- **EmbeddingsFactory**: Create Embeddings instances (Ollama, OpenAI, etc.)
- **Automatic provider registration**: Common providers are auto-registered
- **Extensible**: Register custom providers easily

## Installation

```bash
# Install with specific providers
pip install llm-utils-llm[ollama]  # For Ollama support
pip install llm-utils-llm[openai]  # For OpenAI support
pip install llm-utils-llm[all]     # For all providers
```

## Usage

```python
from llm_utils_llm import LLMFactory, EmbeddingsFactory

# Create LLM instance
llm_config = {
    "provider": "ollama",
    "ollama": {
        "model": "llama3:8b",
        "base_url": "http://localhost:11434"
    }
}
llm = LLMFactory.create_llm(llm_config)

# Create Embeddings instance  
embeddings_config = {
    "provider": "ollama", 
    "ollama": {
        "model": "nomic-embed-text",
        "base_url": "http://localhost:11434"
    }
}
embeddings = EmbeddingsFactory.create_embeddings(embeddings_config)
```