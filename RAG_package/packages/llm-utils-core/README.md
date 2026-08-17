# LLM Utils Core

Core framework for LLM utilities with plugin architecture.

## Features

- Plugin interface and discovery system
- LLM and Embeddings factories  
- Minimal dependencies for lightweight installation

## Installation

```bash
uv add llm-utils-core
```

For additional LLM providers:
```bash  
uv add "llm-utils-core[providers]"
```

## Usage

```python
from llm_utils_core import TaskPlugin, load_plugins

# Discover available plugins
plugins = load_plugins()

# For LLM functionality, use the llm-utils-llm package:
# from llm_utils_llm import LLMFactory
# config = {"provider": "ollama", "ollama": {"model": "llama3:8b"}}
# llm = LLMFactory.create_llm(config)
```