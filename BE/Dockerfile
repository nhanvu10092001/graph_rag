FROM python:3.11-slim

# Install uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    tesseract-ocr \
    tesseract-ocr-vie \
    tesseract-ocr-eng \
    poppler-utils \
    netcat-openbsd \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Copy dependencies first for Docker layer caching
COPY BE/requirements.txt /app/BE/requirements.txt
RUN uv pip install --system -r /app/BE/requirements.txt

# 2. Copy and install RAG_package workspace member packages in editable mode
COPY RAG_package /app/RAG_package
RUN uv pip install --system \
    -e /app/RAG_package/packages/llm-utils-core \
    -e /app/RAG_package/packages/llm-utils-llm \
    -e /app/RAG_package/packages/llm-utils-vector \
    -e /app/RAG_package/packages/llm-utils-parser \
    -e /app/RAG_package/packages/llm-utils-reranker \
    -e /app/RAG_package/packages/llm-utils-rag \
    -e /app/RAG_package/packages/llm-utils-graph-rag \
    -e /app/RAG_package/packages/llm-utils-subagent \
    -e /app/RAG_package/packages/llm-utils-mcp \
    -e /app/RAG_package/packages/agent_tool_packages

# 3. Copy Backend application source code
COPY BE /app/BE
WORKDIR /app/BE

EXPOSE 8000

# Make sure entrypoint script is executable
RUN chmod +x /app/BE/docker-entrypoint.sh

ENTRYPOINT ["/app/BE/docker-entrypoint.sh"]
