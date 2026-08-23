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

# Copy and install RAG_package workspace member packages
COPY RAG_package /app/RAG_package
WORKDIR /app/RAG_package
RUN uv pip install --system \
    -e packages/llm-utils-core \
    -e packages/llm-utils-llm \
    -e packages/llm-utils-vector \
    -e packages/llm-utils-parser \
    -e packages/llm-utils-reranker \
    -e packages/llm-utils-rag \
    -e packages/llm-utils-graph-rag \
    -e packages/llm-utils-subagent \
    -e packages/llm-utils-mcp \
    -e packages/agent_tool_packages

# Copy Backend application
COPY BE /app/BE
WORKDIR /app/BE

# Install Backend Python requirements
RUN uv pip install --system -r requirements.txt

EXPOSE 8000

# Make sure entrypoint script is executable
RUN chmod +x /app/BE/docker-entrypoint.sh

ENTRYPOINT ["/app/BE/docker-entrypoint.sh"]
