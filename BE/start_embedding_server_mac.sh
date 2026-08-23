#!/bin/bash

# Start llama-server hosting nomic-ai/nomic-embed-text-v1.5 on port 8082
LLAMA_SERVER=$(which llama-server)

if [ -z "$LLAMA_SERVER" ]; then
  if [ -f "/opt/homebrew/bin/llama-server" ]; then
    LLAMA_SERVER="/opt/homebrew/bin/llama-server"
  else
    echo "Error: llama-server not found. Please install via brew: brew install llama.cpp"
    exit 1
  fi
fi

# Check if port 8082 is already in use
EXISTING_PID=$(lsof -ti :8082)
if [ -n "$EXISTING_PID" ]; then
  echo "Port 8082 is currently in use by process $EXISTING_PID. Stopping previous instance..."
  kill -9 $EXISTING_PID 2>/dev/null
  sleep 1
fi

echo "Starting llama-server for nomic-ai/nomic-embed-text-v1.5 on http://localhost:8082/v1..."

$LLAMA_SERVER \
  --hf-repo nomic-ai/nomic-embed-text-v1.5-GGUF \
  --hf-file nomic-embed-text-v1.5.f16.gguf \
  --port 8082 \
  --embedding \
  --alias nomic-ai/nomic-embed-text-v1.5 \
  -c 4096
