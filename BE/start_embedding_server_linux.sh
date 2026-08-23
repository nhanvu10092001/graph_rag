#!/bin/bash

docker rm -f vllm-vn-embed vllm-qwen3b 2>/dev/null

# Embedding first - small GPU footprint
docker run --gpus all -d \
  -p 8082:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --name vllm-vn-embed \
  vllm/vllm-openai:latest \
  nomic-ai/nomic-embed-text-v1.5 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --trust-remote-code \
  --gpu-memory-utilization 0.10 \
  --enforce-eager

echo "Waiting for embedding model..."
until curl -sf http://localhost:8082/v1/models > /dev/null 2>&1; do
  sleep 3
done
echo "Embedding ready."
