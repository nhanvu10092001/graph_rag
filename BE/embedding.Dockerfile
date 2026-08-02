FROM python:3.11-slim

RUN pip install --no-cache-dir \
    "infinity_emb[all]" \
    einops

ENV HF_HOME=/data

EXPOSE 8081

ENTRYPOINT ["infinity_emb", "v2", \
    "--model-id", "nomic-ai/nomic-embed-text-v1.5", \
    "--port", "8081", \
    "--engine", "torch", \
    "--dtype", "float32"]
