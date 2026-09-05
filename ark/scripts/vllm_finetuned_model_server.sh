export PROJECT_DIR="/n/holylfs06/LABS/mzitnik_lab/Users/jpolonuer/graph-reasoning-agents-refactor"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"
cd "benchmarks/stark"

MODEL_NAME="Qwen3-8B"

python -m vllm.entrypoints.openai.api_server \
  --model data/finetuning/amazon/${MODEL_NAME}/explorer/merged \
  --served-model-name ${MODEL_NAME}-graphagent \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3