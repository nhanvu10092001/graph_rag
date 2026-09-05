#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# ARK Knowledge Distillation Pipeline
# Teacher → trajectory generation → fine-tune student (LoRA)
# ============================================================================

# ── CONFIG ──────────────────────────────────────────────────────────────────
GRAPH_NAME="prime"                         # prime | mag | amazon
TEACHER_MODEL="gemini-3.6-flash-high"      # tên model teacher
STUDENT_MODEL="Qwen/Qwen3-0.6B"           # model student để fine-tune
STUDENT_VLLM_PORT=8000                     # port vLLM serve student

NUMBER_OF_AGENTS=3                         # 3 agents per question (rejection sampling)
MAX_STEPS=10                               # max steps per trajectory (paper dùng 20)
TRAIN_LIMIT=600                            # số queries train (paper: 600 hoặc 6000)
VAL_LIMIT=200                              # số queries val

# ── TEACHER MODE ────────────────────────────────────────────────────────────
# Chọn 1 trong 3 mode bên dưới, comment 2 cái còn lại
TEACHER_MODE="anthropic_proxy"             # anthropic_proxy | openai_compat | vllm_qwen

# Mode 1: Gemini qua Anthropic-format proxy (localhost:8080)
PROXY_URL="http://localhost:8080"
PROXY_RPM=10

# Mode 2: Bất kỳ model nào qua OpenAI-compatible API (vLLM, Ollama, SGLang...)
OPENAI_BASE_URL="http://localhost:8000/v1"
OPENAI_RPM=60

# Mode 3: Qwen trên vLLM (có thinking support)
QWEN_VLLM_PORT=8000
# ============================================================================


cd "$(dirname "$0")"

echo "============================================"
echo " ARK Distillation Pipeline"
echo " Graph:   ${GRAPH_NAME}"
echo " Teacher: ${TEACHER_MODEL}"
echo " Student: ${STUDENT_MODEL}"
echo " Mode:    ${TEACHER_MODE}"
echo "============================================"


# ── STEP 0: Install ─────────────────────────────────────────────────────────
echo ""
echo "[Step 0] Installing dependencies..."
if command -v uv &>/dev/null; then
    uv sync
else
    echo "ERROR: uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi


# ── STEP 1: Generate teacher trajectories (train split) ────────────────────
echo ""
echo "[Step 1] Generating teacher trajectories on TRAIN split..."

case "${TEACHER_MODE}" in
    anthropic_proxy)
        export PROXY_URL="${PROXY_URL}"
        export RPM_LIMIT="${PROXY_RPM}"
        ;;
    openai_compat)
        export OPENAI_API_BASE="${OPENAI_BASE_URL}"
        export RPM_LIMIT="${OPENAI_RPM}"
        ;;
    vllm_qwen)
        export VLLM_PORT="${QWEN_VLLM_PORT}"
        ;;
    *)
        echo "ERROR: Unknown TEACHER_MODE=${TEACHER_MODE}"
        exit 1
        ;;
esac

# Train trajectories
uv run python main.py \
    --graph_name "${GRAPH_NAME}" \
    --model_name "${TEACHER_MODEL}" \
    --split train \
    --limit "${TRAIN_LIMIT}" \
    --number_of_agents "${NUMBER_OF_AGENTS}" \
    --max_steps "${MAX_STEPS}"

# Val trajectories
echo ""
echo "[Step 1b] Generating teacher trajectories on VAL split..."
uv run python main.py \
    --graph_name "${GRAPH_NAME}" \
    --model_name "${TEACHER_MODEL}" \
    --split val \
    --limit "${VAL_LIMIT}" \
    --number_of_agents "${NUMBER_OF_AGENTS}" \
    --max_steps "${MAX_STEPS}"


# ── STEP 2: Evaluate teacher quality ───────────────────────────────────────
echo ""
echo "[Step 2] Evaluating teacher trajectories..."

TEACHER_SHORT=$(echo "${TEACHER_MODEL}" | tr '/' '-' | awk -F'-' '{print $NF}')

echo "--- Train split ---"
uv run python eval.py \
    --graph_name "${GRAPH_NAME}" \
    --model_name "${TEACHER_MODEL}" \
    --split train

echo "--- Val split ---"
uv run python eval.py \
    --graph_name "${GRAPH_NAME}" \
    --model_name "${TEACHER_MODEL}" \
    --split val


# ── STEP 3: Fine-tune student with LoRA ───────────────────────────────────
echo ""
echo "[Step 3] Fine-tuning ${STUDENT_MODEL} on teacher trajectories..."

TEACHER_DIR_NAME="graph_explorer_${TEACHER_SHORT}"

cat > fine_tuning/params_run.yaml <<YAML
graph_name: "${GRAPH_NAME}"
model_name: "${STUDENT_MODEL}"
train_trajectories_dir: "data/experiments/${GRAPH_NAME}/${TEACHER_DIR_NAME}/train"
val_trajectories_dir: "data/experiments/${GRAPH_NAME}/${TEACHER_DIR_NAME}/val"
output_dir: "data/finetuning"
train_queries_limit: ${TRAIN_LIMIT}
val_queries_limit: ${VAL_LIMIT}
wandb_project: "graph-fine-tuning"
use_qlora: false
rejection_sampling: true

lora:
  r: 32
  lora_alpha: 64
  lora_dropout: 0.1
  bias: "none"
  task_type: "CAUSAL_LM"
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

training:
  max_length: 8192
  packing: false
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 2
  eval_accumulation_steps: 4
  num_train_epochs: 3
  learning_rate: 0.0001
  warmup_steps: 50
  weight_decay: 0.01
  logging_steps: 10
  eval_steps: 200
  save_steps: 500
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  greater_is_better: false
  optim: "paged_adamw_32bit"
  max_grad_norm: 1.0
  assistant_only_loss: true
YAML

uv run python finetune.py \
    --graph_name "${GRAPH_NAME}" \
    --model_name "${STUDENT_MODEL}"


# ── STEP 4: Serve fine-tuned student via vLLM ─────────────────────────────
echo ""
echo "[Step 4] Serving fine-tuned student model..."

STUDENT_SHORT=$(echo "${STUDENT_MODEL}" | awk -F'/' '{print $NF}')
ADAPTER_DIR="data/finetuning/${GRAPH_NAME}_${STUDENT_SHORT}"

if [ ! -d "${ADAPTER_DIR}" ]; then
    echo "Looking for adapter directory..."
    ADAPTER_DIR=$(find data/finetuning -maxdepth 1 -type d -name "*${GRAPH_NAME}*" | head -1)
fi

echo "Adapter: ${ADAPTER_DIR}"

vllm serve "${STUDENT_MODEL}" \
    --port "${STUDENT_VLLM_PORT}" \
    --enable-lora \
    --lora-modules "graphagent=${ADAPTER_DIR}" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --enforce-eager \
    --max-model-len 8192 &

VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"

echo "Waiting for vLLM to be ready..."
for i in $(seq 1 300); do
    if curl -s "http://localhost:${STUDENT_VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        echo "vLLM is ready!"
        break
    fi
    sleep 2
done


# ── STEP 5: Run student inference on test set ─────────────────────────────
echo ""
echo "[Step 5] Running student inference on TEST split..."

export VLLM_PORT="${STUDENT_VLLM_PORT}"

uv run python main.py \
    --graph_name "${GRAPH_NAME}" \
    --model_name "graphagent" \
    --split test \
    --number_of_agents "${NUMBER_OF_AGENTS}" \
    --max_steps "${MAX_STEPS}"


# ── STEP 6: Evaluate student ──────────────────────────────────────────────
echo ""
echo "[Step 6] Evaluating student model..."

uv run python eval.py \
    --graph_name "${GRAPH_NAME}" \
    --model_name "graphagent" \
    --split test


# ── Cleanup ───────────────────────────────────────────────────────────────
echo ""
echo "Stopping vLLM server..."
kill "${VLLM_PID}" 2>/dev/null || true

echo ""
echo "============================================"
echo " Pipeline complete!"
echo " Teacher trajectories: data/experiments/${GRAPH_NAME}/${TEACHER_DIR_NAME}/"
echo " Fine-tuned adapter:   ${ADAPTER_DIR}"
echo " Student results:      data/experiments/${GRAPH_NAME}/graph_explorer_graphagent/test/"
echo "============================================"
