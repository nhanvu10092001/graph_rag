#!/usr/bin/env bash
# ============================================================================
# Fine-tune (oracle trajectories - direct, no teacher distillation) a student
# model of any size, reusing the oracle dataset already built by
# scripts/build_oracle_trajectories.py.
#
# Usage:
#   ./scripts/tune_model.sh <HF_MODEL_ID> [--graph <name>] [--gpu <idx>]
#
# Examples:
#   ./scripts/tune_model.sh Qwen/Qwen3-0.6B
#   ./scripts/tune_model.sh Qwen/Qwen3-0.6B --gpu 3
#
# Batch / max_length are chosen from model size. GPU is auto-picked as the one
# with most free memory unless --gpu is given or CUDA_VISIBLE_DEVICES is set.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:?usage: $0 <HF_MODEL_ID> [--graph name] [--gpu idx]}"
shift || true
GRAPH="prime"
GPU=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --graph) GRAPH="$2"; shift 2 ;;
        --gpu)   GPU="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# --- Infer model size and hyper-params --------------------------------------
SIZE_STR=$(basename "$MODEL")
case "$SIZE_STR" in
    *0.6B*|*1.7B*)   BATCH=16; ACCUM=4; MAX_LEN=4096 ;;
    *4B*)            BATCH=4;  ACCUM=4; MAX_LEN=4096 ;;
    *8B*)            BATCH=1;  ACCUM=4; MAX_LEN=8192 ;;
    *)               BATCH=1;  ACCUM=4; MAX_LEN=4096 ;;
esac

# --- Pick GPU (explicit > CUDA_VISIBLE_DEVICES > freest) ----------------------
if [[ -z "$GPU" && -z "${CUDA_VISIBLE_DEVICES:-}" ]] && command -v nvidia-smi >/dev/null; then
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2 -nr | head -1 | cut -d, -f1)
    echo "Auto-picked GPU $GPU (most free memory). Override with --gpu or CUDA_VISIBLE_DEVICES."
fi
if [[ -n "$GPU" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

echo "============================================"
echo " Tune model:   ${MODEL}"
echo " Graph:        ${GRAPH}"
echo " Oracle data:  data/experiments/${GRAPH}/graph_explorer_oracle_traj/{train,val}"
echo " GPU:          ${CUDA_VISIBLE_DEVICES:-auto}"
echo " Batch/Accum:  ${BATCH}/${ACCUM}  max_length=${MAX_LEN}"
echo "============================================"

# --- Write params.yaml --------------------------------------------------------
cat > fine_tuning/params.yaml <<YAML
graph_name: "${GRAPH}"
model_name: "${MODEL}"
train_trajectories_dir: "data/experiments/${GRAPH}/graph_explorer_oracle_traj/train"
val_trajectories_dir: "data/experiments/${GRAPH}/graph_explorer_oracle_traj/val"
output_dir: "data/finetuning"
train_queries_limit: 600
val_queries_limit: 200
wandb_project: "graph-fine-tuning"
use_qlora: false
rejection_sampling: false

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
  max_length: ${MAX_LEN}
  packing: false
  per_device_train_batch_size: ${BATCH}
  per_device_eval_batch_size: ${BATCH}
  gradient_accumulation_steps: ${ACCUM}
  eval_accumulation_steps: 4
  num_train_epochs: 3
  learning_rate: 0.0001
  warmup_steps: 50
  weight_decay: 0.01
  logging_steps: 10
  eval_steps: 300
  save_steps: 1500
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  greater_is_better: false
  optim: "adamw_torch"
  max_grad_norm: 1.0
  assistant_only_loss: true
YAML

echo "[Step] Running fine-tuning for ${MODEL}..."
WANDB_MODE=offline WANDB_DISABLED=true uv run python finetune.py \
    --graph_name "${GRAPH}" \
    --model_name "${MODEL}"

echo ""
echo "[Done] Fine-tuned adapter: find latest run under data/finetuning/${GRAPH}/$(basename "$MODEL")/"
