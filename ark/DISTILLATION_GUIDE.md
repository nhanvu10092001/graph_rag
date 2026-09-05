# ARK Knowledge Distillation Pipeline

Teacher model (lớn) generate trajectories → Fine-tune student model (nhỏ) bằng LoRA SFT.

## Tổng quan

```
Teacher (Gemini / GPT-4.1 / bất kỳ LLM lớn)
    │
    │  Chạy agent trên mỗi câu hỏi → sinh trajectory (chuỗi tool calls)
    │  3 agents/question, chọn trajectory tốt nhất (rejection sampling)
    │
    ▼
Trajectory JSONs (data/experiments/<graph>/<model>/train/)
    │
    │  Dùng làm training data cho SFT
    │
    ▼
Student (Qwen 0.6B-8B) + LoRA fine-tuning
    │
    ▼
Serve qua vLLM → Evaluate trên test set
```

## Số liệu từ paper (ACL 2026)

| Dataset | Train queries | Val queries | Test queries |
|---------|--------------|-------------|-------------|
| PRIME   | 6,162        | 2,240       | 2,016       |
| MAG     | 7,993        | 2,664       | 2,664       |
| AMAZON  | 5,915        | 1,547       | 1,638       |

- **Teacher**: GPT-4.1, 3 agents/question, T_max=20
- **Student**: Qwen3-8B, LoRA r=32
- Variant `-600`: train trên 600 queries, variant `-6000`: train trên ~6000 queries

---

## Step 0: Cài đặt

```bash
cd ark
uv sync
```

Yêu cầu: Python ≥ 3.10, uv, GPU (fine-tuning + vLLM).

## Step 1: Generate teacher trajectories

### Cách kết nối teacher model

**Option A — Gemini qua Anthropic proxy (localhost:8080):**
```bash
PROXY_URL=http://localhost:8080 RPM_LIMIT=30 python main.py \
    --graph_name prime \
    --model_name gemini-3.6-flash-high \
    --split train \
    --number_of_agents 3 \
    --max_steps 10 \
    --limit 600
```

**Option B — Bất kỳ model nào qua OpenAI-compatible API (vLLM, Ollama, SGLang...):**
```bash
OPENAI_API_BASE=http://localhost:8000/v1 RPM_LIMIT=60 python main.py \
    --graph_name prime \
    --model_name my-model-name \
    --split train \
    --number_of_agents 3 \
    --max_steps 10 \
    --limit 600
```

**Option C — Qwen trên vLLM (có thinking support):**
```bash
VLLM_PORT=8000 python main.py \
    --graph_name prime \
    --model_name Qwen/Qwen3-32B \
    --split train \
    --number_of_agents 3 \
    --max_steps 10 \
    --limit 600
```

### Chạy thêm val split
```bash
# Cùng env vars như trên, đổi --split và --limit
... python main.py \
    --graph_name prime \
    --model_name <teacher> \
    --split val \
    --number_of_agents 3 \
    --max_steps 10 \
    --limit 200
```

### Ước tính thời gian

| max_steps | Calls/question | Thời gian/question (RPM=30) | 600 questions |
|-----------|---------------|----------------------------|---------------|
| 30        | ~90           | ~3 phút                    | ~30 giờ       |
| 20        | ~60           | ~2 phút                    | ~20 giờ       |
| **10**    | **~30**       | **~1 phút**                | **~10 giờ**   |

> Paper cho thấy 3 agents × T_max=10 **tốt hơn** 1 agent × T_max=30 (Table 8).

### Lưu ý
- Script auto-skip question đã process → stop rồi chạy lại an toàn.
- Output lưu tại: `data/experiments/<graph>/graph_explorer_<model>/train/`
- Mỗi question = 1 file JSON chứa trajectories của cả 3 agents.
- Không cần embedding model — mặc định dùng BM25 search.

## Step 2: Evaluate teacher trajectories

```bash
uv run python eval.py \
    --graph_name prime \
    --model_name gemini-3.6-flash-high \
    --split train
```

Output: Hit@1, Hit@5, Recall@10, Recall@20, MRR.

> Nếu Recall@20 < 30% → trajectories chất lượng kém, fine-tune sẽ không hiệu quả.

## Step 3: Fine-tune student model

### Chuẩn bị config

Sửa `fine_tuning/params.yaml`:

```yaml
graph_name: "prime"
model_name: "Qwen/Qwen3-0.6B"          # model student
train_trajectories_dir: "data/experiments/prime/graph_explorer_gemini-3.6-flash-high/train"
val_trajectories_dir: "data/experiments/prime/graph_explorer_gemini-3.6-flash-high/val"
output_dir: "data/finetuning"
train_queries_limit: 600
val_queries_limit: 200
rejection_sampling: true                 # chọn trajectory best recall@20 từ 3 agents

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
  max_length: 8192                       # giảm nếu thiếu VRAM
  per_device_train_batch_size: 4         # tăng được vì model nhỏ
  gradient_accumulation_steps: 2
  num_train_epochs: 3
  learning_rate: 0.0001                  # 1e-4
  warmup_steps: 50
  eval_steps: 200
  save_steps: 500
  load_best_model_at_end: true
  assistant_only_loss: true              # chỉ tính loss trên phần assistant response
```

### Chạy fine-tuning

```bash
uv run python finetune.py \
    --graph_name prime \
    --model_name "Qwen/Qwen3-0.6B"
```

Output: LoRA adapter lưu tại `data/finetuning/`.

### Gợi ý VRAM

| Student model | Batch size | max_length | VRAM ước tính |
|---------------|-----------|------------|---------------|
| Qwen3-0.6B   | 4         | 8192       | ~8 GB         |
| Qwen3-1.7B   | 2         | 8192       | ~16 GB        |
| Qwen3-8B     | 1         | 16384      | ~40 GB        |

## Step 4: Serve fine-tuned model qua vLLM

```bash
vllm serve Qwen/Qwen3-0.6B \
    --port 8000 \
    --enable-lora \
    --lora-modules graphagent=data/finetuning/<adapter_folder> \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --enforce-eager \
    --max-model-len 8192
```

Kiểm tra server ready:
```bash
curl http://localhost:8000/v1/models
```

## Step 5: Run student inference trên test set

```bash
VLLM_PORT=8000 uv run python main.py \
    --graph_name prime \
    --model_name graphagent \
    --split test \
    --number_of_agents 3 \
    --max_steps 10
```

## Step 6: Evaluate student

```bash
uv run python eval.py \
    --graph_name prime \
    --model_name graphagent \
    --split test
```

So sánh với teacher metrics để đánh giá hiệu quả distillation.

---

## Script tự động

Toàn bộ pipeline đã được gói trong `run_distillation.sh`. Sửa phần CONFIG rồi chạy:

```bash
cd ark
./run_distillation.sh
```

## Cấu trúc output

```
ark/data/
├── experiments/
│   └── prime/
│       ├── graph_explorer_gemini-3.6-flash-high/   # teacher trajectories
│       │   ├── train/
│       │   │   ├── 9117.json
│       │   │   ├── 7370.json
│       │   │   └── ...
│       │   └── val/
│       └── graph_explorer_graphagent/               # student results
│           └── test/
└── finetuning/                                      # LoRA adapter
    └── prime_Qwen3-0.6B/
```

## Tools trong ARK Agent

| Tool | Chức năng |
|------|-----------|
| `search_in_graph` | BM25 keyword search trên toàn graph, trả về top-k nodes |
| `search_in_neighborhood` | Mở rộng k-hop từ 1 node, trả về neighbors |
| `add_to_answer` | Thêm node vào danh sách answer |
| `finish` | Kết thúc trajectory |
