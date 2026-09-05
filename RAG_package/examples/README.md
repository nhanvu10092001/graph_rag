# ARK (Adaptive Retriever of Knowledge) — Demo & Report

## Tổng quan

Thư mục này chứa notebook demo end-to-end cho pipeline **ARK** — multi-agent graph exploration system sử dụng LangGraph, cùng với report format ACL.

### Cấu trúc thư mục

```
examples/
├── demo_ark_tools.ipynb        # Notebook demo chính
├── demo_graph_rag.py           # Script demo Graph RAG cơ bản
├── qwen3_4b_finetuned/         # Symlink → model Qwen3-4B fine-tuned (merged, INT4)
├── qwen3_0.6b_distilled/       # Symlink → model Qwen3-0.6B distilled (LoRA adapter, INT4)
├── qwen3_4b_evaluation.png     # Biểu đồ evaluation Qwen3-4B
├── report_acl_format/           # Report LaTeX theo format ACL
│   ├── acl_paper.tex           # Source LaTeX chính
│   ├── acl.sty                 # ACL style file
│   ├── acl_natbib.bst          # Bibliography style
│   ├── references.bib          # References
│   ├── custom.bib              # Custom bibliography entries
│   └── report.pdf              # PDF đã compile
└── README.md                   # File này
```

---

## Hướng dẫn chạy `demo_ark_tools.ipynb`

### Prerequisites

#### 1. Neo4j Database

Notebook cần kết nối Neo4j để lưu trữ Knowledge Graph. Có 2 cách:

**Option A — Neo4j Aura (Cloud, miễn phí):**
- Tạo tài khoản tại [https://neo4j.com/cloud/aura-free/](https://neo4j.com/cloud/aura-free/)
- Tạo một Free Instance
- Lưu lại `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`

**Option B — Neo4j Local (Docker):**
```bash
# Từ thư mục root graph_rag/
docker compose up neo4j -d
# URI: bolt://localhost:7687
# Username: neo4j / Password: password123
```

#### 2. LLM Backend (chọn 1 trong 4)

| Option | Yêu cầu | Ghi chú |
|--------|----------|---------|
| `qwen3-4b` | GPU >= 4GB VRAM | Fine-tuned model, cần download weights |
| `qwen3-0.6b` | GPU >= 2GB VRAM | Distilled model, cần download weights |
| `gemini` | Gemini proxy server đang chạy | Dùng Anthropic Messages API proxy |
| `openai` | `OPENAI_API_KEY` | Đơn giản nhất, không cần GPU |

#### 3. Python Dependencies

```bash
# Cài từ requirements (nếu chạy local)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes peft
pip install langchain-core langgraph langchain-neo4j "neo4j>=5.15.0" pydantic
pip install langchain-huggingface langchain-anthropic langchain-openai
pip install nest-asyncio rank-bm25

# Hoặc dùng uv (từ thư mục RAG_package/)
cd RAG_package && uv sync
```

**Google Colab:** Uncomment cell đầu tiên trong notebook (Section 0) rồi chạy. Sau đó restart runtime.

---

### Chạy từng bước

#### Bước 1 — Mở notebook

**Local (Jupyter):**
```bash
cd RAG_package/examples
jupyter notebook demo_ark_tools.ipynb
# hoặc
jupyter lab demo_ark_tools.ipynb
```

**Google Colab:**
- Upload file `demo_ark_tools.ipynb` lên Colab
- Chọn Runtime → Change runtime type → **T4 GPU** (nếu dùng Qwen models)

**VS Code:**
- Mở file `demo_ark_tools.ipynb` trực tiếp, VS Code sẽ dùng Jupyter extension

#### Bước 2 — Cấu hình (Section 1)

Sửa các biến trong cell `Configuration & Model Selection`:

```python
# Chọn model
MODEL_CHOICE = "openai"  # hoặc "qwen3-4b", "qwen3-0.6b", "gemini"

# Nếu dùng OpenAI
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4o-mini"  # hoặc "gpt-4o"

# Nếu dùng Qwen (đảm bảo symlink/path đúng)
MODEL_PATHS = {
    "qwen3-4b": "./qwen3_4b_finetuned",
    "qwen3-0.6b": "./qwen3_0.6b_distilled",
}

# Neo4j connection
NEO4J_URI = "neo4j+s://your-instance.databases.neo4j.io"
NEO4J_USERNAME = "your-username"
NEO4J_PASSWORD = "your-password"
NEO4J_DATABASE = "your-database"

# ARK parameters
ARK_N_AGENTS = 3      # Số agents chạy song song
ARK_MAX_STEPS = 30    # Max steps mỗi agent
ARK_TEMPERATURE = 0.7 # Temperature cho stochastic exploration
ARK_TOP_K_FUSED = 10  # Top-K nodes sau Voting Rank Fusion
```

#### Bước 3 — Load Model (Section 2)

Chạy cell load model. Tùy `MODEL_CHOICE`:
- **OpenAI/Gemini**: Load nhanh (chỉ init API client)
- **Qwen3-4B**: ~30s, cần ~3.5GB VRAM (INT4 quantized)
- **Qwen3-0.6B**: ~15s, cần ~1.2GB VRAM (LoRA adapter + INT4)

#### Bước 4 — Kết nối Neo4j (Section 3)

Cell sẽ tạo `Neo4jGraphStore` và test connection. Output mong đợi:
```
Neo4j connected: [{'ok': 1}]
```

#### Bước 5 — Indexing (Section 4)

**Lần đầu:** Chạy cell indexing để nạp 36 sample facts vào Neo4j. Mất ~2-5 phút (tùy LLM speed).

**Lần sau:** Skip cell indexing nếu data đã có. Uncomment dòng `MATCH (n) DETACH DELETE n` nếu muốn reset.

Cell cuối sẽ tạo fulltext index và show schema.

#### Bước 6 — Test ARK Tools (Section 5)

Chạy từng cell để test 4 tools riêng lẻ:
- `global_search` — Tìm nodes theo keyword
- `neighborhood_exploration` — Duyệt neighbors, rank bằng BM25
- `add_to_answer` — Đánh dấu nodes là kết quả
- `finish` — Báo hiệu dừng

#### Bước 7 — ARK Agent (Section 6)

Test single agent trajectory:
```python
query = "What country is Alice's employer located in?"
result = await run_single_ark_agent(query, max_steps=15)
```

Xem trace để hiểu agent đã gọi tools nào, theo thứ tự nào.

#### Bước 8 — ARK Retriever (Section 7)

Chạy full pipeline multi-agent + Voting Rank Fusion:
```python
docs = await ark_retrieve(query, n_agents=3, max_steps=15, top_k=5)
```

3 agents chạy **song song**, kết quả được fuse bằng voting.

#### Bước 9 — End-to-End QA (Section 8)

ARK retrieve → LLM answer:
```python
answer = await ark_qa("Where is the headquarters of the partner company of Alice's employer?")
```

#### Bước 10 — Evaluation Report (Section 9)

Xem kết quả evaluation của Qwen3-4B fine-tuned và Qwen3-0.6B distilled trên STaRK-PrimeKG benchmark.

---

### Chuẩn bị model weights (nếu dùng Qwen)

#### Qwen3-4B Fine-tuned

Model đã được merge (LoRA → full model) và lưu tại:
```
ark/data/finetuning/prime/Qwen3-4B/f4f2/merged/
```

Symlink đã được tạo sẵn:
```bash
ls -la qwen3_4b_finetuned
# -> ../../ark/data/finetuning/prime/Qwen3-4B/f4f2/merged/
```

Nếu chưa có, tạo lại:
```bash
cd RAG_package/examples
ln -sf ../../ark/data/finetuning/prime/Qwen3-4B/f4f2/merged qwen3_4b_finetuned
```

#### Qwen3-0.6B Distilled

Model dùng LoRA adapter (chưa merge), load qua `AutoPeftModelForCausalLM`:
```
ark/data/finetuning/prime/Qwen3-0.6B/cfc5/
```

Symlink:
```bash
cd RAG_package/examples
ln -sf ../../ark/data/finetuning/prime/Qwen3-0.6B/cfc5 qwen3_0.6b_distilled
```

---

### Troubleshooting

| Vấn đề | Giải pháp |
|--------|-----------|
| `Neo4j connection failed` | Kiểm tra URI/credentials, đảm bảo Neo4j đang chạy |
| `CUDA out of memory` | Giảm `max_new_tokens`, dùng model nhỏ hơn (0.6B), hoặc chuyển sang OpenAI |
| `No nodes found for subquery` | Fulltext index chưa được tạo, chạy lại `ensure_fulltext_index()` |
| `IProgress not found` | `pip install ipywidgets` rồi restart kernel |
| `nest_asyncio` error | Đảm bảo `nest_asyncio.apply()` được gọi trước khi dùng `await` |
| Agent chạy quá lâu | Giảm `ARK_MAX_STEPS` xuống 10-15 |
| `bitsandbytes` error | Cần CUDA toolkit: `pip install bitsandbytes --prefer-binary` |

---

## Report ACL Format

Thư mục `report_acl_format/` chứa report LaTeX theo chuẩn ACL (Association for Computational Linguistics):

### Compile PDF

```bash
cd report_acl_format

# Compile với bibliography
pdflatex acl_paper.tex
bibtex acl_paper
pdflatex acl_paper.tex
pdflatex acl_paper.tex

# Hoặc dùng latexmk
latexmk -pdf acl_paper.tex
```

### Nội dung report

Report trình bày toàn bộ framework HLG LLM Utils, bao gồm:
- Property Knowledge Graph construction (Neo4j)
- Hierarchical Leiden community detection
- 4 retrieval strategies: Local, Global, Cypher Self-Correction, ARK
- ARK multi-agent parallel exploration với Voting Rank Fusion
- Qwen3-4B fine-tuning và Qwen3-0.6B distillation results
- Full-stack deployment architecture

### Sửa đổi report

- `acl_paper.tex` — Source chính, sửa nội dung ở đây
- `references.bib` — Thêm references mới vào file này
- `custom.bib` — References tùy chỉnh thêm
- `acl.sty` — Style file ACL (không cần sửa)
