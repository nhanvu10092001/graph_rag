<!-- ARK: Autonomous Knowledge Graph Exploration with Adaptive Breadth-Depth Retrieval -->

[![Paper](https://img.shields.io/badge/Paper-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2601.13969)
[![Code](https://img.shields.io/badge/Code-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mims-harvard/ark)

## Introduction

Retrieving evidence for language model queries from knowledge graphs requires balancing broad search across the graph with multi-hop traversal to follow relational links. Similarity-based retrievers provide coverage but remain shallow, whereas traversal-based methods rely on selecting seed nodes to start exploration, which can fail when queries span multiple entities and relations.

We introduce **ARK: Adaptive Retriever of Knowledge**, an agentic KG retriever that gives a language model control over this breadth-depth tradeoff using a two-operation toolset:

- **Global Search**: Lexical search (BM25) over node descriptors for broad discovery
- **Neighborhood Exploration**: One-hop expansion that composes into multi-hop traversal

ARK alternates between breadth-oriented discovery and depth-oriented expansion without depending on fragile seed selection, a pre-set hop depth, or requiring retrieval training. ARK adapts tool use to queries, using global search for language-heavy queries and neighborhood exploration for relation-heavy queries.

**Key Results on STaRK Benchmark:**

- **59.1%** average Hit@1 and **67.4** average MRR
- Improves average Hit@1 by up to **31.4%** and average MRR by up to **28.0%** over retrieval-based and agentic training-free methods
- Distilled 8B model retains up to **98.5%** of the teacher's Hit@1 rate via label-free imitation

<p align="center">
<img src="data/images/figure-1.png?raw=true" width="100%" title="ARK interacts with a KG through a minimal two-tool interface for adaptive retrieval.">
</p>

## Benchmark

ARK is evaluated on [STaRK](https://github.com/snap-stanford/stark), a benchmark for entity-level retrieval over heterogeneous, text-rich knowledge graphs ([Wu et al., 2024](https://arxiv.org/abs/2404.13207)).

STaRK comprises three large, heterogeneous knowledge graphs:

| Dataset | Domain | Entities | Relations | Avg. Degree |
|---------|--------|----------|-----------|-------------|
| **AMAZON** | E-commerce | ~1M | ~9.4M | 18.2 |
| **MAG** | Academic | ~1.9M | ~39.8M | 43.5 |
| **PRIME** | Biomedical | ~129K | ~8.1M | 125.2 |

Each node is associated with text-rich attributes, making STaRK a natural testbed for hybrid retrieval over structured and textual signals.

## Usage Instructions

### 1. Clone and Install

Clone this repository and set up your environment:

```bash
git clone https://github.com/mims-harvard/ark.git
cd ark
```

Install dependencies using [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync
```

For running local models with VLLM, install it separately:

```bash
uv pip install vllm --torch-backend=auto
```

### 2. Download STaRK Data

Download the STaRK benchmark data from the [official repository](https://github.com/snap-stanford/stark):

```bash
# Clone STaRK repository to get the raw data
git clone https://github.com/snap-stanford/stark.git

# Follow STaRK instructions to download the knowledge graphs
# The data should be placed in benchmarks/stark/data/raw_graphs/
```

For detailed instructions on downloading STaRK data, please refer to the [STaRK paper](https://arxiv.org/abs/2404.13207) and repository.

### 3. Preprocess Data

Convert the raw graph data to parquet format for efficient loading:

```bash
cd benchmarks/stark/preprocessing

# Preprocess each graph
python amazon_to_parquet.py
python mag_to_parquet.py
python prime_to_parquet.py
```

This will create parquet files in `benchmarks/stark/data/graphs/{graph_name}/`.

### 4. Configure Environment

Create a `.env` file in the project root with your API keys:

```bash
# For Azure OpenAI (GPT-4.1)
AZURE_API_KEY=your_azure_api_key
AZURE_API_BASE=your_azure_endpoint

# For OpenAI
OPENAI_API_KEY=your_openai_api_key
```

### 5. Run Experiments

Navigate to the STaRK benchmark directory and create symlinks:

```bash
cd benchmarks/stark
ln -s ../../src src
```

Run ARK on a specific graph:

```bash
# Run on PRIME with GPT-4.1 (default: 3 parallel agents)
python main.py --graph_name prime --model_name azure/gpt-4.1 --split test

# Run on MAG
python main.py --graph_name mag --model_name azure/gpt-4.1 --split test

# Run on AMAZON
python main.py --graph_name amazon --model_name azure/gpt-4.1 --split test
```

**Available arguments:**

- `--graph_name`: Graph to evaluate on (`prime`, `mag`, `amazon`)
- `--model_name`: Model to use (`azure/gpt-4.1`, `Qwen/Qwen3-8B`, etc.)
- `--split`: Data split (`train`, `val`, `test`)
- `--number_of_agents`: Number of parallel agents (default: 3)
- `--limit`: Limit number of queries (for debugging)

### 6. Evaluate Results

After running experiments, evaluate the results:

```bash
python eval.py --graph_name prime --model_name azure/gpt-4.1 --split test
```

This will output metrics including Hit@1, Hit@5, Recall@10, Recall@20, and MRR.

### 7. Fine-tune Models (Optional)

ARK supports distillation of the retrieval policy into smaller models via label-free trajectory imitation.

#### Generate Training Trajectories

First, run ARK with the teacher model on training data:

```bash
python main.py --graph_name prime --model_name azure/gpt-4.1 --split train
python main.py --graph_name prime --model_name azure/gpt-4.1 --split val
```

#### Run Fine-tuning

Fine-tune a Qwen model on the collected trajectories:

```bash
python finetune.py --graph_name prime --model_name Qwen/Qwen3-8B --train_queries_limit 6000
```

Configure fine-tuning parameters in `fine_tuning/params.yaml`:

```yaml
graph_name: "prime"
model_name: "Qwen/Qwen3-8B"
train_queries_limit: 6000
val_queries_limit: 200

lora:
  r: 32
  lora_alpha: 64
  lora_dropout: 0.1

training:
  max_length: 16384
  num_train_epochs: 1
  learning_rate: 0.00001
```

#### Serve Fine-tuned Model

Start a VLLM server with the fine-tuned model:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model data/finetuning/prime/Qwen3-8B/explorer/merged \
  --served-model-name Qwen3-8B-graphagent \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Then run evaluation with the fine-tuned model:

```bash
python main.py --graph_name prime --model_name Qwen3-8B-graphagent --split test
```

## Citation

ARK is released under the MIT License. If you use ARK, please consider citing our paper:

```bibtex
@misc{polonuer2026autonomousknowledgegraphexploration,
      title={Autonomous Knowledge Graph Exploration with Adaptive Breadth-Depth Retrieval}, 
      author={Joaquín Polonuer and Lucas Vittor and Iñaki Arango and Ayush Noori and David A. Clifton and Luciano Del Corro and Marinka Zitnik},
      year={2026},
      eprint={2601.13969},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.13969}, 
}
```

## Contact

For any questions or feedback, please open an issue in the [GitHub repository](https://github.com/mims-harvard/ark/issues/new) or contact [Luciano Del Corro](mailto:delcorrol@udesa.edu.ar) and [Marinka Zitnik](mailto:marinka@hms.harvard.edu).
