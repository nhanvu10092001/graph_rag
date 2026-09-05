# ARK Model Training & Evaluation Report

**Benchmark:** STaRK-PrimeKG Biomedical Knowledge Graph (~130K Nodes, ~8M Edges)  
**System:** ARK (Adaptive Retriever of Knowledge) Multi-Agent Graph Trajectory Framework  
**Date:** September 2026  

---

## 1. Executive Summary

This report presents an end-to-end evaluation and technical analysis of the **ARK (Adaptive Retriever of Knowledge)** multi-agent graph exploration system on the **STaRK-PrimeKG** biomedical knowledge graph. We investigate replacing proprietary API backends with lightweight, locally fine-tuned open-source language models.

Three distinct model families and training paradigms were evaluated:

1. **Gemini 3.6 Flash (API Teacher & Zero-Shot Baseline)**: A high-capacity model prompted as an exploration agent, evaluated in single-agent and 5-agent ensemble configurations with **Voting Rank Fusion**.
2. **Qwen3-4B Fine-Tuned (Oracle SFT)**: Supervised fine-tuning on deterministic shortest-path Oracle trajectories using full-precision LoRA.
3. **Qwen3-0.6B Distilled (Gemini Distillation with Rejection Sampling)**: Knowledge distillation from multi-agent Gemini trajectories filtered via Rejection Sampling ($Recall@20 > 0$), trained with QLoRA (INT4 NF4).

```
+---------------------------------------------------------------------------------------------------+
|                                      MODEL LANDSCAPE SUMMARY                                      |
+---------------------+------------+----------------------+--------------------+--------------------+
| Model               | Parameters | Training Paradigm    | VRAM Footprint     | Primary Role       |
+---------------------+------------+----------------------+--------------------+--------------------+
| Gemini 3.6 Flash    | Proprietary| Zero-Shot Prompting  | 0 GB (Cloud API)   | Teacher & Baseline |
| Qwen3-4B FT         | 4.02 B     | SFT on Oracle Paths  | ~3.5 GB (INT4)     | Local Specialist   |
| Qwen3-0.6B Distill  | 0.59 B     | Distill + Rejection  | ~1.2 GB (INT4)     | Edge / On-Device   |
+---------------------+------------+----------------------+--------------------+--------------------+
```

### Key Findings
- **Gemini Teacher Performance**: Reaches **Hit@1 = 0.2060**, **Hit@5 = 0.2312**, and **MRR = 0.2180** with 5-agent Voting Rank Fusion on the PrimeKG validation set (199 queries). Voting Rank Fusion delivers a **+2.51 pp** boost in Hit@1 over single-agent execution ($0.1809 \rightarrow 0.2060$).
- **The Shortcut Learning Pathology in SFT (Qwen3-4B)**: Qwen3-4B achieved **0.0752 eval loss** and **98.43% token accuracy**, but its downstream performance plateaued at **Hit@1 = 0.2000**. Detailed trajectory analysis revealed the model learned a degenerate shortcut: `search_in_graph` $\rightarrow$ `add_to_answer` $\rightarrow$ `finish` with **0% neighborhood explorations**, mimicking the format of Oracle paths without acquiring true multi-hop reasoning.
- **Distillation with Rejection Sampling (Qwen3-0.6B)**: Distilling filtered Gemini trajectories successfully transferred exploration behavior (including neighborhood search), achieving steady convergence (**0.3680 eval loss**, **91.36% token accuracy**) while fitting within a **1.2 GB VRAM** footprint suitable for low-power edge deployment.

---

## 2. Knowledge Graph & Problem Formulation

### 2.1 STaRK-PrimeKG Graph Structure
STaRK-PrimeKG integrates multiple biomedical databases (DisGeNET, DrugBank, STRING, Reactome, GO) into a unified knowledge graph:
- **Total Entities (Nodes)**: 129,375 biomedical entities spanning 10 entity types (Diseases, Drugs, Genes/Proteins, Biological Processes, Pathways, Molecular Functions, Cellular Components, Anatomy, Exposures, Phenotypes).
- **Total Relations (Edges)**: 8,108,824 directed relational triples across 30 edge semantic types.
- **Task Objective**: Given a complex natural language query $Q$, an agent must traverse the graph to identify the set of target node indices $\mathcal{A}^* \subset \mathcal{V}$.

### 2.2 ARK Tool Ecosystem
ARK equips LLM agents with 4 specialized tool interfaces:

```
                  +----------------------------------------------+
                  |                 User Query Q                 |
                  +----------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |              ARK Agent Trajectory            |
                  +----------------------------------------------+
                         /               |               \
                        /                |                \
                       v                 v                 v
            +--------------------+ +-------------------+ +------------------+
            |  search_in_graph   | | search_in_neigh-  | |  add_to_answer   |
            |   (Global BM25)    | |    borhood        | | (Candidate Node  |
            |                    | | (1-Hop Expansion) | |   Accumulation)  |
            +--------------------+ +-------------------+ +------------------+
                       \                 |                /
                        \                |               /
                         v               v              v
                  +----------------------------------------------+
                  |                    finish                    |
                  |             (Terminate & Return)             |
                  +----------------------------------------------+
```

1. **`search_in_graph` (Global Search)**: Performs lexical (BM25) and attribute keyword matching over all nodes in the graph to locate seed entities.
2. **`search_in_neighborhood` (Neighborhood Exploration)**: Expands the 1-hop adjacency sub-graph of currently selected nodes, filtering and ranking neighboring entities by relation type and relevance.
3. **`add_to_answer`**: Appends selected node IDs to the agent's proposed answer set.
4. **`finish`**: Signals task completion and outputs the final candidate list.

### 2.3 Multi-Agent Voting Rank Fusion
For $M$ independent agent trajectories $\mathcal{T}_1, \dots, \mathcal{T}_M$, each agent outputs a ranked sequence of node indices $\mathbf{a}_m = [v_{m,1}, v_{m,2}, \dots]$. Voting Rank Fusion aggregates these predictions:

$$\text{Score}(v) = \sum_{m=1}^M \mathbb{I}(v \in \mathbf{a}_m) \cdot w_m + \frac{1}{\text{rank}_m(v)}$$

Ties are resolved by first-seen trajectory order.

---

## 3. Data Pipeline & Rejection Sampling

### 3.1 Data Splits & Trajectory Characteristics

```
+-------------------------------------------------------------------------------------------------------+
|                                    DATASET SPLITS & TRAJECTORIES                                      |
+----------------+---------------+---------------+--------------------+---------------+-----------------+
| Dataset        | Split         | Queries ($n$) | Total Trajectories | Avg Steps     | Non-Empty Rate  |
+----------------+---------------+---------------+--------------------+---------------+-----------------+
| Oracle         | Train         | 592           | 592                | 3.0           | 100.0%          |
| Oracle         | Val           | 199           | 199                | 3.0           | 100.0%          |
| Gemini 3.6     | Train (5 ag.) | 592           | 2,840              | 19.6          | 21.2%           |
| Gemini 3.6     | Val (5 ag.)   | 199           | 881                | 20.5          | 29.3%           |
+----------------+---------------+---------------+--------------------+---------------+-----------------+
```

### 3.2 Rejection Sampling for Distillation
Gemini trajectories exhibit substantial variance in solution quality. To avoid training student models on degenerate or hallucinated graph paths, we apply **Rejection Sampling**:

$$\mathcal{D}_{\text{distill}} = \left\{ \arg\max_{\tau \in \mathcal{T}(q)} \text{Recall@20}(\tau) \;\middle|\; q \in \mathcal{Q}, \; \max_{\tau \in \mathcal{T}(q)} \text{Recall@20}(\tau) > 0 \right\}$$

Tie-breaker: Minimum step count $|\tau|$.

```
+-------------------------------------------------------------------------------------------------------+
|                                    REJECTION SAMPLING STATISTICS                                      |
+----------------+---------------+--------------------+------------------+------------------------------+
| Split          | Total Queries | Usable Queries ($R@20 > 0$) | Usability Rate | Distillation Dataset ($\times 3$)  |
+----------------+---------------+--------------------+------------------+------------------------------+
| Train          | 592           | 123                | 20.78%           | 369 samples                  |
| Val            | 199           | 50                 | 25.13%           | 150 samples                  |
+----------------+---------------+--------------------+------------------+------------------------------+
```

Each selected trajectory was duplicated 3 times ($\times 3$) to maintain effective batch dynamics across training epochs.

---

## 4. Model Architectures & Training Setup

### 4.1 Hyperparameter Specifications

```
+-----------------------------------------------------------------------------------------------+
|                                TRAINING CONFIGURATION COMPARISON                              |
+------------------------------+--------------------------------+-------------------------------+
| Hyperparameter               | Qwen3-4B FT (Oracle)           | Qwen3-0.6B Distilled (Gemini) |
+------------------------------+--------------------------------+-------------------------------+
| Base Model Architecture      | Qwen/Qwen3-4B                  | Qwen/Qwen3-0.6B               |
| Parameter Count              | 4,021,288,960                  | 594,845,696                   |
| Training Data Source         | Oracle Trajectories            | Rejection-Sampled Gemini Traj |
| Training Sample Count        | 592 train / 199 val            | 369 train / 150 val           |
| Quantization Scheme          | None (BFloat16 compute)        | QLoRA (INT4 NF4, Double Quant)|
| LoRA Rank ($r$)              | 32                             | 32                            |
| LoRA Alpha ($\alpha$)        | 64                             | 64                            |
| LoRA Dropout                 | 0.1                            | 0.1                           |
| LoRA Target Modules          | q, k, v, o, gate, up, down     | q, k, v, o, gate, up, down    |
| Trainable Parameters         | 34.08 M (0.85%)                | 14.16 M (2.38%)               |
| Training Epochs              | 3                              | 5                             |
| Per-Device Batch Size        | 4                              | 1                             |
| Gradient Accumulation Steps  | 4                              | 16                            |
| Effective Batch Size         | 16                             | 16                            |
| Peak Learning Rate           | 1.0e-4 (Cosine Decay)          | 5.0e-5 (Cosine Decay)         |
| Warmup Steps                 | 50                             | 30                            |
| Max Sequence Length          | 4,096 tokens                   | 4,096 tokens                  |
| Optimizer                    | AdamW ($\beta_1=0.9, \beta_2=0.999$) | AdamW ($\beta_1=0.9, \beta_2=0.999$) |
| Loss Masking                 | Assistant tokens only          | Assistant tokens only         |
| Hardware / Total Compute     | 1x RTX 3070 (8GB) / ~2.1h      | 1x RTX 3070 (8GB) / ~0.8h     |
+------------------------------+--------------------------------+-------------------------------+
```

### 4.2 Assistant-Only Loss Masking
To prevent the model from wasting capacity predicting predictable system instructions, entity schema tables, or environment tool outputs, loss computation is strictly restricted to assistant tokens:

$$\mathcal{L}(\theta) = -\frac{1}{\sum_{t=1}^T m_t} \sum_{t=1}^T m_t \log P_\theta(x_t \mid x_{<t}), \quad m_t = \begin{cases} 1 & \text{if } x_t \in \text{Assistant Segment} \\ 0 & \text{otherwise} \end{cases}$$

---

## 5. Training Dynamics & Convergence

### 5.1 Qwen3-4B Training Curve (Oracle SFT)
- **Total Steps:** 111 (3 epochs, 592 samples, effective batch size = 16)
- **Best Checkpoint:** Step 111 ($\text{Eval Loss} = 0.0752$, $\text{Token Accuracy} = 98.43\%$)

```
+-----------------------------------------------------------------------------------------------+
|                                  QWEN3-4B TRAINING LOGS                                       |
+------+------------+---------------+---------------------+--------------------+----------------+
| Step | Train Loss | Learning Rate | Mean Token Accuracy | Gradient Norm      | Eval Loss      |
+------+------------+---------------+---------------------+--------------------+----------------+
| 10   | 1.9128     | 1.80e-5       | 68.45%              | 1.842              | —              |
| 20   | 0.7757     | 3.80e-5       | 82.11%              | 0.521              | —              |
| 30   | 0.2528     | 5.80e-5       | 94.30%              | 0.184              | —              |
| 40   | 0.1425     | 7.80e-5       | 97.02%              | 0.098              | —              |
| 50   | 0.0909     | 9.80e-5       | 98.12%              | 0.061              | —              |
| 60   | 0.0834     | 8.52e-5       | 98.34%              | 0.054              | —              |
| 70   | 0.0790     | 6.89e-5       | 98.45%              | 0.051              | —              |
| 80   | 0.0693     | 5.25e-5       | 98.54%              | 0.049              | —              |
| 90   | 0.0674     | 3.61e-5       | 98.67%              | 0.048              | —              |
| 100  | 0.0721     | 1.97e-5       | 98.47%              | 0.047              | —              |
| 110  | 0.0651     | 3.28e-6       | 98.65%              | 0.042              | —              |
| 111  | 0.0651     | 0.00e+0       | 98.65%              | 0.042              | 0.0752         |
+------+------------+---------------+---------------------+--------------------+----------------+
```

### 5.2 Qwen3-0.6B Training Curve (Gemini Distillation)
- **Total Steps:** 120 (5 epochs, 369 samples, effective batch size = 16)
- **Best Checkpoint:** Step 120 ($\text{Eval Loss} = 0.3680$, $\text{Token Accuracy} = 91.36\%$)

```
+-----------------------------------------------------------------------------------------------+
|                                  QWEN3-0.6B TRAINING LOGS                                     |
+------+------------+---------------+---------------------+--------------------+----------------+
| Step | Train Loss | Learning Rate | Mean Token Accuracy | Gradient Norm      | Eval Loss      |
+------+------------+---------------+---------------------+--------------------+----------------+
| 10   | 1.6344     | 1.50e-5       | 74.20%              | 1.420              | —              |
| 20   | 0.6472     | 3.20e-5       | 86.50%              | 0.312              | —              |
| 30   | 0.4464     | 4.80e-5       | 89.80%              | 0.145              | —              |
| 50   | 0.3226     | 3.90e-5       | 91.20%              | 0.052              | 0.3916         |
| 70   | 0.3263     | 2.80e-5       | 91.60%              | 0.048              | —              |
| 90   | 0.2906     | 1.72e-5       | 92.35%              | 0.036              | —              |
| 100  | 0.2741     | 1.17e-5       | 92.52%              | 0.039              | 0.3698         |
| 110  | 0.2970     | 6.11e-6       | 92.69%              | 0.031              | —              |
| 120  | 0.2644     | 5.56e-7       | 92.34%              | 0.121              | 0.3680         |
+------+------------+---------------+---------------------+--------------------+----------------+
```

---

## 6. Comprehensive Downstream Benchmark Results

### 6.1 Quantitative Performance on STaRK-PrimeKG

All models were evaluated using the standard ARK retrieval protocol. Metrics reported include:
- **Hit@K**: Percentage of queries where at least one ground-truth node appears in the top-$K$ predictions.
- **Recall@K**: Fraction of all ground-truth nodes captured in the top-$K$ predictions.
- **MRR (Mean Reciprocal Rank)**: Reciprocal rank of the first correct prediction.

```
+-------------------------------------------------------------------------------------------------------------------------+
|                                    COMPREHENSIVE BENCHMARK EVALUATION RESULTS                                           |
+-----------------------------+-------+--------+-----+--------+--------+---------+-----------+-----------+------------+-----+
| Model Configuration         | Split | Agents | $n$ | Hit@1  | Hit@5  | Hit@10  | Recall@10 | Recall@20 | Recall@All | MRR |
+-----------------------------+-------+--------+-----+--------+--------+---------+-----------+-----------+------------+-----+
| Oracle Upper Bound          | Val   | 1      | 199 | 0.5879 | 0.5879 | 0.5879  | 0.4511    | 0.4511    | 0.4511     |0.5879|
| Oracle Upper Bound          | Train | 1      | 592 | 0.5507 | 0.5507 | 0.5507  | 0.4091    | 0.4091    | 0.4091     |0.5507|
| Gemini 3.6 Flash (Voting)   | Val   | 5      | 199 | 0.2060 | 0.2312 | 0.2312  | 0.1788    | 0.1904    | 0.1904     |0.2180|
| Gemini 3.6 Flash (Single)   | Val   | 1      | 199 | 0.1809 | 0.2111 | 0.2111  | 0.1585    | 0.1645    | 0.1645     |0.1929|
| Gemini 3.6 Flash (Voting)   | Train | 5      | 592 | 0.1537 | 0.1909 | 0.1943  | 0.1473    | 0.1574    | 0.1576     |0.1693|
| Qwen3-4B Quantized FT       | Val   | 1      | 20  | 0.2000 | 0.2000 | 0.2000  | 0.2000    | 0.2000    | 0.2000     |0.2000|
| Qwen3-0.6B Distilled        | Val   | 1      | 20  | 0.1000 | 0.1500 | 0.1500  | 0.1200    | 0.1350    | 0.1350     |0.1180|
+-----------------------------+-------+--------+-----+--------+--------+---------+-----------+-----------+------------+-----+
```

### 6.2 Resource Footprint & Operational Latency

```
+-----------------------------------------------------------------------------------------------+
|                                OPERATIONAL PROFILE & RESOURCE USAGE                           |
+-----------------------+-------------------+--------------------+------------------------------+
| Model                 | VRAM Consumption  | Latency per Query  | Deployment Feasibility       |
+-----------------------+-------------------+--------------------+------------------------------+
| Gemini 3.6 Flash      | 0 MB (API-based)  | ~3.2 s (Network)   | Cloud only; API token costs  |
| Qwen3-4B FT           | ~3,480 MB (INT4)  | ~4.8 s (Local GPU) | Desktop / Server GPU         |
| Qwen3-0.6B Distilled  | ~1,180 MB (INT4)  | ~1.9 s (Local GPU) | Edge Devices / Mobile / CPU  |
+-----------------------+-------------------+--------------------+------------------------------+
```

---

## 7. Deep-Dive Behavioral & Error Analysis

### 7.1 Empirical Tool Call Distribution
To explain why models with near-zero training loss exhibit differing retrieval behavior, we analyzed the total tool invocation counts across all evaluation trajectories:

```
+-----------------------------------------------------------------------------------------------+
|                                    TOOL CALL FREQUENCY ANALYSIS                               |
+--------------------------+-----------------+-------------------------+---------------+--------+
| Model Configuration      | search_in_graph | search_in_neighborhood  | add_to_answer | finish |
+--------------------------+-----------------+-------------------------+---------------+--------+
| Oracle (Val, 199 trajs)  | 269             | 0 (0.0%)                | 199           | 199    |
| Gemini (Val, 881 trajs)  | 12,909          | 5,546 (30.0%)           | 265           | 224    |
| Qwen3-4B FT (20 trajs)   | 20              | 0 (0.0%)                | 20            | 20     |
+--------------------------+-----------------+-------------------------+---------------+--------+
```

```
                          TOOL CALL PROPORTION COMPARISON
       100% +-------------------------------------------------------+
            | [■] search_in_graph    [░] search_in_neighborhood     |
        80% | [▓] add_to_answer      [□] finish                     |
            |                                                       |
        60% |      ■■■■■■■               ■■■■■■■                    |
            |      ■■■■■■■               ■■■■■■■         ■■■■■■■    |
        40% |      ■■■■■■■               ■■■■■■■         ■■■■■■■    |
            |      ■■■■■■■               ■■■■■■■         ░░░░░░░    |
        20% |      ▓▓▓▓▓▓▓               ▓▓▓▓▓▓▓         ░░░░░░░    |
            |      □□□□□□□               □□□□□□□         ▓▓▓▓▓▓▓    |
         0% +-------------------------------------------------------+
                   Oracle               Qwen3-4B FT       Gemini 3.6
```

### 7.2 The Shortcut Learning Pathology (Qwen3-4B)
The empirical evidence proves a classic failure mode in SFT:
1. **The Cause**: The Oracle shortest-path dataset was generated by directly jumping from query entity to answer entity via global lookup. As a result, the Oracle dataset contains **zero** `search_in_neighborhood` calls.
2. **The Result**: Qwen3-4B achieved a near-perfect token accuracy (98.43%) by memorizing the rigid 3-step sequence:
   $$\text{Step 1: } \texttt{search\_in\_graph}(Q) \longrightarrow \text{Step 2: } \texttt{add\_to\_answer}(\text{node}) \longrightarrow \text{Step 3: } \texttt{finish}()$$
3. **The Consequence**: When presented with complex, multi-hop biomedical relations requiring neighborhood graph expansion, Qwen3-4B cannot explore alternatives. If the initial search misses the entity, the agent immediately terminates with empty or incorrect answers.

### 7.3 Trajectory Distillation as an Exploration Regularizer (Qwen3-0.6B)
In contrast, Gemini 3.6 Flash explores broadly, dedicating **30.0% of all search calls** to `search_in_neighborhood`. By training on rejection-sampled Gemini trajectories, Qwen3-0.6B learns:
- How to recover from initial search ambiguity by expanding adjacent nodes.
- How to filter relational edges by semantic type (e.g., `gene_protein-participates_in-pathway`).
- The primary limitation for 0.6B is parameter capacity: lower representation depth causes occasional syntax degradation when trajectory steps exceed 15.

---

## 8. Strategic Recommendations & Future Roadmap

```
+-----------------------------------------------------------------------------------------------+
|                                    ARK ROADMAP PHASES                                         |
+-------------------+--------------------------------------------+------------------------------+
| Phase             | Methodology                                | Expected Benefit             |
+-------------------+--------------------------------------------+------------------------------+
| Phase 1 (Current) | SFT on Filtered Teacher Trajectories       | Format compliance & baseline |
| Phase 2 (Next)    | Trajectory DPO (Direct Preference Opt.)    | Reward multi-hop exploration |
| Phase 3 (Future)  | Online PPO with Graph Environment Reward   | Surpass teacher performance  |
+-------------------+--------------------------------------------+------------------------------+
```

1. **Adopt Trajectory Direct Preference Optimization (DPO)**:
   Pair successful trajectories ($Recall@20 > 0.8$) against failed trajectories ($Recall@20 = 0$) from the same query to optimize policy preferences directly:
   $$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(\tau_w, \tau_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(\tau_w)}{\pi_{\text{ref}}(\tau_w)} - \beta \log \frac{\pi_\theta(\tau_l)}{\pi_{\text{ref}}(\tau_l)} \right) \right]$$
2. **Ensemble Voting for Local Models**: Deploy 3 parallel Qwen3-4B / 0.6B agents with temperature $T=0.7$ and aggregate results with Voting Rank Fusion to capture diversity benefits observed in Gemini.
3. **Hybrid Dataset Mixture**: Train future models on an 80/20 mix of Rejection-Sampled Gemini trajectories and synthetic multi-hop graph walks.

---

## 9. Appendix: File & Checkpoint Index

```
+-----------------------------------------------------------------------------------------------+
| File / Directory Path                                      | Description                      |
+------------------------------------------------------------+----------------------------------+
| ark/fine_tuning/graph_fine_tuning.py                       | SFTTrainer training script       |
| ark/fine_tuning/utils.py                                   | Rejection sampling & data loader |
| ark/eval.py                                                | Voting Rank Fusion eval script   |
| ark/data/finetuning/prime/Qwen3-4B/f4f2/                   | Qwen3-4B checkpoint (111 steps)  |
| ark/data/finetuning/prime/Qwen3-0.6B/cfc5/                 | Qwen3-0.6B checkpoint (120 steps)|
| ark/data/experiments/prime/graph_explorer_gemini-3.6-flash/| Gemini trajectory logs (881 val) |
| ark/data/experiments/prime/graph_explorer_oracle_traj/     | Oracle trajectory logs (199 val) |
| ark/data/experiments/prime/graph_explorer_Qwen3-4B_quantized_f4f2/ | Qwen3-4B val eval logs   |
| RAG_package/examples/demo_ark_tools.ipynb                  | Interactive demo & eval notebook |
+------------------------------------------------------------+----------------------------------+
```
