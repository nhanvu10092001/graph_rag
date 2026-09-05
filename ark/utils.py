import os
import yaml
import json
import pandas as pd
from src.models.LiteLLMModel import LiteLLMModel
from src.models.QwenModel import QwenModel
from src.models.VLLMQwenModel import VLLMQwenModel
from src.models.FineTunedQwenModel import FineTunedQwenModel
from src.models.AnthropicProxyModel import AnthropicProxyModel
from src.models.OpenAICompatibleModel import OpenAICompatibleModel
from src.core.graph import Graph
from src.agents.graph_explorer.graph_explorer import GraphExplorerAgent
from pydantic import BaseModel, model_validator
from typing import Literal, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def graph_data_dir(graph_name: str) -> Path:
    """Directory containing nodes.parquet and edges.parquet for this graph."""
    return Path("data/graphs") / graph_name


def qa_data_dir(graph_name: str) -> Path:
    """Directory containing stark_qa/ and split/ for this graph (STARK layout)."""
    return Path("data/qa") / graph_name


class GraphExplorerConfig(BaseModel):
    graph_name: str
    model_name: str
    finetune_path: Optional[str] = None
    quantized: bool = False
    enable_thinking: bool = False
    search_mode: Literal["bm25", "embeddings"] = "bm25"
    embedding_model: str = "azure/text-embedding-3-large"
    split: str = "test"
    limit: Optional[int] = None
    max_steps: int = 10
    number_of_agents: int = 1
    results_dir: Optional[str] = None

    @model_validator(mode="after")
    def validate_finetune_path(self):
        if self.finetune_path and self.graph_name not in self.finetune_path:
            raise ValueError(
                f"Finetune path '{self.finetune_path}' must include graph name '{self.graph_name}'"
            )
        return self


def setup_ablation_results_dir(graph_explorer_config, tool_to_remove: str):
    graph_name = graph_explorer_config.graph_name
    split = graph_explorer_config.split

    model_name = graph_explorer_config.model_name.split("/")[-1]
    experiment_name = f"graph_explorer_{model_name}_without_{tool_to_remove}"

    if graph_explorer_config.quantized:
        experiment_name += "_quantized"

    if graph_explorer_config.finetune_path:
        experiment_name += f"_{graph_explorer_config.finetune_path.split('/')[-1]}"

    if graph_explorer_config.enable_thinking:
        experiment_name += "_thinking"

    results_dir = Path(f"data/ablations/{graph_name}/{experiment_name}/{split}")

    results_dir.mkdir(parents=True, exist_ok=True)

    config_dict = graph_explorer_config.model_dump()

    config_file_path = results_dir / "config.yaml"
    with open(config_file_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)

    return results_dir


def setup_graph_explorer_results_dir(graph_explorer_config):
    graph_name = graph_explorer_config.graph_name
    split = graph_explorer_config.split

    model_name = graph_explorer_config.model_name.split("/")[-1]
    experiment_name = f"graph_explorer_{model_name}"

    if graph_explorer_config.quantized:
        experiment_name += "_quantized"

    if graph_explorer_config.finetune_path:
        experiment_name += f"_{graph_explorer_config.finetune_path.split('/')[-1]}"

    if graph_explorer_config.enable_thinking:
        experiment_name += "_thinking"

    if graph_explorer_config.search_mode == "embeddings":
        emb_model_short = graph_explorer_config.embedding_model.split("/")[-1]
        experiment_name += f"_embeddings_{emb_model_short}"

    results_dir = Path(f"data/experiments/{graph_name}/{experiment_name}/{split}")

    results_dir.mkdir(parents=True, exist_ok=True)

    config_dict = graph_explorer_config.model_dump()

    config_file_path = results_dir / "config.yaml"
    with open(config_file_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)

    return results_dir


def iterate_qas(graph_name: str, split: str | None = None, limit=None):
    base = qa_data_dir(graph_name)
    qas = pd.read_csv(base / "stark_qa" / "stark_qa.csv")
    if split:
        split_indices = [
            int(line.strip()) for line in open(base / "split" / f"{split}.index")
        ]
        qas = qas.iloc[split_indices]

    question_ids = qas["id"].tolist()
    questions = qas["query"].tolist()
    answer_indices_list = qas["answer_ids"].apply(json.loads).tolist()

    if limit:
        return list(zip(question_ids, questions, answer_indices_list))[:limit]
    return list(zip(question_ids, questions, answer_indices_list))


def save_log(log, results_dir, question_id):
    out = Path(results_dir) / f"{question_id}.json"
    with open(out, "w") as f:
        json.dump(log, f, indent=4)


def load_model(graph_explorer_config):
    n_agents = graph_explorer_config.number_of_agents

    if "azure" in graph_explorer_config.model_name.lower():
        model = LiteLLMModel(
            name=graph_explorer_config.model_name,
            max_workers=n_agents,
        )
    elif "gemini" in graph_explorer_config.model_name.lower():
        proxy_url = os.environ.get("PROXY_URL", "http://localhost:8080")
        rpm = int(os.environ.get("RPM_LIMIT", "10"))
        model = AnthropicProxyModel(
            name=graph_explorer_config.model_name,
            base_url=proxy_url,
            rpm=rpm,
            max_workers=n_agents,
        )
    elif "qwen" in graph_explorer_config.model_name.lower():
        if (
            "graphagent" in graph_explorer_config.model_name.lower()
            and graph_explorer_config.enable_thinking
        ):
            raise ValueError("Enable thinking is not supported for finetuned models")

        if graph_explorer_config.finetune_path:
            merged_path = os.path.join(graph_explorer_config.finetune_path, "merged")
            if os.path.isdir(merged_path):
                model = QwenModel(
                    name=merged_path,
                    quantized=graph_explorer_config.quantized,
                    enable_thinking=graph_explorer_config.enable_thinking,
                )
                model.max_workers = 1
            else:
                model = FineTunedQwenModel(
                    name=graph_explorer_config.model_name,
                    finetune_path=graph_explorer_config.finetune_path,
                    quantized=graph_explorer_config.quantized,
                    enable_thinking=graph_explorer_config.enable_thinking,
                )
        else:
            vllm_port = os.environ.get("VLLM_PORT", "8000")
            try:
                import urllib.request
                urllib.request.urlopen(f"http://localhost:{vllm_port}/v1/models", timeout=1)
                model = VLLMQwenModel(
                    server_base_url=f"http://localhost:{vllm_port}/v1",
                    name=graph_explorer_config.model_name,
                    enable_thinking=graph_explorer_config.enable_thinking,
                    max_workers=n_agents,
                )
            except Exception:
                model = QwenModel(
                    name=graph_explorer_config.model_name,
                    quantized=graph_explorer_config.quantized,
                    enable_thinking=graph_explorer_config.enable_thinking,
                )
                model.max_workers = 1
    else:
        openai_base = os.environ.get("OPENAI_API_BASE")
        if openai_base:
            rpm = os.environ.get("RPM_LIMIT")
            model = OpenAICompatibleModel(
                name=graph_explorer_config.model_name,
                base_url=openai_base,
                rpm=int(rpm) if rpm else None,
                max_workers=n_agents,
            )
        else:
            raise ValueError(
                f"Unsupported model: {graph_explorer_config.model_name}. "
                f"Set OPENAI_API_BASE env var to use an OpenAI-compatible endpoint."
            )

    return model


def load_graph(graph_explorer_config):
    graph = Graph(
        name=graph_explorer_config.graph_name,
        path=graph_data_dir(graph_explorer_config.graph_name),
        search_mode=graph_explorer_config.search_mode,
        embedding_model=graph_explorer_config.embedding_model,
    )
    return graph


def load_agent(graph, model, graph_explorer_config):

    with open(
        f"prompts/system_prompt.md",
        "r",
    ) as f:
        system_prompt = f.read().format(node_types=graph.node_types, edge_types=graph.edge_types)

    agent = GraphExplorerAgent(graph=graph, model=model, system_prompt=system_prompt)

    return agent
