import os
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.core.logger import logger
from utils import (
    GraphExplorerConfig,
    iterate_qas,
    load_agent,
    load_graph,
    load_model,
    save_log,
    setup_ablation_results_dir,
)
from src.agents.graph_explorer.graph_explorer import GraphExplorerAgent
from src.agents.graph_explorer.tools.search_in_neighborhood import SearchInNeighborhoodTool
import time


def schema_without_query(cls) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": cls.name(),
            "description": cls.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "node_index": {
                        "type": "integer",
                        "description": "The index of the node to search around",
                    },
                    "node_type": {
                        "type": "string",
                        "description": "The type of nodes to search in",
                    },
                    "edge_type": {
                        "type": "string",
                        "description": "The type of edges to traverse when finding neighbors",
                    },
                },
                "required": [],
            },
        },
    }


def schema_without_filters(cls) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": cls.name(),
            "description": cls.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "node_index": {
                        "type": "integer",
                        "description": "The index of the node to search around",
                    },
                    "query": {
                        "type": "string",
                        "description": "The keyword to search for. You can also leave this empty and get all nodes of the specified type",
                    },
                },
                "required": [],
            },
        },
    }


def load_agent(graph, model):
    with open(
        "prompts/system_prompt.md",
        "r",
    ) as f:
        system_prompt = f.read().format(node_types=graph.node_types, edge_types=graph.edge_types)

    return GraphExplorerAgent(graph=graph, model=model, system_prompt=system_prompt)


def run_single_agent_for_question(graph, model, max_steps: int, question: str, tool_to_remove: str):

    if tool_to_remove == "neighbor_queries":
        SearchInNeighborhoodTool.schema = classmethod(schema_without_query)
    elif tool_to_remove == "neighbor_filters":
        SearchInNeighborhoodTool.schema = classmethod(schema_without_filters)

    agent = load_agent(graph, model)

    if tool_to_remove not in ["neighbor_queries", "neighbor_filters"]:
        agent.tools = [tool for tool in agent.tools if tool.name() != tool_to_remove]

    agent.find_nodes(
        query=f"Find nodes that answer the question: {question}",
        max_steps=max_steps,
    )
    return {
        "message_history": agent.message_history,
        "agent_answer_indices": [int(node.index) for node in agent.selected_nodes],
        "steps": agent.step,
        "step_times": agent.step_times,
    }


def run_agents_for_question(
    graph,
    model,
    max_steps: int,
    number_of_agents: int,
    question: str,
    tool_to_remove: str,
) -> list[dict]:
    max_workers = number_of_agents
    if "azure" in model.name.lower():
        max_workers = 3

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_single_agent_for_question,
                graph,
                model,
                max_steps=max_steps,
                question=question,
                tool_to_remove=tool_to_remove,
            )
            for _ in range(number_of_agents)
        ]
        return [future.result() for future in futures]


def main():

    parser = ArgumentParser()
    parser.add_argument("--graph_name", type=str, default="prime")
    parser.add_argument("--model_name", type=str, default="azure/gpt-4.1")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--number_of_agents", type=int, default=3)
    parser.add_argument("--finetune_path", type=str, default=None)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    if args.limit == -1:
        args.limit = None

    graph_explorer_config = GraphExplorerConfig(
        graph_name=args.graph_name,
        model_name=args.model_name,
        finetune_path=args.finetune_path,
        quantized=False,
        enable_thinking=False,
        split=args.split,
        max_steps=20,
        limit=args.limit,
        number_of_agents=args.number_of_agents,
    )

    graph = load_graph(graph_explorer_config)
    model = load_model(graph_explorer_config)

    tools_to_ablate = [
        "neighbor_queries",
        "neighbor_filters",
        "search_in_neighborhood",
    ]

    for tool_to_remove in tools_to_ablate:
        results_dir = setup_ablation_results_dir(
            graph_explorer_config, tool_to_remove=tool_to_remove
        )
        logger.info(f"Results will be saved to: {results_dir}")
        errors = 0
        for question_id, question, answer_indices in iterate_qas(
            graph_explorer_config.graph_name,
            graph_explorer_config.split,
            limit=graph_explorer_config.limit,
        ):
            if os.path.exists(results_dir / f"{question_id}.json"):
                logger.info(f"Skipping {question_id} as it was already processed.")
                continue
            try:
                logger.info(f"Processing question {question_id}: {question}")

                trajectories = run_agents_for_question(
                    graph=graph,
                    model=model,
                    max_steps=graph_explorer_config.max_steps,
                    number_of_agents=graph_explorer_config.number_of_agents,
                    question=question,
                    tool_to_remove=tool_to_remove,
                )

                save_log(
                    {
                        "question": question,
                        "answer_indices": answer_indices,
                        "trajectories": trajectories,
                    },
                    results_dir=results_dir,
                    question_id=question_id,
                )

                logger.info(f"Successfully processed question {question_id}")
            except Exception as e:
                logger.error(f"Error processing question {question_id}: {e}")
                errors += 1
                time.sleep(5)
                if errors > 30:
                    raise e

        logger.info("Graph explorer completed.")


if __name__ == "__main__":
    main()
