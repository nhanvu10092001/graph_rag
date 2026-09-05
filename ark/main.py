import os
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

from src.core.logger import logger
from utils import (
    GraphExplorerConfig,
    iterate_qas,
    load_agent,
    load_graph,
    load_model,
    save_log,
    setup_graph_explorer_results_dir,
)
from src.agents.graph_explorer.graph_explorer import GraphExplorerAgent
import time


def load_agent(graph, model):
    with open(
        "prompts/system_prompt.md",
        "r",
    ) as f:
        system_prompt = f.read().format(node_types=graph.node_types, edge_types=graph.edge_types)

    return GraphExplorerAgent(graph=graph, model=model, system_prompt=system_prompt)


def run_single_agent_for_question(graph, model, max_steps: int, question: str):

    agent = load_agent(graph, model)
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
) -> list[dict]:
    with ThreadPoolExecutor(max_workers=model.max_workers) as executor:
        futures = [
            executor.submit(
                run_single_agent_for_question,
                graph,
                model,
                max_steps=max_steps,
                question=question,
            )
            for _ in range(number_of_agents)
        ]
        return [future.result() for future in futures]


def main():
    # allow_abbrev=False: e.g. --graph must not abbreviate --graph_name vs --graph-name.
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("--graph_name", "--graph-name", "--graph", type=str, default="prime")
    parser.add_argument("--model_name", "--model-name", type=str, default="azure/gpt-4.1")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--number_of_agents", "--number-of-agents", type=int, default=3)
    parser.add_argument("--finetune_path", "--finetune-path", type=str, default=None)
    parser.add_argument(
        "--search_mode",
        "--search-mode",
        type=str,
        choices=["bm25", "embeddings"],
        default="bm25",
    )
    parser.add_argument(
        "--embedding_model", "--embedding-model", type=str, default="azure/text-embedding-3-large"
    )
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max_steps", "--max-steps", type=int, default=30)
    parser.add_argument("--quantized", action="store_true", default=False)
    args = parser.parse_args()

    if args.limit == -1:
        args.limit = None

    graph_explorer_config = GraphExplorerConfig(
        graph_name=args.graph_name,
        model_name=args.model_name,
        finetune_path=args.finetune_path,
        quantized=args.quantized,
        enable_thinking=False,
        search_mode=args.search_mode,
        embedding_model=args.embedding_model,
        split=args.split,
        max_steps=args.max_steps,
        limit=args.limit,
        number_of_agents=args.number_of_agents,
    )

    graph = load_graph(graph_explorer_config)
    model = load_model(graph_explorer_config)

    errors = 0
    results_dir = setup_graph_explorer_results_dir(graph_explorer_config)
    logger.info(f"Results will be saved to: {results_dir}")
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
            if errors > 200:
                raise e

    logger.info("Graph explorer completed.")


if __name__ == "__main__":
    main()
