"""Build direct (oracle) trajectories from ground-truth answers.

For each question, construct a single trajectory that reaches the correct answer
directly, without relying on a (low-quality) teacher:

    search_in_graph(node name) x k   ->   add_to_answer(all answer_indices)   ->   finish

The search step is real (runs on the actual graph) so the tool responses are
authentic and surface the answer nodes. Output JSONs match the schema produced
by main.py (question / answer_indices / trajectories), so the existing
finetune pipeline (fine_tuning.utils.get_split_dataset) consumes them unchanged.
"""
import argparse
import json
import random
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import GraphExplorerConfig, load_graph, save_log
from src.agents.graph_explorer.tools.search_in_graph import SearchInGraphTool

MAX_SEARCHES = 8  # bound context length when a question has many answers


def build_tool_call(name: str, arguments: dict) -> dict:
    call_id = "call_" + "".join(random.choices(string.digits, k=6))
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _node_or_none(graph, idx):
    try:
        return graph.get_node_by_index(idx)
    except ValueError:
        return None


def build_message_history(question, answer_indices, graph, search_tool):
    agent_answer = list(dict.fromkeys(answer_indices))  # dedupe, keep order

    # Only indices that actually exist in the graph can be retrieved / added.
    valid_nodes = [
        (idx, node)
        for idx in agent_answer
        if (node := _node_or_none(graph, idx)) is not None
    ]

    messages = [
        {"role": "system", "content": ""},  # dropped by message_history[1:]
        {"role": "user", "content": f"Find nodes that answer the question: {question}"},
    ]

    # --- step 1: search_in_graph ---
    # Prefer querying each (valid) answer node's name so the tool response
    # surfaces the answer. Fall back to the question text so there is always
    # at least one search step (teaches the search tool).
    search_queries = [node.name for _, node in valid_nodes[:MAX_SEARCHES]]
    if not search_queries:
        search_queries = [question]
    search_queries = search_queries[:MAX_SEARCHES]

    search_calls = [build_tool_call("search_in_graph", {"query": q, "size": 5}) for q in search_queries]
    messages.append(
        {"role": "assistant", "reasoning_content": None, "content": None, "tool_calls": search_calls}
    )
    for call, q in zip(search_calls, search_queries):
        messages.append(
            {"role": "tool", "content": search_tool({"query": q, "size": 5}), "tool_call_id": call["id"]}
        )

    # --- step 2: add_to_answer with the ground-truth indices that exist ---
    answer_nodes = [
        {"node_index": int(idx), "reasoning": f"Answer: {node.name} — relevant to: {question}"}
        for idx, node in valid_nodes
    ]
    add_call = build_tool_call("add_to_answer", {"answer_nodes": answer_nodes})
    messages.append(
        {"role": "assistant", "reasoning_content": None, "content": None, "tool_calls": [add_call]}
    )

    if answer_nodes:
        add_response = "Added the following nodes to the answer:\n" + "\n".join(
            f"- [{idx}] {node.name} (type: {node.type})\n"
            f"  Reasoning: Answer: {node.name} — relevant to: {question}"
            for idx, node in valid_nodes
        )
    else:
        add_response = "No valid nodes were added to the answer."
    messages.append({"role": "tool", "content": add_response, "tool_call_id": add_call["id"]})

    # --- step 3: finish ---
    comment = "Exploration completed. Found and added all relevant nodes for the question."
    finish_call = build_tool_call("finish", {"comment": comment})
    messages.append(
        {"role": "assistant", "reasoning_content": None, "content": None, "tool_calls": [finish_call]}
    )
    messages.append(
        {"role": "tool", "content": f"Exploration finished. Agent's comment: {comment}", "tool_call_id": finish_call["id"]}
    )

    trajectory = {
        "message_history": messages,
        "agent_answer_indices": [int(idx) for idx, _ in valid_nodes],
        "steps": 3,
        "step_times": [],
    }
    return trajectory


def process_split(src_dir, out_dir, graph, search_tool):
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_files = n_skipped = 0
    for path in sorted(src_dir.glob("*.json")):
        with open(path) as f:
            log = json.load(f)
        question = log["question"]
        answer_indices = log["answer_indices"]
        if not answer_indices:
            n_skipped += 1
            continue

        trajectory = build_message_history(question, answer_indices, graph, search_tool)
        save_log(
            {"question": question, "answer_indices": answer_indices, "trajectories": [trajectory]},
            results_dir=out_dir,
            question_id=path.stem,
        )
        n_files += 1
    return n_files, n_skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_name", default="prime")
    parser.add_argument(
        "--src_dir",
        default=None,
        help="Teacher trajectory dir. Default: data/experiments/<graph>/graph_explorer_gemini-3.6-flash-high",
    )
    parser.add_argument(
        "--out_dir", default=None, help="Default: data/experiments/<graph>/graph_explorer_oracle_traj"
    )
    args = parser.parse_args()

    src_base = args.src_dir or f"data/experiments/{args.graph_name}/graph_explorer_gemini-3.6-flash-high"
    out_base = args.out_dir or f"data/experiments/{args.graph_name}/graph_explorer_oracle_traj"

    graph = load_graph(GraphExplorerConfig(graph_name=args.graph_name, model_name="oracle-build", split="train"))
    search_tool = SearchInGraphTool(graph)

    for split in ("train", "val"):
        src_dir = Path(src_base) / split
        out_dir = Path(out_base) / split
        n, skipped = process_split(src_dir, out_dir, graph, search_tool)
        print(f"[{split}] wrote {n} oracle files -> {out_dir} (skipped {skipped})")


if __name__ == "__main__":
    main()
