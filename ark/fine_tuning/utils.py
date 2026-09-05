import json
from datasets import Dataset
from tqdm import tqdm

from src.agents.graph_explorer.tools.search_in_graph import SearchInGraphTool
from src.agents.graph_explorer.tools.search_in_neighborhood import SearchInNeighborhoodTool
from src.agents.graph_explorer.tools.add_to_answers import AddToAnswer
from src.agents.graph_explorer.tools.finish import FinishTool


def calculate_recall_at_k(predicted, gt, k=20):
    """Calculate recall at k"""
    gt_answer_indices = set(gt)
    agent_answer_indices = set(predicted[:k])
    if not gt_answer_indices:
        return 0.0
    return len(gt_answer_indices.intersection(agent_answer_indices)) / len(gt_answer_indices)


def load_log_dataset(json_logs, config):
    """Load and format log data for fine-tuning"""
    samples = []
    tools = [
        SearchInGraphTool.schema(),
        SearchInNeighborhoodTool.schema(),
        AddToAnswer.schema(),
        FinishTool.schema(),
    ]
    for log in json_logs:
        if config.rejection_sampling:
            for trajectory in log["trajectories"]:
                trajectory["recall@20"] = calculate_recall_at_k(
                    trajectory["agent_answer_indices"], log["answer_indices"], k=20
                )

            # Best performance is defined by highest recall@20, and in case of ties, by fewest steps.
            best_trajectory = sorted(
                log["trajectories"], key=lambda x: (-x["recall@20"], len(x["message_history"]))
            )[0]

            if best_trajectory["recall@20"] == 0.0:
                continue

            # We will add to the training data the most performant trajectory, as long as the recall is not zero.
            # We do it three times to mantains compatibility with the no-rejection sampling case.
            for _ in range(3):
                samples.append({"messages": best_trajectory["message_history"][1:], "tools": tools})

        else:
            for trajectory in log["trajectories"]:
                samples.append({"messages": trajectory["message_history"][1:], "tools": tools})

    return Dataset.from_list(samples).shuffle()


def get_split_dataset(split, config):
    """Load and format log data for fine-tuning for a specific split"""
    if split == "train":
        trajectories_dir = config.train_trajectories_dir
        queries_limit = config.train_queries_limit
    elif split == "val":
        trajectories_dir = config.val_trajectories_dir
        queries_limit = config.val_queries_limit
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'")

    # Load log files
    log_files = list(trajectories_dir.glob("*.json"))
    if not log_files:
        raise FileNotFoundError(f"No log files found in {trajectories_dir}")

    json_logs = [
        json.load(open(path, "r")) for path in tqdm(log_files, desc=f"Loading {split} trajectories")
    ]

    # Apply queries limit
    if queries_limit != -1:
        json_logs = json_logs[:queries_limit]

    return load_log_dataset(json_logs, config)
