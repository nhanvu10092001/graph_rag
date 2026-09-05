import os
import json
import random
import pandas as pd
from pathlib import Path
from collections import Counter


def safe_open_json(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Removing corrupted log file: {file_path}")
        os.remove(file_path)
        return None


def load_metrics_df(logs_dir_or_json_files, max_agents=None, max_steps=None, sorting_criteria="voting"):
    # Support both Path object (logs_dir) and list of paths (json_files)
    if isinstance(logs_dir_or_json_files, Path):
        json_files = sorted(logs_dir_or_json_files.glob("*.json"), key=lambda f: f.stat().st_ctime)
    else:
        json_files = logs_dir_or_json_files

    data = []
    for json_file in json_files:
        log_data = safe_open_json(json_file)

        if log_data is None:
            continue  # Skip corrupted files

        # Extract key information from each log entry
        record = {
            "question_id": str(json_file).split("/")[-1].replace(".json", ""),
            "question": log_data.get("question", ""),
            "answer_indices": log_data.get("answer_indices", []),
            "trajectories": log_data.get("trajectories", []),
            "time_taken": log_data.get("time_taken", None),
        }

        if max_agents is None:
            max_agents = len(record["trajectories"])

        if len(record["trajectories"]) < max_agents:
            print(
                f"Warning: Expected {max_agents} agents but found {len(record['trajectories'])} in {json_file}"
            )
            max_agents = len(record["trajectories"])

        trajectories = record["trajectories"][:max_agents]

        # Calculate latency if max_steps is provided
        if max_steps is not None:
            latency = max([traj["step_times"][min(max_steps - 1, len(traj["step_times"]) - 1)] for traj in trajectories])
            record["latency"] = latency
        else:
            record["latency"] = None

        # Filter trajectories by max_steps if provided
        if max_steps is not None:
            agents_answer_indices = [
                traj.get("agent_answer_indices", [])
                for traj in trajectories
                if traj.get("steps") <= max_steps
            ]
        else:
            agents_answer_indices = [
                traj.get("agent_answer_indices", []) for traj in trajectories
            ]

        flat = [idx for sublist in agents_answer_indices for idx in sublist]

        if sorting_criteria == "voting":
            # Count occurrences across all trajectories
            counts = Counter(flat)
            
            # Track minimum depth (position within trajectory) for each node
            min_depth = {}
            for traj_indices in agents_answer_indices:
                for depth, idx in enumerate(traj_indices):
                    if idx not in min_depth or depth < min_depth[idx]:
                        min_depth[idx] = depth
            
            # Sort by count (descending), then by minimum depth (ascending)
            record["combined_answer_indices"] = sorted(
                counts.keys(), key=lambda x: (-counts[x], min_depth.get(x, float('inf')))
            )
        elif sorting_criteria == "order":
            # Keep first appearance of each node in concatenation order
            seen = set()
            ordered = []
            for idx in flat:
                if idx not in seen:
                    seen.add(idx)
                    ordered.append(idx)
            record["combined_answer_indices"] = ordered
        elif sorting_criteria == "random":
            # Collect unique nodes and randomly shuffle
            unique_indices = list(set(flat))
            random.shuffle(unique_indices)
            record["combined_answer_indices"] = unique_indices
        else:
            raise ValueError(f"Unknown sorting_criteria: {sorting_criteria}. Use 'voting', 'order', or 'random'.")

        data.append(record)

    df = pd.DataFrame(data).reset_index(drop=True)

    df["recall@all"] = df.apply(
        lambda row: len(
            set(row["answer_indices"]).intersection(set(row["combined_answer_indices"]))
        )
        / len(set(row["answer_indices"])),
        axis=1,
    )
    df["hit@1"] = df.apply(
        lambda row: (
            row["combined_answer_indices"][0] in row["answer_indices"]
            if row["combined_answer_indices"]
            else False
        ),
        axis=1,
    )
    df["hit@5"] = df.apply(
        lambda row: len(
            set(row["answer_indices"]).intersection(set(row["combined_answer_indices"][:5]))
        )
        > 0,
        axis=1,
    )
    df["hit@10"] = df.apply(
        lambda row: len(
            set(row["answer_indices"]).intersection(set(row["combined_answer_indices"][:10]))
        )
        > 0,
        axis=1,
    )
    df["recall@10"] = df.apply(
        lambda row: len(
            set(row["answer_indices"]).intersection(set(row["combined_answer_indices"][:10]))
        )
        / len(set(row["answer_indices"])),
        axis=1,
    )
    df["recall@20"] = df.apply(
        lambda row: len(
            set(row["answer_indices"]).intersection(set(row["combined_answer_indices"][:20]))
        )
        / len(set(row["answer_indices"])),
        axis=1,
    )

    def reciprocal_rank(row):
        for rank, idx in enumerate(row["combined_answer_indices"], 1):
            if idx in row["answer_indices"]:
                return 1 / rank
        return 0

    df["MRR"] = df.apply(reciprocal_rank, axis=1)

    return df

