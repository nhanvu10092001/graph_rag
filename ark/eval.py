import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def safe_open_json(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Removing corrupted log file: {file_path}")
        os.remove(file_path)
        return None


def load_metrics_df(logs_dir, max_agents=None):
    json_files = sorted(logs_dir.glob("*.json"), key=lambda f: f.stat().st_ctime)

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

        agents_answer_indices = [
            traj.get("agent_answer_indices", []) for traj in record["trajectories"]
        ][:max_agents]

        flat = [idx for sublist in agents_answer_indices for idx in sublist]
        counts = Counter(flat)
        first_seen = {}
        for i, idx in enumerate(flat):
            first_seen.setdefault(idx, i)
        record["combined_answer_indices"] = sorted(
            counts.keys(), key=lambda x: (-counts[x], first_seen[x])
        )

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


parser = ArgumentParser()
parser.add_argument("--graph_name", type=str, default="mag")
parser.add_argument("--model_name", type=str, default="azure/gpt-4.1")
parser.add_argument("--split", type=str, default="test")
args = parser.parse_args()


performance_df = pd.DataFrame()

logs_dir = Path(f"data/experiments/{args.graph_name}/graph_explorer_{args.model_name.split('/')[-1]}/{args.split}")
df = load_metrics_df(logs_dir, max_agents=3)

metrics = [
    ("n", len(df)),
    (f"Hit@1", float(round(df[f"hit@1"].mean(), 3))),
    (f"Hit@5", float(round(df[f"hit@5"].mean(), 3))),
    (f"Recall@10", float(round(df[f"recall@10"].mean(), 3))),
    (f"Recall@20", float(round(df[f"recall@20"].mean(), 3))),
    (f"Recall@all", float(round(df[f"recall@all"].mean(), 3))),
    (f"MRR", float(round(df[f"MRR"].mean(), 3))),
    (f"TimeTakenMean", float(round(df[f"time_taken"].mean(), 3))),
    (f"TimeTakenStd", float(round(df[f"time_taken"].std(), 3))),
]
print(f"Model: graph_explorer_{args.model_name.split('/')[-1]}")
for k, v in metrics:
    print(f"  {k}: {v}")
print()