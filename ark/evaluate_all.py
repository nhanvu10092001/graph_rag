import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

def safe_open_json(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        return None

def compute_metrics(logs_dir, max_agents=None):
    logs_dir = Path(logs_dir)
    if not logs_dir.exists():
        return None, None

    json_files = sorted(logs_dir.glob("*.json"), key=lambda f: f.stat().st_ctime)
    if not json_files:
        return None, None

    data = []
    tool_counts = Counter()
    total_steps = []
    agent_counts_list = []

    for json_file in json_files:
        log_data = safe_open_json(json_file)
        if log_data is None:
            continue

        trajectories = log_data.get("trajectories", [])
        if not trajectories:
            continue

        actual_agents = len(trajectories)
        agent_counts_list.append(actual_agents)
        eff_agents = actual_agents if max_agents is None else min(max_agents, actual_agents)

        record = {
            "question_id": json_file.stem,
            "question": log_data.get("question", ""),
            "answer_indices": log_data.get("answer_indices", []),
            "time_taken": log_data.get("time_taken", None),
        }

        agents_answer_indices = [
            traj.get("agent_answer_indices", []) for traj in trajectories[:eff_agents]
        ]

        # Analyze tool calls and steps
        for traj in trajectories[:eff_agents]:
            history = traj.get("message_history", [])
            steps = traj.get("steps", len(history))
            total_steps.append(steps)
            for msg in history:
                if isinstance(msg, dict):
                    # Check tool calls
                    if "tool_calls" in msg:
                        for tc in msg["tool_calls"]:
                            fn_name = tc.get("function", {}).get("name", tc.get("name", "unknown"))
                            tool_counts[fn_name] += 1
                    content = msg.get("content", "") or ""
                    if "<tool_call>" in content:
                        import re
                        calls = re.findall(r'"name":\s*"([^"]+)"', content)
                        for c in calls:
                            tool_counts[c] += 1

        flat = [idx for sublist in agents_answer_indices for idx in sublist]
        counts = Counter(flat)
        first_seen = {}
        for i, idx in enumerate(flat):
            first_seen.setdefault(idx, i)
        record["combined_answer_indices"] = sorted(
            counts.keys(), key=lambda x: (-counts[x], first_seen[x])
        )

        data.append(record)

    if not data:
        return None, None

    df = pd.DataFrame(data).reset_index(drop=True)

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
        ) > 0,
        axis=1,
    )
    df["hit@10"] = df.apply(
        lambda row: len(
            set(row["answer_indices"]).intersection(set(row["combined_answer_indices"][:10]))
        ) > 0,
        axis=1,
    )
    df["hit@20"] = df.apply(
        lambda row: len(
            set(row["answer_indices"]).intersection(set(row["combined_answer_indices"][:20]))
        ) > 0,
        axis=1,
    )
    df["recall@1"] = df.apply(
        lambda row: (
            len(set(row["answer_indices"]).intersection(set(row["combined_answer_indices"][:1])))
            / max(len(set(row["answer_indices"])), 1)
        ),
        axis=1,
    )
    df["recall@5"] = df.apply(
        lambda row: (
            len(set(row["answer_indices"]).intersection(set(row["combined_answer_indices"][:5])))
            / max(len(set(row["answer_indices"])), 1)
        ),
        axis=1,
    )
    df["recall@10"] = df.apply(
        lambda row: (
            len(set(row["answer_indices"]).intersection(set(row["combined_answer_indices"][:10])))
            / max(len(set(row["answer_indices"])), 1)
        ),
        axis=1,
    )
    df["recall@20"] = df.apply(
        lambda row: (
            len(set(row["answer_indices"]).intersection(set(row["combined_answer_indices"][:20])))
            / max(len(set(row["answer_indices"])), 1)
        ),
        axis=1,
    )
    df["recall@all"] = df.apply(
        lambda row: (
            len(set(row["answer_indices"]).intersection(set(row["combined_answer_indices"])))
            / max(len(set(row["answer_indices"])), 1)
        ),
        axis=1,
    )

    def reciprocal_rank(row):
        for rank, idx in enumerate(row["combined_answer_indices"], 1):
            if idx in row["answer_indices"]:
                return 1.0 / rank
        return 0.0

    df["MRR"] = df.apply(reciprocal_rank, axis=1)

    summary = {
        "n": len(df),
        "avg_agents": float(np.mean(agent_counts_list)) if agent_counts_list else 1.0,
        "avg_steps": float(np.mean(total_steps)) if total_steps else 0.0,
        "hit@1": float(df["hit@1"].mean()),
        "hit@5": float(df["hit@5"].mean()),
        "hit@10": float(df["hit@10"].mean()),
        "hit@20": float(df["hit@20"].mean()),
        "recall@1": float(df["recall@1"].mean()),
        "recall@5": float(df["recall@5"].mean()),
        "recall@10": float(df["recall@10"].mean()),
        "recall@20": float(df["recall@20"].mean()),
        "recall@all": float(df["recall@all"].mean()),
        "MRR": float(df["MRR"].mean()),
        "tool_counts": dict(tool_counts),
    }

    return df, summary

def main():
    experiments_dir = Path("data/experiments/prime")
    models = [
        ("Oracle Upper Bound (Val)", experiments_dir / "graph_explorer_oracle_traj/val", 1),
        ("Oracle Upper Bound (Train)", experiments_dir / "graph_explorer_oracle_traj/train", 1),
        ("Gemini 3.6 Flash (Val, 5-Agent Voting)", experiments_dir / "graph_explorer_gemini-3.6-flash-high/val", 5),
        ("Gemini 3.6 Flash (Val, 3-Agent Voting)", experiments_dir / "graph_explorer_gemini-3.6-flash-high/val", 3),
        ("Gemini 3.6 Flash (Val, 1-Agent Single)", experiments_dir / "graph_explorer_gemini-3.6-flash-high/val", 1),
        ("Gemini 3.6 Flash (Train, 5-Agent Voting)", experiments_dir / "graph_explorer_gemini-3.6-flash-high/train", 5),
        ("Gemini 3.6 Flash (Train, 1-Agent Single)", experiments_dir / "graph_explorer_gemini-3.6-flash-high/train", 1),
        ("Qwen3-0.6B Distilled (Val, cfc5)", experiments_dir / "graph_explorer_Qwen3-0.6B_quantized_cfc5/val", 1),
        ("Qwen3-0.6B Default Base (Val, Zero-Shot)", experiments_dir / "graph_explorer_Qwen3-0.6B_quantized/val", 1),
    ]

    print("=" * 115)
    print("ARK MULTI-AGENT GRAPH RAG — COMPREHENSIVE MODEL EVALUATION REPORT")
    print("Benchmark: STaRK-PrimeKG Biomedical Knowledge Graph (~130K Nodes, ~8M Edges)")
    print("=" * 115)
    print(f"{'Model Configuration':<40} | {'Split':<5} | {'Ag':<2} | {'n':<4} | {'Hit@1':<6} | {'Hit@5':<6} | {'Hit@10':<6} | {'Rec@10':<6} | {'Rec@20':<6} | {'MRR':<6}")
    print("-" * 115)

    for name, path, max_ag in models:
        split = "val" if "val" in str(path) else "train"
        df, summary = compute_metrics(path, max_agents=max_ag)
        if summary is None:
            continue
        print(f"{name:<40} | {split:<5} | {max_ag:<2} | {summary['n']:<4} | {summary['hit@1']:.4f} | {summary['hit@5']:.4f} | {summary['hit@10']:.4f} | {summary['recall@10']:.4f} | {summary['recall@20']:.4f} | {summary['MRR']:.4f}")

    print("=" * 115)

if __name__ == "__main__":
    main()
