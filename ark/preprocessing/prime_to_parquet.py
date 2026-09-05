"""
Prime STaRK preprocessing - limited to ~6000 nodes.
Samples answer nodes from QA data + 1-hop neighbors.
"""

import ast
import csv
import gc
import json
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch

from pathlib import Path

from summary_utils import add_summary_to_prime

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_NODES = 6000

# ── Step 1: Collect answer node indices from QA ──────────────────────────

print("Collecting answer node indices from QA data...")
answer_nodes = set()
with open(DATA_DIR / "qa/prime/stark_qa/stark_qa.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ids = ast.literal_eval(row["answer_ids"])
        if isinstance(ids, list):
            answer_nodes.update(ids)
        else:
            answer_nodes.add(ids)
print(f"  Answer nodes: {len(answer_nodes)}")

# ── Step 2: Load edges, find 1-hop neighbors, cap at MAX_NODES ───────────

print("Loading edge tensors...")
edge_index = torch.load(DATA_DIR / "raw_graphs/prime/edge_index.pt", weights_only=True)
edge_types_tensor = torch.load(DATA_DIR / "raw_graphs/prime/edge_types.pt", weights_only=True)

with open(DATA_DIR / "raw_graphs/prime/edge_type_dict.pkl", "rb") as f:
    edge_type_dict = pickle.load(f)

src_all = edge_index[0].numpy()
dst_all = edge_index[1].numpy()
etype_all = np.array([edge_type_dict[int(t)] for t in edge_types_tensor], dtype=object)
del edge_index, edge_types_tensor, edge_type_dict
gc.collect()

print("Finding 1-hop neighbors...")
answer_arr = np.array(sorted(answer_nodes))
mask = np.isin(src_all, answer_arr) | np.isin(dst_all, answer_arr)
neighbor_nodes = set(src_all[mask].tolist()) | set(dst_all[mask].tolist())
all_needed = answer_nodes | neighbor_nodes
print(f"  Answer + 1-hop: {len(all_needed)} nodes")

if len(all_needed) > MAX_NODES:
    random.seed(42)
    answer_budget = int(MAX_NODES * 0.8)
    neighbor_budget = MAX_NODES - answer_budget
    sampled_answers = set(random.sample(sorted(answer_nodes), min(answer_budget, len(answer_nodes))))
    sampled_arr = np.array(sorted(sampled_answers))
    mask_local = np.isin(src_all, sampled_arr) | np.isin(dst_all, sampled_arr)
    local_neighbors = set(src_all[mask_local].tolist()) | set(dst_all[mask_local].tolist())
    extra = sorted(local_neighbors - sampled_answers)
    sampled_neighbors = set(random.sample(extra, min(neighbor_budget, len(extra))))
    all_needed = sampled_answers | sampled_neighbors
    print(f"  Capped to {len(all_needed)} nodes ({len(sampled_answers)} answer + {len(sampled_neighbors)} neighbors)")

needed_arr = np.array(sorted(all_needed))

print("Filtering edges...")
mask_both = np.isin(src_all, needed_arr) & np.isin(dst_all, needed_arr)
edges_df = pd.DataFrame({
    "start_node_index": src_all[mask_both],
    "end_node_index": dst_all[mask_both],
    "type": etype_all[mask_both],
}).drop_duplicates()
del src_all, dst_all, etype_all
gc.collect()

print(f"  Filtered edges: {len(edges_df)}")
os.makedirs(DATA_DIR / "graphs/prime/", exist_ok=True)
edges_df.to_parquet(DATA_DIR / "graphs/prime/edges.parquet")
del edges_df
gc.collect()

# ── Step 3: Load node_info, filter to needed indices ─────────────────────

print("Loading node_info...")
with open(DATA_DIR / "raw_graphs/prime/node_info.pkl", "rb") as f:
    node_info_full = pickle.load(f)

node_info = {k: v for k, v in node_info_full.items() if k in all_needed}
del node_info_full
gc.collect()
print(f"  Kept {len(node_info)} nodes")

nodes_df = pd.DataFrame(node_info.values())
nodes_df["index"] = list(node_info.keys())
del node_info
gc.collect()

if "id" in nodes_df.columns:
    nodes_df = nodes_df.drop(columns=["id"])
nodes_df["details"] = nodes_df["details"].apply(lambda x: json.dumps({} if pd.isna(x) else x))

print("Adding summaries...")
nodes_df["summary"] = nodes_df.apply(lambda row: add_summary_to_prime(row), axis=1)

print("Saving nodes...")
nodes_df[["index", "name", "type", "summary"]].to_parquet(DATA_DIR / "graphs/prime/nodes.parquet")

print(f"Done! Saved {len(nodes_df)} nodes to {DATA_DIR / 'graphs/prime/'}")
