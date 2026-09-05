"""
Amazon STaRK preprocessing - memory-efficient version.

Limits the graph to nodes referenced in QA answer_ids + their 1-hop neighbors,
then caps at MAX_NODES to avoid OOM from the 5.1GB node_info.pkl.
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

from summary_utils import add_summary_amazon

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_NODES = 6000

# ── Step 1: Collect target node indices from QA data ──────────────────────

print("Collecting answer node indices from QA data...")
answer_nodes = set()
qa_csv = DATA_DIR / "qa/amazon/stark_qa/stark_qa.csv"
with open(qa_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        ids = ast.literal_eval(row["answer_ids"])
        if isinstance(ids, list):
            answer_nodes.update(ids)
        else:
            answer_nodes.add(ids)
print(f"  Answer nodes from QA: {len(answer_nodes)}")

# ── Step 2: Load edges (small tensors), find 1-hop neighbors ─────────────

print("Loading edge tensors...")
edge_index = torch.load(DATA_DIR / "raw_graphs/amazon/edge_index.pt", weights_only=True)
edge_types_tensor = torch.load(DATA_DIR / "raw_graphs/amazon/edge_types.pt", weights_only=True)

with open(DATA_DIR / "raw_graphs/amazon/edge_type_dict.pkl", "rb") as f:
    edge_type_dict = pickle.load(f)

edge_index_bcc = torch.load(
    DATA_DIR / "raw_graphs/amazon/cache/brand-category-color/edge_index.pt", weights_only=True
)
edge_types_bcc = torch.load(
    DATA_DIR / "raw_graphs/amazon/cache/brand-category-color/edge_types.pt", weights_only=True
)
with open(
    DATA_DIR / "raw_graphs/amazon/cache/brand-category-color/edge_type_dict.pkl", "rb"
) as f:
    edge_type_dict_bcc = pickle.load(f)

src_all = np.concatenate([edge_index[0].numpy(), edge_index_bcc[0].numpy()])
dst_all = np.concatenate([edge_index[1].numpy(), edge_index_bcc[1].numpy()])
etype_all = []
for t in edge_types_tensor:
    etype_all.append(edge_type_dict[int(t)])
for t in edge_types_bcc:
    etype_all.append(edge_type_dict_bcc[int(t)])
etype_all = np.array(etype_all, dtype=object)

del edge_index, edge_types_tensor, edge_type_dict
del edge_index_bcc, edge_types_bcc, edge_type_dict_bcc
gc.collect()

print("Finding 1-hop neighbors of answer nodes...")
answer_arr = np.array(sorted(answer_nodes))
mask_src = np.isin(src_all, answer_arr)
mask_dst = np.isin(dst_all, answer_arr)
neighbor_mask = mask_src | mask_dst
neighbor_nodes = set(src_all[neighbor_mask].tolist()) | set(dst_all[neighbor_mask].tolist())
all_needed = answer_nodes | neighbor_nodes
print(f"  Answer + 1-hop neighbors: {len(all_needed)} nodes")

if len(all_needed) > MAX_NODES:
    random.seed(42)
    # Sample answer nodes down to ~80% of budget, leave 20% for neighbors
    answer_budget = int(MAX_NODES * 0.8)
    neighbor_budget = MAX_NODES - answer_budget
    sampled_answers = set(random.sample(sorted(answer_nodes), min(answer_budget, len(answer_nodes))))
    # Pick neighbors connected to sampled answers
    sampled_arr = np.array(sorted(sampled_answers))
    mask_s = np.isin(src_all, sampled_arr)
    mask_d = np.isin(dst_all, sampled_arr)
    local_neighbors = set(src_all[mask_s | mask_d].tolist()) | set(dst_all[mask_s | mask_d].tolist())
    extra = sorted(local_neighbors - sampled_answers)
    sampled_neighbors = set(random.sample(extra, min(neighbor_budget, len(extra))))
    all_needed = sampled_answers | sampled_neighbors
    print(f"  Capped to {len(all_needed)} nodes ({len(sampled_answers)} answer + {len(sampled_neighbors)} neighbors)")

needed_arr = np.array(sorted(all_needed))

print("Filtering edges to subgraph...")
mask_both = np.isin(src_all, needed_arr) & np.isin(dst_all, needed_arr)
edges_df = pd.DataFrame({
    "start_node_index": src_all[mask_both],
    "end_node_index": dst_all[mask_both],
    "type": etype_all[mask_both],
}).drop_duplicates()

del src_all, dst_all, etype_all, mask_src, mask_dst, neighbor_mask, mask_both
gc.collect()

print(f"  Filtered edges: {len(edges_df)}")
os.makedirs(DATA_DIR / "graphs/amazon/", exist_ok=True)
edges_df.to_parquet(DATA_DIR / "graphs/amazon/edges.parquet")
del edges_df
gc.collect()
print("  Edges saved.")

# ── Step 3: Load node_info.pkl, filter immediately ────────────────────────

print(f"Loading node_info.pkl — will filter to {len(all_needed)} nodes...")
gc.disable()
with open(DATA_DIR / "raw_graphs/amazon/node_info.pkl", "rb") as f:
    node_info_full = pickle.load(f)
gc.enable()

node_info = {k: v for k, v in node_info_full.items() if k in all_needed}
del node_info_full
gc.collect()
print(f"  Kept {len(node_info)} nodes from main node_info")

print("Loading brand-category-color node_info...")
with open(DATA_DIR / "raw_graphs/amazon/cache/brand-category-color/node_info.pkl", "rb") as f:
    node_info_bcc = pickle.load(f)
node_info_bcc_filtered = {k: v for k, v in node_info_bcc.items() if k in all_needed}
del node_info_bcc
gc.collect()
print(f"  Kept {len(node_info_bcc_filtered)} nodes from brand-category-color")

node_info.update(node_info_bcc_filtered)
del node_info_bcc_filtered
gc.collect()

nodes_df = pd.DataFrame(node_info.values(), index=node_info.keys())
nodes_df["index"] = nodes_df.index
del node_info
gc.collect()

# ── Step 4: Clean review/QA fields ────────────────────────────────────────


def clean_review_list(review_list):
    if not isinstance(review_list, list):
        return None
    cleaned_list = []
    for review_dict in review_list:
        cleaned = {}
        for key, value in review_dict.items():
            if pd.isna(value) or value is np.nan:
                cleaned[key] = None
            elif isinstance(value, (np.float64, np.float32)):
                cleaned[key] = float(value)
            elif isinstance(value, (np.int64, np.int32)):
                cleaned[key] = int(value)
            elif isinstance(value, np.bool_):
                cleaned[key] = bool(value)
            else:
                cleaned[key] = value
        cleaned_list.append(cleaned)
    return cleaned_list


if "review" in nodes_df.columns:
    print("Cleaning reviews...")
    nodes_df["review"] = nodes_df["review"].apply(clean_review_list)
    nodes_df["review"] = nodes_df["review"].apply(json.dumps)

if "qa" in nodes_df.columns:
    print("Cleaning qas...")
    nodes_df["qa"] = nodes_df["qa"].apply(clean_review_list)
    nodes_df["qa"] = nodes_df["qa"].apply(json.dumps)

# ── Step 5: Add node types, names, summaries ──────────────────────────────

node_types_bcc = torch.load(
    DATA_DIR / "raw_graphs/amazon/cache/brand-category-color/node_types.pt", weights_only=True
)
with open(DATA_DIR / "raw_graphs/amazon/cache/brand-category-color/node_type_dict.pkl", "rb") as f:
    node_types_dict_bcc = pickle.load(f)

type_map = {}
for idx, t in enumerate(node_types_bcc):
    type_map[idx] = node_types_dict_bcc[int(t)]
del node_types_bcc, node_types_dict_bcc

nodes_df["type"] = nodes_df["index"].map(type_map)
del type_map
gc.collect()

print("Adding names...")
name_column = {
    "product": "title",
    "brand": "brand_name",
    "color": "color_name",
    "category": "category_name",
}
nodes_df["name"] = nodes_df.apply(
    lambda row: row.get(name_column.get(row["type"], ""), "") if pd.notna(row["type"]) else "",
    axis=1,
)

print("Adding summaries...")
nodes_df["summary"] = nodes_df.apply(
    lambda row: add_summary_amazon(row) if pd.notna(row["type"]) else "", axis=1
)

print("Saving nodes...")
nodes_df["name"] = nodes_df["name"].astype(str)
nodes_df[["index", "name", "type", "summary"]].to_parquet(DATA_DIR / "graphs/amazon/nodes.parquet")

print(f"Done! Saved {len(nodes_df)} nodes to {DATA_DIR / 'graphs/amazon/'}")
