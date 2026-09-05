import os

import pandas as pd

os.makedirs("data/graphs/parquet/prime", exist_ok=True)
os.makedirs("data/graphs/parquet/mag", exist_ok=True)
os.makedirs("data/graphs/parquet/amazon", exist_ok=True)

pd.read_csv("data/graphs/csv/prime/nodes.csv", low_memory=False).to_parquet(
    "data/graphs/parquet/prime/nodes.parquet"
)
pd.read_csv("data/graphs/csv/prime/edges.csv", low_memory=False).to_parquet(
    "data/graphs/parquet/prime/edges.parquet"
)

pd.read_csv("data/graphs/csv/mag/nodes.csv", low_memory=False).to_parquet(
    "data/graphs/parquet/mag/nodes.parquet"
)
pd.read_csv("data/graphs/csv/mag/edges.csv", low_memory=False).to_parquet(
    "data/graphs/parquet/mag/edges.parquet"
)

pd.read_csv("data/graphs/csv/amazon/nodes.csv", low_memory=False).to_parquet(
    "data/graphs/parquet/amazon/nodes.parquet"
)
pd.read_csv("data/graphs/csv/amazon/edges.csv", low_memory=False).to_parquet(
    "data/graphs/parquet/amazon/edges.parquet"
)
