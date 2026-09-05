from pathlib import Path
import polars as pl

from src.core.index import GraphIndex
from src.core.embedding_index import EmbeddingIndex
from src.core.logger import logger
from src.core.node import Node


class GraphPath:
    def __init__(self, path_as_list: list):
        self.path_as_list = self._validate_path_as_list(path_as_list)

    @staticmethod
    def _validate_path_as_list(path_as_list):
        for i in range(0, len(path_as_list), 2):
            assert isinstance(
                path_as_list[i], Node
            ), f"Expected Node at index {i}, got {type(path_as_list[i])}"
        for i in range(1, len(path_as_list), 2):
            assert isinstance(
                path_as_list[i], str
            ), f"Expected str at index {i}, got {type(path_as_list[i])}"
        return path_as_list

    def __hash__(self):
        # A frozenset is an immutable, unordered collection of unique elements.
        # Here, we use it to ensure that the hash is the same for a path and its reverse,
        # since {path, reversed_path} as a frozenset will be equal for both.
        fwd = tuple(self.path_as_list)
        rev = tuple(reversed(self.path_as_list))
        return hash(frozenset([fwd, rev]))

    def __eq__(self, other):
        if not isinstance(other, GraphPath):
            return False
        fwd_self = tuple(self.path_as_list)
        rev_self = tuple(reversed(self.path_as_list))
        fwd_other = tuple(other.path_as_list)
        rev_other = tuple(reversed(other.path_as_list))
        return frozenset([fwd_self, rev_self]) == frozenset([fwd_other, rev_other])

    def __repr__(self):
        return " <-> ".join(
            [str(node) if isinstance(node, Node) else node for node in self.path_as_list]
        )

    def __str__(self):
        return repr(self)


class Graph:
    def __init__(self, name, path: Path, search_mode: str = "bm25", embedding_model: str = "azure/text-embedding-3-large"):
        self.name = name
        self.path = path
        self.search_mode = search_mode

        self.nodes_df = pl.read_parquet(path / "nodes.parquet")
        self.edges_df = pl.read_parquet(path / "edges.parquet")

        self.index = self.load_or_create_index(path / "index")

        if search_mode == "embeddings":
            self.embedding_index = self.load_or_create_embedding_index(path / "embedding_index", embedding_model)

    def load_or_create_index(self, path: Path):
        if path.exists():
            return GraphIndex(path)

        return GraphIndex.from_nodes_df(path, self.nodes_df)

    def load_or_create_embedding_index(self, path: Path, model: str):
        return EmbeddingIndex.from_nodes_df(path, self.nodes_df, model=model)

    @property
    def node_types(self) -> list[str]:
        return self.nodes_df["type"].unique().to_list()

    @property
    def edge_types(self) -> list[str]:
        return self.edges_df["type"].unique().to_list()

    def search_nodes(self, query: str, k: int = 10):
        if self.search_mode == "embeddings":
            return self.embedding_index.search(query, k)
        return self.index.search(query, k)

    def search_nodes_by_indices(self, query: str, candidate_indices: list[int], k: int = 10):
        if self.search_mode == "embeddings":
            return self.embedding_index.search_in_subset(query, candidate_indices, k)
        raise NotImplementedError("search_nodes_by_indices is only supported for embedding search mode")

    def get_node_by_index(self, index: int) -> Node:
        df = self.nodes_df.filter(pl.col("index") == index)
        if df.height == 0:
            raise ValueError(f"Node with index {index} not found")
        row = df.to_dicts()[0]
        return Node.from_df_row(row)

    def get_indices_of_neighbors(self, node_index: int) -> set[int]:
        neighbors_start = self.edges_df.filter(pl.col("start_node_index") == node_index).select(
            pl.col("end_node_index").alias("neighbor_index")
        )
        neighbors_end = self.edges_df.filter(pl.col("end_node_index") == node_index).select(
            pl.col("start_node_index").alias("neighbor_index")
        )
        neighbor_indices = pl.concat([neighbors_start, neighbors_end]).unique()

        return set(neighbor_indices["neighbor_index"].to_list()) 

    def get_khop_subgraph(self, node: Node, k: int):
        first_hop_neighbors_indices = self.get_indices_of_neighbors(node.index) | {node.index}

        if k == 1:
            khop_indices = first_hop_neighbors_indices

        if k == 2:
            neighbors_of_first_hop_start = set(
                self.edges_df.filter(pl.col("start_node_index").is_in(first_hop_neighbors_indices))[
                    "end_node_index"
                ]
                .unique()
                .to_list()
            )
            neighbors_of_first_hop_end = set(
                self.edges_df.filter(pl.col("end_node_index").is_in(first_hop_neighbors_indices))[
                    "start_node_index"
                ]
                .unique()
                .to_list()
            )

            khop_indices = (
                first_hop_neighbors_indices
                | neighbors_of_first_hop_start
                | neighbors_of_first_hop_end
            )

        khop_nodes_df = self.nodes_df.filter(pl.col("index").is_in(khop_indices))
        khop_edges_df = self.edges_df.filter(pl.col("start_node_index").is_in(khop_indices) | pl.col("end_node_index").is_in(khop_indices))

        return khop_nodes_df.clone(), khop_edges_df.clone()

    def find_paths_of_length_2(self, src: Node, dst: Node):
        if src.index == dst.index:
            return []

        # Build direct connections, normalizing column order so polars can concatenate
        direct_src = self.edges_df.filter(
            (pl.col("start_node_index") == src.index) & (pl.col("end_node_index") == dst.index)
        ).rename({"start_node_index": "src_index", "end_node_index": "dst_index"})
        direct_dst = self.edges_df.filter(
            (pl.col("start_node_index") == dst.index) & (pl.col("end_node_index") == src.index)
        ).rename({"start_node_index": "dst_index", "end_node_index": "src_index"})

        if len(direct_src.columns) != len(direct_dst.columns) or set(direct_src.columns) != set(
            direct_dst.columns
        ):
            common_cols = sorted(set(direct_src.columns) & set(direct_dst.columns))
            direct_src = direct_src.select(common_cols)
            direct_dst = direct_dst.select(common_cols)
        else:
            ordered_cols = sorted(direct_src.columns)
            direct_src = direct_src.select(ordered_cols)
            direct_dst = direct_dst.select(ordered_cols)

        direct_connections = (
            pl.concat([direct_src, direct_dst])
            if direct_src.height or direct_dst.height
            else pl.DataFrame()
        )

        src_mediators = self.edges_df.filter(pl.col("start_node_index") == src.index).rename(
            {
                "start_node_index": "src_index",
                "end_node_index": "neighbor",
            }
        )
        mediator_src = self.edges_df.filter(pl.col("end_node_index") == src.index).rename(
            {
                "start_node_index": "neighbor",
                "end_node_index": "src_index",
            }
        )
        # Normalize column order before concatenation
        if set(src_mediators.columns) != set(mediator_src.columns):
            common_cols = sorted(set(src_mediators.columns) & set(mediator_src.columns))
            src_mediators = src_mediators.select(common_cols)
            mediator_src = mediator_src.select(common_cols)
        else:
            ordered_cols = sorted(src_mediators.columns)
            src_mediators = src_mediators.select(ordered_cols)
            mediator_src = mediator_src.select(ordered_cols)

        src_edges = pl.concat([src_mediators, mediator_src]).unique()

        mediator_dst = self.edges_df.filter(pl.col("end_node_index") == dst.index).rename(
            {
                "start_node_index": "neighbor",
                "end_node_index": "dst_index",
            }
        )
        dst_mediators = self.edges_df.filter(pl.col("start_node_index") == dst.index).rename(
            {
                "start_node_index": "dst_index",
                "end_node_index": "neighbor",
            }
        )

        # Normalize column order before concatenation
        if set(mediator_dst.columns) != set(dst_mediators.columns):
            common_cols = sorted(set(mediator_dst.columns) & set(dst_mediators.columns))
            mediator_dst = mediator_dst.select(common_cols)
            dst_mediators = dst_mediators.select(common_cols)
        else:
            ordered_cols = sorted(mediator_dst.columns)
            mediator_dst = mediator_dst.select(ordered_cols)
            dst_mediators = dst_mediators.select(ordered_cols)

        dst_edges = pl.concat([mediator_dst, dst_mediators]).unique()

        # Ensure distinct column names for join
        src_edges = src_edges.rename({"type": "type_1"})
        dst_edges = dst_edges.rename({"type": "type_2"})

        paths_with_mediator = src_edges.join(dst_edges, on="neighbor", how="inner").unique()

        paths = set()

        for row in direct_connections.to_dicts():
            path = GraphPath(
                path_as_list=[
                    self.get_node_by_index(row["src_index"]),
                    row["type"],
                    self.get_node_by_index(row["dst_index"]),
                ]
            )
            paths.add(path)

        for row in paths_with_mediator.to_dicts():
            path = GraphPath(
                path_as_list=[
                    self.get_node_by_index(row["src_index"]),
                    row["type_1"],
                    self.get_node_by_index(row["neighbor"]),
                    row["type_2"],
                    self.get_node_by_index(row["dst_index"]),
                ]
            )
            paths.add(path)

        return paths
