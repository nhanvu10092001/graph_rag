import polars as pl
from typing import Any
from src.agents.graph_explorer.tools.tool import Tool
from src.agents.graph_explorer.tools.utils import filter_by_node_type, filter_by_query
from src.core.graph import Node


class SearchInNeighborhoodTool(Tool):
    MAX_RESULTS_SHOWN: int = 15

    def __init__(self, graph):
        super().__init__(graph=graph)
        self.MAX_RESULTS_SHOWN = 20

    @classmethod
    def name(cls) -> str:
        return "search_in_neighborhood"

    @classmethod
    def description(cls) -> str:
        return "Search in the immediate neighborhood (1 hop) for a specific keyword, in nodes with a specific type, optionally filtering by edge type. Shows relation types between reference node and neighbors."

    def _format_candidates(
        self,
        candidates: pl.DataFrame,
        node: Node,
        query: str,
        node_type: str,
        edge_type: str,
    ):
        search_desc = f"[{node.index}] {node.name}"
        params = []
        if query:
            params.append(f"query='{query}'")
        if node_type:
            params.append(f"node_type={node_type}")
        if edge_type:
            params.append(f"edge_type={edge_type}")
        if params:
            search_desc += f" ({', '.join(params)})"

        if len(candidates) == 0:
            return f"Search around {search_desc} found no results. Consider widening your search or changing the strategy.\n"

        candidates_to_show = candidates.head(self.MAX_RESULTS_SHOWN)
        formatted_candidates = []
        for i, row in enumerate(candidates_to_show.to_dicts()):
            formatted_node = ""

            if row["edges_to_candidate"]:
                edges_to = [e for e in row["edges_to_candidate"] if e is not None]
                if edges_to:
                    formatted_node += f"-{', '.join(edges_to)}-> "

            candidate_index = row["index"]
            candidate_name = row["name"]
            candidate_type = row["type"]
            formatted_node += f"[{candidate_index}] {candidate_name} ({candidate_type})"

            if row["edges_from_candidate"]:
                edges_from = [e for e in row["edges_from_candidate"] if e is not None]
                if edges_from:
                    formatted_node += f" <-{', '.join(edges_from)}-"

            if row["match_explanation"]:
                formatted_node += f"\n{row['match_explanation']}"

            formatted_candidates.append(f"{i}. {formatted_node}")

        response = f"Search around {search_desc} found {len(candidates)} result{'s' if len(candidates) != 1 else ''}:\n\n"
        response += "\n\n".join(formatted_candidates)

        if len(candidates) > self.MAX_RESULTS_SHOWN:
            response += f"\n\n(Showing {self.MAX_RESULTS_SHOWN} of {len(candidates)} results. You may want to refine your search.)"

        return response

    def _filter_by_edge_type(
        self,
        candidates: pl.DataFrame,
        khop_edges: pl.DataFrame,
        node_index: int,
        edge_type: str,
    ):
        filtered_khop_edges = khop_edges.filter(pl.col("type") == edge_type)
        dst_indices = filtered_khop_edges.filter(pl.col("start_node_index") == node_index)[
            "end_node_index"
        ].to_list()
        src_indices = filtered_khop_edges.filter(pl.col("end_node_index") == node_index)[
            "start_node_index"
        ].to_list()
        filtered_neighbors_indices = set(dst_indices).union(set(src_indices))

        if not filtered_neighbors_indices:
            return candidates.head(0)

        return candidates.filter(pl.col("index").is_in(list(filtered_neighbors_indices)))

    def __call__(self, args: dict) -> str:
        node_index: int = args["node_index"]
        if "query" in self.schema()['function']['parameters']['properties'].keys():                        
            query: str = args.get("query", "")
        else:
            query = ""
            
        if "node_type" in self.schema()['function']['parameters']['properties'].keys():                        
            node_type: str = args.get("node_type", "")
        else:
            node_type = ""
            
        if "edge_type" in self.schema()['function']['parameters']['properties'].keys():                        
            edge_type: str = args.get("edge_type", "")
        else:
            edge_type = ""
            
        node = self.graph.get_node_by_index(node_index)
        candidates, khop_edges_df = self.graph.get_khop_subgraph(node, k=1)

        if len(candidates) > 0:
            candidates = candidates.with_columns(pl.lit("").alias("match_explanation"))

        if node_type:
            candidates = filter_by_node_type(candidates, node_type)
        if edge_type:
            candidates = self._filter_by_edge_type(candidates, khop_edges_df, node.index, edge_type)
        if query and len(candidates) > 0:
            if self.graph.search_mode == "embeddings":
                candidate_indices = candidates["index"].to_list()
                ranked_indices, ranked_scores = self.graph.search_nodes_by_indices(
                    query, candidate_indices, k=self.MAX_RESULTS_SHOWN
                )
                if ranked_indices:
                    score_map = {idx: score for idx, score in zip(ranked_indices, ranked_scores)}
                    candidates = candidates.filter(pl.col("index").is_in(ranked_indices))
                    candidates = candidates.with_columns(
                        pl.col("index")
                        .map_elements(lambda idx: score_map.get(idx, 0.0), return_dtype=pl.Float64)
                        .alias("similarity")
                    )
                    candidates = candidates.sort("similarity", descending=True)
                    candidates = candidates.with_columns(
                        pl.col("similarity")
                        .map_elements(
                            lambda s: f"Similarity: {s:.3f}",
                            return_dtype=pl.Utf8,
                        )
                        .alias("match_explanation")
                    )
                else:
                    candidates = candidates.head(0)
            else:
                candidates = filter_by_query(candidates, query)

        if len(candidates) > 0:
            def edges_to_candidate(idx: int) -> list[str]:
                return khop_edges_df.filter(
                    (pl.col("start_node_index") == node.index) & (pl.col("end_node_index") == idx)
                )["type"].to_list()

            def edges_from_candidate(idx: int) -> list[str]:
                return khop_edges_df.filter(
                    (pl.col("end_node_index") == node.index) & (pl.col("start_node_index") == idx)
                )["type"].to_list()

            candidates = candidates.with_columns(
                pl.col("index")
                .map_elements(edges_to_candidate, return_dtype=pl.List(pl.Utf8))
                .alias("edges_to_candidate"),
                pl.col("index")
                .map_elements(edges_from_candidate, return_dtype=pl.List(pl.Utf8))
                .alias("edges_from_candidate"),
            )
        else:
            candidates = candidates.with_columns(
                pl.lit("").alias("edges_to_candidate"),
                pl.lit("").alias("edges_from_candidate"),
            )

        return self._format_candidates(candidates, node, query, node_type, edge_type)

    @classmethod
    def schema(cls) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": cls.name(),
                "description": cls.description(),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_index": {
                            "type": "integer",
                            "description": "The index of the node to search around",
                        },
                        "query": {
                            "type": "string",
                            "description": "The keyword to search for. You can also leave this empty and get all nodes of the specified type",
                        },
                        "node_type": {
                            "type": "string",
                            "description": "The type of nodes to search in",
                        },
                        "edge_type": {
                            "type": "string",
                            "description": "The type of edges to traverse when finding neighbors",
                        },
                    },
                    "required": [],
                },
            },
        }
