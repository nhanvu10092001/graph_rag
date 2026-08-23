import re
import logging
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from llm_utils_graph_rag.graph_store import Neo4jGraphStore

logger = logging.getLogger(__name__)

def sanitize_lucene_query(query: str) -> str:
    """Escape special characters for Neo4j Lucene fulltext search."""
    special_chars = r'[\+\-\&\|\!\(\)\{\}\[\]\^\"\~\*\?\:\\]'
    sanitized = re.sub(special_chars, ' ', query)
    # Return as an AND-joined string of words for stricter matching, or let it be OR-joined
    words = [w for w in sanitized.split() if w.strip()]
    return " AND ".join(words) if words else sanitized


class GlobalSearchInput(BaseModel):
    subquery: str = Field(description="Subquery string to search node text attributes.")
    k: int = Field(default=5, description="Maximum number of candidate nodes to retrieve.")


class GlobalSearchTool(BaseTool):
    name: str = "global_search"
    description: str = "Retrieves broad candidate nodes across the graph matching the subquery using fulltext search."
    args_schema: Type[BaseModel] = GlobalSearchInput

    graph_store: Neo4jGraphStore

    def _run(self, subquery: str, k: int = 5, run_manager=None) -> str:
        sanitized_query = sanitize_lucene_query(subquery)
        if not sanitized_query:
            return "Query is empty after sanitization."

        cypher = """
        CALL db.index.fulltext.queryNodes('entity_fulltext', $search_phrase) YIELD node, score
        RETURN node.id AS id, node.type AS type, node.description AS description,
               node.source_documents AS source_documents, score
        LIMIT $k
        """
        params = {
            "search_phrase": sanitized_query,
            "k": k
        }

        try:
            results = self.graph_store.query(cypher, params)
            if not results:
                return f"No nodes found for subquery: '{subquery}'"

            output = []
            for r in results:
                output.append(f"Node ID: {r['id']} | Type: {r['type']} | Score: {r['score']:.3f}\nDescription: {r['description']}")

            return "\n\n".join(output)
        except Exception as e:
            logger.error(f"GlobalSearchTool error: {e}")
            return f"Error executing global search: {e}"


class NeighborhoodExplorationInput(BaseModel):
    node_id: str = Field(description="ID of the starting node to expand from.")
    subquery: str = Field(description="Subquery to rank the 1-hop neighbor nodes.")
    filter_node_types: Optional[List[str]] = Field(default=None, description="Optional allowed node types.")
    filter_edge_types: Optional[List[str]] = Field(default=None, description="Optional allowed relation edge types.")
    k: int = Field(default=5, description="Maximum number of neighbors to return.")


class NeighborhoodExplorationTool(BaseTool):
    name: str = "neighborhood_exploration"
    description: str = "Retrieves 1-hop adjacent nodes from node_id, filtered by node/edge types and ranked by subquery."
    args_schema: Type[BaseModel] = NeighborhoodExplorationInput

    graph_store: Neo4jGraphStore

    def _run(
        self,
        node_id: str,
        subquery: str,
        filter_node_types: Optional[List[str]] = None,
        filter_edge_types: Optional[List[str]] = None,
        k: int = 5,
        run_manager=None
    ) -> str:
        sanitized_query = sanitize_lucene_query(subquery)
        if not sanitized_query:
            return "Subquery is empty after sanitization."

        cypher = """
        MATCH (n:Entity {id: $node_id})-[r]-(m:Entity)
        WHERE ($filter_node_types IS NULL OR size($filter_node_types) = 0 OR m.type IN $filter_node_types)
          AND ($filter_edge_types IS NULL OR size($filter_edge_types) = 0 OR type(r) IN $filter_edge_types)
        RETURN DISTINCT m.id AS id, m.type AS type, m.description AS description,
                        type(r) AS rel_type, r.description AS rel_description
        """
        params = {
            "node_id": node_id,
            "filter_node_types": filter_node_types or [],
            "filter_edge_types": filter_edge_types or []
        }

        try:
            results = self.graph_store.query(cypher, params)
            if not results:
                return f"No neighbors found for node '{node_id}' with given filters."

            # Simple python-based text scoring (term overlap) since we have a subquery
            # For a production robust approach we could use rank_bm25, but basic overlap works for small neighbor sets.
            # Let's import BM25 from the existing RAG package if possible, or use a simple heuristic.
            try:
                from rank_bm25 import BM25Okapi
                corpus = [f"{r['id']} {r['description']}".lower().split() for r in results]
                tokenized_query = sanitized_query.lower().split()
                bm25 = BM25Okapi(corpus)
                scores = bm25.get_scores(tokenized_query)
                for i, r in enumerate(results):
                    r['score'] = scores[i]

                # Sort by score DESC
                results = sorted(results, key=lambda x: x['score'], reverse=True)
            except ImportError:
                # Fallback if rank_bm25 is not available
                logger.warning("rank_bm25 not installed, falling back to simple term overlap for neighborhood ranking.")
                tokenized_query = set(sanitized_query.lower().split())
                for r in results:
                    text_tokens = set(f"{r['id']} {r['description']}".lower().split())
                    r['score'] = len(tokenized_query.intersection(text_tokens))
                results = sorted(results, key=lambda x: x['score'], reverse=True)

            # Limit to top k
            top_results = results[:k]

            output = []
            for r in top_results:
                output.append(
                    f"Neighbor ID: {r['id']} | Type: {r['type']} | Rel: -[{r['rel_type']}]- | Score: {r['score']:.3f}\n"
                    f"Description: {r['description']}\nRel Desc: {r.get('rel_description', '')}"
                )

            return "\n\n".join(output)
        except Exception as e:
            logger.error(f"NeighborhoodExplorationTool error: {e}")
            return f"Error executing neighborhood exploration: {e}"


class AnswerNodeEntry(BaseModel):
    node_id: str = Field(description="The ID of the node to add as an answer.")
    reasoning: str = Field(description="Explanation of why this node is relevant to answering the question.")


class AddToAnswerInput(BaseModel):
    answer_nodes: List[AnswerNodeEntry] = Field(
        description="List of nodes to add to the answer. Each entry must include the node_id and your reasoning for selecting it. Pass EVERY node that could be an answer."
    )


class AddToAnswerTool(BaseTool):
    name: str = "add_to_answer"
    description: str = (
        "Add selected nodes to your answer. Call this after finding relevant nodes via search. "
        "Each node must include the node_id (as seen in search results) and your reasoning for why it is relevant."
    )
    args_schema: Type[BaseModel] = AddToAnswerInput

    def _run(self, answer_nodes: List[Dict[str, Any]], run_manager=None) -> str:
        if not answer_nodes:
            return "No nodes were provided to add to the answer."

        added_ids = []
        for entry in answer_nodes:
            node_id = entry.get("node_id", "") if isinstance(entry, dict) else entry.node_id
            if node_id:
                added_ids.append(node_id)

        if not added_ids:
            return "No valid node IDs found in the provided answer_nodes."

        return f"Added {len(added_ids)} node(s) to answer: {added_ids}"


class FinishInput(BaseModel):
    comment: str = Field(default="", description="Optional comment explaining why exploration is complete.")


class FinishTool(BaseTool):
    name: str = "finish"
    description: str = (
        "Signal that exploration is complete. Call this when you have found all relevant nodes "
        "and added them to the answer via add_to_answer, or when you have exhausted useful exploration paths."
    )
    args_schema: Type[BaseModel] = FinishInput

    def _run(self, comment: str = "", run_manager=None) -> str:
        if comment:
            return f"Exploration complete. {comment}"
        return "Exploration complete."
