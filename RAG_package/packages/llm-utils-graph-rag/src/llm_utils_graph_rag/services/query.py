"""Graph Retrieval Service — retrieves semantic nodes and paths from Neo4j.

Supports Local Search (entity-focused) with optional community context enrichment.
"""

import logging
from typing import Any, Dict, List, Set, Tuple, Optional
from langchain_core.embeddings import Embeddings
from ..graph_store import Neo4jGraphStore
from llm_utils_reranker import BaseReranker

logger = logging.getLogger(__name__)


class GraphQueryService:
    """Handles Vector-based node lookup and graph traversal retrieval."""

    def __init__(
        self,
        graph_store: Neo4jGraphStore,
        embeddings: Embeddings,
        reranker: Optional[BaseReranker] = None,
    ):
        self.graph_store = graph_store
        self.embeddings = embeddings
        self.reranker = reranker

    def retrieve_relevant_subgraph(
        self,
        query: str,
        top_k_entities: int = 5,
        allowed_docs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Lookup nodes by vector similarity, apply reranking if configured, and fetch surrounding relationships."""
        self.graph_store.connect()

        # 1. Embed query
        query_vector = self.embeddings.embed_query(query)

        # 2. Vector search in Neo4j for closest entity nodes
        # If reranking is enabled, fetch a larger candidate pool
        top_k_search = max(top_k_entities * 3, 15) if self.reranker else top_k_entities

        if allowed_docs is not None:
            vector_search_cypher = """
            CALL db.index.vector.queryNodes('entity_embeddings', $top_k, $vector)
            YIELD node, score
            WHERE any(doc IN node.source_documents WHERE doc IN $allowed_docs)
            RETURN node.id AS id, node.type AS type, node.description AS description, score
            """
        else:
            vector_search_cypher = """
            CALL db.index.vector.queryNodes('entity_embeddings', $top_k, $vector)
            YIELD node, score
            RETURN node.id AS id, node.type AS type, node.description AS description, score
            """
        
        try:
            params = {
                "top_k": top_k_search,
                "vector": query_vector
            }
            if allowed_docs is not None:
                params["allowed_docs"] = allowed_docs
                
            matched_nodes = self.graph_store.query(vector_search_cypher, params)
        except Exception as e:
            logger.warning(f"Neo4j vector search query failed: {e}. Falling back to name matching.")
            # Fallback to simple ILIKE search on node IDs if vector search is not configured yet
            if allowed_docs is not None:
                fallback_cypher = """
                MATCH (n:Entity)
                WHERE (n.id CONTAINS toUpper($query) OR n.description CONTAINS $query)
                  AND any(doc IN n.source_documents WHERE doc IN $allowed_docs)
                RETURN n.id AS id, n.type AS type, n.description AS description, 1.0 AS score
                LIMIT $top_k
                """
            else:
                fallback_cypher = """
                MATCH (n:Entity)
                WHERE n.id CONTAINS toUpper($query) OR n.description CONTAINS $query
                RETURN n.id AS id, n.type AS type, n.description AS description, 1.0 AS score
                LIMIT $top_k
                """
            try:
                fallback_params = {
                    "query": query,
                    "top_k": top_k_search
                }
                if allowed_docs is not None:
                    fallback_params["allowed_docs"] = allowed_docs
                matched_nodes = self.graph_store.query(fallback_cypher, fallback_params)
            except Exception as fe:
                logger.error(f"Fallback graph retrieval failed: {fe}")
                return {"entities": [], "relationships": [], "context_str": ""}

        if not matched_nodes:
            return {"entities": [], "relationships": [], "context_str": "No matching knowledge graph entities found."}

        # Apply reranking if configured
        if self.reranker and len(matched_nodes) > 1:
            from langchain_core.documents import Document
            
            docs = []
            for node in matched_nodes:
                content = f"{node['id']} ({node['type']}): {node['description']}"
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "id": node["id"],
                        "type": node["type"],
                        "description": node["description"],
                        "score": node.get("score", 1.0)
                    }
                ))
            
            reranked_docs = self.reranker.rerank_sync(query, docs, top_k=top_k_entities)
            
            matched_nodes = []
            for doc in reranked_docs:
                matched_nodes.append({
                    "id": doc.metadata["id"],
                    "type": doc.metadata["type"],
                    "description": doc.metadata["description"],
                    "score": doc.metadata.get("rerank_score", doc.metadata.get("score", 1.0))
                })

        # 3. Retrieve 1-hop relationships for the matched nodes
        relationships: List[Dict[str, Any]] = []
        seen_triplets: Set[Tuple[str, str, str]] = set()

        for node in matched_nodes:
            node_id = node["id"]
            if allowed_docs is not None:
                traversal_cypher = """
                MATCH (n:Entity {id: $node_id})-[r]-(m:Entity)
                WHERE any(doc IN r.source_documents WHERE doc IN $allowed_docs)
                RETURN n.id AS source, type(r) AS rel, m.id AS target, r.description AS description
                LIMIT 15
                """
            else:
                traversal_cypher = """
                MATCH (n:Entity {id: $node_id})-[r]-(m:Entity)
                RETURN n.id AS source, type(r) AS rel, m.id AS target, r.description AS description
                LIMIT 15
                """
            try:
                traversal_params = {"node_id": node_id}
                if allowed_docs is not None:
                    traversal_params["allowed_docs"] = allowed_docs
                edges = self.graph_store.query(traversal_cypher, traversal_params)
                for edge in edges:
                    triplet = (edge["source"], edge["rel"], edge["target"])
                    # De-duplicate relationships
                    reversed_triplet = (edge["target"], edge["rel"], edge["source"])
                    if triplet not in seen_triplets and reversed_triplet not in seen_triplets:
                        seen_triplets.add(triplet)
                        relationships.append({
                            "source": edge["source"],
                            "relationship": edge["rel"],
                            "target": edge["target"],
                            "description": edge.get("description", "")
                        })
            except Exception as e:
                logger.error(f"Error traversing relationships for node {node_id}: {e}")

        # 4. Enrich with community context (if communities exist)
        community_context = self._get_community_context_for_entities(matched_nodes)

        # 5. Synthesize context string
        context_lines = ["Matched Entities:"]
        for node in matched_nodes:
            context_lines.append(f"- {node['id']} ({node['type']}): {node['description']} (Score: {node['score']:.4f})")
            
        if relationships:
            context_lines.append("\nKnowledge Graph Relationships:")
            for rel in relationships:
                desc_str = f" ({rel['description']})" if rel["description"] else ""
                context_lines.append(f"- {rel['source']} -[{rel['relationship']}]-> {rel['target']}{desc_str}")
        else:
            context_lines.append("\nNo connected graph relationships found for these entities.")

        if community_context:
            context_lines.append("\nCommunity Context (higher-level themes):")
            for cc in community_context:
                context_lines.append(f"- [{cc['title']}]: {cc['summary']}")

        context_str = "\n".join(context_lines)

        return {
            "entities": matched_nodes,
            "relationships": relationships,
            "community_context": community_context,
            "context_str": context_str,
            "search_mode": "local",
        }

    def _get_community_context_for_entities(
        self, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Retrieve community summaries that contain the matched entities."""
        if not entities:
            return []

        entity_ids = [e["id"] for e in entities]

        try:
            result = self.graph_store.query("""
            MATCH (e:Entity)-[:BELONGS_TO]->(c:Community)
            WHERE e.id IN $entity_ids
            RETURN DISTINCT c.id AS id, c.title AS title, c.summary AS summary,
                   c.importance_score AS importance_score
            ORDER BY c.importance_score DESC
            LIMIT 3
            """, {"entity_ids": entity_ids})
            return result
        except Exception as e:
            logger.debug(f"Community context lookup skipped (communities may not exist): {e}")
            return []
