"""Community Detection & Summarization Service — Leiden-based hierarchical communities."""

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field
from ..graph_store import Neo4jGraphStore

logger = logging.getLogger(__name__)


class CommunitySummaryReport(BaseModel):
    title: str = Field(description="Short descriptive title for this community (max 10 words)")
    summary: str = Field(description="Comprehensive paragraph summarizing key themes, facts, and connections")
    findings: List[str] = Field(default_factory=list, description="List of key findings or insights about this community")
    importance_score: float = Field(default=0.5, description="Float from 0.0 to 1.0 indicating significance")

# ── Prompts ──────────────────────────────────────────────────────────────────

COMMUNITY_SUMMARY_PROMPT = """You are an AI assistant that generates a structured summary report for a group (community) of related entities from a knowledge graph.

Given the following entities and their relationships within this community, produce a JSON report with:
1. "title": A short descriptive title for this community (max 10 words).
2. "summary": A comprehensive paragraph summarizing the key themes, facts, and connections within this community.
3. "findings": A list of 3-5 key findings or insights about this community, each as a short sentence.
4. "importance_score": A float from 0.0 to 1.0 indicating how informative or significant this community is.

Entities:
{entities}

Relationships:
{relationships}

Respond with raw JSON only, no markdown formatting or explanations.
"""


class CommunityDetectionService:
    """Detects communities in the Neo4j knowledge graph using Leiden algorithm
    and generates LLM-powered summaries for each community."""

    def __init__(
        self,
        graph_store: Neo4jGraphStore,
        llm: BaseLanguageModel,
        embeddings: Embeddings,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.graph_store = graph_store
        self.llm = llm
        self.embeddings = embeddings
        self.config = config or {}

        self.algorithm = self.config.get("algorithm", "leiden")
        self.resolution = float(self.config.get("resolution", 1.0))
        self.max_levels = int(self.config.get("max_levels", 3))

    # ── Public API ────────────────────────────────────────────────────────

    def detect_communities(self) -> Dict[str, Any]:
        """Run community detection on the Entity graph.

        Tries Neo4j GDS first (if available), falls back to Python igraph.
        Returns stats about detected communities.
        """
        self.graph_store.connect()
        logger.info(f"Starting community detection (algorithm={self.algorithm}, resolution={self.resolution})")

        # Try Neo4j GDS first
        if self._has_gds():
            logger.info("Using Neo4j GDS for community detection.")
            stats = self._detect_with_gds()
        else:
            logger.info("Neo4j GDS not available. Using Python igraph fallback.")
            stats = self._detect_with_igraph()

        logger.info(f"Community detection complete: {stats}")
        return stats

    def generate_community_summaries(self) -> Dict[str, Any]:
        """Generate LLM summaries for all detected communities at each level."""
        self.graph_store.connect()

        # Setup vector index for community summaries
        self._setup_community_vector_index()

        # Get all distinct community levels
        levels = self._get_community_levels()
        if not levels:
            return {"status": "warning", "message": "No communities found. Run detect_communities() first."}

        total_summaries = 0
        for level in levels:
            community_ids = self._get_community_ids_at_level(level)
            logger.info(f"Generating summaries for {len(community_ids)} communities at level {level}...")

            for comm_id in community_ids:
                try:
                    self._summarize_community(comm_id, level)
                    total_summaries += 1
                except Exception as e:
                    logger.error(f"Failed to summarize community {comm_id} at level {level}: {e}")

        return {
            "status": "success",
            "total_summaries": total_summaries,
            "levels": levels,
        }

    def rebuild_communities(self) -> Dict[str, Any]:
        """Full rebuild: clear old communities, re-detect, and re-summarize."""
        self.graph_store.connect()
        logger.info("Rebuilding communities...")

        # 1. Clear existing community data
        self._clear_communities()

        # 2. Re-detect
        detection_stats = self.detect_communities()

        # 3. Re-summarize
        summary_stats = self.generate_community_summaries()

        return {
            "status": "success",
            "detection": detection_stats,
            "summaries": summary_stats,
        }

    def get_community_summaries(
        self,
        level: int = 0,
        allowed_docs: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve all community summaries at a given level, optionally filtered by documents."""
        self.graph_store.connect()

        # Check if Community label exists in Neo4j to avoid DBMS schema warnings
        try:
            label_check = self.graph_store.query(
                'CALL db.labels() YIELD label WHERE label = "Community" RETURN count(label) AS cnt'
            )
            if not label_check or label_check[0].get("cnt", 0) == 0:
                return []
        except Exception:
            pass

        if allowed_docs is not None:
            cypher = """
            MATCH (c:Community {level: $level})
            OPTIONAL MATCH (c)<-[:BELONGS_TO]-(e:Entity)
            WITH c, collect(e) AS members
            WHERE any(e IN members WHERE any(doc IN e.source_documents WHERE doc IN $allowed_docs))
            RETURN c.id AS id, c.title AS title, c.summary AS summary,
                   c.findings AS findings, c.importance_score AS importance_score,
                   c.entity_count AS entity_count, c.level AS level
            ORDER BY c.importance_score DESC
            """
            results = self.graph_store.query(cypher, {"level": level, "allowed_docs": allowed_docs})
        else:
            cypher = """
            MATCH (c:Community {level: $level})
            RETURN c.id AS id, c.title AS title, c.summary AS summary,
                   c.findings AS findings, c.importance_score AS importance_score,
                   c.entity_count AS entity_count, c.level AS level
            ORDER BY c.importance_score DESC
            """
            results = self.graph_store.query(cypher, {"level": level})

        return results

    def search_communities_by_vector(
        self,
        query: str,
        top_k: int = 5,
        level: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Vector search over community summaries."""
        self.graph_store.connect()
        query_vector = self.embeddings.embed_query(query)

        cypher = """
        CALL db.index.vector.queryNodes('community_embeddings', $top_k, $vector)
        YIELD node, score
        WHERE ($level IS NULL OR node.level = $level)
        RETURN node.id AS id, node.title AS title, node.summary AS summary,
               node.findings AS findings, node.importance_score AS importance_score,
               node.level AS level, score
        """
        try:
            results = self.graph_store.query(cypher, {
                "top_k": top_k,
                "vector": query_vector,
                "level": level,
            })
            return results
        except Exception as e:
            logger.warning(f"Community vector search failed: {e}")
            return []

    def _get_community_prop_name(self, level: int) -> str:
        return f"community_level_{level}"

    def _create_community_nodes_and_relationships(self) -> None:
        """Immediately create :Community nodes and :BELONGS_TO relationships after detection."""
        for level in range(self.max_levels):
            prop_name = self._get_community_prop_name(level)
            community_ids = self._get_community_ids_at_level(level)
            for comm_id in community_ids:
                comm_node_id = f"community_lvl{level}_{comm_id}"

                cypher_comm = """
                MERGE (c:Community {id: $id})
                SET c.level = $level,
                    c.community_id = $comm_id,
                    c.title = COALESCE(c.title, $title),
                    c.summary = COALESCE(c.summary, 'Community partition created. Run generate_community_summaries() to generate LLM summary.'),
                    c.importance_score = COALESCE(c.importance_score, 0.5),
                    c.entity_count = COALESCE(c.entity_count, 0)
                """
                self.graph_store.query(cypher_comm, {
                    "id": comm_node_id,
                    "level": level,
                    "comm_id": comm_id,
                    "title": f"Community {comm_id} (level {level})"
                })

                cypher_belongs = f"""
                MATCH (e:Entity)
                WHERE e.`{prop_name}` = $comm_id
                MATCH (c:Community {{id: $comm_id_node}})
                MERGE (e)-[r:BELONGS_TO]->(c)
                SET r.level = $level
                WITH c, count(e) AS cnt
                SET c.entity_count = cnt
                """
                self.graph_store.query(cypher_belongs, {
                    "comm_id": comm_id,
                    "comm_id_node": comm_node_id,
                    "level": level,
                })

    def _has_gds(self) -> bool:
        """Check if Neo4j GDS plugin is available."""
        try:
            result = self.graph_store.query("RETURN gds.version() AS version")
            logger.info(f"Neo4j GDS version: {result[0]['version']}")
            return True
        except Exception:
            return False

    def _detect_with_gds(self) -> Dict[str, Any]:
        """Run Leiden via Neo4j GDS projected graph."""
        # 1. Create in-memory graph projection
        try:
            self.graph_store.query("CALL gds.graph.drop('entity_graph', false)")
        except Exception:
            pass  # Graph didn't exist

        self.graph_store.query("""
        CALL gds.graph.project(
            'entity_graph',
            'Entity',
            {ALL: {type: '*', orientation: 'UNDIRECTED'}}
        )
        """)

        # 2. Run Leiden community detection
        algo_name = "gds.leiden" if self.algorithm == "leiden" else "gds.louvain"
        result = self.graph_store.query(f"""
        CALL {algo_name}.write('entity_graph', {{
            writeProperty: 'community_level_0',
            resolution: $resolution
        }})
        YIELD communityCount, modularity
        RETURN communityCount, modularity
        """, {"resolution": self.resolution})

        stats = {
            "algorithm": self.algorithm,
            "backend": "neo4j_gds",
            "community_count_level_0": result[0]["communityCount"] if result else 0,
            "modularity": result[0]["modularity"] if result else 0,
        }

        # 3. Create hierarchical levels by adjusting resolution
        for level in range(self.max_levels):
            adjusted_res = self.resolution * (0.5 ** level)  # Coarser at higher levels
            prop_name = self._get_community_prop_name(level)
            try:
                r = self.graph_store.query(f"""
                CALL {algo_name}.write('entity_graph', {{
                    writeProperty: $prop,
                    resolution: $res
                }})
                YIELD communityCount
                RETURN communityCount
                """, {"prop": prop_name, "res": adjusted_res})
                stats[f"community_count_level_{level}"] = r[0]["communityCount"] if r else 0
            except Exception as e:
                logger.warning(f"Failed to detect communities at level {level}: {e}")

        # 4. Write BELONGS_TO relationships and create Community placeholder nodes
        self._create_community_nodes_and_relationships()

        # Cleanup projection
        try:
            self.graph_store.query("CALL gds.graph.drop('entity_graph', false)")
        except Exception:
            pass

        return stats

    def _detect_with_igraph(self) -> Dict[str, Any]:
        """Run Leiden using Python igraph + leidenalg library."""
        import igraph as ig
        import leidenalg

        # 1. Export graph from Neo4j
        nodes_result = self.graph_store.query("""
        MATCH (e:Entity)
        RETURN e.id AS id, e.type AS type, e.description AS description
        """)

        edges_result = self.graph_store.query("""
        MATCH (e1:Entity)-[r]->(e2:Entity)
        RETURN e1.id AS source, e2.id AS target, type(r) AS rel_type
        """)

        if not nodes_result:
            return {"algorithm": self.algorithm, "backend": "igraph", "community_count_level_0": 0}

        # 2. Build igraph Graph
        node_ids = [n["id"] for n in nodes_result]
        node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

        g = ig.Graph(n=len(node_ids), directed=False)
        g.vs["name"] = node_ids

        edges = []
        for e in edges_result:
            src_idx = node_id_to_idx.get(e["source"])
            tgt_idx = node_id_to_idx.get(e["target"])
            if src_idx is not None and tgt_idx is not None:
                edges.append((src_idx, tgt_idx))

        g.add_edges(edges)
        # Remove self-loops and multi-edges
        g.simplify()

        stats = {"algorithm": self.algorithm, "backend": "igraph"}

        # 3. Run Leiden at multiple resolution levels
        for level in range(self.max_levels):
            adjusted_res = self.resolution * (0.5 ** level)

            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=adjusted_res,
            )

            community_assignments = partition.membership
            num_communities = len(set(community_assignments))
            stats[f"community_count_level_{level}"] = num_communities

            # 4. Write community assignments back to Neo4j
            prop_name = self._get_community_prop_name(level)
            for idx, comm_id in enumerate(community_assignments):
                node_id = node_ids[idx]
                self.graph_store.query(
                    f"MATCH (e:Entity {{id: $id}}) SET e.`{prop_name}` = $comm_id",
                    {"id": node_id, "comm_id": comm_id},
                )

        # 5. Write BELONGS_TO relationships and create Community placeholder nodes
        self._create_community_nodes_and_relationships()

        return stats

    # ── Community Summarization ──────────────────────────────────────────

    def _summarize_community(self, community_id: int, level: int) -> None:
        """Generate and store a summary for a single community."""
        prop_name = self._get_community_prop_name(level)

        # Gather entities in this community
        entities_result = self.graph_store.query(f"""
        MATCH (e:Entity)
        WHERE e.`{prop_name}` = $comm_id
        RETURN e.id AS id, e.type AS type, e.description AS description
        """, {"comm_id": community_id})

        if not entities_result:
            return

        entity_ids = [e["id"] for e in entities_result]

        # Gather relationships between community members
        rels_result = self.graph_store.query(f"""
        MATCH (e1:Entity)-[r]->(e2:Entity)
        WHERE e1.`{prop_name}` = $comm_id AND e2.`{prop_name}` = $comm_id
        RETURN e1.id AS source, type(r) AS rel, e2.id AS target, r.description AS description
        """, {"comm_id": community_id})

        # Format for prompt
        entities_text = "\n".join(
            f"- {e['id']} ({e['type']}): {e['description']}" for e in entities_result
        )
        rels_text = "\n".join(
            f"- {r['source']} -[{r['rel']}]-> {r['target']}: {r.get('description', '')}"
            for r in rels_result
        ) if rels_result else "No internal relationships."

        # Generate summary with LLM using Structured Output
        prompt = COMMUNITY_SUMMARY_PROMPT.format(entities=entities_text, relationships=rels_text)
        report_dict = None
        if hasattr(self.llm, "with_structured_output"):
            try:
                structured_llm = self.llm.with_structured_output(CommunitySummaryReport)
                res = structured_llm.invoke(prompt)
                if isinstance(res, CommunitySummaryReport):
                    report_dict = res.dict()
                elif isinstance(res, dict):
                    report_dict = res
            except Exception as e:
                logger.warning(f"Structured output for community summary failed ({e}), falling back to text parsing.")

        if not report_dict:
            try:
                response = self.llm.invoke(prompt)
                raw_content = getattr(response, "content", response)
                if isinstance(raw_content, str):
                    response_text = raw_content
                elif isinstance(raw_content, list):
                    text_parts = []
                    for part in raw_content:
                        if isinstance(part, str):
                            text_parts.append(part)
                        elif isinstance(part, dict):
                            if part.get("type") == "text" and part.get("text"):
                                text_parts.append(part["text"])
                            elif "text" in part and isinstance(part["text"], str):
                                text_parts.append(part["text"])
                        elif hasattr(part, "text") and getattr(part, "text"):
                            text_parts.append(str(getattr(part, "text")))
                    response_text = "".join(text_parts)
                else:
                    response_text = str(raw_content)

                response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)

                # Parse JSON
                json_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
                if json_match:
                    report_dict = json.loads(json_match.group(1))
                else:
                    report_dict = json.loads(response_text)
            except Exception as e:
                logger.error(f"Failed to generate/parse community summary: {e}")
                report_dict = {
                    "title": f"Community {community_id}",
                    "summary": f"A group of {len(entities_result)} related entities.",
                    "findings": [],
                    "importance_score": 0.5,
                }
        report = report_dict

        # Generate embedding for the summary
        summary_text = f"{report.get('title', '')}: {report.get('summary', '')}"
        embedding = self.embeddings.embed_query(summary_text)

        # Create unique community node ID
        comm_node_id = f"community_lvl{level}_{community_id}"

        # Store Community node in Neo4j
        self.graph_store.query("""
        MERGE (c:Community {id: $id})
        SET c.level = $level,
            c.title = $title,
            c.summary = $summary,
            c.findings = $findings,
            c.importance_score = $importance_score,
            c.entity_count = $entity_count,
            c.embedding = $embedding,
            c.community_id = $community_id
        """, {
            "id": comm_node_id,
            "level": level,
            "title": report.get("title", f"Community {community_id}"),
            "summary": report.get("summary", ""),
            "findings": json.dumps(report.get("findings", []), ensure_ascii=False),
            "importance_score": float(report.get("importance_score", 0.5)),
            "entity_count": len(entities_result),
            "embedding": embedding,
            "community_id": community_id,
        })

        # Create BELONGS_TO relationships from entities to this community
        for entity_id in entity_ids:
            self.graph_store.query("""
            MATCH (e:Entity {id: $entity_id})
            MATCH (c:Community {id: $comm_id})
            MERGE (e)-[r:BELONGS_TO]->(c)
            SET r.level = $level
            """, {
                "entity_id": entity_id,
                "comm_id": comm_node_id,
                "level": level,
            })

    # ── Helpers ───────────────────────────────────────────────────────────

    def _write_belongs_to_relationships(self) -> None:
        """Write BELONGS_TO relationships from community property to Community nodes.
        Called after detection to ensure Community nodes exist before summarization."""
        # Community nodes are created during summarization, so this is a no-op placeholder.
        # The actual BELONGS_TO edges are created in _summarize_community().
        pass

    def _get_community_levels(self) -> List[int]:
        """Get all distinct community levels that have been detected."""
        levels = []
        for level in range(self.max_levels):
            prop_name = self._get_community_prop_name(level)
            try:
                result = self.graph_store.query(f"""
                MATCH (e:Entity)
                WHERE e.`{prop_name}` IS NOT NULL
                RETURN DISTINCT 1 AS has_data LIMIT 1
                """)
                if result:
                    levels.append(level)
            except Exception:
                pass
        return levels

    def _get_community_ids_at_level(self, level: int) -> List[int]:
        """Get all distinct community IDs at a given level."""
        prop_name = self._get_community_prop_name(level)
        result = self.graph_store.query(f"""
        MATCH (e:Entity)
        WHERE e.`{prop_name}` IS NOT NULL
        RETURN DISTINCT e.`{prop_name}` AS comm_id
        ORDER BY comm_id
        """)
        return [r["comm_id"] for r in result]

    def _clear_communities(self) -> None:
        """Remove community nodes and BELONGS_TO relationships."""
        self.graph_store.query("MATCH (c:Community) DETACH DELETE c")

        # Remove community properties from entities
        for level in range(self.max_levels):
            prop_name = self._get_community_prop_name(level)
            try:
                self.graph_store.query(f"""
                MATCH (e:Entity)
                WHERE e.`{prop_name}` IS NOT NULL
                REMOVE e.`{prop_name}`
                """)
            except Exception:
                pass

        logger.info("Cleared existing community data.")

    def _setup_community_vector_index(self) -> None:
        """Create vector index for community summary embeddings."""
        if getattr(self, "_index_initialized", False):
            return
        test_emb = self.embeddings.embed_query("test")
        dimension = len(test_emb)

        try:
            check = self.graph_store.query("SHOW INDEXES YIELD name WHERE name = 'community_embeddings' RETURN name")
            if not check:
                self.graph_store.query("""
                CREATE VECTOR INDEX `community_embeddings` IF NOT EXISTS
                FOR (n:Community) ON (n.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: $dimension,
                    `vector.similarity_function`: 'cosine'
                }}
                """, {"dimension": dimension})
                logger.info("Successfully set up Neo4j vector index 'community_embeddings'.")
            self._index_initialized = True
        except Exception as e:
            logger.warning(f"Could not create community vector index: {e}")
