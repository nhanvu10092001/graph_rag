"""Global Search Service — Map-Reduce over community summaries for big-picture queries."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from ..graph_store import Neo4jGraphStore
from .community import CommunityDetectionService

logger = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────────────────────

COMMUNITY_RELEVANCE_PROMPT = """You are a relevance judge. Given a user query and a community summary from a knowledge graph, determine if this community contains information relevant to answering the query.

Query: {query}

Community Title: {title}
Community Summary: {summary}

Respond with ONLY "relevant" or "irrelevant". No other words.
"""

MAP_PROMPT = """You are an analytical assistant. Based on the community information below from a knowledge graph, generate a partial answer to the user's query. Focus only on what this community can contribute.

Query: {query}

Community: {title}
Summary: {summary}
Key Findings: {findings}

Member Entities:
{entities}

Key Relationships:
{relationships}

Generate a concise, factual partial answer based only on this community's information. If this community has no relevant information, respond with "NO_RELEVANT_INFO".
"""

REDUCE_PROMPT = """You are a synthesis assistant. Multiple partial answers have been generated from different communities (groups of related entities) in a knowledge graph. Combine them into one coherent, comprehensive final answer.

Query: {query}

Partial Answers:
{partial_answers}

Instructions:
- Synthesize all partial answers into a single coherent response.
- Remove redundancies but preserve all unique information.
- If there are contradictions, note them.
- If all partial answers indicate no relevant info, say so clearly.
- Structure the answer clearly with key points.
"""


def _get_content_str(response: Any) -> str:
    """Safely extract string text content from LLM response (handling string or list of content blocks)."""
    raw_content = getattr(response, "content", response)
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
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
        return "".join(text_parts)
    return str(raw_content)


class GlobalSearchService:
    """Handles global (corpus-wide) queries using map-reduce over community summaries."""

    def __init__(
        self,
        graph_store: Neo4jGraphStore,
        llm: BaseLanguageModel,
        embeddings: Embeddings,
        community_service: CommunityDetectionService,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.graph_store = graph_store
        self.llm = llm
        self.embeddings = embeddings
        self.community_service = community_service
        self.config = config or {}

    def search(
        self,
        query: str,
        level: int = 0,
        max_communities: int = 10,
        allowed_docs: Optional[List[str]] = None,
        group_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute a global search using map-reduce over community summaries.

        Args:
            query: The user's question.
            level: Community hierarchy level (0 = finest, higher = coarser).
            max_communities: Max number of communities to process in map phase.
            allowed_docs: Optional document filter for group-based access control.
            group_id: Optional group ID filter.

        Returns:
            Dict with response, partial_answers, selected_communities, etc.
        """
        target_db = f"group_{group_id}" if group_id is not None else None
        self.graph_store.connect(database=target_db)
        logger.info(f"Global Search: query='{query}', level={level}, group_id={group_id}, database={target_db}")

        # 1. Dynamic Community Selection — find relevant communities
        selected_communities = self._select_relevant_communities(
            query, level, max_communities, allowed_docs, group_id
        )

        if not selected_communities:
            return {
                "response": "No relevant communities found in the knowledge graph for this query.",
                "partial_answers": [],
                "selected_communities": [],
                "search_mode": "global",
            }

        logger.info(f"Selected {len(selected_communities)} relevant communities for map phase.")

        # 2. Map Phase — generate partial answers from each community
        partial_answers = []
        for community in selected_communities:
            try:
                partial = self._map_community(query, community, group_id=group_id)
                if partial and "NO_RELEVANT_INFO" not in partial:
                    partial_answers.append({
                        "community_id": community.get("id", "unknown"),
                        "community_title": community.get("title", "Unknown"),
                        "answer": partial,
                    })
            except Exception as e:
                logger.error(f"Map phase failed for community {community.get('id')}: {e}")

        if not partial_answers:
            return {
                "response": "The knowledge graph communities did not contain relevant information for this query.",
                "partial_answers": [],
                "selected_communities": [c.get("title") for c in selected_communities],
                "search_mode": "global",
            }

        # 3. Reduce Phase — synthesize partial answers
        final_response = self._reduce(query, partial_answers)

        return {
            "response": final_response,
            "partial_answers": partial_answers,
            "selected_communities": [c.get("title") for c in selected_communities],
            "search_mode": "global",
        }

    # ── Community Selection ──────────────────────────────────────────────

    def _select_relevant_communities(
        self,
        query: str,
        level: int,
        max_communities: int,
        allowed_docs: Optional[List[str]] = None,
        group_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Select relevant communities using a two-stage approach:
        1. Vector similarity on community summaries (broad retrieval)
        2. LLM relevance filtering (precision)
        """
        # Stage 1: Vector search to get candidate communities
        candidates = self.community_service.search_communities_by_vector(
            query, top_k=max_communities * 2, level=level, group_id=group_id
        )

        if not candidates:
            # Fallback: get all communities at this level
            candidates = self.community_service.get_community_summaries(
                level=level, allowed_docs=allowed_docs, group_id=group_id
            )

        if not candidates:
            return []

        # If document or group filtering is needed and we got results from vector search,
        # apply filter now
        if allowed_docs is not None or group_id is not None:
            target_db = f"group_{group_id}" if group_id is not None else None
            filtered = []
            for c in candidates:
                comm_id = c.get("id", "")
                # Check if any entity in this community belongs to allowed docs or group_id
                check_result = self.graph_store.query("""
                MATCH (e:Entity)-[r:BELONGS_TO]->(c:Community {id: $comm_id})
                WHERE ($group_id IS NULL OR c.group_id = $group_id OR $group_id IN COALESCE(e.group_ids, []))
                  AND ($allowed_docs IS NULL OR any(doc IN e.source_documents WHERE doc IN $allowed_docs))
                RETURN count(e) AS cnt
                """, {"comm_id": comm_id, "allowed_docs": allowed_docs, "group_id": group_id}, database=target_db)
                if check_result and check_result[0]["cnt"] > 0:
                    filtered.append(c)
            candidates = filtered

        # Stage 2: LLM relevance filtering
        relevant = []
        for candidate in candidates[:max_communities * 2]:
            if self._is_community_relevant(query, candidate):
                relevant.append(candidate)
                if len(relevant) >= max_communities:
                    break

        return relevant

    def _is_community_relevant(self, query: str, community: Dict[str, Any]) -> bool:
        """Use LLM to determine if a community is relevant to the query."""
        title = community.get("title", "")
        summary = community.get("summary", "")

        if not summary:
            return False

        prompt = COMMUNITY_RELEVANCE_PROMPT.format(
            query=query, title=title, summary=summary
        )

        try:
            response = self.llm.invoke(prompt)
            result = _get_content_str(response).strip().lower()
            result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
            return "relevant" in result
        except Exception as e:
            logger.warning(f"Relevance check failed: {e}. Assuming relevant.")
            return True

    # ── Map Phase ────────────────────────────────────────────────────────

    def _map_community(self, query: str, community: Dict[str, Any], group_id: Optional[int] = None) -> str:
        """Generate a partial answer from a single community's context."""
        target_db = f"group_{group_id}" if group_id is not None else None
        comm_id = community.get("id", "")
        title = community.get("title", "")
        summary = community.get("summary", "")
        findings = community.get("findings", "[]")

        # Parse findings
        if isinstance(findings, str):
            try:
                findings_list = json.loads(findings)
                findings_text = "\n".join(f"- {f}" for f in findings_list)
            except json.JSONDecodeError:
                findings_text = findings
        elif isinstance(findings, list):
            findings_text = "\n".join(f"- {f}" for f in findings)
        else:
            findings_text = str(findings)

        # Fetch member entities
        entities_result = self.graph_store.query("""
        MATCH (e:Entity)-[:BELONGS_TO]->(c:Community {id: $comm_id})
        RETURN e.id AS id, e.type AS type, e.description AS description
        LIMIT 20
        """, {"comm_id": comm_id}, database=target_db)

        entities_text = "\n".join(
            f"- {e['id']} ({e['type']}): {e['description']}" for e in entities_result
        ) if entities_result else "No entities available."

        # Fetch internal relationships
        rels_result = self.graph_store.query("""
        MATCH (e1:Entity)-[:BELONGS_TO]->(c:Community {id: $comm_id})
        MATCH (e2:Entity)-[:BELONGS_TO]->(c)
        MATCH (e1)-[r]->(e2)
        WHERE NOT type(r) = 'BELONGS_TO'
        RETURN e1.id AS source, type(r) AS rel, e2.id AS target, r.description AS description
        LIMIT 30
        """, {"comm_id": comm_id}, database=target_db)

        rels_text = "\n".join(
            f"- {r['source']} -[{r['rel']}]-> {r['target']}: {r.get('description', '')}"
            for r in rels_result
        ) if rels_result else "No internal relationships."

        prompt = MAP_PROMPT.format(
            query=query,
            title=title,
            summary=summary,
            findings=findings_text,
            entities=entities_text,
            relationships=rels_text,
        )

        response = self.llm.invoke(prompt)
        result = _get_content_str(response)
        result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
        return result

    # ── Reduce Phase ─────────────────────────────────────────────────────

    def _reduce(self, query: str, partial_answers: List[Dict[str, Any]]) -> str:
        """Synthesize partial answers into a final coherent response."""
        # Format partial answers
        pa_text = ""
        for i, pa in enumerate(partial_answers, 1):
            pa_text += f"\n--- Partial Answer {i} (from: {pa['community_title']}) ---\n"
            pa_text += pa["answer"]
            pa_text += "\n"

        prompt = REDUCE_PROMPT.format(query=query, partial_answers=pa_text)

        response = self.llm.invoke(prompt)
        result = _get_content_str(response)
        result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
        return result
