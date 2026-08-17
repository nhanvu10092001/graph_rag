"""LangGraph orchestration workflow for Neo4j Graph RAG with self-correcting Cypher loop."""

import logging
import re
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import END, START, StateGraph
from ..graph_store import Neo4jGraphStore
from ..services.query import GraphQueryService

logger = logging.getLogger(__name__)


class GraphRAGState(TypedDict):
    """Workflow state structure."""

    query: str
    routing: str  # "vector", "graph", "hybrid"
    cypher_query: str
    cypher_records: List[Dict[str, Any]]
    graph_context: str
    response: str
    error: str
    iteration: int
    schema: str


class LangGraphRAGWorkflow:
    """Orchestrates RAG retrieval and self-correcting query execution using LangGraph."""

    def __init__(
        self,
        graph_store: Neo4jGraphStore,
        graph_query_service: GraphQueryService,
        llm: BaseLanguageModel,
    ):
        self.graph_store = graph_store
        self.graph_query_service = graph_query_service
        self.llm = llm

    def _route_query(self, state: GraphRAGState) -> Dict[str, Any]:
        """Node: Decides retrieval routing."""
        query = state["query"]
        prompt = f"""Analyze the user query and choose the best retrieval strategy.
Options:
- "graph": Use this if the query asks about connections, paths, structures, relationships, or requires executing database joins (e.g., "Who does Alice work with?", "How is X connected to Y?").
- "hybrid": Use this for general informational queries where both keyword/semantic lookup of concepts AND local relationships would be useful.
- "vector": Use this if the query is a simple factual lookup of a single topic or document snippet.

Respond with ONLY one word: "graph", "hybrid", or "vector".

Query: {query}"""
        try:
            res = self.llm.invoke(prompt)
            route = res.content.strip().lower() if hasattr(res, "content") else str(res).strip().lower()
            route = re.sub(r"<think>.*?</think>", "", route, flags=re.DOTALL).strip()
            # Sanitize
            if route not in ["graph", "hybrid", "vector"]:
                route = "hybrid"
        except Exception:
            route = "hybrid"

        logger.info(f"Query routing decision: {route}")
        return {"routing": route, "iteration": 0, "schema": self.graph_store.get_schema()}

    def _generate_cypher(self, state: GraphRAGState) -> Dict[str, Any]:
        """Node: Generate Cypher query using Neo4j schema."""
        query = state["query"]
        schema = state["schema"]

        prompt = f"""Based on the Neo4j Graph Schema below, write a Cypher query to answer the user's question.

Schema:
{schema}

Question:
{query}

Guidelines:
- Return ONLY the executable Cypher query. Do not wrap in markdown or include explanations.
- Query should match Entity nodes using: MATCH (n:Entity {{id: ...}}) or MATCH (n:Entity) WHERE n.id = ...
- Retrieve properties like n.type, n.description, and relationships.
- Keep output concise and correct.
"""
        res = self.llm.invoke(prompt)
        cypher = res.content if hasattr(res, "content") else str(res)
        cypher = re.sub(r"<think>.*?</think>", "", cypher, flags=re.DOTALL)
        # Strip markdown syntax if LLM ignored instructions
        cypher = re.sub(r"```cypher|```", "", cypher).strip()
        
        logger.info(f"Generated Cypher: {cypher}")
        return {"cypher_query": cypher, "error": ""}

    def _execute_cypher(self, state: GraphRAGState) -> Dict[str, Any]:
        """Node: Execute Cypher and handle errors."""
        cypher = state["cypher_query"]
        try:
            records = self.graph_store.query(cypher)
            logger.info(f"Cypher query executed successfully. Retrieved {len(records)} records.")
            
            # Format records to text context
            context_parts = []
            for i, record in enumerate(records):
                record_str = ", ".join([f"{k}: {v}" for k, v in record.items()])
                context_parts.append(f"Record {i+1}: {record_str}")
            
            return {
                "cypher_records": records,
                "graph_context": "\n".join(context_parts) if context_parts else "No matching graph database records found.",
                "error": ""
            }
        except Exception as e:
            logger.warning(f"Cypher execution failed: {e}")
            return {"error": str(e)}

    def _correct_cypher(self, state: GraphRAGState) -> Dict[str, Any]:
        """Node: Self-correct failing Cypher queries."""
        cypher = state["cypher_query"]
        error = state["error"]
        schema = state["schema"]
        iteration = state["iteration"] + 1

        logger.info(f"Auto-correcting Cypher (Iteration {iteration})...")

        prompt = f"""The following Cypher query failed to run on Neo4j.
Query:
{cypher}

Error received:
{error}

Graph Schema:
{schema}

Please rewrite the Cypher query to fix the syntax or schema issue. Return ONLY the new Cypher query. No explanations.
"""
        res = self.llm.invoke(prompt)
        new_cypher = res.content if hasattr(res, "content") else str(res)
        new_cypher = re.sub(r"<think>.*?</think>", "", new_cypher, flags=re.DOTALL)
        new_cypher = re.sub(r"```cypher|```", "", new_cypher).strip()

        return {"cypher_query": new_cypher, "iteration": iteration, "error": ""}

    def _fallback_retrieve(self, state: GraphRAGState) -> Dict[str, Any]:
        """Node: Fallback retrieval (Vector search + traversal) when Cypher fails or isn't chosen."""
        query = state["query"]
        res = self.graph_query_service.retrieve_relevant_subgraph(query)
        return {"graph_context": res["context_str"]}

    def _synthesize_response(self, state: GraphRAGState) -> Dict[str, Any]:
        """Node: Synthesizes final answer from context."""
        query = state["query"]
        context = state.get("graph_context", "")

        prompt = f"""You are an intelligent knowledge assistant. Answer the user question based on the provided Knowledge Graph context.

Context:
{context}

Question:
{query}

Answer:"""
        res = self.llm.invoke(prompt)
        response_text = res.content if hasattr(res, "content") else str(res)
        response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
        return {"response": response_text}

    # ── Conditional Routing logic ──────────────────────────────────────

    def _decide_execution_path(self, state: GraphRAGState) -> str:
        """Route from Query Router."""
        routing = state["routing"]
        if routing == "graph":
            return "generate_cypher"
        else:
            return "fallback_retrieve"

    def _check_execution_result(self, state: GraphRAGState) -> str:
        """Check if execution failed, deciding to correct or exit."""
        error = state.get("error", "")
        iteration = state.get("iteration", 0)

        if error:
            if iteration < 3:
                return "correct_cypher"
            else:
                logger.warning("Max auto-correction attempts reached. Falling back to traversal retrieval.")
                return "fallback_retrieve"
        return "synthesize"

    # ── Compile Graph ──────────────────────────────────────────────────

    def compile(self):
        """Compile the LangGraph workflow."""
        workflow = StateGraph(GraphRAGState)

        # Register nodes
        workflow.add_node("route_query", self._route_query)
        workflow.add_node("generate_cypher", self._generate_cypher)
        workflow.add_node("execute_cypher", self._execute_cypher)
        workflow.add_node("correct_cypher", self._correct_cypher)
        workflow.add_node("fallback_retrieve", self._fallback_retrieve)
        workflow.add_node("synthesize", self._synthesize_response)

        # Set entry
        workflow.set_entry_point("route_query")

        # Routing edges
        workflow.add_conditional_edges(
            "route_query",
            self._decide_execution_path,
            {
                "generate_cypher": "generate_cypher",
                "fallback_retrieve": "fallback_retrieve"
            }
        )

        workflow.add_edge("generate_cypher", "execute_cypher")

        workflow.add_conditional_edges(
            "execute_cypher",
            self._check_execution_result,
            {
                "correct_cypher": "correct_cypher",
                "fallback_retrieve": "fallback_retrieve",
                "synthesize": "synthesize"
            }
        )

        workflow.add_edge("correct_cypher", "execute_cypher")
        workflow.add_edge("fallback_retrieve", "synthesize")
        workflow.add_edge("synthesize", END)

        return workflow.compile()
