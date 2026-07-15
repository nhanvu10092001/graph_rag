"""LangGraph Deep Agent setup equipped with Neo4j Graph RAG tools."""

import logging
from typing import Any, Dict
from langchain_core.tools import StructuredTool
from deepagents import create_deep_agent
from llm_utils_llm import EmbeddingsFactory
from llm_utils_graph_rag.plugins.graph_rag_plugin import GraphRAGPlugin
from config import settings

logger = logging.getLogger(__name__)

# 1. Initialize LLM dynamically with Tool Calling support
if settings.llm_provider == "ollama":
    try:
        from langchain_ollama import ChatOllama
        logger.info("Using ChatOllama from langchain_ollama")
        llm = ChatOllama(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature
        )
    except ImportError:
        from langchain_community.chat_models import ChatOllama
        logger.info("Using ChatOllama from langchain_community")
        llm = ChatOllama(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature
        )
elif settings.llm_provider == "openai":
    from langchain_openai import ChatOpenAI
    logger.info("Using ChatOpenAI from langchain_openai")
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        openai_api_key=settings.openai_api_key or None
    )
else:
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


# 2. Construct plugin configuration dict from YAML settings
plugin_config = {
    "neo4j": {
        "uri": settings.neo4j_uri,
        "username": settings.neo4j_username,
        "password": settings.neo4j_password,
        "database": settings.neo4j_database,
    },
    "llm": {},  # Loaded separately above
    "embeddings": {
        "provider": settings.embeddings_provider,
        "ollama": {
            "model": settings.embeddings_model,
            "base_url": settings.embeddings_base_url
        } if settings.embeddings_provider == "ollama" else {},
        "openai": {
            "model": settings.embeddings_model,
            "openai_api_key": settings.openai_api_key or None
        } if settings.embeddings_provider == "openai" else {}
    }
}

# Initialize Graph RAG plugin
logger.info("Initializing Graph RAG Plugin...")
graph_rag_plugin = GraphRAGPlugin(plugin_config)

# Instantiate query service and stores dynamically
embeddings = EmbeddingsFactory.create_embeddings(plugin_config["embeddings"])
query_service = graph_rag_plugin.graph_store
# Re-create GraphQueryService using our embeddings and store
from llm_utils_graph_rag.services.query import GraphQueryService
graph_query_service = GraphQueryService(graph_rag_plugin.graph_store, embeddings)

# Create GraphIndexingService for indexation
from llm_utils_graph_rag.services.indexing import GraphIndexingService
graph_indexing_service = GraphIndexingService(graph_rag_plugin.graph_store, llm, embeddings)


# 3. Define LangChain StructuredTool for the Agent
def query_knowledge_graph(query: str) -> str:
    """Queries the Neo4j Knowledge Graph using semantic search and relationship traversal.
    
    Use this tool to discover entities (e.g. people, companies, cities), their descriptions, 
    and how they are connected to other entities (relationships).
    """
    logger.info(f"Agent is querying Knowledge Graph for: '{query}'")
    try:
        res = graph_query_service.retrieve_relevant_subgraph(query)
        context_str = res.get("context_str", "")
        if not context_str or "No matching knowledge graph entities found" in context_str:
            return f"Query: '{query}'. Result: No matching entities or relationships found in the Knowledge Graph."
        return f"Query: '{query}'. Knowledge Graph Subgraph Results:\n{context_str}"
    except Exception as e:
        logger.error(f"Error querying knowledge graph tool: {e}")
        return f"Error occurred while querying Knowledge Graph: {str(e)}"


# Construct the tool list
graph_rag_tool = StructuredTool.from_function(
    func=query_knowledge_graph,
    name="query_knowledge_graph",
    description="Query the Neo4j database to find connected facts, people, companies, relationships, and locations."
)

agent_tools = [graph_rag_tool]

# 4. Create the Deep Agent using LangGraph prebuilt create_react_agent
system_prompt = (
    "You are a helpful AI assistant (Deep Agent) equipped with access to a structured Neo4j Knowledge Graph.\n"
    "Your goal is to answer queries by querying the knowledge graph. When asked about entities (e.g., people, organizations, locations) "
    "or their relationships/connections, you MUST use the `query_knowledge_graph` tool to fetch accurate context.\n\n"
    "After retrieving graph nodes and relations, synthesize the connections in your final answer so the user understands "
    "how the facts link together (e.g., 'Alice works at Heligate, which is located in Hanoi, the capital of Vietnam.')."
)

logger.info(f"Creating LangGraph Deep Agent with LLM model: {settings.llm_model}")
deep_agent = create_deep_agent(
    model=llm,
    tools=agent_tools,
    instructions=system_prompt
)
