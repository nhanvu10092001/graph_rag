"""Interactive demo for Neo4j Graph RAG using LangGraph."""

import os
import sys
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Add parent dir to path to import packages locally
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "llm-utils-core", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "llm-utils-llm", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "llm-utils-vector", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "llm-utils-graph-rag", "src"))

from llm_utils_graph_rag.plugins.graph_rag_plugin import GraphRAGPlugin


def main():
    print("🕸️ Starting Neo4j Graph RAG Demo...")
    print("====================================")
    
    # 1. Initialize plugin
    # Read config from env
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_pass = os.getenv("NEO4J_PASSWORD", "password") # Default container password
    
    config = {
        "neo4j": {
            "uri": neo4j_uri,
            "username": neo4j_user,
            "password": neo4j_pass,
            "database": os.getenv("NEO4J_DATABASE", "neo4j")
        },
        "llm": {
            "provider": "ollama",
            "ollama": {
                "model": "llama3:8b",
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                "temperature": 0.0
            }
        },
        "embeddings": {
            "provider": "ollama",
            "ollama": {
                "model": "nomic-embed-text",
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            }
        }
    }
    
    plugin = GraphRAGPlugin(config)
    
    # Test connection
    if not plugin.graph_store.test_connection():
        print(f"❌ Failed to connect to Neo4j at {neo4j_uri}.")
        print("Please make sure Neo4j is running:")
        print("docker run -d --name neo4j-rag -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest")
        return

    print("✅ Connected to Neo4j Graph Database successfully.")

    # 2. Index connected information
    facts = [
        "Alice is a software engineer who works at Heligate company.",
        "Heligate is located in Hanoi, which is the capital of Vietnam.",
        "Heligate was founded in 2016 and specializes in AI solutions.",
        "Heligate is partners with Google.",
        "Google is headquartered in Mountain View, California."
    ]

    print("\n📝 Indexing facts into Knowledge Graph...")
    for i, fact in enumerate(facts):
        print(f"[{i+1}/{len(facts)}] Fact: \"{fact}\"")
        res = plugin.run({"action": "index", "text": fact})
        if res.get("status") == "success":
            print(f"   Successfully indexed node: {res.get('indexed_entities')} entities, {res.get('indexed_relationships')} relationships.")
        else:
            print(f"   ❌ Indexing failed: {res.get('error')}")

    # Print schema
    print("\n📊 Current Database Schema:")
    print(plugin.graph_store.get_schema())

    # 3. Querying the graph
    queries = [
        "What country is Alice's employer located in?",
        "What partner companies does Alice's employer have?",
        "Where is the headquarters of the partner company of Alice's employer located?"
    ]

    print("\n🤖 Running Graph Queries with LangGraph and Cypher Self-Correction...")
    for q in queries:
        print(f"\n❓ Question: \"{q}\"")
        res = plugin.run({"action": "query", "query": q})
        if res.get("status") == "success":
            print(f"   🛣️ Routing decision: {res.get('routing')}")
            print(f"   💻 Cypher Query used: {res.get('cypher_query')}")
            print(f"   💡 Answer: {res.get('response')}")
        else:
            print(f"   ❌ Query execution failed: {res.get('error')}")

    print("\n🎉 Demo completed successfully!")


if __name__ == "__main__":
    main()
