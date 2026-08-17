"""Basic unit tests for llm-utils-graph-rag package."""

import pytest
from unittest.mock import MagicMock
from llm_utils_graph_rag.config import Neo4jConfig
from llm_utils_graph_rag.graph_store import Neo4jGraphStore
from llm_utils_graph_rag.services.indexing import GraphIndexingService
from llm_utils_graph_rag.services.query import GraphQueryService


def test_neo4j_config_validation():
    """Test config validation logic."""
    cfg = Neo4jConfig(uri="bolt://localhost:7687", username="neo4j", password="password")
    cfg.validate()  # Should not raise

    invalid_cfg = Neo4jConfig(uri="", username="neo4j", password="")
    with pytest.raises(ValueError):
        invalid_cfg.validate()


def test_graph_indexing_service():
    """Test entity & relation extraction with mocks."""
    mock_store = MagicMock(spec=Neo4jGraphStore)
    mock_llm = MagicMock()
    mock_embeddings = MagicMock()

    # Mock LLM return value
    mock_response = MagicMock()
    mock_response.content = '{"entities": [{"id": "ALICE", "type": "PERSON", "description": "Works at Heligate"}], "relationships": []}'
    mock_llm.invoke.return_value = mock_response
    mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]

    svc = GraphIndexingService(mock_store, mock_llm, mock_embeddings)
    res = svc.index_text("Alice works at Heligate")

    assert res["status"] == "success"
    assert res["extracted_entities"] == 1
    assert res["indexed_entities"] == 1
    assert mock_llm.invoke.called


def test_graph_query_service():
    """Test subgraph retrieval with mocks."""
    mock_store = MagicMock(spec=Neo4jGraphStore)
    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.return_value = [0.1, 0.2]

    # Mock matched nodes
    mock_store.query.side_effect = [
        [{"id": "ALICE", "type": "PERSON", "description": "Works at Heligate", "score": 0.95}], # First call (vector query)
        [{"source": "ALICE", "rel": "WORKS_AT", "target": "HELIGATE", "description": "Software engineer"}] # Second call (traversal query)
    ]

    svc = GraphQueryService(mock_store, mock_embeddings)
    res = svc.retrieve_relevant_subgraph("Alice's workplace")

    assert len(res["entities"]) == 1
    assert res["entities"][0]["id"] == "ALICE"
    assert len(res["relationships"]) == 1
    assert res["relationships"][0]["source"] == "ALICE"
    assert "Matched Entities:" in res["context_str"]
    assert "WORKS_AT" in res["context_str"]
