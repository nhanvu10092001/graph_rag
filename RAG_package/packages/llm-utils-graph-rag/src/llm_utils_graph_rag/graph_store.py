"""Neo4j graph database store wrapper."""

import logging
from typing import Any, Dict, List, Optional
from langchain_community.graphs import Neo4jGraph
from .config import Neo4jConfig

logger = logging.getLogger(__name__)


class Neo4jGraphStore:
    """Wrapper around LangChain's Neo4jGraph implementation."""

    def __init__(self, config: Neo4jConfig):
        self.config = config
        self._graph: Optional[Neo4jGraph] = None

    def connect(self) -> Neo4jGraph:
        """Establish connection to Neo4j database."""
        if self._graph is None:
            try:
                logger.info(f"Connecting to Neo4j at {self.config.uri} (database: {self.config.database})")
                self._graph = Neo4jGraph(
                    url=self.config.uri,
                    username=self.config.username,
                    password=self.config.password,
                    database=self.config.database,
                )
                logger.info("Successfully connected to Neo4j database.")
                self.ensure_fulltext_index()
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                raise ConnectionError(f"Failed to connect to Neo4j at {self.config.uri}: {e}") from e
        return self._graph

    def ensure_fulltext_index(self) -> None:
        """Ensure fulltext index exists for ARK global search."""
        if self._graph is None:
            return
        try:
            # Create fulltext index on Entity nodes for ARK global search
            cypher = """
            CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
            FOR (n:Entity) ON EACH [n.id, n.description, n.type]
            """
            self._graph.query(cypher)
            logger.info("Ensured 'entity_fulltext' index exists.")
        except Exception as e:
            logger.warning(f"Failed to create fulltext index (may already exist or not supported): {e}")

    def test_connection(self) -> bool:
        """Test if connection to Neo4j works."""
        try:
            graph = self.connect()
            # Simple query to test
            graph.query("RETURN 1 AS test_val")
            return True
        except Exception as e:
            logger.warning(f"Neo4j connection test failed: {e}")
            return False

    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run Cypher query and return dictionary list."""
        graph = self.connect()
        logger.debug(f"Running Cypher: {cypher} with params {params}")
        return graph.query(cypher, params or {})

    def get_schema(self) -> str:
        """Return the schema of the graph database."""
        graph = self.connect()
        # Fetch schema from LangChain graph store
        graph.refresh_schema()
        return graph.schema
