"""Configuration classes for Neo4j Graph RAG provider."""

import os
from dataclasses import dataclass


@dataclass
class Neo4jConfig:
    """Configuration for Neo4j database."""

    uri: str
    username: str
    password: str
    database: str = "neo4j"

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        """Create configuration from environment variables."""
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        database = os.getenv("NEO4J_DATABASE", "neo4j")
        
        return cls(
            uri=uri,
            username=username,
            password=password,
            database=database,
        )

    def validate(self) -> None:
        """Validate Neo4j configuration."""
        if not self.uri:
            raise ValueError("Neo4j URI is required")
        if not self.username:
            raise ValueError("Neo4j username is required")
        if not self.password:
            raise ValueError("Neo4j password is required")
