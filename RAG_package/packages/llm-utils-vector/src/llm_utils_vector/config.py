"""Configuration classes for vector store providers."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class VectorStoreConfig(ABC):
    """Base configuration class for vector store providers."""

    @abstractmethod
    def validate(self) -> None:
        """Validate the configuration."""
        pass

    @abstractmethod
    def get_cache_key(self, collection_name: str, is_async: bool) -> str:
        """Generate cache key for this configuration."""
        pass


@dataclass
class QdrantConfig(VectorStoreConfig):
    """Configuration for Qdrant vector store."""

    host: str
    port: int
    api_key: str
    https: bool = False

    @classmethod
    def from_env(cls) -> "QdrantConfig":
        """Create configuration from environment variables."""
        api_key = os.getenv("QDRANT_API_KEY")
        if not api_key:
            raise ValueError("QDRANT_API_KEY environment variable is required")

        return cls(
            host=os.getenv("QDRANT_HOST", "172.16.6.155"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            api_key=api_key,
            https=os.getenv("QDRANT_HTTPS", "false").lower() == "true",
        )

    def validate(self) -> None:
        """Validate Qdrant configuration."""
        if not self.api_key:
            raise ValueError("Qdrant API key is required")
        if not self.host:
            raise ValueError("Qdrant host is required")
        if not (1 <= self.port <= 65535):
            raise ValueError("Qdrant port must be between 1 and 65535")

    def get_cache_key(self, collection_name: str, is_async: bool) -> str:
        """Generate cache key for Qdrant configuration."""
        async_suffix = "async" if is_async else "sync"
        return f"qdrant:{self.host}:{self.port}:{collection_name}:{async_suffix}"


@dataclass
class PGVectorConfig(VectorStoreConfig):
    """Configuration for PGVector store."""

    host: str
    port: int
    user: str
    password: str
    database: str

    @classmethod
    def from_env(cls) -> "PGVectorConfig":
        """Create configuration from environment variables."""
        password = os.getenv("PGVECTOR_PASSWORD")
        if not password:
            raise ValueError("PGVECTOR_PASSWORD environment variable is required")

        return cls(
            host=os.getenv("PGVECTOR_HOST", "172.16.6.31"),
            port=int(os.getenv("PGVECTOR_PORT", "5432")),
            user=os.getenv("PGVECTOR_USER", "heligate"),
            password=password,
            database=os.getenv("PGVECTOR_DATABASE", "vector_db"),
        )

    def validate(self) -> None:
        """Validate PGVector configuration."""
        if not self.password:
            raise ValueError("PGVector password is required")
        if not self.host:
            raise ValueError("PGVector host is required")
        if not (1 <= self.port <= 65535):
            raise ValueError("PGVector port must be between 1 and 65535")
        if not self.user:
            raise ValueError("PGVector user is required")
        if not self.database:
            raise ValueError("PGVector database is required")

    def get_cache_key(self, collection_name: str, is_async: bool) -> str:
        """Generate cache key for PGVector configuration."""
        async_suffix = "async" if is_async else "sync"
        return f"pgvector:{self.host}:{self.port}:{self.database}:{collection_name}:{async_suffix}"

    def get_connection_string(self, is_async: bool = False) -> str:
        """Generate connection string for PGVector."""
        driver = "postgresql+psycopg" if not is_async else "postgresql+psycopg"
        return f"{driver}://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
