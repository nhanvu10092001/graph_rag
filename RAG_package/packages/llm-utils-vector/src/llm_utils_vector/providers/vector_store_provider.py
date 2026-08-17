"""Base class for vector store providers."""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class VectorStoreProvider(ABC):
    """Base class for vector store providers."""

    def __init__(self, config: Any):
        self.config = config
        self.config.validate()

    @abstractmethod
    def create_vectorstore(
        self, embeddings: Embeddings, collection_name: str, **kwargs
    ) -> VectorStore:
        """Create a new vector store instance."""
        pass

    @abstractmethod
    def get_existing_vectorstore(
        self, embeddings: Embeddings, collection_name: str, **kwargs
    ) -> VectorStore:
        """Get an existing vector store instance."""
        pass