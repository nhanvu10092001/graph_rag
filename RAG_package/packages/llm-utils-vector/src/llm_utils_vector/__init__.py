"""LLM Utils Vector - Vector store factories and implementations."""

from .factories import (
    VectorStoreManager as VectorStoreFactory,
    VectorStoreFactory as PureVectorStoreFactory,
    VectorStoreManager,
)
from .config import QdrantConfig, PGVectorConfig, VectorStoreConfig
from .providers import (
    QdrantProviderSync,
    QdrantProviderAsync,
    PGVectorProviderSync,
    PGVectorProviderAsync,
    VectorStoreProvider,
)
from .exceptions import (
    VectorStoreError,
    VectorStoreConfigError,
    VectorStoreConnectionError,
    VectorStoreProviderError,
    VectorStoreNotFoundError,
    VectorStoreDocumentError,
)

__version__ = "0.1.0"

__all__ = [
    # Main factory (backward compatible with full functionality)
    "VectorStoreFactory",
    # Manager (with caching and document operations)
    "VectorStoreManager", 
    # Pure factory pattern (without caching/document operations)
    "PureVectorStoreFactory",
    # Configuration classes
    "VectorStoreConfig",
    "QdrantConfig",
    "PGVectorConfig",
    # Provider classes
    "VectorStoreProvider",
    "QdrantProviderSync",
    "QdrantProviderAsync",
    "PGVectorProviderSync",
    "PGVectorProviderAsync",
    # Exceptions
    "VectorStoreError",
    "VectorStoreConfigError",
    "VectorStoreConnectionError",
    "VectorStoreProviderError",
    "VectorStoreNotFoundError",
    "VectorStoreDocumentError",
]
