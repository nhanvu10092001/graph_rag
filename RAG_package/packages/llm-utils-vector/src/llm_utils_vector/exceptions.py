"""Custom exceptions for vector store operations."""


class VectorStoreError(Exception):
    """Base exception for vector store operations."""

    pass


class VectorStoreConfigError(VectorStoreError):
    """Exception raised for configuration-related errors."""

    pass


class VectorStoreConnectionError(VectorStoreError):
    """Exception raised for connection-related errors."""

    pass


class VectorStoreProviderError(VectorStoreError):
    """Exception raised for provider-specific errors."""

    pass


class VectorStoreNotFoundError(VectorStoreError):
    """Exception raised when a vector store provider is not found."""

    pass


class VectorStoreDocumentError(VectorStoreError):
    """Exception raised for document operation errors."""

    pass
