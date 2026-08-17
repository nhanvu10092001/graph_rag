"""Cache management for vector store instances."""

from typing import Dict, Optional

from langchain_core.vectorstores import VectorStore


class VectorStoreCache:
    """Manages caching of vector store instances with sync/async separation."""

    def __init__(self):
        """Initialize empty caches for sync and async instances."""
        self._sync_instances: Dict[str, VectorStore] = {}
        self._async_instances: Dict[str, VectorStore] = {}

    def get_sync_instance(self, cache_key: str) -> Optional[VectorStore]:
        """Get cached synchronous vector store instance.

        Args:
            cache_key: Cache key for the instance

        Returns:
            Cached vector store instance or None if not found
        """
        return self._sync_instances.get(cache_key)

    def set_sync_instance(self, cache_key: str, instance: VectorStore) -> None:
        """Cache synchronous vector store instance.

        Args:
            cache_key: Cache key for the instance
            instance: Vector store instance to cache
        """
        self._sync_instances[cache_key] = instance

    def get_async_instance(self, cache_key: str) -> Optional[VectorStore]:
        """Get cached asynchronous vector store instance.

        Args:
            cache_key: Cache key for the instance

        Returns:
            Cached vector store instance or None if not found
        """
        return self._async_instances.get(cache_key)

    def set_async_instance(self, cache_key: str, instance: VectorStore) -> None:
        """Cache asynchronous vector store instance.

        Args:
            cache_key: Cache key for the instance
            instance: Vector store instance to cache
        """
        self._async_instances[cache_key] = instance

    def clear_all(self) -> None:
        """Clear all cached vector store instances."""
        self._sync_instances.clear()
        self._async_instances.clear()

    def clear_sync(self) -> None:
        """Clear only synchronous cached instances."""
        self._sync_instances.clear()

    def clear_async(self) -> None:
        """Clear only asynchronous cached instances."""
        self._async_instances.clear()
