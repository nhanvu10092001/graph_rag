"""Refactored factory for creating vector store instances with proper sync/async separation."""

from typing import List, Dict

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ..config import QdrantConfig, PGVectorConfig
from ..providers import (
    QdrantProviderSync,
    QdrantProviderAsync,
    PGVectorProviderSync,
    PGVectorProviderAsync,
    VectorStoreProvider,
)
from ..exceptions import (
    VectorStoreNotFoundError,
    VectorStoreProviderError,
)
from ..logger import logger


class VectorStoreFactory:
    """Pure factory for creating vector store instances."""

    _sync_providers: Dict[str, VectorStoreProvider] = {}
    _async_providers: Dict[str, VectorStoreProvider] = {}
    _initialized = False

    @classmethod
    def _ensure_initialized(cls):
        """Ensure providers are initialized."""
        if not cls._initialized:
            cls._initialize_providers()
            cls._initialized = True

    @classmethod
    def _initialize_providers(cls):
        """Initialize built-in providers."""
        # Initialize Qdrant providers
        try:
            qdrant_config = QdrantConfig.from_env()
            cls._sync_providers["qdrant"] = QdrantProviderSync(qdrant_config)
            cls._async_providers["qdrant"] = QdrantProviderAsync(qdrant_config)
        except Exception:
            # Qdrant not configured or dependencies not available
            pass

        # Initialize PGVector providers
        try:
            pgvector_config = PGVectorConfig.from_env()
            cls._sync_providers["pgvector"] = PGVectorProviderSync(pgvector_config)
            cls._async_providers["pgvector"] = PGVectorProviderAsync(pgvector_config)
        except Exception:
            # PGVector not configured or dependencies not available
            pass

    @classmethod
    def _initialize_providers_with_config(cls, config: dict):
        """Initialize providers with custom configuration."""
        # Initialize Qdrant providers with custom config
        if "qdrant" in config["vector_store"]:
            try:
                qdrant_config_dict = config["vector_store"]["qdrant"]
                qdrant_config = QdrantConfig(
                    host=qdrant_config_dict.get("host", "localhost"),
                    port=qdrant_config_dict.get("port", 6333),
                    api_key=qdrant_config_dict.get("api_key", ""),
                    https=qdrant_config_dict.get("https", False)
                )
                cls._sync_providers["qdrant"] = QdrantProviderSync(qdrant_config)
                cls._async_providers["qdrant"] = QdrantProviderAsync(qdrant_config)
            except Exception:
                # Qdrant configuration failed
                pass

        # Initialize PGVector providers with custom config
        if "pgvector" in config["vector_store"]:
            try:
                pgvector_config_dict = config["vector_store"]["pgvector"]
                pgvector_config = PGVectorConfig(
                    host=pgvector_config_dict.get("host", "localhost"),
                    port=pgvector_config_dict.get("port", 5432),
                    user=pgvector_config_dict.get("user", "postgres"),
                    password=pgvector_config_dict.get("password", ""),
                    database=pgvector_config_dict.get("database", "vector_db")
                )
                cls._sync_providers["pgvector"] = PGVectorProviderSync(pgvector_config)
                cls._async_providers["pgvector"] = PGVectorProviderAsync(pgvector_config)
            except Exception:
                # PGVector configuration failed
                pass

    @classmethod
    def register_sync_provider(cls, name: str, provider: VectorStoreProvider):
        """Register a synchronous vector store provider.

        Args:
            name: Provider name (e.g., "qdrant", "pgvector")
            provider: Provider instance
        """
        cls._sync_providers[name.lower()] = provider

    @classmethod
    def register_async_provider(cls, name: str, provider: VectorStoreProvider):
        """Register an asynchronous vector store provider.

        Args:
            name: Provider name (e.g., "qdrant", "pgvector")
            provider: Provider instance
        """
        cls._async_providers[name.lower()] = provider

    @classmethod
    def create_vectorstore(
        cls,
        provider: str,
        embeddings: Embeddings,
        collection_name: str = "documents",
        **kwargs,
    ) -> VectorStore:
        """Create empty vector store instance synchronously.

        Args:
            provider: Vector store provider ("qdrant" or "pgvector")
            embeddings: Embeddings instance
            collection_name: Name of the collection/table
            **kwargs: Additional provider-specific options

        Returns:
            VectorStore: Empty configured vector store instance

        Raises:
            VectorStoreNotFoundError: If provider is not supported
            VectorStoreProviderError: If provider configuration fails
        """
        logger.info(f"Creating vector store with provider: {provider}, collection: {collection_name}")
        cls._ensure_initialized()
        provider = provider.lower()

        if provider not in cls._sync_providers:
            logger.error(f"Sync provider '{provider}' not found. Available: {list(cls._sync_providers.keys())}")
            raise VectorStoreNotFoundError(
                f"Sync provider '{provider}' not found. Available: {list(cls._sync_providers.keys())}"
            )

        try:
            result = cls._sync_providers[provider].create_vectorstore(
                embeddings, collection_name, **kwargs
            )
            logger.info(f"Successfully created vector store with provider: {provider}")
            return result
        except Exception as e:
            logger.error(f"Failed to create sync vector store with provider '{provider}': {str(e)}")
            raise VectorStoreProviderError(
                f"Failed to create sync vector store with provider '{provider}': {str(e)}"
            ) from e

    @classmethod
    def get_existing_vectorstore(
        cls,
        provider: str,
        embeddings: Embeddings,
        collection_name: str = "documents",
        **kwargs,
    ) -> VectorStore:
        """Get existing vector store instance synchronously.

        Args:
            provider: Vector store provider ("qdrant" or "pgvector")
            embeddings: Embeddings instance
            collection_name: Name of the collection/table
            **kwargs: Additional provider-specific options

        Returns:
            VectorStore: Configured vector store instance

        Raises:
            VectorStoreNotFoundError: If provider is not supported
            VectorStoreProviderError: If provider configuration fails
        """
        cls._ensure_initialized()
        provider = provider.lower()

        if provider not in cls._sync_providers:
            raise VectorStoreNotFoundError(
                f"Sync provider '{provider}' not found. Available: {list(cls._sync_providers.keys())}"
            )

        try:
            return cls._sync_providers[provider].get_existing_vectorstore(
                embeddings, collection_name, **kwargs
            )
        except Exception as e:
            raise VectorStoreProviderError(
                f"Failed to get existing sync vector store with provider '{provider}': {str(e)}"
            ) from e

    @classmethod
    async def create_vectorstore_async(
        cls,
        provider: str,
        embeddings: Embeddings,
        collection_name: str = "documents",
        **kwargs,
    ) -> VectorStore:
        """Create empty vector store instance asynchronously.

        Args:
            provider: Vector store provider ("qdrant" or "pgvector")
            embeddings: Embeddings instance
            collection_name: Name of the collection/table
            **kwargs: Additional provider-specific options

        Returns:
            VectorStore: Empty configured vector store instance

        Raises:
            VectorStoreNotFoundError: If provider is not supported
            VectorStoreProviderError: If provider configuration fails
        """
        cls._ensure_initialized()
        provider = provider.lower()

        if provider not in cls._async_providers:
            raise VectorStoreNotFoundError(
                f"Async provider '{provider}' not found. Available: {list(cls._async_providers.keys())}"
            )

        try:
            return cls._async_providers[provider].create_vectorstore(
                embeddings, collection_name, **kwargs
            )
        except Exception as e:
            raise VectorStoreProviderError(
                f"Failed to create async vector store with provider '{provider}': {str(e)}"
            ) from e

    @classmethod
    async def get_existing_vectorstore_async(
        cls,
        provider: str,
        embeddings: Embeddings,
        collection_name: str = "documents",
        **kwargs,
    ) -> VectorStore:
        """Get existing vector store instance asynchronously.

        Args:
            provider: Vector store provider ("qdrant" or "pgvector")
            embeddings: Embeddings instance
            collection_name: Name of the collection/table
            **kwargs: Additional provider-specific options

        Returns:
            VectorStore: Configured vector store instance

        Raises:
            VectorStoreNotFoundError: If provider is not supported
            VectorStoreProviderError: If provider configuration fails
        """
        cls._ensure_initialized()
        provider = provider.lower()

        if provider not in cls._async_providers:
            raise VectorStoreNotFoundError(
                f"Async provider '{provider}' not found. Available: {list(cls._async_providers.keys())}"
            )

        try:
            return cls._async_providers[provider].get_existing_vectorstore(
                embeddings, collection_name, **kwargs
            )
        except Exception as e:
            raise VectorStoreProviderError(
                f"Failed to get existing async vector store with provider '{provider}': {str(e)}"
            ) from e

    @classmethod
    def get_available_providers(cls) -> Dict[str, List[str]]:
        """Get list of available providers.

        Returns:
            Dict with 'sync' and 'async' keys containing provider names
        """
        cls._ensure_initialized()
        return {
            "sync": list(cls._sync_providers.keys()),
            "async": list(cls._async_providers.keys()),
        }

    @classmethod
    def create_vectorstore_with_config(
        cls,
        provider: str,
        embeddings: Embeddings,
        collection_name: str = "documents",
        config: dict = None,
        **kwargs,
    ) -> VectorStore:
        """Create empty vector store instance synchronously with custom configuration.

        Args:
            provider: Vector store provider ("qdrant" or "pgvector")
            embeddings: Embeddings instance
            collection_name: Name of the collection/table
            config: Custom configuration dict containing provider configs
            **kwargs: Additional provider-specific options

        Returns:
            VectorStore: Empty configured vector store instance

        Raises:
            VectorStoreNotFoundError: If provider is not supported
            VectorStoreProviderError: If provider configuration fails
        """
        if config:
            cls._initialize_providers_with_config(config)
        else:
            cls._ensure_initialized()
        
        provider = provider.lower()

        if provider not in cls._sync_providers:
            raise VectorStoreNotFoundError(
                f"Sync provider '{provider}' not found. Available: {list(cls._sync_providers.keys())}"
            )

        try:
            return cls._sync_providers[provider].create_vectorstore(
                embeddings, collection_name, **kwargs
            )
        except Exception as e:
            raise VectorStoreProviderError(
                f"Failed to create sync vector store with provider '{provider}': {str(e)}"
            ) from e

    @classmethod
    def get_existing_vectorstore_with_config(
        cls,
        provider: str,
        embeddings: Embeddings,
        collection_name: str = "documents",
        config: dict = None,
        **kwargs,
    ) -> VectorStore:
        """Get existing vector store instance synchronously with custom configuration.

        Args:
            provider: Vector store provider ("qdrant" or "pgvector")
            embeddings: Embeddings instance
            collection_name: Name of the collection/table
            config: Custom configuration dict containing provider configs
            **kwargs: Additional provider-specific options

        Returns:
            VectorStore: Configured vector store instance

        Raises:
            VectorStoreNotFoundError: If provider is not supported
            VectorStoreProviderError: If provider configuration fails
        """
        if config:
            cls._initialize_providers_with_config(config)
        else:
            cls._ensure_initialized()
        
        provider = provider.lower()

        if provider not in cls._sync_providers:
            raise VectorStoreNotFoundError(
                f"Sync provider '{provider}' not found. Available: {list(cls._sync_providers.keys())}"
            )

        try:
            return cls._sync_providers[provider].get_existing_vectorstore(
                embeddings, collection_name, **kwargs
            )
        except Exception as e:
            raise VectorStoreProviderError(
                f"Failed to get existing sync vector store with provider '{provider}': {str(e)}"
            ) from e

    @classmethod
    async def create_vectorstore_async_with_config(
        cls,
        provider: str,
        embeddings: Embeddings,
        collection_name: str = "documents",
        config: dict = None,
        **kwargs,
    ) -> VectorStore:
        """Create empty vector store instance asynchronously with custom configuration.

        Args:
            provider: Vector store provider ("qdrant" or "pgvector")
            embeddings: Embeddings instance
            collection_name: Name of the collection/table
            config: Custom configuration dict containing provider configs
            **kwargs: Additional provider-specific options

        Returns:
            VectorStore: Empty configured vector store instance

        Raises:
            VectorStoreNotFoundError: If provider is not supported
            VectorStoreProviderError: If provider configuration fails
        """
        if config:
            cls._initialize_providers_with_config(config)
        else:
            cls._ensure_initialized()
        
        provider = provider.lower()

        if provider not in cls._async_providers:
            raise VectorStoreNotFoundError(
                f"Async provider '{provider}' not found. Available: {list(cls._async_providers.keys())}"
            )

        try:
            return cls._async_providers[provider].create_vectorstore(
                embeddings, collection_name, **kwargs
            )
        except Exception as e:
            raise VectorStoreProviderError(
                f"Failed to create async vector store with provider '{provider}': {str(e)}"
            ) from e

    @classmethod
    async def get_existing_vectorstore_async_with_config(
        cls,
        provider: str,
        embeddings: Embeddings,
        collection_name: str = "documents",
        config: dict = None,
        **kwargs,
    ) -> VectorStore:
        """Get existing vector store instance asynchronously with custom configuration.

        Args:
            provider: Vector store provider ("qdrant" or "pgvector")
            embeddings: Embeddings instance
            collection_name: Name of the collection/table
            config: Custom configuration dict containing provider configs
            **kwargs: Additional provider-specific options

        Returns:
            VectorStore: Configured vector store instance

        Raises:
            VectorStoreNotFoundError: If provider is not supported
            VectorStoreProviderError: If provider configuration fails
        """
        if config:
            cls._initialize_providers_with_config(config)
        else:
            cls._ensure_initialized()
        
        provider = provider.lower()

        if provider not in cls._async_providers:
            raise VectorStoreNotFoundError(
                f"Async provider '{provider}' not found. Available: {list(cls._async_providers.keys())}"
            )

        try:
            return cls._async_providers[provider].get_existing_vectorstore(
                embeddings, collection_name, **kwargs
            )
        except Exception as e:
            raise VectorStoreProviderError(
                f"Failed to get existing async vector store with provider '{provider}': {str(e)}"
            ) from e
