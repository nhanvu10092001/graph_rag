"""Backward compatibility manager that combines factory, cache, and document operations."""

from typing import List, Dict

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from .vector_store_factory import VectorStoreFactory
from .vector_store_cache import VectorStoreCache
from .vector_store_document_manager import VectorStoreDocumentManager


class VectorStoreManager:
    """Manager that combines factory, cache, and document operations for backward compatibility."""

    _cache = VectorStoreCache()

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
        """
        return VectorStoreFactory.create_vectorstore(
            provider, embeddings, collection_name, **kwargs
        )

    @classmethod
    def get_existing_vectorstore(
        cls,
        provider: str,
        embeddings: Embeddings,
        collection_name: str = "documents",
        **kwargs,
    ) -> VectorStore:
        """Get existing vector store instance synchronously (with caching).

        Args:
            provider: Vector store provider ("qdrant" or "pgvector")
            embeddings: Embeddings instance
            collection_name: Name of the collection/table
            **kwargs: Additional provider-specific options

        Returns:
            VectorStore: Configured vector store instance
        """
        # Ensure factory is initialized
        VectorStoreFactory._ensure_initialized()

        # Generate cache key using provider's config
        provider_obj = VectorStoreFactory._sync_providers.get(provider.lower())
        if not provider_obj:
            return VectorStoreFactory.get_existing_vectorstore(
                provider, embeddings, collection_name, **kwargs
            )

        cache_key = provider_obj.config.get_cache_key(collection_name, is_async=False)

        # Check cache first
        cached_instance = cls._cache.get_sync_instance(cache_key)
        if cached_instance:
            return cached_instance

        # Create new instance and cache it
        instance = VectorStoreFactory.get_existing_vectorstore(
            provider, embeddings, collection_name, **kwargs
        )
        cls._cache.set_sync_instance(cache_key, instance)
        return instance

    @classmethod
    def add_documents_to_vectorstore(
        cls,
        vectorstore: VectorStore,
        documents: List[Document],
        **kwargs,
    ) -> VectorStore:
        """Add documents to an existing vector store synchronously.

        Args:
            vectorstore: Vector store instance to add documents to
            documents: Documents to index
            **kwargs: Additional provider-specific options

        Returns:
            VectorStore: The vector store instance with added documents
        """
        return VectorStoreDocumentManager.add_documents(
            vectorstore, documents, **kwargs
        )

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
        """
        return await VectorStoreFactory.create_vectorstore_async(
            provider, embeddings, collection_name, **kwargs
        )

    @classmethod
    async def get_existing_vectorstore_async(
        cls,
        provider: str,
        embeddings: Embeddings,
        collection_name: str = "documents",
        **kwargs,
    ) -> VectorStore:
        """Get existing vector store instance asynchronously (with caching).

        Args:
            provider: Vector store provider ("qdrant" or "pgvector")
            embeddings: Embeddings instance
            collection_name: Name of the collection/table
            **kwargs: Additional provider-specific options

        Returns:
            VectorStore: Configured vector store instance
        """
        # Ensure factory is initialized
        VectorStoreFactory._ensure_initialized()

        # Generate cache key using provider's config
        provider_obj = VectorStoreFactory._async_providers.get(provider.lower())
        if not provider_obj:
            return await VectorStoreFactory.get_existing_vectorstore_async(
                provider, embeddings, collection_name, **kwargs
            )

        cache_key = provider_obj.config.get_cache_key(collection_name, is_async=True)

        # Check cache first
        cached_instance = cls._cache.get_async_instance(cache_key)
        if cached_instance:
            return cached_instance

        # Create new instance and cache it
        instance = await VectorStoreFactory.get_existing_vectorstore_async(
            provider, embeddings, collection_name, **kwargs
        )
        cls._cache.set_async_instance(cache_key, instance)
        return instance

    @classmethod
    async def add_documents_to_vectorstore_async(
        cls,
        vectorstore: VectorStore,
        documents: List[Document],
        **kwargs,
    ) -> VectorStore:
        """Add documents to an existing vector store asynchronously.

        Args:
            vectorstore: Vector store instance to add documents to
            documents: Documents to index
            **kwargs: Additional provider-specific options

        Returns:
            VectorStore: The vector store instance with added documents
        """
        return await VectorStoreDocumentManager.add_documents_async(
            vectorstore, documents, **kwargs
        )

    @classmethod
    def clear_cache(cls):
        """Clear all cached vector store instances."""
        cls._cache.clear_all()

    @classmethod
    def get_available_providers(cls) -> Dict[str, List[str]]:
        """Get list of available providers."""
        return VectorStoreFactory.get_available_providers()

    @classmethod
    def register_sync_provider(cls, name: str, provider):
        """Register a synchronous vector store provider."""
        VectorStoreFactory.register_sync_provider(name, provider)

    @classmethod
    def register_async_provider(cls, name: str, provider):
        """Register an asynchronous vector store provider."""
        VectorStoreFactory.register_async_provider(name, provider)

    @classmethod
    def get_factory(cls):
        """Get the underlying VectorStoreFactory instance for direct access.
        
        Returns:
            VectorStoreFactory: The factory class for direct method calls
        """
        VectorStoreFactory._ensure_initialized()
        return VectorStoreFactory
    
    @classmethod
    def initialize_factory_with_config(cls, config: dict):
        """Initialize the factory with custom configuration.
        
        Args:
            config: Configuration dict containing provider configs
        """
        VectorStoreFactory._initialize_providers_with_config(config)
    
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
        """
        return VectorStoreFactory.create_vectorstore_with_config(
            provider, embeddings, collection_name, config, **kwargs
        )

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
        """
        return VectorStoreFactory.get_existing_vectorstore_with_config(
            provider, embeddings, collection_name, config, **kwargs
        )

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
        """
        return await VectorStoreFactory.create_vectorstore_async_with_config(
            provider, embeddings, collection_name, config, **kwargs
        )

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
        """
        return await VectorStoreFactory.get_existing_vectorstore_async_with_config(
            provider, embeddings, collection_name, config, **kwargs
        )
