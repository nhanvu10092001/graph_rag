"""Qdrant provider implementations."""

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ..config import QdrantConfig
from ..exceptions import VectorStoreConnectionError, VectorStoreProviderError
from .vector_store_provider import VectorStoreProvider


class QdrantProviderSync(VectorStoreProvider):
    """Synchronous Qdrant provider."""

    def __init__(self, config: QdrantConfig):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        """Get or create Qdrant client."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient

                self._client = QdrantClient(
                    host=self.config.host,
                    port=self.config.port,
                    api_key=self.config.api_key,
                    https=self.config.https,
                )
            except ImportError as e:
                raise VectorStoreProviderError(
                    "Qdrant dependencies not installed. Install with: pip install qdrant-client langchain-qdrant"
                ) from e
            except Exception as e:
                raise VectorStoreConnectionError(
                    f"Failed to connect to Qdrant at {self.config.host}:{self.config.port}: {str(e)}"
                ) from e

        return self._client

    def create_vectorstore(
        self, embeddings: Embeddings, collection_name: str, **kwargs
    ) -> VectorStore:
        """Create a new Qdrant vector store."""
        try:
            from langchain_qdrant import QdrantVectorStore
            from qdrant_client.http.models import Distance, VectorParams

            client = self._get_client()
            
            # Create collection if it doesn't exist
            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]

            if collection_name not in collection_names:
                print(f"Collection '{collection_name}' is not found. Creating...")
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=len(embeddings.embed_query("test")), distance=Distance.COSINE)
                )
            else:
                print(f"Collection '{collection_name}' is found. Skipping creation...")
            
            return QdrantVectorStore(
                client=client,
                embedding=embeddings,
                collection_name=collection_name,
                **kwargs,
            )
        except ImportError as e:
            raise VectorStoreProviderError(
                "Qdrant dependencies not installed. Install with: pip install qdrant-client langchain-qdrant"
            ) from e
        except Exception as e:
            raise VectorStoreProviderError(
                f"Failed to create Qdrant vector store: {str(e)}"
            ) from e

    def get_existing_vectorstore(
        self, embeddings: Embeddings, collection_name: str, **kwargs
    ) -> VectorStore:
        """Get existing Qdrant vector store."""
        # For Qdrant, getting existing is same as creating (it connects to existing collection)
        return self.create_vectorstore(embeddings, collection_name, **kwargs)


class QdrantProviderAsync(VectorStoreProvider):
    """Asynchronous Qdrant provider."""

    def __init__(self, config: QdrantConfig):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        """Get or create Qdrant client."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient

                self._client = QdrantClient(
                    host=self.config.host,
                    port=self.config.port,
                    api_key=self.config.api_key,
                    https=self.config.https,
                )
            except ImportError as e:
                raise VectorStoreProviderError(
                    "Qdrant dependencies not installed. Install with: pip install qdrant-client langchain-qdrant"
                ) from e
            except Exception as e:
                raise VectorStoreConnectionError(
                    f"Failed to connect to Qdrant at {self.config.host}:{self.config.port}: {str(e)}"
                ) from e

        return self._client

    def create_vectorstore(
        self, embeddings: Embeddings, collection_name: str, **kwargs
    ) -> VectorStore:
        """Create a new Qdrant vector store."""
        try:
            from langchain_qdrant import QdrantVectorStore
            from qdrant_client.http.models import Distance, VectorParams
            client = self._get_client()

            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]

            if collection_name not in collection_names:
                print(f"Collection '{collection_name}' is not found. Creating...")
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=len(embeddings.embed_query("test")), distance=Distance.COSINE)
                )
            else:
                print(f"Collection '{collection_name}' is found. Skipping creation...")
            
            return QdrantVectorStore(
                client=client,
                embedding=embeddings,
                collection_name=collection_name,
            )
        except ImportError as e:
            raise VectorStoreProviderError(
                "Qdrant dependencies not installed. Install with: pip install qdrant-client langchain-qdrant"
            ) from e
        except Exception as e:
            raise VectorStoreProviderError(
                f"Failed to create Qdrant vector store: {str(e)}"
            ) from e

    def get_existing_vectorstore(
        self, embeddings: Embeddings, collection_name: str, **kwargs
    ) -> VectorStore:
        """Get existing Qdrant vector store."""
        # For Qdrant, getting existing is same as creating (it connects to existing collection)
        return self.create_vectorstore(embeddings, collection_name, **kwargs)
