"""PGVector provider implementations."""

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ..config import PGVectorConfig
from ..exceptions import VectorStoreConnectionError, VectorStoreProviderError
from .vector_store_provider import VectorStoreProvider


class PGVectorProviderSync(VectorStoreProvider):
    """Synchronous PGVector provider."""

    def __init__(self, config: PGVectorConfig):
        super().__init__(config)
        self._engine = None

    def _get_engine(self):
        """Get or create sync SQLAlchemy engine."""
        if self._engine is None:
            try:
                from sqlalchemy import create_engine

                connection_string = self.config.get_connection_string(is_async=False)
                self._engine = create_engine(connection_string)
            except ImportError as e:
                raise VectorStoreProviderError(
                    "PGVector dependencies not installed. Install with: pip install langchain-postgres psycopg[binary]"
                ) from e
            except Exception as e:
                raise VectorStoreConnectionError(
                    f"Failed to connect to PGVector at {self.config.host}:{self.config.port}: {str(e)}"
                ) from e

        return self._engine

    def create_vectorstore(
        self, embeddings: Embeddings, collection_name: str, **kwargs
    ) -> VectorStore:
        """Create a new PGVector store, auto-creating tables if needed."""
        try:
            from langchain_postgres import PGVector

            return PGVector(
                embeddings=embeddings,
                connection=self._get_engine(),
                collection_name=collection_name,
                create_extension=True,
                **kwargs,
            )
        except ImportError as e:
            raise VectorStoreProviderError(
                "PGVector dependencies not installed. Install with: pip install langchain-postgres psycopg[binary]"
            ) from e
        except Exception as e:
            raise VectorStoreProviderError(
                f"Failed to create PGVector store: {str(e)}"
            ) from e

    def get_existing_vectorstore(
        self, embeddings: Embeddings, collection_name: str, **kwargs
    ) -> VectorStore:
        """Get existing PGVector store."""
        return self.create_vectorstore(embeddings, collection_name, **kwargs)


class PGVectorProviderAsync(VectorStoreProvider):
    """Asynchronous PGVector provider."""

    def __init__(self, config: PGVectorConfig):
        super().__init__(config)
        self._engine = None

    def _get_engine(self):
        """Get or create async SQLAlchemy engine."""
        if self._engine is None:
            try:
                from sqlalchemy.ext.asyncio import create_async_engine

                connection_string = self.config.get_connection_string(is_async=True)
                self._engine = create_async_engine(connection_string)
            except ImportError as e:
                raise VectorStoreProviderError(
                    "PGVector dependencies not installed. Install with: pip install langchain-postgres psycopg[binary]"
                ) from e
            except Exception as e:
                raise VectorStoreConnectionError(
                    f"Failed to connect to PGVector at {self.config.host}:{self.config.port}: {str(e)}"
                ) from e

        return self._engine

    def create_vectorstore(
        self, embeddings: Embeddings, collection_name: str, **kwargs
    ) -> VectorStore:
        """Create a new PGVector store, auto-creating tables if needed."""
        try:
            from langchain_postgres import PGVector
            from sqlalchemy.engine import Engine
            from sqlalchemy.ext.asyncio import AsyncEngine
            class CustomPGVector(PGVector):
                """Custom PGVector class with method to close the DB connection."""

                def close(self):
                    """Close the connection to the vector store."""
                    if isinstance(self._engine, Engine):
                        self._engine.dispose()
                async def aclose(self):
                    """Asynchronously close the connection to the vector store."""
                    if isinstance(self._async_engine, AsyncEngine):
                        await self._async_engine.dispose()
            return CustomPGVector(
                embeddings=embeddings,
                connection=self._get_engine(),
                collection_name=collection_name,
                create_extension=True,
                **kwargs,
            )
        except ImportError as e:
            raise VectorStoreProviderError(
                "PGVector dependencies not installed. Install with: pip install langchain-postgres psycopg[binary]"
            ) from e
        except Exception as e:
            raise VectorStoreProviderError(
                f"Failed to create PGVector store: {str(e)}"
            ) from e

    def get_existing_vectorstore(
        self, embeddings: Embeddings, collection_name: str, **kwargs
    ) -> VectorStore:
        """Get existing PGVector store."""
        return self.create_vectorstore(embeddings, collection_name, **kwargs)