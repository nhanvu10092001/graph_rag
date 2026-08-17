"""Document management operations for vector stores."""

from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from ..exceptions import VectorStoreDocumentError


class VectorStoreDocumentManager:
    """Manages document operations for vector stores."""

    @staticmethod
    def add_documents(
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

        Raises:
            VectorStoreDocumentError: If adding documents fails
        """
        try:
            vectorstore.add_documents(documents, **kwargs)
            return vectorstore
        except Exception as e:
            raise VectorStoreDocumentError(
                f"Failed to add documents to vector store: {str(e)}"
            ) from e

    @staticmethod
    async def add_documents_async(
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

        Raises:
            VectorStoreDocumentError: If adding documents fails
        """
        try:
            response = await vectorstore.aadd_documents(documents, **kwargs)
            return response
        except Exception as e:
            raise VectorStoreDocumentError(
                f"Failed to add documents to vector store: {str(e)}"
            ) from e
