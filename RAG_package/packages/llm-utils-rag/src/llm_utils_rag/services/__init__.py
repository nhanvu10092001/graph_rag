"""RAG services — focused modules extracted from the RAGPlugin god class."""

from .indexing_service import IndexingService
from .query_service import QueryService
from .deletion_service import DeletionService
from .connection_manager import ConnectionManager

__all__ = [
    "IndexingService",
    "QueryService",
    "DeletionService",
    "ConnectionManager",
]
