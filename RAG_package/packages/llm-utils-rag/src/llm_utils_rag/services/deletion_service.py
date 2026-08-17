"""Deletion service — handles vector removal from stores.

Extracts the deletion logic from RAGPlugin, using ConnectionManager
for pooled database/Qdrant access.
"""

import asyncio
import logging
from typing import Any, Dict

from psycopg2 import sql

from .connection_manager import ConnectionManager

logger = logging.getLogger(__name__)


class DeletionService:
    """Handles vector deletion from configured vector stores.

    Usage:
        svc = DeletionService(config)
        result = svc.delete_sync(context)
        result = await svc.delete_async(context)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._conn_mgr = ConnectionManager(config)

    # ── Public API ─────────────────────────────────────────────────────

    def delete_sync(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Delete vectors synchronously by unique_filename."""
        collection_name = context.get("collection_name")
        if not collection_name:
            raise ValueError("collection_name is required for deletion")

        unique_filename = context.get("unique_filename")
        if not unique_filename:
            raise ValueError("unique_filename is required for deletion")

        result = self._delete_from_configured_store(collection_name, unique_filename)

        if result["success"]:
            message = (
                f"Successfully deleted {result['count']} vectors from "
                f"{result['provider']} collection '{collection_name}' "
                f"with unique_filename '{unique_filename}'"
            )
            status = "success"
        else:
            message = f"Failed to delete from {result['provider']}: {result['error']}"
            status = "error"

        return {
            "status": status,
            "message": message,
            "count": result["count"],
            "collection_name": collection_name,
            "unique_filename": unique_filename,
            "provider": result["provider"],
            "error": result["error"],
        }

    async def delete_async(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Delete vectors asynchronously (runs sync in executor)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.delete_sync, context)

    # ── Provider dispatch ──────────────────────────────────────────────

    def _delete_from_configured_store(
        self, collection_name: str, unique_filename: str
    ) -> Dict[str, Any]:
        vector_provider = self.config.get("vector_store", {}).get("provider", "qdrant")
        result = {"provider": vector_provider, "success": False, "count": 0, "error": None}

        try:
            if vector_provider.lower() == "pgvector":
                if "pgvector" not in self.config.get("vector_store", {}):
                    raise ValueError("PGVector provider selected but pgvector config not found")
                result["count"] = self._delete_pg(collection_name, unique_filename)
                result["success"] = True
            else:
                result["count"] = self._delete_qdrant(collection_name, unique_filename)
                result["success"] = True
        except Exception as e:
            result["error"] = str(e)

        return result

    # ── PGVector deletion ──────────────────────────────────────────────

    def _delete_pg(self, collection_name: str, unique_filename: str) -> int:
        conn = None
        try:
            conn = self._conn_mgr.get_pg_connection()
            cursor = conn.cursor()

            delete_query = sql.SQL(
                """
                DELETE FROM langchain_pg_embedding
                WHERE cmetadata::jsonb ->> 'unique_filename' = %s
                AND collection_id = (
                    SELECT uuid FROM langchain_pg_collection
                    WHERE name = %s
                )
                """
            )
            cursor.execute(delete_query, (unique_filename, collection_name))
            deleted_count = cursor.rowcount
            conn.commit()
            cursor.close()
            return deleted_count
        except Exception as e:
            if conn:
                conn.rollback()
            raise ValueError(f"Database deletion failed: {e}") from e
        finally:
            self._conn_mgr.release_pg_connection(conn)

    # ── Qdrant deletion ────────────────────────────────────────────────

    def _delete_qdrant(self, collection_name: str, unique_filename: str) -> int:
        return self._find_and_delete_from_qdrant(
            collection_name, {"unique_filename": unique_filename}
        )

    def _find_and_delete_from_qdrant(
        self, collection_name: str, search_criteria: Dict[str, Any]
    ) -> int:
        try:
            from qdrant_client.http import models

            client = self._conn_mgr.get_qdrant_client()
            must_conditions = []

            metadata_keys = [
                "unique_filename", "filename", "source_path", "category"
            ]
            for key in metadata_keys:
                if key in search_criteria:
                    must_conditions.append(
                        models.FieldCondition(
                            key=f"metadata.{key}",
                            match=models.MatchValue(value=search_criteria[key]),
                        )
                    )

            if not must_conditions and search_criteria.get("delete_all"):
                client.delete(
                    collection_name=collection_name,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(must=[])
                    ),
                )
                try:
                    info = client.get_collection(collection_name)
                    return info.points_count if hasattr(info, "points_count") else 0
                except Exception:
                    return 0

            elif must_conditions:
                filt = models.Filter(must=must_conditions)
                scroll_result = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=filt,
                    limit=10000,
                    with_payload=False,
                    with_vectors=False,
                )
                count = len(scroll_result[0])
                client.delete(
                    collection_name=collection_name,
                    points_selector=models.FilterSelector(filter=filt),
                )
                return count
            else:
                raise ValueError(
                    "No search criteria provided and delete_all not set to True"
                )

        except Exception as e:
            raise ValueError(f"Qdrant deletion failed: {e}") from e

    def close(self):
        """Release connection resources."""
        self._conn_mgr.close()
