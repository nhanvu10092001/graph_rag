"""Connection manager for database and vector store clients.

Provides pooled connections to PostgreSQL (PGVector) and Qdrant,
eliminating the per-request connection overhead from the original RAGPlugin.
"""

import logging
import os
from typing import Any, Dict, Optional

import psycopg2
from psycopg2 import pool

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages pooled connections to PostgreSQL and Qdrant.

    Usage:
        mgr = ConnectionManager(config)
        conn = mgr.get_pg_connection()
        try:
            ...
        finally:
            mgr.release_pg_connection(conn)

        client = mgr.get_qdrant_client()
    """

    _pg_pool: Optional[pool.ThreadedConnectionPool] = None
    _qdrant_client = None

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    # ── PostgreSQL (PGVector) ──────────────────────────────────────────

    def _ensure_pg_pool(self) -> pool.ThreadedConnectionPool:
        """Lazily create the PG connection pool."""
        if self._pg_pool is None or self._pg_pool.closed:
            pgvector_config = self.config.get("vector_store", {}).get("pgvector", {})

            host = pgvector_config.get("host", os.getenv("PGVECTOR_HOST", "localhost"))
            port = pgvector_config.get("port", int(os.getenv("PGVECTOR_PORT", "5432")))
            user = pgvector_config.get("user", os.getenv("PGVECTOR_USER", "postgres"))
            password = pgvector_config.get("password", os.getenv("PGVECTOR_PASSWORD"))
            database = pgvector_config.get(
                "database", os.getenv("PGVECTOR_DATABASE", "vector_db")
            )

            if not password:
                raise ValueError(
                    "PGVector password not found in config or environment variables"
                )

            min_conn = pgvector_config.get("pool_min", 1)
            max_conn = pgvector_config.get("pool_max", 10)

            self._pg_pool = pool.ThreadedConnectionPool(
                minconn=min_conn,
                maxconn=max_conn,
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
            )
            logger.info("PostgreSQL connection pool created (%d–%d)", min_conn, max_conn)

        return self._pg_pool

    def get_pg_connection(self):
        """Get a connection from the pool."""
        try:
            pg_pool = self._ensure_pg_pool()
            return pg_pool.getconn()
        except Exception as e:
            raise ValueError(f"Database connection failed: {e}") from e

    def release_pg_connection(self, conn):
        """Return a connection to the pool."""
        if self._pg_pool is not None and conn is not None:
            self._pg_pool.putconn(conn)

    # ── Qdrant ─────────────────────────────────────────────────────────

    def get_qdrant_client(self):
        """Get or create a Qdrant client (singleton per manager instance)."""
        if self._qdrant_client is not None:
            return self._qdrant_client

        try:
            from qdrant_client import QdrantClient

            qdrant_config = self.config.get("vector_store", {}).get("qdrant", {})

            host = qdrant_config.get("host", os.getenv("QDRANT_HOST", "localhost"))
            port = qdrant_config.get("port", int(os.getenv("QDRANT_PORT", "6333")))
            api_key = qdrant_config.get("api_key", os.getenv("QDRANT_API_KEY"))
            https = qdrant_config.get(
                "https", os.getenv("QDRANT_HTTPS", "false").lower() == "true"
            )

            self._qdrant_client = QdrantClient(
                host=host,
                port=port,
                api_key=api_key,
                https=https,
            )
            logger.info("Qdrant client connected to %s:%s", host, port)
            return self._qdrant_client

        except ImportError as e:
            raise ValueError(
                "Qdrant dependencies not installed. Install with: pip install qdrant-client"
            ) from e
        except Exception as e:
            raise ValueError(f"Failed to connect to Qdrant: {e}") from e

    # ── Lifecycle ──────────────────────────────────────────────────────

    def close(self):
        """Close all managed connections."""
        if self._pg_pool is not None and not self._pg_pool.closed:
            self._pg_pool.closeall()
            logger.info("PostgreSQL connection pool closed")

        if self._qdrant_client is not None:
            try:
                self._qdrant_client.close()
            except Exception:
                pass
            self._qdrant_client = None
            logger.info("Qdrant client closed")
