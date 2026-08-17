"""Production hardening utilities — retry logic, input validation, and resource cleanup.

Provides decorators and helpers for making RAG pipeline operations
more resilient in production environments.
"""

import asyncio
import logging
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

MAX_QUERY_LENGTH = 10_000
MAX_FILE_SIZE_MB = 100
MAX_COLLECTION_NAME_LENGTH = 256
VALID_ACTIONS = {"index", "query", "delete"}


# ── Input Validation ───────────────────────────────────────────────────

class InputValidator:
    """Validates RAG pipeline inputs before processing."""

    @staticmethod
    def validate_query(query: str):
        """Validate query string."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if len(query) > MAX_QUERY_LENGTH:
            raise ValueError(
                f"Query too long: {len(query)} chars (max {MAX_QUERY_LENGTH})"
            )

    @staticmethod
    def validate_collection_name(name: str):
        """Validate collection name format."""
        if not name or not name.strip():
            raise ValueError("Collection name cannot be empty")
        if len(name) > MAX_COLLECTION_NAME_LENGTH:
            raise ValueError(
                f"Collection name too long: {len(name)} chars (max {MAX_COLLECTION_NAME_LENGTH})"
            )

    @staticmethod
    def validate_action(action: str):
        """Validate action type."""
        if action not in VALID_ACTIONS:
            raise ValueError(
                f"Unknown action: '{action}'. Valid actions: {VALID_ACTIONS}"
            )

    @staticmethod
    def validate_context(context: Dict[str, Any]):
        """Validate the full context dict."""
        if not isinstance(context, dict):
            raise TypeError(f"Context must be a dict, got {type(context).__name__}")

        action = context.get("action", "query")
        InputValidator.validate_action(action)

        if action == "query":
            InputValidator.validate_query(context.get("query", ""))
            InputValidator.validate_collection_name(
                context.get("collection_name", "")
            )

        elif action == "delete":
            InputValidator.validate_collection_name(
                context.get("collection_name", "")
            )
            unique_filename = context.get("unique_filename", "")
            if not unique_filename:
                raise ValueError("unique_filename is required for deletion")


# ── Retry Decorator ────────────────────────────────────────────────────

def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = (ConnectionError, TimeoutError, OSError),
):
    """Decorator for retrying sync functions with exponential backoff.

    Usage:
        @retry(max_attempts=3)
        def call_external_service():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            "retry.attempt %s/%s for %s failed: %s. Retrying in %.1fs",
                            attempt, max_attempts, func.__name__, e, delay,
                        )
                        import time
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            "retry.exhausted all %s attempts for %s: %s",
                            max_attempts, func.__name__, e,
                        )

            raise last_exception

        return wrapper
    return decorator


def async_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = (ConnectionError, TimeoutError, OSError),
):
    """Decorator for retrying async functions with exponential backoff.

    Usage:
        @async_retry(max_attempts=3)
        async def call_external_service():
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            "async_retry.attempt %s/%s for %s failed: %s. Retrying in %.1fs",
                            attempt, max_attempts, func.__name__, e, delay,
                        )
                        await asyncio.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            "async_retry.exhausted all %s attempts for %s: %s",
                            max_attempts, func.__name__, e,
                        )

            raise last_exception

        return wrapper
    return decorator


# ── Resource Cleanup ───────────────────────────────────────────────────

@contextmanager
def managed_vectorstore(vectorstore):
    """Context manager ensuring vectorstore is properly closed.

    Usage:
        with managed_vectorstore(vs) as vs:
            vs.add_documents(docs)
    """
    try:
        yield vectorstore
    finally:
        if hasattr(vectorstore, "close"):
            try:
                vectorstore.close()
            except Exception as e:
                logger.warning("Error closing vectorstore: %s", e)


@asynccontextmanager
async def managed_vectorstore_async(vectorstore):
    """Async context manager ensuring vectorstore is properly closed.

    Usage:
        async with managed_vectorstore_async(vs) as vs:
            await vs.aadd_documents(docs)
    """
    try:
        yield vectorstore
    finally:
        if hasattr(vectorstore, "aclose"):
            try:
                await vectorstore.aclose()
            except Exception as e:
                logger.warning("Error closing async vectorstore: %s", e)
        elif hasattr(vectorstore, "close"):
            try:
                vectorstore.close()
            except Exception as e:
                logger.warning("Error closing vectorstore: %s", e)
