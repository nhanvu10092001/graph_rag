"""Shared polling helpers for agent tools.

Extracted from the duplicated patterns in:
    - image_app/img_blurring/_utils.py (Redis polling)
    - chatbot_app/auto_quotation/api_utils.py (Redis polling)
    - image_app/img_ocr/router.py (MongoDB status check pattern)
"""

import asyncio
import json
from typing import Any

from .clients import get_mongo_collection, get_redis_client


class ToolTimeoutError(Exception):
    """Raised when polling exceeds the timeout."""

    def __init__(self, strategy: str, timeout: int, identifier: str):
        self.strategy = strategy
        self.timeout = timeout
        self.identifier = identifier
        super().__init__(
            f"Tool timeout: {strategy} polling for '{identifier}' "
            f"exceeded {timeout}s. The task may still be processing — "
            f"try again later or check the status manually."
        )


async def poll_redis(
    request_id: str,
    timeout: int = 60,
    interval: float = 1.0,
) -> dict:
    """Poll Redis for a result posted by a consumer webhook.

    This is the generalized version of image_app/img_blurring/_utils.py::get_redis_result
    and chatbot_app/auto_quotation/api_utils.py::get_redis_result.

    Flow:
        Consumer processes job → POSTs to webhook → webhook stores in Redis:
            key=request_id, value=json({"result": <data>, "timestamp": "ISO8601"})
        This function polls that key until found or timeout.

    Args:
        request_id: The unique request ID to poll for.
        timeout: Maximum seconds to wait.
        interval: Seconds between poll attempts.

    Returns:
        The parsed "result" dict from Redis.

    Raises:
        ToolTimeoutError: If result not found within timeout.
    """
    redis_client = get_redis_client()
    iterations = int(timeout / interval)

    for _ in range(iterations):
        raw = redis_client.get(request_id)
        if raw:
            parsed = json.loads(raw)
            result = parsed.get("result")
            if result is not None:
                return result
        await asyncio.sleep(interval)

    raise ToolTimeoutError(
        strategy="REDIS_POLL",
        timeout=timeout,
        identifier=request_id,
    )


async def poll_mongodb(
    db_name: str,
    collection_name: str,
    id_field: str,
    id_value: str,
    status_field: str,
    done_value: str = "processed",
    timeout: int = 120,
    interval: float = 2.0,
) -> dict:
    """Poll MongoDB until a document's status field reaches the done value.

    This is the generalized version of the pattern used in:
        - image_app/img_ocr/router.py::get_processed_image (checks ocr_result.re_inv_ocr_status)
        - image_app/img_blurring/router.py::get_processed_image (checks blurred_image.status)

    Args:
        db_name: MongoDB database name.
        collection_name: MongoDB collection name.
        id_field: Field name to match (e.g. "image_id").
        id_value: Value to match (e.g. the UUID).
        status_field: Dot-notation path to status (e.g. "ocr_result.re_inv_ocr_status").
        done_value: Value indicating completion (default "processed").
        timeout: Maximum seconds to wait.
        interval: Seconds between poll attempts.

    Returns:
        The full MongoDB document as dict.

    Raises:
        ToolTimeoutError: If status not reached within timeout.
    """
    collection = get_mongo_collection(db_name, collection_name)
    iterations = int(timeout / interval)

    for _ in range(iterations):
        doc = await collection.find_one({id_field: id_value})
        if doc:
            # Navigate dot-notation status field (e.g. "ocr_result.re_inv_ocr_status")
            value = _get_nested_field(doc, status_field)
            if value == done_value:
                # Convert ObjectId to str for JSON serialization
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
                return doc
        await asyncio.sleep(interval)

    raise ToolTimeoutError(
        strategy="MONGODB_POLL",
        timeout=timeout,
        identifier=f"{id_field}={id_value}",
    )


def _get_nested_field(doc: dict, field_path: str) -> Any:
    """Safely get a nested field from a dict using dot notation.

    Example: _get_nested_field(doc, "ocr_result.re_inv_ocr_status")
    """
    keys = field_path.split(".")
    current = doc
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return None
    return current
