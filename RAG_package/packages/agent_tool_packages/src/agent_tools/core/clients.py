"""Shared infrastructure clients for agent tools.

Provides singleton-like access to MQ producers, Redis, MongoDB, and S3 clients.
All clients are lazily initialized from environment variables — same pattern
used by the existing FastAPI apps (image_app/misc/, chatbot_app/misc/).
"""

import os
from functools import lru_cache
from typing import Any

import boto3
import redis
from motor.motor_asyncio import AsyncIOMotorClient

# ── MQ Producer Cache ────────────────────────────────────────────────

_mq_factory = None
_producers: dict[str, Any] = {}


def _get_mq_factory():
    """Lazy-init MQ factory. Reuses mq_packages create_factory()."""
    global _mq_factory
    if _mq_factory is None:
        from mq.mq_factory_creator import create_factory

        mq_type = os.getenv("MQ_TYPE", "kafka")
        _mq_factory = create_factory(mq_type)
    return _mq_factory


def get_producer(queue_name: str):
    """Get or create a producer for the given queue.

    Producers are cached per queue_name to avoid repeated connections.
    Same pattern as image_app/img_blurring/router.py:
        producer = mq_factory.create_producer(queue_name=BLURRING_CONSUMER_QUEUE)
    """
    if queue_name not in _producers:
        factory = _get_mq_factory()
        _producers[queue_name] = factory.create_producer(queue_name=queue_name)
    return _producers[queue_name]


# ── Redis Client ─────────────────────────────────────────────────────

_redis_client = None


def get_redis_client() -> redis.Redis:
    """Get shared Redis client.

    Same connection as image_app/misc/redis.py and chatbot_app/misc/redis.py.
    """
    global _redis_client
    if _redis_client is None:
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", 6379))
        _redis_client = redis.Redis(host=host, port=port, decode_responses=True)
    return _redis_client


# ── MongoDB Client (async) ───────────────────────────────────────────

_mongo_client = None


def get_mongo_client() -> AsyncIOMotorClient:
    """Get shared async MongoDB client.

    Same connection as image_app/misc/db.py (uses motor).
    """
    global _mongo_client
    if _mongo_client is None:
        mongo_url = os.getenv("MONGO_URL", "mongodb://localhost:27017")
        _mongo_client = AsyncIOMotorClient(mongo_url)
    return _mongo_client


def get_mongo_db(db_name: str):
    """Get a MongoDB database by name."""
    return get_mongo_client()[db_name]


def get_mongo_collection(db_name: str, collection_name: str):
    """Get a MongoDB collection by db_name and collection_name."""
    return get_mongo_db(db_name)[collection_name]


# ── S3/MinIO Client ──────────────────────────────────────────────────

_s3_client = None


def get_s3_client():
    """Get shared S3/MinIO client.

    Same connection as image_app/misc/file_storage.py.
    """
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            endpoint_url=os.getenv("FS_ENDPOINT"),
            aws_access_key_id=os.getenv("FS_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("FS_SECRET_KEY"),
        )
    return _s3_client


def get_s3_bucket() -> str:
    """Get the default S3 bucket name."""
    return os.getenv("FS_BUCKET", "")
