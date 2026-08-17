"""MinIO object storage client and helper operations."""

import logging
import threading
from datetime import datetime
import boto3
from botocore.client import Config

from app.config import settings

logger = logging.getLogger("BE.services.file_storage")

_s3_client = None
_lock = threading.Lock()


def _get_s3_client():
    """Lazily initialize and return the S3 client for MinIO."""
    global _s3_client
    if _s3_client is not None:
        return _s3_client
    with _lock:
        if _s3_client is not None:
            return _s3_client

        _protocol = "https" if settings.minio_secure else "http"
        _s3_client = boto3.client(
            "s3",
            endpoint_url=f"{_protocol}://{settings.minio_endpoint}",
            aws_access_key_id=settings.minio_access_key,
            aws_secret_access_key=settings.minio_secret_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1"
        )
        return _s3_client


def init_minio():
    """Checks and creates MinIO bucket if it does not exist."""
    logger.info(f"Checking MinIO bucket '{settings.minio_bucket}'...")
    client = _get_s3_client()
    try:
        client.head_bucket(Bucket=settings.minio_bucket)
        logger.info(f"MinIO bucket '{settings.minio_bucket}' already exists.")
    except Exception:
        logger.info(f"Creating MinIO bucket '{settings.minio_bucket}'...")
        try:
            client.create_bucket(Bucket=settings.minio_bucket)
            logger.info(f"MinIO bucket '{settings.minio_bucket}' created.")
        except Exception as e:
            logger.error(f"Failed to create MinIO bucket: {e}")
            raise e


def upload_file_to_minio(file_bytes: bytes, filename: str) -> str:
    """Uploads file content to MinIO and returns the unique MinIO object key."""
    # Generate unique key name based on timestamp to avoid collisions
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    minio_key = f"{timestamp}_{filename}"

    client = _get_s3_client()
    client.put_object(
        Bucket=settings.minio_bucket,
        Key=minio_key,
        Body=file_bytes
    )
    logger.info(f"Uploaded '{filename}' to MinIO as '{minio_key}'.")
    return minio_key


def get_file_from_minio(minio_key: str) -> bytes:
    """Retrieves file content from MinIO."""
    client = _get_s3_client()
    response = client.get_object(Bucket=settings.minio_bucket, Key=minio_key)
    return response["Body"].read()


def delete_file_from_minio(minio_key: str) -> None:
    """Deletes a file from MinIO."""
    client = _get_s3_client()
    try:
        client.delete_object(Bucket=settings.minio_bucket, Key=minio_key)
        logger.info(f"Deleted '{minio_key}' from MinIO.")
    except Exception as e:
        logger.error(f"Failed to delete '{minio_key}' from MinIO: {e}")
        raise e
