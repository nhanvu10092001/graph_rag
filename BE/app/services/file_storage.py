"""MinIO object storage client and helper operations."""

import logging
from datetime import datetime
import boto3
from botocore.client import Config

from app.config import settings

logger = logging.getLogger("BE.services.file_storage")

# We use boto3 with path-style addressing for MinIO compatibility
s3_client = boto3.client(
    "s3",
    endpoint_url=f"http://{settings.minio_endpoint}",
    aws_access_key_id=settings.minio_access_key,
    aws_secret_access_key=settings.minio_secret_key,
    config=Config(signature_version="s3v4"),
    region_name="us-east-1"
)


def init_minio():
    """Checks and creates MinIO bucket if it does not exist."""
    logger.info(f"Checking MinIO bucket '{settings.minio_bucket}'...")
    try:
        s3_client.head_bucket(Bucket=settings.minio_bucket)
        logger.info(f"MinIO bucket '{settings.minio_bucket}' already exists.")
    except Exception:
        logger.info(f"Creating MinIO bucket '{settings.minio_bucket}'...")
        try:
            s3_client.create_bucket(Bucket=settings.minio_bucket)
            logger.info(f"MinIO bucket '{settings.minio_bucket}' created.")
        except Exception as e:
            logger.error(f"Failed to create MinIO bucket: {e}")
            raise e


def upload_file_to_minio(file_bytes: bytes, filename: str) -> str:
    """Uploads file content to MinIO and returns the unique MinIO object key."""
    # Generate unique key name based on timestamp to avoid collisions
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    minio_key = f"{timestamp}_{filename}"

    s3_client.put_object(
        Bucket=settings.minio_bucket,
        Key=minio_key,
        Body=file_bytes
    )
    logger.info(f"Uploaded '{filename}' to MinIO as '{minio_key}'.")
    return minio_key


def get_file_from_minio(minio_key: str) -> bytes:
    """Retrieves file content from MinIO."""
    response = s3_client.get_object(Bucket=settings.minio_bucket, Key=minio_key)
    return response["Body"].read()
