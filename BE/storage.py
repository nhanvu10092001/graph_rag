"""Postgres and MinIO storage operations for document upload and indexing metadata."""

import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import boto3
from botocore.client import Config
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from config import settings

logger = logging.getLogger("BE.storage")

# ── Postgres setup ──────────────────────────────────────────────────────────

DATABASE_URL = f"postgresql://{settings.pg_user}:{settings.pg_password}@{settings.pg_host}:{settings.pg_port}/{settings.pg_database}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Document(Base):
    """Postgres model representing uploaded documents and their indexing metadata."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    minio_key = Column(String, unique=True, nullable=False)
    status = Column(String, default="pending")  # pending, processing, indexed, failed
    entity_count = Column(Integer, default=0)
    relationship_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


# ── MinIO setup ──────────────────────────────────────────────────────────────

# We use boto3 with path-style addressing for MinIO compat
s3_client = boto3.client(
    "s3",
    endpoint_url=f"http://{settings.minio_endpoint}",
    aws_access_key_id=settings.minio_access_key,
    aws_secret_access_key=settings.minio_secret_key,
    config=Config(signature_version="s3v4"),
    region_name="us-east-1"
)


# ── Storage Helpers ──────────────────────────────────────────────────────────

def init_storage():
    """Initializes Postgres tables and creates MinIO bucket if not exists."""
    logger.info("Initializing Postgres tables...")
    Base.metadata.create_all(bind=engine)

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


def get_db():
    """Database session generator helper."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_document_metadata(filename: str, minio_key: str) -> Dict[str, Any]:
    """Saves initial document metadata to Postgres."""
    db = SessionLocal()
    try:
        doc = Document(filename=filename, minio_key=minio_key, status="pending")
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return {
            "id": doc.id,
            "filename": doc.filename,
            "minio_key": doc.minio_key,
            "status": doc.status,
            "created_at": doc.created_at.isoformat()
        }
    finally:
        db.close()


def update_document_status(doc_id: int, status: str, entity_count: int = 0, relationship_count: int = 0) -> None:
    """Updates the status and stats of a document in Postgres."""
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            doc.status = status
            if entity_count > 0:
                doc.entity_count = entity_count
            if relationship_count > 0:
                doc.relationship_count = relationship_count
            db.commit()
    finally:
        db.close()


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


def list_documents() -> List[Dict[str, Any]]:
    """Lists all uploaded documents and metadata from Postgres."""
    db = SessionLocal()
    try:
        docs = db.query(Document).order_by(Document.created_at.desc()).all()
        return [
            {
                "id": d.id,
                "filename": d.filename,
                "minio_key": d.minio_key,
                "status": d.status,
                "entity_count": d.entity_count,
                "relationship_count": d.relationship_count,
                "created_at": d.created_at.isoformat()
            }
            for d in docs
        ]
    finally:
        db.close()
