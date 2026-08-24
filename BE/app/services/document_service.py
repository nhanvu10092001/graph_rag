"""Database and business logic services for document management and indexing."""

import logging
import asyncio
from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import Document
from app.services.file_storage import get_file_from_minio, delete_file_from_minio
from app.services.registry import get_services

logger = logging.getLogger("BE.services.document_service")


def _get_session(db: Optional[Session]) -> tuple[Session, bool]:
    if db is not None:
        return db, False
    return SessionLocal(), True


def save_document_metadata(filename: str, minio_key: str, db: Optional[Session] = None) -> Dict[str, Any]:
    """Saves initial document metadata to Postgres."""
    db, owns = _get_session(db)
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
        if owns:
            db.close()


def update_document_status(doc_id: int, status: str, entity_count: int = 0, relationship_count: int = 0, db: Optional[Session] = None) -> None:
    """Updates the status and statistics of a document."""
    db, owns = _get_session(db)
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
        if owns:
            db.close()


def list_documents(db: Optional[Session] = None) -> List[Dict[str, Any]]:
    """Lists all uploaded documents from Postgres."""
    db, owns = _get_session(db)
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
        if owns:
            db.close()


async def index_document_background(doc_id: int, minio_key: str, filename: str):
    """Asynchronous background task to extract text, run Graph RAG indexing, and update database."""
    try:
        update_document_status(doc_id, "processing")

        file_bytes = get_file_from_minio(minio_key)

        logger.info(f"Indexing document {doc_id} ('{filename}') via GraphIndexingService...")
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(None, lambda: get_services().graph_indexing_service.index_document(file_bytes, filename))

        entity_count = res.get("indexed_entities", 0)
        relationship_count = res.get("indexed_relationships", 0)

        update_document_status(
            doc_id,
            "indexed",
            entity_count=entity_count,
            relationship_count=relationship_count
        )
        logger.info(f"Document {doc_id} ('{filename}') indexed successfully. Entities: {entity_count}, Relations: {relationship_count}")
    except Exception as e:
        logger.error(f"Failed to index document {doc_id} ('{filename}'): {e}", exc_info=True)
        update_document_status(doc_id, "failed")


def delete_document(doc_id: int, db: Optional[Session] = None) -> Dict[str, Any]:
    """Deletes document record from Postgres, file object from MinIO, and data from Graph store."""
    db, owns = _get_session(db)
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            raise ValueError(f"Document with ID {doc_id} not found.")

        # 1. Delete from Neo4j Graph
        try:
            get_services().graph_indexing_service.delete_document_from_graph(doc.filename)
        except Exception as e:
            logger.warning(f"Failed to delete document from graph for doc {doc_id}: {e}")

        # 2. Delete from MinIO
        try:
            delete_file_from_minio(doc.minio_key)
        except Exception as e:
            logger.warning(f"Failed to delete file from MinIO for doc {doc_id}: {e}")

        # 3. Delete from Postgres
        db.delete(doc)
        db.commit()

        logger.info(f"Document {doc_id} ('{doc.filename}') deleted successfully.")
        return {
            "status": "success",
            "message": f"Document {doc_id} deleted successfully."
        }
    except Exception:
        if owns:
            db.rollback()
        raise
    finally:
        if owns:
            db.close()
