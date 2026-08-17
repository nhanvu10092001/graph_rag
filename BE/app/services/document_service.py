"""Database and business logic services for document management and indexing."""

import logging
import asyncio
from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import Document, Group
from app.services.file_storage import get_file_from_minio, delete_file_from_minio
from app.services.registry import get_services

logger = logging.getLogger("BE.services.document_service")


def _get_session(db: Optional[Session]) -> tuple[Session, bool]:
    if db is not None:
        return db, False
    return SessionLocal(), True


def save_document_metadata(filename: str, minio_key: str, group_id: Optional[int] = None, db: Optional[Session] = None) -> Dict[str, Any]:
    """Saves initial document metadata to Postgres."""
    db, owns = _get_session(db)
    try:
        doc = Document(filename=filename, minio_key=minio_key, status="pending", group_id=group_id)
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return {
            "id": doc.id,
            "filename": doc.filename,
            "minio_key": doc.minio_key,
            "status": doc.status,
            "group_id": doc.group_id,
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


def list_documents(group_id: Optional[int] = None, db: Optional[Session] = None) -> List[Dict[str, Any]]:
    """Lists all uploaded documents from Postgres, optionally filtered by group_id."""
    db, owns = _get_session(db)
    try:
        query = db.query(Document)
        if group_id is not None:
            query = query.filter(Document.group_id == group_id)
        docs = query.order_by(Document.created_at.desc()).all()
        return [
            {
                "id": d.id,
                "filename": d.filename,
                "minio_key": d.minio_key,
                "status": d.status,
                "entity_count": d.entity_count,
                "relationship_count": d.relationship_count,
                "group_id": d.group_id,
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
        logger.error(f"Failed to index document {doc_id} ('{filename}'): {e}")
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
        if owns:
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


def create_group(name: str, db: Optional[Session] = None) -> Dict[str, Any]:
    """Creates a new document group."""
    db, owns = _get_session(db)
    try:
        existing = db.query(Group).filter(Group.name == name).first()
        if existing:
            raise ValueError(f"Group with name '{name}' already exists.")

        group = Group(name=name)
        db.add(group)
        db.commit()
        db.refresh(group)
        return {
            "id": group.id,
            "name": group.name,
            "created_at": group.created_at.isoformat()
        }
    finally:
        if owns:
            db.close()


def list_groups(db: Optional[Session] = None) -> List[Dict[str, Any]]:
    """Lists all document groups."""
    db, owns = _get_session(db)
    try:
        groups = db.query(Group).order_by(Group.created_at.desc()).all()
        return [
            {
                "id": g.id,
                "name": g.name,
                "created_at": g.created_at.isoformat()
            }
            for g in groups
        ]
    finally:
        if owns:
            db.close()


def delete_group(group_id: int, db: Optional[Session] = None) -> Dict[str, Any]:
    """Deletes a group and all its documents atomically."""
    db, owns = _get_session(db)
    try:
        group = db.query(Group).filter(Group.id == group_id).first()
        if not group:
            raise ValueError(f"Group with ID {group_id} not found.")

        docs = db.query(Document).filter(Document.group_id == group_id).all()

        for doc in docs:
            try:
                delete_document(doc.id, db=db)
            except Exception as e:
                logger.warning(f"Failed to delete document {doc.id} during group delete: {e}")

        db.delete(group)
        if owns:
            db.commit()

        logger.info(f"Group {group_id} ('{group.name}') and its documents deleted successfully.")
        return {
            "status": "success",
            "message": f"Group '{group.name}' and all associated documents deleted successfully."
        }
    except Exception:
        if owns:
            db.rollback()
        raise
    finally:
        if owns:
            db.close()
