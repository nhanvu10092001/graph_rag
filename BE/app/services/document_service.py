"""Database and business logic services for document management and indexing."""

import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any

from app.database import SessionLocal
from app.models import Document
from app.services.file_storage import get_file_from_minio, delete_file_from_minio
from app.parser import extract_text
from app.agent import graph_indexing_service

logger = logging.getLogger("BE.services.document_service")


def save_document_metadata(filename: str, minio_key: str) -> Dict[str, Any]:
    """Saves initial document metadata to Postgres. Creates and closes its own DB connection."""
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
    """Updates the status and statistics of a document. Creates and closes its own DB connection."""
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


def list_documents() -> List[Dict[str, Any]]:
    """Lists all uploaded documents from Postgres. Creates and closes its own DB connection."""
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


async def index_document_background(doc_id: int, minio_key: str, filename: str):
    """Asynchronous background task to extract text, run Graph RAG indexing, and update database."""
    try:
        # 1. Update status to processing
        update_document_status(doc_id, "processing")
        
        # 2. Fetch content from MinIO
        file_bytes = get_file_from_minio(minio_key)
        
        # 3. Extract text content
        text = extract_text(filename, file_bytes)
        
        # 4. Ingest text into Graph RAG
        logger.info(f"Indexing document {doc_id} ('{filename}') via GraphIndexingService...")
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(None, lambda: graph_indexing_service.index_text(text, source_doc=filename))
        
        # 5. Extract statistics
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


def delete_document(doc_id: int) -> Dict[str, Any]:
    """Deletes document record from Postgres, file object from MinIO, and data from Graph store."""
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            raise ValueError(f"Document with ID {doc_id} not found.")
        
        # 1. Delete from Neo4j Graph
        try:
            graph_indexing_service.delete_document_from_graph(doc.filename)
        except Exception as e:
            logger.warning(f"Failed to delete document from graph for doc {doc_id}: {e}")
            
        # 2. Delete from MinIO
        try:
            delete_file_from_minio(doc.minio_key)
        except Exception as e:
            logger.warning(f"Failed to delete file from MinIO for doc {doc_id}: {e}")
        
        # 2. Delete from Postgres
        db.delete(doc)
        db.commit()
        
        logger.info(f"Document {doc_id} ('{doc.filename}') deleted successfully.")
        return {
            "status": "success",
            "message": f"Document {doc_id} deleted successfully."
        }
    finally:
        db.close()

