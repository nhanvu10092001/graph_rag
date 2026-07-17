"""Database models representing schema definitions."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base


class Group(Base):
    """Postgres model representing a group/namespace of documents."""
    __tablename__ = "document_groups"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Document(Base):
    """Postgres model representing uploaded documents and their indexing metadata."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    minio_key = Column(String, unique=True, nullable=False)
    status = Column(String, default="pending")  # pending, processing, indexed, failed
    entity_count = Column(Integer, default=0)
    relationship_count = Column(Integer, default=0)
    group_id = Column(Integer, ForeignKey("document_groups.id", ondelete="CASCADE"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    group = relationship("Group", backref="documents")
