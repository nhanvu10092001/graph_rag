"""Database models representing schema definitions."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime
from app.database import Base


class Document(Base):
    """Postgres model representing uploaded documents and their indexing metadata."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    minio_key = Column(String, unique=True, nullable=False)
    status = Column(String, default="pending")  # pending, processing, indexed, failed
    entity_count = Column(Integer, default=0)
    relationship_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
