"""SQLAlchemy database connection setup."""

import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.config import settings

logger = logging.getLogger("BE.database")

DATABASE_URL = f"postgresql://{settings.pg_user}:{settings.pg_password}@{settings.pg_host}:{settings.pg_port}/{settings.pg_database}"

logger.info(f"Connecting to database at {settings.pg_host}:{settings.pg_port}/{settings.pg_database}")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Database session generator helper."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
