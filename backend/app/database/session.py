from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.database.base import Base

def create_db_engine():
    # Attempt primary database connection (PostgreSQL)
    try:
        engine = create_engine(
            settings.DATABASE_URL,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={"connect_timeout": 5}
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"Database connected to primary PostgreSQL engine: {settings.DATABASE_URL.split('@')[-1]}")
        return engine
    except Exception as e:
        logger.warning(f"Could not connect to primary PostgreSQL ({e}). Falling back to SQLite: {settings.SQLITE_FALLBACK_URL}")
        engine = create_engine(
            settings.SQLITE_FALLBACK_URL,
            connect_args={"check_same_thread": False}
        )
        return engine

engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initializes tables and creates initial seed users and records if not present."""
    from backend.app import models  # Ensure all models are registered with Base
    Base.metadata.create_all(bind=engine)
    logger.info("Database schemas initialized.")

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
