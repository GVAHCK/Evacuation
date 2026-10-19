from datetime import datetime, timezone
import uuid
from sqlalchemy import Column, String, DateTime, JSON
from backend.app.database.base import Base

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    user_id = Column(String(36), nullable=True)
    session_id = Column(String(36), nullable=True)
    action = Column(String(100), nullable=False)
    details = Column(JSON, nullable=True)
