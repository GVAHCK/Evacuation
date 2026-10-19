from datetime import datetime, timezone
import uuid
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from backend.app.database.base import Base

class Scenario(Base):
    __tablename__ = "scenarios"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(150), nullable=False)
    description = Column(String(500), nullable=True)
    network = Column(String(50), default="336_nodes", nullable=False)
    origin = Column(String(50), nullable=False)
    destination = Column(String(50), nullable=False)
    agent_count = Column(Integer, default=4, nullable=False)
    hazard_type = Column(String(50), default="flood", nullable=False)  # normal, flood, fire, combined
    hazard_config = Column(JSON, nullable=True)
    traffic_config = Column(JSON, nullable=True)
    model = Column(String(50), default="offline_to_online", nullable=False)
    seed = Column(Integer, default=42, nullable=False)
    created_by = Column(String(36), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    creator = relationship("User", back_populates="scenarios")
    evacuation_sessions = relationship("EvacuationSession", back_populates="scenario")
