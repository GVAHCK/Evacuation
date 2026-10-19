from datetime import datetime, timezone
import uuid
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from backend.app.database.base import Base

class EvacuationSession(Base):
    __tablename__ = "evacuation_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    scenario_id = Column(String(36), ForeignKey("scenarios.id"), nullable=True)
    model = Column(String(50), default="offline_to_online", nullable=False)
    status = Column(String(30), default="active", nullable=False)  # active, completed, stopped, failed
    started_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    origin = Column(String(50), nullable=False)
    destination = Column(String(50), nullable=False)
    agent_count = Column(Integer, default=4, nullable=False)

    distance = Column(Float, default=0.0, nullable=False)
    travel_time = Column(Float, default=0.0, nullable=False)
    hazard_exposure = Column(Float, default=0.0, nullable=False)
    traffic_exposure = Column(Float, default=0.0, nullable=False)
    risk_score = Column(Float, default=0.0, nullable=False)
    inference_latency = Column(Float, default=0.0, nullable=False)
    success = Column(Boolean, default=False, nullable=False)

    fallback_used = Column(Boolean, default=False, nullable=False)
    fallback_reason = Column(Text, nullable=True)

    user = relationship("User", back_populates="evacuation_sessions")
    scenario = relationship("Scenario", back_populates="evacuation_sessions")
    agents = relationship("Agent", back_populates="session", cascade="all, delete-orphan")
    routes = relationship("Route", back_populates="session", cascade="all, delete-orphan")
    hazards = relationship("Hazard", back_populates="session", cascade="all, delete-orphan")
    traffic_snapshots = relationship("TrafficSnapshot", back_populates="session", cascade="all, delete-orphan")
