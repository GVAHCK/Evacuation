from datetime import datetime, timezone
import uuid
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from backend.app.database.base import Base

class TrafficSnapshot(Base):
    __tablename__ = "traffic_snapshots"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("evacuation_sessions.id"), nullable=False)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    average_speed = Column(Float, default=11.1, nullable=False)  # m/s (~40 km/h)
    congestion_level = Column(Float, default=0.2, nullable=False)  # 0.0 - 1.0
    queue_length = Column(Float, default=0.0, nullable=False)
    active_vehicles = Column(Integer, default=0, nullable=False)

    session = relationship("EvacuationSession", back_populates="traffic_snapshots")
