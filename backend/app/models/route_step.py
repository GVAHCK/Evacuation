from datetime import datetime, timezone
import uuid
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from backend.app.database.base import Base

class RouteStep(Base):
    __tablename__ = "route_steps"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    route_id = Column(String(36), ForeignKey("routes.id"), nullable=False)
    sequence = Column(Integer, nullable=False)
    node_id = Column(String(50), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    hazard_level = Column(Float, default=0.0, nullable=False)
    traffic_level = Column(Float, default=0.0, nullable=False)

    route = relationship("Route", back_populates="steps")
