import uuid
from sqlalchemy import Column, String, Integer, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship
from backend.app.database.base import Base

class Route(Base):
    __tablename__ = "routes"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("evacuation_sessions.id"), nullable=False)
    agent_index = Column(Integer, default=0, nullable=False)
    total_distance = Column(Float, default=0.0, nullable=False)
    estimated_time = Column(Float, default=0.0, nullable=False)
    risk_score = Column(Float, default=0.0, nullable=False)
    nodes = Column(JSON, nullable=False)  # List of node IDs
    coordinates = Column(JSON, nullable=False)  # List of [lat, lon] pairs
    decision_factors = Column(JSON, nullable=True)  # travel_time, distance, hazard_exposure, traffic_exposure, etc.

    session = relationship("EvacuationSession", back_populates="routes")
    steps = relationship("RouteStep", back_populates="route", cascade="all, delete-orphan")
