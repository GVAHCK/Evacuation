import uuid
from sqlalchemy import Column, String, Integer, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship
from backend.app.database.base import Base

class Agent(Base):
    __tablename__ = "agents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("evacuation_sessions.id"), nullable=False)
    agent_index = Column(Integer, nullable=False)
    current_node = Column(String(50), nullable=False)
    destination_node = Column(String(50), nullable=False)
    status = Column(String(30), default="evacuating", nullable=False)  # evacuating, evacuated, trapped, rerouted
    route = Column(JSON, nullable=True)  # List of node IDs
    distance_travelled = Column(Float, default=0.0, nullable=False)
    time_elapsed = Column(Float, default=0.0, nullable=False)
    hazard_exposure = Column(Float, default=0.0, nullable=False)

    session = relationship("EvacuationSession", back_populates="agents")
