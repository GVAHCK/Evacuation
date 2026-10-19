import uuid
from sqlalchemy import Column, String, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship
from backend.app.database.base import Base

class Hazard(Base):
    __tablename__ = "hazards"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("evacuation_sessions.id"), nullable=False)
    hazard_type = Column(String(50), nullable=False)  # flood, fire, combined
    severity = Column(Float, default=1.0, nullable=False)
    center_lat = Column(Float, nullable=False)
    center_lon = Column(Float, nullable=False)
    radius = Column(Float, default=300.0, nullable=False)
    affected_nodes = Column(JSON, nullable=True)
    affected_edges = Column(JSON, nullable=True)

    session = relationship("EvacuationSession", back_populates="hazards")
