from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class TrafficConfig(BaseModel):
    density_regime: str = Field("medium", description="low, medium, high, extreme")
    congestion_bias: float = Field(0.2, ge=0.0, le=1.0)
    incident_edges: Optional[List[str]] = None

class TrafficStateResponse(BaseModel):
    density: float
    congestion: float
    average_speed_mps: float
    queue_length_meters: float
    active_vehicles: int
    bottleneck_edges: List[str] = []
