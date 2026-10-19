from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class HazardConfig(BaseModel):
    hazard_type: str = Field("flood", description="normal, flood, fire, combined")
    severity: float = Field(1.0, ge=0.0, le=5.0)
    center_lat: Optional[float] = Field(None, ge=-90.0, le=90.0)
    center_lon: Optional[float] = Field(None, ge=-180.0, le=180.0)
    radius_meters: float = Field(350.0, ge=10.0, le=5000.0)
    propagation_rate: float = Field(0.05, ge=0.0, le=1.0)
    wind_direction_deg: Optional[float] = Field(45.0, ge=0.0, le=360.0)  # For wildfire
    wind_speed_mps: Optional[float] = Field(5.0, ge=0.0, le=50.0)

class HazardArea(BaseModel):
    type: str
    severity: float
    center: List[float]  # [lat, lon]
    radius: float
    affected_nodes: List[str] = []
    affected_edges: List[str] = []
    geometry_polygon: Optional[List[List[float]]] = None

class HazardSimulationRequest(BaseModel):
    hazard_config: HazardConfig
    steps: int = Field(5, ge=1, le=50)

class HazardSimulationResponse(BaseModel):
    hazard_type: str
    severity: float
    active_zones: List[HazardArea]
    closed_edges_count: int
    degraded_edges_count: int
