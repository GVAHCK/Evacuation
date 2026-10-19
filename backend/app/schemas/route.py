from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from backend.app.schemas.hazard import HazardConfig, HazardArea
from backend.app.schemas.traffic import TrafficConfig

class RouteDecisionFactors(BaseModel):
    travel_time_seconds: float
    distance_meters: float
    hazard_exposure_score: float
    traffic_exposure_score: float
    agent_conflicts: int
    route_risk_index: float

class AgentRoute(BaseModel):
    agent_id: int
    agent_type: str
    origin_node: str
    destination_node: str
    nodes: List[str]
    coordinates: List[List[float]]  # [[lat, lon], ...]
    distance_meters: float
    estimated_time_seconds: float
    hazard_exposure: float
    traffic_exposure: float
    risk_score: float
    decision_factors: RouteDecisionFactors

class RouteCalculationRequest(BaseModel):
    origin: Union[str, List[float], Dict[str, Any]] = Field(..., description="Node ID, [lat, lon], or {latitude, longitude}")
    destination: Union[str, List[float], Dict[str, Any]] = Field(..., description="Node ID, [lat, lon], or {latitude, longitude}")
    agent_count: int = Field(4, ge=1, le=8, description="Number of heterogeneous evacuation agents")
    hazard_config: Optional[HazardConfig] = None
    traffic_config: Optional[TrafficConfig] = None

class RouteCalculationResponse(BaseModel):
    session_id: str
    model: str = "offline_to_online"
    requested_model: str = "offline_to_online"
    actual_model: str = "offline_to_online"
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    success: bool
    inference_latency_ms: float
    primary_route: AgentRoute
    agent_routes: List[AgentRoute]
    hazard_summary: Optional[Dict[str, Any]] = None
    traffic_summary: Optional[Dict[str, Any]] = None
    route_decision_factors: RouteDecisionFactors
