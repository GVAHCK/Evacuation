from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from backend.app.schemas.hazard import HazardConfig, HazardArea
from backend.app.schemas.traffic import TrafficConfig
from backend.app.schemas.route import RouteDecisionFactors, AgentRoute

class EvacuationStartRequest(BaseModel):
    origin: Union[str, List[float], Dict[str, Any]] = Field(..., description="Node ID, [lat, lon], or {latitude, longitude}")
    destination: Union[str, List[float], Dict[str, Any]] = Field(..., description="Node ID, [lat, lon], or {latitude, longitude}")
    agent_count: int = Field(4, ge=1, le=8)
    hazard_config: Optional[Union[HazardConfig, Dict[str, Any]]] = None
    hazard_type: Optional[str] = None
    traffic_config: Optional[Union[TrafficConfig, Dict[str, Any]]] = None
    scenario_id: Optional[str] = None
    seed: Optional[int] = 42
    speed_factor: Optional[float] = Field(1.0, description="Simulation playback speed multiplier")

class AgentLiveState(BaseModel):
    id: int
    agent_type: str
    latitude: float
    longitude: float
    current_node: str
    destination_node: str
    status: str  # evacuating, evacuated, trapped, rerouted
    progress: float  # 0.0 to 1.0
    distance_remaining_m: float
    route: List[str]

class TrafficLiveState(BaseModel):
    density: float
    congestion: float
    average_speed: float

class HazardLiveState(BaseModel):
    type: str
    severity: float
    geometry: Optional[List[List[float]]] = None
    center: Optional[List[float]] = None
    radius: Optional[float] = None

class RouteLiveState(BaseModel):
    coordinates: List[List[float]]
    distance: float
    eta: float
    risk: float

class MetricsLiveState(BaseModel):
    hazard_exposure: float
    traffic_exposure: float
    elapsed_time: float
    total_conflicts: int = 0
    agents_evacuated: int = 0
    agents_remaining: int = 4

class WebSocketEvacuationMessage(BaseModel):
    event: str  # session_started, agent_update, traffic_update, hazard_update, route_update, reroute, warning, agent_evacuated, session_completed, error
    session_id: str
    timestamp: float
    status: str
    agents: List[AgentLiveState]
    traffic: TrafficLiveState
    hazards: List[HazardLiveState]
    route: RouteLiveState
    metrics: MetricsLiveState
    message: Optional[str] = None

class EvacuationSessionResponse(BaseModel):
    id: str
    user_id: Optional[str]
    scenario_id: Optional[str]
    model: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    origin: str
    destination: str
    agent_count: int
    distance: float
    travel_time: float
    hazard_exposure: float
    traffic_exposure: float
    risk_score: float
    inference_latency: float
    success: bool
    fallback_used: bool
    fallback_reason: Optional[str] = None

    model_config = {"from_attributes": True}

class EvacuationResultResponse(BaseModel):
    session: EvacuationSessionResponse
    routes: List[AgentRoute]
    route_decision_factors: RouteDecisionFactors
    benchmark_comparison_available: bool = True
