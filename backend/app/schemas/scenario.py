from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from backend.app.schemas.hazard import HazardConfig
from backend.app.schemas.traffic import TrafficConfig

class ScenarioBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=150)
    description: Optional[str] = None
    network: str = Field("336_nodes", description="Network topology identifier")
    origin: str
    destination: str
    agent_count: int = Field(4, ge=1, le=8)
    hazard_type: str = Field("flood", description="normal, flood, fire, combined")
    hazard_config: Optional[HazardConfig] = None
    traffic_config: Optional[TrafficConfig] = None
    model: str = "offline_to_online"
    seed: int = 42

class ScenarioCreate(ScenarioBase):
    pass

class ScenarioUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    origin: Optional[str] = None
    destination: Optional[str] = None
    agent_count: Optional[int] = None
    hazard_type: Optional[str] = None
    hazard_config: Optional[HazardConfig] = None
    traffic_config: Optional[TrafficConfig] = None

class ScenarioResponse(ScenarioBase):
    id: str
    created_by: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}

