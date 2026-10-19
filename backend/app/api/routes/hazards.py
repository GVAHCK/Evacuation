from typing import List
from fastapi import APIRouter
from backend.app.schemas.hazard import HazardArea, HazardSimulationRequest, HazardSimulationResponse
from backend.app.hazards.hazard_manager import HazardManager

router = APIRouter(prefix="/hazards", tags=["Hazards"])

# Shared global hazard manager for active operational preview
global_hazard_manager = HazardManager()

@router.get("/active", response_model=List[HazardArea], summary="Get currently active hazard zones and road closures")
def get_active_hazards():
    return global_hazard_manager.get_active_hazard_areas()

@router.post("/simulate", response_model=HazardSimulationResponse, summary="Simulate prospective dynamic hazard spread")
def simulate_hazard(req: HazardSimulationRequest):
    sim_mgr = HazardManager()
    sim_mgr.configure(req.hazard_config)
    for _ in range(req.steps):
        sim_mgr.step(dt=2.0)

    closed = sim_mgr.get_closed_edges()
    degraded = sim_mgr.get_degraded_edges()

    return HazardSimulationResponse(
        hazard_type=req.hazard_config.hazard_type,
        severity=req.hazard_config.severity,
        active_zones=sim_mgr.get_active_hazard_areas(),
        closed_edges_count=len(closed),
        degraded_edges_count=len(degraded)
    )
