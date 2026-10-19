from fastapi import APIRouter
from backend.app.schemas.traffic import TrafficStateResponse
from backend.app.simulation.sumo_manager import sumo_manager

router = APIRouter(prefix="/traffic", tags=["Traffic"])

@router.get("/current", response_model=TrafficStateResponse, summary="Get current road network traffic state and bottlenecks")
def get_current_traffic():
    return sumo_manager.step(dt=0.0)
