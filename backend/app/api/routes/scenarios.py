from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.api.deps import get_current_user, require_researcher_or_admin
from backend.app.models.user import User
from backend.app.models.scenario import Scenario
from backend.app.schemas.scenario import ScenarioCreate, ScenarioUpdate, ScenarioResponse
from backend.app.schemas.evacuation import EvacuationStartRequest, EvacuationSessionResponse
from backend.app.services.evacuation_service import evacuation_service

router = APIRouter(prefix="/scenarios", tags=["Scenarios"])

@router.get("", response_model=List[ScenarioResponse], summary="List all saved evacuation scenarios")
def list_scenarios(db: Session = Depends(get_db)):
    scenarios = db.query(Scenario).all()
    return scenarios

@router.post("", response_model=ScenarioResponse, status_code=status.HTTP_201_CREATED, summary="Create a new scenario")
def create_scenario(
    req: ScenarioCreate,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    scenario = Scenario(
        name=req.name,
        description=req.description,
        network=req.network,
        origin=req.origin,
        destination=req.destination,
        agent_count=req.agent_count,
        hazard_type=req.hazard_type,
        hazard_config=req.hazard_config.model_dump() if req.hazard_config else None,
        traffic_config=req.traffic_config.model_dump() if req.traffic_config else None,
        model="offline_to_online",
        seed=req.seed,
        created_by=user.id if user else None
    )
    db.add(scenario)
    db.commit()
    db.refresh(scenario)
    return scenario

@router.get("/{scenario_id}", response_model=ScenarioResponse, summary="Get scenario by ID")
def get_scenario(scenario_id: str, db: Session = Depends(get_db)):
    scenario = db.query(Scenario).filter(Scenario.id == scenario_id).first()
    if not scenario:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "SCENARIO_NOT_FOUND", "message": f"Scenario '{scenario_id}' not found"}
        )
    return scenario

@router.patch("/{scenario_id}", response_model=ScenarioResponse, summary="Update scenario configuration")
def update_scenario(
    scenario_id: str,
    update_data: ScenarioUpdate,
    db: Session = Depends(get_db)
):
    scenario = db.query(Scenario).filter(Scenario.id == scenario_id).first()
    if not scenario:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "SCENARIO_NOT_FOUND", "message": f"Scenario '{scenario_id}' not found"}
        )

    for field, value in update_data.model_dump(exclude_unset=True).items():
        setattr(scenario, field, value)

    db.commit()
    db.refresh(scenario)
    return scenario

@router.delete("/{scenario_id}", summary="Delete scenario")
def delete_scenario(scenario_id: str, db: Session = Depends(get_db)):
    scenario = db.query(Scenario).filter(Scenario.id == scenario_id).first()
    if not scenario:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "SCENARIO_NOT_FOUND", "message": f"Scenario '{scenario_id}' not found"}
        )
    db.delete(scenario)
    db.commit()
    return {"code": "SCENARIO_DELETED", "message": f"Scenario '{scenario_id}' deleted"}

@router.post("/{scenario_id}/run", response_model=EvacuationSessionResponse, summary="Launch evacuation session from scenario")
def run_scenario(
    scenario_id: str,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    scenario = db.query(Scenario).filter(Scenario.id == scenario_id).first()
    if not scenario:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "SCENARIO_NOT_FOUND", "message": f"Scenario '{scenario_id}' not found"}
        )

    req = EvacuationStartRequest(
        origin=scenario.origin,
        destination=scenario.destination,
        agent_count=scenario.agent_count,
        hazard_config=scenario.hazard_config,
        traffic_config=scenario.traffic_config,
        scenario_id=scenario.id,
        seed=scenario.seed
    )
    session = evacuation_service.start_session(req, user_id=user.id if user else None, db=db)
    return session
