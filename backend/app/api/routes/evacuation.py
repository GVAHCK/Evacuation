from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.api.deps import get_current_user
from backend.app.models.user import User
from backend.app.models.evacuation_session import EvacuationSession
from backend.app.models.route import Route
from backend.app.models.agent import Agent
from backend.app.schemas.evacuation import (
    EvacuationStartRequest,
    EvacuationSessionResponse,
    EvacuationResultResponse
)
from backend.app.schemas.route import AgentRoute, RouteDecisionFactors
from backend.app.services.evacuation_service import evacuation_service

from backend.app.core.rate_limiter import compute_rate_limiter

router = APIRouter(prefix="/evacuation", tags=["Evacuation"])

@router.post("/start", response_model=EvacuationSessionResponse, summary="Start live multi-agent evacuation session", dependencies=[Depends(compute_rate_limiter)])
async def start_evacuation(
    req: EvacuationStartRequest,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user_id = current_user.id if current_user else None
    session = evacuation_service.start_session(req, user_id=user_id, db=db)
    return session

def verify_session_ownership(session: EvacuationSession, current_user: Optional[User]):
    if session.user_id:
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "AUTHENTICATION_REQUIRED", "message": "Authentication required to access private session"}
            )
        if current_user.id != session.user_id and current_user.role not in ("ADMIN", "RESEARCHER"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"code": "FORBIDDEN", "message": "Access forbidden: session belongs to another user"}
            )

@router.get("/{session_id}", response_model=EvacuationSessionResponse, summary="Get evacuation session status")
def get_evacuation_session(
    session_id: str,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    session = db.query(EvacuationSession).filter(EvacuationSession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "SESSION_NOT_FOUND", "message": f"Session '{session_id}' not found"}
        )
    verify_session_ownership(session, current_user)
    return session

@router.post("/{session_id}/stop", response_model=EvacuationSessionResponse, summary="Stop active evacuation session")
async def stop_evacuation(
    session_id: str,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    session = db.query(EvacuationSession).filter(EvacuationSession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "SESSION_NOT_FOUND", "message": f"Session '{session_id}' not found"}
        )
    verify_session_ownership(session, current_user)
    stopped_session = evacuation_service.stop_session(session_id, db)
    return stopped_session

@router.post("/{session_id}/pause", summary="Pause active evacuation session")
async def pause_evacuation(
    session_id: str,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    session = db.query(EvacuationSession).filter(EvacuationSession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "SESSION_NOT_FOUND", "message": f"Session '{session_id}' not found"}
        )
    verify_session_ownership(session, current_user)
    success = evacuation_service.pause_session(session_id)
    return {"code": "SESSION_PAUSED" if success else "SESSION_NOT_ACTIVE", "paused": success, "session_id": session_id}

@router.post("/{session_id}/resume", summary="Resume paused evacuation session")
async def resume_evacuation(
    session_id: str,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    session = db.query(EvacuationSession).filter(EvacuationSession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "SESSION_NOT_FOUND", "message": f"Session '{session_id}' not found"}
        )
    verify_session_ownership(session, current_user)
    success = evacuation_service.resume_session(session_id)
    return {"code": "SESSION_RESUMED" if success else "SESSION_NOT_ACTIVE", "resumed": success, "session_id": session_id}

@router.post("/{session_id}/speed", summary="Adjust simulation speed factor (0.1x to 10.0x)")
async def set_evacuation_speed(
    session_id: str,
    speed_factor: float = 1.0,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    session = db.query(EvacuationSession).filter(EvacuationSession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "SESSION_NOT_FOUND", "message": f"Session '{session_id}' not found"}
        )
    verify_session_ownership(session, current_user)
    success = evacuation_service.set_speed_factor(session_id, speed_factor)
    return {"code": "SPEED_UPDATED", "speed_factor": speed_factor, "session_id": session_id}

@router.get("/{session_id}/events", summary="Get chronological operational events and reroutes for session")
def get_session_events(
    session_id: str,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    session = db.query(EvacuationSession).filter(EvacuationSession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "SESSION_NOT_FOUND", "message": f"Session '{session_id}' not found"}
        )
    verify_session_ownership(session, current_user)
    events = evacuation_service.get_events(session_id)
    # If in-memory events are empty (e.g. historical completed session), synthesize from stored state
    if not events:
        events = [
            {
                "id": f"evt_init_{session_id}",
                "session_id": session_id,
                "timestamp": session.started_at.isoformat() if session.started_at else None,
                "type": "session_started",
                "message": f"Evacuation session initialized with {session.agent_count} agents via {session.model}."
            }
        ]
        if session.status in ("completed", "stopped", "incomplete"):
            events.append({
                "id": f"evt_term_{session_id}",
                "session_id": session_id,
                "timestamp": session.completed_at.isoformat() if session.completed_at else None,
                "type": "session_completed" if session.success else "session_stopped",
                "message": f"Evacuation session concluded with status '{session.status}'."
            })
    return events

@router.get("/{session_id}/agents/{agent_id}", summary="Get specific agent trajectory, progress, and telemetry")
def get_agent_detail(
    session_id: str,
    agent_id: int,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    session = db.query(EvacuationSession).filter(EvacuationSession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "SESSION_NOT_FOUND", "message": f"Session '{session_id}' not found"}
        )
    verify_session_ownership(session, current_user)

    agent = db.query(Agent).filter(Agent.session_id == session_id, Agent.agent_index == agent_id).first()
    route = db.query(Route).filter(Route.session_id == session_id, Route.agent_index == agent_id).first()
    if not route:
        # Fall back to primary route if distinct route record not created
        route = db.query(Route).filter(Route.session_id == session_id, Route.agent_index == 0).first()

    return {
        "session_id": session_id,
        "agent_id": agent_id,
        "status": agent.status if agent else ("evacuated" if session.success else "active"),
        "current_node": agent.current_node if agent else session.origin,
        "destination_node": agent.destination_node if agent else session.destination,
        "distance_travelled": agent.distance_travelled if agent else session.distance,
        "time_elapsed": agent.time_elapsed if agent else session.travel_time,
        "hazard_exposure": agent.hazard_exposure if agent else session.hazard_exposure,
        "nodes": route.nodes if route else [],
        "coordinates": route.coordinates if route else [],
        "decision_factors": route.decision_factors if route else {}
    }

@router.post("/{session_id}/reroute", summary="Trigger dynamic reroute for active evacuation session", dependencies=[Depends(compute_rate_limiter)])
async def trigger_reroute(session_id: str):
    evacuation_service.trigger_reroute(session_id)
    return {"code": "REROUTE_TRIGGERED", "message": "Dynamic route recalculation initiated."}

@router.get("/{session_id}/result", response_model=EvacuationResultResponse, summary="Get full final result of an evacuation session")
def get_evacuation_result(
    session_id: str,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    session = db.query(EvacuationSession).filter(EvacuationSession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "SESSION_NOT_FOUND", "message": f"Session '{session_id}' not found"}
        )
    verify_session_ownership(session, current_user)

    # Fetch stored routes and agents
    routes = db.query(Route).filter(Route.session_id == session_id).all()
    agents = db.query(Agent).filter(Agent.session_id == session_id).all()

    agent_routes = []
    for r in routes:
        factors = RouteDecisionFactors(**(r.decision_factors or {
            "travel_time_seconds": r.estimated_time,
            "distance_meters": r.total_distance,
            "hazard_exposure_score": r.risk_score,
            "traffic_exposure_score": 0.25,
            "agent_conflicts": 0,
            "route_risk_index": r.risk_score
        }))
        agent_routes.append(AgentRoute(
            agent_id=r.agent_index,
            agent_type="Standard Cohort",
            origin_node=r.nodes[0] if r.nodes else session.origin,
            destination_node=r.nodes[-1] if r.nodes else session.destination,
            nodes=r.nodes or [],
            coordinates=r.coordinates or [],
            distance_meters=r.total_distance,
            estimated_time_seconds=r.estimated_time,
            hazard_exposure=session.hazard_exposure,
            traffic_exposure=session.traffic_exposure,
            risk_score=r.risk_score,
            decision_factors=factors
        ))

    default_factors = agent_routes[0].decision_factors if agent_routes else RouteDecisionFactors(
        travel_time_seconds=session.travel_time,
        distance_meters=session.distance,
        hazard_exposure_score=session.hazard_exposure,
        traffic_exposure_score=session.traffic_exposure,
        agent_conflicts=0,
        route_risk_index=session.risk_score
    )

    return EvacuationResultResponse(
        session=session,
        routes=agent_routes,
        route_decision_factors=default_factors
    )
