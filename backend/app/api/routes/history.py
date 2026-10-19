from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.app.database.session import get_db
from backend.app.models.evacuation_session import EvacuationSession
from backend.app.schemas.evacuation import EvacuationSessionResponse, EvacuationResultResponse
from backend.app.api.routes.evacuation import get_evacuation_result

from backend.app.api.deps import get_current_user
from backend.app.models.user import User

router = APIRouter(prefix="/history", tags=["History"])

@router.get("", response_model=List[EvacuationSessionResponse], summary="List completed and previous evacuation sessions")
def list_history(
    limit: int = 50,
    offset: int = 0,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(EvacuationSession)
    if current_user and current_user.role not in ("ADMIN", "RESEARCHER"):
        query = query.filter((EvacuationSession.user_id == current_user.id) | (EvacuationSession.user_id.is_(None)))
    sessions = (
        query
        .order_by(desc(EvacuationSession.started_at))
        .offset(offset)
        .limit(limit)
        .all()
    )
    return sessions

@router.get("/analytics/summary", summary="Get aggregate evacuation performance and safety analytics from database")
def get_history_analytics(
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(EvacuationSession)
    if current_user and current_user.role not in ("ADMIN", "RESEARCHER"):
        query = query.filter((EvacuationSession.user_id == current_user.id) | (EvacuationSession.user_id.is_(None)))
    sessions = query.all()

    total = len(sessions)
    if total == 0:
        return {
            "total_sessions": 0,
            "completed_sessions": 0,
            "joint_success_rate": 0.0,
            "mean_evacuation_time_s": 0.0,
            "mean_distance_meters": 0.0,
            "mean_hazard_exposure": 0.0,
            "mean_risk_score": 0.0,
            "total_agents_evacuated": 0,
            "sessions_evaluated": 0,
            "disclaimer": "Aggregated from authoritative PostgreSQL evacuation_sessions records."
        }

    completed = [s for s in sessions if s.status == "completed" or s.success]
    completed_count = len(completed)
    success_rate = (completed_count / total) * 100.0

    travel_times = [s.travel_time for s in completed if s.travel_time > 0]
    distances = [s.distance for s in completed if s.distance > 0]
    hazards = [s.hazard_exposure for s in sessions if s.hazard_exposure is not None]
    risks = [s.risk_score for s in sessions if s.risk_score is not None]
    agents = sum(s.agent_count for s in completed if s.agent_count)

    mean_time = float(sum(travel_times) / len(travel_times)) if travel_times else 0.0
    mean_dist = float(sum(distances) / len(distances)) if distances else 0.0
    mean_haz = float(sum(hazards) / len(hazards)) if hazards else 0.0
    mean_risk = float(sum(risks) / len(risks)) if risks else 0.0

    return {
        "total_sessions": total,
        "completed_sessions": completed_count,
        "joint_success_rate": round(success_rate, 2),
        "mean_evacuation_time_s": round(mean_time, 1),
        "mean_distance_meters": round(mean_dist, 1),
        "mean_hazard_exposure": round(mean_haz, 4),
        "mean_risk_score": round(mean_risk, 4),
        "total_agents_evacuated": agents,
        "sessions_evaluated": total,
        "disclaimer": "Aggregated from authoritative PostgreSQL evacuation_sessions records."
    }

@router.get("/{session_id}", response_model=EvacuationResultResponse, summary="Get full details and telemetry of a historical evacuation session")
def get_historical_session(
    session_id: str,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return get_evacuation_result(session_id=session_id, current_user=current_user, db=db)
