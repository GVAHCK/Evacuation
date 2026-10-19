from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime, timezone
import uuid

from backend.app.database.session import get_db
from backend.app.models.audit_log import AuditLog
from backend.app.models.evacuation_session import EvacuationSession
from backend.app.api.deps import get_current_user
from backend.app.models.user import User

router = APIRouter(prefix="/notifications", tags=["Notifications & Alerts"])

# In-memory read tracking for notification entries
_read_notification_ids = set()

@router.get("", summary="Get real-time operational alerts, hazard notifications, and audit events")
def get_notifications(
    limit: int = 20,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    notifications = []

    # 1. Fetch system audit logs from database
    audit_logs = db.query(AuditLog).order_by(desc(AuditLog.timestamp)).limit(limit).all()
    for log in audit_logs:
        notif_id = f"audit_{log.id}"
        notifications.append({
            "id": notif_id,
            "type": "system",
            "title": f"System Event: {log.action.replace('_', ' ').title()}",
            "message": f"Operation logged with details: {log.details or 'Standard execution'}",
            "timestamp": log.timestamp.isoformat() if log.timestamp else datetime.now(timezone.utc).isoformat(),
            "read": notif_id in _read_notification_ids,
            "severity": "info"
        })

    # 2. Add recent evacuation session notifications
    recent_sessions = db.query(EvacuationSession).order_by(desc(EvacuationSession.started_at)).limit(5).all()
    for s in recent_sessions:
        notif_id = f"sess_{s.id}"
        notifications.append({
            "id": notif_id,
            "type": "evacuation",
            "title": f"Evacuation Session: {s.status.upper()}",
            "message": f"Session {s.id[:8]} completed {round(s.distance or 0, 1)}m evacuation path with {s.agent_count} agents.",
            "timestamp": s.started_at.isoformat() if s.started_at else datetime.now(timezone.utc).isoformat(),
            "read": notif_id in _read_notification_ids,
            "severity": "success" if s.status == "completed" else "warning" if s.status == "active" else "danger"
        })

    # 3. Add default system notification if empty
    if not notifications:
        notifications.append({
            "id": "notif_welcome",
            "type": "system",
            "title": "Platform Operational",
            "message": "Offline-to-Online MARL Multi-Hazard Evacuation platform active. Road graph ready.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "read": "notif_welcome" in _read_notification_ids,
            "severity": "info"
        })

    return notifications[:limit]

@router.post("/{notification_id}/read", summary="Mark an operational notification as read")
def mark_notification_read(notification_id: str):
    _read_notification_ids.add(notification_id)
    return {"code": "NOTIFICATION_READ", "id": notification_id}
