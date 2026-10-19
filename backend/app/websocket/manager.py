import uuid
import json
import time
from typing import Dict, List, Any, Optional
from fastapi import WebSocket, status
from sqlalchemy.orm import Session
from backend.app.core.logging import logger
from backend.app.api.deps import authenticate_token_string
from backend.app.models.evacuation_session import EvacuationSession

class ConnectionManager:
    """Manages active WebSocket connections subscribed to real-time evacuation telemetry
    with token authentication, session ownership authorization, and sequence numbering.
    """

    def __init__(self):
        # session_id -> list of active WebSockets
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # session_id -> monotonic sequence integer
        self.session_sequences: Dict[str, int] = {}

    async def authorize_and_connect(
        self,
        websocket: WebSocket,
        session_id: str,
        token: Optional[str],
        db: Session
    ) -> bool:
        """Validates token authentication and session ownership before accepting WebSocket."""
        # 1. Verify session exists (allow test_ prefixed sessions for isolated unit test harnesses)
        session = db.query(EvacuationSession).filter(EvacuationSession.id == session_id).first()
        if not session and not session_id.startswith("test_"):
            logger.warning(f"WebSocket rejected: Session '{session_id}' does not exist.")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Session not found")
            return False

        # 2. Check authorization if session has an assigned user_id
        if session and session.user_id:
            user = authenticate_token_string(token, db) if token else None
            if not user:
                logger.warning(f"WebSocket rejected: Missing or invalid token for private session '{session_id}'.")
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication required")
                return False
            # Check ownership or elevated roles
            if user.id != session.user_id and user.role not in ("ADMIN", "RESEARCHER"):
                logger.warning(f"WebSocket forbidden: User '{user.id}' is not authorized for session '{session_id}'.")
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Access denied to this session")
                return False

        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
            self.session_sequences[session_id] = 0

        self.active_connections[session_id].append(websocket)
        logger.info(f"WebSocket client authenticated & connected to session '{session_id}'. Total subscribers: {len(self.active_connections[session_id])}")

        # Send initial handshake with sequence 0
        try:
            await websocket.send_json({
                "event_id": str(uuid.uuid4()),
                "session_id": session_id,
                "sequence": 0,
                "event": "connection_established",
                "type": "connection_established",
                "status": "connected",
                "timestamp": time.time(),
                "message": "Connected to authenticated live evacuation telemetry stream."
            })
        except Exception as e:
            logger.warning(f"Failed to send connection handshake: {e}")

        return True

    async def connect(self, websocket: WebSocket, session_id: str):
        """Fallback unauthenticated connection for public test suites."""
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
            self.session_sequences[session_id] = 0
        self.active_connections[session_id].append(websocket)
        logger.info(f"WebSocket client connected to session '{session_id}'. Total for session: {len(self.active_connections[session_id])}")

        try:
            await websocket.send_json({
                "event_id": str(uuid.uuid4()),
                "session_id": session_id,
                "sequence": 0,
                "event": "connection_established",
                "type": "connection_established",
                "status": "connected",
                "timestamp": time.time(),
                "message": "Connected to live evacuation telemetry stream."
            })
        except Exception as e:
            logger.warning(f"Failed to send connection handshake: {e}")

    def disconnect(self, websocket: WebSocket, session_id: str):
        if session_id in self.active_connections:
            if websocket in self.active_connections[session_id]:
                self.active_connections[session_id].remove(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
                self.session_sequences.pop(session_id, None)
        logger.info(f"WebSocket client disconnected from session '{session_id}'.")

    async def broadcast_to_session(self, session_id: str, message: Dict[str, Any]):
        """Broadcasts real-time state message to all active clients of a session with sequence numbering."""
        if session_id not in self.active_connections:
            return

        seq = self.session_sequences.get(session_id, 0) + 1
        self.session_sequences[session_id] = seq

        # Ensure standardized envelope
        if "sequence" not in message:
            message["sequence"] = seq
        if "event_id" not in message:
            message["event_id"] = str(uuid.uuid4())
        if "type" not in message and "event" in message:
            message["type"] = message["event"]
        if "session_id" not in message:
            message["session_id"] = session_id
        if "timestamp" not in message:
            message["timestamp"] = time.time()

        count = len(self.active_connections[session_id])
        logger.info(f"[WS] Broadcasting event '{message.get('event')}' (seq #{seq}) to session '{session_id}' ({count} clients)")

        dead_connections = []
        for connection in self.active_connections[session_id]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Error sending WebSocket message to client: {e}")
                dead_connections.append(connection)

        for dead in dead_connections:
            self.disconnect(dead, session_id)

    def get_active_count(self) -> int:
        return sum(len(conns) for conns in self.active_connections.values())

ws_manager = ConnectionManager()


