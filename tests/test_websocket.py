import pytest
from starlette.testclient import TestClient
from backend.app.main import app

def test_websocket_connection_and_ping():
    with TestClient(app) as client:
        session_id = "test_ws_session_123"
        with client.websocket_connect(f"/ws/evacuation/{session_id}") as websocket:
            # 1. Verify instantaneous connection handshake frame
            handshake = websocket.receive_json()
            assert handshake["event"] == "connection_established"
            assert handshake["session_id"] == session_id
            assert handshake["status"] == "connected"

            # 2. Verify bidirectional ping/pong
            websocket.send_text("ping")
            data = websocket.receive_text()
            assert data == "pong"
