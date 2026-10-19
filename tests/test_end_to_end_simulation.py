import pytest
import time
from fastapi.testclient import TestClient
from backend.app.main import app

def test_full_end_to_end_evacuation_flow():
    """
    Complete End-to-End Simulation Test:
    User -> Login -> Destination -> Flood Hazard -> Offline-to-Online MARL ->
    Route -> Live WebSocket -> Dynamic Re-routing -> Evacuation Complete -> Result & History.
    """
    client = TestClient(app)

    # 1. System Health & Readiness Check
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    assert health_data["status"] == "healthy"
    assert health_data["model_status"] == "ready"
    print("\n[E2E] System Health & Model Status: HEALTHY / READY (CUDA/CPU)")

    # 2. Authentication: User Registration & Login
    ts = int(time.time())
    email = f"officer_e2e_{ts}@evacuation.gov"
    reg_resp = client.post("/api/v1/auth/register", json={
        "name": "Captain Evacuation",
        "email": email,
        "password": "SecurePassword123!",
        "role": "RESEARCHER"
    })
    assert reg_resp.status_code in [200, 201], reg_resp.text
    print(f"[E2E] User Registered: {email}")

    login_resp = client.post("/api/v1/auth/login", json={
        "email": email,
        "password": "SecurePassword123!"
    })
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("[E2E] User Authenticated - JWT Token Acquired")

    # 3. Model Verification
    model_resp = client.get("/api/v1/models/offline-to-online", headers=headers)
    assert model_resp.status_code == 200
    model_info = model_resp.json()
    assert model_info["model_name"] == "offline_to_online"
    assert model_info["parameter_count"] == 327204
    print(f"[E2E] Model Verified: {model_info['model_name']} ({model_info['parameter_count']} params)")

    # 4. Multi-Agent Route Calculation (Direct Offline-to-Online MARL)
    calc_payload = {
        "origin": [13.0036872, 80.220072],
        "destination": [13.0069116, 80.2403241],
        "agent_count": 4,
        "hazard_config": {
            "type": "flood",
            "center": [13.0050, 80.2300],
            "severity": 0.8,
            "radius": 450.0
        },
        "traffic_config": {
            "initial_density": 0.35
        }
    }
    route_calc_resp = client.post("/api/v1/routes/calculate", json=calc_payload, headers=headers)
    assert route_calc_resp.status_code == 200
    route_calc_data = route_calc_resp.json()
    assert route_calc_data["model"] == "offline_to_online"
    assert len(route_calc_data["primary_route"]["coordinates"]) > 0
    assert len(route_calc_data["agent_routes"]) == 4
    assert route_calc_data["route_decision_factors"]["travel_time_seconds"] > 0
    print(f"[E2E] Offline-to-Online MARL Route Generated: {len(route_calc_data['primary_route']['coordinates'])} waypoints, Latency: {route_calc_data['inference_latency_ms']:.2f} ms")

    # 5. Start Evacuation Session
    start_payload = {
        "origin": [13.0036872, 80.220072],
        "destination": [13.0069116, 80.2403241],
        "agent_count": 4,
        "hazard_type": "flood",
        "hazard_config": {
            "severity": 0.85,
            "center": [13.0050, 80.2300],
            "radius": 450.0
        },
        "traffic_config": {"initial_density": 0.3},
        "speed_factor": 1.0
    }
    start_resp = client.post("/api/v1/evacuation/start", json=start_payload, headers=headers)
    assert start_resp.status_code == 200
    session_data = start_resp.json()
    session_id = session_data["id"]
    print(f"[E2E] Evacuation Session Started: {session_id} (Status: {session_data['status']})")

    # 6. WebSocket Live Telemetry Connection & Verification (Authenticated with JWT)
    with client.websocket_connect(f"/ws/evacuation/{session_id}?token={token}") as websocket:
        # Handshake frame
        frame1 = websocket.receive_json()
        assert frame1["session_id"] == session_id
        assert frame1["status"] in ["active", "evacuating", "started", "connected"]
        print(f"[E2E] WebSocket Connected - Frame 1: Event={frame1.get('event')}, Status={frame1['status']}")

        # Send ping to test bidirectional communication
        websocket.send_text("ping")
        pong = websocket.receive_text()
        assert pong == "pong"
        print("[E2E] WebSocket Ping/Pong: VERIFIED")

    # 7. Trigger Dynamic Re-routing (simulate hazard breach / congestion shift)
    reroute_resp = client.post(f"/api/v1/evacuation/{session_id}/reroute", headers=headers)
    assert reroute_resp.status_code == 200
    reroute_data = reroute_resp.json()
    assert reroute_data["code"] == "REROUTE_TRIGGERED"
    print(f"[E2E] Dynamic Rerouting Request Triggered: {reroute_data['message']}")

    # 8. Complete Evacuation / Stop Session
    stop_resp = client.post(f"/api/v1/evacuation/{session_id}/stop", headers=headers)
    assert stop_resp.status_code == 200
    print(f"[E2E] Evacuation Session Stopped/Completed")

    # 9. Verify Final Result
    result_resp = client.get(f"/api/v1/evacuation/{session_id}/result", headers=headers)
    assert result_resp.status_code == 200
    result_data = result_resp.json()
    assert result_data["session"]["id"] == session_id
    assert result_data["session"]["model"] == "offline_to_online"
    assert result_data["session"]["agent_count"] == 4
    assert result_data["session"]["distance"] > 0, f"Expected positive distance, got {result_data['session']['distance']}"
    assert result_data["session"]["travel_time"] > 0, f"Expected positive travel time, got {result_data['session']['travel_time']}"
    assert len(result_data.get("routes", [])) > 0, "Expected at least one route in result"
    print(f"[E2E] Final Result Retrieved: Status={result_data['session']['status']}, Distance={result_data['session']['distance']:.1f}m, TravelTime={result_data['session']['travel_time']:.1f}s")

    # 10. Verify History & Audit Records
    history_resp = client.get("/api/v1/history", headers=headers)
    assert history_resp.status_code == 200
    history_list = history_resp.json()
    assert any(s["id"] == session_id for s in history_list)
    print(f"[E2E] History Verified: Session {session_id} found in user session log")

    session_detail_resp = client.get(f"/api/v1/history/{session_id}", headers=headers)
    assert session_detail_resp.status_code == 200
    detail_data = session_detail_resp.json()
    assert detail_data["session"]["id"] == session_id
    assert len(detail_data.get("routes", [])) > 0, f"Expected saved routes, got {len(detail_data.get('routes', []))}"
    print(f"[E2E] Full Session Audit Details Verified: {len(detail_data.get('routes', []))} agent route records preserved")
    print("============================================================")
    print("END-TO-END ACCEPTANCE TEST COMPLETED: 100% PASS")
    print("============================================================")

if __name__ == "__main__":
    test_full_end_to_end_evacuation_flow()
