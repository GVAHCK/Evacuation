import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.routing.graph_manager import graph_manager

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c

def test_health_endpoints(client):
    res = client.get("/health")
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_status"] == "ready"
    assert data["road_graph_nodes"] > 0

    res_live = client.get("/live")
    assert res_live.status_code == 200

    res_ready = client.get("/ready")
    assert res_ready.status_code == 200

def test_auth_and_user_flow(client):
    # Register test user
    email = "testuser@evac.ai"
    reg_res = client.post("/api/v1/auth/register", json={
        "name": "Integration Tester",
        "email": email,
        "password": "securepassword123",
        "role": "USER"
    })
    # If already registered, status may be 400, else 201
    assert reg_res.status_code in (201, 400)

    # Login
    login_res = client.post("/api/v1/auth/login", json={
        "email": email,
        "password": "securepassword123"
    })
    assert login_res.status_code == 200, login_res.text
    tokens = login_res.json()
    assert "access_token" in tokens
    assert "refresh_token" in tokens

    token = tokens["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Get /users/me
    me_res = client.get("/api/v1/users/me", headers=headers)
    assert me_res.status_code == 200
    user_data = me_res.json()
    assert user_data["email"] == email

def test_models_endpoint(client):
    res = client.get("/api/v1/models")
    assert res.status_code == 200
    models = res.json()
    assert len(models) == 1
    assert models[0]["model_name"] == "offline_to_online"
    assert models[0]["status"] == "ready"

def test_route_calculation_offline_marl(client):
    origin = graph_manager.nodes[0]
    dest = graph_manager.nodes[-1]

    calc_res = client.post("/api/v1/routes/calculate", json={
        "origin": origin,
        "destination": dest,
        "agent_count": 4,
        "hazard_config": {
            "hazard_type": "flood",
            "severity": 1.2
        }
    })
    assert calc_res.status_code == 200, calc_res.text
    route_data = calc_res.json()
    assert route_data["model"] == "offline_to_online"
    assert route_data["success"] is True
    assert len(route_data["agent_routes"]) == 4
    assert len(route_data["primary_route"]["coordinates"]) > 1
    assert route_data["route_decision_factors"]["distance_meters"] > 0

def test_research_endpoints(client):
    bench_res = client.get("/api/v1/research/benchmark")
    assert bench_res.status_code == 200
    data = bench_res.json()
    assert data["model_name"] == "Offline-to-Online MARL"
    assert "evaluation_summary" in data

    zero_shot_res = client.get("/api/v1/research/zero-shot")
    assert zero_shot_res.status_code == 200

    ablation_res = client.get("/api/v1/research/ablation")
    assert ablation_res.status_code == 200

def test_evacuation_session_lifecycle(client):
    origin = graph_manager.nodes[0]
    dest = graph_manager.nodes[-1]

    start_res = client.post("/api/v1/evacuation/start", json={
        "origin": origin,
        "destination": dest,
        "agent_count": 4,
        "hazard_config": {
            "hazard_type": "flood",
            "severity": 1.0
        }
    })
    assert start_res.status_code == 200, start_res.text
    session = start_res.json()
    session_id = session["id"]
    assert session["status"] == "active"

    # Check status
    get_res = client.get(f"/api/v1/evacuation/{session_id}")
    assert get_res.status_code == 200

    # Stop session
    stop_res = client.post(f"/api/v1/evacuation/{session_id}/stop")
    assert stop_res.status_code == 200
    assert stop_res.json()["status"] == "stopped"

    # Get result
    result_res = client.get(f"/api/v1/evacuation/{session_id}/result")
    assert result_res.status_code == 200
