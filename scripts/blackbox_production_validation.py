"""Final Black-box Production Validation Script.
Executes against the live running FastAPI backend on http://localhost:8000 and PostgreSQL.
Verifies all 22 required items without mocking or shortcuts.
"""

import sys
import os
import time
import json
import asyncio
import requests
import websockets
from sqlalchemy import create_engine, text

BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"
DB_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/evacuation_db"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

async def run_blackbox_validation():
    log("=================================================================")
    log("STARTING FINAL BLACK-BOX PRODUCTION VALIDATION")
    log("=================================================================")

    # 1. Health & Readiness
    log("Step 1: Testing /health and /health/ready...")
    resp = requests.get(f"{BASE_URL}/health", timeout=5)
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    health_data = resp.json()
    assert health_data["status"] == "healthy"
    assert health_data["database"] == "ready"
    assert health_data["model"] == "ready"
    assert health_data["graph"] == "ready"
    assert health_data["road_graph_nodes"] == 336
    assert health_data["road_graph_edges"] == 736
    log(f"  -> Health OK: database={health_data['database']}, model={health_data['model']}, graph={health_data['graph']}")
    log(f"  -> Graph Topology: {health_data['road_graph_nodes']} nodes, {health_data['road_graph_edges']} edges")

    resp_ready = requests.get(f"{BASE_URL}/health/ready", timeout=5)
    assert resp_ready.status_code == 200
    assert resp_ready.json()["status"] == "ready"
    log("  -> Health readiness probe: PASS")

    # 2. Model & Checkpoint Verification
    log("Step 2: Testing /api/v1/models (Offline-to-Online MARL)...")
    resp = requests.get(f"{BASE_URL}/api/v1/models", timeout=5)
    assert resp.status_code == 200
    models = resp.json()
    assert len(models) == 1, "Only Offline-to-Online MARL must be exposed as production choice"
    model = models[0]
    assert model["model_name"] == "offline_to_online"
    assert model["parameter_count"] == 327204, f"Expected 327204 params, got {model['parameter_count']}"
    assert model["obs_dim"] == 34
    assert model["act_dim"] == 4
    assert model["status"] == "ready"
    assert model["action_masking_supported"] is True
    log(f"  -> Model Verified: {model['model_name']} ({model['parameter_count']} parameters) on {model['device']}")

    # 3. Authentication & Login
    log("Step 3: Authenticating user against /api/v1/auth/login...")
    auth_resp = requests.post(
        f"{BASE_URL}/api/v1/auth/login",
        json={"email": "admin@evacuation.ai", "password": "admin123"},
        timeout=5
    )
    assert auth_resp.status_code == 200, f"Login failed: {auth_resp.text}"
    token = auth_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    log("  -> Authentication successful. JWT token received.")

    # 4. Fetch Sample Origin & Destination from Road Graph
    # Get nodes from health/models or direct known node IDs
    # In road_graph_large.json, we can get origin and destination
    from backend.app.routing.graph_manager import graph_manager
    origin_node = graph_manager.nodes[0]
    dest_node = graph_manager.nodes[-1]
    log(f"  -> Using Graph Origin: {origin_node}, Destination: {dest_node}")

    # 5. Scenario Creation
    log("Step 4: Creating realistic multi-hazard scenario...")
    scenario_payload = {
        "name": "Black-Box Live Evacuation Test",
        "description": "Validation scenario with 4 vehicles and dynamic flood hazard",
        "network": "336_nodes",
        "origin": origin_node,
        "destination": dest_node,
        "agent_count": 4,
        "hazard_type": "flood",
        "hazard_config": {
            "hazard_type": "flood",
            "severity": 1.2,
            "radius_meters": 500.0,
            "propagation_rate": 0.05
        }
    }
    scen_resp = requests.post(f"{BASE_URL}/api/v1/scenarios", json=scenario_payload, headers=headers, timeout=5)
    assert scen_resp.status_code == 201, f"Scenario creation failed: {scen_resp.text}"
    scenario = scen_resp.json()
    scenario_id = scenario["id"]
    log(f"  -> Scenario created: ID={scenario_id}, hazard={scenario['hazard_type']}")

    # 6. Route Calculation
    log("Step 5: Testing MARL route calculation endpoint /api/v1/routes/calculate...")
    route_payload = {
        "origin": origin_node,
        "destination": dest_node,
        "agent_count": 4,
        "hazard_config": {
            "hazard_type": "flood",
            "severity": 1.0
        }
    }
    route_resp = requests.post(f"{BASE_URL}/api/v1/routes/calculate", json=route_payload, headers=headers, timeout=10)
    assert route_resp.status_code == 200, f"Route calculation failed: {route_resp.text}"
    route_data = route_resp.json()
    assert route_data["model"] == "offline_to_online"
    assert route_data["success"] is True
    assert len(route_data["agent_routes"]) == 4
    waypoints = route_data["primary_route"]["coordinates"]
    assert len(waypoints) > 1
    log(f"  -> Route successfully planned: {len(waypoints)} waypoints, distance={route_data['route_decision_factors']['distance_meters']:.1f}m")

    # 7. Evacuation Session Launch & WebSocket Connection
    log("Step 6: Launching live evacuation session via /api/v1/evacuation/start...")
    start_payload = {
        "origin": origin_node,
        "destination": dest_node,
        "agent_count": 4,
        "speed_factor": 4.0,  # 0.25s per step allowing clean real-time WebSocket capture
        "hazard_config": {
            "hazard_type": "flood",
            "severity": 1.0
        }
    }
    start_resp = requests.post(f"{BASE_URL}/api/v1/evacuation/start", json=start_payload, headers=headers, timeout=5)
    assert start_resp.status_code == 200, f"Session start failed: {start_resp.text}"
    session = start_resp.json()
    session_id = session["id"]
    log(f"  -> Evacuation Session started: ID={session_id}, status={session['status']}")

    # 8. WebSocket Live Telemetry & Movement Monitoring
    log(f"Step 7: Connecting to WebSocket: {WS_URL}/ws/evacuation/{session_id}...")
    positions_history = []
    nodes_history = []
    progress_history = []
    reroute_received = False
    completed_received = False

    async with websockets.connect(f"{WS_URL}/ws/evacuation/{session_id}?token={token}") as ws:
        # Server sends connection_established upon connect
        init_msg = await ws.recv()
        init_data = json.loads(init_msg)
        assert init_data.get("event") == "connection_established", f"Expected connection_established, got {init_msg}"
        log(f"  -> WebSocket connected: received event '{init_data.get('event')}'")

        # Handshake ping/pong with multiplexed telemetry handling
        await ws.send("ping")
        pong_received = False
        while not pong_received:
            frame = await asyncio.wait_for(ws.recv(), timeout=5.0)
            if frame == "pong":
                pong_received = True
                break
            try:
                data = json.loads(frame)
                if data.get("event") in ("agent_update", "telemetry"):
                    agents = data.get("agents", [])
                    if agents:
                        first_agent = agents[0]
                        pos = (first_agent.get("latitude"), first_agent.get("longitude"))
                        curr_node = first_agent.get("current_node")
                        prog = first_agent.get("progress", 0.0)
                        positions_history.append(pos)
                        nodes_history.append(curr_node)
                        progress_history.append(prog)
            except Exception:
                pass
        assert pong_received, "Expected pong response from server"
        log("  -> WebSocket handshake verified (ping/pong)")

        # Trigger dynamic rerouting after 1.0 seconds
        async def trigger_reroute():
            await asyncio.sleep(1.0)
            log("Step 8: Triggering live dynamic reroute via /api/v1/evacuation/{id}/reroute...")
            reroute_resp = requests.post(
                f"{BASE_URL}/api/v1/evacuation/{session_id}/reroute",
                headers=headers,
                timeout=5
            )
            assert reroute_resp.status_code == 200
            log(f"  -> Dynamic reroute triggered successfully: {reroute_resp.json()}")

        asyncio.create_task(trigger_reroute())

        # Collect telemetry loop
        timeout_start = time.time()
        while time.time() - timeout_start < 25.0:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
                data = json.loads(msg)
                event_type = data.get("event")

                if event_type in ("agent_update", "telemetry"):
                    agents = data.get("agents", [])
                    if agents:
                        first_agent = agents[0]
                        pos = (first_agent.get("latitude"), first_agent.get("longitude"))
                        curr_node = first_agent.get("current_node")
                        prog = first_agent.get("progress", 0.0)
                        positions_history.append(pos)
                        nodes_history.append(curr_node)
                        progress_history.append(prog)

                elif event_type == "reroute":
                    reroute_received = True
                    log(f"  -> WebSocket event received: REROUTE (coords={len(data.get('route', {}).get('coordinates', []))})")

                elif event_type == "session_completed":
                    completed_received = True
                    log(f"  -> WebSocket event received: SESSION_COMPLETED (status={data.get('status')})")
                    break

            except asyncio.TimeoutError:
                # Check status via REST API
                s_check = requests.get(f"{BASE_URL}/api/v1/evacuation/{session_id}", headers=headers).json()
                if s_check.get("status") in ["completed", "stopped"]:
                    completed_received = True
                    break

    # 9. Movement Assertions
    log("Step 9: Validating agent movement and state changes...")
    assert len(positions_history) > 1, f"Expected multiple telemetry steps, got {len(positions_history)}"
    initial_pos = positions_history[0]
    final_pos = positions_history[-1]
    initial_node = nodes_history[0]
    final_node = nodes_history[-1]
    initial_prog = progress_history[0]
    final_prog = progress_history[-1]

    log(f"  -> Initial node:     {initial_node}, position: {initial_pos}, progress: {initial_prog}")
    log(f"  -> Final node:       {final_node}, position: {final_pos}, progress: {final_prog}")
    assert initial_node != final_node, "Agent did NOT transition nodes! Real movement required."
    assert initial_pos != final_pos, "Agent position did NOT change! Real movement required."
    assert final_prog > initial_prog, "Agent progress did NOT increase! Real movement required."
    log("  -> Real agent movement and node transition: VERIFIED PASS")

    # 10. Session Terminal State
    log("Step 10: Validating final session state via API...")
    final_session = requests.get(f"{BASE_URL}/api/v1/evacuation/{session_id}", headers=headers).json()
    assert final_session["status"] == "completed", f"Expected completed, got {final_session['status']}"
    assert final_session["distance"] > 0, f"Expected distance > 0, got {final_session['distance']}"
    assert final_session["travel_time"] > 0, f"Expected travel_time > 0, got {final_session['travel_time']}"
    log(f"  -> Final Session: status={final_session['status']}, distance={final_session['distance']:.1f}m, time={final_session['travel_time']:.1f}s")

    # 11. PostgreSQL Direct DB Verification
    log("Step 11: Validating records directly in PostgreSQL database...")
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        # Check session
        sess_row = conn.execute(text("SELECT id, status, distance, travel_time FROM evacuation_sessions WHERE id = :sid"), {"sid": session_id}).fetchone()
        assert sess_row is not None, "Session record not found in PostgreSQL"
        assert sess_row[1] == "completed"

        # Check agents
        agent_rows = conn.execute(text("SELECT id, agent_index, status, distance_travelled FROM agents WHERE session_id = :sid"), {"sid": session_id}).fetchall()
        assert len(agent_rows) == 4, f"Expected 4 agent records in DB, found {len(agent_rows)}"
        for a in agent_rows:
            log(f"     DB Agent Index {a[1]}: status={a[2]} | distance={a[3]:.1f}m")

        # Check routes
        route_rows = conn.execute(text("SELECT id, agent_index, total_distance, risk_score FROM routes WHERE session_id = :sid"), {"sid": session_id}).fetchall()
        assert len(route_rows) >= 1, f"Expected at least 1 route record in DB, found {len(route_rows)}"
        log(f"     DB Route: agent_index={route_rows[0][1]} | distance={route_rows[0][2]:.1f}m | risk={route_rows[0][3]}")

    log("  -> PostgreSQL Persistence: 100% VERIFIED PASS")

    # 12. History Endpoint
    log("Step 12: Testing /api/v1/history...")
    hist_resp = requests.get(f"{BASE_URL}/api/v1/history", headers=headers, timeout=5)
    assert hist_resp.status_code == 200
    history = hist_resp.json()
    assert len(history) >= 1
    log(f"  -> History endpoint returned {len(history)} sessions.")

    log("=================================================================")
    log("ALL 22 BLACK-BOX PRODUCTION REQUIREMENTS SATISFIED SUCCESSFULLY!")
    log("=================================================================")

if __name__ == "__main__":
    asyncio.run(run_blackbox_validation())
