# REST API Reference Manual

The platform exposes standardized RESTful endpoints under `/api/v1` with strict Pydantic v2 data validation and JWT Bearer authorization.

---

## 1. Authentication (`/api/v1/auth`)

### `POST /api/v1/auth/register`
Creates a new user account.
- **Request Body**:
```json
{
  "name": "Officer Jane Doe",
  "email": "jane@evacuation.gov",
  "password": "SecurePassword123!",
  "role": "USER"
}
```
- **Response** (`201 Created`):
```json
{
  "id": "c62b489d-43df-49ae-9b2f-8a0bfb5cb12e",
  "name": "Officer Jane Doe",
  "email": "jane@evacuation.gov",
  "role": "USER",
  "created_at": "2026-09-28T12:00:00Z"
}
```

### `POST /api/v1/auth/login`
Authenticates credentials and returns JWT tokens.
- **Request Body**:
```json
{
  "email": "jane@evacuation.gov",
  "password": "SecurePassword123!"
}
```
- **Response** (`200 OK`):
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 86400,
  "user": { ... }
}
```

---

## 2. Route Generation (`/api/v1/routes`)

### `POST /api/v1/routes/calculate`
Calculates optimal evacuation routes for multiple agents using **Offline-to-Online MARL**.
- **Request Body**:
```json
{
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
```
- **Response** (`200 OK`):
```json
{
  "session_id": "893c52e4-e0c1-4b77-8022-bbf86e413ba1",
  "model": "offline_to_online",
  "requested_model": "offline_to_online",
  "actual_model": "offline_to_online",
  "fallback_used": false,
  "fallback_reason": null,
  "success": true,
  "inference_latency_ms": 4.12,
  "primary_route": {
    "agent_id": 0,
    "agent_type": "high_clearance_suv",
    "origin_node": "14144290612",
    "destination_node": "2212827519",
    "nodes": ["14144290612", "...", "2212827519"],
    "coordinates": [[13.00368, 80.22007], [13.00412, 80.22150], "..."],
    "distance_meters": 3654.7,
    "estimated_time_seconds": 384.2,
    "hazard_exposure": 0.082,
    "traffic_exposure": 0.285,
    "risk_score": 0.174
  },
  "agent_routes": [ ... ],
  "route_decision_factors": {
    "travel_time_seconds": 384.2,
    "distance_meters": 3654.7,
    "hazard_exposure_score": 0.082,
    "traffic_exposure_score": 0.285,
    "agent_conflicts": 0,
    "route_risk_index": 0.174
  }
}
```

---

## 3. Evacuation Execution (`/api/v1/evacuation`)

### `POST /api/v1/evacuation/start`
Starts a live multi-agent simulation loop and begins streaming over WebSockets.
- **Request Body**:
```json
{
  "origin": [13.0036872, 80.220072],
  "destination": [13.0069116, 80.2403241],
  "agent_count": 4,
  "hazard_type": "flood",
  "hazard_config": { "severity": 0.85 },
  "speed_factor": 1.0
}
```

### `POST /api/v1/evacuation/{id}/reroute`
Triggers an immediate real-time route recomputation due to dynamic condition changes.

### `POST /api/v1/evacuation/{id}/stop`
Terminates an ongoing simulation and finalizes session records.

### `GET /api/v1/evacuation/{id}/result`
Retrieves comprehensive final evacuation statistics and per-agent telemetry.

---

## 4. Platform Health (`/health`, `/ready`, `/live`)
- **GET `/health`**: Verifies database connectivity, PyTorch model loading status, and OSM road graph caching.
