# Real-Time WebSocket Telemetry Protocol

## Endpoint

```
ws://<HOST>:8000/ws/evacuation/{session_id}
```

---

## Connection Handshake

Upon establishing the WebSocket connection, the backend immediately emits a handshake frame:
```json
{
  "event": "connection_established",
  "session_id": "a29c7871-649b-412d-bb13-cc60a345f196",
  "status": "connected",
  "timestamp": 1790599812.45,
  "message": "Connected to live evacuation telemetry stream."
}
```

---

## Bi-Directional Ping / Pong

Clients can send raw text:
```
ping
```
The server immediately responds:
```
pong
```

---

## Real-Time Telemetry State Frame

Every simulation step (~1.0s), the backend broadcasts the full synchronized state:

```json
{
  "event": "step_update",
  "session_id": "a29c7871-649b-412d-bb13-cc60a345f196",
  "timestamp": 1790599815.2,
  "status": "active",
  "agents": [
    {
      "id": 0,
      "agent_type": "high_clearance_suv",
      "latitude": 13.00412,
      "longitude": 80.22150,
      "current_node": "14144290612",
      "destination_node": "2212827519",
      "status": "evacuating",
      "progress": 0.25,
      "distance_remaining_m": 2740.5,
      "route": ["14144290612", "...", "2212827519"]
    },
    { ... }
  ],
  "traffic": {
    "density": 0.28,
    "congestion": 0.22,
    "average_speed": 10.4
  },
  "hazards": [
    {
      "type": "flood",
      "severity": 0.85,
      "center": [13.0050, 80.2300],
      "radius": 465.0,
      "geometry": [[13.0048, 80.2295], "..."]
    }
  ],
  "route": {
    "coordinates": [[13.00412, 80.22150], "..."],
    "distance": 2740.5,
    "eta": 288.0,
    "risk": 0.18
  },
  "metrics": {
    "hazard_exposure": 0.045,
    "traffic_exposure": 0.28,
    "elapsed_time": 14.0,
    "total_conflicts": 0,
    "agents_evacuated": 1,
    "agents_remaining": 3
  }
}
```

---

## Dynamic Re-routing Frame

When road blockages occur and the route is recalculated:
```json
{
  "event": "route_update",
  "session_id": "a29c7871-649b-412d-bb13-cc60a345f196",
  "timestamp": 1790599820.0,
  "status": "active",
  "message": "Route updated due to changing hazard and traffic conditions.",
  "route": { ... }
}
```
The React Native client receives this frame and automatically updates the route geometry and displays an in-app alert banner.
