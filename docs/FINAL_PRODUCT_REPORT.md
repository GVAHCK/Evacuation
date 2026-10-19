# Final Product Technical & Engineering Report

## Executive Summary

The **Multi-Hazard Evacuation Intelligence Platform** is a research-grade emergency coordination system that unifies Multi-Agent Reinforcement Learning with real-time geospatial road networks, stochastic flood and fire propagation models, microscopic traffic simulation, a high-performance Python FastAPI backend, and a modern React Native mobile application.

---

## Technical Specifications Summary

### 1. Unified Single-Backend Architecture
- **Framework**: FastAPI (Python 3.11+) with Uvicorn.
- **Decision**: No Node.js or Express backend. All ML inference, graph algorithms, and physics simulations are executed natively in Python.
- **Database**: Dual resilient storage engine supporting PostgreSQL 16 as primary and SQLite 3 as local fallback.
- **Security**: Direct `bcrypt` password hashing (cost factor: 12) and JWT authentication with HS256 algorithm.

### 2. Reinforcement Learning Routing Engine
- **Model**: **Offline-to-Online MARL** (`best_model.pt`, 327,204 parameters).
- **Inference Speed**: ~4 ms (CUDA GPU) / ~15 ms (CPU).
- **Observation**: Exact 34-dimensional normalized vector (14 state/global features + 20 successor edge attributes).
- **Action Masking**: Hard topological constraints masking impassable, flooded, or topologically invalid transitions.
- **Transparent Decision Factors**: Replaces hallucinated explanations with exact numerical metrics:
  - `travel_time_seconds`
  - `distance_meters`
  - `hazard_exposure_score`
  - `traffic_exposure_score`
  - `agent_conflicts`
  - `route_risk_index`

### 3. Spatial Digital Twin & Simulation
- **Road Network**: 336-node, 736-edge urban network extracted from OpenStreetMap.
- **Graph Manager**: In-memory Dijkstra all-pairs shortest path caching with sub-millisecond lookup.
- **Hazard Engines**: Cellular automata flood propagation and anisotropic thermal fire diffusion with dynamic road closure triggers.
- **SUMO Integration**: Resilient non-blocking TraCI controller with automatic graceful fallback to kinematic traffic models.

### 4. Real-Time Telemetry & Mobile Client
- **WebSocket Protocol**: `/ws/evacuation/{session_id}` broadcasting state updates (~1.0s cadence) and route update events.
- **Mobile Client**: React Native 0.76 with Expo SDK 52 and TypeScript.
- **UX & Design**: Map-centric dark emergency UI, vector route rendering, live hazard contours, multi-agent position markers, and dynamic re-routing banners.
- **Audit Logging**: Full end-to-end telemetry persisted to database and accessible via the history and session audit screens.

---

## Automated Verification Suite

All 8 automated backend and integration tests pass with 100% success rate:
- Health and model readiness checks
- Authentication, JWT issuance, and user profile management
- Model metadata and parameter verification
- Multi-agent route calculation using Offline-to-Online MARL
- Research benchmark and zero-shot transfer access
- Evacuation session lifecycle and database persistence
- Bidirectional WebSocket communication and ping/pong
- Full end-to-end evacuation simulation flow
