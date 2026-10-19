# System Architecture & Technical Specification

## Overview

The **Reinforcement-Learning-Based Multi-Hazard Evacuation Routing Platform** transforms research-grade multi-agent reinforcement learning into a production emergency evacuation system. It coordinates multi-agent vehicular routing during dynamic disasters (urban floods, flash fires, and emergent congestion) over real-world OpenStreetMap (OSM) road networks with SUMO microscopic traffic digital twin integration.

---

## High-Level Topology

```
                 ┌────────────────────────────────────────────────────────┐
                 │             REACT NATIVE / EXPO CLIENT                 │
                 │                                                        │
                 │ • Map-Centric Dynamic Navigation (MapRenderer)         │
                 │ • Emergency Route Preview & Transparent Factors       │
                 │ • Live WebSocket Telemetry & Agent Tracking            │
                 │ • Dynamic Re-Routing Push Alerts                       │
                 │ • Scenario Builder & Research Mode                     │
                 └───────────────────────────┬────────────────────────────┘
                                             │
                               HTTP REST (JSON) + WebSocket (/ws)
                                             │
                                             ▼
                 ┌────────────────────────────────────────────────────────┐
                 │                 FASTAPI BACKEND ENGINE                 │
                 │                                                        │
                 │ • API Routers (Auth, Evacuation, Routes, Hazards)      │
                 │ • Token Authentication (JWT Access & Refresh, bcrypt)   │
                 │ • WebSocket ConnectionManager & Broadcast Broker       │
                 │ • Dual Resilient Storage: PostgreSQL 16 & SQLite       │
                 │ • Evacuation Loop & Dynamic Rerouting Engine           │
                 └──────────────┬──────────────────────────┬──────────────┘
                                │                          │
                                ▼                          ▼
     ┌────────────────────────────────────┐   ┌───────────────────────────┐
     │      OFFLINE-TO-ONLINE MARL        │   │    SPATIAL DIGITAL TWIN   │
     │                                    │   │                           │
     │ • PyTorch 2.5.1 Checkpoint Loader  │   │ • 336-Node OSM Road Graph │
     │ • Singleton Model Manager (CUDA)   │   │ • GraphManager (Dijkstra) │
     │ • 34-Dim Observation Vector Builder│   │ • SUMO TraCI Controller   │
     │ • Hard Action Masking Engine       │   │ • Cellular Flood Engine   │
     │ • Sub-4ms Multi-Agent Inference    │   │ • Thermal Fire Engine     │
     └────────────────────────────────────┘   └───────────────────────────┘
```

---

## Core System Subsystems

### 1. Unified FastAPI Backend
- **Technology**: Python 3.11+, FastAPI, Uvicorn, SQLAlchemy 2.x, Pydantic v2.
- **Role**: Single backend for the entire system (No Express/Node.js backend). Python directly interfaces with PyTorch, NetworkX, and SUMO without serialization bridges.
- **Endpoints**:
  - `/api/v1/auth`: Registration, JWT issuance, token refresh, and logout.
  - `/api/v1/routes`: Direct Offline-to-Online MARL route calculation with action masking.
  - `/api/v1/evacuation`: Multi-agent evacuation session initiation, stepping, stop, dynamic rerouting, and results.
  - `/ws/evacuation/{session_id}`: Real-time bi-directional telemetry broadcast over WebSocket.
  - `/api/v1/hazards`: Dynamic flood depth and fire contour configuration.
  - `/api/v1/traffic`: Real-time density, congestion index, and average speed from SUMO.
  - `/api/v1/research`: Access to peer-reviewed benchmark datasets (500 scenarios, zero-shot transfers, statistical significance tables).

### 2. Production AI Engine: Offline-to-Online MARL
- **Trained Artifact**: `models/marl_six/offline_online/seed_42/best_model.pt`.
- **Architectural Policy**: Multi-Agent Actor-Critic with centralized training and decentralized execution.
- **Model Parameters**: 327,204 float parameters.
- **Hardware Acceleration**: Automatic CUDA detection with seamless CPU fallback.
- **Action Masking**: Hard constraints masking closed, flooded, or topologically invalid road successors.
- **Transparent Decision Factors**: Instead of pseudo-natural-language hallucinations, the engine exposes mathematically defined numerical factors:
  - `travel_time_seconds`
  - `distance_meters`
  - `hazard_exposure_score`
  - `traffic_exposure_score`
  - `agent_conflicts`
  - `route_risk_index`

### 3. Spatial Digital Twin & Hazard Propagation
- **Road Network**: 336-node, 736-edge urban network extracted from OpenStreetMap (`data/processed/road_graph_large.json`).
- **GraphManager**: Caches topological adjacency, node GPS coordinates, and pre-computed all-pairs shortest path tables for fast sub-millisecond lookups.
- **Flood Engine**: Cellular automata model simulating elevation-dependent hydrological flow and water depth accumulation.
- **Fire Engine**: 2D anisotropic thermal diffusion model simulating wind-driven flame front propagation.
- **Road Closures**: Edges exceeding critical water depth (>0.3m) or thermal intensity (>0.6) are dynamically closed, forcing real-time re-routing.

### 4. SUMO Traffic Digital Twin
- **Controller**: `backend/app/simulation/sumo_manager.py`.
- **Integration**: TraCI (Traffic Control Interface) running as a non-blocking background subprocess.
- **Graceful Degradation**: If SUMO binaries are unavailable on the host, the system seamlessly transitions to a mathematical kinematic wave traffic model without interrupting FastAPI or mobile clients.

### 5. Mobile Application Client
- **Framework**: React Native 0.76 with Expo SDK 52 and TypeScript.
- **State Management**: Zustand stores for authentication and real-time evacuation telemetry.
- **Mapping**: Dynamic vector map renderer (`MapRenderer`) displaying road links, hazard polygons, multi-agent positions, and recommended vs alternative paths.
- **Navigation**: Full native stack with splash, onboarding, authentication, dashboard, live map, evacuation preview, live tracking, and historical audit screens.
