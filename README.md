# Multi-Hazard Evacuation Intelligence Platform

[![Platform: FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Client: React Native](https://img.shields.io/badge/Mobile-React%20Native%20%2F%20Expo-61DAFB.svg?style=flat&logo=react)](https://reactnative.dev)
[![AI: Offline-to-Online MARL](https://img.shields.io/badge/AI-Offline--to--Online%20MARL-8B5CF6.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Database: PostgreSQL 16](https://img.shields.io/badge/Database-PostgreSQL%2016-336791.svg?style=flat&logo=postgresql)](https://www.postgresql.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end, mobile-first emergency evacuation intelligence platform powered by **Offline-to-Online Multi-Agent Reinforcement Learning (MARL)**, OpenStreetMap road networks, dynamic flood and fire hazard propagation, microscopic SUMO traffic digital twin integration, a high-performance Python FastAPI backend, and a modern React Native mobile app.

---

## Architecture Overview

```
                 ┌────────────────────────────────────────────────────────┐
                 │             REACT NATIVE / EXPO CLIENT                 │
                 │                                                        │
                 │ • Map-Centric Dynamic Navigation (MapRenderer)         │
                 │ • Emergency Route Preview & Transparent Factors        │
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

## Key System Features

1. **Exclusively Offline-to-Online MARL in Production**:
   - The production routing engine uses the benchmark-winning **Offline-to-Online MARL** model (`seed_42/best_model.pt`, 327,204 parameters).
   - Generates coordinated multi-vehicle routes in sub-4ms on GPU (~15ms on CPU).
   - Enforces hard action masking preventing transitions into flooded roads, fire zones, or topologically disconnected links.
2. **Transparent Route Decision Factors**:
   - Exposes mathematically defined numerical decision factors (`travel_time_seconds`, `distance_meters`, `hazard_exposure_score`, `traffic_exposure_score`, `agent_conflicts`, `route_risk_index`) rather than natural-language hallucinations.
3. **Dynamic Re-Routing Engine**:
   - When flood waters rise, road blockages occur, or traffic congestion spikes, the backend dynamically calculates safe alternate routes and pushes `route_update` frames over WebSockets.
4. **Single Backend Architecture**:
   - Built exclusively in **FastAPI** (No Node.js/Express backend).
   - Python directly interfaces with PyTorch, NetworkX, and SUMO without serialization bridges.
5. **Dual Resilient Database Storage**:
   - Connects to **PostgreSQL 16** as primary engine with automatic seamless fallback to **SQLite 3** (`evacuation.db`).
6. **Mobile-First Emergency Application**:
   - Built with **React Native (0.86.3)**, **Expo SDK 57**, **React 19.2.3**, and **TypeScript**.
   - Features an **Apple Liquid Glass-inspired Design System** (`LiquidGlassTokens`, `GlassCard`, `GlassButton`, `GlassHeader`, `GlassPill`) with translucent surfaces, specular refraction edge highlights, and high-contrast emergency accents.
   - Comprehensive `@expo/vector-icons` iconography with screen-reader accessibility labels.
   - Hardware-backed token persistence using `expo-secure-store`.
   - 15 functional screens: Splash, Onboarding, Login, Register, Dashboard, Live Map, Start Evacuation Wizard, Route Preview, Live Simulation Tracking, Results, History, Session Audit, Scenario Builder, Research Benchmarks, Profile, and Settings.
7. **Comprehensive Security & Production Hardening**:
   - Explicit CORS allowlist with strict origin filtering (no wildcards with credentials).
   - JWT token type separation (`access` vs `refresh`), bcrypt password hashing, and token expiration.
   - Cross-user session isolation and tenant authorization enforcement (HTTP 403 Forbidden).
   - Authenticated WebSocket telemetry stream (`?token=...`) with session ownership validation (Code 1008 policy violation on intrusion).
   - In-memory sliding-window token bucket rate limiter protecting authentication and MARL routing endpoints.
   - PyTorch `weights_only=True` checkpoint loading to mitigate deserialization vulnerabilities.

---

## Quick Start

### 1. Database
Launch PostgreSQL 16 container:
```bash
docker compose up -d postgres
```
*(If Docker is unavailable, the backend automatically uses SQLite `evacuation.db`).*

### 2. FastAPI Backend
Run the backend server on port 8000:
```bash
# Windows
.\run_py.ps1 -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

# Linux / macOS
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```
- API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

### 3. React Native Mobile App
```bash
cd mobile
npm install
npx expo-doctor
npx expo start
```
- Press `a` for Android Emulator.
- Press `w` for Web Browser.
- Scan QR code with Expo Go for physical devices.

---

## Automated Acceptance Test Suite

Run the full automated test suite (46 tests across 10 suites covering Unit, API, WebSocket, Observation Builder, Hard Action Masking, Security Matrix, and Section 36 E2E Live Simulation):
```powershell
.\run_py.ps1 -m pytest tests/ -v
```

Execute the 22-requirement black-box production validation:
```powershell
.\run_py.ps1 scripts/blackbox_production_validation.py
```

Execute the Section 36 Live Simulation Movement & Completion test:
```powershell
.\run_py.ps1 -m pytest tests/test_live_simulation_and_movement.py -v
```

Verify research artifact integrity:
```powershell
.\run_py.ps1 scripts/verify_research_integrity.py
```

Verify Python environment and dependencies:
```powershell
.\run_py.ps1 scripts/verify_python_environment.py
```

Verify mobile application assets:
```powershell
.\run_py.ps1 scripts/verify_mobile_assets.py
```

---

## Production Documentation Index

- [Production Setup & Operator Runbook](docs/PRODUCTION_SETUP.md)
- [Production Architecture & System Audit](docs/PRODUCTION_AUDIT.md)
- [Production Validation Matrix & Test Evidence](docs/PRODUCTION_VALIDATION.md)
- [Final Production Completion Report](docs/FINAL_PRODUCTION_COMPLETION_REPORT.md)
- [Mobile Application Guide](docs/MOBILE.md)
- [REST API Reference](docs/API.md)
- [Real-Time WebSocket Protocol](docs/WEBSOCKET.md)
- [Database Schema & ERD](docs/DATABASE.md)
- [Production Docker Deployment](docs/DEPLOYMENT.md)
- [End-to-End Operational Lifecycle](docs/END_TO_END_FLOW.md)
- [Troubleshooting & FAQ](docs/TROUBLESHOOTING.md)
- [Scientific Benchmark & Research Specification](docs/RESEARCH.md)
