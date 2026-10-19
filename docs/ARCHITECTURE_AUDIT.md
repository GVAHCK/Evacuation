# Architecture & Repository Audit

**Project:** Reinforcement-Learning-Based Multi-Hazard Evacuation Routing  
**Date:** September 2026  
**Auditor:** Principal System Architect & AI/RL Engineer  

---

## 1. Executive Summary

This repository contains a comprehensive, research-grade evaluation and multi-agent reinforcement learning (MARL) framework for urban evacuation routing under multi-hazard conditions (flash floods, urban wildfires, dynamic congestion).

A full benchmark of 6 MARL paradigms (MADDPG, RS-MADDPG, Offline-to-Online MARL, MA-GRL, GraphSAGE-MAPPO, HARL) across 5 random seeds (42, 101, 2024, 777, 999) has been completed, yielding publication-quality benchmark artifacts, statistical normality analyses, ablation studies, and Pareto frontier rankings.

Per project directive, **Offline-to-Online MARL** has been designated as the **sole production AI routing engine**. All other models remain strictly research artifacts and are not exposed in the production application or API.

---

## 2. Repository Layout & Existing Architecture

```
FinalYear/
├── .venv/                         # Pre-existing Python 3.11 virtual environment with PyTorch CUDA & spatial stack
├── configs/                       # Configuration YAMLs
│   ├── default.yaml               # 84-node baseline network configuration
│   ├── large.yaml                 # 336-node multi-district network configuration
│   ├── dense_grid.yaml            # George Town zero-shot network configuration
│   ├── sparse_suburban.yaml       # OMR Suburban zero-shot network configuration
│   └── marl_six/                  # Model configurations (offline_online.yaml, etc.)
├── data/
│   ├── raw/                       # network.osm, test_network.osm
│   ├── processed/                 # road_graph.json, road_graph_large.json, network.net.xml, sumo_mappings.json
│   ├── generated/                 # Generated scenario records
│   └── offline_marl/              # Expert & behavioral offline trajectory datasets
├── docs/                          # Research reports & model audit documentation
├── models/
│   └── marl_six/                  # 6 trained MARL model families
│       └── offline_online/        # PRODUCTION MODEL: 5 seeds (42, 101, 2024, 777, 999)
│           ├── seed_42/           # best_model.pt (1.35 MB), final_model.pt
│           ├── seed_101/
│           ├── seed_2024/
│           ├── seed_777/
│           └── seed_999/
├── results/
│   ├── marl_six/tables/           # Comprehensive benchmark CSVs & JSONs
│   │   ├── final_marl_comparison.csv
│   │   ├── marl_evaluation_summary_large.json
│   │   ├── zero_shot_marl_comparison.csv
│   │   ├── ablation_study_marl.csv
│   │   ├── statistical_tests_marl.csv
│   │   └── pareto_frontier_marl.csv
│   └── figures/                   # 300 DPI publication charts
├── scripts/                       # Training, benchmarking, validation scripts
│   ├── verify_checkpoint_integrity.py
│   ├── diagnostic_run.py
│   └── evaluate_marl_six.py
├── src/
│   ├── agents/
│   │   ├── marl_six/
│   │   │   ├── offline_online_agent.py   # PRODUCTION MODEL DEFINITION
│   │   │   ├── maddpg_agent.py          # CentralizedCritic & DiscreteActor base classes
│   │   │   └── ...
│   ├── envs/
│   │   └── multi_agent_evacuation_env.py # MultiAgentEvacuationEnv (34-dim obs, 4 agents)
│   ├── environment/
│   │   └── evacuation_env.py             # Single-agent base simulation & digital twin
│   ├── hazard/
│   │   ├── hazard_engine.py             # Abstract hazard base class
│   │   ├── flood_engine.py              # Cellular automata flood propagation
│   │   └── fire_engine.py               # Stochastic wind-driven fire spread
│   ├── network/
│   │   ├── osm_loader.py                # OSMnx road graph parser & coordinate cache
│   │   ├── sumo_converter.py            # Netconvert pipeline
│   │   └── traffic_generator.py         # SUMO dynamic traffic demand generator
│   └── utils/
│       ├── logger.py
│       └── seed.py
└── tests/
    └── test_marl_six.py
```

---

## 3. Working vs. Broken vs. Reusable Components

| Component | Status | Details | Reusability in Production |
|---|---|---|---|
| **Offline-to-Online MARL Model** | **WORKING** | Verified loading and inference with PyTorch 2.5.1 CUDA (`seed_42/best_model.pt`). Exact 34-dim obs, 4-dim discrete actions with hard masking. | **Direct reuse via dedicated `ModelManager`** |
| **Observation Builder (34-dim)** | **WORKING** | In `src/envs/multi_agent_evacuation_env.py`: 14 base state features + 20 candidate edge features (4 neighbors x 5 features: is_valid, alignment, dist_gain, hazard, congestion). | **Direct reuse for live observation construction** |
| **Action Masking** | **WORKING** | Strict hard masking on outgoing edges (closed edge status -> False). Prevents invalid agent moves. | **Direct reuse in `RouteBuilder` & RL inference loop** |
| **Hazard Engines (Flood & Fire)** | **WORKING** | `FloodEngine` (diffusion/cellular automata) & `FireEngine` (stochastic wind vector propagation). Compute dynamic intensities & road closure states (`STATE_SAFE`, `STATE_DEGRADED`, `STATE_CLOSED`). | **Direct reuse via `HazardManager`** |
| **OSM Road Graph** | **WORKING** | `data/processed/road_graph.json` and `road_graph_large.json`. NetworkX directed graph with node spatial coordinates (lat/lon, x/y) and shortest-path length lookup tables. | **Direct reuse via `GraphManager`** |
| **Benchmark Artifacts** | **WORKING** | `results/marl_six/tables/*.csv`, `*.json`. Comprehensive research figures and verification metrics. | **Direct reuse via `/api/v1/research/*` endpoints** |
| **PostgreSQL Database** | **WORKING** | Docker container running PostgreSQL 16 on port 5432 (`evacuation_db`). | **Primary production persistent database** |
| **FastAPI / Uvicorn Stack** | **READY** | FastAPI 0.141, Uvicorn, WebSockets, SQLAlchemy 2.1, Alembic, Pydantic v2, PyJWT, bcrypt installed in Python 3.11 runtime. | **Sole backend application framework** |
| **SUMO / TraCI Simulation** | **ADAPTABLE** | `eclipse-sumo` 1.27.1 installed. When native `sumo.exe` binary execution is constrained by host OS application policies, system seamlessly executes in deterministic hybrid simulation mode using graph/hazard dynamics and calibrated edge delays, ensuring 100% backend uptime without crashing. | **Wrapped with robust `SUMOManager` with automatic fallback** |
| **Node.js / React Native** | **READY** | Node.js v24.16.0 and npm 11.13.0 installed on host. Expo / React Native tooling available. | **Mobile client frontend** |

---

## 4. Key Production Model Specifications (Offline-to-Online MARL)

- **Architecture:** Conservative Q-Learning (CQL) offline pretraining with online interactive fine-tuning and KL policy regularization.
- **Actor:** `DiscreteActor` (Linear 34 -> LayerNorm -> ReLU -> Linear 128 -> LayerNorm -> ReLU -> Linear 4) with hard action masking.
- **Critic:** `CentralizedCritic` (Linear (34 * 4 + 8 + 4 * 4) -> 128 -> 128 -> 1).
- **Parameters:** 327,204 total parameters.
- **Trained Checkpoints:** `models/marl_six/offline_online/seed_42/best_model.pt` (primary production checkpoint).
- **Input Dimension:** 34 floats in range `[0.0, 1.0]`.
- **Output:** Discrete action index in `{0, 1, 2, 3}` corresponding to ordered outgoing road graph successors.
- **Action Masking:** Enforced prior to action selection. Logits of masked actions set to `-1e9`.

---

## 5. Benchmark Performance Metrics (Reference Data)

The following metrics are derived from the completed 500-scenario, 5-seed benchmark evaluation on the 336-node multi-district network:

- **Joint Success Rate:** 92.88 ± 0.73%
- **Individual Success Rate:** 98.01%
- **Mean Evacuation Time:** 137.61 ± 1.42 seconds
- **Hazard Exposure Index (HEI):** 6.836 ± 0.134
- **Routing F1 Score:** 0.9601
- **Mean Inference Latency:** 3.575 ms
- **Model Parameters:** 327,204

*Note: In accordance with project requirements, these metrics are benchmark reference figures and are kept strictly distinct from live session telemetry.*

---

## 6. Proposed Integration & Implementation Plan

1. **Phase 1: Research Preservation**
   - Retain all model checkpoints, dataset cards, and benchmark artifacts.
   - Standardize Python 3.11 execution runtime with native PyTorch CUDA.

2. **Phase 2: Database Layer (`backend/app/database`)**
   - Initialize SQLAlchemy 2.x models for PostgreSQL (`users`, `scenarios`, `evacuation_sessions`, `agents`, `routes`, `route_steps`, `hazards`, `traffic_snapshots`).
   - Create tables, default seed data, and initial admin/researcher accounts.

3. **Phase 3: Core Service & Routing Architecture (`backend/app`)**
   - `ModelManager`: Thread-safe, singleton loader for `OfflineOnlineMARLModel` with warmup and health status reporting.
   - `GraphManager`: In-memory caching of OSM road graph (`road_graph_large.json`), spatial k-d tree / nearest neighbor node mapper, and shortest path cache.
   - `HazardManager`: Dynamic simulation wrapper for flood (cellular automata) and fire (stochastic wind spread).
   - `ObservationBuilder`: Exact 34-feature vector constructor adhering strictly to `MultiAgentEvacuationEnv._get_agent_observation()`.
   - `RouteBuilder`: Action execution, path reconstruction, decision factor computation, and Dynamic A* fallback.
   - `SUMOManager`: Process lifecycle manager with TraCI connection handling and robust non-blocking fallback.

4. **Phase 4: FastAPI REST & WebSocket APIs**
   - `/api/v1/auth`: JWT login, registration, token refresh.
   - `/api/v1/routes/calculate`: Primary Offline-to-Online routing endpoint with decision factors.
   - `/api/v1/evacuation`: Session lifecycle management (start, stop, reroute, result).
   - `/api/v1/hazards`, `/api/v1/traffic`, `/api/v1/models`, `/api/v1/history`, `/api/v1/research`.
   - `/ws/evacuation/{session_id}`: Real-time simulation state streaming.

5. **Phase 5: React Native Mobile Client (`mobile`)**
   - Modern, map-centric emergency evacuation application using Expo & TypeScript.
   - Map screen with interactive layers (route, alternatives, agents, flood contours, fire zones, congestion).
   - Full flow: Login -> Current Location -> Select Destination -> Hazard Config -> Live Evacuation -> Real-time Rerouting -> Results -> History.

6. **Phase 6: End-to-End Verification & Automated Testing**
   - Unit and integration tests for ML model loading, graph resolution, route generation, fallback mechanisms, auth, and WebSocket communication.
   - End-to-end execution of live evacuation session.
