# FastAPI Backend Technical Guide

## Architecture

The backend is built exclusively on **FastAPI** (Python 3.11+) as the single backend for the platform. Python directly executes the PyTorch RL model, NetworkX graph algorithms, hazard propagation engines, and SUMO TraCI client.

---

## Directory Structure

```
backend/
├── app/
│   ├── main.py                     # Application entry point, CORS, and WebSocket router
│   ├── core/
│   │   ├── config.py               # Pydantic v2 BaseSettings configuration
│   │   ├── logging.py              # Centralized logging setup
│   │   └── security.py             # bcrypt password hashing & JWT encoding/decoding
│   ├── api/
│   │   ├── deps.py                 # Dependency injection (DB session, JWT auth)
│   │   └── routes/                 # Modular REST routes
│   │       ├── auth.py             # /auth/register, /auth/login, /auth/refresh
│   │       ├── users.py            # /users/me
│   │       ├── routes.py           # /routes/calculate (Offline-to-Online MARL)
│   │       ├── evacuation.py       # /evacuation/start, stop, reroute, result
│   │       ├── hazards.py          # /hazards/active, /hazards/simulate
│   │       ├── traffic.py          # /traffic/current
│   │       ├── models.py           # /models/offline-to-online
│   │       ├── history.py          # /history, /history/{id}
│   │       ├── scenarios.py        # /scenarios CRUD & run
│   │       ├── research.py         # /research/benchmark, /research/zero-shot
│   │       └── health.py           # /health, /ready, /live
│   ├── database/
│   │   ├── session.py              # PostgreSQL engine with SQLite fallback
│   │   └── base.py                 # DeclarativeBase definition
│   ├── models/                     # SQLAlchemy 2.0 ORM entities
│   │   ├── user.py
│   │   ├── scenario.py
│   │   ├── evacuation_session.py
│   │   ├── agent.py
│   │   ├── route.py
│   │   ├── hazard.py
│   │   ├── traffic.py
│   │   └── audit_log.py
│   ├── schemas/                    # Pydantic v2 data validation models
│   ├── ml/                         # Offline-to-Online MARL inference & builder
│   │   ├── model_manager.py        # Thread-safe PyTorch singleton
│   │   └── observation_builder.py  # 34-dim observation & action masking
│   ├── routing/                    # Spatial graph manager & route builder
│   │   ├── graph_manager.py        # 336-node OSM road graph cache
│   │   ├── route_builder.py        # Multi-agent MARL routing & factors
│   │   ├── route_validator.py      # Route continuity & safety validator
│   │   └── fallback_router.py      # Dynamic A* fallback
│   ├── hazards/                    # Cellular flood & thermal fire simulation
│   │   └── hazard_manager.py
│   ├── simulation/                 # SUMO TraCI & traffic state manager
│   │   └── sumo_manager.py
│   └── websocket/                  # Live telemetry broadcasting
│       └── manager.py
```

---

## Authentication Flow

1. **Register**: `POST /api/v1/auth/register` creates a user account with a salted `bcrypt` hash (cost factor: 12).
2. **Login**: `POST /api/v1/auth/login` verifies credentials and issues:
   - `access_token` (JWT with HS256 algorithm, 24-hour expiration)
   - `refresh_token` (JWT, 7-day expiration)
3. **Protected Endpoints**: Require `Authorization: Bearer <access_token>` header.

---

## Running the Backend

```bash
.\run_py.bat -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```
