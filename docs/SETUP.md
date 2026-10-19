# Full Platform Setup & Installation Guide

This guide walks you through setting up and running the complete **Multi-Hazard Evacuation Intelligence Platform**.

---

## 1. Prerequisites

- **Python**: 3.11.x
- **Node.js**: v18+ or v20 LTS
- **Docker & Docker Compose**: (Optional, for running PostgreSQL 16 container)
- **Git**: For version management
- **SUMO** (Eclipse Simulation of Urban MObility): Optional, for microscopic traffic simulation

---

## 2. Environment Configuration

Copy the example environment configuration into `.env`:

```bash
cp .env.example .env
```

Ensure the key parameters match your local setup:
```ini
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/evacuation_db
JWT_SECRET=super-secret-marl-evacuation-key-2026-secure-token
JWT_REFRESH_SECRET=super-secret-refresh-marl-key-2026-secure-token
MODEL_CHECKPOINT=models/marl_six/offline_online/seed_42/best_model.pt
MODEL_DEVICE=cuda
OSM_GRAPH_PATH=data/processed/road_graph_large.json
```

---

## 3. Database Setup

### Option A: Using Docker (Recommended)
Launch the isolated PostgreSQL 16 container:
```bash
docker compose up -d postgres
```

The database container runs on `localhost:5432` with database `evacuation_db`.

### Option B: Automatic SQLite Fallback
If PostgreSQL or Docker is not installed, the platform's resilient database engine automatically initializes and manages a local SQLite file (`evacuation.db`) with identical schema definitions.

---

## 4. Starting the FastAPI Backend

Run using the project launcher:
```bash
# On Windows
.\run_py.bat -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# On Linux / macOS
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Verify backend health:
```bash
curl http://localhost:8000/health
```

Expected JSON response:
```json
{
  "status": "healthy",
  "database": "connected",
  "model_status": "ready",
  "model_device": "cuda",
  "road_graph_nodes": 336
}
```

Interactive OpenAPI documentation is available at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 5. Starting the React Native Mobile Application

Navigate to the `mobile/` directory:
```bash
cd mobile
npm install
```

Start the Expo development server:
```bash
npx expo start
```

### Running on Device / Emulator:
- **Android Emulator**: Press `a` in the terminal. The app will communicate with the backend at `http://10.0.2.2:8000`.
- **Physical Device**: Scan the QR code using Expo Go. Ensure your phone and PC are on the same Wi-Fi network and update the API base URL in `mobile/src/screens/SettingsScreen.tsx` to your PC's LAN IP.
- **Web Browser**: Press `w` in the terminal to launch the web client.

---

## 6. Running Acceptance Tests

Run the complete automated test suite (8 tests covering Auth, Routing, Models, WebSockets, and End-to-End Simulation):
```bash
.\run_py.bat -m pytest tests/test_backend_api.py tests/test_websocket.py tests/test_end_to_end_simulation.py -v
```

All 8 tests must pass with 100% success.
