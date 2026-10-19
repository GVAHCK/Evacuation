# Production Setup & Operator Runbook

This guide contains verified, end-to-end instructions for deploying, running, and validating the **Reinforcement-Learning Multi-Hazard Evacuation System** in production and local environments.

---

## 1. System Architecture & Prerequisites

The platform consists of:
- **Mobile Client**: React Native + Expo SDK 52 + TypeScript + vector map rendering
- **Backend API**: FastAPI + Python 3.11 + PyTorch (CUDA 12.1 / CPU fallback)
- **Database**: PostgreSQL 16 on port 5432 (`evacuation_db`)
- **Spatial Network**: OpenStreetMap graph (336 nodes, 736 edges)
- **Traffic Engine**: SUMO / TraCI with resilient digital twin fallback
- **Production AI Model**: Offline-to-Online Multi-Agent Reinforcement Learning (MARL) checkpoint with exactly **327,204 parameters**

### System Requirements
- **OS**: Windows 11 / Windows Server / Linux
- **Python**: 3.11.x (managed via project virtual environment `.venv`)
- **Node.js**: v18+ or v20 LTS
- **Docker**: Docker Desktop for Windows (with WSL2 backend)

---

## 2. Windows Environment Scripts

To prevent Windows execution policy restrictions and Smart App Control interference, use the verified project execution scripts:
- **PowerShell**: `.\run_py.ps1`
- **Command Prompt (CMD)**: `.\run_py.bat`

Verify your Python environment dependencies:
```powershell
.\run_py.ps1 scripts/verify_python_environment.py
```
Expected output:
```
============================================================
VERIFYING PYTHON PRODUCTION DEPENDENCIES
============================================================
[PASS] PyTorch (torch) version: 2.5.1+cu121
[PASS] NumPy version: 2.4.6
...
ALL PRODUCTION DEPENDENCIES IMPORTED SUCCESSFULLY!
```

---

## 3. Database Initialization (PostgreSQL)

1. Launch the PostgreSQL production container:
```powershell
docker compose up -d postgres
```
2. Verify container health:
```powershell
docker ps --filter "name=evac_postgres"
```
3. Canonical connection string:
```
postgresql+psycopg2://postgres:postgres@localhost:5432/evacuation_db
```
The database engine automatically verifies connectivity on startup and initializes schemas if not present (`init_db()`).

---

## 4. Model & Artifact Verification

Verify all 30 research model checkpoints and tables are 100% intact:
```powershell
.\run_py.ps1 scripts/verify_research_integrity.py
```

The production application strictly consumes the pre-trained Offline-to-Online MARL model:
- **Path**: `models/marl_six/offline_online/seed_42/best_model.pt`
- **Parameter Count**: `327,204`
- **Observation Space**: `34 dimensions`
- **Action Space**: `4 dimensions` with hard action masking

---

## 5. Starting the Backend API

Launch the production FastAPI server:
```powershell
.\run_py.ps1 -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```
Or with CMD:
```cmd
.\run_py.bat -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Verify backend readiness:
```powershell
curl http://localhost:8000/health
curl http://localhost:8000/health/ready
```
Expected JSON output:
```json
{
  "status": "ready",
  "database": "ready",
  "model": "ready",
  "graph": "ready",
  "hazards": "ready",
  "simulation": "ready"
}
```

Interactive OpenAPI Swagger UI is available at:
`http://localhost:8000/docs`

---

## 6. Mobile Application Setup (React Native + Expo)

1. Navigate to the mobile directory:
```powershell
cd mobile
```
2. Install dependencies:
```powershell
npm install
```
3. Validate Expo environment and assets:
```powershell
npx expo-doctor
cd ..
.\run_py.ps1 scripts/verify_mobile_assets.py
cd mobile
```
4. Check TypeScript types:
```powershell
npx tsc --noEmit
```
5. Configure backend endpoint in `mobile/.env`:
- **Android Emulator**: `EXPO_PUBLIC_API_URL=http://10.0.2.2:8000`
- **Physical Device**: `EXPO_PUBLIC_API_URL=http://<YOUR_PC_LAN_IP>:8000`
  *(Find your PC LAN IP by running `ipconfig`)*

6. Launch the development server:
```powershell
npx expo start -c
```
- Press `a` for Android Emulator.
- Scan the displayed QR code with the Expo Go app on a physical device.

---

## 7. Running the Automated Test Suite

Run the full automated test suite (35 tests):
```powershell
.\run_py.ps1 -m pytest tests/ -v
```

Run the Section 36 Live Simulation & Movement Verification test:
```powershell
.\run_py.ps1 tests/test_live_simulation_and_movement.py
```
Expected output:
```
=================================================================
ALL 20 CRITICAL SECTION 36 SIMULATION REQUIREMENTS SATISFIED!
=================================================================
```
