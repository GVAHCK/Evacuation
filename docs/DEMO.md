# Deterministic Emergency Evacuation Demo Guide

## Demo Scenario Parameters

- **Network**: 336-node urban road network (`road_graph_large.json`).
- **Agents**: 4 heterogeneous evacuation vehicles (High-Clearance SUV, City Commuter, Evacuation Bus, Emergency Response Van).
- **Origin**: Central Relief Shelter & Evacuation Hub (`14144290612`, Lat: `13.0036872`, Lon: `80.220072`).
- **Destination**: Northern High-Ground Community Stadium (`2212827519`, Lat: `13.0069116`, Lon: `80.2403241`).
- **Distance**: 3,654 meters (3.65 km).
- **Hazard**: Dynamic Flood Scenario (Severity: 0.85, Center: `[13.0050, 80.2300]`, Radius: 450 meters).
- **Model**: **Offline-to-Online MARL** (`best_model.pt`, 327,204 parameters).
- **Seed**: `42` (Deterministic reproducibility).

---

## Step-by-Step Live Walkthrough

1. **Launch FastAPI Backend**:
```bash
.\run_py.bat -m uvicorn backend.app.main:app --port 8000
```

2. **Open Mobile Client**:
Launch Expo in `mobile/`:
```bash
npx expo start
```

3. **Authenticate**:
- Login with demo researcher credentials:
  - **Email**: `officer@evac.gov`
  - **Password**: `Evacuation2026!`

4. **Navigate to Start Evacuation Wizard**:
- **Step 1 (Origin)**: Select `Central Relief Shelter`.
- **Step 2 (Destination)**: Select `Northern High-Ground Community Stadium`.
- **Step 3 (Hazard)**: Choose `Flood Hazard (Severity: High)`.
- **Step 4 (Agents)**: Set 4 Agents.
- Tap **Preview Recommended Route**.

5. **Observe Offline-to-Online MARL Route**:
- Inspect the 26 waypoints avoiding the central flood zone.
- Review transparent **Route Decision Factors** (Travel time ~380s, Hazard exposure <0.1, Risk <0.2).

6. **Confirm & Start Evacuation**:
- Real-time simulation launches.
- Map displays moving agent markers, flood contours, and traffic queues.

7. **Dynamic Re-routing in Action**:
- Trigger a dynamic hazard spread via the UI or `POST /api/v1/evacuation/{id}/reroute`.
- Notice the **DYNAMIC RE-ROUTING EXECUTED** banner appearing instantly on mobile.
- Route safely shifts around closed road segments in sub-second response time.

8. **Evacuation Complete**:
- Agents reach shelter. Results screen displays comprehensive metrics and saves the audit record into the persistent history log.
