# Mobile Application Architecture & Deployment Guide

## 1. Overview
The mobile application is a high-performance cross-platform emergency evacuation interface built with:
- **Framework**: React Native + Expo SDK 52
- **Language**: TypeScript (Strict typing, 0 compilation errors)
- **State Management**: Zustand stores with persistent security token storage via `expo-secure-store`
- **Networking**: Axios HTTP client with auto-retry and JWT interceptors + native WebSocket client with auto-reconnect and heartbeat ping/pong
- **Mapping**: Dynamic vector SVG map renderer (`MapRenderer.tsx`) for the 336-node, 736-edge OpenStreetMap road network

---

## 2. Directory Structure

```
mobile/
├── assets/                       # Production branded assets
│   ├── icon.png                 # App icon (1024x1024)
│   ├── splash.png               # Splash screen (2048x2048)
│   ├── adaptive-icon.png        # Android adaptive foreground (1024x1024)
│   └── favicon.png              # Web favicon (48x48)
├── src/
│   ├── api/
│   │   ├── client.ts            # Configurable API base URL, interceptors
│   │   ├── auth.ts              # Authentication endpoints
│   │   ├── routes.ts            # Route calculation endpoints
│   │   ├── evacuation.ts        # Evacuation session lifecycle
│   │   ├── history.ts           # Past session audit queries
│   │   └── models.ts            # Model status inspection
│   ├── components/
│   │   ├── MapRenderer.tsx      # SVG vector road network, agents, hazards
│   │   ├── MetricCard.tsx       # Live telemetry HUD components
│   │   └── ...
│   ├── navigation/
│   │   └── RootNavigator.tsx    # Stack navigation across all screens
│   ├── screens/                 # All 15+ verified screens
│   │   ├── SplashScreen.tsx
│   │   ├── LoginScreen.tsx
│   │   ├── RegisterScreen.tsx
│   │   ├── HomeScreen.tsx
│   │   ├── LiveMapScreen.tsx
│   │   ├── ScenarioBuilderScreen.tsx
│   │   ├── EvacuationPreviewScreen.tsx
│   │   ├── LiveEvacuationScreen.tsx
│   │   ├── EvacuationResultScreen.tsx
│   │   ├── HistoryScreen.tsx
│   │   ├── SessionDetailScreen.tsx
│   │   ├── ResearchScreen.tsx
│   │   ├── ProfileScreen.tsx
│   │   └── SettingsScreen.tsx
│   └── store/
│       ├── authStore.ts         # User session, JWT lifecycle, secure token loading
│       └── evacuationStore.ts   # Session telemetry, WebSocket stream, rerouting events
├── app.json                     # Expo configuration
├── package.json                 # Aligned Expo SDK 52 dependencies
└── tsconfig.json                # TypeScript compiler configuration
```

---

## 3. Configuration & Networking

### Dynamic Endpoint Resolution
The mobile client uses `EXPO_PUBLIC_API_URL` to configure the backend API and WebSocket endpoints:
- **Android Emulator**: `http://10.0.2.2:8000` (translates to host PC loopback)
- **Physical Android Device**: `http://<YOUR_PC_LAN_IP>:8000` (e.g. `http://192.168.1.15:8000`)
- **iOS Simulator / Web**: `http://localhost:8000`

To set the variable:
Create or edit `mobile/.env`:
```ini
EXPO_PUBLIC_API_URL=http://10.0.2.2:8000
```
Or override dynamically at runtime within the app via **Settings Screen**.

---

## 4. State Management Lifecycle

### `authStore`
- Manages user login, registration, token persistence, and logout.
- Uses `expo-secure-store` to retrieve stored JWT tokens upon application boot via `loadToken()`.
- Rejects unauthenticated requests with clean redirects to `LoginScreen`.

### `evacuationStore`
- Tracks the active evacuation session lifecycle through states:
  `idle` -> `creating` -> `starting` -> `running` -> `rerouting` -> `completed` / `stopped` / `failed`
- Maintains WebSocket subscription to `/ws/evacuation/{session_id}`.
- Sends heartbeat ping every 10 seconds to maintain connection health.
- Employs automatic reconnect with exponential backoff if connectivity drops.
- Dispatches dynamic reroute updates immediately upon receiving `event: "reroute"`.

---

## 5. Map Rendering (`MapRenderer.tsx`)
The custom SVG-based map renderer projects OSM geographic coordinates $(lat, lon)$ into scalable canvas units:
- Renders 736 network edges with hazard-severity coloring (blue for flood, red for fire, purple for combined).
- Renders origin, destination, and multi-agent cohorts with distinct markers.
- Updates live agent coordinates dynamically on every WebSocket telemetry frame.
- High performance without requiring external Google Maps API keys or third-party billing.

---

## 6. Verification & Quality Gates

Run asset verification:
```powershell
.\run_py.ps1 scripts/verify_mobile_assets.py
```

Run Expo environment check:
```powershell
cd mobile
npx expo-doctor
```
Result: **18/18 checks passed!**

Compile TypeScript:
```powershell
npx tsc --noEmit
```
Result: **0 errors.**
