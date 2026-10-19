# React Native Mobile Application Setup Guide

## Architecture

The mobile client is built using **React Native 0.76**, **Expo SDK 52**, and **TypeScript**. It connects to the FastAPI backend via REST APIs and real-time WebSockets to provide a map-centric emergency routing and live evacuation interface.

---

## Directory Structure

```
mobile/
├── App.tsx                     # Top-level application component
├── index.js                    # Expo root entry point
├── package.json                # Dependencies and build scripts
├── app.json                    # Expo project configuration
├── tsconfig.json               # TypeScript configuration
└── src/
    ├── api/
    │   └── client.ts           # Axios HTTP client & WebSocket base URL
    ├── components/
    │   ├── MapRenderer.tsx     # Vector road graph, hazard zones & agent markers
    │   └── RouteDecisionCard.tsx # Transparent route decision factors display
    ├── navigation/
    │   ├── RootNavigator.tsx   # Stack navigator (Splash, Auth, Main, Details)
    │   └── TabNavigator.tsx    # Bottom tab navigator (Home, Map, Evacuate, History, Research)
    ├── screens/
    │   ├── Splash.tsx          # Animated loading & token verification
    │   ├── OnboardingScreen.tsx# Feature overview & hazard capabilities
    │   ├── Login.tsx           # Authentication screen
    │   ├── Register.tsx        # New account registration
    │   ├── HomeScreen.tsx      # Evacuation dashboard & quick actions
    │   ├── LiveMapScreen.tsx   # Map with active hazards and shelters
    │   ├── StartEvacuationScreen.tsx # 4-step wizard (Origin, Dest, Hazard, Agents)
    │   ├── EvacuationPreviewScreen.tsx # Route preview & decision factors
    │   ├── LiveEvacuationScreen.tsx # Real-time simulation tracking & rerouting alerts
    │   ├── EvacuationResultScreen.tsx # Final outcome, travel time & exposure
    │   ├── HistoryScreen.tsx   # Historical evacuation log
    │   ├── SessionDetailsScreen.tsx # Detailed telemetry audit for past sessions
    │   ├── ScenarioBuilderScreen.tsx # Custom multi-hazard scenario designer
    │   ├── ResearchScreen.tsx  # Benchmark metrics & statistical validation
    │   ├── ProfileScreen.tsx   # User profile & logout
    │   └── SettingsScreen.tsx  # API URL & simulation preferences
    ├── store/
    │   ├── authStore.ts        # Zustand auth state & JWT token persistence
    │   └── evacuationStore.ts  # Active session, route, and WebSocket telemetry
    ├── theme/
    │   └── colors.ts           # Curated dark emergency color palette
    └── types/
        └── index.ts            # TypeScript interfaces matching backend schemas
```

---

## Installation & Running

1. **Install Dependencies**:
```bash
cd mobile
npm install
```

2. **Configure API Base URL**:
In `mobile/src/screens/SettingsScreen.tsx` or `mobile/src/api/client.ts`, configure the API URL:
- **Android Emulator**: `http://10.0.2.2:8000` (Default)
- **iOS Simulator**: `http://localhost:8000`
- **Physical Device**: `http://<YOUR_LAN_IP>:8000`

3. **Start Development Server**:
```bash
npx expo start
```
