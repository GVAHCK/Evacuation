# SUMO Traffic Simulation Integration Guide

## Overview

The platform couples the microscopic traffic simulation engine **Eclipse SUMO** (Simulation of Urban MObility) with the Offline-to-Online MARL routing model via Python's **TraCI** (Traffic Control Interface).

---

## SUMO Architecture & Lifecycle

1. **Manager**: `backend/app/simulation/sumo_manager.py` manages the SUMO background process lifecycle.
2. **Network Artifacts**:
   - `data/processed/network.net.xml` (Compiled SUMO road network)
   - `data/processed/road_graph_large.json` (Mapped OSM road graph)
3. **Resilient High-Throughput Mode**:
   - SUMO execution runs in a non-blocking background thread.
   - If SUMO is not installed on the host machine or if a TraCI communication error occurs, `SUMOManager` automatically degrades gracefully to a mathematical kinematic wave model, calculating congestion density from agent proximity and active road bottlenecks.
   - **SUMO crashes or missing binaries will NEVER crash the FastAPI backend or mobile clients.**

---

## Environment Variables

Configure in `.env`:
```ini
SUMO_HOME=/usr/share/sumo
SUMO_BINARY=sumo
```
*(On headless Linux servers, set `SUMO_BINARY=sumo` instead of `sumo-gui`).*

---

## Traffic Telemetry Extracted per Step

- `density`: Vehicular density normalized across active edges ($0.0$ to $1.0$).
- `congestion`: Bottleneck index representing halting queues and delay factor.
- `average_speed`: Current network travel velocity (meters/second).
- `edge_impedances`: Dynamic travel times factoring in lane saturation and reduced speeds through hazardous water/smoke zones.
