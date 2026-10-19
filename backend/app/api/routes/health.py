import time
from fastapi import APIRouter, Response, status
from sqlalchemy import text

from backend.app.core.config import settings
from backend.app.database.session import engine
from backend.app.ml.model_manager import model_manager
from backend.app.routing.graph_manager import graph_manager

from backend.app.simulation.sumo_manager import sumo_manager
from backend.app.websocket.manager import ws_manager

router = APIRouter(tags=["Health"])

@router.get("/health", summary="Comprehensive platform health check")
def health_check(response: Response):
    # Check DB
    db_ok = False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    # Check Model
    model_status = model_manager.status
    model_ok = (model_status == "ready")

    # Check Graph
    graph_ok = (graph_manager.graph is not None and len(graph_manager.nodes) > 0)

    # Subsystem statuses
    sim_status = "ready" if sumo_manager else "fallback"
    hazards_status = "ready"
    websocket_status = "ready"

    overall_healthy = db_ok and model_ok and graph_ok
    if not overall_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "healthy" if overall_healthy else "degraded",
        "ready": overall_healthy,
        "database": "ready" if db_ok else "unavailable",
        "model": "ready" if model_ok else "unavailable",
        "graph": "ready" if graph_ok else "unavailable",
        "hazards": hazards_status,
        "simulation": sim_status,
        "websocket": websocket_status,
        "model_status": model_status,
        "model_device": model_manager.device,
        "road_graph_nodes": len(graph_manager.nodes),
        "road_graph_edges": graph_manager.graph.number_of_edges() if graph_manager.graph else 0,
        "evacuation_centers_configured": len(graph_manager.evacuation_centers),
        "version": settings.VERSION,
        "timestamp": time.time()
    }

@router.get("/health/ready", summary="Standardized subsystem readiness probe")
def health_readiness(response: Response):
    db_ok = False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    model_ok = (model_manager.status == "ready")
    graph_ok = (graph_manager.graph is not None and len(graph_manager.nodes) > 0)
    overall_ready = db_ok and model_ok and graph_ok

    if not overall_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ready" if overall_ready else "not_ready",
        "database": "ready" if db_ok else "unavailable",
        "model": "ready" if model_ok else "unavailable",
        "graph": "ready" if graph_ok else "unavailable",
        "hazards": "ready",
        "simulation": "ready"
    }

@router.get("/ready", summary="Readiness probe")
def readiness_check(response: Response):
    if model_manager.status != "ready" or not graph_manager.graph:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"ready": False, "reason": "Model or Graph not ready"}
    return {"ready": True}

@router.get("/live", summary="Liveness probe")
def liveness_check():
    return {"live": True}

@router.get("/health/diagnostics", summary="Detailed platform diagnostics for connection, database, and simulation engines")
def health_diagnostics():
    from datetime import datetime, timezone
    from backend.app.services.evacuation_service import evacuation_service

    # Measure DB ping latency
    t0 = time.perf_counter()
    db_ok = False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    db_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    has_sumo = sumo_manager.is_available() if sumo_manager else False

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": {
            "status": "connected" if db_ok else "unavailable",
            "ping_latency_ms": db_latency_ms,
            "engine": "PostgreSQL 16/18"
        },
        "model": {
            "status": model_manager.status,
            "model_name": model_manager.model_name,
            "device": str(model_manager.device),
            "parameter_count": model_manager.parameter_count,
            "observation_dim": model_manager.obs_dim,
            "action_masking": "Enforced"
        },
        "road_network": {
            "nodes": len(graph_manager.nodes),
            "edges": graph_manager.graph.number_of_edges() if graph_manager.graph else 0,
            "shelters": len(graph_manager.evacuation_centers),
            "status": "loaded"
        },
        "simulation": {
            "primary_engine": "SUMO/TraCI (Microscopic)" if has_sumo else "BPR Digital Twin (Active Production Engine)",
            "sumo_available": has_sumo,
            "fallback_active": not has_sumo,
            "running_sessions": len(evacuation_service.active_tasks),
            "max_agents_supported": 32
        },
        "websocket": {
            "status": "online",
            "active_connections": ws_manager.get_active_count() if hasattr(ws_manager, "get_active_count") else 0
        },
        "api_gateway": {
            "version": settings.VERSION,
            "environment": settings.ENV,
            "cors_configured": True
        }
    }


