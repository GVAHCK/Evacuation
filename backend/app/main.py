import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Query, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.database.session import init_db, SessionLocal
from backend.app.services.auth_service import auth_service
from backend.app.ml.model_manager import model_manager
from backend.app.routing.graph_manager import graph_manager
from backend.app.simulation.sumo_manager import sumo_manager
from backend.app.websocket.manager import ws_manager

from backend.app.api.routes import (
    auth,
    users,
    scenarios,
    evacuation,
    routes,
    hazards,
    traffic,
    models,
    history,
    health,
    research,
    notifications
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Platform startup and graceful shutdown lifecycle."""
    logger.info(f"Starting {settings.PROJECT_NAME} (v{settings.VERSION})...")
    # 1. Initialize DB schemas & seed accounts
    init_db()
    with SessionLocal() as db:
        auth_service.seed_default_users(db)

    # 2. Touch GraphManager and ModelManager singletons to warm up
    _ = graph_manager.nodes
    _ = model_manager.get_metadata()
    logger.info(f"Startup complete. Model status: {model_manager.status}. Road graph nodes: {len(graph_manager.nodes)}.")

    yield

    # Clean shutdown
    logger.info("Initiating platform shutdown...")
    sumo_manager.stop()
    logger.info("Shutdown sequence complete.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Production API for Reinforcement-Learning-Based Multi-Hazard Evacuation Routing. "
        "Powered exclusively by Offline-to-Online Multi-Agent Reinforcement Learning (MARL), "
        "OpenStreetMap road networks, cellular automata hazard engines, and SUMO traffic coupling."
    ),
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["Authorization", "Content-Type", "Accept", "X-Request-ID"],
)

# Request ID & Audit Logging Middleware
@app.middleware("http")
async def audit_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    t0 = time.perf_counter()

    response = await call_next(request)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-MS"] = f"{latency_ms:.2f}"

    if not request.url.path.startswith("/health") and not request.url.path.startswith("/live"):
        logger.info(f"[{request.method}] {request.url.path} -> {response.status_code} ({latency_ms:.1f}ms) [ReqID: {request_id}]")

    return response

# Standardized Error Handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    detail = exc.detail
    if isinstance(detail, dict):
        code = detail.get("code", "HTTP_ERROR")
        message = detail.get("message", str(exc.detail))
        extra = detail
    else:
        code = "HTTP_ERROR"
        message = str(exc.detail)
        extra = {}

    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": code,
            "message": message,
            "details": extra,
            "request_id": req_id
        }
    )

from fastapi.encoders import jsonable_encoder

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "code": "VALIDATION_ERROR",
            "message": "Invalid request parameters",
            "details": jsonable_encoder(exc.errors()),
            "request_id": req_id
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.error(f"Unhandled server exception: {exc} [ReqID: {req_id}]", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "code": "INTERNAL_SERVER_ERROR",
            "message": "An internal server error occurred while processing the evacuation operation",
            "details": str(exc),
            "request_id": req_id
        }
    )

# WebSocket Endpoint for Live Evacuation Telemetry
@app.websocket("/ws/evacuation/{session_id}")
async def websocket_evacuation_endpoint(
    websocket: WebSocket,
    session_id: str,
    token: Optional[str] = Query(None)
):
    with SessionLocal() as db:
        connected = await ws_manager.authorize_and_connect(websocket, session_id, token, db)
        if not connected:
            return

    try:
        while True:
            # Receive client ping or control signals
            data = await websocket.receive_text()
            if len(data) > 65536:
                await websocket.close(code=status.WS_1009_MESSAGE_TOO_BIG, reason="Message too large")
                break
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, session_id)
    except Exception as e:
        logger.warning(f"WebSocket exception on session '{session_id}': {e}")
        ws_manager.disconnect(websocket, session_id)

# Mount REST API Routers
api_v1 = settings.API_V1_PREFIX
app.include_router(health.router)
app.include_router(auth.router, prefix=api_v1)
app.include_router(users.router, prefix=api_v1)
app.include_router(scenarios.router, prefix=api_v1)
app.include_router(evacuation.router, prefix=api_v1)
app.include_router(routes.router, prefix=api_v1)
app.include_router(hazards.router, prefix=api_v1)
app.include_router(traffic.router, prefix=api_v1)
app.include_router(models.router, prefix=api_v1)
app.include_router(history.router, prefix=api_v1)
app.include_router(research.router, prefix=api_v1)
app.include_router(notifications.router, prefix=api_v1)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=False)
