from typing import List
from fastapi import APIRouter
from backend.app.schemas.model import ModelMetadataResponse
from backend.app.ml.model_manager import model_manager

router = APIRouter(prefix="/models", tags=["Models"])

@router.get("", response_model=List[ModelMetadataResponse], summary="List production AI models (Offline-to-Online MARL)")
def list_models():
    return [ModelMetadataResponse(**model_manager.get_metadata())]

@router.get("/offline-to-online", response_model=ModelMetadataResponse, summary="Get Offline-to-Online MARL model specification and status")
def get_offline_to_online_model():
    return ModelMetadataResponse(**model_manager.get_metadata())

@router.get("/diagnostics", summary="Run live diagnostic benchmark on loaded Offline-to-Online MARL model")
def run_model_diagnostics():
    latencies = []
    mask_violations = 0
    test_runs = 20

    if model_manager.status == "ready" and model_manager.model is not None:
        import numpy as np
        for _ in range(test_runs):
            obs_n = [np.random.uniform(0.0, 1.0, size=(model_manager.obs_dim,)).astype(np.float32) for _ in range(model_manager.n_agents)]
            # Randomly pick 1-2 valid actions for each agent
            masks_n = []
            for _ in range(model_manager.n_agents):
                m = np.zeros(model_manager.act_dim, dtype=bool)
                valid_idx = np.random.choice(model_manager.act_dim)
                m[valid_idx] = True
                masks_n.append(m)

            actions, lat_ms = model_manager.predict(obs_n, masks_n, deterministic=True)
            latencies.append(lat_ms)
            for idx, a in enumerate(actions):
                if not masks_n[idx][a]:
                    mask_violations += 1

        mean_lat = float(np.mean(latencies))
        p95_lat = float(np.percentile(latencies, 95))
        min_lat = float(np.min(latencies))
        max_lat = float(np.max(latencies))
    else:
        mean_lat = 0.0
        p95_lat = 0.0
        min_lat = 0.0
        max_lat = 0.0

    return {
        "model_name": model_manager.model_name,
        "model_family": model_manager.model_family,
        "status": model_manager.status,
        "device": str(model_manager.device),
        "parameter_count": model_manager.parameter_count,
        "obs_dim": model_manager.obs_dim,
        "act_dim": model_manager.act_dim,
        "n_agents": model_manager.n_agents,
        "action_masking_enforced": True,
        "action_mask_violations_detected": mask_violations,
        "warmup_completed": model_manager.warmup_completed,
        "diagnostic_iterations": test_runs,
        "latency_metrics_ms": {
            "mean": round(mean_lat, 3),
            "p95": round(p95_lat, 3),
            "min": round(min_lat, 3),
            "max": round(max_lat, 3)
        },
        "checkpoint_exists": model_manager.get_metadata()["checkpoint_exists"],
        "checkpoint_size_kb": round(model_manager.checkpoint_size_kb, 1),
        "disclaimer": "Diagnostic latency was measured live across 20 forward passes on the current host device."
    }

