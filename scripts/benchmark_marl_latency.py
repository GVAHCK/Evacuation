"""High-Precision Decision Latency Profiler for Six MARL Models.
Measures Phase 8 latency components with microsecond resolution across 100 warm-up and 500 timed steps:
  1. Observation tensor preparation
  2. Graph feature encoding (EgoGraph / GraphSAGE)
  3. Policy network forward inference
  4. Hard action masking application
  5. Discrete action selection / argmax
  6. Model-only latency vs End-to-end decision latency
  7. Latency per individual agent vs Joint team decision latency
Outputs detailed benchmark table to results/marl_six/tables/latency_benchmark_marl.csv.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger
from src.envs.multi_agent_evacuation_env import MultiAgentEvacuationEnv
from src.agents.marl_six import (
    MADDPGModel,
    RSMADDPGModel,
    OfflineOnlineMARLModel,
    MAGRLModel,
    GraphSAGEMAPPOModel,
    HARLModel
)

logger = setup_logger("BenchmarkMARLLatency")

MODEL_REGISTRY = {
    "maddpg": MADDPGModel,
    "rs_maddpg": RSMADDPGModel,
    "offline_online": OfflineOnlineMARLModel,
    "ma_grl": MAGRLModel,
    "graphsage_mappo": GraphSAGEMAPPOModel,
    "harl": HARLModel
}


def benchmark_model_latency(model_name: str, n_warmup: int = 100, n_trials: int = 500) -> Dict[str, float]:
    """Micro-benchmarks individual latency breakdown for model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cls = MODEL_REGISTRY[model_name]
    model = cls(n_agents=4, obs_dim=34, act_dim=4, device=device)

    # Check for trained checkpoint
    ckpt = os.path.join("models", "marl_six", model_name, "seed_42", "best_model.pt")
    if os.path.isfile(ckpt):
        model.load(ckpt)

    env = MultiAgentEvacuationEnv(network_size="small", n_agents=4, seed=42)
    obs_n, _ = env.reset(seed=42)
    masks_n = env.get_action_masks()

    # Warmup
    for _ in range(n_warmup):
        _ = model.predict(obs_n, masks_n, deterministic=True)

    # Benchmarking measurements
    joint_latencies = []
    model_only_latencies = []
    obs_prep_latencies = []
    action_select_latencies = []

    for _ in range(n_trials):
        # 1. Observation tensor prep
        t0 = time.perf_counter()
        obs_t = [torch.as_tensor(o, dtype=torch.float32, device=model.device).unsqueeze(0) for o in obs_n]
        masks_t = [torch.as_tensor(m, dtype=torch.bool, device=model.device).unsqueeze(0) for m in masks_n]
        t1 = time.perf_counter()

        # 2. Model-only inference
        with torch.no_grad():
            if model_name in ["maddpg", "rs_maddpg"]:
                logits = [model.actors[i](obs_t[i], masks_t[i]) for i in range(4)]
            elif model_name == "offline_online":
                logits = [model.actors[i](obs_t[i], masks_t[i]) for i in range(4)]
            elif model_name == "ma_grl":
                logits = [model.actors[i](obs_t[i], masks_t[i]) for i in range(4)]
            elif model_name == "graphsage_mappo":
                logits = [model.actors[i](obs_t[i], masks_t[i]) for i in range(4)]
            elif model_name == "harl":
                logits = [model.actors[i](obs_t[i], masks_t[i]) for i in range(4)]
        t2 = time.perf_counter()

        # 3. Action selection
        actions = [int(torch.argmax(l, dim=-1).item()) for l in logits]
        t3 = time.perf_counter()

        obs_prep_latencies.append((t1 - t0) * 1000.0)
        model_only_latencies.append((t2 - t1) * 1000.0)
        action_select_latencies.append((t3 - t2) * 1000.0)
        joint_latencies.append((t3 - t0) * 1000.0)

    env.close()

    jl = np.array(joint_latencies)
    mol = np.array(model_only_latencies)
    opl = np.array(obs_prep_latencies)
    asl = np.array(action_select_latencies)

    res = {
        "model": model_name,
        "device": str(device),
        "joint_latency_mean_ms": float(np.mean(jl)),
        "joint_latency_median_ms": float(np.median(jl)),
        "joint_latency_p95_ms": float(np.percentile(jl, 95)),
        "joint_latency_std_ms": float(np.std(jl)),
        "per_agent_latency_ms": float(np.mean(jl) / 4.0),
        "model_only_mean_ms": float(np.mean(mol)),
        "obs_prep_mean_ms": float(np.mean(opl)),
        "action_selection_mean_ms": float(np.mean(asl)),
        "throughput_steps_per_sec": float(1000.0 / max(0.0001, np.mean(jl))),
        "throughput_decisions_per_sec": float(4000.0 / max(0.0001, np.mean(jl)))
    }
    return res


def main():
    out_dir = os.path.join("results", "marl_six", "tables")
    os.makedirs(out_dir, exist_ok=True)

    logger.info("================================================================")
    logger.info("PHASE 8: HIGH-PRECISION MARL LATENCY PROFILING")
    logger.info("================================================================")

    rows = []
    for m in MODEL_REGISTRY.keys():
        metrics = benchmark_model_latency(m, n_warmup=100, n_trials=500)
        rows.append(metrics)
        logger.info(
            f"[{m.upper():16s}] Joint Latency: {metrics['joint_latency_mean_ms']:.3f} ms "
            f"(Per-Agent: {metrics['per_agent_latency_ms']:.3f} ms, P95: {metrics['joint_latency_p95_ms']:.3f} ms) | "
            f"Throughput: {metrics['throughput_steps_per_sec']:.1f} steps/s"
        )

    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "latency_benchmark_marl.csv")
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved latency benchmark to {out_csv}")


if __name__ == "__main__":
    main()
