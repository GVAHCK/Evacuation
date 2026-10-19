"""Zero-Shot Generalization Benchmark for Hybrid RL Models:
Evaluates hybrid models on two completely unseen road topologies:
  1. George Town (Dense Urban Grid, 366 nodes, 994 edges) - 250 test scenarios
  2. OMR Suburban (Sparse Linear Corridor, 336 nodes, 870 edges) - 250 test scenarios
Calculates Generalization Gap for Success, Evacuation Time, and HEI vs in-distribution.
"""

import os
import sys
import json
import time
import argparse
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN

from src.utils.logger import setup_logger
from src.environment.evacuation_env import EvacuationEnv
from src.baselines.dijkstra import DijkstraBaseline
from src.baselines.astar import AStarBaseline
from src.baselines.random_routing import RandomValidRoutingBaseline
from src.agents.gnn_ppo_agent import load_gnn_ppo_agent
from src.agents.recurrent_ppo_agent import load_recurrent_ppo_agent
from src.agents.gnn_qrdqn_agent import MaskedGNNQRDQNAgent
from src.agents.safety_shield_agent import SafetyShieldPPOAgent
from src.agents.tri_hybrid_agent import TriHybridEvacuationAgent
from scripts.evaluate_hybrid_models import evaluate_single_agent

logger = setup_logger("ZeroShotEvaluator")

REGIONS = {
    "dense_grid": {
        "name": "George Town (Dense Urban Grid)",
        "config_path": "configs/dense_grid.yaml",
        "scenarios_path": "data/generated/test_dense_grid_scenarios.json"
    },
    "sparse_suburban": {
        "name": "OMR Suburban (Sparse Linear Corridor)",
        "config_path": "configs/sparse_suburban.yaml",
        "scenarios_path": "data/generated/test_sparse_suburban_scenarios.json"
    }
}


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Cross-Region Evaluation")
    parser.add_argument("--mode", type=str, default="full", choices=["quick", "full"],
                        help="Quick (20 scenarios) or full (250 scenarios) evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Seed model to evaluate")
    args = parser.parse_args()

    all_records = []
    summaries = []

    for region_key, reg_info in REGIONS.items():
        reg_name = reg_info["name"]
        cfg_path = reg_info["config_path"]
        scen_path = reg_info["scenarios_path"]

        with open(scen_path, "r", encoding="utf-8") as f:
            scenarios = json.load(f)

        if args.mode == "quick":
            scenarios = scenarios[:20]

        logger.info(f"=== Evaluating Region: {reg_name} ({len(scenarios)} scenarios) ===")
        dijkstra = DijkstraBaseline()

        # Classical Baselines
        baselines = [
            ("Dynamic Dijkstra", dijkstra, "dijkstra"),
            ("Dynamic A*", AStarBaseline(), "astar"),
            ("Random Valid", RandomValidRoutingBaseline(seed=42), "random")
        ]
        for b_name, b_agent, b_type in baselines:
            recs, summ = evaluate_single_agent(
                agent=b_agent,
                agent_name=f"{b_name} (Zero-Shot)",
                agent_type=b_type,
                config_path=cfg_path,
                scenarios=scenarios,
                seed=42,
                dijkstra_baseline=dijkstra
            )
            summ["region"] = reg_name
            all_records.extend(recs)
            summaries.append(summ)

        # Baseline PPO & DQN (Large trained)
        ppo_path = "models/large/ppo/seed_42/best_model.zip"
        dqn_path = "models/large/dqn/seed_42/best_model.zip"
        if os.path.isfile(ppo_path):
            base_ppo = MaskablePPO.load(ppo_path)
            recs, summ = evaluate_single_agent(base_ppo, "Baseline MaskablePPO (Zero-Shot)", "baseline_ppo", cfg_path, scenarios, 42, dijkstra)
            summ["region"] = reg_name
            all_records.extend(recs)
            summaries.append(summ)

        if os.path.isfile(dqn_path):
            base_dqn = DQN.load(dqn_path)
            recs, summ = evaluate_single_agent(base_dqn, "Baseline Masked DQN (Zero-Shot)", "baseline_dqn", cfg_path, scenarios, 42, dijkstra)
            summ["region"] = reg_name
            all_records.extend(recs)
            summaries.append(summ)

        # Hybrid Models
        seed = args.seed
        gnn_ppo_path = f"models/hybrid/gnn_ppo/seed_{seed}/best_model.zip"
        loaded_gnn_ppo = None
        if os.path.isfile(gnn_ppo_path):
            loaded_gnn_ppo = load_gnn_ppo_agent(gnn_ppo_path)
            recs, summ = evaluate_single_agent(loaded_gnn_ppo, "GNN-MaskablePPO (Zero-Shot)", "gnn_ppo", cfg_path, scenarios, seed, dijkstra)
            summ["region"] = reg_name
            all_records.extend(recs)
            summaries.append(summ)

        lstm_path = f"models/hybrid/lstm_ppo/seed_{seed}/best_model.pt"
        if os.path.isfile(lstm_path):
            lstm_agent = load_recurrent_ppo_agent(lstm_path)
            recs, summ = evaluate_single_agent(lstm_agent, "RecurrentMaskablePPO (Zero-Shot)", "lstm_ppo", cfg_path, scenarios, seed, dijkstra)
            summ["region"] = reg_name
            all_records.extend(recs)
            summaries.append(summ)

        qrdqn_path = f"models/hybrid/gnn_qrdqn/seed_{seed}/best_model.zip"
        loaded_qrdqn = None
        if os.path.isfile(qrdqn_path):
            loaded_qrdqn = MaskedGNNQRDQNAgent(qrdqn_path)
            recs, summ = evaluate_single_agent(loaded_qrdqn, "GNN-QRDQN (Zero-Shot)", "gnn_qrdqn", cfg_path, scenarios, seed, dijkstra)
            summ["region"] = reg_name
            all_records.extend(recs)
            summaries.append(summ)

        if loaded_gnn_ppo is not None:
            shield = SafetyShieldPPOAgent(base_ppo_agent=loaded_gnn_ppo, alpha=0.50, beta=0.35, gamma=0.15)
            recs, summ = evaluate_single_agent(shield, "SafetyShieldPPO (Zero-Shot)", "safety_shield", cfg_path, scenarios, seed, dijkstra)
            summ["region"] = reg_name
            all_records.extend(recs)
            summaries.append(summ)

        if loaded_gnn_ppo is not None and loaded_qrdqn is not None:
            tri_agent = TriHybridEvacuationAgent(ppo_agent=loaded_gnn_ppo, qrdqn_agent=loaded_qrdqn, dijkstra_baseline=dijkstra)
            recs, summ = evaluate_single_agent(tri_agent, "TriHybridAgent (Zero-Shot)", "tri_hybrid", cfg_path, scenarios, seed, dijkstra)
            summ["region"] = reg_name
            all_records.extend(recs)
            summaries.append(summ)

    out_dir = "results/tables"
    os.makedirs(out_dir, exist_ok=True)
    raw_df = pd.DataFrame(all_records)
    summ_df = pd.DataFrame(summaries)

    raw_path = os.path.join(out_dir, "raw_hybrid_zero_shot_records.csv")
    summ_path = os.path.join(out_dir, "hybrid_zero_shot_comparison.csv")

    raw_df.to_csv(raw_path, index=False)
    summ_df.to_csv(summ_path, index=False)

    print("\n" + "=" * 100)
    print("ZERO-SHOT CROSS-REGION EVALUATION SUMMARY")
    print(summ_df[["region", "model", "success_rate", "evacuation_time_mean", "hei_mean", "distance_mean", "timeout_rate"]].to_string(index=False))
    print("=" * 100)
    print(f"Saved: {summ_path}")


if __name__ == "__main__":
    main()
