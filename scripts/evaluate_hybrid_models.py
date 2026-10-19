"""Comprehensive Hybrid Models Evaluation Suite:
Benchmarks all five hybrid models + historical baselines on unseen test scenarios:
  1. Small 84-node network (100 unseen test scenarios)
  2. Large 336-node network (500 unseen test scenarios)
Measures Success Rate, Mean/Median/P95 Evacuation Time, HEI, Travel Distance,
Timeout Rate, Return, Routing Accuracy, Weighted F1, and Inference Latency.
Strictly asserts action_mask[action] == True on every decision step.
"""

import os
import sys
import json
import time
import argparse
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
from src.metrics.evaluator import bootstrap_ci

logger = setup_logger("EvaluateHybridModels")

SEEDS = [42, 101, 2024, 777, 999]


def evaluate_single_agent(
    agent: Any,
    agent_name: str,
    agent_type: str,
    config_path: str,
    scenarios: List[Dict[str, Any]],
    seed: int,
    dijkstra_baseline: DijkstraBaseline
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluates an agent across a scenario set, collecting per-episode and per-step metrics."""
    env = EvacuationEnv(config_path=config_path, scenarios=scenarios, use_traci=False, curriculum_stage=4)

    records = []
    successes = []
    times = []
    heis = []
    dists = []
    timeouts = []
    returns = []
    latencies = []
    y_true_all = []
    y_pred_all = []

    for scen in scenarios:
        obs, info = env.reset(options={"scenario": scen})
        done = False
        ep_ret = 0.0

        # Reset recurrent state if supported
        if hasattr(agent, "reset_state"):
            agent.reset_state(batch_size=1)

        while not done:
            mask = env.action_masks()
            t0 = time.perf_counter()

            # Predict action according to agent interface
            if agent_type == "gnn_ppo" or agent_type == "baseline_ppo":
                action, _ = agent.predict(obs, action_masks=mask, deterministic=True)
                action = int(action)
            elif agent_type == "lstm_ppo":
                action, _ = agent.predict(obs, action_masks=mask, deterministic=True)
                action = int(action)
            elif agent_type == "gnn_qrdqn":
                action, _ = agent.predict(obs, action_masks=mask, deterministic=True)
                action = int(action)
            elif agent_type == "baseline_dqn":
                obs_t = torch.as_tensor(obs, device=agent.device).unsqueeze(0)
                with torch.no_grad():
                    q = agent.q_net(obs_t).squeeze(0).cpu().numpy()
                q[~mask] = -1e9
                action = int(np.argmax(q))
            elif agent_type == "safety_shield" or agent_type == "tri_hybrid":
                action = agent.select_action(env)
            else:
                # Dijkstra, A*, Random
                action = agent.select_action(env)

            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt_ms)

            # Strict Assertion: Action must be permitted by action_masks()
            assert mask[action], f"CRITICAL: Agent {agent_name} picked invalid masked action {action}"

            # Ground truth action from Dijkstra for routing accuracy
            opt_act = dijkstra_baseline.select_action(env)
            y_true_all.append(int(opt_act))
            y_pred_all.append(int(action))

            obs, r, term, trunc, step_info = env.step(action)
            ep_ret += r
            done = term or trunc

        succ = bool(step_info.get("success", False))
        tt = float(step_info.get("cumulative_travel_time", 0.0))
        hei = float(step_info.get("hazard_exposure_index", 0.0))
        dist = float(step_info.get("cumulative_distance", 0.0))
        to = bool(step_info.get("timeout", False))

        successes.append(succ)
        times.append(tt)
        heis.append(hei)
        dists.append(dist)
        timeouts.append(to)
        returns.append(ep_ret)

        records.append({
            "scenario_id": scen.get("scenario_id", ""),
            "model": agent_name,
            "seed": seed,
            "success": succ,
            "travel_time": tt,
            "hei": hei,
            "distance": dist,
            "timeout": to,
            "return": ep_ret
        })

    env.close()

    # Classification & Agreement vs Dijkstra
    acc = accuracy_score(y_true_all, y_pred_all) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average="weighted", zero_division=0)

    summary = {
        "model": agent_name,
        "seed": seed,
        "success_rate": float(np.mean(successes)) * 100.0,
        "evacuation_time_mean": float(np.mean(times)),
        "evacuation_time_median": float(np.median(times)),
        "evacuation_time_p95": float(np.percentile(times, 95)),
        "hei_mean": float(np.mean(heis)),
        "distance_mean": float(np.mean(dists)),
        "timeout_rate": float(np.mean(timeouts)) * 100.0,
        "return_mean": float(np.mean(returns)),
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "routing_accuracy": float(acc),
        "weighted_f1": float(f1)
    }

    return records, summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid RL Models")
    parser.add_argument("--network", type=str, default="small", choices=["small", "large"],
                        help="Network scale to evaluate on (small=84-node, large=336-node)")
    parser.add_argument("--mode", type=str, default="full", choices=["quick", "full"],
                        help="Quick test (20 scenarios) or full evaluation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Seeds to evaluate across")
    args = parser.parse_args()

    if args.network == "small":
        config_path = "configs/default.yaml"
        scen_path = "data/generated/test_scenarios.json"
    else:
        config_path = "configs/large.yaml"
        scen_path = "data/generated/test_large_scenarios.json"

    with open(scen_path, "r", encoding="utf-8") as f:
        test_scenarios = json.load(f)

    if args.mode == "quick":
        test_scenarios = test_scenarios[:20]

    logger.info(f"Loaded {len(test_scenarios)} test scenarios from {scen_path} for {args.network} network.")

    dijkstra = DijkstraBaseline()
    all_records = []
    summaries = []

    # 1. Classical Baselines
    baselines = [
        ("Dynamic Dijkstra", dijkstra, "dijkstra"),
        ("Dynamic A*", AStarBaseline(), "astar"),
        ("Random Valid", RandomValidRoutingBaseline(seed=42), "random")
    ]
    for b_name, b_agent, b_type in baselines:
        recs, summ = evaluate_single_agent(
            agent=b_agent,
            agent_name=b_name,
            agent_type=b_type,
            config_path=config_path,
            scenarios=test_scenarios,
            seed=42,
            dijkstra_baseline=dijkstra
        )
        all_records.extend(recs)
        summaries.append(summ)

    # 2. Historical Baselines (PPO & DQN)
    ppo_path = "models/ppo/seed_42/best_model.zip" if args.network == "small" else "models/large/ppo/seed_42/best_model.zip"
    dqn_path = "models/dqn/seed_42/best_model.zip" if args.network == "small" else "models/large/dqn/seed_42/best_model.zip"

    if os.path.isfile(ppo_path):
        base_ppo = MaskablePPO.load(ppo_path)
        recs, summ = evaluate_single_agent(base_ppo, "Baseline MaskablePPO", "baseline_ppo", config_path, test_scenarios, 42, dijkstra)
        all_records.extend(recs)
        summaries.append(summ)

    if os.path.isfile(dqn_path):
        base_dqn = DQN.load(dqn_path)
        recs, summ = evaluate_single_agent(base_dqn, "Baseline Masked DQN", "baseline_dqn", config_path, test_scenarios, 42, dijkstra)
        all_records.extend(recs)
        summaries.append(summ)

    # 3. Hybrid Models
    for seed in args.seeds:
        # GNN-PPO
        gnn_ppo_path = f"models/hybrid/gnn_ppo/seed_{seed}/best_model.zip"
        loaded_gnn_ppo = None
        if os.path.isfile(gnn_ppo_path):
            loaded_gnn_ppo = load_gnn_ppo_agent(gnn_ppo_path)
            recs, summ = evaluate_single_agent(loaded_gnn_ppo, f"GNN-MaskablePPO (seed {seed})", "gnn_ppo", config_path, test_scenarios, seed, dijkstra)
            all_records.extend(recs)
            summaries.append(summ)

        # Recurrent-PPO (LSTM)
        lstm_path = f"models/hybrid/lstm_ppo/seed_{seed}/best_model.pt"
        if os.path.isfile(lstm_path):
            lstm_agent = load_recurrent_ppo_agent(lstm_path)
            recs, summ = evaluate_single_agent(lstm_agent, f"RecurrentMaskablePPO (seed {seed})", "lstm_ppo", config_path, test_scenarios, seed, dijkstra)
            all_records.extend(recs)
            summaries.append(summ)

        # GNN-QRDQN
        qrdqn_path = f"models/hybrid/gnn_qrdqn/seed_{seed}/best_model.zip"
        loaded_qrdqn = None
        if os.path.isfile(qrdqn_path):
            loaded_qrdqn = MaskedGNNQRDQNAgent(qrdqn_path)
            recs, summ = evaluate_single_agent(loaded_qrdqn, f"GNN-QRDQN (seed {seed})", "gnn_qrdqn", config_path, test_scenarios, seed, dijkstra)
            all_records.extend(recs)
            summaries.append(summ)

        # SafetyShieldPPO
        if loaded_gnn_ppo is not None:
            shield_agent = SafetyShieldPPOAgent(base_ppo_agent=loaded_gnn_ppo, alpha=0.50, beta=0.35, gamma=0.15)
            recs, summ = evaluate_single_agent(shield_agent, f"SafetyShieldPPO (seed {seed})", "safety_shield", config_path, test_scenarios, seed, dijkstra)
            all_records.extend(recs)
            summaries.append(summ)

        # TriHybridAgent
        if loaded_gnn_ppo is not None and loaded_qrdqn is not None:
            tri_agent = TriHybridEvacuationAgent(ppo_agent=loaded_gnn_ppo, qrdqn_agent=loaded_qrdqn, dijkstra_baseline=dijkstra)
            recs, summ = evaluate_single_agent(tri_agent, f"TriHybridAgent (seed {seed})", "tri_hybrid", config_path, test_scenarios, seed, dijkstra)
            all_records.extend(recs)
            summaries.append(summ)

    # Save outputs
    out_dir = "results/tables"
    os.makedirs(out_dir, exist_ok=True)
    raw_df = pd.DataFrame(all_records)
    summ_df = pd.DataFrame(summaries)

    raw_path = os.path.join(out_dir, f"raw_hybrid_{args.network}_records.csv")
    summ_path = os.path.join(out_dir, f"hybrid_{args.network}_comparison.csv")

    raw_df.to_csv(raw_path, index=False)
    summ_df.to_csv(summ_path, index=False)

    print("\n" + "=" * 100)
    print(f"EVALUATION COMPLETE ({args.network.upper()} NETWORK, {len(test_scenarios)} SCENARIOS)")
    print(summ_df[["model", "success_rate", "evacuation_time_mean", "hei_mean", "distance_mean", "timeout_rate", "latency_mean_ms", "weighted_f1"]].to_string(index=False))
    print("=" * 100)
    print(f"Summary saved: {summ_path}")
    print(f"Raw records: {raw_path}")


if __name__ == "__main__":
    main()
