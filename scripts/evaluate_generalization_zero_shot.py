"""Zero-Shot Cross-Region Generalization Evaluation Suite:
Evaluates existing trained PPO & DQN models (trained ONLY on 336-node multi-district network)
on two completely unseen road topologies:
1. George Town (Dense Urban Grid, 366 nodes, 994 edges) - 250 test scenarios
2. OMR Suburban (Sparse Linear Corridor, 336 nodes, 870 edges) - 250 test scenarios
Also benchmarks Dynamic Dijkstra, Dynamic A*, and Random Valid Routing on each region.
Computes full metrics, classification agreement vs Dijkstra, and the Generalization Gap.
"""

import os
import sys
import json
import time
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
from src.metrics.evaluator import bootstrap_ci

logger = setup_logger("GeneralizationEvaluator")

SEEDS = [42, 101, 2024, 777, 999]

EVAL_REGIONS = {
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

# In-distribution baseline reference metrics from the 336-node multi-district evaluation
IN_DIST_BASELINE = {
    "PPO": {
        "success_rate": 96.4,
        "evacuation_time": 133.13,
        "hei": 10.91,
        "return": 26.95,
        "accuracy": 94.70,
        "f1": 0.9287
    },
    "DQN": {
        "success_rate": 91.8,
        "evacuation_time": 161.97,
        "hei": 16.07,
        "return": -17.65,
        "accuracy": 88.35,
        "f1": 0.8621
    },
    "Dijkstra": {
        "success_rate": 97.0,
        "evacuation_time": 137.27,
        "hei": 7.68,
        "return": 43.98
    }
}


def evaluate_single_run(
    model_or_baseline,
    model_name: str,
    seed_tag: str,
    config_path: str,
    test_scenarios: List[Dict[str, Any]],
    is_ppo: bool = False,
    is_dqn: bool = False,
    dijkstra_agent: Any = None
) -> Tuple[List[Dict[str, Any]], Dict[str, float], List[int], List[int]]:
    """Evaluates one model instance on a given region's test scenarios."""
    env = EvacuationEnv(config_path=config_path, scenarios=test_scenarios, use_traci=False, curriculum_stage=4)

    records = []
    successes = []
    times = []
    heis = []
    dists = []
    timeouts = []
    returns = []
    inf_times = []
    y_true_all = []
    y_pred_all = []

    for scen in test_scenarios:
        obs, info = env.reset(options={"scenario": scen})
        done = False
        ep_ret = 0.0

        while not done:
            t0 = time.perf_counter()
            mask = env.action_masks()

            if is_ppo:
                action, _ = model_or_baseline.predict(obs, action_masks=mask, deterministic=True)
                action = int(action)
            elif is_dqn:
                obs_t = torch.as_tensor(obs, device=model_or_baseline.device).unsqueeze(0)
                with torch.no_grad():
                    q = model_or_baseline.q_net(obs_t).squeeze(0).cpu().numpy()
                q[~mask] = -1e9
                action = int(np.argmax(q))
            else:
                action = model_or_baseline.select_action(env)

            dt_ms = (time.perf_counter() - t0) * 1000.0
            inf_times.append(dt_ms)

            # Ground truth action from Dijkstra on this specific region
            if dijkstra_agent is not None:
                opt_act = dijkstra_agent.select_action(env)
                y_true_all.append(int(opt_act))
                y_pred_all.append(int(action))

            obs, r, term, trunc, step_info = env.step(action)
            ep_ret += r
            done = term or trunc

        succ = step_info.get("success", False)
        tt = step_info.get("cumulative_travel_time", 0.0)
        hei = step_info.get("hazard_exposure_index", 0.0)
        dist = step_info.get("cumulative_distance", 0.0)
        to = step_info.get("timeout", False)

        successes.append(succ)
        times.append(tt)
        heis.append(hei)
        dists.append(dist)
        timeouts.append(to)
        returns.append(ep_ret)

        records.append({
            "model": model_name,
            "seed": str(seed_tag),
            "scenario_id": scen["scenario_id"],
            "region": scen.get("region", "unknown"),
            "hazard_type": scen["hazard_type"],
            "traffic_level": scen["traffic_level"],
            "success": succ,
            "evacuation_time": tt,
            "hei": hei,
            "distance": dist,
            "timeout": to,
            "return": ep_ret
        })

    env.close()

    summary = {
        "success_rate": float(np.mean(successes) * 100.0),
        "mean_evac_time": float(np.mean(times)),
        "std_evac_time": float(np.std(times)),
        "mean_hei": float(np.mean(heis)),
        "std_hei": float(np.std(heis)),
        "mean_dist": float(np.mean(dists)),
        "timeout_rate": float(np.mean(timeouts) * 100.0),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_inf_time_ms": float(np.mean(inf_times)) if inf_times else 0.0
    }

    return records, summary, y_true_all, y_pred_all


def main():
    logger.info("=== STARTING PHASE C: ZERO-SHOT CROSS-REGION GENERALIZATION EVALUATION ===")

    # Load trained models from the 336-node region
    ppo_models = {}
    dqn_models = {}
    for s in SEEDS:
        p_path = f"models/large/ppo/seed_{s}/best_model.zip"
        d_path = f"models/large/dqn/seed_{s}/best_model.zip"
        if os.path.exists(p_path):
            ppo_models[s] = MaskablePPO.load(p_path)
        if os.path.exists(d_path):
            dqn_models[s] = DQN.load(d_path)

    logger.info(f"Loaded {len(ppo_models)} PPO checkpoints and {len(dqn_models)} DQN checkpoints.")

    all_evaluation_records = []
    region_benchmark_rows = []
    classification_rows = []
    gap_rows = []

    for reg_key, reg_info in EVAL_REGIONS.items():
        reg_name = reg_info["name"]
        cfg_path = reg_info["config_path"]
        scens_path = reg_info["scenarios_path"]

        logger.info(f"\n==========================================================================")
        logger.info(f"EVALUATING REGION: {reg_name} ({reg_key})")
        logger.info(f"==========================================================================")

        with open(scens_path, "r", encoding="utf-8") as f:
            scenarios = json.load(f)

        dijkstra_agent = DijkstraBaseline()
        astar_agent = AStarBaseline()
        random_agent = RandomValidRoutingBaseline()

        # 1. Evaluate Dynamic Dijkstra
        logger.info(f"[{reg_key}] Evaluating Dynamic Dijkstra...")
        dijk_recs, dijk_summ, _, _ = evaluate_single_run(
            dijkstra_agent, "Dynamic Dijkstra", "fixed", cfg_path, scenarios
        )
        all_evaluation_records.extend(dijk_recs)
        dijk_ci_time = bootstrap_ci([r["evacuation_time"] for r in dijk_recs])
        dijk_ci_hei = bootstrap_ci([r["hei"] for r in dijk_recs])

        # 2. Evaluate Dynamic A*
        logger.info(f"[{reg_key}] Evaluating Dynamic A*...")
        astar_recs, astar_summ, _, _ = evaluate_single_run(
            astar_agent, "Dynamic A*", "fixed", cfg_path, scenarios
        )
        all_evaluation_records.extend(astar_recs)

        # 3. Evaluate Random Valid
        logger.info(f"[{reg_key}] Evaluating Random Valid Routing...")
        rand_recs, rand_summ, _, _ = evaluate_single_run(
            random_agent, "Random Valid", "fixed", cfg_path, scenarios
        )
        all_evaluation_records.extend(rand_recs)

        # 4. Evaluate Zero-Shot PPO across 5 seeds
        ppo_summaries = []
        ppo_times_all_seeds = []
        ppo_y_true = []
        ppo_y_pred = []

        for s in SEEDS:
            logger.info(f"[{reg_key}] Evaluating Zero-Shot PPO Seed {s}...")
            p_recs, p_summ, yt, yp = evaluate_single_run(
                ppo_models[s], "MaskablePPO (Zero-Shot)", str(s), cfg_path, scenarios,
                is_ppo=True, dijkstra_agent=(dijkstra_agent if s == 42 else None)
            )
            all_evaluation_records.extend(p_recs)
            ppo_summaries.append(p_summ)
            ppo_times_all_seeds.extend([r["evacuation_time"] for r in p_recs])
            if s == 42:
                ppo_y_true = yt
                ppo_y_pred = yp

        p_succ_m = float(np.mean([s["success_rate"] for s in ppo_summaries]))
        p_succ_s = float(np.std([s["success_rate"] for s in ppo_summaries]))
        p_time_m = float(np.mean([s["mean_evac_time"] for s in ppo_summaries]))
        p_time_s = float(np.std([s["mean_evac_time"] for s in ppo_summaries]))
        p_hei_m = float(np.mean([s["mean_hei"] for s in ppo_summaries]))
        p_hei_s = float(np.std([s["mean_hei"] for s in ppo_summaries]))
        p_dist_m = float(np.mean([s["mean_dist"] for s in ppo_summaries]))
        p_to_m = float(np.mean([s["timeout_rate"] for s in ppo_summaries]))
        p_ret_m = float(np.mean([s["mean_return"] for s in ppo_summaries]))
        p_ret_s = float(np.std([s["mean_return"] for s in ppo_summaries]))
        p_inf_m = float(np.mean([s["mean_inf_time_ms"] for s in ppo_summaries]))

        ppo_ci_time = bootstrap_ci([r["evacuation_time"] for r in dijk_recs])  # sample CI
        ppo_ci_hei = bootstrap_ci([r["hei"] for r in dijk_recs])

        # 5. Evaluate Zero-Shot DQN across 5 seeds
        dqn_summaries = []
        for s in SEEDS:
            logger.info(f"[{reg_key}] Evaluating Zero-Shot DQN Seed {s}...")
            d_recs, d_summ, _, _ = evaluate_single_run(
                dqn_models[s], "Masked DQN (Zero-Shot)", str(s), cfg_path, scenarios,
                is_dqn=True
            )
            all_evaluation_records.extend(d_recs)
            dqn_summaries.append(d_summ)

        d_succ_m = float(np.mean([s["success_rate"] for s in dqn_summaries]))
        d_succ_s = float(np.std([s["success_rate"] for s in dqn_summaries]))
        d_time_m = float(np.mean([s["mean_evac_time"] for s in dqn_summaries]))
        d_time_s = float(np.std([s["mean_evac_time"] for s in dqn_summaries]))
        d_hei_m = float(np.mean([s["mean_hei"] for s in dqn_summaries]))
        d_hei_s = float(np.std([s["mean_hei"] for s in dqn_summaries]))
        d_dist_m = float(np.mean([s["mean_dist"] for s in dqn_summaries]))
        d_to_m = float(np.mean([s["timeout_rate"] for s in dqn_summaries]))
        d_ret_m = float(np.mean([s["mean_return"] for s in dqn_summaries]))
        d_ret_s = float(np.std([s["mean_return"] for s in dqn_summaries]))
        d_inf_m = float(np.mean([s["mean_inf_time_ms"] for s in dqn_summaries]))

        # Per-step routing classification metrics for PPO
        acc = accuracy_score(ppo_y_true, ppo_y_pred)
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(ppo_y_true, ppo_y_pred, average="weighted", zero_division=0)
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(ppo_y_true, ppo_y_pred, average="macro", zero_division=0)

        classification_rows.append({
            "Region": reg_name,
            "Region_Key": reg_key,
            "Total_Steps": len(ppo_y_true),
            "Accuracy": round(acc * 100.0, 2),
            "Weighted_Precision": round(p_w, 4),
            "Weighted_Recall": round(r_w, 4),
            "Weighted_F1": round(f1_w, 4),
            "Macro_F1": round(f1_m, 4)
        })

        # Compile benchmark table rows
        region_benchmark_rows.extend([
            {
                "Region": reg_name,
                "Model": "MaskablePPO (Zero-Shot)",
                "Success_Rate": f"{p_succ_m:.1f} ± {p_succ_s:.2f}%",
                "Evacuation_Time_s": f"{p_time_m:.2f} ± {p_time_s:.2f}",
                "HEI": f"{p_hei_m:.2f} ± {p_hei_s:.2f}",
                "Distance_m": f"{p_dist_m:.1f}",
                "Timeout_Rate": f"{p_to_m:.1f}%",
                "Return": f"{p_ret_m:.2f} ± {p_ret_s:.2f}",
                "Inference_ms": f"{p_inf_m:.3f}"
            },
            {
                "Region": reg_name,
                "Model": "Masked DQN (Zero-Shot)",
                "Success_Rate": f"{d_succ_m:.1f} ± {d_succ_s:.2f}%",
                "Evacuation_Time_s": f"{d_time_m:.2f} ± {d_time_s:.2f}",
                "HEI": f"{d_hei_m:.2f} ± {d_hei_s:.2f}",
                "Distance_m": f"{d_dist_m:.1f}",
                "Timeout_Rate": f"{d_to_m:.1f}%",
                "Return": f"{d_ret_m:.2f} ± {d_ret_s:.2f}",
                "Inference_ms": f"{d_inf_m:.3f}"
            },
            {
                "Region": reg_name,
                "Model": "Dynamic Dijkstra (Region Ground Truth)",
                "Success_Rate": f"{dijk_summ['success_rate']:.1f}%",
                "Evacuation_Time_s": f"{dijk_summ['mean_evac_time']:.2f}",
                "HEI": f"{dijk_summ['mean_hei']:.2f}",
                "Distance_m": f"{dijk_summ['mean_dist']:.1f}",
                "Timeout_Rate": f"{dijk_summ['timeout_rate']:.1f}%",
                "Return": f"{dijk_summ['mean_return']:.2f}",
                "Inference_ms": f"{dijk_summ['mean_inf_time_ms']:.3f}"
            },
            {
                "Region": reg_name,
                "Model": "Dynamic A*",
                "Success_Rate": f"{astar_summ['success_rate']:.1f}%",
                "Evacuation_Time_s": f"{astar_summ['mean_evac_time']:.2f}",
                "HEI": f"{astar_summ['mean_hei']:.2f}",
                "Distance_m": f"{astar_summ['mean_dist']:.1f}",
                "Timeout_Rate": f"{astar_summ['timeout_rate']:.1f}%",
                "Return": f"{astar_summ['mean_return']:.2f}",
                "Inference_ms": f"{astar_summ['mean_inf_time_ms']:.3f}"
            },
            {
                "Region": reg_name,
                "Model": "Random Valid Routing",
                "Success_Rate": f"{rand_summ['success_rate']:.1f}%",
                "Evacuation_Time_s": f"{rand_summ['mean_evac_time']:.2f}",
                "HEI": f"{rand_summ['mean_hei']:.2f}",
                "Distance_m": f"{rand_summ['mean_dist']:.1f}",
                "Timeout_Rate": f"{rand_summ['timeout_rate']:.1f}%",
                "Return": f"{rand_summ['mean_return']:.2f}",
                "Inference_ms": f"{rand_summ['mean_inf_time_ms']:.3f}"
            }
        ])

        # Compute generalization gap vs In-Distribution
        ppo_in = IN_DIST_BASELINE["PPO"]
        gap_rows.append({
            "Region": reg_name,
            "Model": "MaskablePPO",
            "In_Dist_Success": f"{ppo_in['success_rate']:.1f}%",
            "Zero_Shot_Success": f"{p_succ_m:.1f}%",
            "Success_Gap": f"{p_succ_m - ppo_in['success_rate']:+.1f}%",
            "In_Dist_Time_s": f"{ppo_in['evacuation_time']:.2f}",
            "Zero_Shot_Time_s": f"{p_time_m:.2f}",
            "Time_Delta_s": f"{p_time_m - ppo_in['evacuation_time']:+.2f}s",
            "In_Dist_HEI": f"{ppo_in['hei']:.2f}",
            "Zero_Shot_HEI": f"{p_hei_m:.2f}",
            "HEI_Delta": f"{p_hei_m - ppo_in['hei']:+.2f}",
            "In_Dist_Return": f"{ppo_in['return']:.2f}",
            "Zero_Shot_Return": f"{p_ret_m:.2f}",
            "Return_Delta": f"{p_ret_m - ppo_in['return']:+.2f}",
            "Routing_Agreement_vs_Dijkstra": f"{acc * 100.0:.2f}% (F1: {f1_w:.4f})"
        })

    # Save all tables
    os.makedirs("results/tables", exist_ok=True)

    pd.DataFrame(all_evaluation_records).to_csv("results/tables/raw_generalization_records.csv", index=False)
    df_bench = pd.DataFrame(region_benchmark_rows)
    df_bench.to_csv("results/tables/generalization_benchmark_comparison.csv", index=False)
    df_gap = pd.DataFrame(gap_rows)
    df_gap.to_csv("results/tables/generalization_gap_analysis.csv", index=False)
    df_clf = pd.DataFrame(classification_rows)
    df_clf.to_csv("results/tables/classification_metrics_generalization.csv", index=False)

    print("\n" + "=" * 110)
    print("CROSS-REGION ZERO-SHOT BENCHMARK COMPARISON")
    print("=" * 110)
    print(df_bench.to_string(index=False))

    print("\n" + "=" * 110)
    print("GENERALIZATION GAP ANALYSIS (IN-DISTRIBUTION VS ZERO-SHOT TRANSFER)")
    print("=" * 110)
    print(df_gap.to_string(index=False))

    print("\n" + "=" * 110)
    print("PER-STEP ROUTING ACCURACY VS DIJKSTRA ON UNSEEN TOPOLOGIES")
    print("=" * 110)
    print(df_clf.to_string(index=False))


if __name__ == "__main__":
    main()
