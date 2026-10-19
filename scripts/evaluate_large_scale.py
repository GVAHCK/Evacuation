"""Large-Scale Evaluation Suite:
Evaluates 5 PPO seeds, 5 DQN seeds, Dynamic Dijkstra, Dynamic A*, and Random
across 500 strictly unseen test scenarios on the Multi-District Urban Road Network.
Computes multi-seed statistical metrics (mean +- std), paired Wilcoxon and t-tests,
per-step action classification metrics, and episode outcome metrics.
"""

import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
from scipy import stats
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

logger = setup_logger("LargeScaleEvaluator")

SEEDS = [42, 101, 2024, 777, 999]
CONFIG_PATH = "configs/large.yaml"
TEST_SCENARIOS_PATH = "data/generated/test_large_scenarios.json"


def evaluate_single_run(
    model_or_baseline,
    model_name: str,
    seed_tag: str,
    is_ppo: bool = False,
    is_dqn: bool = False,
    test_scenarios: List[Dict[str, Any]] = None,
    dijkstra_agent: Any = None
) -> Tuple[List[Dict[str, Any]], Dict[str, float], List[int], List[int]]:
    """Evaluates one model instance on the test scenarios, tracking steps and metrics."""
    env = EvacuationEnv(config_path=CONFIG_PATH, scenarios=test_scenarios, use_traci=False, curriculum_stage=4)

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

            # Ground-truth action from Dijkstra for classification comparison
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
            "seed": seed_tag,
            "scenario_id": scen.get("scenario_id", "unknown"),
            "success": succ,
            "evacuation_time": tt,
            "hei": hei,
            "distance": dist,
            "timeout": to,
            "return": ep_ret
        })

    env.close()

    summary = {
        "success_rate": float(np.mean(successes)) * 100.0,
        "mean_evac_time": float(np.mean(times)),
        "median_evac_time": float(np.median(times)),
        "mean_hei": float(np.mean(heis)),
        "mean_dist": float(np.mean(dists)),
        "timeout_rate": float(np.mean(timeouts)) * 100.0,
        "mean_return": float(np.mean(returns)),
        "mean_inf_time_ms": float(np.mean(inf_times))
    }

    return records, summary, y_true_all, y_pred_all


def main():
    logger.info("=" * 80)
    logger.info("PHASE 5: LARGE-SCALE STATISTICAL EVALUATION (500 UNSEEN TEST SCENARIOS)")
    logger.info("=" * 80)

    test_scenarios = json.load(open(TEST_SCENARIOS_PATH))
    all_records = []
    dijkstra_agent = DijkstraBaseline()

    # 1. Evaluate PPO across 5 seeds
    ppo_summaries = []
    ppo_seed_times = {}
    ppo_y_true = []
    ppo_y_pred = []

    for seed in SEEDS:
        ckpt_path = f"models/large/ppo/seed_{seed}/best_model.zip"
        logger.info(f"Evaluating Large MaskablePPO (seed {seed})...")
        model = MaskablePPO.load(ckpt_path)
        recs, summ, yt, yp = evaluate_single_run(
            model, "PPO", str(seed), is_ppo=True, test_scenarios=test_scenarios, dijkstra_agent=dijkstra_agent
        )
        all_records.extend(recs)
        ppo_summaries.append(summ)
        ppo_seed_times[seed] = [r["evacuation_time"] for r in recs]
        if seed == 42:
            ppo_y_true = yt
            ppo_y_pred = yp

    # 2. Evaluate DQN across 5 seeds
    dqn_summaries = []
    dqn_seed_times = {}
    for seed in SEEDS:
        ckpt_path = f"models/large/dqn/seed_{seed}/best_model.zip"
        logger.info(f"Evaluating Large Masked DQN (seed {seed})...")
        model = DQN.load(ckpt_path)
        recs, summ, _, _ = evaluate_single_run(
            model, "DQN", str(seed), is_dqn=True, test_scenarios=test_scenarios
        )
        all_records.extend(recs)
        dqn_summaries.append(summ)
        dqn_seed_times[seed] = [r["evacuation_time"] for r in recs]

    # 3. Evaluate Baselines
    logger.info("Evaluating Large Dynamic Dijkstra...")
    dijkstra_recs, dijkstra_summ, _, _ = evaluate_single_run(
        dijkstra_agent, "Dijkstra", "fixed", test_scenarios=test_scenarios
    )
    all_records.extend(dijkstra_recs)

    logger.info("Evaluating Large Dynamic A*...")
    astar_agent = AStarBaseline()
    astar_recs, astar_summ, _, _ = evaluate_single_run(
        astar_agent, "A*", "fixed", test_scenarios=test_scenarios
    )
    all_records.extend(astar_recs)

    logger.info("Evaluating Large Random Valid...")
    rand_agent = RandomValidRoutingBaseline()
    rand_recs, rand_summ, _, _ = evaluate_single_run(
        rand_agent, "Random", "fixed", test_scenarios=test_scenarios
    )
    all_records.extend(rand_recs)

    # 4. Save raw records
    df_raw = pd.DataFrame(all_records)
    raw_path = "results/tables/raw_large_scale_records.csv"
    df_raw.to_csv(raw_path, index=False)
    logger.info(f"Saved {len(df_raw)} raw evaluation records to {raw_path}")

    # 5. Statistical Hypothesis Tests (PPO seed 42 vs Dijkstra across 500 test scenarios)
    ppo_times_42 = ppo_seed_times[42]
    dijk_times = [r["evacuation_time"] for r in dijkstra_recs]
    dqn_times_42 = dqn_seed_times[42]

    diff_dijk = np.array(ppo_times_42) - np.array(dijk_times)
    ttest_res = stats.ttest_rel(ppo_times_42, dijk_times)
    try:
        wtest_res = stats.wilcoxon(diff_dijk)
        w_stat, w_pval = float(wtest_res.statistic), float(wtest_res.pvalue)
    except Exception:
        w_stat, w_pval = 0.0, 1.0

    stat_results = {
        "sample_size_N": len(ppo_times_42),
        "degrees_of_freedom": len(ppo_times_42) - 1,
        "non_zero_differences": int(np.count_nonzero(diff_dijk)),
        "PPO_vs_Dijkstra_Time": {
            "paired_t_stat": float(ttest_res.statistic),
            "paired_t_pvalue": float(ttest_res.pvalue),
            "wilcoxon_stat": w_stat,
            "wilcoxon_pvalue": w_pval
        }
    }
    with open("results/tables/statistical_tests_large.json", "w", encoding="utf-8") as f:
        json.dump(stat_results, f, indent=2)

    # 6. Aggregate Benchmark Table (Mean +- Std across 5 seeds)
    def agg_metric(summaries, key):
        vals = [s[key] for s in summaries]
        return np.mean(vals), np.std(vals)

    ppo_ci_time = bootstrap_ci(ppo_times_42)
    ppo_ci_hei = bootstrap_ci([r["hei"] for r in all_records if r["model"] == "PPO" and r["seed"] == "42"])
    dijk_ci_time = bootstrap_ci(dijk_times)
    dijk_ci_hei = bootstrap_ci([r["hei"] for r in dijkstra_recs])

    final_rows = []

    # PPO across 5 seeds
    p_succ_m, p_succ_s = agg_metric(ppo_summaries, "success_rate")
    p_time_m, p_time_s = agg_metric(ppo_summaries, "mean_evac_time")
    p_hei_m, p_hei_s = agg_metric(ppo_summaries, "mean_hei")
    p_dist_m, p_dist_s = agg_metric(ppo_summaries, "mean_dist")
    p_to_m, p_to_s = agg_metric(ppo_summaries, "timeout_rate")
    p_ret_m, p_ret_s = agg_metric(ppo_summaries, "mean_return")
    p_inf_m, _ = agg_metric(ppo_summaries, "mean_inf_time_ms")

    final_rows.append({
        "Model": "MaskablePPO (Proposed, 5 seeds)",
        "Success_Rate": f"{p_succ_m:.1f} ± {p_succ_s:.2f}%",
        "Evacuation_Time_s": f"{p_time_m:.2f} ± {p_time_s:.2f}",
        "Evac_Time_95_CI": f"[{ppo_ci_time[0]:.1f}, {ppo_ci_time[1]:.1f}]",
        "HEI": f"{p_hei_m:.2f} ± {p_hei_s:.2f}",
        "HEI_95_CI": f"[{ppo_ci_hei[0]:.2f}, {ppo_ci_hei[1]:.2f}]",
        "Travel_Distance_m": f"{p_dist_m:.1f} ± {p_dist_s:.1f}",
        "Timeout_Rate": f"{p_to_m:.1f}%",
        "Return": f"{p_ret_m:.2f} ± {p_ret_s:.2f}",
        "Inference_ms": f"{p_inf_m:.3f}"
    })

    # DQN across 5 seeds
    d_succ_m, d_succ_s = agg_metric(dqn_summaries, "success_rate")
    d_time_m, d_time_s = agg_metric(dqn_summaries, "mean_evac_time")
    d_hei_m, d_hei_s = agg_metric(dqn_summaries, "mean_hei")
    d_dist_m, d_dist_s = agg_metric(dqn_summaries, "mean_dist")
    d_to_m, d_to_s = agg_metric(dqn_summaries, "timeout_rate")
    d_ret_m, d_ret_s = agg_metric(dqn_summaries, "mean_return")
    d_inf_m, _ = agg_metric(dqn_summaries, "mean_inf_time_ms")

    dqn_ci_time = bootstrap_ci(dqn_times_42)
    dqn_ci_hei = bootstrap_ci([r["hei"] for r in all_records if r["model"] == "DQN" and r["seed"] == "42"])

    final_rows.append({
        "Model": "Masked DQN (5 seeds)",
        "Success_Rate": f"{d_succ_m:.1f} ± {d_succ_s:.2f}%",
        "Evacuation_Time_s": f"{d_time_m:.2f} ± {d_time_s:.2f}",
        "Evac_Time_95_CI": f"[{dqn_ci_time[0]:.1f}, {dqn_ci_time[1]:.1f}]",
        "HEI": f"{d_hei_m:.2f} ± {d_hei_s:.2f}",
        "HEI_95_CI": f"[{dqn_ci_hei[0]:.2f}, {dqn_ci_hei[1]:.2f}]",
        "Travel_Distance_m": f"{d_dist_m:.1f} ± {d_dist_s:.1f}",
        "Timeout_Rate": f"{d_to_m:.1f}%",
        "Return": f"{d_ret_m:.2f} ± {d_ret_s:.2f}",
        "Inference_ms": f"{d_inf_m:.3f}"
    })

    # Dijkstra
    final_rows.append({
        "Model": "Dynamic Dijkstra",
        "Success_Rate": f"{dijkstra_summ['success_rate']:.1f}%",
        "Evacuation_Time_s": f"{dijkstra_summ['mean_evac_time']:.2f}",
        "Evac_Time_95_CI": f"[{dijk_ci_time[0]:.1f}, {dijk_ci_time[1]:.1f}]",
        "HEI": f"{dijkstra_summ['mean_hei']:.2f}",
        "HEI_95_CI": f"[{dijk_ci_hei[0]:.2f}, {dijk_ci_hei[1]:.2f}]",
        "Travel_Distance_m": f"{dijkstra_summ['mean_dist']:.1f}",
        "Timeout_Rate": f"{dijkstra_summ['timeout_rate']:.1f}%",
        "Return": f"{dijkstra_summ['mean_return']:.2f}",
        "Inference_ms": f"{dijkstra_summ['mean_inf_time_ms']:.3f}"
    })

    # A*
    astar_ci_time = bootstrap_ci([r["evacuation_time"] for r in astar_recs])
    astar_ci_hei = bootstrap_ci([r["hei"] for r in astar_recs])
    final_rows.append({
        "Model": "Dynamic A*",
        "Success_Rate": f"{astar_summ['success_rate']:.1f}%",
        "Evacuation_Time_s": f"{astar_summ['mean_evac_time']:.2f}",
        "Evac_Time_95_CI": f"[{astar_ci_time[0]:.1f}, {astar_ci_time[1]:.1f}]",
        "HEI": f"{astar_summ['mean_hei']:.2f}",
        "HEI_95_CI": f"[{astar_ci_hei[0]:.2f}, {astar_ci_hei[1]:.2f}]",
        "Travel_Distance_m": f"{astar_summ['mean_dist']:.1f}",
        "Timeout_Rate": f"{astar_summ['timeout_rate']:.1f}%",
        "Return": f"{astar_summ['mean_return']:.2f}",
        "Inference_ms": f"{astar_summ['mean_inf_time_ms']:.3f}"
    })

    # Random
    rand_ci_time = bootstrap_ci([r["evacuation_time"] for r in rand_recs])
    rand_ci_hei = bootstrap_ci([r["hei"] for r in rand_recs])
    final_rows.append({
        "Model": "Random Valid Routing",
        "Success_Rate": f"{rand_summ['success_rate']:.1f}%",
        "Evacuation_Time_s": f"{rand_summ['mean_evac_time']:.2f}",
        "Evac_Time_95_CI": f"[{rand_ci_time[0]:.1f}, {rand_ci_time[1]:.1f}]",
        "HEI": f"{rand_summ['mean_hei']:.2f}",
        "HEI_95_CI": f"[{rand_ci_hei[0]:.2f}, {rand_ci_hei[1]:.2f}]",
        "Travel_Distance_m": f"{rand_summ['mean_dist']:.1f}",
        "Timeout_Rate": f"{rand_summ['timeout_rate']:.1f}%",
        "Return": f"{rand_summ['mean_return']:.2f}",
        "Inference_ms": f"{rand_summ['mean_inf_time_ms']:.3f}"
    })

    df_final = pd.DataFrame(final_rows)
    final_csv = "results/tables/final_large_scale_comparison.csv"
    df_final.to_csv(final_csv, index=False)

    print("\n" + "=" * 110)
    print("FINAL LARGE-SCALE BENCHMARK COMPARISON (500 TEST SCENARIOS, 5 SEEDS)")
    print("=" * 110)
    print(df_final.to_string(index=False))
    print("=" * 110 + "\n")

    # 7. Classification metrics
    if ppo_y_true and ppo_y_pred:
        acc = accuracy_score(ppo_y_true, ppo_y_pred)
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(ppo_y_true, ppo_y_pred, average="weighted", zero_division=0)
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(ppo_y_true, ppo_y_pred, average="macro", zero_division=0)
        clf_data = [{
            "Level": "Per-Step Routing (Large)",
            "Total_Steps": len(ppo_y_true),
            "Accuracy": round(acc, 4),
            "Weighted_Precision": round(p_w, 4),
            "Weighted_Recall": round(r_w, 4),
            "Weighted_F1": round(f1_w, 4),
            "Macro_F1": round(f1_m, 4)
        }]
        pd.DataFrame(clf_data).to_csv("results/tables/classification_metrics_large.csv", index=False)
        print(f"Per-Step Routing Accuracy on Large Network: {acc*100:.2f}% (Weighted F1: {f1_w:.4f})")


if __name__ == "__main__":
    main()
