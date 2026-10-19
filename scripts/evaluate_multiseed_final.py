"""Final Multi-seed Evaluation Suite with paired Wilcoxon signed-rank and t-tests across 100 unseen test scenarios."""

import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Tuple

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN

from src.utils.logger import setup_logger
from src.environment.evacuation_env import EvacuationEnv
from src.baselines.dijkstra import DijkstraBaseline
from src.baselines.astar import AStarBaseline
from src.baselines.random_routing import RandomValidRoutingBaseline
from src.metrics.evaluator import bootstrap_ci

logger = setup_logger("FinalEvaluation")

SEEDS = [42, 101, 2024]
TEST_SCENARIOS_PATH = "data/generated/test_scenarios.json"


def evaluate_single_run(
    model_or_baseline,
    model_name: str,
    seed_tag: str,
    is_ppo: bool = False,
    is_dqn: bool = False,
    test_scenarios: List[Dict[str, Any]] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Evaluates one model instance on the 100 test scenarios."""
    env = EvacuationEnv(scenarios=test_scenarios, use_traci=False, curriculum_stage=4)

    records = []
    successes = []
    times = []
    heis = []
    dists = []
    timeouts = []
    returns = []
    inf_times = []

    for scen in test_scenarios:
        obs, info = env.reset(options={"scenario": scen})
        done = False
        ep_ret = 0.0
        ep_steps = 0

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

            obs, r, term, trunc, step_info = env.step(action)
            ep_ret += r
            ep_steps += 1
            done = term or trunc

        succ = step_info.get("success", False)
        hei = step_info.get("hazard_exposure_index", 0.0)
        tt = step_info.get("cumulative_travel_time", 0.0)
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

    return records, summary


def main():
    logger.info("=" * 80)
    logger.info("PHASE D: FINAL STATISTICAL BENCHMARK EVALUATION (100 UNSEEN SCENARIOS)")
    logger.info("=" * 80)

    test_scenarios = json.load(open(TEST_SCENARIOS_PATH))
    all_records = []

    # 1. Evaluate PPO across 3 seeds
    ppo_summaries = []
    ppo_seed_times = {}
    ppo_seed_returns = {}
    for seed in SEEDS:
        ckpt_path = f"models/ppo/seed_{seed}/best_model.zip"
        logger.info(f"Evaluating PPO (seed {seed})...")
        model = MaskablePPO.load(ckpt_path)
        recs, summ = evaluate_single_run(model, "PPO", str(seed), is_ppo=True, test_scenarios=test_scenarios)
        all_records.extend(recs)
        ppo_summaries.append(summ)
        ppo_seed_times[seed] = [r["evacuation_time"] for r in recs]
        ppo_seed_returns[seed] = [r["return"] for r in recs]

    # 2. Evaluate DQN across 3 seeds
    dqn_summaries = []
    dqn_seed_times = {}
    dqn_seed_returns = {}
    dqn_seed_heis = {}
    for seed in SEEDS:
        ckpt_path = f"models/dqn/seed_{seed}/best_model.zip"
        logger.info(f"Evaluating DQN (seed {seed})...")
        model = DQN.load(ckpt_path)
        recs, summ = evaluate_single_run(model, "DQN", str(seed), is_dqn=True, test_scenarios=test_scenarios)
        all_records.extend(recs)
        dqn_summaries.append(summ)
        dqn_seed_times[seed] = [r["evacuation_time"] for r in recs]
        dqn_seed_returns[seed] = [r["return"] for r in recs]
        dqn_seed_heis[seed] = [r["hei"] for r in recs]

    # 3. Evaluate Baselines
    logger.info("Evaluating Dijkstra Baseline...")
    dijkstra = DijkstraBaseline()
    dijkstra_recs, dijkstra_summ = evaluate_single_run(dijkstra, "Dijkstra", "fixed", test_scenarios=test_scenarios)
    all_records.extend(dijkstra_recs)

    logger.info("Evaluating A* Baseline...")
    astar = AStarBaseline()
    astar_recs, astar_summ = evaluate_single_run(astar, "A*", "fixed", test_scenarios=test_scenarios)
    all_records.extend(astar_recs)

    logger.info("Evaluating Random Baseline...")
    rand_agent = RandomValidRoutingBaseline()
    rand_recs, rand_summ = evaluate_single_run(rand_agent, "Random", "fixed", test_scenarios=test_scenarios)
    all_records.extend(rand_recs)

    # 4. Save raw records
    df_raw = pd.DataFrame(all_records)
    df_raw.to_csv("results/tables/raw_multiseed_records.csv", index=False)

    # 5. Statistical Hypothesis Tests (PPO seed 42 vs Dijkstra & PPO vs DQN)
    ppo_times_42 = ppo_seed_times[42]
    dijk_times = [r["evacuation_time"] for r in dijkstra_recs]
    dqn_times_42 = dqn_seed_times[42]

    ppo_rets_42 = ppo_seed_returns[42]
    dijk_rets = [r["return"] for r in dijkstra_recs]
    dqn_rets_42 = dqn_seed_returns[42]

    ppo_heis_42 = [r["hei"] for r in all_records if r["model"] == "PPO" and r["seed"] == "42"]
    dqn_heis_42 = dqn_seed_heis[42]

    # Paired t-test and Wilcoxon signed-rank test
    ttest_time_dijk = stats.ttest_rel(ppo_times_42, dijk_times)
    wtest_time_dijk = stats.wilcoxon(np.array(ppo_times_42) - np.array(dijk_times))

    ttest_time_dqn = stats.ttest_rel(ppo_times_42, dqn_times_42)
    wtest_time_dqn = stats.wilcoxon(np.array(ppo_times_42) - np.array(dqn_times_42))

    ttest_hei_dqn = stats.ttest_rel(ppo_heis_42, dqn_heis_42)
    wtest_hei_dqn = stats.wilcoxon(np.array(ppo_heis_42) - np.array(dqn_heis_42))

    stat_results = {
        "PPO_vs_Dijkstra_Time": {
            "paired_t_stat": float(ttest_time_dijk.statistic),
            "paired_t_pvalue": float(ttest_time_dijk.pvalue),
            "wilcoxon_stat": float(wtest_time_dijk.statistic),
            "wilcoxon_pvalue": float(wtest_time_dijk.pvalue),
        },
        "PPO_vs_DQN_Time": {
            "paired_t_stat": float(ttest_time_dqn.statistic),
            "paired_t_pvalue": float(ttest_time_dqn.pvalue),
            "wilcoxon_stat": float(wtest_time_dqn.statistic),
            "wilcoxon_pvalue": float(wtest_time_dqn.pvalue),
        },
        "PPO_vs_DQN_HEI": {
            "paired_t_stat": float(ttest_hei_dqn.statistic),
            "paired_t_pvalue": float(ttest_hei_dqn.pvalue),
            "wilcoxon_stat": float(wtest_hei_dqn.statistic),
            "wilcoxon_pvalue": float(wtest_hei_dqn.pvalue),
        }
    }

    with open("results/tables/statistical_tests.json", "w", encoding="utf-8") as f:
        json.dump(stat_results, f, indent=2)

    # 6. Aggregate Final Multi-Seed Table (Mean +- Std)
    def agg_metric(summaries, key):
        vals = [s[key] for s in summaries]
        return np.mean(vals), np.std(vals)

    # Bootstrap CIs for representative runs
    ppo_ci_time = bootstrap_ci(ppo_times_42)
    ppo_ci_hei = bootstrap_ci(ppo_heis_42)
    dijk_ci_time = bootstrap_ci(dijk_times)
    dijk_ci_hei = bootstrap_ci([r["hei"] for r in dijkstra_recs])

    final_rows = []

    # PPO
    p_succ_m, p_succ_s = agg_metric(ppo_summaries, "success_rate")
    p_time_m, p_time_s = agg_metric(ppo_summaries, "mean_evac_time")
    p_hei_m, p_hei_s = agg_metric(ppo_summaries, "mean_hei")
    p_dist_m, p_dist_s = agg_metric(ppo_summaries, "mean_dist")
    p_to_m, p_to_s = agg_metric(ppo_summaries, "timeout_rate")
    p_ret_m, p_ret_s = agg_metric(ppo_summaries, "mean_return")
    p_inf_m, _ = agg_metric(ppo_summaries, "mean_inf_time_ms")

    final_rows.append({
        "Model": "PPO (Proposed, 3 seeds)",
        "Success_Rate": f"{p_succ_m:.1f} ± {p_succ_s:.1f}%",
        "Evacuation_Time_s": f"{p_time_m:.2f} ± {p_time_s:.2f}",
        "Evac_Time_95_CI": f"[{ppo_ci_time[0]:.1f}, {ppo_ci_time[1]:.1f}]",
        "HEI": f"{p_hei_m:.2f} ± {p_hei_s:.2f}",
        "HEI_95_CI": f"[{ppo_ci_hei[0]:.2f}, {ppo_ci_hei[1]:.2f}]",
        "Travel_Distance_m": f"{p_dist_m:.1f} ± {p_dist_s:.1f}",
        "Timeout_Rate": f"{p_to_m:.1f}%",
        "Return": f"{p_ret_m:.2f} ± {p_ret_s:.2f}",
        "Inference_ms": f"{p_inf_m:.3f}"
    })

    # DQN
    d_succ_m, d_succ_s = agg_metric(dqn_summaries, "success_rate")
    d_time_m, d_time_s = agg_metric(dqn_summaries, "mean_evac_time")
    d_hei_m, d_hei_s = agg_metric(dqn_summaries, "mean_hei")
    d_dist_m, d_dist_s = agg_metric(dqn_summaries, "mean_dist")
    d_to_m, d_to_s = agg_metric(dqn_summaries, "timeout_rate")
    d_ret_m, d_ret_s = agg_metric(dqn_summaries, "mean_return")
    d_inf_m, _ = agg_metric(dqn_summaries, "mean_inf_time_ms")

    dqn_ci_time = bootstrap_ci(dqn_times_42)
    dqn_ci_hei = bootstrap_ci(dqn_heis_42)

    final_rows.append({
        "Model": "DQN (3 seeds)",
        "Success_Rate": f"{d_succ_m:.1f} ± {d_succ_s:.1f}%",
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
    df_final.to_csv("results/tables/final_multiseed_comparison.csv", index=False)
    # Also overwrite model_comparison.csv with this statistically rigorous table
    df_final.to_csv("results/tables/model_comparison.csv", index=False)

    print("\n" + "=" * 110)
    print("FINAL STATISTICALLY VALID BENCHMARK COMPARISON (100 UNSEEN SCENARIOS, 3 SEEDS)")
    print("=" * 110)
    print(df_final.to_string(index=False))
    print("=" * 110 + "\n")

    print("STATISTICAL SIGNIFICANCE TESTS:")
    print(json.dumps(stat_results, indent=2))


if __name__ == "__main__":
    main()
