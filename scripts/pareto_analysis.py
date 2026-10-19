"""Multi-Objective Pareto Analysis & Model Selection Suite:
Identifies non-dominated Pareto frontier and computes composite multi-objective score:
  - Success Rate (weight 0.30)
  - Safety / 1 - HEI (weight 0.20)
  - Efficiency / 1 - Time (weight 0.15)
  - Timeout Avoidance (weight 0.10)
  - Distance Minimization (weight 0.10)
  - Routing F1 Agreement (weight 0.05)
  - Multi-seed Stability (weight 0.05)
  - Latency (weight 0.05)
Identifies:
  - Best Reliability Model
  - Best Safety Model
  - Best Efficiency Model
  - Best Routing Model
  - Best Zero-Shot Model
  - Best Real-Time Model
  - Best Overall Pareto Model
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

# Project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger

logger = setup_logger("ParetoAnalysis")


def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """Finds the pareto-efficient points for a minimization problem.

    :param costs: (n_points, n_objectives) array where all objectives are to be minimized
    :return: (n_points,) boolean array indicating Pareto efficiency
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep any point with a lower cost on at least one dimension
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True  # Self comparison is not strictly less
    return is_efficient


def compute_pareto_rankings(
    summary_csv: str = "results/tables/hybrid_large_comparison.csv",
    zero_shot_csv: str = "results/tables/hybrid_zero_shot_comparison.csv"
) -> Dict[str, Any]:
    """Computes Pareto dominance, composite performance score, and automatic winner selection."""
    if not os.path.isfile(summary_csv):
        logger.warning(f"{summary_csv} not found.")
        return {}

    df = pd.read_csv(summary_csv)
    df["model_family"] = df["model"].str.replace(r"\s*\(seed\s*\d+\)", "", regex=True).str.strip()
    grouped = df.groupby("model_family").agg({
        "success_rate": ["mean", "std"],
        "evacuation_time_mean": ["mean", "std"],
        "hei_mean": ["mean", "std"],
        "distance_mean": ["mean", "std"],
        "timeout_rate": ["mean"],
        "latency_mean_ms": ["mean"],
        "latency_p95_ms": ["mean"],
        "weighted_f1": ["mean"],
        "routing_accuracy": ["mean"]
    }).reset_index()

    # Flatten columns
    grouped.columns = [
        "model", "success_mean", "success_std", "time_mean", "time_std",
        "hei_mean", "hei_std", "dist_mean", "dist_std", "timeout_mean",
        "latency_mean", "latency_p95", "f1_mean", "acc_mean"
    ]
    grouped["success_std"] = grouped["success_std"].fillna(0.0)
    grouped["time_std"] = grouped["time_std"].fillna(0.0)
    grouped["hei_std"] = grouped["hei_std"].fillna(0.0)

    # 1. Normalize objectives to [0, 1] where 1 is best, 0 is worst
    def norm_max(s):
        r = s.max() - s.min()
        return (s - s.min()) / r if r > 1e-6 else np.ones_like(s)

    def norm_min(s):
        r = s.max() - s.min()
        return (s.max() - s) / r if r > 1e-6 else np.ones_like(s)

    norm_success = norm_max(grouped["success_mean"])
    norm_safety = norm_min(grouped["hei_mean"])
    norm_time = norm_min(grouped["time_mean"])
    norm_timeout = norm_min(grouped["timeout_mean"])
    norm_dist = norm_min(grouped["dist_mean"])
    norm_f1 = norm_max(grouped["f1_mean"])
    norm_stability = norm_min(grouped["time_std"] + grouped["hei_std"])
    norm_latency = norm_min(grouped["latency_mean"])

    # Composite Performance Score
    composite = (
        0.30 * norm_success +
        0.20 * norm_safety +
        0.15 * norm_time +
        0.10 * norm_timeout +
        0.10 * norm_dist +
        0.05 * norm_f1 +
        0.05 * norm_stability +
        0.05 * norm_latency
    ) * 100.0

    grouped["composite_score"] = np.round(composite, 2)

    # 2. Pareto Frontier: Minimization format [-success, time, hei, timeout, dist, latency]
    cost_matrix = np.column_stack([
        -grouped["success_mean"].values,
        grouped["time_mean"].values,
        grouped["hei_mean"].values,
        grouped["timeout_mean"].values,
        grouped["dist_mean"].values,
        grouped["latency_mean"].values
    ])
    pareto_mask = is_pareto_efficient(cost_matrix)
    grouped["pareto_optimal"] = pareto_mask

    # 3. Categorical Winners
    best_reliability = grouped.sort_values(by=["success_mean", "timeout_mean"], ascending=[False, True]).iloc[0]["model"]
    best_safety = grouped.sort_values(by="hei_mean", ascending=True).iloc[0]["model"]
    best_efficiency = grouped.sort_values(by="time_mean", ascending=True).iloc[0]["model"]
    best_routing = grouped.sort_values(by="f1_mean", ascending=False).iloc[0]["model"]
    best_realtime = grouped.sort_values(by="latency_mean", ascending=True).iloc[0]["model"]
    best_pareto = grouped[grouped["pareto_optimal"]].sort_values(by="composite_score", ascending=False).iloc[0]["model"]

    winners = {
        "best_reliability_model": best_reliability,
        "best_safety_model": best_safety,
        "best_efficiency_model": best_efficiency,
        "best_routing_model": best_routing,
        "best_real_time_model": best_realtime,
        "best_overall_pareto_model": best_pareto
    }

    # Save
    out_dir = "results/tables"
    os.makedirs(out_dir, exist_ok=True)
    rank_csv = os.path.join(out_dir, "pareto_rankings.csv")
    win_json = os.path.join(out_dir, "pareto_winners.json")

    grouped.to_csv(rank_csv, index=False)
    with open(win_json, "w", encoding="utf-8") as f:
        json.dump(winners, f, indent=2)

    logger.info(f"Pareto rankings saved to {rank_csv} and winners to {win_json}")
    return {"rankings": grouped.to_dict(orient="records"), "winners": winners}


if __name__ == "__main__":
    compute_pareto_rankings()
