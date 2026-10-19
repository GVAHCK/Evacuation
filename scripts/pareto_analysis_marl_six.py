"""Multi-Objective Pareto Optimality and Category Winner Analysis for Six MARL Models.
Analyzes trade-offs across:
  1. Joint Success Rate (Max)
  2. Hazard Exposure Index HEI (Min)
  3. Evacuation Travel Time (Min)
  4. Bottleneck Conflict Count (Min)
  5. Decision Inference Latency (Min)
Determines the non-dominated Pareto frontier and assigns category crowns.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger

logger = setup_logger("ParetoAnalysisMARL")


def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """Finds Pareto-efficient points in multi-dimensional minimization space.
    costs: [n_points, n_objectives] where ALL objectives are to be MINIMIZED.
    """
    n_points = costs.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        if is_efficient[i]:
            # Keep any point that is strictly better in at least one objective or equal
            is_efficient[is_efficient] = np.any(costs[is_efficient] < costs[i], axis=1) | np.all(costs[is_efficient] == costs[i], axis=1)
            is_efficient[i] = True
    return is_efficient


def run_pareto_analysis(summary_csv: str, out_dir: str):
    """Computes Pareto frontier and determines category winner awards."""
    if not os.path.isfile(summary_csv):
        logger.error(f"Summary CSV {summary_csv} not found.")
        return

    df = pd.read_csv(summary_csv)
    # Aggregate across seeds per model
    model_agg = df.groupby("model").agg({
        "joint_success_rate": "mean",
        "indiv_success_rate": "mean",
        "mean_hei": "mean",
        "mean_travel_time": "mean",
        "mean_conflicts": "mean",
        "mean_route_overlap": "mean",
        "mean_coordination_efficiency": "mean",
        "mean_latency_ms": "mean"
    }).reset_index()

    # Formulate minimization cost matrix:
    # 1. Negative Joint Success (to minimize)
    # 2. Mean HEI
    # 3. Mean Travel Time
    # 4. Mean Conflicts
    # 5. Mean Latency
    costs = np.column_stack([
        -model_agg["joint_success_rate"].values,
        model_agg["mean_hei"].values,
        model_agg["mean_travel_time"].values,
        model_agg["mean_conflicts"].values,
        model_agg["mean_latency_ms"].values
    ])

    pareto_mask = is_pareto_efficient(costs)
    model_agg["is_pareto_optimal"] = pareto_mask

    # Export Pareto results
    pareto_csv = os.path.join(out_dir, "pareto_results.csv")
    model_agg.to_csv(pareto_csv, index=False)
    pareto_frontier_csv = os.path.join(out_dir, "pareto_frontier_marl.csv")
    model_agg.to_csv(pareto_frontier_csv, index=False)
    logger.info(f"Saved Pareto results to {pareto_csv} and {pareto_frontier_csv}")

    # Determine Category Winners
    winners = {
        "highest_joint_success": {
            "model": model_agg.loc[model_agg["joint_success_rate"].idxmax(), "model"],
            "value": float(model_agg["joint_success_rate"].max()),
            "metric": "Joint Success Rate (%)"
        },
        "lowest_hazard_exposure": {
            "model": model_agg.loc[model_agg["mean_hei"].idxmin(), "model"],
            "value": float(model_agg["mean_hei"].min()),
            "metric": "Mean HEI"
        },
        "fastest_evacuation": {
            "model": model_agg.loc[model_agg["mean_travel_time"].idxmin(), "model"],
            "value": float(model_agg["mean_travel_time"].min()),
            "metric": "Travel Time (s)"
        },
        "lowest_bottleneck_conflicts": {
            "model": model_agg.loc[model_agg["mean_conflicts"].idxmin(), "model"],
            "value": float(model_agg["mean_conflicts"].min()),
            "metric": "Total Intersection Conflicts"
        },
        "lowest_route_overlap": {
            "model": model_agg.loc[model_agg["mean_route_overlap"].idxmin(), "model"],
            "value": float(model_agg["mean_route_overlap"].min()),
            "metric": "Mean Route Overlap (Jaccard)"
        },
        "highest_coordination_efficiency": {
            "model": model_agg.loc[model_agg["mean_coordination_efficiency"].idxmax(), "model"],
            "value": float(model_agg["mean_coordination_efficiency"].max()),
            "metric": "Coordination Efficiency Index"
        },
        "fastest_inference_latency": {
            "model": model_agg.loc[model_agg["mean_latency_ms"].idxmin(), "model"],
            "value": float(model_agg["mean_latency_ms"].min()),
            "metric": "Step Inference Latency (ms)"
        },
        "pareto_efficient_cohort": list(model_agg.loc[model_agg["is_pareto_optimal"], "model"].values)
    }

    winners_json = os.path.join(out_dir, "category_winners_marl.json")
    with open(winners_json, "w", encoding="utf-8") as f:
        json.dump(winners, f, indent=2)

    logger.info("Category Winners:")
    for k, v in winners.items():
        if k != "pareto_efficient_cohort":
            logger.info(f"  - {k:32s}: {v['model']} ({v['value']:.3f} {v['metric']})")
        else:
            logger.info(f"  - Pareto Optimal Models: {v}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Objective Pareto Analysis for MARL")
    parser.add_argument("--network", type=str, default="small", choices=["small", "large"])
    args = parser.parse_args()

    summary_csv = os.path.join("results", "marl_six", "tables", f"marl_evaluation_summary_{args.network}.csv")
    out_dir = os.path.join("results", "marl_six", "tables")
    os.makedirs(out_dir, exist_ok=True)

    run_pareto_analysis(summary_csv, out_dir)


if __name__ == "__main__":
    main()
