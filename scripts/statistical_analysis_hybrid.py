"""Statistical Hypothesis Testing Suite for Hybrid RL Models:
Tests:
  - Shapiro-Wilk test for normality
  - Paired Student's t-test
  - Wilcoxon signed-rank test
  - Cohen's d effect size
  - 95% Bootstrap Confidence Intervals
Pairs:
  - Hybrid vs Baseline MaskablePPO
  - Hybrid vs Dynamic Dijkstra
  - Hybrid vs Baseline Masked DQN
Generates results/tables/hybrid_statistical_tests.json and CSV.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any

# Project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger
from src.metrics.evaluator import bootstrap_ci

logger = setup_logger("StatisticalAnalysisHybrid")


def compute_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Cohen's d effect size for paired samples."""
    diff = x - y
    std_diff = np.std(diff, ddof=1)
    if std_diff < 1e-8:
        return 0.0
    return float(np.mean(diff) / std_diff)


def run_statistical_suite(
    data_path: str = "results/tables/raw_hybrid_large_records.csv",
    baseline_model: str = "Baseline MaskablePPO",
    hybrid_model_keywords: List[str] = None
) -> Dict[str, Any]:
    """Runs normality checks, paired t-tests, Wilcoxon signed-rank tests, and Cohen's d."""
    if not os.path.isfile(data_path):
        logger.warning(f"Data path {data_path} not found.")
        return {}

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} evaluation records from {data_path}")

    all_models = df["model"].unique()
    logger.info(f"Models in dataset: {all_models}")

    target_metrics = ["travel_time", "hei", "distance", "return"]
    results = {}
    rows = []

    # Filter base models cleanly
    base_df = df[df["model"] == "Baseline MaskablePPO"]
    dijk_df = df[df["model"] == "Dynamic Dijkstra"]
    dqn_df = df[df["model"] == "Baseline Masked DQN"]

    hybrid_models = [m for m in all_models if any(k in m for k in ["GNN", "Recurrent", "SafetyShield", "TriHybrid"])]

    for hybrid_name in hybrid_models:
        hyb_df = df[df["model"] == hybrid_name]
        results[hybrid_name] = {}

        for comp_name, comp_df in [
            ("vs_BaselinePPO", base_df),
            ("vs_DynamicDijkstra", dijk_df),
            ("vs_BaselineDQN", dqn_df)
        ]:
            if comp_df.empty or hyb_df.empty:
                continue

            results[hybrid_name][comp_name] = {}

            # Match on scenario_id
            merged = pd.merge(hyb_df, comp_df, on="scenario_id", suffixes=("_hyb", "_comp"))
            if len(merged) < 10:
                continue

            for metric in target_metrics:
                m_hyb = merged[f"{metric}_hyb"].values
                m_comp = merged[f"{metric}_comp"].values
                diff = m_hyb - m_comp

                # Normality
                try:
                    sw_stat, sw_p = stats.shapiro(diff[:500])
                    is_normal = bool(sw_p > 0.05)
                except Exception:
                    sw_stat, sw_p, is_normal = 0.0, 0.0, False

                # Paired t-test
                t_stat, t_p = stats.ttest_rel(m_hyb, m_comp)

                # Wilcoxon signed-rank test
                try:
                    w_stat, w_p = stats.wilcoxon(m_hyb, m_comp, alternative="two-sided")
                except Exception:
                    w_stat, w_p = 0.0, 1.0

                # Effect size
                d = compute_cohens_d(m_hyb, m_comp)
                ci_low, ci_high = bootstrap_ci(list(diff), n_boot=1000)

                stat_record = {
                    "metric": metric,
                    "n_pairs": len(merged),
                    "mean_diff": round(float(np.mean(diff)), 4),
                    "ci_95": [round(ci_low, 4), round(ci_high, 4)],
                    "shapiro_stat": round(float(sw_stat), 4),
                    "shapiro_p": float(f"{sw_p:.4e}"),
                    "is_normal": is_normal,
                    "t_statistic": round(float(t_stat), 4),
                    "t_p_value": float(f"{t_p:.4e}"),
                    "wilcoxon_statistic": round(float(w_stat), 4),
                    "wilcoxon_p_value": float(f"{w_p:.4e}"),
                    "cohens_d": round(float(d), 4),
                    "statistically_significant": bool(w_p < 0.05)
                }
                results[hybrid_name][comp_name][metric] = stat_record

                rows.append({
                    "model": hybrid_name,
                    "comparison": comp_name,
                    "metric": metric,
                    "mean_diff": round(float(np.mean(diff)), 3),
                    "ci_95": f"[{ci_low:.2f}, {ci_high:.2f}]",
                    "wilcoxon_stat": round(float(w_stat), 2),
                    "wilcoxon_p": float(f"{w_p:.4e}"),
                    "cohens_d": round(float(d), 3),
                    "p_lt_0.05": bool(w_p < 0.05)
                })

    # Save
    out_dir = "results/tables"
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "hybrid_statistical_tests.json")
    csv_path = os.path.join(out_dir, "hybrid_statistical_tests.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"Statistical test results saved to {json_path} and {csv_path}")
    return results


if __name__ == "__main__":
    run_statistical_suite()
