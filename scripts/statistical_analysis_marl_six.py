"""Statistical Significance and Hypothesis Testing for the Six MARL Models.
Executes Shapiro-Wilk normality tests, paired Wilcoxon signed-rank tests, paired t-tests,
Cohen's d effect sizes, and Holm-Bonferroni corrections across all model pairs.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger

logger = setup_logger("StatisticalAnalysisMARL")


def compute_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Calculates Cohen's d effect size for paired or independent samples."""
    diff = x - y
    std_diff = np.std(diff, ddof=1)
    if std_diff < 1e-8:
        return 0.0
    return float(np.mean(diff) / std_diff)


def run_statistical_tests(scenario_records_csv: str, output_dir: str):
    """Performs pairwise statistical tests across all 6 MARL architectures."""
    if not os.path.isfile(scenario_records_csv):
        logger.error(f"Records file {scenario_records_csv} not found.")
        return

    df = pd.read_csv(scenario_records_csv)
    models = sorted(df["model"].unique())
    logger.info(f"Loaded scenario records for models: {models} ({len(df)} total rows)")

    metrics_to_test = ["hei", "travel_time", "return", "conflicts", "coordination_efficiency"]

    # 1. Normality Tests (Shapiro-Wilk)
    normality_rows = []
    for m in models:
        sub_m = df[df["model"] == m]
        for metric in metrics_to_test:
            vals = sub_m[metric].dropna().values
            if len(vals) >= 8:
                stat, p_val = stats.shapiro(vals[:5000])
                is_normal = bool(p_val > 0.05)
                normality_rows.append({
                    "model": m,
                    "metric": metric,
                    "shapiro_w": float(stat),
                    "p_value": float(p_val),
                    "is_normal": is_normal
                })

    norm_df = pd.DataFrame(normality_rows)
    norm_csv = os.path.join(output_dir, "normality_tests_marl.csv")
    norm_df.to_csv(norm_csv, index=False)
    logger.info(f"Saved Shapiro-Wilk normality tests to {norm_csv}")

    # 2. Pairwise Significance Tests
    pairwise_rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1 = models[i]
            m2 = models[j]

            df_m1 = df[df["model"] == m1].groupby("scenario_id").mean(numeric_only=True)
            df_m2 = df[df["model"] == m2].groupby("scenario_id").mean(numeric_only=True)

            common_scens = df_m1.index.intersection(df_m2.index)
            if len(common_scens) < 5:
                continue

            for metric in metrics_to_test:
                v1 = df_m1.loc[common_scens, metric].values
                v2 = df_m2.loc[common_scens, metric].values

                # Paired t-test
                t_stat, t_pval = stats.ttest_rel(v1, v2)

                # Wilcoxon signed-rank test
                try:
                    diff = v1 - v2
                    if np.all(diff == 0):
                        w_stat, w_pval = 0.0, 1.0
                    else:
                        w_stat, w_pval = stats.wilcoxon(v1, v2)
                except Exception:
                    w_stat, w_pval = np.nan, np.nan

                # Effect size
                d = compute_cohens_d(v1, v2)

                pairwise_rows.append({
                    "model_1": m1,
                    "model_2": m2,
                    "metric": metric,
                    "n_pairs": len(common_scens),
                    "mean_1": float(np.mean(v1)),
                    "mean_2": float(np.mean(v2)),
                    "diff_mean": float(np.mean(v1) - np.mean(v2)),
                    "cohens_d": d,
                    "t_statistic": float(t_stat),
                    "t_pvalue": float(t_pval),
                    "wilcoxon_stat": float(w_stat),
                    "wilcoxon_pvalue": float(w_pval)
                })

    test_df = pd.DataFrame(pairwise_rows)

    # Holm-Bonferroni Correction
    if not test_df.empty:
        for metric in metrics_to_test:
            sub_idx = test_df[test_df["metric"] == metric].index
            p_vals = test_df.loc[sub_idx, "wilcoxon_pvalue"].values
            valid_mask = ~np.isnan(p_vals)

            adj_p = np.full(len(p_vals), np.nan)
            if np.any(valid_mask):
                order = np.argsort(p_vals[valid_mask])
                sorted_p = p_vals[valid_mask][order]
                m_tests = len(sorted_p)
                adj = np.minimum(1.0, sorted_p * (m_tests - np.arange(m_tests)))
                # Monotonicity enforcement
                for k in range(1, m_tests):
                    adj[k] = max(adj[k], adj[k - 1])
                adj_p_valid = np.empty_like(adj)
                adj_p_valid[order] = adj
                adj_p[valid_mask] = adj_p_valid

            test_df.loc[sub_idx, "wilcoxon_p_holm"] = adj_p
            test_df.loc[sub_idx, "significant_after_correction"] = test_df.loc[sub_idx, "wilcoxon_p_holm"] < 0.05

    tests_csv = os.path.join(output_dir, "statistical_tests.csv")
    test_df.to_csv(tests_csv, index=False)
    tests_marl_csv = os.path.join(output_dir, "statistical_tests_marl.csv")
    test_df.to_csv(tests_marl_csv, index=False)
    logger.info(f"Saved pairwise statistical tests with Holm-Bonferroni corrections to {tests_csv} and {tests_marl_csv}")


def main():
    parser = argparse.ArgumentParser(description="Statistical Testing across Six MARL Models")
    parser.add_argument("--network", type=str, default="small", choices=["small", "large"])
    args = parser.parse_args()

    records_csv = os.path.join("results", "marl_six", "tables", f"marl_scenario_records_{args.network}.csv")
    out_dir = os.path.join("results", "marl_six", "tables")
    os.makedirs(out_dir, exist_ok=True)

    run_statistical_tests(records_csv, out_dir)


if __name__ == "__main__":
    main()
